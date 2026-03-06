use std::collections::BTreeMap;

use flexpart_gpu::config::ReleaseConfig;
use flexpart_gpu::coords::GridDomain;
use flexpart_gpu::gpu::GpuError;
use flexpart_gpu::io::TimeBoundsBehavior;
use flexpart_gpu::particles::ParticleStore;
use flexpart_gpu::physics::VelocityToGridScale;
use flexpart_gpu::simulation::{
    BackwardReceptorConfig, BackwardSourceCollection, BackwardSourceRegionConfig,
    BackwardTimeLoopConfig, BackwardTimeLoopDriver, ForwardStepForcing, ForwardTimeLoopConfig,
    ForwardTimeLoopDriver, MetTimeBracket, TimeLoopError,
};
use flexpart_gpu::wind::{uniform_wind_field, SurfaceFields, WindFieldGrid};
use ndarray::Array1;

const GRID_NX: usize = 64;
const GRID_NY: usize = 64;
const GRID_NZ: usize = 16;
const PARTICLE_CAPACITY: usize = 8;

const SOURCE_X_LON: f64 = 7.0;
const SOURCE_Y_LAT: f64 = 5.0;
const SOURCE_Z_M: f64 = 1.0;
const RECEPTOR_X_LON: f64 = 10.0;
const RECEPTOR_Y_LAT: f64 = 5.0;
const RECEPTOR_Z_M: f64 = 1.0;
const PARTICLE_MASS_KG: f64 = 1.0;

const CONSISTENCY_ABS_MASS_EPSILON_KG: f32 = 1.0e-6;
const CONSISTENCY_REL_MASS_EPSILON: f32 = 1.0e-6;
const CONSISTENCY_RATIO_EPSILON: f32 = 1.0e-6;

fn synthetic_wind_grid(nx: usize, ny: usize, nz: usize) -> WindFieldGrid {
    WindFieldGrid::new(
        nx,
        ny,
        nz,
        nz,
        nz,
        1.0,
        1.0,
        0.0,
        0.0,
        Array1::from_iter((0..nz).map(|k| k as f32)),
    )
}

fn synthetic_surface_fields(nx: usize, ny: usize) -> SurfaceFields {
    let mut surface = SurfaceFields::zeros(nx, ny);
    surface.surface_pressure_pa.fill(101_325.0);
    surface.u10_ms.fill(1.0);
    surface.v10_ms.fill(0.0);
    surface.temperature_2m_k.fill(290.0);
    surface.dewpoint_2m_k.fill(285.0);
    surface.precip_large_scale_mm_h.fill(0.0);
    surface.precip_convective_mm_h.fill(0.0);
    surface.sensible_heat_flux_w_m2.fill(0.0);
    surface.solar_radiation_w_m2.fill(0.0);
    surface.surface_stress_n_m2.fill(0.0);
    surface.friction_velocity_ms.fill(0.0);
    surface.convective_velocity_scale_ms.fill(0.0);
    surface.mixing_height_m.fill(1_000.0);
    surface.tropopause_height_m.fill(10_000.0);
    surface.inv_obukhov_length_per_m.fill(0.0);
    surface
}

#[derive(Debug, Clone, Copy)]
struct RegionBounds {
    lon_min: f64,
    lon_max: f64,
    lat_min: f64,
    lat_max: f64,
    z_min_m: f64,
    z_max_m: f64,
}

impl RegionBounds {
    fn contains_particle(
        &self,
        particle: &flexpart_gpu::particles::Particle,
        release_grid: &GridDomain,
    ) -> bool {
        let lon = release_grid.xlon0 + particle.grid_x() * release_grid.dx;
        let lat = release_grid.ylat0 + particle.grid_y() * release_grid.dy;
        let z = f64::from(particle.pos_z);
        (self.lon_min..=self.lon_max).contains(&lon)
            && (self.lat_min..=self.lat_max).contains(&lat)
            && (self.z_min_m..=self.z_max_m).contains(&z)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct RegionCollection {
    hit_count: usize,
    total_mass_kg: f32,
}

fn collect_region(
    store: &ParticleStore,
    release_grid: &GridDomain,
    bounds: RegionBounds,
) -> RegionCollection {
    let mut hit_count = 0usize;
    let mut total_mass_kg = 0.0_f32;
    for particle in store.as_slice() {
        if !particle.is_active() {
            continue;
        }
        if bounds.contains_particle(particle, release_grid) {
            hit_count += 1;
            total_mass_kg += particle.mass[0];
        }
    }
    RegionCollection {
        hit_count,
        total_mass_kg,
    }
}

#[derive(Debug, Clone, PartialEq)]
struct SourceReceptorConsistencyMetric {
    forward_receptor_hit_count: usize,
    backward_source_hit_count: usize,
    forward_receptor_mass_kg: f32,
    backward_source_mass_kg: f32,
    absolute_mass_error_kg: f32,
    relative_mass_error: f32,
    mass_closure_ratio: f32,
}

impl SourceReceptorConsistencyMetric {
    fn evaluate(
        forward_receptor: RegionCollection,
        backward_source: BackwardSourceCollection,
    ) -> Self {
        let absolute_mass_error_kg =
            (forward_receptor.total_mass_kg - backward_source.total_mass_kg).abs();
        let relative_mass_error = if forward_receptor.total_mass_kg > 0.0 {
            absolute_mass_error_kg / forward_receptor.total_mass_kg
        } else if backward_source.total_mass_kg > 0.0 {
            f32::INFINITY
        } else {
            0.0
        };
        let mass_closure_ratio = if forward_receptor.total_mass_kg > 0.0 {
            backward_source.total_mass_kg / forward_receptor.total_mass_kg
        } else if backward_source.total_mass_kg == 0.0 {
            1.0
        } else {
            0.0
        };
        Self {
            forward_receptor_hit_count: forward_receptor.hit_count,
            backward_source_hit_count: backward_source.hit_count,
            forward_receptor_mass_kg: forward_receptor.total_mass_kg,
            backward_source_mass_kg: backward_source.total_mass_kg,
            absolute_mass_error_kg,
            relative_mass_error,
            mass_closure_ratio,
        }
    }
}

struct SourceReceptorConsistencyHarness {
    release_grid: GridDomain,
}

impl SourceReceptorConsistencyHarness {
    fn new() -> Self {
        Self {
            release_grid: GridDomain {
                xlon0: 0.0,
                ylat0: 0.0,
                dx: 1.0,
                dy: 1.0,
                nx: GRID_NX,
                ny: GRID_NY,
            },
        }
    }

    fn forward_config() -> ForwardTimeLoopConfig {
        ForwardTimeLoopConfig {
            start_timestamp: "20240101000000".to_string(),
            end_timestamp: "20240101000002".to_string(),
            timestep_seconds: 1,
            time_bounds_behavior: TimeBoundsBehavior::Strict,
            velocity_to_grid_scale: VelocityToGridScale::IDENTITY,
            ..ForwardTimeLoopConfig::default()
        }
    }

    fn forward_releases() -> Vec<ReleaseConfig> {
        vec![ReleaseConfig {
            name: "source_release".to_string(),
            start_time: "20240101000000".to_string(),
            end_time: "20240101000000".to_string(),
            lon: SOURCE_X_LON,
            lat: SOURCE_Y_LAT,
            z_min: SOURCE_Z_M,
            z_max: SOURCE_Z_M,
            mass_kg: PARTICLE_MASS_KG,
            particle_count: 1,
            raw: BTreeMap::new(),
        }]
    }

    fn backward_config() -> BackwardTimeLoopConfig {
        BackwardTimeLoopConfig {
            start_timestamp: "20240101000002".to_string(),
            end_timestamp: "20240101000000".to_string(),
            timestep_seconds: 1,
            time_bounds_behavior: TimeBoundsBehavior::Strict,
            velocity_to_grid_scale: VelocityToGridScale::IDENTITY,
            receptors: vec![BackwardReceptorConfig {
                name: "receptor_a".to_string(),
                lon: RECEPTOR_X_LON,
                lat: RECEPTOR_Y_LAT,
                z_m: RECEPTOR_Z_M,
                particle_count: 1,
                mass_kg: PARTICLE_MASS_KG,
            }],
            source_regions: vec![BackwardSourceRegionConfig {
                name: "source_a".to_string(),
                lon_min: SOURCE_X_LON - 2.0,
                lon_max: SOURCE_X_LON + 2.0,
                lat_min: SOURCE_Y_LAT - 2.0,
                lat_max: SOURCE_Y_LAT + 2.0,
                z_min_m: 0.0,
                z_max_m: 16.0,
            }],
            ..BackwardTimeLoopConfig::default()
        }
    }

    fn receptor_bounds() -> RegionBounds {
        RegionBounds {
            lon_min: RECEPTOR_X_LON - 2.0,
            lon_max: RECEPTOR_X_LON + 2.0,
            lat_min: RECEPTOR_Y_LAT - 2.0,
            lat_max: RECEPTOR_Y_LAT + 2.0,
            z_min_m: 0.0,
            z_max_m: 16.0,
        }
    }

    fn shared_met<'a>(
        &'a self,
        wind_t0: &'a flexpart_gpu::wind::WindField3D,
        wind_t1: &'a flexpart_gpu::wind::WindField3D,
        surface_t0: &'a SurfaceFields,
        surface_t1: &'a SurfaceFields,
    ) -> MetTimeBracket<'a> {
        MetTimeBracket {
            wind_t0,
            wind_t1,
            surface_t0,
            surface_t1,
            time_t0_seconds: 1_704_067_200,
            time_t1_seconds: 1_704_067_202,
        }
    }

    fn run(&self) -> Result<SourceReceptorConsistencyMetric, TimeLoopError> {
        let wind_grid = synthetic_wind_grid(GRID_NX, GRID_NY, GRID_NZ);
        let wind_t0 = uniform_wind_field(&wind_grid, 1.0, 0.0, 0.0);
        let wind_t1 = uniform_wind_field(&wind_grid, 1.0, 0.0, 0.0);
        let surface_t0 = synthetic_surface_fields(GRID_NX, GRID_NY);
        let surface_t1 = synthetic_surface_fields(GRID_NX, GRID_NY);
        let met = self.shared_met(&wind_t0, &wind_t1, &surface_t0, &surface_t1);
        let forcing = ForwardStepForcing::default();

        let mut forward_driver = pollster::block_on(ForwardTimeLoopDriver::new(
            Self::forward_config(),
            &Self::forward_releases(),
            self.release_grid.clone(),
            PARTICLE_CAPACITY,
        ))?;
        let forward_reports = pollster::block_on(forward_driver.run_to_end(&met, &forcing))?;
        assert_eq!(
            forward_reports.len(),
            3,
            "forward run should include three timesteps"
        );

        let forward_receptor = collect_region(
            forward_driver.particle_store(),
            &self.release_grid,
            Self::receptor_bounds(),
        );

        let mut backward_driver = pollster::block_on(BackwardTimeLoopDriver::new(
            Self::backward_config(),
            self.release_grid.clone(),
            PARTICLE_CAPACITY,
        ))?;
        let backward_reports = pollster::block_on(backward_driver.run_to_end(&met, &forcing))?;
        assert_eq!(
            backward_reports.len(),
            3,
            "backward run should include three timesteps"
        );
        let backward_source = backward_reports
            .last()
            .expect("backward run should produce at least one report")
            .source_collections
            .get("source_a")
            .cloned()
            .expect("source collection for source_a should exist");

        Ok(SourceReceptorConsistencyMetric::evaluate(
            forward_receptor,
            backward_source,
        ))
    }
}

#[test]
fn backward_source_receptor_consistency_matches_forward_transport() {
    let harness = SourceReceptorConsistencyHarness::new();
    let metric = match harness.run() {
        Ok(metric) => metric,
        Err(TimeLoopError::Gpu(GpuError::NoAdapter)) => {
            eprintln!("No GPU adapter found — skipping backward source-receptor consistency test");
            return;
        }
        Err(err) => panic!("unexpected source-receptor consistency error: {err}"),
    };

    assert_eq!(
        metric.forward_receptor_hit_count, 1,
        "forward transport should place exactly one particle in the receptor envelope"
    );
    assert_eq!(
        metric.backward_source_hit_count, 1,
        "backward transport should place exactly one particle in the source envelope"
    );
    assert!(
        (metric.forward_receptor_mass_kg - 1.0).abs() < 1.0e-6,
        "forward receptor mass should match released mass"
    );
    assert!(
        (metric.backward_source_mass_kg - 1.0).abs() < 1.0e-6,
        "backward source mass should match released mass"
    );
    assert!(
        metric.absolute_mass_error_kg <= CONSISTENCY_ABS_MASS_EPSILON_KG,
        "absolute mass mismatch too large: {} > {}",
        metric.absolute_mass_error_kg,
        CONSISTENCY_ABS_MASS_EPSILON_KG
    );
    assert!(
        metric.relative_mass_error <= CONSISTENCY_REL_MASS_EPSILON,
        "relative mass mismatch too large: {} > {}",
        metric.relative_mass_error,
        CONSISTENCY_REL_MASS_EPSILON
    );
    assert!(
        (metric.mass_closure_ratio - 1.0).abs() <= CONSISTENCY_RATIO_EPSILON,
        "mass closure ratio should be ~1.0, got {}",
        metric.mass_closure_ratio
    );
}
