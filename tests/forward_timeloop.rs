use std::collections::BTreeMap;

use flexpart_gpu::config::ReleaseConfig;
use flexpart_gpu::coords::GridDomain;
use flexpart_gpu::gpu::GpuError;
use flexpart_gpu::io::TimeBoundsBehavior;
use flexpart_gpu::particles::{
    ParticleSpatialSortOptions, SpatialSortBounds, SpatialSortBoundsMode,
};
use flexpart_gpu::physics::VelocityToGridScale;
use flexpart_gpu::simulation::{
    ForwardSpatialSortConfig, ForwardStepForcing, ForwardTimeLoopConfig, ForwardTimeLoopDriver,
    MetTimeBracket, TimeLoopError,
};
use flexpart_gpu::wind::{uniform_wind_field, SurfaceFields, WindFieldGrid};
use ndarray::Array1;

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
    surface.u10_ms.fill(2.0);
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

#[test]
fn test_forward_timeloop_synthetic_uniform_wind_is_deterministic() {
    let config = ForwardTimeLoopConfig {
        start_timestamp: "20240101000000".to_string(),
        end_timestamp: "20240101000002".to_string(),
        timestep_seconds: 1,
        time_bounds_behavior: TimeBoundsBehavior::Strict,
        velocity_to_grid_scale: VelocityToGridScale::IDENTITY,
        ..ForwardTimeLoopConfig::default()
    };

    let releases = vec![ReleaseConfig {
        name: "synthetic_point".to_string(),
        start_time: "20240101000000".to_string(),
        end_time: "20240101000000".to_string(),
        lon: 10.0,
        lat: 5.0,
        z_min: 1.0,
        z_max: 1.0,
        mass_kg: 1.0,
        particle_count: 1,
        raw: BTreeMap::new(),
    }];
    let release_grid = GridDomain {
        xlon0: 0.0,
        ylat0: 0.0,
        dx: 1.0,
        dy: 1.0,
        nx: 64,
        ny: 64,
    };

    let mut driver = match pollster::block_on(ForwardTimeLoopDriver::new(
        config,
        &releases,
        release_grid,
        8,
    )) {
        Ok(driver) => driver,
        Err(TimeLoopError::Gpu(GpuError::NoAdapter)) => return,
        Err(err) => panic!("driver initialization failed: {err}"),
    };

    let wind_grid = synthetic_wind_grid(64, 64, 16);
    let wind_t0 = uniform_wind_field(&wind_grid, 1.0, 0.0, 0.0);
    let wind_t1 = uniform_wind_field(&wind_grid, 3.0, 0.0, 0.0);
    let surface_t0 = synthetic_surface_fields(64, 64);
    let surface_t1 = synthetic_surface_fields(64, 64);
    let met = MetTimeBracket {
        wind_t0: &wind_t0,
        wind_t1: &wind_t1,
        surface_t0: &surface_t0,
        surface_t1: &surface_t1,
        time_t0_seconds: driver.current_time_seconds(),
        time_t1_seconds: driver.current_time_seconds() + 2,
    };

    let forcing = ForwardStepForcing::default();
    let reports =
        pollster::block_on(driver.run_to_end(&met, &forcing)).expect("timeloop run should succeed");

    assert_eq!(reports.len(), 3);
    assert_eq!(reports[0].released_count, 1);
    assert_eq!(reports[1].released_count, 0);
    assert_eq!(reports[2].released_count, 0);

    let expected_alpha = [0.0_f32, 0.5_f32, 1.0_f32];
    for (report, expected) in reports.iter().zip(expected_alpha) {
        assert!(
            (report.interpolation_alpha - expected).abs() < 1.0e-6,
            "unexpected interpolation alpha: got {}, expected {expected}",
            report.interpolation_alpha
        );
        assert!(
            report
                .dry_deposition_probability
                .iter()
                .all(|p| p.abs() < 1.0e-8),
            "dry deposition should be zero for zero forcing"
        );
        assert!(
            report
                .wet_deposition_probability
                .iter()
                .all(|p| p.abs() < 1.0e-8),
            "wet deposition should be zero for zero forcing"
        );
    }

    let released_slot = reports[0]
        .released_slots
        .first()
        .copied()
        .expect("one slot should be released in first step");
    let particle = driver
        .particle_store()
        .get(released_slot)
        .expect("released slot should still exist");
    assert!(particle.is_active());

    // Particle released at x=10 with u=[1..3] over 3 steps.
    // Pure advection yields x≈16, but Langevin turbulence adds stochastic
    // displacement. We verify the particle moved substantially in +x and
    // stayed near the initial y/z within turbulence bounds.
    assert!(
        particle.grid_x() > 12.0 && particle.grid_x() < 20.0,
        "particle should drift in +x direction: grid_x={}",
        particle.grid_x()
    );
    assert!(
        (particle.grid_y() - 5.0).abs() < 2.0,
        "y should stay near initial: grid_y={}",
        particle.grid_y()
    );
    assert!(
        particle.pos_z >= 0.0 && particle.pos_z < 16.0,
        "z should stay in domain: pos_z={}",
        particle.pos_z
    );
    assert!((particle.mass[0] - 1.0).abs() < 1.0e-6);
}

#[test]
fn test_forward_timeloop_optional_spatial_sort_reorders_particle_slots() {
    let config = ForwardTimeLoopConfig {
        start_timestamp: "20240101000000".to_string(),
        end_timestamp: "20240101000001".to_string(),
        timestep_seconds: 1,
        time_bounds_behavior: TimeBoundsBehavior::Strict,
        velocity_to_grid_scale: VelocityToGridScale::IDENTITY,
        spatial_sort: Some(ForwardSpatialSortConfig {
            interval_steps: 1,
            sort_options: ParticleSpatialSortOptions {
                bits_per_axis: 10,
                bounds_mode: SpatialSortBoundsMode::Explicit(SpatialSortBounds {
                    x_min: 0.0,
                    x_max: 64.0,
                    y_min: 0.0,
                    y_max: 64.0,
                    z_min: 0.0,
                    z_max: 1000.0,
                }),
                include_inverse_map: false,
            },
        }),
        ..ForwardTimeLoopConfig::default()
    };

    let releases = vec![
        ReleaseConfig {
            name: "high_x".to_string(),
            start_time: "20240101000000".to_string(),
            end_time: "20240101000000".to_string(),
            lon: 30.0,
            lat: 5.0,
            z_min: 1.0,
            z_max: 1.0,
            mass_kg: 1.0,
            particle_count: 1,
            raw: BTreeMap::new(),
        },
        ReleaseConfig {
            name: "low_x".to_string(),
            start_time: "20240101000000".to_string(),
            end_time: "20240101000000".to_string(),
            lon: 10.0,
            lat: 5.0,
            z_min: 1.0,
            z_max: 1.0,
            mass_kg: 1.0,
            particle_count: 1,
            raw: BTreeMap::new(),
        },
    ];
    let release_grid = GridDomain {
        xlon0: 0.0,
        ylat0: 0.0,
        dx: 1.0,
        dy: 1.0,
        nx: 64,
        ny: 64,
    };

    let mut driver = match pollster::block_on(ForwardTimeLoopDriver::new(
        config,
        &releases,
        release_grid,
        8,
    )) {
        Ok(driver) => driver,
        Err(TimeLoopError::Gpu(GpuError::NoAdapter)) => return,
        Err(err) => panic!("driver initialization failed: {err}"),
    };

    let wind_grid = synthetic_wind_grid(64, 64, 16);
    let wind_t0 = uniform_wind_field(&wind_grid, 0.0, 0.0, 0.0);
    let wind_t1 = uniform_wind_field(&wind_grid, 0.0, 0.0, 0.0);
    let surface_t0 = synthetic_surface_fields(64, 64);
    let surface_t1 = synthetic_surface_fields(64, 64);
    let met = MetTimeBracket {
        wind_t0: &wind_t0,
        wind_t1: &wind_t1,
        surface_t0: &surface_t0,
        surface_t1: &surface_t1,
        time_t0_seconds: driver.current_time_seconds(),
        time_t1_seconds: driver.current_time_seconds() + 1,
    };

    let forcing = ForwardStepForcing::default();
    let reports =
        pollster::block_on(driver.run_to_end(&met, &forcing)).expect("timeloop run should succeed");
    assert_eq!(reports.len(), 2);

    let slot0 = driver.particle_store().get(0).expect("slot 0 exists");
    let slot1 = driver.particle_store().get(1).expect("slot 1 exists");
    assert_eq!(
        slot0.release_point, 1,
        "spatial sort should place low-x particle first"
    );
    assert_eq!(slot1.release_point, 0);
}
