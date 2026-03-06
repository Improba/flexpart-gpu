use std::collections::BTreeMap;
use std::f32::consts::PI;
use std::path::PathBuf;

use flexpart_gpu::config::ReleaseConfig;
use flexpart_gpu::coords::GridDomain;
use flexpart_gpu::gpu::{
    ConcentrationGridShape, ConcentrationGriddingParams, GpuError, MAX_OUTPUT_LEVELS,
};
use flexpart_gpu::io::TimeBoundsBehavior;
use flexpart_gpu::physics::VelocityToGridScale;
use flexpart_gpu::simulation::{
    ForwardStepForcing, ForwardTimeLoopConfig, ForwardTimeLoopDriver, MetTimeBracket,
    ParticleForcingField, TimeLoopError,
};
use flexpart_gpu::wind::{SurfaceFields, WindField3D, WindFieldGrid};
use ndarray::Array1;
use serde::Serialize;

const R_EARTH: f64 = 6_371_000.0;

const U_WIND: f32 = 5.0;
const V_WIND: f32 = -3.0;
const W_WIND: f32 = 0.0;

const OUT_NX: usize = 32;
const OUT_NY: usize = 32;
const OUT_NZ: usize = 10;
const OUT_DX: f64 = 0.1;
const OUT_DY: f64 = 0.1;
const OUTLON0: f64 = 9.5;
const OUTLAT0: f64 = 8.5;
const OUTHEIGHTS: [f32; OUT_NZ] = [
    100.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 5000.0,
];

const WIND_NZ: usize = 8;
const WIND_HEIGHTS: [f32; WIND_NZ] = [
    50.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0,
];

const RELEASE_LON: f64 = 10.0;
const RELEASE_LAT: f64 = 10.0;
const RELEASE_Z: f64 = 50.0;
const PARTICLE_COUNT: u64 = 10_000;
const MASS_KG: f64 = 1.0;

const START_TS: &str = "20240101000000";
const END_TS: &str = "20240101060000";
const DT_SECONDS: i64 = 900;

const SP_PA: f32 = 101_325.0;
const T2M_K: f32 = 289.0;
const TD2M_K: f32 = 284.0;
const SSHF_W_M2: f32 = 40.0;
const SSR_W_M2: f32 = 220.0;
const BLH_M: f32 = 3000.0;

#[derive(Serialize)]
struct ValidationOutput {
    grid: GridInfo,
    particle_count_per_cell: Vec<u32>,
    concentration_mass_kg: Vec<f32>,
    total_particles_active: usize,
    total_steps: usize,
    particle_z_stats: Option<ParticleZStats>,
}

#[derive(Serialize)]
struct ParticleZStats {
    count: usize,
    min_m: f32,
    max_m: f32,
    mean_m: f32,
    std_m: f32,
    lon_mean: f64,
    lat_mean: f64,
    lon_std: f64,
    lat_std: f64,
}

#[derive(Serialize)]
struct GridInfo {
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    xlon0: f64,
    ylat0: f64,
    heights_m: Vec<f32>,
}

fn main() {
    env_logger::init();

    let output_path: PathBuf = std::env::var("OUTPUT_PATH")
        .unwrap_or_else(|_| "target/validation/gpu_concentration.json".to_string())
        .into();
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).expect("create output directory");
    }

    let particle_count: u64 = std::env::var("PARTICLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(PARTICLE_COUNT);

    eprintln!("=== fortran-validation (GPU) ===");
    eprintln!("domain: {OUT_NX}x{OUT_NY}x{OUT_NZ}, dx={OUT_DX}°, dy={OUT_DY}°");
    eprintln!("origin: ({OUTLON0}, {OUTLAT0})");
    eprintln!("wind: u={U_WIND}, v={V_WIND}, w={W_WIND} m/s");
    eprintln!("release: ({RELEASE_LON}, {RELEASE_LAT}), z={RELEASE_Z}m, {particle_count} particles");
    eprintln!("simulation: {START_TS} → {END_TS}, dt={DT_SECONDS}s");

    let center_lat = RELEASE_LAT as f32;
    let lat_rad = (center_lat as f64) * std::f64::consts::PI / 180.0;
    let dx_m = R_EARTH * lat_rad.cos() * (OUT_DX * PI as f64 / 180.0);
    let dy_m = R_EARTH * (OUT_DY * PI as f64 / 180.0);
    let velocity_scale = VelocityToGridScale {
        x_grid_per_meter: (1.0 / dx_m) as f32,
        y_grid_per_meter: (1.0 / dy_m) as f32,
        z_grid_per_meter: 1.0,
        level_heights_m: {
            let mut h = [0.0_f32; 16];
            for (i, &v) in WIND_HEIGHTS.iter().enumerate() {
                h[i] = v;
            }
            h
        },
    };
    eprintln!(
        "velocity scale: x={:.3e}, y={:.3e} grid/m",
        velocity_scale.x_grid_per_meter, velocity_scale.y_grid_per_meter
    );

    let release_grid = GridDomain {
        xlon0: OUTLON0,
        ylat0: OUTLAT0,
        dx: OUT_DX,
        dy: OUT_DY,
        nx: OUT_NX,
        ny: OUT_NY,
    };

    let releases = vec![ReleaseConfig {
        name: "comparison".to_string(),
        start_time: START_TS.to_string(),
        end_time: START_TS.to_string(),
        lon: RELEASE_LON,
        lat: RELEASE_LAT,
        z_min: RELEASE_Z,
        z_max: RELEASE_Z,
        mass_kg: MASS_KG,
        particle_count,
        raw: BTreeMap::new(),
    }];

    let config = ForwardTimeLoopConfig {
        start_timestamp: START_TS.to_string(),
        end_timestamp: END_TS.to_string(),
        timestep_seconds: DT_SECONDS,
        time_bounds_behavior: TimeBoundsBehavior::Clamp,
        velocity_to_grid_scale: velocity_scale,
        sync_particle_store_each_step: true,
        collect_deposition_probabilities_each_step: false,
        ..ForwardTimeLoopConfig::default()
    };

    eprintln!("initializing GPU driver...");
    let mut driver = match pollster::block_on(ForwardTimeLoopDriver::new(
        config,
        &releases,
        release_grid,
        particle_count as usize,
    )) {
        Ok(d) => d,
        Err(TimeLoopError::Gpu(GpuError::NoAdapter)) => {
            eprintln!("ERROR: no GPU adapter found");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("GPU driver ready");

    let wind_grid = WindFieldGrid::new(
        OUT_NX, OUT_NY, WIND_NZ, WIND_NZ, WIND_NZ,
        OUT_DX as f32, OUT_DY as f32,
        OUTLON0 as f32, OUTLAT0 as f32,
        Array1::from_vec(WIND_HEIGHTS.to_vec()),
    );
    let wind_t0 = build_uniform_wind(&wind_grid, U_WIND, V_WIND, W_WIND);
    let wind_t1 = build_uniform_wind(&wind_grid, U_WIND, V_WIND, W_WIND);
    let surface_t0 = build_surface_fields(OUT_NX, OUT_NY);
    let surface_t1 = build_surface_fields(OUT_NX, OUT_NY);

    let start_secs = 1_704_067_200_i64; // 2024-01-01 00:00:00 UTC
    let end_secs = start_secs + 6 * 3600;
    let met = MetTimeBracket {
        wind_t0: &wind_t0,
        wind_t1: &wind_t1,
        surface_t0: &surface_t0,
        surface_t1: &surface_t1,
        time_t0_seconds: start_secs,
        time_t1_seconds: end_secs,
    };

    let forcing = ForwardStepForcing {
        dry_deposition_velocity_m_s: ParticleForcingField::Uniform(0.0),
        wet_scavenging_coefficient_s_inv: ParticleForcingField::Uniform(0.0),
        wet_precipitating_fraction: ParticleForcingField::Uniform(0.0),
        rho_grad_over_rho: 0.0,
    };

    eprintln!("running simulation...");
    let reports = pollster::block_on(driver.run_to_end(&met, &forcing))
        .expect("simulation failed");

    let total_steps = reports.len();
    let active = reports.last().map(|r| r.active_particle_count).unwrap_or(0);
    eprintln!("completed {total_steps} steps, {active} active particles");

    let z_stats = {
        let store = driver.particle_store();
        let particles = store.as_slice();
        let active_particles: Vec<_> = particles.iter()
            .filter(|p| p.is_active())
            .collect();
        if !active_particles.is_empty() {
            let n = active_particles.len() as f64;
            let active_zs: Vec<f32> = active_particles.iter().map(|p| p.pos_z).collect();
            let min_z = active_zs.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_z = active_zs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean_z: f32 = active_zs.iter().sum::<f32>() / active_zs.len() as f32;
            let var_z: f32 = active_zs.iter().map(|z| (z - mean_z).powi(2)).sum::<f32>() / active_zs.len() as f32;
            let std_z = var_z.sqrt();

            let lons: Vec<f64> = active_particles.iter()
                .map(|p| OUTLON0 + (p.cell_x as f64 + p.pos_x as f64) * OUT_DX)
                .collect();
            let lats: Vec<f64> = active_particles.iter()
                .map(|p| OUTLAT0 + (p.cell_y as f64 + p.pos_y as f64) * OUT_DY)
                .collect();
            let lon_mean: f64 = lons.iter().sum::<f64>() / n;
            let lat_mean: f64 = lats.iter().sum::<f64>() / n;
            let lon_var: f64 = lons.iter().map(|l| (l - lon_mean).powi(2)).sum::<f64>() / n;
            let lat_var: f64 = lats.iter().map(|l| (l - lat_mean).powi(2)).sum::<f64>() / n;

            eprintln!("particle z stats: n={}, min={:.1}m, max={:.1}m, mean={:.1}m, std={:.1}m",
                active_zs.len(), min_z, max_z, mean_z, std_z);
            eprintln!("particle h stats: lon_mean={:.4}°, lat_mean={:.4}°, lon_std={:.4}°, lat_std={:.4}°",
                lon_mean, lat_mean, lon_var.sqrt(), lat_var.sqrt());
            Some(ParticleZStats {
                count: active_zs.len(),
                min_m: min_z,
                max_m: max_z,
                mean_m: mean_z,
                std_m: std_z,
                lon_mean,
                lat_mean,
                lon_std: lon_var.sqrt(),
                lat_std: lat_var.sqrt(),
            })
        } else {
            None
        }
    };

    eprintln!("accumulating concentration grid...");
    let conc = pollster::block_on(driver.accumulate_concentration_grid(
        ConcentrationGridShape {
            nx: OUT_NX,
            ny: OUT_NY,
            nz: OUT_NZ,
        },
        ConcentrationGriddingParams {
            species_index: 0,
            mass_scale: 1_000_000.0,
            outheights: {
                let mut h = [0.0_f32; MAX_OUTPUT_LEVELS];
                for (i, &v) in OUTHEIGHTS.iter().enumerate() {
                    h[i] = v;
                }
                h
            },
        },
    ))
    .expect("concentration gridding failed");

    let total_count: u32 = conc.particle_count_per_cell.iter().sum();
    eprintln!("total gridded particles: {total_count}");

    let output = ValidationOutput {
        grid: GridInfo {
            nx: OUT_NX,
            ny: OUT_NY,
            nz: OUT_NZ,
            dx: OUT_DX,
            dy: OUT_DY,
            xlon0: OUTLON0,
            ylat0: OUTLAT0,
            heights_m: OUTHEIGHTS.to_vec(),
        },
        particle_count_per_cell: conc.particle_count_per_cell,
        concentration_mass_kg: conc.concentration_mass_kg,
        total_particles_active: active,
        total_steps,
        particle_z_stats: z_stats,
    };

    let json = serde_json::to_string_pretty(&output).expect("serialize output");
    std::fs::write(&output_path, &json).expect("write output file");
    eprintln!("wrote {}", output_path.display());

    std::process::exit(0);
}

fn build_uniform_wind(grid: &WindFieldGrid, u: f32, v: f32, w: f32) -> WindField3D {
    let mut field = WindField3D::zeros(grid.nx, grid.ny, grid.nz);
    field.u_ms.fill(u);
    field.v_ms.fill(v);
    field.w_ms.fill(w);
    field.temperature_k.fill(285.0);
    field.specific_humidity.fill(0.005);
    field.pressure_pa.fill(100_000.0);
    field.air_density_kg_m3.fill(1.2);
    field.density_gradient_kg_m2.fill(-0.0008);
    field
}

fn build_surface_fields(nx: usize, ny: usize) -> SurfaceFields {
    let mut s = SurfaceFields::zeros(nx, ny);
    s.surface_pressure_pa.fill(SP_PA);
    s.u10_ms.fill(U_WIND);
    s.v10_ms.fill(V_WIND);
    s.temperature_2m_k.fill(T2M_K);
    s.dewpoint_2m_k.fill(TD2M_K);
    s.sensible_heat_flux_w_m2.fill(SSHF_W_M2);
    s.solar_radiation_w_m2.fill(SSR_W_M2);
    s.surface_stress_n_m2.fill(0.2);
    s.friction_velocity_ms.fill(0.35);
    s.convective_velocity_scale_ms.fill(0.1);
    s.mixing_height_m.fill(BLH_M);
    s.tropopause_height_m.fill(10_000.0);
    s.inv_obukhov_length_per_m.fill(0.0);
    s
}
