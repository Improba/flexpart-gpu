//! End-to-end physics validation test (Phase E — CI gate).
//!
//! Runs a short but realistic simulation with:
//! - Uniform wind (u=5, v=-3 m/s)
//! - Active Langevin turbulence
//! - PBL boundary reflections
//! - Known release position
//!
//! Verifies:
//! 1. Horizontal advection: center of mass moves in the correct direction
//! 2. PBL confinement: all particles stay within [0, mixing_height]
//! 3. Vertical mixing: particles spread throughout the PBL
//! 4. Mass conservation: total mass unchanged (no deposition)
//! 5. Particle count preservation: all particles remain active

use std::collections::BTreeMap;

use flexpart_gpu::config::ReleaseConfig;
use flexpart_gpu::coords::GridDomain;
use flexpart_gpu::gpu::GpuError;
use flexpart_gpu::io::TimeBoundsBehavior;
use flexpart_gpu::physics::VelocityToGridScale;
use flexpart_gpu::simulation::{
    ForwardStepForcing, ForwardTimeLoopConfig, ForwardTimeLoopDriver, MetTimeBracket,
    TimeLoopError,
};
use flexpart_gpu::wind::{SurfaceFields, WindField3D, WindFieldGrid};
use ndarray::Array1;

const NX: usize = 64;
const NY: usize = 64;
const WIND_NZ: usize = 8;
const DX_DEG: f64 = 0.1;
const DY_DEG: f64 = 0.1;
const XLON0: f64 = 6.0;
const YLAT0: f64 = 6.0;

const RELEASE_LON: f64 = 9.0;
const RELEASE_LAT: f64 = 9.0;
const RELEASE_Z: f64 = 50.0;
const PARTICLE_COUNT: u64 = 500;
const MASS_KG: f64 = 1.0;

const U_WIND: f32 = 5.0;
const V_WIND: f32 = -3.0;
const W_WIND: f32 = 0.0;

const BLH_M: f32 = 1500.0;
const UST: f32 = 0.35;
const WST: f32 = 0.1;
const SSHF: f32 = 40.0;
const SSR: f32 = 220.0;

const DT: i64 = 300;
const N_STEPS: usize = 12;
const SIM_DURATION_S: i64 = DT * N_STEPS as i64;

const WIND_HEIGHTS: [f32; 8] = [0.0, 100.0, 500.0, 1500.0, 3000.0, 5000.0, 10000.0, 20000.0];

const R_EARTH: f64 = 6_371_000.0;
const PI: f32 = std::f32::consts::PI;

fn wind_grid() -> WindFieldGrid {
    WindFieldGrid::new(
        NX, NY, WIND_NZ, WIND_NZ, WIND_NZ,
        DX_DEG as f32, DY_DEG as f32,
        XLON0 as f32, YLAT0 as f32,
        Array1::from_vec(WIND_HEIGHTS.to_vec()),
    )
}

fn uniform_wind(grid: &WindFieldGrid) -> WindField3D {
    let mut field = WindField3D::zeros(grid.nx, grid.ny, grid.nz);
    field.u_ms.fill(U_WIND);
    field.v_ms.fill(V_WIND);
    field.w_ms.fill(W_WIND);
    field
}

fn surface_fields() -> SurfaceFields {
    let mut s = SurfaceFields::zeros(NX, NY);
    s.surface_pressure_pa.fill(101_325.0);
    s.u10_ms.fill(U_WIND);
    s.v10_ms.fill(V_WIND);
    s.temperature_2m_k.fill(289.0);
    s.dewpoint_2m_k.fill(284.0);
    s.sensible_heat_flux_w_m2.fill(SSHF);
    s.solar_radiation_w_m2.fill(SSR);
    s.surface_stress_n_m2.fill(0.2);
    s.friction_velocity_ms.fill(UST);
    s.convective_velocity_scale_ms.fill(WST);
    s.mixing_height_m.fill(BLH_M);
    s.tropopause_height_m.fill(10_000.0);
    s.inv_obukhov_length_per_m.fill(0.0);
    s
}

fn velocity_scale() -> VelocityToGridScale {
    let lat_rad = (RELEASE_LAT as f32) * PI / 180.0;
    let dx_m = R_EARTH * f64::from(lat_rad.cos()) * (DX_DEG * std::f64::consts::PI / 180.0);
    let dy_m = R_EARTH * (DY_DEG * std::f64::consts::PI / 180.0);
    VelocityToGridScale {
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
    }
}

#[test]
fn physics_validation_advection_turbulence_pbl() {
    let release_grid = GridDomain {
        xlon0: XLON0,
        ylat0: YLAT0,
        dx: DX_DEG,
        dy: DY_DEG,
        nx: NX,
        ny: NY,
    };

    let releases = vec![ReleaseConfig {
        name: "validation".to_string(),
        start_time: "20240101000000".to_string(),
        end_time: "20240101000000".to_string(),
        lon: RELEASE_LON,
        lat: RELEASE_LAT,
        z_min: RELEASE_Z,
        z_max: RELEASE_Z,
        mass_kg: MASS_KG,
        particle_count: PARTICLE_COUNT,
        raw: BTreeMap::new(),
    }];

    let config = ForwardTimeLoopConfig {
        start_timestamp: "20240101000000".to_string(),
        end_timestamp: format!(
            "20240101{:02}{:02}00",
            SIM_DURATION_S / 3600,
            (SIM_DURATION_S % 3600) / 60
        ),
        timestep_seconds: DT,
        time_bounds_behavior: TimeBoundsBehavior::Clamp,
        velocity_to_grid_scale: velocity_scale(),
        sync_particle_store_each_step: true,
        collect_deposition_probabilities_each_step: false,
        ..ForwardTimeLoopConfig::default()
    };

    let mut driver = match pollster::block_on(ForwardTimeLoopDriver::new(
        config,
        &releases,
        release_grid,
        PARTICLE_COUNT as usize,
    )) {
        Ok(d) => d,
        Err(TimeLoopError::Gpu(GpuError::NoAdapter)) => {
            eprintln!("No GPU adapter — skipping physics validation test");
            return;
        }
        Err(err) => panic!("driver init failed: {err}"),
    };

    let wg = wind_grid();
    let wind_t0 = uniform_wind(&wg);
    let wind_t1 = uniform_wind(&wg);
    let surface_t0 = surface_fields();
    let surface_t1 = surface_fields();

    let start_secs = 1_704_067_200_i64;
    let met = MetTimeBracket {
        wind_t0: &wind_t0,
        wind_t1: &wind_t1,
        surface_t0: &surface_t0,
        surface_t1: &surface_t1,
        time_t0_seconds: start_secs,
        time_t1_seconds: start_secs + SIM_DURATION_S,
    };

    let forcing = ForwardStepForcing::default();
    let reports = pollster::block_on(driver.run_to_end(&met, &forcing))
        .expect("simulation should complete");

    assert_eq!(
        reports.len(),
        N_STEPS + 1,
        "should run correct number of steps"
    );

    let store = driver.particle_store();
    let particles = store.as_slice();
    let active: Vec<_> = particles.iter().filter(|p| p.is_active()).collect();
    let active_count = active.len();

    // --- 1. Particle count preservation ---
    assert_eq!(
        active_count, PARTICLE_COUNT as usize,
        "all particles should remain active"
    );

    // --- 2. Mass conservation (no deposition in this test) ---
    let total_mass: f64 = active.iter().map(|p| f64::from(p.mass[0])).sum();
    let expected_mass = MASS_KG;
    let mass_rel_err = (total_mass - expected_mass).abs() / expected_mass;
    assert!(
        mass_rel_err < 1.0e-5,
        "mass should be conserved: got {total_mass:.9e}, expected {expected_mass:.9e}, \
         relative error {mass_rel_err:.2e}"
    );

    // --- 3. Horizontal advection direction ---
    let release_grid_x = (RELEASE_LON - XLON0) / DX_DEG;
    let release_grid_y = (RELEASE_LAT - YLAT0) / DY_DEG;
    let mean_gx: f64 =
        active.iter().map(|p| p.grid_x()).sum::<f64>() / active_count as f64;
    let mean_gy: f64 =
        active.iter().map(|p| p.grid_y()).sum::<f64>() / active_count as f64;

    // u=+5 m/s → particles should move in +x (eastward)
    assert!(
        mean_gx > release_grid_x,
        "COM should move east (u>0): mean_gx={mean_gx:.2}, release_x={release_grid_x:.2}"
    );
    // v=-3 m/s → particles should move in -y (southward)
    assert!(
        mean_gy < release_grid_y,
        "COM should move south (v<0): mean_gy={mean_gy:.2}, release_y={release_grid_y:.2}"
    );

    let mean_gx_f32 = mean_gx as f32;
    let mean_gy_f32 = mean_gy as f32;
    let release_gx_f32 = release_grid_x as f32;
    let release_gy_f32 = release_grid_y as f32;

    // --- 4. PBL confinement ---
    let zs: Vec<f32> = active.iter().map(|p| p.pos_z).collect();
    let min_z = zs.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_z = zs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean_z: f32 = zs.iter().sum::<f32>() / zs.len() as f32;
    let var_z: f32 = zs.iter().map(|z| (z - mean_z).powi(2)).sum::<f32>() / zs.len() as f32;
    let std_z = var_z.sqrt();

    assert!(
        min_z >= 0.0,
        "no particle should be below ground: min_z={min_z}"
    );
    assert!(
        max_z <= BLH_M + 1.0,
        "all particles should be within PBL: max_z={max_z}, BLH={BLH_M}"
    );

    // --- 5. Vertical mixing ---
    // After 12 steps of 300s = 1h with active turbulence and BLH=1500m,
    // particles released at 50m should have started mixing vertically.
    // The mean z should be above the release height (turbulence drives mixing).
    assert!(
        mean_z > RELEASE_Z as f32 + 10.0,
        "turbulence should mix particles above release height: mean_z={mean_z:.1}m, \
         release_z={RELEASE_Z}m"
    );
    // Standard deviation should be non-trivial — particles are spreading
    assert!(
        std_z > 10.0,
        "particles should have vertical spread: std_z={std_z:.1}m"
    );

    eprintln!("=== Physics validation PASS ===");
    eprintln!("  active particles: {active_count}");
    eprintln!("  mass relative error: {mass_rel_err:.2e}");
    eprintln!("  COM grid: ({mean_gx_f32:.2}, {mean_gy_f32:.2}), release: ({release_gx_f32:.1}, {release_gy_f32:.1})");
    eprintln!("  z stats: min={min_z:.1}m, max={max_z:.1}m, mean={mean_z:.1}m, std={std_z:.1}m");
    eprintln!("  PBL: [0, {BLH_M}]m");
}
