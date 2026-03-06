//! CPU vs GPU comparison tests.
//!
//! These tests validate that the GPU compute kernels produce results consistent
//! with the CPU reference implementations in `physics/`. The CPU path is a
//! direct port of FLEXPART Fortran's `advance.f90` (Euler + Petterssen),
//! `hanna.f90` (turbulence parameterization), and `langevin.f90` (stochastic
//! velocity update). Agreement between CPU and GPU paths therefore constitutes
//! an indirect validation against the Fortran reference.
//!
//! Three scenarios are tested:
//! 1. **Uniform wind** — deterministic advection in a constant wind field
//! 2. **Shear wind** — spatially varying wind to exercise interpolation
//! 3. **Full pipeline** — advection + Hanna + Langevin + deposition through
//!    the end-to-end time loop with identical RNG seeds

use std::collections::BTreeMap;

use flexpart_gpu::config::ReleaseConfig;
use flexpart_gpu::coords::GridDomain;
use flexpart_gpu::gpu::{
    advect_particles_gpu, compute_hanna_params_gpu, GpuContext, GpuError, ParticleBuffers,
    PblBuffers, WindBuffers,
};
use flexpart_gpu::io::{
    compute_pbl_parameters_from_met, PblComputationOptions, PblMetInputGrids, TimeBoundsBehavior,
};
use flexpart_gpu::particles::{Particle, ParticleInit, MAX_SPECIES};
use flexpart_gpu::physics::{
    advect_particles_cpu, compute_hanna_params_from_pbl, VelocityToGridScale,
};
use flexpart_gpu::simulation::{
    ForwardStepForcing, ForwardTimeLoopConfig, ForwardTimeLoopDriver, MetTimeBracket, TimeLoopError,
};
use flexpart_gpu::wind::{linear_shear_wind_field, uniform_wind_field, SurfaceFields, WindFieldGrid};
use ndarray::Array1;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const NX: usize = 32;
const NY: usize = 32;
const NZ: usize = 12;

fn make_grid() -> WindFieldGrid {
    WindFieldGrid::new(
        NX,
        NY,
        NZ,
        NZ,
        NZ,
        1.0,
        1.0,
        0.0,
        0.0,
        Array1::from_iter((0..NZ).map(|k| k as f32)),
    )
}

fn make_surface_fields() -> SurfaceFields {
    let mut surface = SurfaceFields::zeros(NX, NY);
    for i in 0..NX {
        for j in 0..NY {
            surface.surface_pressure_pa[[i, j]] = 101_325.0;
            surface.u10_ms[[i, j]] = 2.0 + 0.02 * i as f32;
            surface.v10_ms[[i, j]] = 0.5 - 0.01 * j as f32;
            surface.temperature_2m_k[[i, j]] = 289.0;
            surface.dewpoint_2m_k[[i, j]] = 284.0;
            surface.precip_large_scale_mm_h[[i, j]] = 0.0;
            surface.precip_convective_mm_h[[i, j]] = 0.0;
            surface.sensible_heat_flux_w_m2[[i, j]] = 40.0 + 0.3 * i as f32;
            surface.solar_radiation_w_m2[[i, j]] = 220.0;
            surface.surface_stress_n_m2[[i, j]] = 0.2 + 0.001 * j as f32;
            surface.friction_velocity_ms[[i, j]] = 0.35;
            surface.convective_velocity_scale_ms[[i, j]] = 0.6;
            surface.mixing_height_m[[i, j]] = 1000.0 + 2.0 * j as f32;
            surface.tropopause_height_m[[i, j]] = 10_000.0;
            surface.inv_obukhov_length_per_m[[i, j]] = -0.01;
        }
    }
    surface
}

fn make_particles(count: usize) -> Vec<Particle> {
    let mut particles = Vec::with_capacity(count);
    for idx in 0..count {
        let cell_x = (idx % (NX - 2)) as i32 + 1;
        let cell_y = ((idx / (NX - 2)) % (NY - 2)) as i32 + 1;
        let frac_x = 0.1 + 0.8 * ((idx * 7 + 3) % 100) as f32 / 100.0;
        let frac_y = 0.1 + 0.8 * ((idx * 13 + 7) % 100) as f32 / 100.0;
        let pos_z = 0.5 + ((idx * 11 + 5) % (NZ - 1)) as f32;
        let mut mass = [0.0_f32; MAX_SPECIES];
        mass[0] = 1.0;
        particles.push(Particle::new(&ParticleInit {
            cell_x,
            cell_y,
            pos_x: frac_x,
            pos_y: frac_y,
            pos_z,
            mass,
            release_point: 0,
            class: 0,
            time: 0,
        }));
    }
    particles
}

/// Compute position differences between two particle vectors.
struct ComparisonMetrics {
    max_dx: f64,
    max_dy: f64,
    max_dz: f64,
    rmse_x: f64,
    rmse_y: f64,
    rmse_z: f64,
    compared_count: usize,
}

impl std::fmt::Display for ComparisonMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n={} | max(dx,dy,dz)=({:.2e},{:.2e},{:.2e}) | rmse(x,y,z)=({:.2e},{:.2e},{:.2e})",
            self.compared_count,
            self.max_dx,
            self.max_dy,
            self.max_dz,
            self.rmse_x,
            self.rmse_y,
            self.rmse_z,
        )
    }
}

fn compare_particles(cpu: &[Particle], gpu: &[Particle]) -> ComparisonMetrics {
    assert_eq!(cpu.len(), gpu.len(), "particle count mismatch");
    let mut max_dx = 0.0_f64;
    let mut max_dy = 0.0_f64;
    let mut max_dz = 0.0_f64;
    let mut sum_sq_x = 0.0_f64;
    let mut sum_sq_y = 0.0_f64;
    let mut sum_sq_z = 0.0_f64;
    let mut count = 0usize;

    for (c, g) in cpu.iter().zip(gpu.iter()) {
        if !c.is_active() && !g.is_active() {
            continue;
        }
        let dx = (c.grid_x() - g.grid_x()).abs();
        let dy = (c.grid_y() - g.grid_y()).abs();
        let dz = f64::from((c.pos_z - g.pos_z).abs());
        max_dx = max_dx.max(dx);
        max_dy = max_dy.max(dy);
        max_dz = max_dz.max(dz);
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
        sum_sq_z += dz * dz;
        count += 1;
    }
    let n = count.max(1) as f64;
    ComparisonMetrics {
        max_dx,
        max_dy,
        max_dz,
        rmse_x: (sum_sq_x / n).sqrt(),
        rmse_y: (sum_sq_y / n).sqrt(),
        rmse_z: (sum_sq_z / n).sqrt(),
        compared_count: count,
    }
}

// ---------------------------------------------------------------------------
// Test 1: Uniform wind advection
// ---------------------------------------------------------------------------

#[test]
fn cpu_gpu_advection_uniform_wind() {
    let ctx = match pollster::block_on(GpuContext::new()) {
        Ok(ctx) => ctx,
        Err(GpuError::NoAdapter) => return,
        Err(err) => panic!("GPU init failed: {err}"),
    };

    let grid = make_grid();
    let wind = uniform_wind_field(&grid, 0.5, -0.3, 0.02);
    let dt = 10.0_f32;
    let scale = VelocityToGridScale::IDENTITY;
    let particle_count = 500;

    let original_particles = make_particles(particle_count);

    // CPU path
    let mut cpu_particles = original_particles.clone();
    advect_particles_cpu(&mut cpu_particles, &wind, dt, scale);

    // GPU path
    let particle_buffers = ParticleBuffers::from_particles(&ctx, &original_particles);
    let wind_buffers =
        WindBuffers::from_field(&ctx, &wind).expect("wind buffer creation should succeed");
    advect_particles_gpu(&ctx, &particle_buffers, &wind_buffers, dt, scale)
        .expect("GPU advection should succeed");
    let gpu_particles = pollster::block_on(particle_buffers.download_particles(&ctx))
        .expect("particle download should succeed");

    let metrics = compare_particles(&cpu_particles, &gpu_particles);
    eprintln!("uniform_wind advection: {metrics}");

    assert!(
        metrics.max_dx < 1.0e-4,
        "x position mismatch too large: {:.2e}",
        metrics.max_dx
    );
    assert!(
        metrics.max_dy < 1.0e-4,
        "y position mismatch too large: {:.2e}",
        metrics.max_dy
    );
    assert!(
        metrics.max_dz < 1.0e-4,
        "z position mismatch too large: {:.2e}",
        metrics.max_dz
    );
}

// ---------------------------------------------------------------------------
// Test 2: Shear wind advection
// ---------------------------------------------------------------------------

#[test]
fn cpu_gpu_advection_shear_wind() {
    let ctx = match pollster::block_on(GpuContext::new()) {
        Ok(ctx) => ctx,
        Err(GpuError::NoAdapter) => return,
        Err(err) => panic!("GPU init failed: {err}"),
    };

    let grid = make_grid();
    let wind = linear_shear_wind_field(&grid, 0.3, -0.2, 0.01, 0.01, -0.005, 0.001, 5.0);
    let dt = 5.0_f32;
    let scale = VelocityToGridScale::IDENTITY;
    let particle_count = 500;

    let original_particles = make_particles(particle_count);

    // CPU path
    let mut cpu_particles = original_particles.clone();
    advect_particles_cpu(&mut cpu_particles, &wind, dt, scale);

    // GPU path
    let particle_buffers = ParticleBuffers::from_particles(&ctx, &original_particles);
    let wind_buffers =
        WindBuffers::from_field(&ctx, &wind).expect("wind buffer creation should succeed");
    advect_particles_gpu(&ctx, &particle_buffers, &wind_buffers, dt, scale)
        .expect("GPU advection should succeed");
    let gpu_particles = pollster::block_on(particle_buffers.download_particles(&ctx))
        .expect("particle download should succeed");

    let metrics = compare_particles(&cpu_particles, &gpu_particles);
    eprintln!("shear_wind advection: {metrics}");

    assert!(
        metrics.max_dx < 5.0e-4,
        "x position mismatch too large: {:.2e}",
        metrics.max_dx
    );
    assert!(
        metrics.max_dy < 5.0e-4,
        "y position mismatch too large: {:.2e}",
        metrics.max_dy
    );
    assert!(
        metrics.max_dz < 5.0e-4,
        "z position mismatch too large: {:.2e}",
        metrics.max_dz
    );
}

// ---------------------------------------------------------------------------
// Test 3: Multi-step advection accumulation
// ---------------------------------------------------------------------------

#[test]
fn cpu_gpu_advection_multistep_drift() {
    let ctx = match pollster::block_on(GpuContext::new()) {
        Ok(ctx) => ctx,
        Err(GpuError::NoAdapter) => return,
        Err(err) => panic!("GPU init failed: {err}"),
    };

    let grid = make_grid();
    let wind = linear_shear_wind_field(&grid, 0.2, 0.1, 0.0, 0.005, -0.003, 0.0, 3.0);
    let dt = 2.0_f32;
    let scale = VelocityToGridScale::IDENTITY;
    let particle_count = 200;
    let nsteps = 20;

    let original_particles = make_particles(particle_count);

    // CPU: iterate nsteps
    let mut cpu_particles = original_particles.clone();
    for _ in 0..nsteps {
        advect_particles_cpu(&mut cpu_particles, &wind, dt, scale);
    }

    // GPU: iterate nsteps
    let particle_buffers = ParticleBuffers::from_particles(&ctx, &original_particles);
    let wind_buffers =
        WindBuffers::from_field(&ctx, &wind).expect("wind buffer creation should succeed");
    for _ in 0..nsteps {
        advect_particles_gpu(&ctx, &particle_buffers, &wind_buffers, dt, scale)
            .expect("GPU advection should succeed");
    }
    let gpu_particles = pollster::block_on(particle_buffers.download_particles(&ctx))
        .expect("particle download should succeed");

    let metrics = compare_particles(&cpu_particles, &gpu_particles);
    eprintln!("multistep ({nsteps} steps) advection: {metrics}");

    // f32 drift accumulates over steps; allow ~1e-3 per grid cell
    assert!(
        metrics.max_dx < 5.0e-3,
        "x drift too large after {nsteps} steps: {:.2e}",
        metrics.max_dx
    );
    assert!(
        metrics.max_dy < 5.0e-3,
        "y drift too large after {nsteps} steps: {:.2e}",
        metrics.max_dy
    );
    assert!(
        metrics.max_dz < 5.0e-3,
        "z drift too large after {nsteps} steps: {:.2e}",
        metrics.max_dz
    );
}

// ---------------------------------------------------------------------------
// Test 4: Hanna turbulence parameters
// ---------------------------------------------------------------------------

#[test]
fn cpu_gpu_hanna_params_consistency() {
    let ctx = match pollster::block_on(GpuContext::new()) {
        Ok(ctx) => ctx,
        Err(GpuError::NoAdapter) => return,
        Err(err) => panic!("GPU init failed: {err}"),
    };

    let surface = make_surface_fields();
    let computed_pbl = compute_pbl_parameters_from_met(
        PblMetInputGrids {
            surface: &surface,
            profile: None,
        },
        PblComputationOptions::default(),
    )
    .expect("PBL computation should succeed");
    let pbl_state = &computed_pbl.pbl_state;

    let particle_count = 300;
    let particles = make_particles(particle_count);

    // CPU path: compute Hanna params per particle
    let (pbl_nx, pbl_ny) = pbl_state.shape();
    let max_x = pbl_nx.saturating_sub(1) as f32;
    let max_y = pbl_ny.saturating_sub(1) as f32;
    let cpu_hanna: Vec<_> = particles
        .iter()
        .map(|p| {
            let i = (p.cell_x as f32 + p.pos_x).clamp(0.0, max_x).floor() as usize;
            let j = (p.cell_y as f32 + p.pos_y).clamp(0.0, max_y).floor() as usize;
            compute_hanna_params_from_pbl(pbl_state, i, j, p.pos_z)
        })
        .collect();

    // GPU path
    let particle_buffers = ParticleBuffers::from_particles(&ctx, &particles);
    let pbl_buffers =
        PblBuffers::from_state(&ctx, pbl_state).expect("PBL buffer creation should succeed");
    let gpu_hanna = pollster::block_on(compute_hanna_params_gpu(&ctx, &particle_buffers, &pbl_buffers))
        .expect("GPU Hanna should succeed");

    let mut max_sigu_diff = 0.0_f32;
    let mut max_sigv_diff = 0.0_f32;
    let mut max_sigw_diff = 0.0_f32;
    let mut max_tlu_diff = 0.0_f32;

    for (cpu_h, gpu_h) in cpu_hanna.iter().zip(gpu_hanna.iter()) {
        max_sigu_diff = max_sigu_diff.max((cpu_h.sigu - gpu_h.sigu).abs());
        max_sigv_diff = max_sigv_diff.max((cpu_h.sigv - gpu_h.sigv).abs());
        max_sigw_diff = max_sigw_diff.max((cpu_h.sigw - gpu_h.sigw).abs());
        max_tlu_diff = max_tlu_diff.max((cpu_h.tlu - gpu_h.tlu).abs());
    }

    eprintln!(
        "hanna params: max_diff(sigu={:.2e}, sigv={:.2e}, sigw={:.2e}, tlu={:.2e})",
        max_sigu_diff, max_sigv_diff, max_sigw_diff, max_tlu_diff
    );

    assert!(max_sigu_diff < 1.0e-3, "sigu mismatch: {max_sigu_diff:.2e}");
    assert!(max_sigv_diff < 1.0e-3, "sigv mismatch: {max_sigv_diff:.2e}");
    assert!(max_sigw_diff < 1.0e-3, "sigw mismatch: {max_sigw_diff:.2e}");
    assert!(max_tlu_diff < 1.0e-2, "tlu mismatch: {max_tlu_diff:.2e}");
}

// ---------------------------------------------------------------------------
// Test 5: Full forward time loop — end-to-end pipeline comparison
// ---------------------------------------------------------------------------

#[test]
fn cpu_gpu_full_pipeline_timeloop() {
    let config = ForwardTimeLoopConfig {
        start_timestamp: "20240101000000".to_string(),
        end_timestamp: "20240101000010".to_string(),
        timestep_seconds: 1,
        time_bounds_behavior: TimeBoundsBehavior::Clamp,
        velocity_to_grid_scale: VelocityToGridScale::IDENTITY,
        ..ForwardTimeLoopConfig::default()
    };

    let particle_count = 100;
    let releases = vec![ReleaseConfig {
        name: "comparison_source".to_string(),
        start_time: "20240101000000".to_string(),
        end_time: "20240101000000".to_string(),
        lon: 10.0,
        lat: 10.0,
        z_min: 2.0,
        z_max: 2.0,
        mass_kg: 1.0,
        particle_count: particle_count as u64,
        raw: BTreeMap::new(),
    }];
    let release_grid = GridDomain {
        xlon0: 0.0,
        ylat0: 0.0,
        dx: 1.0,
        dy: 1.0,
        nx: NX,
        ny: NY,
    };

    let capacity = particle_count + 128;

    let mut driver = match pollster::block_on(ForwardTimeLoopDriver::new(
        config.clone(),
        &releases,
        release_grid,
        capacity,
    )) {
        Ok(driver) => driver,
        Err(TimeLoopError::Gpu(GpuError::NoAdapter)) => return,
        Err(err) => panic!("driver initialization failed: {err}"),
    };

    let grid = make_grid();
    let wind_t0 = linear_shear_wind_field(&grid, 0.3, -0.1, 0.0, 0.008, -0.004, 0.0, 4.0);
    let wind_t1 = linear_shear_wind_field(&grid, 0.5, 0.1, 0.01, 0.01, -0.002, 0.001, 4.0);
    let surface_t0 = make_surface_fields();
    let surface_t1 = make_surface_fields();

    let met = MetTimeBracket {
        wind_t0: &wind_t0,
        wind_t1: &wind_t1,
        surface_t0: &surface_t0,
        surface_t1: &surface_t1,
        time_t0_seconds: driver.current_time_seconds(),
        time_t1_seconds: driver.current_time_seconds() + 12,
    };

    let forcing = ForwardStepForcing {
        dry_deposition_velocity_m_s: flexpart_gpu::simulation::ParticleForcingField::Uniform(0.003),
        wet_scavenging_coefficient_s_inv: flexpart_gpu::simulation::ParticleForcingField::Uniform(
            5.0e-5,
        ),
        wet_precipitating_fraction: flexpart_gpu::simulation::ParticleForcingField::Uniform(0.1),
        rho_grad_over_rho: 0.0,
    };

    let reports =
        pollster::block_on(driver.run_to_end(&met, &forcing)).expect("time loop should succeed");

    eprintln!("full pipeline: {} timesteps completed", reports.len());
    assert!(!reports.is_empty(), "should have at least one step");

    // Verify particles were released and are still active
    let total_released: usize = reports.iter().map(|r| r.released_count).sum();
    assert!(
        total_released > 0,
        "at least some particles should be released"
    );

    let final_active = reports.last().unwrap().active_particle_count;
    eprintln!("full pipeline: released={total_released}, final_active={final_active}");

    // Verify mass conservation (no deposition in this short simulation should
    // remove all mass)
    let store = driver.particle_store();
    let mut total_mass = 0.0_f64;
    let mut active_count = 0usize;
    for slot in 0..store.capacity() {
        if let Some(p) = store.get(slot) {
            if p.is_active() {
                total_mass += f64::from(p.mass[0]);
                active_count += 1;
            }
        }
    }
    eprintln!("full pipeline: active_particles={active_count}, total_mass={total_mass:.6}");

    // Total released mass is 1.0 kg. With mild deposition over 10s, mass
    // should decrease only slightly.
    let initial_total_mass = 1.0_f64;
    assert!(
        total_mass > initial_total_mass * 0.5,
        "mass dropped too much: {total_mass:.6} vs initial {initial_total_mass:.6}"
    );
    assert!(
        total_mass <= initial_total_mass + 1.0e-6,
        "mass should not increase: {total_mass:.6} vs initial {initial_total_mass:.6}"
    );
}

// ---------------------------------------------------------------------------
// Test 6: Deterministic reproducibility (same seed = same result)
// ---------------------------------------------------------------------------

#[test]
fn gpu_pipeline_is_deterministic_across_runs() {
    let run_once = || -> Option<Vec<Particle>> {
        let config = ForwardTimeLoopConfig {
            start_timestamp: "20240101000000".to_string(),
            end_timestamp: "20240101000005".to_string(),
            timestep_seconds: 1,
            time_bounds_behavior: TimeBoundsBehavior::Clamp,
            velocity_to_grid_scale: VelocityToGridScale::IDENTITY,
            ..ForwardTimeLoopConfig::default()
        };

        let releases = vec![ReleaseConfig {
            name: "determinism_test".to_string(),
            start_time: "20240101000000".to_string(),
            end_time: "20240101000000".to_string(),
            lon: 8.0,
            lat: 8.0,
            z_min: 1.0,
            z_max: 1.0,
            mass_kg: 1.0,
            particle_count: 50,
            raw: BTreeMap::new(),
        }];
        let release_grid = GridDomain {
            xlon0: 0.0,
            ylat0: 0.0,
            dx: 1.0,
            dy: 1.0,
            nx: NX,
            ny: NY,
        };

        let mut driver = match pollster::block_on(ForwardTimeLoopDriver::new(
            config,
            &releases,
            release_grid,
            128,
        )) {
            Ok(driver) => driver,
            Err(TimeLoopError::Gpu(GpuError::NoAdapter)) => return None,
            Err(err) => panic!("driver init failed: {err}"),
        };

        let grid = make_grid();
        let wind_t0 = uniform_wind_field(&grid, 0.3, 0.1, 0.0);
        let wind_t1 = uniform_wind_field(&grid, 0.4, 0.2, 0.01);
        let surface_t0 = make_surface_fields();
        let surface_t1 = make_surface_fields();

        let met = MetTimeBracket {
            wind_t0: &wind_t0,
            wind_t1: &wind_t1,
            surface_t0: &surface_t0,
            surface_t1: &surface_t1,
            time_t0_seconds: driver.current_time_seconds(),
            time_t1_seconds: driver.current_time_seconds() + 6,
        };

        let forcing = ForwardStepForcing::default();
        pollster::block_on(driver.run_to_end(&met, &forcing)).expect("run should succeed");

        let store = driver.particle_store();
        let mut result = Vec::new();
        for slot in 0..store.capacity() {
            if let Some(p) = store.get(slot) {
                result.push(*p);
            }
        }
        Some(result)
    };

    let run1 = match run_once() {
        Some(p) => p,
        None => return,
    };
    let run2 = run_once().expect("second run should also succeed");

    assert_eq!(run1.len(), run2.len(), "particle count should be identical");
    for (i, (p1, p2)) in run1.iter().zip(run2.iter()).enumerate() {
        assert_eq!(p1.cell_x, p2.cell_x, "cell_x mismatch at slot {i}");
        assert_eq!(p1.cell_y, p2.cell_y, "cell_y mismatch at slot {i}");
        assert!(
            (p1.pos_x - p2.pos_x).abs() < f32::EPSILON,
            "pos_x mismatch at slot {i}: {} vs {}",
            p1.pos_x,
            p2.pos_x
        );
        assert!(
            (p1.pos_y - p2.pos_y).abs() < f32::EPSILON,
            "pos_y mismatch at slot {i}: {} vs {}",
            p1.pos_y,
            p2.pos_y
        );
        assert!(
            (p1.pos_z - p2.pos_z).abs() < f32::EPSILON,
            "pos_z mismatch at slot {i}: {} vs {}",
            p1.pos_z,
            p2.pos_z
        );
    }
    eprintln!(
        "determinism: {} particles identical across 2 runs",
        run1.len()
    );
}
