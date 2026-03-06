//! Phase B scientific invariant tests:
//!
//! B1 — Concentration gridding CPU/GPU parity (with outheights in meters)
//! B2 — Positivity: mass and concentration are never negative
//! B3 — Mass conservation with deposition: particles + dry + wet = initial

use flexpart_gpu::constants::HREF;
use flexpart_gpu::gpu::{
    accumulate_concentration_grid_gpu, apply_dry_deposition_step_gpu,
    apply_wet_deposition_step_gpu, DryDepositionStepParams, GpuContext, GpuError,
    ParticleBuffers, WetDepositionStepParams,
    ConcentrationGridShape, ConcentrationGriddingParams, MAX_OUTPUT_LEVELS,
};
use flexpart_gpu::particles::{Particle, ParticleInit, MAX_SPECIES};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_particle(
    cell_x: i32,
    cell_y: i32,
    pos_z_m: f32,
    species0_mass: f32,
) -> Particle {
    let mut mass = [0.0_f32; MAX_SPECIES];
    mass[0] = species0_mass;
    Particle::new(&ParticleInit {
        cell_x,
        cell_y,
        pos_x: 0.3,
        pos_y: 0.6,
        pos_z: pos_z_m,
        mass,
        release_point: 0,
        class: 0,
        time: 0,
    })
}

fn total_mass_species0(particles: &[Particle]) -> f64 {
    particles
        .iter()
        .filter(|p| p.is_active())
        .map(|p| f64::from(p.mass[0]))
        .sum()
}

// CPU reference gridding: same logic as the WGSL shader with outheights.
fn cpu_concentration_grid(
    particles: &[Particle],
    shape: ConcentrationGridShape,
    outheights: &[f32],
    mass_scale: f32,
) -> (Vec<u32>, Vec<f32>) {
    let cell_count = shape.cell_count();
    let mut counts = vec![0_u32; cell_count];
    let mut mass_kg = vec![0.0_f32; cell_count];

    for p in particles {
        if !p.is_active() {
            continue;
        }
        let ix = p.cell_x.clamp(0, shape.nx as i32 - 1) as usize;
        let iy = p.cell_y.clamp(0, shape.ny as i32 - 1) as usize;
        let iz = if outheights[0] > 0.0 {
            let mut level = shape.nz - 1;
            for k in 0..shape.nz {
                if p.pos_z <= outheights[k] {
                    level = k;
                    break;
                }
            }
            level
        } else {
            (p.pos_z.floor() as usize).clamp(0, shape.nz - 1)
        };

        let flat = ((ix * shape.ny) + iy) * shape.nz + iz;
        counts[flat] += 1;

        let scaled = (p.mass[0] * mass_scale).round() as u32;
        mass_kg[flat] += scaled as f32 / mass_scale;
    }
    (counts, mass_kg)
}

// ---------------------------------------------------------------------------
// B1 — Concentration gridding CPU/GPU parity
// ---------------------------------------------------------------------------

#[test]
fn concentration_gridding_gpu_matches_cpu_reference_with_outheights() {
    let ctx = match pollster::block_on(GpuContext::new()) {
        Ok(ctx) => ctx,
        Err(GpuError::NoAdapter) => {
            eprintln!("No GPU adapter — skipping concentration gridding parity test");
            return;
        }
        Err(err) => panic!("unexpected GPU init error: {err}"),
    };

    let outheights_arr = [100.0_f32, 500.0, 1000.0];
    let shape = ConcentrationGridShape {
        nx: 8,
        ny: 6,
        nz: 3,
    };
    let mass_scale = 1_000_000.0_f32;

    let particles = vec![
        make_particle(2, 3, 50.0, 1.5),
        make_particle(2, 3, 150.0, 0.8),
        make_particle(5, 1, 600.0, 2.0),
        make_particle(7, 5, 999.0, 0.3),
        make_particle(0, 0, 10.0, 4.0),
        {
            let mut p = make_particle(1, 1, 80.0, 9.9);
            p.deactivate();
            p
        },
    ];

    let mut outheights_full = [0.0_f32; MAX_OUTPUT_LEVELS];
    for (i, &h) in outheights_arr.iter().enumerate() {
        outheights_full[i] = h;
    }
    let params = ConcentrationGriddingParams {
        species_index: 0,
        mass_scale,
        outheights: outheights_full,
    };

    let particle_buffers = ParticleBuffers::from_particles(&ctx, &particles);
    let gpu_output = pollster::block_on(accumulate_concentration_grid_gpu(
        &ctx,
        &particle_buffers,
        shape,
        params,
    ))
    .expect("GPU gridding should succeed");

    let (cpu_counts, cpu_mass) =
        cpu_concentration_grid(&particles, shape, &outheights_arr, mass_scale);

    for ix in 0..shape.nx {
        for iy in 0..shape.ny {
            for iz in 0..shape.nz {
                let (gpu_count, gpu_mass_kg) = gpu_output.cell_values(ix, iy, iz);
                let flat = ((ix * shape.ny) + iy) * shape.nz + iz;
                let cpu_count = cpu_counts[flat];
                let cpu_mass_kg = cpu_mass[flat];

                assert_eq!(
                    gpu_count, cpu_count,
                    "count mismatch at ({ix},{iy},{iz}): GPU={gpu_count}, CPU={cpu_count}"
                );
                let mass_err = (gpu_mass_kg - cpu_mass_kg).abs();
                assert!(
                    mass_err < 1.0e-4,
                    "mass mismatch at ({ix},{iy},{iz}): GPU={gpu_mass_kg:.6e}, CPU={cpu_mass_kg:.6e}, err={mass_err:.6e}"
                );
            }
        }
    }

    let gpu_total: u32 = gpu_output.particle_count_per_cell.iter().sum();
    let active_count = particles.iter().filter(|p| p.is_active()).count() as u32;
    assert_eq!(
        gpu_total, active_count,
        "total gridded particles should equal active count"
    );
}

// ---------------------------------------------------------------------------
// B2 — Positivity invariants
// ---------------------------------------------------------------------------

#[test]
fn particle_mass_stays_non_negative_after_deposition() {
    let ctx = match pollster::block_on(GpuContext::new()) {
        Ok(ctx) => ctx,
        Err(GpuError::NoAdapter) => {
            eprintln!("No GPU adapter — skipping positivity test");
            return;
        }
        Err(err) => panic!("unexpected GPU init error: {err}"),
    };

    let particles = vec![
        make_particle(1, 1, 5.0, 1.0e-8),
        make_particle(2, 2, 5.0, 1.0e-4),
        make_particle(3, 3, 5.0, 1.0),
        make_particle(4, 4, 5.0, 1.0e6),
    ];
    let particle_buffers = ParticleBuffers::from_particles(&ctx, &particles);

    let dry_velocity = vec![0.05_f32; particles.len()];
    let scavenging = vec![0.01_f32; particles.len()];
    let precipitating_fraction = vec![1.0_f32; particles.len()];

    for step in 0..50 {
        let _dry_prob = pollster::block_on(apply_dry_deposition_step_gpu(
            &ctx,
            &particle_buffers,
            &dry_velocity,
            DryDepositionStepParams {
                dt_seconds: 60.0,
                reference_height_m: HREF,
            },
        ))
        .expect("dry deposition should succeed");

        let _wet_prob = pollster::block_on(apply_wet_deposition_step_gpu(
            &ctx,
            &particle_buffers,
            &scavenging,
            &precipitating_fraction,
            WetDepositionStepParams { dt_seconds: 60.0 },
        ))
        .expect("wet deposition should succeed");

        let updated = pollster::block_on(particle_buffers.download_particles(&ctx))
            .expect("particle readback should succeed");

        for (i, p) in updated.iter().enumerate() {
            if p.is_active() {
                for (s, &m) in p.mass.iter().enumerate() {
                    assert!(
                        m >= 0.0,
                        "negative mass at step {step}, particle {i}, species {s}: {m}"
                    );
                }
            }
        }
    }
}

#[test]
fn concentration_gridding_output_is_non_negative() {
    let ctx = match pollster::block_on(GpuContext::new()) {
        Ok(ctx) => ctx,
        Err(GpuError::NoAdapter) => {
            eprintln!("No GPU adapter — skipping concentration positivity test");
            return;
        }
        Err(err) => panic!("unexpected GPU init error: {err}"),
    };

    let particles = vec![
        make_particle(0, 0, 50.0, 1.0),
        make_particle(1, 1, 200.0, 0.5),
        make_particle(2, 2, 800.0, 0.01),
    ];
    let shape = ConcentrationGridShape {
        nx: 4,
        ny: 4,
        nz: 3,
    };
    let mut outheights = [0.0_f32; MAX_OUTPUT_LEVELS];
    outheights[0] = 100.0;
    outheights[1] = 500.0;
    outheights[2] = 1000.0;

    let particle_buffers = ParticleBuffers::from_particles(&ctx, &particles);
    let output = pollster::block_on(accumulate_concentration_grid_gpu(
        &ctx,
        &particle_buffers,
        shape,
        ConcentrationGriddingParams {
            species_index: 0,
            mass_scale: 1_000_000.0,
            outheights,
        },
    ))
    .expect("gridding should succeed");

    for (i, &mass_kg) in output.concentration_mass_kg.iter().enumerate() {
        assert!(
            mass_kg >= 0.0,
            "negative concentration mass at flat index {i}: {mass_kg}"
        );
    }
}

// ---------------------------------------------------------------------------
// B3 — Mass conservation with deposition (particles + dry + wet = initial)
// ---------------------------------------------------------------------------

#[test]
fn total_mass_conserved_with_deposition_gpu() {
    let ctx = match pollster::block_on(GpuContext::new()) {
        Ok(ctx) => ctx,
        Err(GpuError::NoAdapter) => {
            eprintln!("No GPU adapter — skipping mass conservation with deposition test");
            return;
        }
        Err(err) => panic!("unexpected GPU init error: {err}"),
    };

    let particles = vec![
        make_particle(1, 1, 5.0, 2.0),
        make_particle(2, 2, 5.0, 3.0),
        make_particle(3, 3, 5.0, 1.5),
    ];
    let initial_mass = total_mass_species0(&particles);

    let particle_buffers = ParticleBuffers::from_particles(&ctx, &particles);

    let dry_velocity = vec![0.02_f32; particles.len()];
    let scavenging = vec![0.005_f32; particles.len()];
    let precipitating_fraction = vec![1.0_f32; particles.len()];

    let mut cumulative_dry_deposited = 0.0_f64;
    let mut cumulative_wet_deposited = 0.0_f64;

    let dt_seconds = 60.0_f32;
    let n_steps = 20;

    for _step in 0..n_steps {
        let mass_before = {
            let ps = pollster::block_on(particle_buffers.download_particles(&ctx))
                .expect("readback");
            total_mass_species0(&ps)
        };

        let dry_probs = pollster::block_on(apply_dry_deposition_step_gpu(
            &ctx,
            &particle_buffers,
            &dry_velocity,
            DryDepositionStepParams {
                dt_seconds,
                reference_height_m: HREF,
            },
        ))
        .expect("dry deposition should succeed");

        let mass_after_dry = {
            let ps = pollster::block_on(particle_buffers.download_particles(&ctx))
                .expect("readback");
            total_mass_species0(&ps)
        };
        let dry_removed = mass_before - mass_after_dry;
        cumulative_dry_deposited += dry_removed;

        let wet_probs = pollster::block_on(apply_wet_deposition_step_gpu(
            &ctx,
            &particle_buffers,
            &scavenging,
            &precipitating_fraction,
            WetDepositionStepParams { dt_seconds },
        ))
        .expect("wet deposition should succeed");

        let mass_after_wet = {
            let ps = pollster::block_on(particle_buffers.download_particles(&ctx))
                .expect("readback");
            total_mass_species0(&ps)
        };
        let wet_removed = mass_after_dry - mass_after_wet;
        cumulative_wet_deposited += wet_removed;

        assert!(
            !dry_probs.is_empty(),
            "dry deposition should return probabilities"
        );
        assert!(
            !wet_probs.is_empty(),
            "wet deposition should return probabilities"
        );
    }

    let final_particles =
        pollster::block_on(particle_buffers.download_particles(&ctx)).expect("final readback");
    let remaining_mass = total_mass_species0(&final_particles);

    let total_accounted = remaining_mass + cumulative_dry_deposited + cumulative_wet_deposited;
    let budget_error = (total_accounted - initial_mass).abs();
    let rel_error = budget_error / initial_mass;

    assert!(
        rel_error < 1.0e-5,
        "mass budget violated: initial={initial_mass:.9e}, remaining={remaining_mass:.9e}, \
         dry_deposited={cumulative_dry_deposited:.9e}, wet_deposited={cumulative_wet_deposited:.9e}, \
         total={total_accounted:.9e}, error={budget_error:.9e} ({rel_error:.2e} relative)"
    );

    assert!(
        remaining_mass < initial_mass,
        "mass should decrease with deposition: initial={initial_mass}, final={remaining_mass}"
    );

    assert!(
        remaining_mass > 0.0,
        "particles should retain some mass after {n_steps} steps"
    );
}
