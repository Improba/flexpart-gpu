use flexpart_gpu::gpu::{
    advect_particles_gpu, update_particles_turbulence_langevin_gpu, GpuContext, GpuError,
    ParticleBuffers, WindBuffers,
};
use flexpart_gpu::particles::{Particle, ParticleInit, MAX_SPECIES};
use flexpart_gpu::physics::{
    advect_particles_cpu, compute_hanna_params, philox_counter_add,
    update_particles_turbulence_langevin_with_rng_cpu, HannaInputs, LangevinStep, PhiloxCounter,
    PhiloxKey, PhiloxRng, VelocityToGridScale,
};
use flexpart_gpu::wind::WindField3D;

const GRID_NX: usize = 40;
const GRID_NY: usize = 32;
const GRID_NZ: usize = 20;
const PARTICLE_COUNT: usize = 96;
const WORKFLOW_STEPS: usize = 10;
const ADVECT_DT_SECONDS: f32 = 2.5;
const TURBULENCE_DT_SECONDS: f32 = 30.0;
const RHO_GRAD_OVER_RHO: f32 = 2.0e-4;
const PHILOX_KEY: PhiloxKey = [0xA11C_E123, 0xBADC_0FFE];
const PHILOX_BASE_COUNTER: PhiloxCounter = [0x0000_1000, 0x0000_2000, 0x0000_3000, 0x0000_4000];

// Mass should remain invariant because this workflow intentionally excludes dry/wet deposition.
// We still use a small tolerance to absorb f32 summation-order differences between CPU/GPU paths.
const MASS_EPSILON_ABS: f64 = 1.0e-6;
const MASS_EPSILON_REL: f64 = 2.0e-6;

fn deterministic_wind_field() -> WindField3D {
    let mut field = WindField3D::zeros(GRID_NX, GRID_NY, GRID_NZ);
    for i in 0..GRID_NX {
        for j in 0..GRID_NY {
            for k in 0..GRID_NZ {
                let x = i as f32;
                let y = j as f32;
                let z = k as f32;
                field.u_ms[[i, j, k]] = 0.12 + 0.015 * x - 0.006 * y + 0.010 * z;
                field.v_ms[[i, j, k]] = -0.07 + 0.004 * x + 0.012 * y - 0.005 * z;
                field.w_ms[[i, j, k]] = 0.015 + 0.003 * x - 0.002 * y + 0.006 * z;
            }
        }
    }
    field
}

fn make_particle(index: usize) -> Particle {
    let x = 8.0 + (index % 12) as f32 * 1.35;
    let y = 6.0 + ((index / 12) % 8) as f32 * 1.10;
    let z = 2.5 + (index % 9) as f32 * 0.70;
    let cell_x = x.floor() as i32;
    let cell_y = y.floor() as i32;

    let mut mass = [0.0_f32; MAX_SPECIES];
    mass[0] = 0.8 + (index % 5) as f32 * 0.15;
    mass[1] = 0.2 + (index % 7) as f32 * 0.05;
    mass[2] = 0.05 + (index % 3) as f32 * 0.02;

    Particle::new(&ParticleInit {
        cell_x,
        cell_y,
        pos_x: x - cell_x as f32,
        pos_y: y - cell_y as f32,
        pos_z: z,
        mass,
        release_point: (index % 4) as i32,
        class: (index % 3) as i32,
        time: 0,
    })
}

fn deterministic_particles() -> Vec<Particle> {
    (0..PARTICLE_COUNT).map(make_particle).collect()
}

fn hanna_params_for_particles(particles: &[Particle]) -> Vec<flexpart_gpu::pbl::HannaParams> {
    particles
        .iter()
        .enumerate()
        .map(|(idx, particle)| {
            let unstable = idx % 3 == 0;
            let ol = if unstable {
                -60.0 - (idx % 5) as f32 * 10.0
            } else {
                120.0 + (idx % 7) as f32 * 15.0
            };
            let wst = if unstable {
                1.1 + (idx % 4) as f32 * 0.15
            } else {
                0.0
            };
            let h = 700.0 + (idx % 4) as f32 * 120.0;
            let z = particle.pos_z.clamp(5.0, h - 5.0);
            compute_hanna_params(HannaInputs {
                ust: 0.22 + (idx % 6) as f32 * 0.02,
                wst,
                ol,
                h,
                z,
            })
        })
        .collect()
}

fn total_mass(particles: &[Particle]) -> f64 {
    particles
        .iter()
        .filter(|p| p.is_active())
        .map(|particle| particle.mass.iter().map(|m| f64::from(*m)).sum::<f64>())
        .sum::<f64>()
}

fn assert_mass_conserved(initial_mass: f64, final_mass: f64, label: &str) {
    let tolerance = MASS_EPSILON_ABS.max(MASS_EPSILON_REL * initial_mass.abs());
    let delta = (final_mass - initial_mass).abs();
    assert!(
        delta <= tolerance,
        "{label} mass drift exceeds tolerance: initial={initial_mass:.9e}, final={final_mass:.9e}, delta={delta:.9e}, tolerance={tolerance:.9e}"
    );
}

fn run_cpu_workflow(initial_particles: &[Particle], field: &WindField3D) -> Vec<Particle> {
    let mut particles = initial_particles.to_vec();
    let mut rng = PhiloxRng::new(PHILOX_KEY, PHILOX_BASE_COUNTER);
    let step = LangevinStep::legacy(TURBULENCE_DT_SECONDS, RHO_GRAD_OVER_RHO);

    for _ in 0..WORKFLOW_STEPS {
        let hanna = hanna_params_for_particles(&particles);
        update_particles_turbulence_langevin_with_rng_cpu(&mut particles, &hanna, step, &mut rng)
            .expect("cpu Langevin update should succeed");
        advect_particles_cpu(
            &mut particles,
            field,
            ADVECT_DT_SECONDS,
            VelocityToGridScale::IDENTITY,
        );
    }

    particles
}

fn run_gpu_workflow(
    ctx: &GpuContext,
    initial_particles: &[Particle],
    field: &WindField3D,
) -> Vec<Particle> {
    let wind_buffers = WindBuffers::from_field(ctx, field).expect("wind upload should succeed");
    let particle_buffers = ParticleBuffers::from_particles(ctx, initial_particles);
    let step = LangevinStep::legacy(TURBULENCE_DT_SECONDS, RHO_GRAD_OVER_RHO);

    let mut counter = PHILOX_BASE_COUNTER;
    for _ in 0..WORKFLOW_STEPS {
        let host_particles = pollster::block_on(particle_buffers.download_particles(ctx))
            .expect("intermediate particle readback should succeed");
        let hanna = hanna_params_for_particles(&host_particles);
        counter = update_particles_turbulence_langevin_gpu(
            ctx,
            &particle_buffers,
            &hanna,
            step,
            PHILOX_KEY,
            counter,
        )
        .expect("gpu Langevin update should succeed");
        advect_particles_gpu(
            ctx,
            &particle_buffers,
            &wind_buffers,
            ADVECT_DT_SECONDS,
            VelocityToGridScale::IDENTITY,
        )
        .expect("gpu advection should succeed");
    }

    let expected_counter = philox_counter_add(
        PHILOX_BASE_COUNTER,
        (WORKFLOW_STEPS.saturating_mul(initial_particles.len())) as u64,
    );
    assert_eq!(
        counter, expected_counter,
        "Philox counter progression should remain deterministic"
    );

    pollster::block_on(particle_buffers.download_particles(ctx))
        .expect("final particle readback should succeed")
}

#[test]
fn mass_is_conserved_cpu_advection_plus_turbulence_without_deposition() {
    let particles = deterministic_particles();
    let field = deterministic_wind_field();
    let initial_mass = total_mass(&particles);

    let final_particles = run_cpu_workflow(&particles, &field);
    let final_mass = total_mass(&final_particles);
    assert_mass_conserved(initial_mass, final_mass, "cpu");
    assert_eq!(
        final_particles.iter().filter(|p| p.is_active()).count(),
        particles.len(),
        "no particles should be removed when deposition is disabled"
    );
}

#[test]
fn mass_is_conserved_gpu_advection_plus_turbulence_without_deposition() {
    let ctx = match pollster::block_on(GpuContext::new()) {
        Ok(ctx) => ctx,
        Err(GpuError::NoAdapter) => {
            eprintln!("No GPU adapter found — skipping mass conservation GPU integration test");
            return;
        }
        Err(err) => panic!("unexpected GPU init error: {err}"),
    };

    let particles = deterministic_particles();
    let field = deterministic_wind_field();
    let initial_mass = total_mass(&particles);

    let final_particles = run_gpu_workflow(&ctx, &particles, &field);
    let final_mass = total_mass(&final_particles);
    assert_mass_conserved(initial_mass, final_mass, "gpu");
    assert_eq!(
        final_particles.iter().filter(|p| p.is_active()).count(),
        particles.len(),
        "no particles should be removed when deposition is disabled"
    );
}
