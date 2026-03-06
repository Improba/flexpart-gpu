use flexpart_gpu::constants::HREF;
use flexpart_gpu::gpu::{
    apply_dry_deposition_step_gpu, apply_wet_deposition_step_gpu, GpuContext, GpuError,
    ParticleBuffers, WetDepositionStepParams,
};
use flexpart_gpu::particles::{Particle, ParticleInit, MAX_SPECIES};
use flexpart_gpu::physics::{
    apply_wet_scavenging_mass_step, dry_deposition_probability_step, WetScavengingStep,
};

const STEP_COUNT: usize = 12;
const DT_SECONDS: f32 = 60.0;
const DRY_VDEP_M_S: f32 = 0.024;
const WET_SCAVENGING_LAMBDA_S_INV: f32 = 1.6e-3;
const WET_PRECIPITATING_FRACTION: f32 = 1.0;
const PARTICLE_HEIGHT_M: f32 = 5.0;
const INITIAL_SPECIES_MASS: [f32; MAX_SPECIES] = [1.0, 0.4, 0.2, 0.1];

// Error budget documentation for D-06:
// - CPU path: deposition helpers run in f32 and are compared against f64 analytical decay;
//   expected accumulated error <= 2e-6 (absolute and relative) over 12 steps.
// - GPU path: `exp` and rounding can differ slightly by adapter/driver; expected error
//   <= 2e-5 (absolute and relative) while preserving deterministic monotonic decay.
const CPU_ABS_TOLERANCE: f64 = 2.0e-6;
const CPU_REL_TOLERANCE: f64 = 2.0e-6;
const GPU_ABS_TOLERANCE: f64 = 2.0e-5;
const GPU_REL_TOLERANCE: f64 = 2.0e-5;

fn make_particle_with_mass(species_mass: [f32; MAX_SPECIES], pos_z: f32) -> Particle {
    Particle::new(&ParticleInit {
        cell_x: 2,
        cell_y: 3,
        pos_x: 0.25,
        pos_y: 0.75,
        pos_z,
        mass: species_mass,
        release_point: 0,
        class: 0,
        time: 0,
    })
}

fn total_mass(particle: &Particle) -> f64 {
    particle.mass.iter().map(|m| f64::from(*m)).sum::<f64>()
}

fn analytical_mass_dry(initial_mass: f64, step: usize) -> f64 {
    let dry_rate_s_inv = f64::from(DRY_VDEP_M_S / (2.0 * HREF));
    let elapsed_time_s = step as f64 * f64::from(DT_SECONDS);
    initial_mass * (-dry_rate_s_inv * elapsed_time_s).exp()
}

fn analytical_mass_wet(initial_mass: f64, step: usize) -> f64 {
    let wet_rate_s_inv = f64::from(WET_SCAVENGING_LAMBDA_S_INV);
    let elapsed_time_s = step as f64 * f64::from(DT_SECONDS);
    initial_mass * (-wet_rate_s_inv * elapsed_time_s).exp()
}

fn analytical_mass_combined(initial_mass: f64, step: usize) -> f64 {
    let dry_rate_s_inv = f64::from(DRY_VDEP_M_S / (2.0 * HREF));
    let wet_rate_s_inv = f64::from(WET_SCAVENGING_LAMBDA_S_INV);
    let elapsed_time_s = step as f64 * f64::from(DT_SECONDS);
    initial_mass * (-(dry_rate_s_inv + wet_rate_s_inv) * elapsed_time_s).exp()
}

fn assert_with_tolerance(label: &str, simulated: f64, analytical: f64, abs_tol: f64, rel_tol: f64) {
    let tolerance = abs_tol.max(rel_tol * analytical.abs());
    let error = (simulated - analytical).abs();
    assert!(
        error <= tolerance,
        "{label} mismatch: simulated={simulated:.9e}, analytical={analytical:.9e}, error={error:.9e}, tolerance={tolerance:.9e}"
    );
}

fn assert_monotonic_decay(masses: &[f64], label: &str) {
    for pair in masses.windows(2) {
        assert!(
            pair[1] <= pair[0] + 1.0e-12,
            "{label} mass must be non-increasing: previous={:.9e}, current={:.9e}",
            pair[0],
            pair[1]
        );
    }
}

fn run_dry_cpu_evolution(initial_mass: f64) -> Vec<f64> {
    let mut mass = initial_mass;
    let mut evolution = Vec::with_capacity(STEP_COUNT);
    for _ in 0..STEP_COUNT {
        let probability = dry_deposition_probability_step(DRY_VDEP_M_S, DT_SECONDS, HREF);
        mass *= 1.0 - f64::from(probability);
        evolution.push(mass);
    }
    evolution
}

fn run_wet_cpu_evolution(initial_mass: f64) -> Vec<f64> {
    let mut mass = initial_mass as f32;
    let mut evolution = Vec::with_capacity(STEP_COUNT);
    let step = WetScavengingStep {
        scavenging_coefficient_s_inv: WET_SCAVENGING_LAMBDA_S_INV,
        dt_seconds: DT_SECONDS,
        precipitating_fraction: WET_PRECIPITATING_FRACTION,
    };
    for _ in 0..STEP_COUNT {
        let (remaining, _deposited) = apply_wet_scavenging_mass_step(mass, step);
        mass = remaining;
        evolution.push(f64::from(mass));
    }
    evolution
}

fn run_combined_cpu_evolution(initial_mass: f64) -> Vec<f64> {
    let wet_step = WetScavengingStep {
        scavenging_coefficient_s_inv: WET_SCAVENGING_LAMBDA_S_INV,
        dt_seconds: DT_SECONDS,
        precipitating_fraction: WET_PRECIPITATING_FRACTION,
    };

    let mut mass = initial_mass;
    let mut evolution = Vec::with_capacity(STEP_COUNT);
    for _ in 0..STEP_COUNT {
        let dry_prob = f64::from(dry_deposition_probability_step(
            DRY_VDEP_M_S,
            DT_SECONDS,
            HREF,
        ));
        mass *= 1.0 - dry_prob;
        let (remaining, _deposited) = apply_wet_scavenging_mass_step(mass as f32, wet_step);
        mass = f64::from(remaining);
        evolution.push(mass);
    }
    evolution
}

fn run_dry_gpu_evolution(ctx: &GpuContext, initial_mass: [f32; MAX_SPECIES]) -> Vec<f64> {
    let particles = vec![make_particle_with_mass(initial_mass, PARTICLE_HEIGHT_M)];
    let particle_buffers = ParticleBuffers::from_particles(ctx, &particles);
    let dry_velocity = vec![DRY_VDEP_M_S];
    let mut evolution = Vec::with_capacity(STEP_COUNT);

    for _ in 0..STEP_COUNT {
        let _probability = pollster::block_on(apply_dry_deposition_step_gpu(
            ctx,
            &particle_buffers,
            &dry_velocity,
            flexpart_gpu::gpu::DryDepositionStepParams {
                dt_seconds: DT_SECONDS,
                reference_height_m: HREF,
            },
        ))
        .expect("dry deposition GPU step should succeed");
        let updated = pollster::block_on(particle_buffers.download_particles(ctx))
            .expect("dry deposition particle readback should succeed");
        evolution.push(total_mass(&updated[0]));
    }
    evolution
}

fn run_wet_gpu_evolution(ctx: &GpuContext, initial_mass: [f32; MAX_SPECIES]) -> Vec<f64> {
    let particles = vec![make_particle_with_mass(initial_mass, PARTICLE_HEIGHT_M)];
    let particle_buffers = ParticleBuffers::from_particles(ctx, &particles);
    let scavenging = vec![WET_SCAVENGING_LAMBDA_S_INV];
    let precipitating_fraction = vec![WET_PRECIPITATING_FRACTION];
    let mut evolution = Vec::with_capacity(STEP_COUNT);

    for _ in 0..STEP_COUNT {
        let _probability = pollster::block_on(apply_wet_deposition_step_gpu(
            ctx,
            &particle_buffers,
            &scavenging,
            &precipitating_fraction,
            WetDepositionStepParams {
                dt_seconds: DT_SECONDS,
            },
        ))
        .expect("wet deposition GPU step should succeed");
        let updated = pollster::block_on(particle_buffers.download_particles(ctx))
            .expect("wet deposition particle readback should succeed");
        evolution.push(total_mass(&updated[0]));
    }
    evolution
}

fn run_combined_gpu_evolution(ctx: &GpuContext, initial_mass: [f32; MAX_SPECIES]) -> Vec<f64> {
    let particles = vec![make_particle_with_mass(initial_mass, PARTICLE_HEIGHT_M)];
    let particle_buffers = ParticleBuffers::from_particles(ctx, &particles);
    let dry_velocity = vec![DRY_VDEP_M_S];
    let scavenging = vec![WET_SCAVENGING_LAMBDA_S_INV];
    let precipitating_fraction = vec![WET_PRECIPITATING_FRACTION];
    let mut evolution = Vec::with_capacity(STEP_COUNT);

    for _ in 0..STEP_COUNT {
        let _dry_probability = pollster::block_on(apply_dry_deposition_step_gpu(
            ctx,
            &particle_buffers,
            &dry_velocity,
            flexpart_gpu::gpu::DryDepositionStepParams {
                dt_seconds: DT_SECONDS,
                reference_height_m: HREF,
            },
        ))
        .expect("combined dry deposition GPU step should succeed");
        let _wet_probability = pollster::block_on(apply_wet_deposition_step_gpu(
            ctx,
            &particle_buffers,
            &scavenging,
            &precipitating_fraction,
            WetDepositionStepParams {
                dt_seconds: DT_SECONDS,
            },
        ))
        .expect("combined wet deposition GPU step should succeed");
        let updated = pollster::block_on(particle_buffers.download_particles(ctx))
            .expect("combined deposition particle readback should succeed");
        evolution.push(total_mass(&updated[0]));
    }
    evolution
}

#[test]
fn deposition_mass_evolution_cpu_matches_analytical_exponential_decay() {
    let initial_particle = make_particle_with_mass(INITIAL_SPECIES_MASS, PARTICLE_HEIGHT_M);
    let initial_mass = total_mass(&initial_particle);

    let dry = run_dry_cpu_evolution(initial_mass);
    let wet = run_wet_cpu_evolution(initial_mass);
    let combined = run_combined_cpu_evolution(initial_mass);

    assert_monotonic_decay(&dry, "cpu dry");
    assert_monotonic_decay(&wet, "cpu wet");
    assert_monotonic_decay(&combined, "cpu combined");

    for step in 1..=STEP_COUNT {
        let dry_expected = analytical_mass_dry(initial_mass, step);
        let wet_expected = analytical_mass_wet(initial_mass, step);
        let combined_expected = analytical_mass_combined(initial_mass, step);
        let decomposed_expected = dry_expected * (wet_expected / initial_mass);

        assert_with_tolerance(
            "cpu dry decay",
            dry[step - 1],
            dry_expected,
            CPU_ABS_TOLERANCE,
            CPU_REL_TOLERANCE,
        );
        assert_with_tolerance(
            "cpu wet decay",
            wet[step - 1],
            wet_expected,
            CPU_ABS_TOLERANCE,
            CPU_REL_TOLERANCE,
        );
        assert_with_tolerance(
            "cpu combined decay",
            combined[step - 1],
            combined_expected,
            CPU_ABS_TOLERANCE,
            CPU_REL_TOLERANCE,
        );
        assert_with_tolerance(
            "cpu combined decomposition",
            combined[step - 1],
            decomposed_expected,
            CPU_ABS_TOLERANCE,
            CPU_REL_TOLERANCE,
        );
    }
}

#[test]
fn deposition_mass_evolution_gpu_matches_analytical_exponential_decay_when_adapter_available() {
    let ctx = match pollster::block_on(GpuContext::new()) {
        Ok(ctx) => ctx,
        Err(GpuError::NoAdapter) => {
            eprintln!("No GPU adapter found — skipping deposition exponential-decay GPU test");
            return;
        }
        Err(err) => panic!("unexpected GPU init error: {err}"),
    };

    let initial_particle = make_particle_with_mass(INITIAL_SPECIES_MASS, PARTICLE_HEIGHT_M);
    let initial_mass = total_mass(&initial_particle);

    let dry = run_dry_gpu_evolution(&ctx, INITIAL_SPECIES_MASS);
    let wet = run_wet_gpu_evolution(&ctx, INITIAL_SPECIES_MASS);
    let combined = run_combined_gpu_evolution(&ctx, INITIAL_SPECIES_MASS);

    assert_monotonic_decay(&dry, "gpu dry");
    assert_monotonic_decay(&wet, "gpu wet");
    assert_monotonic_decay(&combined, "gpu combined");

    for step in 1..=STEP_COUNT {
        let dry_expected = analytical_mass_dry(initial_mass, step);
        let wet_expected = analytical_mass_wet(initial_mass, step);
        let combined_expected = analytical_mass_combined(initial_mass, step);
        let decomposed_expected = dry_expected * (wet_expected / initial_mass);

        assert_with_tolerance(
            "gpu dry decay",
            dry[step - 1],
            dry_expected,
            GPU_ABS_TOLERANCE,
            GPU_REL_TOLERANCE,
        );
        assert_with_tolerance(
            "gpu wet decay",
            wet[step - 1],
            wet_expected,
            GPU_ABS_TOLERANCE,
            GPU_REL_TOLERANCE,
        );
        assert_with_tolerance(
            "gpu combined decay",
            combined[step - 1],
            combined_expected,
            GPU_ABS_TOLERANCE,
            GPU_REL_TOLERANCE,
        );
        assert_with_tolerance(
            "gpu combined decomposition",
            combined[step - 1],
            decomposed_expected,
            GPU_ABS_TOLERANCE,
            GPU_REL_TOLERANCE,
        );
    }
}
