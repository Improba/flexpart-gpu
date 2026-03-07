use std::collections::BTreeMap;
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use flexpart_gpu::config::ReleaseConfig;
use flexpart_gpu::coords::GridDomain;
use flexpart_gpu::gpu::{
    accumulate_concentration_grid_gpu, advect_particles_gpu, apply_dry_deposition_step_gpu,
    apply_wet_deposition_step_gpu, auto_tune_key_kernels, compute_hanna_params_gpu,
    encode_langevin_fused_gpu, save_autotune_report_default,
    update_particles_turbulence_langevin_gpu, ConcentrationGridShape,
    ConcentrationGriddingParams, DryDepositionStepParams, GpuContext, GpuError,
    LangevinFusedDispatchKernel, MAX_OUTPUT_LEVELS, ParticleBuffers,
    PblBuffers, ScopedWorkgroupOverride, WetDepositionStepParams, WindBuffers,
    WorkgroupAutoTuneOptions, WorkgroupKernel,
};
use flexpart_gpu::io::{
    compute_pbl_parameters_from_met, interpolate_surface_fields_linear,
    interpolate_wind_field_linear, PblComputationOptions, PblMetInputGrids, TimeBoundsBehavior,
};
use flexpart_gpu::particles::{Particle, ParticleInit, MAX_SPECIES};
use flexpart_gpu::physics::{
    advect_particles_cpu, compute_hanna_params_from_pbl, dry_deposition_probability_step,
    in_dry_deposition_layer, update_particles_turbulence_langevin_with_rng_cpu,
    wet_scavenging_probability_step, LangevinStep, PhiloxRng, VelocityToGridScale,
    WetScavengingStep,
};
use flexpart_gpu::simulation::{
    ForwardStepForcing, ForwardTimeLoopConfig, ForwardTimeLoopDriver, MetTimeBracket,
    ParticleForcingField, TimeLoopError,
};
use flexpart_gpu::wind::{SurfaceFields, WindField3D};

const BENCH_GRID_NX: usize = 64;
const BENCH_GRID_NY: usize = 64;
const BENCH_GRID_NZ: usize = 24;
const BENCH_WIND_BRACKET_SECONDS: i64 = 600;
const BENCH_INTERPOLATION_TARGET_SECONDS: i64 = BENCH_WIND_BRACKET_SECONDS / 2;
const BENCH_LANGEVIN_KEY: [u32; 2] = [0xDECA_FBAD, 0x1234_5678];
const BENCH_LANGEVIN_COUNTER: [u32; 4] = [0, 0, 0, 0];

const SCENARIOS: [ParticleScenario; 3] = [
    ParticleScenario {
        name: "1M",
        target_particles: 1_000_000,
    },
    ParticleScenario {
        name: "10M",
        target_particles: 10_000_000,
    },
    ParticleScenario {
        name: "100M",
        target_particles: 100_000_000,
    },
];

#[derive(Clone, Copy)]
struct ParticleScenario {
    name: &'static str,
    target_particles: usize,
}

#[derive(Clone, Copy)]
struct BenchRuntimeConfig {
    max_particles: usize,
    sample_size: usize,
    warm_up_seconds: u64,
    measurement_seconds: u64,
}

impl BenchRuntimeConfig {
    fn from_env() -> Self {
        Self {
            max_particles: parse_env_usize("FLEXPART_BENCH_MAX_PARTICLES", 1_000_000).max(1),
            sample_size: parse_env_usize("FLEXPART_BENCH_SAMPLE_SIZE", 10).max(10),
            warm_up_seconds: parse_env_usize("FLEXPART_BENCH_WARMUP_SECS", 1) as u64,
            measurement_seconds: parse_env_usize("FLEXPART_BENCH_MEASUREMENT_SECS", 3) as u64,
        }
    }

    fn effective_particles(self, scenario: ParticleScenario) -> usize {
        scenario.target_particles.min(self.max_particles).max(1)
    }

    fn scenario_label(self, scenario: ParticleScenario) -> String {
        let effective = self.effective_particles(scenario);
        if effective == scenario.target_particles {
            scenario.name.to_string()
        } else {
            format!(
                "{} (scaled:{})",
                scenario.name,
                format_particle_count(effective)
            )
        }
    }
}

fn parse_env_usize(name: &str, default_value: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default_value)
}

fn parse_env_bool(name: &str, default_value: bool) -> bool {
    std::env::var(name).map_or(default_value, |value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn parse_env_u32_list(name: &str) -> Option<Vec<u32>> {
    let raw = std::env::var(name).ok()?;
    let candidates: Vec<u32> = raw
        .split(',')
        .filter_map(|value| value.trim().parse::<u32>().ok())
        .collect();
    if candidates.is_empty() {
        None
    } else {
        Some(candidates)
    }
}

fn format_particle_count(value: usize) -> String {
    if value >= 1_000_000 {
        format!("{:.2}M", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.2}K", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

fn configure_group(
    group: &mut criterion::BenchmarkGroup<'_, WallTime>,
    runtime: BenchRuntimeConfig,
) {
    group.sample_size(runtime.sample_size);
    group.warm_up_time(Duration::from_secs(runtime.warm_up_seconds));
    group.measurement_time(Duration::from_secs(runtime.measurement_seconds));
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn fraction_from_seed(seed: u64) -> f32 {
    // Exactly deterministic in [0, 1).
    let upper = (splitmix64(seed) >> 32) as u32;
    (upper as f32) / (u32::MAX as f32 + 1.0)
}

fn deterministic_particles(count: usize) -> Vec<Particle> {
    let x_span = BENCH_GRID_NX.saturating_sub(1).max(1);
    let y_span = BENCH_GRID_NY.saturating_sub(1).max(1);
    let z_span = BENCH_GRID_NZ.saturating_sub(1).max(1);
    let xy_span = x_span.saturating_mul(y_span).max(1);

    let mut particles = Vec::with_capacity(count);
    for index in 0..count {
        let i = index as u64;
        let cell_x = (index % x_span) as i32;
        let cell_y = ((index / x_span) % y_span) as i32;
        let layer = ((index / xy_span) % z_span) as f32;

        let frac_x = 0.05 + 0.9 * fraction_from_seed(i.wrapping_mul(3).wrapping_add(1));
        let frac_y = 0.05 + 0.9 * fraction_from_seed(i.wrapping_mul(5).wrapping_add(7));
        let frac_z = 0.05 + 0.9 * fraction_from_seed(i.wrapping_mul(11).wrapping_add(13));
        let max_z = BENCH_GRID_NZ.saturating_sub(1) as f32;
        let pos_z = if max_z > 0.0 {
            (layer + frac_z).min(max_z - f32::EPSILON)
        } else {
            0.0
        };

        let mut mass = [0.0_f32; MAX_SPECIES];
        mass[0] = 1.0 + (index % 17) as f32 * 0.01;
        mass[1] = 0.5 + (index % 11) as f32 * 0.005;

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

fn deterministic_wind_field(phase: f32) -> WindField3D {
    let mut wind = WindField3D::zeros(BENCH_GRID_NX, BENCH_GRID_NY, BENCH_GRID_NZ);
    for i in 0..BENCH_GRID_NX {
        for j in 0..BENCH_GRID_NY {
            for k in 0..BENCH_GRID_NZ {
                let x = i as f32;
                let y = j as f32;
                let z = k as f32;
                wind.u_ms[[i, j, k]] = 0.2 + 0.01 * x + 0.001 * y + 0.0005 * z + phase;
                wind.v_ms[[i, j, k]] = -0.1 + 0.008 * y - 0.0007 * x + 0.0003 * z + phase * 0.5;
                wind.w_ms[[i, j, k]] = 0.05 + 0.0009 * z - 0.0002 * y + phase * 0.25;
                wind.temperature_k[[i, j, k]] = 285.0 + 0.02 * z + phase;
                wind.specific_humidity[[i, j, k]] = 0.004 + 0.00001 * x + phase * 0.0001;
                wind.pressure_pa[[i, j, k]] = 100_000.0 - 11.0 * z + phase * 10.0;
                wind.air_density_kg_m3[[i, j, k]] = 1.2 - 0.0008 * z + phase * 0.001;
                wind.density_gradient_kg_m2[[i, j, k]] = -0.0008;
            }
        }
    }
    wind
}

fn deterministic_surface_fields(phase: f32) -> SurfaceFields {
    let mut surface = SurfaceFields::zeros(BENCH_GRID_NX, BENCH_GRID_NY);
    for i in 0..BENCH_GRID_NX {
        for j in 0..BENCH_GRID_NY {
            let x = i as f32;
            let y = j as f32;
            surface.surface_pressure_pa[[i, j]] = 101_000.0 + 0.5 * x - 0.4 * y;
            surface.u10_ms[[i, j]] = 2.0 + 0.02 * x + phase;
            surface.v10_ms[[i, j]] = 1.0 - 0.015 * y + phase * 0.5;
            surface.temperature_2m_k[[i, j]] = 289.0 + 0.01 * y + phase;
            surface.dewpoint_2m_k[[i, j]] = 284.0 + 0.01 * y;
            surface.precip_large_scale_mm_h[[i, j]] = 0.0;
            surface.precip_convective_mm_h[[i, j]] = 0.0;
            surface.sensible_heat_flux_w_m2[[i, j]] = 50.0 + 0.1 * x;
            surface.solar_radiation_w_m2[[i, j]] = 250.0 + 0.2 * y;
            surface.surface_stress_n_m2[[i, j]] = 0.25 + 0.001 * x;
            surface.friction_velocity_ms[[i, j]] = 0.35;
            surface.convective_velocity_scale_ms[[i, j]] = 0.8;
            surface.mixing_height_m[[i, j]] = 900.0 + 0.5 * y;
            surface.tropopause_height_m[[i, j]] = 10_000.0;
            surface.inv_obukhov_length_per_m[[i, j]] = -0.01 + phase * 0.001;
        }
    }
    surface
}

fn deterministic_scalar_field(count: usize, base: f32, step: f32, modulo: usize) -> Vec<f32> {
    (0..count)
        .map(|idx| base + step * (idx % modulo) as f32)
        .collect()
}

fn compute_hanna_params_cpu_for_particles(
    particles: &[Particle],
    pbl_state: &flexpart_gpu::pbl::PblState,
) -> Vec<flexpart_gpu::pbl::HannaParams> {
    let (nx, ny) = pbl_state.shape();
    let max_x = nx.saturating_sub(1) as f32;
    let max_y = ny.saturating_sub(1) as f32;
    particles
        .iter()
        .map(|particle| {
            let i = (particle.cell_x as f32 + particle.pos_x)
                .clamp(0.0, max_x)
                .floor() as usize;
            let j = (particle.cell_y as f32 + particle.pos_y)
                .clamp(0.0, max_y)
                .floor() as usize;
            compute_hanna_params_from_pbl(pbl_state, i, j, particle.pos_z)
        })
        .collect()
}

fn apply_dry_deposition_step_cpu(
    particles: &mut [Particle],
    deposition_velocity_m_s: &[f32],
    params: DryDepositionStepParams,
) -> Vec<f32> {
    debug_assert_eq!(
        particles.len(),
        deposition_velocity_m_s.len(),
        "dry deposition forcing length must match particle slots"
    );
    particles
        .iter_mut()
        .zip(deposition_velocity_m_s.iter().copied())
        .map(|(particle, vdep)| {
            if particle.is_active() && in_dry_deposition_layer(particle, params.reference_height_m)
            {
                let probability = dry_deposition_probability_step(
                    vdep,
                    params.dt_seconds,
                    params.reference_height_m,
                );
                let survival = 1.0 - probability;
                for mass in &mut particle.mass {
                    *mass = (*mass * survival).max(0.0);
                }
                probability
            } else {
                0.0
            }
        })
        .collect()
}

fn apply_wet_deposition_step_cpu(
    particles: &mut [Particle],
    scavenging_coefficient_s_inv: &[f32],
    precipitating_fraction: &[f32],
    params: WetDepositionStepParams,
) -> Vec<f32> {
    debug_assert_eq!(
        particles.len(),
        scavenging_coefficient_s_inv.len(),
        "wet scavenging forcing length must match particle slots"
    );
    debug_assert_eq!(
        particles.len(),
        precipitating_fraction.len(),
        "wet precipitating fraction length must match particle slots"
    );
    particles
        .iter_mut()
        .zip(scavenging_coefficient_s_inv.iter().copied())
        .zip(precipitating_fraction.iter().copied())
        .map(|((particle, scavenging), fraction)| {
            if particle.is_active() {
                let probability = wet_scavenging_probability_step(WetScavengingStep {
                    scavenging_coefficient_s_inv: scavenging,
                    dt_seconds: params.dt_seconds,
                    precipitating_fraction: fraction,
                });
                let survival = 1.0 - probability;
                for mass in &mut particle.mass {
                    *mass = (*mass * survival).max(0.0);
                }
                probability
            } else {
                0.0
            }
        })
        .collect()
}

fn clamp_particle_axis(value: i32, upper_exclusive: usize) -> usize {
    let upper = upper_exclusive.saturating_sub(1);
    let as_usize = usize::try_from(value).unwrap_or(0);
    as_usize.min(upper)
}

fn accumulate_concentration_grid_cpu(
    particles: &[Particle],
    shape: ConcentrationGridShape,
    species_index: usize,
) -> (Vec<u32>, Vec<f32>) {
    if shape.nx == 0 || shape.ny == 0 || shape.nz == 0 {
        return (Vec::new(), Vec::new());
    }
    let cell_count = shape.nx * shape.ny * shape.nz;
    let mut particle_count_per_cell = vec![0_u32; cell_count];
    let mut concentration_mass_kg = vec![0.0_f32; cell_count];

    let max_z = shape.nz.saturating_sub(1) as f32;
    for particle in particles.iter().filter(|particle| particle.is_active()) {
        let ix = clamp_particle_axis(particle.cell_x, shape.nx);
        let iy = clamp_particle_axis(particle.cell_y, shape.ny);
        let iz = particle.pos_z.floor().clamp(0.0, max_z) as usize;
        let flat = ((ix * shape.ny) + iy) * shape.nz + iz;
        particle_count_per_cell[flat] = particle_count_per_cell[flat].saturating_add(1);
        concentration_mass_kg[flat] += particle.mass[species_index];
    }

    (particle_count_per_cell, concentration_mass_kg)
}

fn maybe_run_workgroup_autotune(runtime: BenchRuntimeConfig, ctx: &GpuContext) {
    if !parse_env_bool("FLEXPART_BENCH_AUTOTUNE", false) {
        return;
    }

    let tuning_particles = parse_env_usize(
        "FLEXPART_BENCH_AUTOTUNE_PARTICLES",
        runtime.max_particles.min(250_000),
    )
    .max(1);
    let particles = deterministic_particles(tuning_particles);
    let particle_buffers = ParticleBuffers::from_particles(ctx, &particles);

    let wind = deterministic_wind_field(0.15);
    let wind_buffers = match WindBuffers::from_field(ctx, &wind) {
        Ok(buffers) => buffers,
        Err(err) => {
            eprintln!("workgroup autotune skipped: wind upload failed: {err}");
            return;
        }
    };

    let surface = deterministic_surface_fields(0.05);
    let computed_pbl = match compute_pbl_parameters_from_met(
        PblMetInputGrids {
            surface: &surface,
            profile: None,
        },
        PblComputationOptions::default(),
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("workgroup autotune skipped: PBL compute failed: {err}");
            return;
        }
    };
    let pbl_buffers = match PblBuffers::from_state(ctx, &computed_pbl.pbl_state) {
        Ok(buffers) => buffers,
        Err(err) => {
            eprintln!("workgroup autotune skipped: PBL upload failed: {err}");
            return;
        }
    };
    let dry_velocity = deterministic_scalar_field(tuning_particles, 0.001, 0.00001, 41);
    let wet_scavenging = deterministic_scalar_field(tuning_particles, 0.0005, 0.00001, 73);
    let wet_fraction: Vec<f32> = (0..tuning_particles)
        .map(|idx| ((idx % 100) as f32) / 100.0)
        .collect();
    let grid_shape = ConcentrationGridShape {
        nx: BENCH_GRID_NX,
        ny: BENCH_GRID_NY,
        nz: BENCH_GRID_NZ,
    };
    let mut fused_langevin_kernels: BTreeMap<u32, LangevinFusedDispatchKernel> = BTreeMap::new();

    let mut options = WorkgroupAutoTuneOptions::from_env();
    if let Some(candidates) = parse_env_u32_list("FLEXPART_BENCH_AUTOTUNE_CANDIDATES") {
        options.candidate_sizes = candidates;
    }

    let report = auto_tune_key_kernels(ctx, &options, |kernel, candidate| {
        let _override = ScopedWorkgroupOverride::new(kernel, candidate);
        match kernel {
            WorkgroupKernel::Advection => {
                particle_buffers
                    .upload_particles(ctx, &particles)
                    .map_err(|err| err.to_string())?;
                let started = std::time::Instant::now();
                advect_particles_gpu(
                    ctx,
                    &particle_buffers,
                    &wind_buffers,
                    1.0,
                    VelocityToGridScale::IDENTITY,
                )
                .map_err(|err| err.to_string())?;
                Ok(started.elapsed())
            }
            WorkgroupKernel::HannaParams => {
                let started = std::time::Instant::now();
                let params = pollster::block_on(compute_hanna_params_gpu(
                    ctx,
                    &particle_buffers,
                    &pbl_buffers,
                ))
                .map_err(|err| err.to_string())?;
                std::hint::black_box(params);
                Ok(started.elapsed())
            }
            WorkgroupKernel::Langevin => {
                particle_buffers
                    .upload_particles(ctx, &particles)
                    .map_err(|err| err.to_string())?;
                // Tune against the production turbulence path (fused Hanna+Langevin),
                // not the legacy separated Langevin kernel used only for validation.
                let fused_kernel = fused_langevin_kernels
                    .entry(candidate)
                    .or_insert_with(|| LangevinFusedDispatchKernel::new(ctx));
                let started = std::time::Instant::now();
                let mut encoder = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("bench_autotune_langevin_fused"),
                    });
                let next_counter = encode_langevin_fused_gpu(
                    ctx,
                    &particle_buffers,
                    &pbl_buffers,
                    LangevinStep::legacy(1.0, 2.5e-4),
                    BENCH_LANGEVIN_KEY,
                    BENCH_LANGEVIN_COUNTER,
                    fused_kernel,
                    &mut encoder,
                )
                .map_err(|err| err.to_string())?;
                ctx.queue.submit(Some(encoder.finish()));
                let _ = ctx.device.poll(wgpu::MaintainBase::Wait);
                std::hint::black_box(next_counter);
                Ok(started.elapsed())
            }
            WorkgroupKernel::DryDeposition => {
                particle_buffers
                    .upload_particles(ctx, &particles)
                    .map_err(|err| err.to_string())?;
                let started = std::time::Instant::now();
                let output = pollster::block_on(apply_dry_deposition_step_gpu(
                    ctx,
                    &particle_buffers,
                    &dry_velocity,
                    DryDepositionStepParams {
                        dt_seconds: 1.0,
                        reference_height_m: 15.0,
                    },
                ))
                .map_err(|err| err.to_string())?;
                std::hint::black_box(output);
                Ok(started.elapsed())
            }
            WorkgroupKernel::WetDeposition => {
                particle_buffers
                    .upload_particles(ctx, &particles)
                    .map_err(|err| err.to_string())?;
                let started = std::time::Instant::now();
                let output = pollster::block_on(apply_wet_deposition_step_gpu(
                    ctx,
                    &particle_buffers,
                    &wet_scavenging,
                    &wet_fraction,
                    WetDepositionStepParams { dt_seconds: 1.0 },
                ))
                .map_err(|err| err.to_string())?;
                std::hint::black_box(output);
                Ok(started.elapsed())
            }
            WorkgroupKernel::ConcentrationGridding => {
                particle_buffers
                    .upload_particles(ctx, &particles)
                    .map_err(|err| err.to_string())?;
                let started = std::time::Instant::now();
                let output = pollster::block_on(accumulate_concentration_grid_gpu(
                    ctx,
                    &particle_buffers,
                    grid_shape,
                    ConcentrationGriddingParams {
                        species_index: 0,
                        mass_scale: 1_000_000.0,
                        outheights: [0.0; MAX_OUTPUT_LEVELS],
                    },
                ))
                .map_err(|err| err.to_string())?;
                std::hint::black_box(output);
                Ok(started.elapsed())
            }
            _ => Err(format!("kernel {kernel:?} not benchmarked in autotune")),
        }
    });

    match report {
        Ok(report) => match save_autotune_report_default(&report) {
            Ok(Some(path)) => {
                eprintln!(
                    "workgroup autotune complete: device={} backend={} cache={}",
                    report.adapter_name,
                    report.backend,
                    path.display()
                );
            }
            Ok(None) => {
                eprintln!(
                    "workgroup autotune complete: device={} backend={} (cache path unavailable)",
                    report.adapter_name, report.backend
                );
            }
            Err(err) => {
                eprintln!("workgroup autotune complete but cache write failed: {err}");
            }
        },
        Err(err) => {
            eprintln!("workgroup autotune failed: {err}");
        }
    }
}

fn bench_gpu_advection(c: &mut Criterion, runtime: BenchRuntimeConfig, ctx: &GpuContext) {
    let wind = deterministic_wind_field(0.0);
    let wind_buffers = WindBuffers::from_field(ctx, &wind).expect("wind upload should succeed");
    let mut group = c.benchmark_group("pipeline_stage_advection");
    configure_group(&mut group, runtime);

    for scenario in SCENARIOS {
        let effective_particles = runtime.effective_particles(scenario);
        let particles = deterministic_particles(effective_particles);
        let particle_buffers = ParticleBuffers::from_particles(ctx, &particles);
        let scenario_label = runtime.scenario_label(scenario);

        group.throughput(Throughput::Elements(
            u64::try_from(effective_particles).unwrap_or(u64::MAX),
        ));
        group.bench_function(BenchmarkId::new("advect", scenario_label), |b| {
            b.iter(|| {
                particle_buffers
                    .upload_particles(ctx, &particles)
                    .expect("particle upload should succeed");
                advect_particles_gpu(
                    ctx,
                    &particle_buffers,
                    &wind_buffers,
                    1.0,
                    VelocityToGridScale::IDENTITY,
                )
                .expect("advection dispatch should succeed");
                std::hint::black_box(());
            });
        });
    }
    group.finish();
}

fn bench_gpu_hanna(c: &mut Criterion, runtime: BenchRuntimeConfig, ctx: &GpuContext) {
    let surface = deterministic_surface_fields(0.0);
    let computed_pbl = compute_pbl_parameters_from_met(
        PblMetInputGrids {
            surface: &surface,
            profile: None,
        },
        PblComputationOptions::default(),
    )
    .expect("deterministic PBL computation should succeed");
    let pbl_buffers =
        PblBuffers::from_state(ctx, &computed_pbl.pbl_state).expect("pbl upload should succeed");

    let mut group = c.benchmark_group("pipeline_stage_hanna");
    configure_group(&mut group, runtime);

    for scenario in SCENARIOS {
        let effective_particles = runtime.effective_particles(scenario);
        let particles = deterministic_particles(effective_particles);
        let particle_buffers = ParticleBuffers::from_particles(ctx, &particles);
        let scenario_label = runtime.scenario_label(scenario);

        group.throughput(Throughput::Elements(
            u64::try_from(effective_particles).unwrap_or(u64::MAX),
        ));
        group.bench_function(BenchmarkId::new("hanna_params", scenario_label), |b| {
            b.iter(|| {
                let params = pollster::block_on(compute_hanna_params_gpu(
                    ctx,
                    &particle_buffers,
                    &pbl_buffers,
                ))
                .expect("hanna dispatch should succeed");
                std::hint::black_box(params);
            });
        });
    }
    group.finish();
}

fn bench_gpu_langevin(c: &mut Criterion, runtime: BenchRuntimeConfig, ctx: &GpuContext) {
    let surface = deterministic_surface_fields(0.05);
    let computed_pbl = compute_pbl_parameters_from_met(
        PblMetInputGrids {
            surface: &surface,
            profile: None,
        },
        PblComputationOptions::default(),
    )
    .expect("deterministic PBL computation should succeed");
    let pbl_buffers =
        PblBuffers::from_state(ctx, &computed_pbl.pbl_state).expect("pbl upload should succeed");

    let mut group = c.benchmark_group("pipeline_stage_langevin");
    configure_group(&mut group, runtime);

    for scenario in SCENARIOS {
        let effective_particles = runtime.effective_particles(scenario);
        let particles = deterministic_particles(effective_particles);
        let particle_buffers = ParticleBuffers::from_particles(ctx, &particles);
        let hanna_params = pollster::block_on(compute_hanna_params_gpu(
            ctx,
            &particle_buffers,
            &pbl_buffers,
        ))
        .expect("hanna params should be available for langevin");
        let scenario_label = runtime.scenario_label(scenario);

        group.throughput(Throughput::Elements(
            u64::try_from(effective_particles).unwrap_or(u64::MAX),
        ));
        group.bench_function(BenchmarkId::new("langevin_update", scenario_label), |b| {
            b.iter(|| {
                particle_buffers
                    .upload_particles(ctx, &particles)
                    .expect("particle upload should succeed");
                let next_counter = update_particles_turbulence_langevin_gpu(
                    ctx,
                    &particle_buffers,
                    &hanna_params,
                    LangevinStep::legacy(1.0, 2.5e-4),
                    [0xDECA_FBAD, 0x1234_5678],
                    [0, 0, 0, 0],
                )
                .expect("langevin dispatch should succeed");
                std::hint::black_box(next_counter);
            });
        });
    }
    group.finish();
}

fn bench_gpu_deposition(c: &mut Criterion, runtime: BenchRuntimeConfig, ctx: &GpuContext) {
    let mut dry_group = c.benchmark_group("pipeline_stage_dry_deposition");
    configure_group(&mut dry_group, runtime);
    for scenario in SCENARIOS {
        let effective_particles = runtime.effective_particles(scenario);
        let particles = deterministic_particles(effective_particles);
        let particle_buffers = ParticleBuffers::from_particles(ctx, &particles);
        let dry_velocity = deterministic_scalar_field(effective_particles, 0.001, 0.00001, 41);
        let scenario_label = runtime.scenario_label(scenario);
        dry_group.throughput(Throughput::Elements(
            u64::try_from(effective_particles).unwrap_or(u64::MAX),
        ));
        dry_group.bench_function(BenchmarkId::new("dry_probability", scenario_label), |b| {
            b.iter(|| {
                particle_buffers
                    .upload_particles(ctx, &particles)
                    .expect("particle upload should succeed");
                let output = pollster::block_on(apply_dry_deposition_step_gpu(
                    ctx,
                    &particle_buffers,
                    &dry_velocity,
                    DryDepositionStepParams {
                        dt_seconds: 1.0,
                        reference_height_m: 15.0,
                    },
                ))
                .expect("dry deposition should succeed");
                std::hint::black_box(output);
            });
        });
    }
    dry_group.finish();

    let mut wet_group = c.benchmark_group("pipeline_stage_wet_deposition");
    configure_group(&mut wet_group, runtime);
    for scenario in SCENARIOS {
        let effective_particles = runtime.effective_particles(scenario);
        let particles = deterministic_particles(effective_particles);
        let particle_buffers = ParticleBuffers::from_particles(ctx, &particles);
        let scavenging = deterministic_scalar_field(effective_particles, 0.0005, 0.00001, 73);
        let precip_fraction: Vec<f32> = (0..effective_particles)
            .map(|idx| ((idx % 100) as f32) / 100.0)
            .collect();
        let scenario_label = runtime.scenario_label(scenario);
        wet_group.throughput(Throughput::Elements(
            u64::try_from(effective_particles).unwrap_or(u64::MAX),
        ));
        wet_group.bench_function(BenchmarkId::new("wet_probability", scenario_label), |b| {
            b.iter(|| {
                particle_buffers
                    .upload_particles(ctx, &particles)
                    .expect("particle upload should succeed");
                let output = pollster::block_on(apply_wet_deposition_step_gpu(
                    ctx,
                    &particle_buffers,
                    &scavenging,
                    &precip_fraction,
                    WetDepositionStepParams { dt_seconds: 1.0 },
                ))
                .expect("wet deposition should succeed");
                std::hint::black_box(output);
            });
        });
    }
    wet_group.finish();
}

fn bench_gpu_concentration_gridding(
    c: &mut Criterion,
    runtime: BenchRuntimeConfig,
    ctx: &GpuContext,
) {
    let mut group = c.benchmark_group("pipeline_stage_concentration_gridding");
    configure_group(&mut group, runtime);

    for scenario in SCENARIOS {
        let effective_particles = runtime.effective_particles(scenario);
        let particles = deterministic_particles(effective_particles);
        let particle_buffers = ParticleBuffers::from_particles(ctx, &particles);
        let scenario_label = runtime.scenario_label(scenario);
        group.throughput(Throughput::Elements(
            u64::try_from(effective_particles).unwrap_or(u64::MAX),
        ));
        group.bench_function(BenchmarkId::new("particle_to_grid", scenario_label), |b| {
            b.iter(|| {
                particle_buffers
                    .upload_particles(ctx, &particles)
                    .expect("particle upload should succeed");
                let output = pollster::block_on(accumulate_concentration_grid_gpu(
                    ctx,
                    &particle_buffers,
                    ConcentrationGridShape {
                        nx: BENCH_GRID_NX,
                        ny: BENCH_GRID_NY,
                        nz: BENCH_GRID_NZ,
                    },
                    ConcentrationGriddingParams {
                        species_index: 0,
                        mass_scale: 1_000_000.0,
                        outheights: [0.0; MAX_OUTPUT_LEVELS],
                    },
                ))
                .expect("concentration gridding should succeed");
                std::hint::black_box(output);
            });
        });
    }
    group.finish();
}

fn bench_cpu_advection(c: &mut Criterion, runtime: BenchRuntimeConfig) {
    let wind = deterministic_wind_field(0.0);
    let mut group = c.benchmark_group("pipeline_stage_advection_cpu");
    configure_group(&mut group, runtime);

    for scenario in SCENARIOS {
        let effective_particles = runtime.effective_particles(scenario);
        let particles = deterministic_particles(effective_particles);
        let scenario_label = runtime.scenario_label(scenario);
        group.throughput(Throughput::Elements(
            u64::try_from(effective_particles).unwrap_or(u64::MAX),
        ));
        group.bench_function(BenchmarkId::new("advect_cpu", scenario_label), |b| {
            b.iter(|| {
                let mut working = particles.clone();
                advect_particles_cpu(&mut working, &wind, 1.0, VelocityToGridScale::IDENTITY);
                std::hint::black_box(working);
            });
        });
    }
    group.finish();
}

fn bench_cpu_hanna(c: &mut Criterion, runtime: BenchRuntimeConfig) {
    let surface = deterministic_surface_fields(0.0);
    let computed_pbl = compute_pbl_parameters_from_met(
        PblMetInputGrids {
            surface: &surface,
            profile: None,
        },
        PblComputationOptions::default(),
    )
    .expect("deterministic PBL computation should succeed");
    let mut group = c.benchmark_group("pipeline_stage_hanna_cpu");
    configure_group(&mut group, runtime);

    for scenario in SCENARIOS {
        let effective_particles = runtime.effective_particles(scenario);
        let particles = deterministic_particles(effective_particles);
        let scenario_label = runtime.scenario_label(scenario);
        group.throughput(Throughput::Elements(
            u64::try_from(effective_particles).unwrap_or(u64::MAX),
        ));
        group.bench_function(BenchmarkId::new("hanna_params_cpu", scenario_label), |b| {
            b.iter(|| {
                let params =
                    compute_hanna_params_cpu_for_particles(&particles, &computed_pbl.pbl_state);
                std::hint::black_box(params);
            });
        });
    }
    group.finish();
}

fn bench_cpu_langevin(c: &mut Criterion, runtime: BenchRuntimeConfig) {
    let surface = deterministic_surface_fields(0.05);
    let computed_pbl = compute_pbl_parameters_from_met(
        PblMetInputGrids {
            surface: &surface,
            profile: None,
        },
        PblComputationOptions::default(),
    )
    .expect("deterministic PBL computation should succeed");
    let mut group = c.benchmark_group("pipeline_stage_langevin_cpu");
    configure_group(&mut group, runtime);

    for scenario in SCENARIOS {
        let effective_particles = runtime.effective_particles(scenario);
        let particles = deterministic_particles(effective_particles);
        let hanna_params =
            compute_hanna_params_cpu_for_particles(&particles, &computed_pbl.pbl_state);
        let scenario_label = runtime.scenario_label(scenario);
        group.throughput(Throughput::Elements(
            u64::try_from(effective_particles).unwrap_or(u64::MAX),
        ));
        group.bench_function(
            BenchmarkId::new("langevin_update_cpu", scenario_label),
            |b| {
                b.iter(|| {
                    let mut working = particles.clone();
                    let mut rng = PhiloxRng::new(BENCH_LANGEVIN_KEY, BENCH_LANGEVIN_COUNTER);
                    update_particles_turbulence_langevin_with_rng_cpu(
                        &mut working,
                        &hanna_params,
                        LangevinStep::legacy(1.0, 2.5e-4),
                        &mut rng,
                    )
                    .expect("cpu langevin update should succeed");
                    std::hint::black_box(working);
                });
            },
        );
    }
    group.finish();
}

fn bench_cpu_deposition(c: &mut Criterion, runtime: BenchRuntimeConfig) {
    let mut dry_group = c.benchmark_group("pipeline_stage_dry_deposition_cpu");
    configure_group(&mut dry_group, runtime);
    for scenario in SCENARIOS {
        let effective_particles = runtime.effective_particles(scenario);
        let particles = deterministic_particles(effective_particles);
        let dry_velocity = deterministic_scalar_field(effective_particles, 0.001, 0.00001, 41);
        let scenario_label = runtime.scenario_label(scenario);
        dry_group.throughput(Throughput::Elements(
            u64::try_from(effective_particles).unwrap_or(u64::MAX),
        ));
        dry_group.bench_function(
            BenchmarkId::new("dry_probability_cpu", scenario_label),
            |b| {
                b.iter(|| {
                    let mut working = particles.clone();
                    let output = apply_dry_deposition_step_cpu(
                        &mut working,
                        &dry_velocity,
                        DryDepositionStepParams {
                            dt_seconds: 1.0,
                            reference_height_m: 15.0,
                        },
                    );
                    std::hint::black_box((working, output));
                });
            },
        );
    }
    dry_group.finish();

    let mut wet_group = c.benchmark_group("pipeline_stage_wet_deposition_cpu");
    configure_group(&mut wet_group, runtime);
    for scenario in SCENARIOS {
        let effective_particles = runtime.effective_particles(scenario);
        let particles = deterministic_particles(effective_particles);
        let scavenging = deterministic_scalar_field(effective_particles, 0.0005, 0.00001, 73);
        let precip_fraction: Vec<f32> = (0..effective_particles)
            .map(|idx| ((idx % 100) as f32) / 100.0)
            .collect();
        let scenario_label = runtime.scenario_label(scenario);
        wet_group.throughput(Throughput::Elements(
            u64::try_from(effective_particles).unwrap_or(u64::MAX),
        ));
        wet_group.bench_function(
            BenchmarkId::new("wet_probability_cpu", scenario_label),
            |b| {
                b.iter(|| {
                    let mut working = particles.clone();
                    let output = apply_wet_deposition_step_cpu(
                        &mut working,
                        &scavenging,
                        &precip_fraction,
                        WetDepositionStepParams { dt_seconds: 1.0 },
                    );
                    std::hint::black_box((working, output));
                });
            },
        );
    }
    wet_group.finish();
}

fn bench_cpu_concentration_gridding(c: &mut Criterion, runtime: BenchRuntimeConfig) {
    let mut group = c.benchmark_group("pipeline_stage_concentration_gridding_cpu");
    configure_group(&mut group, runtime);

    for scenario in SCENARIOS {
        let effective_particles = runtime.effective_particles(scenario);
        let particles = deterministic_particles(effective_particles);
        let scenario_label = runtime.scenario_label(scenario);
        group.throughput(Throughput::Elements(
            u64::try_from(effective_particles).unwrap_or(u64::MAX),
        ));
        group.bench_function(
            BenchmarkId::new("particle_to_grid_cpu", scenario_label),
            |b| {
                b.iter(|| {
                    let output = accumulate_concentration_grid_cpu(
                        &particles,
                        ConcentrationGridShape {
                            nx: BENCH_GRID_NX,
                            ny: BENCH_GRID_NY,
                            nz: BENCH_GRID_NZ,
                        },
                        0,
                    );
                    std::hint::black_box(output);
                });
            },
        );
    }
    group.finish();
}

fn bench_forward_timeloop_e2e(c: &mut Criterion, runtime: BenchRuntimeConfig) {
    let mut group = c.benchmark_group("pipeline_end_to_end_timeloop");
    configure_group(&mut group, runtime);
    let timeloop_sync_host = parse_env_bool("FLEXPART_BENCH_TIMELOOP_SYNC_HOST", false);
    let timeloop_collect_probabilities =
        parse_env_bool("FLEXPART_BENCH_TIMELOOP_COLLECT_PROBABILITIES", false);

    for scenario in SCENARIOS {
        let effective_particles = runtime.effective_particles(scenario);
        let scenario_label = runtime.scenario_label(scenario);

        let mut raw = BTreeMap::new();
        raw.insert("lon_min".to_string(), "10.0".to_string());
        raw.insert("lon_max".to_string(), "30.0".to_string());
        raw.insert("lat_min".to_string(), "5.0".to_string());
        raw.insert("lat_max".to_string(), "25.0".to_string());
        let releases = vec![ReleaseConfig {
            name: "benchmark_release".to_string(),
            start_time: "20240101000000".to_string(),
            end_time: "20240101000000".to_string(),
            lon: 10.0,
            lat: 5.0,
            z_min: 1.0,
            z_max: (BENCH_GRID_NZ.saturating_sub(2)) as f64,
            mass_kg: 1.0,
            particle_count: effective_particles as u64,
            raw,
        }];
        let release_grid = GridDomain {
            xlon0: 0.0,
            ylat0: 0.0,
            dx: 1.0,
            dy: 1.0,
            nx: BENCH_GRID_NX,
            ny: BENCH_GRID_NY,
        };
        let config = ForwardTimeLoopConfig {
            start_timestamp: "20240101000000".to_string(),
            end_timestamp: "20300101000000".to_string(),
            timestep_seconds: 1,
            time_bounds_behavior: TimeBoundsBehavior::Clamp,
            velocity_to_grid_scale: VelocityToGridScale::IDENTITY,
            sync_particle_store_each_step: timeloop_sync_host,
            collect_deposition_probabilities_each_step: timeloop_collect_probabilities,
            ..ForwardTimeLoopConfig::default()
        };

        let mut driver = match pollster::block_on(ForwardTimeLoopDriver::new(
            config,
            &releases,
            release_grid,
            effective_particles,
        )) {
            Ok(driver) => driver,
            Err(TimeLoopError::Gpu(GpuError::NoAdapter)) => return,
            Err(err) => panic!("timeloop benchmark driver initialization should succeed: {err}"),
        };

        let wind_t0 = deterministic_wind_field(0.0);
        let wind_t1 = deterministic_wind_field(0.1);
        let surface_t0 = deterministic_surface_fields(0.0);
        let surface_t1 = deterministic_surface_fields(0.2);
        let met = MetTimeBracket {
            wind_t0: &wind_t0,
            wind_t1: &wind_t1,
            surface_t0: &surface_t0,
            surface_t1: &surface_t1,
            time_t0_seconds: driver.current_time_seconds(),
            time_t1_seconds: driver.current_time_seconds() + 600,
        };

        let forcing = ForwardStepForcing {
            dry_deposition_velocity_m_s: ParticleForcingField::PerParticle(
                deterministic_scalar_field(effective_particles, 0.001, 0.00001, 23),
            ),
            wet_scavenging_coefficient_s_inv: ParticleForcingField::PerParticle(
                deterministic_scalar_field(effective_particles, 0.0005, 0.00001, 29),
            ),
            wet_precipitating_fraction: ParticleForcingField::PerParticle(
                (0..effective_particles)
                    .map(|idx| ((idx % 100) as f32) / 100.0)
                    .collect(),
            ),
            rho_grad_over_rho: 2.5e-4,
        };

        // Prime one step so steady-state timing excludes first-step release spike.
        let _warmup_report = pollster::block_on(driver.run_timestep(&met, &forcing))
            .expect("timeloop warmup step should succeed");

        group.throughput(Throughput::Elements(
            u64::try_from(effective_particles).unwrap_or(u64::MAX),
        ));
        group.bench_function(BenchmarkId::new("forward_step", scenario_label), |b| {
            b.iter(|| {
                let report = pollster::block_on(driver.run_timestep(&met, &forcing))
                    .expect("timeloop step should succeed");
                std::hint::black_box(report);
            });
        });
    }
    group.finish();
}

fn bench_forward_timeloop_e2e_cpu(c: &mut Criterion, runtime: BenchRuntimeConfig) {
    let mut group = c.benchmark_group("pipeline_end_to_end_timeloop_cpu");
    configure_group(&mut group, runtime);

    for scenario in SCENARIOS {
        let effective_particles = runtime.effective_particles(scenario);
        let scenario_label = runtime.scenario_label(scenario);
        let particles = deterministic_particles(effective_particles);
        let wind_t0 = deterministic_wind_field(0.0);
        let wind_t1 = deterministic_wind_field(0.1);
        let surface_t0 = deterministic_surface_fields(0.0);
        let surface_t1 = deterministic_surface_fields(0.2);
        let dry_velocity = deterministic_scalar_field(effective_particles, 0.001, 0.00001, 23);
        let wet_scavenging = deterministic_scalar_field(effective_particles, 0.0005, 0.00001, 29);
        let wet_fraction: Vec<f32> = (0..effective_particles)
            .map(|idx| ((idx % 100) as f32) / 100.0)
            .collect();

        group.throughput(Throughput::Elements(
            u64::try_from(effective_particles).unwrap_or(u64::MAX),
        ));
        group.bench_function(BenchmarkId::new("forward_step_cpu", scenario_label), |b| {
            b.iter(|| {
                let mut working = particles.clone();
                let interpolated_wind = interpolate_wind_field_linear(
                    &wind_t0,
                    &wind_t1,
                    0,
                    BENCH_WIND_BRACKET_SECONDS,
                    BENCH_INTERPOLATION_TARGET_SECONDS,
                    TimeBoundsBehavior::Clamp,
                )
                .expect("cpu wind interpolation should succeed");
                let interpolated_surface = interpolate_surface_fields_linear(
                    &surface_t0,
                    &surface_t1,
                    0,
                    BENCH_WIND_BRACKET_SECONDS,
                    BENCH_INTERPOLATION_TARGET_SECONDS,
                    TimeBoundsBehavior::Clamp,
                )
                .expect("cpu surface interpolation should succeed");
                let computed_pbl = compute_pbl_parameters_from_met(
                    PblMetInputGrids {
                        surface: &interpolated_surface,
                        profile: None,
                    },
                    PblComputationOptions::default(),
                )
                .expect("cpu PBL computation should succeed");
                advect_particles_cpu(
                    &mut working,
                    &interpolated_wind,
                    1.0,
                    VelocityToGridScale::IDENTITY,
                );
                let hanna_params =
                    compute_hanna_params_cpu_for_particles(&working, &computed_pbl.pbl_state);
                let mut rng = PhiloxRng::new(BENCH_LANGEVIN_KEY, BENCH_LANGEVIN_COUNTER);
                update_particles_turbulence_langevin_with_rng_cpu(
                    &mut working,
                    &hanna_params,
                    LangevinStep::legacy(1.0, 2.5e-4),
                    &mut rng,
                )
                .expect("cpu langevin step should succeed");

                let dry_probability = apply_dry_deposition_step_cpu(
                    &mut working,
                    &dry_velocity,
                    DryDepositionStepParams {
                        dt_seconds: 1.0,
                        reference_height_m: 15.0,
                    },
                );
                let wet_probability = apply_wet_deposition_step_cpu(
                    &mut working,
                    &wet_scavenging,
                    &wet_fraction,
                    WetDepositionStepParams { dt_seconds: 1.0 },
                );
                let concentration = accumulate_concentration_grid_cpu(
                    &working,
                    ConcentrationGridShape {
                        nx: BENCH_GRID_NX,
                        ny: BENCH_GRID_NY,
                        nz: BENCH_GRID_NZ,
                    },
                    0,
                );
                std::hint::black_box((dry_probability, wet_probability, concentration));
            });
        });
    }
    group.finish();
}

fn bench_pipeline_suite(c: &mut Criterion) {
    let runtime = BenchRuntimeConfig::from_env();
    match pollster::block_on(GpuContext::new()) {
        Ok(ctx) => {
            maybe_run_workgroup_autotune(runtime, &ctx);

            // CPU-side met interpolation and PBL computation are included via the end-to-end
            // timeloop benchmark. Stage benchmarks focus on the core per-particle kernels.
            bench_gpu_advection(c, runtime, &ctx);
            bench_gpu_hanna(c, runtime, &ctx);
            bench_gpu_langevin(c, runtime, &ctx);
            bench_gpu_deposition(c, runtime, &ctx);
            bench_gpu_concentration_gridding(c, runtime, &ctx);
            bench_forward_timeloop_e2e(c, runtime);
        }
        Err(GpuError::NoAdapter) => {
            c.bench_function("pipeline_gpu_unavailable", |b| {
                b.iter(|| std::hint::black_box("No GPU adapter available"))
            });
        }
        Err(err) => panic!("GPU initialization failed for benchmark suite: {err}"),
    }

    // Always expose a CPU reference baseline so GPU speedups can be quantified
    // even on machines without a functional GPU runtime.
    bench_cpu_advection(c, runtime);
    bench_cpu_hanna(c, runtime);
    bench_cpu_langevin(c, runtime);
    bench_cpu_deposition(c, runtime);
    bench_cpu_concentration_gridding(c, runtime);
    bench_forward_timeloop_e2e_cpu(c, runtime);
}

criterion_group!(benches, bench_pipeline_suite);
criterion_main!(benches);
