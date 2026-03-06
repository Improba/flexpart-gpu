//! GPU Langevin turbulence update dispatch (H-04).
//!
//! Ported from FLEXPART `advance.f90:300-500` through the CPU reference in
//! `src/physics/langevin.rs` (H-03).
//!
//! This dispatch updates `Particle::turb_u`, `Particle::turb_v`, and
//! `Particle::turb_w` in-place for active particles, using:
//! - per-particle [`HannaParams`] (from H-02/H-01)
//! - deterministic Philox RNG (A-05), one block per particle slot
//! - Box-Muller Gaussian forcing
//!
//! MVP assumption: CBL-specific turbulence branching remains out of scope here
//! and matches current H-03 CPU behavior.

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::pbl::HannaParams;
use crate::physics::{philox_counter_add, LangevinStep, PhiloxCounter, PhiloxKey};

use super::{
    render_shader_with_workgroup_size, runtime_workgroup_size, GpuContext, ParticleBuffers,
    WorkgroupKernel,
};

const SHADER_TEMPLATE: &str = include_str!("../shaders/langevin.wgsl");

/// Reusable Langevin dispatch kernel objects (bind group layout + compute pipeline).
pub struct LangevinDispatchKernel {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline: wgpu::ComputePipeline,
    workgroup_size_x: u32,
}

impl LangevinDispatchKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let workgroup_size_x = runtime_workgroup_size(ctx, WorkgroupKernel::Langevin);
        let shader_source = render_shader_with_workgroup_size(SHADER_TEMPLATE, workgroup_size_x);
        let shader = ctx.load_shader("langevin_shader", &shader_source);
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("langevin_bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let pipeline = ctx.create_compute_pipeline(
            "langevin_pipeline",
            &shader,
            "main",
            &[&bind_group_layout],
        );
        Self {
            bind_group_layout,
            pipeline,
            workgroup_size_x,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LangevinDispatchParams {
    key0: u32,
    key1: u32,
    counter0: u32,
    counter1: u32,
    counter2: u32,
    counter3: u32,
    particle_count: u32,
    n_substeps: u32,
    dt_seconds: f32,
    rho_grad_over_rho: f32,
    min_height_m: f32,
    _pad0: u32,
}

/// Errors returned by GPU Langevin turbulence dispatch.
#[derive(Debug, Error)]
pub enum GpuLangevinError {
    #[error("invalid dt_seconds for Langevin update: {dt_seconds}")]
    InvalidTimeStep { dt_seconds: f32 },
    #[error(
        "mismatched input lengths: particle_slots={particle_slots}, hanna_params={hanna_params}"
    )]
    MismatchedInputLengths {
        particle_slots: usize,
        hanna_params: usize,
    },
    #[error("value for {field} does not fit in u32: {value}")]
    ValueTooLarge { field: &'static str, value: usize },
}

fn usize_to_u32(value: usize, field: &'static str) -> Result<u32, GpuLangevinError> {
    u32::try_from(value).map_err(|_| GpuLangevinError::ValueTooLarge { field, value })
}

/// Dispatch the WGSL Langevin turbulence kernel for all particle slots.
///
/// The kernel always consumes one Philox block per slot (active or inactive),
/// then updates only active particle turbulence memory.
///
/// Returns the next Philox counter (`base_counter + particle_count`) so callers
/// can chain deterministic timesteps.
///
/// # Errors
///
/// Returns:
/// - [`GpuLangevinError::InvalidTimeStep`] when `dt_seconds` is not finite or
///   not strictly positive
/// - [`GpuLangevinError::MismatchedInputLengths`] when `hanna_params.len()`
///   differs from the particle buffer slot count
/// - [`GpuLangevinError::ValueTooLarge`] when particle count does not fit in
///   a WGSL `u32`
pub fn update_particles_turbulence_langevin_gpu(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    hanna_params: &[HannaParams],
    step: LangevinStep,
    key: PhiloxKey,
    base_counter: PhiloxCounter,
) -> Result<PhiloxCounter, GpuLangevinError> {
    let particle_count = particles.particle_count();
    if particle_count != hanna_params.len() {
        return Err(GpuLangevinError::MismatchedInputLengths {
            particle_slots: particle_count,
            hanna_params: hanna_params.len(),
        });
    }
    if particle_count == 0 {
        return Ok(base_counter);
    }
    if !step.dt_seconds.is_finite() || step.dt_seconds <= 0.0 {
        return Err(GpuLangevinError::InvalidTimeStep {
            dt_seconds: step.dt_seconds,
        });
    }

    let hanna_params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("langevin_hanna_params"),
            contents: bytemuck::cast_slice(hanna_params),
            usage: wgpu::BufferUsages::STORAGE,
        });
    update_particles_turbulence_langevin_gpu_with_hanna_buffer(
        ctx,
        particles,
        &hanna_params_buffer,
        hanna_params.len(),
        step,
        key,
        base_counter,
    )
}

/// Dispatch Langevin turbulence update using an existing GPU Hanna-parameter buffer.
///
/// This avoids host-side Hanna download/upload between H-02 and H-04 kernels.
pub fn update_particles_turbulence_langevin_gpu_with_hanna_buffer(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    hanna_params_buffer: &wgpu::Buffer,
    hanna_params_len: usize,
    step: LangevinStep,
    key: PhiloxKey,
    base_counter: PhiloxCounter,
) -> Result<PhiloxCounter, GpuLangevinError> {
    let kernel = LangevinDispatchKernel::new(ctx);
    update_particles_turbulence_langevin_gpu_with_hanna_buffer_and_kernel(
        ctx,
        particles,
        hanna_params_buffer,
        hanna_params_len,
        step,
        key,
        base_counter,
        &kernel,
    )
}

/// Dispatch Langevin update using an existing Hanna buffer and prepared kernel.
pub fn update_particles_turbulence_langevin_gpu_with_hanna_buffer_and_kernel(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    hanna_params_buffer: &wgpu::Buffer,
    hanna_params_len: usize,
    step: LangevinStep,
    key: PhiloxKey,
    base_counter: PhiloxCounter,
    kernel: &LangevinDispatchKernel,
) -> Result<PhiloxCounter, GpuLangevinError> {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("langevin_encoder"),
        });
    let next_counter =
        encode_update_particles_turbulence_langevin_gpu_with_hanna_buffer_and_kernel(
            ctx,
            particles,
            hanna_params_buffer,
            hanna_params_len,
            step,
            key,
            base_counter,
            kernel,
            &mut encoder,
        )?;
    ctx.queue.submit(Some(encoder.finish()));
    let _ = ctx.device.poll(wgpu::Maintain::Wait);
    Ok(next_counter)
}

/// Encode Langevin dispatch into a caller-provided command encoder.
pub fn encode_update_particles_turbulence_langevin_gpu_with_hanna_buffer_and_kernel(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    hanna_params_buffer: &wgpu::Buffer,
    hanna_params_len: usize,
    step: LangevinStep,
    key: PhiloxKey,
    base_counter: PhiloxCounter,
    kernel: &LangevinDispatchKernel,
    encoder: &mut wgpu::CommandEncoder,
) -> Result<PhiloxCounter, GpuLangevinError> {
    let particle_count = particles.particle_count();
    if particle_count != hanna_params_len {
        return Err(GpuLangevinError::MismatchedInputLengths {
            particle_slots: particle_count,
            hanna_params: hanna_params_len,
        });
    }
    if particle_count == 0 {
        return Ok(base_counter);
    }
    if !step.dt_seconds.is_finite() || step.dt_seconds <= 0.0 {
        return Err(GpuLangevinError::InvalidTimeStep {
            dt_seconds: step.dt_seconds,
        });
    }

    let params = LangevinDispatchParams {
        key0: key[0],
        key1: key[1],
        counter0: base_counter[0],
        counter1: base_counter[1],
        counter2: base_counter[2],
        counter3: base_counter[3],
        particle_count: usize_to_u32(particle_count, "particle_count")?,
        n_substeps: step.n_substeps,
        dt_seconds: step.dt_seconds,
        rho_grad_over_rho: step.rho_grad_over_rho,
        min_height_m: step.min_height_m,
        _pad0: 0,
    };
    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("langevin_dispatch_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("langevin_bg"),
        layout: &kernel.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: particles.particle_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: hanna_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("langevin_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&kernel.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        super::dispatch_1d(&mut cpass, params.particle_count, kernel.workgroup_size_x);
    }
    let blocks_per_particle: u64 = if step.n_substeps <= 2 { 1 } else { 2 };
    Ok(philox_counter_add(
        base_counter,
        (particle_count as u64) * blocks_per_particle,
    ))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use bytemuck::Zeroable;

    use super::*;
    use crate::gpu::{
        compute_hanna_params_gpu, sample_cbl_vertical_velocity_gpu, CblSamplingInput,
        CblSamplingOutput, GpuError, PblBuffers,
    };
    use crate::particles::{Particle, ParticleInit, MAX_SPECIES};
    use crate::pbl::{HannaParams, PblState};
    use crate::physics::{
        compute_cbl_bigaussian_pdf, compute_hanna_params, obukhov_length_from_inverse,
        sample_cbl_vertical_velocity, update_particles_turbulence_langevin_with_rng_cpu,
        CblPdfInputs, HannaInputs, PhiloxRng,
    };

    fn make_particle(x: f32, y: f32, z: f32) -> Particle {
        let cell_x = x.floor() as i32;
        let cell_y = y.floor() as i32;
        Particle::new(&ParticleInit {
            cell_x,
            cell_y,
            pos_x: x - cell_x as f32,
            pos_y: y - cell_y as f32,
            pos_z: z,
            mass: [0.0; MAX_SPECIES],
            release_point: 0,
            class: 0,
            time: 0,
        })
    }

    fn assert_particle_common_fields_equal(expected: &Particle, actual: &Particle) {
        assert_eq!(actual.cell_x, expected.cell_x);
        assert_eq!(actual.cell_y, expected.cell_y);
        assert_eq!(actual.flags, expected.flags);
        assert_eq!(actual.time, expected.time);
        assert_eq!(actual.timestep, expected.timestep);
        assert_eq!(actual.time_mem, expected.time_mem);
        assert_eq!(actual.time_split, expected.time_split);
        assert_eq!(actual.release_point, expected.release_point);
        assert_eq!(actual.class, expected.class);
        assert_eq!(actual.cbt, expected.cbt);
        assert_eq!(actual.pad0, expected.pad0);
        assert_relative_eq!(actual.pos_x, expected.pos_x, epsilon = 1.0e-7);
        assert_relative_eq!(actual.pos_y, expected.pos_y, epsilon = 1.0e-7);
        assert_relative_eq!(actual.pos_z, expected.pos_z, epsilon = 1.0e-7);
        assert_relative_eq!(actual.vel_u, expected.vel_u, epsilon = 1.0e-7);
        assert_relative_eq!(actual.vel_v, expected.vel_v, epsilon = 1.0e-7);
        assert_relative_eq!(actual.vel_w, expected.vel_w, epsilon = 1.0e-7);
        for (lhs, rhs) in actual.mass.iter().zip(expected.mass.iter()) {
            assert_relative_eq!(*lhs, *rhs, epsilon = 1.0e-7);
        }
    }

    fn expected_hanna_from_cpu(pbl: &PblState, particle: &Particle) -> HannaParams {
        if !particle.is_active() {
            return HannaParams::zeroed();
        }

        let (nx, ny) = pbl.shape();
        let x = (particle.cell_x as f32 + particle.pos_x).clamp(0.0, (nx - 1) as f32);
        let y = (particle.cell_y as f32 + particle.pos_y).clamp(0.0, (ny - 1) as f32);
        let i = x.floor() as usize;
        let j = y.floor() as usize;

        compute_hanna_params(HannaInputs {
            ust: pbl.ustar[[i, j]],
            wst: pbl.wstar[[i, j]],
            ol: obukhov_length_from_inverse(pbl.oli[[i, j]]),
            h: pbl.hmix[[i, j]],
            z: particle.pos_z,
        })
    }

    fn assert_hanna_scalar_close(actual: f32, expected: f32) {
        if expected.is_infinite() {
            assert!(actual.is_infinite());
            assert_eq!(actual.is_sign_positive(), expected.is_sign_positive());
        } else {
            assert_relative_eq!(actual, expected, epsilon = 5.0e-5, max_relative = 5.0e-5);
        }
    }

    fn assert_hanna_params_close(actual: &HannaParams, expected: &HannaParams) {
        assert_hanna_scalar_close(actual.ust, expected.ust);
        assert_hanna_scalar_close(actual.wst, expected.wst);
        assert_hanna_scalar_close(actual.ol, expected.ol);
        assert_hanna_scalar_close(actual.h, expected.h);
        assert_hanna_scalar_close(actual.zeta, expected.zeta);
        assert_hanna_scalar_close(actual.sigu, expected.sigu);
        assert_hanna_scalar_close(actual.sigv, expected.sigv);
        assert_hanna_scalar_close(actual.sigw, expected.sigw);
        assert_hanna_scalar_close(actual.dsigwdz, expected.dsigwdz);
        assert_hanna_scalar_close(actual.dsigw2dz, expected.dsigw2dz);
        assert_hanna_scalar_close(actual.tlu, expected.tlu);
        assert_hanna_scalar_close(actual.tlv, expected.tlv);
        assert_hanna_scalar_close(actual.tlw, expected.tlw);
    }

    fn expected_cbl_output_from_cpu(
        particle: &Particle,
        hanna: &HannaParams,
        sample: CblSamplingInput,
    ) -> CblSamplingOutput {
        if !particle.is_active() {
            return CblSamplingOutput::zeroed();
        }

        let pdf = compute_cbl_bigaussian_pdf(CblPdfInputs {
            z_m: particle.pos_z,
            h_m: hanna.h,
            wstar_m_s: hanna.wst,
            sigma_w_m_s: hanna.sigw,
            obukhov_length_m: hanna.ol,
        });
        let sampled_w_m_s = sample_cbl_vertical_velocity(
            &pdf,
            sample.branch_uniform,
            sample.gaussian_uniform0,
            sample.gaussian_uniform1,
        );

        CblSamplingOutput {
            sampled_w_m_s,
            updraft_weight: pdf.updraft.weight,
            updraft_mean_m_s: pdf.updraft.mean_m_s,
            updraft_sigma_m_s: pdf.updraft.sigma_m_s,
            downdraft_weight: pdf.downdraft.weight,
            downdraft_mean_m_s: pdf.downdraft.mean_m_s,
            downdraft_sigma_m_s: pdf.downdraft.sigma_m_s,
            second_moment_w2: pdf.target_moments.second_moment_w2,
            third_moment_w3: pdf.target_moments.third_moment_w3,
            skewness: pdf.target_moments.skewness,
            transition: pdf.transition,
            z_over_h: pdf.z_over_h,
        }
    }

    fn assert_cbl_output_close(actual: &CblSamplingOutput, expected: &CblSamplingOutput) {
        assert_relative_eq!(
            actual.sampled_w_m_s,
            expected.sampled_w_m_s,
            epsilon = 5.0e-5,
            max_relative = 5.0e-5
        );
        assert_relative_eq!(
            actual.updraft_weight,
            expected.updraft_weight,
            epsilon = 5.0e-5,
            max_relative = 5.0e-5
        );
        assert_relative_eq!(
            actual.updraft_mean_m_s,
            expected.updraft_mean_m_s,
            epsilon = 5.0e-5,
            max_relative = 5.0e-5
        );
        assert_relative_eq!(
            actual.updraft_sigma_m_s,
            expected.updraft_sigma_m_s,
            epsilon = 5.0e-5,
            max_relative = 5.0e-5
        );
        assert_relative_eq!(
            actual.downdraft_weight,
            expected.downdraft_weight,
            epsilon = 5.0e-5,
            max_relative = 5.0e-5
        );
        assert_relative_eq!(
            actual.downdraft_mean_m_s,
            expected.downdraft_mean_m_s,
            epsilon = 5.0e-5,
            max_relative = 5.0e-5
        );
        assert_relative_eq!(
            actual.downdraft_sigma_m_s,
            expected.downdraft_sigma_m_s,
            epsilon = 5.0e-5,
            max_relative = 5.0e-5
        );
        assert_relative_eq!(
            actual.second_moment_w2,
            expected.second_moment_w2,
            epsilon = 5.0e-5,
            max_relative = 5.0e-5
        );
        assert_relative_eq!(
            actual.third_moment_w3,
            expected.third_moment_w3,
            epsilon = 5.0e-4,
            max_relative = 5.0e-4
        );
        assert_relative_eq!(
            actual.skewness,
            expected.skewness,
            epsilon = 5.0e-4,
            max_relative = 5.0e-4
        );
        assert_relative_eq!(
            actual.transition,
            expected.transition,
            epsilon = 5.0e-5,
            max_relative = 5.0e-5
        );
        assert_relative_eq!(
            actual.z_over_h,
            expected.z_over_h,
            epsilon = 5.0e-6,
            max_relative = 5.0e-6
        );
    }

    #[test]
    fn gpu_langevin_matches_cpu_reference_on_deterministic_inputs() {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let mut gpu_input = vec![
            make_particle(0.2, 0.0, 120.0),
            make_particle(1.2, 0.0, 80.0),
            make_particle(2.2, 0.0, 200.0),
            make_particle(0.4, 0.0, 30.0),
        ];
        gpu_input[0].turb_u = 0.8;
        gpu_input[0].turb_v = -0.3;
        gpu_input[0].turb_w = 0.1;
        gpu_input[1].turb_u = -0.5;
        gpu_input[1].turb_v = 1.1;
        gpu_input[1].turb_w = -0.2;
        gpu_input[2].turb_u = 0.4;
        gpu_input[2].turb_v = 0.7;
        gpu_input[2].turb_w = -1.0;
        gpu_input[3].deactivate();
        gpu_input[3].turb_u = 3.0;
        gpu_input[3].turb_v = -2.0;
        gpu_input[3].turb_w = 1.0;

        let hanna = vec![
            compute_hanna_params(HannaInputs {
                ust: 0.30,
                wst: 1.80,
                ol: -50.0,
                h: 900.0,
                z: 120.0,
            }),
            compute_hanna_params(HannaInputs {
                ust: 0.25,
                wst: 0.0,
                ol: f32::INFINITY,
                h: 800.0,
                z: 80.0,
            }),
            compute_hanna_params(HannaInputs {
                ust: 0.35,
                wst: 0.0,
                ol: 100.0,
                h: 700.0,
                z: 200.0,
            }),
            compute_hanna_params(HannaInputs {
                ust: 0.2,
                wst: 0.0,
                ol: 80.0,
                h: 600.0,
                z: 30.0,
            }),
        ];

        let step = LangevinStep::legacy(45.0, 2.5e-4);
        let key = [0xDECA_FBAD, 0x1234_5678];
        let base_counter = [0xA5A5_0001, 0xFACE_B00C, 7, 9];

        let mut expected = gpu_input.clone();
        let mut cpu_rng = PhiloxRng::new(key, base_counter);
        update_particles_turbulence_langevin_with_rng_cpu(
            &mut expected,
            &hanna,
            step,
            &mut cpu_rng,
        )
        .expect("cpu reference update succeeds");

        let particle_buffers = ParticleBuffers::from_particles(&ctx, &gpu_input);
        let next_counter = update_particles_turbulence_langevin_gpu(
            &ctx,
            &particle_buffers,
            &hanna,
            step,
            key,
            base_counter,
        )
        .expect("gpu langevin dispatch succeeds");
        let actual = pollster::block_on(particle_buffers.download_particles(&ctx))
            .expect("particle readback succeeds");

        assert_eq!(actual.len(), expected.len());
        for (cpu, gpu) in expected.iter().zip(actual.iter()) {
            assert_particle_common_fields_equal(cpu, gpu);
            assert_relative_eq!(
                gpu.turb_u,
                cpu.turb_u,
                epsilon = 5.0e-5,
                max_relative = 5.0e-5
            );
            assert_relative_eq!(
                gpu.turb_v,
                cpu.turb_v,
                epsilon = 5.0e-5,
                max_relative = 5.0e-5
            );
            assert_relative_eq!(
                gpu.turb_w,
                cpu.turb_w,
                epsilon = 5.0e-5,
                max_relative = 5.0e-5
            );
        }

        assert_eq!(
            next_counter,
            philox_counter_add(base_counter, gpu_input.len() as u64)
        );
    }

    #[test]
    fn gpu_full_turbulence_step_matches_cpu_reference_with_cbl() {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let mut initial_particles = vec![
            make_particle(0.2, 0.0, 120.0),
            make_particle(1.1, 0.0, 80.0),
            make_particle(2.1, 0.0, 200.0),
            make_particle(3.0, 0.0, 300.0),
            make_particle(3.2, 0.0, 30.0),
        ];
        initial_particles[0].turb_u = 0.8;
        initial_particles[0].turb_v = -0.3;
        initial_particles[0].turb_w = 0.1;
        initial_particles[1].turb_u = -0.5;
        initial_particles[1].turb_v = 1.1;
        initial_particles[1].turb_w = -0.2;
        initial_particles[2].turb_u = 0.4;
        initial_particles[2].turb_v = 0.7;
        initial_particles[2].turb_w = -1.0;
        initial_particles[3].turb_u = -0.2;
        initial_particles[3].turb_v = -0.4;
        initial_particles[3].turb_w = 0.6;
        initial_particles[4].deactivate();

        let mut pbl = PblState::new(4, 1);
        // Unstable (L = -50 m)
        pbl.ustar[[0, 0]] = 0.30;
        pbl.wstar[[0, 0]] = 1.80;
        pbl.hmix[[0, 0]] = 900.0;
        pbl.oli[[0, 0]] = -0.02;
        // Neutral (L = +inf)
        pbl.ustar[[1, 0]] = 0.25;
        pbl.wstar[[1, 0]] = 0.0;
        pbl.hmix[[1, 0]] = 800.0;
        pbl.oli[[1, 0]] = 0.0;
        // Stable (L = +100 m)
        pbl.ustar[[2, 0]] = 0.35;
        pbl.wstar[[2, 0]] = 0.0;
        pbl.hmix[[2, 0]] = 700.0;
        pbl.oli[[2, 0]] = 0.01;
        // Unstable (L = -80 m)
        pbl.ustar[[3, 0]] = 0.28;
        pbl.wstar[[3, 0]] = 1.60;
        pbl.hmix[[3, 0]] = 1_000.0;
        pbl.oli[[3, 0]] = -0.0125;

        let pbl_buffers = PblBuffers::from_state(&ctx, &pbl).expect("pbl upload succeeds");
        let particle_buffers = ParticleBuffers::from_particles(&ctx, &initial_particles);

        let hanna_gpu = pollster::block_on(compute_hanna_params_gpu(
            &ctx,
            &particle_buffers,
            &pbl_buffers,
        ))
        .expect("gpu hanna dispatch succeeds");
        let hanna_cpu: Vec<HannaParams> = initial_particles
            .iter()
            .map(|particle| expected_hanna_from_cpu(&pbl, particle))
            .collect();

        assert_eq!(hanna_gpu.len(), hanna_cpu.len());
        for (gpu, cpu) in hanna_gpu.iter().zip(hanna_cpu.iter()) {
            assert_hanna_params_close(gpu, cpu);
        }

        let step = LangevinStep::legacy(45.0, 2.5e-4);
        let key = [0xDECA_FBAD, 0x1234_5678];
        let base_counter = [0xA5A5_0001, 0xFACE_B00C, 7, 9];

        let mut expected_particles = initial_particles.clone();
        let mut cpu_rng = PhiloxRng::new(key, base_counter);
        update_particles_turbulence_langevin_with_rng_cpu(
            &mut expected_particles,
            &hanna_cpu,
            step,
            &mut cpu_rng,
        )
        .expect("cpu langevin update succeeds");

        let next_counter = update_particles_turbulence_langevin_gpu(
            &ctx,
            &particle_buffers,
            &hanna_gpu,
            step,
            key,
            base_counter,
        )
        .expect("gpu langevin dispatch succeeds");
        let actual_particles = pollster::block_on(particle_buffers.download_particles(&ctx))
            .expect("particle readback succeeds");

        assert_eq!(actual_particles.len(), expected_particles.len());
        for (gpu, cpu) in actual_particles.iter().zip(expected_particles.iter()) {
            assert_particle_common_fields_equal(cpu, gpu);
            assert_relative_eq!(
                gpu.turb_u,
                cpu.turb_u,
                epsilon = 5.0e-5,
                max_relative = 5.0e-5
            );
            assert_relative_eq!(
                gpu.turb_v,
                cpu.turb_v,
                epsilon = 5.0e-5,
                max_relative = 5.0e-5
            );
            assert_relative_eq!(
                gpu.turb_w,
                cpu.turb_w,
                epsilon = 5.0e-5,
                max_relative = 5.0e-5
            );
        }
        assert_eq!(
            next_counter,
            philox_counter_add(base_counter, initial_particles.len() as u64)
        );

        // Deterministic reproducibility with identical seed/counter and inputs.
        let particle_buffers_second = ParticleBuffers::from_particles(&ctx, &initial_particles);
        let hanna_gpu_second = pollster::block_on(compute_hanna_params_gpu(
            &ctx,
            &particle_buffers_second,
            &pbl_buffers,
        ))
        .expect("second gpu hanna dispatch succeeds");
        let next_counter_second = update_particles_turbulence_langevin_gpu(
            &ctx,
            &particle_buffers_second,
            &hanna_gpu_second,
            step,
            key,
            base_counter,
        )
        .expect("second gpu langevin dispatch succeeds");
        let second_particles = pollster::block_on(particle_buffers_second.download_particles(&ctx))
            .expect("second particle readback succeeds");
        assert_eq!(next_counter_second, next_counter);
        for (first, second) in actual_particles.iter().zip(second_particles.iter()) {
            assert_eq!(first.turb_u.to_bits(), second.turb_u.to_bits());
            assert_eq!(first.turb_v.to_bits(), second.turb_v.to_bits());
            assert_eq!(first.turb_w.to_bits(), second.turb_w.to_bits());
        }

        let mut cbl_rng = PhiloxRng::new([0xBADC_0FFE, 0x1020_3040], [3, 5, 7, 11]);
        let cbl_sampling_inputs: Vec<CblSamplingInput> = (0..initial_particles.len())
            .map(|_| {
                let [u0, u1, u2, _u3] = cbl_rng.next_uniform4();
                CblSamplingInput::new(u0, u1, u2)
            })
            .collect();

        let cbl_gpu = pollster::block_on(sample_cbl_vertical_velocity_gpu(
            &ctx,
            &particle_buffers,
            &hanna_gpu,
            &cbl_sampling_inputs,
        ))
        .expect("gpu cbl dispatch succeeds");
        let cbl_cpu: Vec<CblSamplingOutput> = expected_particles
            .iter()
            .zip(hanna_cpu.iter().zip(cbl_sampling_inputs.iter()))
            .map(|(particle, (hanna, sample))| {
                expected_cbl_output_from_cpu(particle, hanna, *sample)
            })
            .collect();

        let mut unstable_cases_checked = 0usize;
        for (idx, (gpu, cpu)) in cbl_gpu.iter().zip(cbl_cpu.iter()).enumerate() {
            if !expected_particles[idx].is_active() {
                assert_eq!(gpu.sampled_w_m_s.to_bits(), 0.0f32.to_bits());
                continue;
            }
            if hanna_cpu[idx].ol < 0.0 {
                assert_cbl_output_close(gpu, cpu);
                unstable_cases_checked += 1;
            }
        }
        assert!(
            unstable_cases_checked > 0,
            "at least one unstable particle is required for CBL parity checks"
        );
    }

    #[test]
    fn gpu_langevin_validates_input_lengths() {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let particles = vec![make_particle(0.2, 0.0, 10.0), make_particle(1.0, 0.0, 20.0)];
        let particle_buffers = ParticleBuffers::from_particles(&ctx, &particles);
        let hanna = vec![compute_hanna_params(HannaInputs {
            ust: 0.3,
            wst: 0.0,
            ol: 80.0,
            h: 500.0,
            z: 10.0,
        })];

        let err = update_particles_turbulence_langevin_gpu(
            &ctx,
            &particle_buffers,
            &hanna,
            LangevinStep::legacy(10.0, 0.0),
            [1, 2],
            [0, 0, 0, 0],
        )
        .expect_err("input length mismatch should fail");
        assert!(matches!(
            err,
            GpuLangevinError::MismatchedInputLengths { .. }
        ));
    }

    #[test]
    fn gpu_langevin_empty_dispatch_keeps_counter() {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let particles: Vec<Particle> = Vec::new();
        let particle_buffers = ParticleBuffers::from_particles(&ctx, &particles);
        let base_counter = [11, 22, 33, 44];
        let next_counter = update_particles_turbulence_langevin_gpu(
            &ctx,
            &particle_buffers,
            &[],
            LangevinStep::legacy(10.0, 0.0),
            [7, 8],
            base_counter,
        )
        .expect("empty dispatch succeeds");
        assert_eq!(next_counter, base_counter);
    }
}
