//! GPU CBL vertical-velocity sampling dispatch (H-06).
//!
//! Dispatches a WGSL kernel that reconstructs the CBL bi-Gaussian vertical
//! velocity PDF and samples one turbulent vertical perturbation `w'` per
//! particle slot.
//!
//! Ported from FLEXPART `cbl.f90` through the CPU reference in
//! `src/physics/cbl.rs` (H-05) and designed to consume:
//! - current particle storage buffer (`ParticleBuffers`)
//! - per-particle Hanna parameter buffer (`HannaParams`, from H-02/H-01)
//! - explicit deterministic sampling uniforms (`CblSamplingInput`)
//!
//! MVP assumptions:
//! - this kernel computes CBL PDF/sampling outputs but does not yet mutate
//!   particle turbulence memory (`Particle::turb_w`) in-place;
//! - branch-constrained reinjection (`CblBranch`) is left to higher-level logic.

use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::particles::ParticleStore;
use crate::pbl::HannaParams;

use super::{download_buffer_typed, GpuBufferError, GpuContext, GpuError, ParticleBuffers};

const WORKGROUP_SIZE_X: u32 = 64;
const SHADER_SOURCE: &str = include_str!("../shaders/cbl.wgsl");

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CblDispatchParams {
    particle_count: u32,
    _pad0: [u32; 3],
}

/// Deterministic uniforms used by one CBL sampling operation.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct CblSamplingInput {
    /// Branch selector in `[0, 1)` for updraft/downdraft mixture sampling.
    pub branch_uniform: f32,
    /// Box-Muller first uniform in `[0, 1)`.
    pub gaussian_uniform0: f32,
    /// Box-Muller second uniform in `[0, 1)`.
    pub gaussian_uniform1: f32,
    /// Padding for 16-byte alignment.
    pub _pad0: f32,
}

impl CblSamplingInput {
    #[must_use]
    pub fn new(branch_uniform: f32, gaussian_uniform0: f32, gaussian_uniform1: f32) -> Self {
        Self {
            branch_uniform,
            gaussian_uniform0,
            gaussian_uniform1,
            _pad0: 0.0,
        }
    }
}

/// Per-particle CBL PDF diagnostics plus sampled vertical velocity `w'`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct CblSamplingOutput {
    /// One sampled turbulent vertical perturbation `w'` [m/s].
    pub sampled_w_m_s: f32,
    /// Updraft mixture weight.
    pub updraft_weight: f32,
    /// Updraft Gaussian mean [m/s].
    pub updraft_mean_m_s: f32,
    /// Updraft Gaussian sigma [m/s].
    pub updraft_sigma_m_s: f32,
    /// Downdraft mixture weight.
    pub downdraft_weight: f32,
    /// Downdraft Gaussian mean [m/s].
    pub downdraft_mean_m_s: f32,
    /// Downdraft Gaussian sigma [m/s].
    pub downdraft_sigma_m_s: f32,
    /// Target second moment E[w'^2].
    pub second_moment_w2: f32,
    /// Target third moment E[w'^3].
    pub third_moment_w3: f32,
    /// Target skewness.
    pub skewness: f32,
    /// CBL transition factor in `[0, 1]`.
    pub transition: f32,
    /// Relative height z/h in `[0, 1]`.
    pub z_over_h: f32,
}

/// Typed GPU IO buffers used by the CBL sampling kernel.
pub struct CblSamplingIoBuffers {
    sampling_inputs: wgpu::Buffer,
    outputs: wgpu::Buffer,
    particle_count: usize,
}

impl CblSamplingIoBuffers {
    fn checked_byte_len<T>(len: usize, field: &'static str) -> Result<u64, GpuCblError> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or(GpuCblError::SizeOverflow { field })?;
        u64::try_from(bytes).map_err(|_| GpuCblError::SizeOverflow { field })
    }

    fn from_inputs(
        ctx: &GpuContext,
        sampling_inputs: &[CblSamplingInput],
    ) -> Result<Self, GpuCblError> {
        let output_size = Self::checked_byte_len::<CblSamplingOutput>(
            sampling_inputs.len(),
            "cbl_sampling_outputs",
        )?;
        let outputs = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cbl_sampling_outputs"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let sampling_inputs_buffer =
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("cbl_sampling_inputs"),
                    contents: bytemuck::cast_slice(sampling_inputs),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        Ok(Self {
            sampling_inputs: sampling_inputs_buffer,
            outputs,
            particle_count: sampling_inputs.len(),
        })
    }

    async fn download_outputs(
        &self,
        ctx: &GpuContext,
    ) -> Result<Vec<CblSamplingOutput>, GpuCblError> {
        download_buffer_typed::<CblSamplingOutput>(
            ctx,
            &self.outputs,
            self.particle_count,
            "cbl_sampling_outputs",
        )
        .await
        .map_err(Into::into)
    }
}

/// Errors returned while running the GPU CBL kernel.
#[derive(Debug, Error)]
pub enum GpuCblError {
    #[error(
        "mismatched input lengths: particle_slots={particle_slots}, hanna_params={hanna_params}, sampling_inputs={sampling_inputs}"
    )]
    MismatchedInputLengths {
        particle_slots: usize,
        hanna_params: usize,
        sampling_inputs: usize,
    },
    #[error("value for {field} does not fit in u32: {value}")]
    ValueTooLarge { field: &'static str, value: usize },
    #[error("byte-size overflow while preparing {field}")]
    SizeOverflow { field: &'static str },
    #[error("buffer operation failed: {0}")]
    Buffer(#[from] GpuBufferError),
}

/// Errors returned by the workflow helper that creates a GPU context internally.
#[derive(Debug, Error)]
pub enum GpuCblWorkflowError {
    #[error("gpu initialization failed: {0}")]
    Gpu(#[from] GpuError),
    #[error("cbl dispatch failed: {0}")]
    Cbl(#[from] GpuCblError),
}

fn usize_to_u32(value: usize, field: &'static str) -> Result<u32, GpuCblError> {
    u32::try_from(value).map_err(|_| GpuCblError::ValueTooLarge { field, value })
}

/// Dispatch the WGSL CBL kernel and return one typed output per particle slot.
///
/// Inactive particles (`flags & 1 == 0`) return zeroed [`CblSamplingOutput`].
pub async fn sample_cbl_vertical_velocity_gpu(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    hanna_params: &[HannaParams],
    sampling_inputs: &[CblSamplingInput],
) -> Result<Vec<CblSamplingOutput>, GpuCblError> {
    let particle_count = particles.particle_count();
    if particle_count != hanna_params.len() || particle_count != sampling_inputs.len() {
        return Err(GpuCblError::MismatchedInputLengths {
            particle_slots: particle_count,
            hanna_params: hanna_params.len(),
            sampling_inputs: sampling_inputs.len(),
        });
    }
    if particle_count == 0 {
        return Ok(Vec::new());
    }

    let params = CblDispatchParams {
        particle_count: usize_to_u32(particle_count, "particle_count")?,
        _pad0: [0; 3],
    };

    let io = CblSamplingIoBuffers::from_inputs(ctx, sampling_inputs)?;
    let hanna_params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cbl_hanna_params"),
            contents: bytemuck::cast_slice(hanna_params),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cbl_dispatch_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader = ctx.load_shader("cbl_shader", SHADER_SOURCE);
    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cbl_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
    let pipeline =
        ctx.create_compute_pipeline("cbl_pipeline", &shader, "main", &[&bind_group_layout]);

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cbl_bg"),
        layout: &bind_group_layout,
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
                resource: io.sampling_inputs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: io.outputs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cbl_encoder"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cbl_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        super::dispatch_1d(&mut cpass, params.particle_count, WORKGROUP_SIZE_X);
    }
    ctx.queue.submit(Some(encoder.finish()));

    io.download_outputs(ctx).await
}

/// Workflow helper that runs CBL sampling on GPU when available.
///
/// Returns:
/// - `Ok(Some(outputs))` when a GPU adapter exists and dispatch succeeds.
/// - `Ok(None)` when no GPU adapter is available (graceful skip).
pub async fn sample_cbl_vertical_velocity_workflow(
    particles: &ParticleStore,
    hanna_params: &[HannaParams],
    sampling_inputs: &[CblSamplingInput],
) -> Result<Option<Vec<CblSamplingOutput>>, GpuCblWorkflowError> {
    let ctx = match GpuContext::new().await {
        Ok(ctx) => ctx,
        Err(GpuError::NoAdapter) => return Ok(None),
        Err(err) => return Err(err.into()),
    };

    let gpu_particles = ParticleBuffers::from_store(&ctx, particles);
    let outputs =
        sample_cbl_vertical_velocity_gpu(&ctx, &gpu_particles, hanna_params, sampling_inputs)
            .await?;
    Ok(Some(outputs))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::gpu::GpuError;
    use crate::particles::{Particle, ParticleInit, ParticleStore, MAX_SPECIES};
    use crate::physics::{
        compute_cbl_bigaussian_pdf, compute_hanna_params, sample_cbl_vertical_velocity,
        CblPdfInputs, HannaInputs,
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

    fn expected_from_cpu(
        particle: &Particle,
        hanna: &HannaParams,
        sampling_input: CblSamplingInput,
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
            sampling_input.branch_uniform,
            sampling_input.gaussian_uniform0,
            sampling_input.gaussian_uniform1,
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

    fn assert_output_close(actual: &CblSamplingOutput, expected: &CblSamplingOutput) {
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
    fn gpu_cbl_sampling_matches_cpu_reference() {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let mut particles = vec![
            make_particle(0.2, 0.0, 120.0),
            make_particle(1.2, 0.0, 80.0),
            make_particle(2.0, 0.0, 200.0),
            make_particle(3.0, 0.0, 30.0),
        ];
        particles[3].deactivate();

        let hanna_params = vec![
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
        let sampling_inputs = vec![
            CblSamplingInput::new(0.1, 0.4, 0.7),
            CblSamplingInput::new(0.8, 0.9, 0.3),
            CblSamplingInput::new(0.2, 0.6, 0.5),
            CblSamplingInput::new(0.4, 0.1, 0.2),
        ];

        let expected: Vec<CblSamplingOutput> = particles
            .iter()
            .zip(hanna_params.iter().zip(sampling_inputs.iter()))
            .map(|(particle, (hanna, sample))| expected_from_cpu(particle, hanna, *sample))
            .collect();

        let particle_buffers = ParticleBuffers::from_particles(&ctx, &particles);
        let actual = pollster::block_on(sample_cbl_vertical_velocity_gpu(
            &ctx,
            &particle_buffers,
            &hanna_params,
            &sampling_inputs,
        ))
        .expect("gpu cbl sampling succeeds");

        assert_eq!(actual.len(), expected.len());
        for (gpu, cpu) in actual.iter().zip(expected.iter()) {
            assert_output_close(gpu, cpu);
        }
    }

    #[test]
    fn workflow_api_gracefully_skips_without_adapter() {
        let mut store = ParticleStore::with_capacity(2);
        store
            .add(make_particle(0.5, 0.5, 100.0))
            .expect("slot 0 available");
        store
            .add(make_particle(1.5, 0.5, 80.0))
            .expect("slot 1 available");

        let hanna_params = vec![
            compute_hanna_params(HannaInputs {
                ust: 0.35,
                wst: 1.5,
                ol: -80.0,
                h: 1000.0,
                z: 100.0,
            }),
            compute_hanna_params(HannaInputs {
                ust: 0.25,
                wst: 0.0,
                ol: 150.0,
                h: 900.0,
                z: 80.0,
            }),
        ];
        let sampling_inputs = vec![
            CblSamplingInput::new(0.3, 0.2, 0.9),
            CblSamplingInput::new(0.7, 0.5, 0.4),
        ];

        let result = pollster::block_on(sample_cbl_vertical_velocity_workflow(
            &store,
            &hanna_params,
            &sampling_inputs,
        ))
        .expect("workflow should gracefully handle missing adapter");

        if let Some(outputs) = result {
            assert_eq!(outputs.len(), store.as_slice().len());
        }
    }
}
