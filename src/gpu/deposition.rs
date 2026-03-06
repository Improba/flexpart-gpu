//! GPU dry deposition probability dispatch (D-02).
//!
//! Ported from FLEXPART deposition probability update:
//! - `advance.f90`: `p = 1 - exp(-vdep * |dt| / (2*href))`
//! - `get_vdep_prob.f90`: near-surface deposition-layer logic
//!
//! This kernel computes per-particle dry-deposition probability and applies the
//! corresponding survival factor to all species masses:
//! `mass_new = mass_old * exp(-vdep * |dt| / (2*href))`.
//!
//! Buffer contract:
//! - binding 0: `array<Particle>` read-write storage buffer.
//! - binding 1: `array<f32>` deposition velocity per particle [m/s].
//! - binding 2: `array<f32>` deposition probability output per particle [-].
//! - binding 3: uniform params `(particle_count, dt_seconds, reference_height_m, pad)`.

use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::constants::HREF;
use crate::particles::{Particle, ParticleStore};

use super::{
    download_buffer_typed, render_shader_with_workgroup_size, runtime_workgroup_size,
    GpuBufferError, GpuContext, GpuError, ParticleBuffers, WorkgroupKernel,
};

const SHADER_TEMPLATE: &str = include_str!("../shaders/dry_deposition.wgsl");

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DryDepositionDispatchParamsRaw {
    particle_count: u32,
    dt_seconds: f32,
    reference_height_m: f32,
    _pad0: f32,
}

/// Dry deposition dispatch parameters.
#[derive(Debug, Clone, Copy)]
pub struct DryDepositionStepParams {
    /// Timestep duration [s]. Absolute value is used in the exponential term.
    pub dt_seconds: f32,
    /// Dry-deposition reference height `href` [m].
    pub reference_height_m: f32,
}

impl Default for DryDepositionStepParams {
    fn default() -> Self {
        Self {
            dt_seconds: 0.0,
            reference_height_m: HREF,
        }
    }
}

/// Reusable dry-deposition dispatch kernel objects.
pub struct DryDepositionDispatchKernel {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline: wgpu::ComputePipeline,
    workgroup_size_x: u32,
}

impl DryDepositionDispatchKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let workgroup_size_x = runtime_workgroup_size(ctx, WorkgroupKernel::DryDeposition);
        let shader_source = render_shader_with_workgroup_size(SHADER_TEMPLATE, workgroup_size_x);
        let shader = ctx.load_shader("dry_deposition_shader", &shader_source);
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("dry_deposition_bgl"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
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
            "dry_deposition_pipeline",
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

/// Typed IO buffers for dry deposition dispatch.
pub struct DryDepositionIoBuffers {
    pub deposition_velocity_m_s: wgpu::Buffer,
    pub deposition_probability: wgpu::Buffer,
    particle_count: usize,
}

impl DryDepositionIoBuffers {
    pub fn from_velocity(
        ctx: &GpuContext,
        deposition_velocity_m_s: &[f32],
    ) -> Result<Self, GpuDryDepositionError> {
        let particle_count = deposition_velocity_m_s.len();
        let velocity_buffer = if deposition_velocity_m_s.is_empty() {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dry_dep_velocity"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        } else {
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("dry_dep_velocity"),
                    contents: bytemuck::cast_slice(deposition_velocity_m_s),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                })
        };

        let probability_byte_len =
            checked_byte_len::<f32>(particle_count, "deposition_probability")?;
        let probability_size = if probability_byte_len == 0 {
            4
        } else {
            u64::try_from(probability_byte_len).map_err(|_| {
                GpuDryDepositionError::SizeOverflow {
                    field: "deposition_probability",
                }
            })?
        };
        let probability_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dry_dep_probability"),
            size: probability_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            deposition_velocity_m_s: velocity_buffer,
            deposition_probability: probability_buffer,
            particle_count,
        })
    }

    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.particle_count
    }

    pub fn upload_deposition_velocity(
        &self,
        ctx: &GpuContext,
        deposition_velocity_m_s: &[f32],
    ) -> Result<(), GpuDryDepositionError> {
        if deposition_velocity_m_s.len() != self.particle_count {
            return Err(GpuDryDepositionError::LengthMismatch {
                field: "deposition_velocity_m_s",
                expected: self.particle_count,
                actual: deposition_velocity_m_s.len(),
            });
        }
        if deposition_velocity_m_s.is_empty() {
            return Ok(());
        }
        ctx.queue.write_buffer(
            &self.deposition_velocity_m_s,
            0,
            bytemuck::cast_slice(deposition_velocity_m_s),
        );
        Ok(())
    }

    pub async fn download_probabilities(
        &self,
        ctx: &GpuContext,
    ) -> Result<Vec<f32>, GpuDryDepositionError> {
        if self.particle_count == 0 {
            return Ok(Vec::new());
        }
        download_buffer_typed::<f32>(
            ctx,
            &self.deposition_probability,
            self.particle_count,
            "deposition_probability",
        )
        .await
        .map_err(Into::into)
    }
}

/// Errors returned by GPU dry deposition dispatch.
#[derive(Debug, Error)]
pub enum GpuDryDepositionError {
    #[error("value for {field} does not fit in u32: {value}")]
    ValueTooLarge { field: &'static str, value: usize },
    #[error("byte-size overflow while preparing {field}")]
    SizeOverflow { field: &'static str },
    #[error("length mismatch for {field}: expected {expected}, got {actual}")]
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("invalid dt_seconds for dry deposition: {dt_seconds}")]
    InvalidTimeStep { dt_seconds: f32 },
    #[error("invalid reference_height_m for dry deposition: {reference_height_m}")]
    InvalidReferenceHeight { reference_height_m: f32 },
    #[error("buffer operation failed: {0}")]
    Buffer(#[from] GpuBufferError),
}

/// Errors for the higher-level dry-deposition particle workflow helper.
#[derive(Debug, Error)]
pub enum GpuDryDepositionWorkflowError {
    #[error("gpu initialization failed: {0}")]
    Gpu(#[from] GpuError),
    #[error("dry deposition dispatch failed: {0}")]
    Deposition(#[from] GpuDryDepositionError),
    #[error("particle buffer readback failed: {0}")]
    Buffer(#[from] GpuBufferError),
}

fn usize_to_u32(value: usize, field: &'static str) -> Result<u32, GpuDryDepositionError> {
    u32::try_from(value).map_err(|_| GpuDryDepositionError::ValueTooLarge { field, value })
}

fn checked_byte_len<T>(len: usize, field: &'static str) -> Result<usize, GpuDryDepositionError> {
    len.checked_mul(size_of::<T>())
        .ok_or(GpuDryDepositionError::SizeOverflow { field })
}

/// Dispatch dry deposition probability computation and in-place mass attenuation.
pub fn dispatch_dry_deposition_probability_gpu(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    io: &DryDepositionIoBuffers,
    params: DryDepositionStepParams,
) -> Result<(), GpuDryDepositionError> {
    let kernel = DryDepositionDispatchKernel::new(ctx);
    dispatch_dry_deposition_probability_gpu_with_kernel(ctx, particles, io, params, &kernel)
}

/// Dispatch dry deposition using a reusable prepared kernel.
pub fn dispatch_dry_deposition_probability_gpu_with_kernel(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    io: &DryDepositionIoBuffers,
    params: DryDepositionStepParams,
    kernel: &DryDepositionDispatchKernel,
) -> Result<(), GpuDryDepositionError> {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dry_deposition_encoder"),
        });
    encode_dry_deposition_probability_gpu_with_kernel(
        ctx,
        particles,
        io,
        params,
        kernel,
        &mut encoder,
    )?;
    ctx.queue.submit(Some(encoder.finish()));
    let _ = ctx.device.poll(wgpu::Maintain::Wait);
    Ok(())
}

/// Encode dry deposition dispatch into a caller-provided command encoder.
pub fn encode_dry_deposition_probability_gpu_with_kernel(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    io: &DryDepositionIoBuffers,
    params: DryDepositionStepParams,
    kernel: &DryDepositionDispatchKernel,
    encoder: &mut wgpu::CommandEncoder,
) -> Result<(), GpuDryDepositionError> {
    let particle_count = particles.particle_count();
    if particle_count == 0 {
        return Ok(());
    }
    if io.particle_count != particle_count {
        return Err(GpuDryDepositionError::LengthMismatch {
            field: "deposition_velocity_m_s",
            expected: particle_count,
            actual: io.particle_count,
        });
    }
    if !params.dt_seconds.is_finite() {
        return Err(GpuDryDepositionError::InvalidTimeStep {
            dt_seconds: params.dt_seconds,
        });
    }
    if !params.reference_height_m.is_finite() || params.reference_height_m <= 0.0 {
        return Err(GpuDryDepositionError::InvalidReferenceHeight {
            reference_height_m: params.reference_height_m,
        });
    }

    debug_assert_eq!(Particle::GPU_SIZE, 96);

    let raw_params = DryDepositionDispatchParamsRaw {
        particle_count: usize_to_u32(particle_count, "particle_count")?,
        dt_seconds: params.dt_seconds,
        reference_height_m: params.reference_height_m,
        _pad0: 0.0,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dry_deposition_params"),
            contents: bytemuck::bytes_of(&raw_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dry_deposition_bg"),
        layout: &kernel.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: particles.particle_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: io.deposition_velocity_m_s.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: io.deposition_probability.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dry_deposition_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&kernel.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        super::dispatch_1d(&mut cpass, raw_params.particle_count, kernel.workgroup_size_x);
    }
    Ok(())
}

/// Minimal callable dry-deposition API for the GPU particle update workflow.
///
/// This convenience helper:
/// 1. creates typed velocity/probability IO buffers,
/// 2. dispatches the WGSL dry-deposition kernel,
/// 3. returns per-particle deposition probability output.
///
/// Particle masses are attenuated in place on the GPU particle buffer.
pub async fn apply_dry_deposition_step_gpu(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    deposition_velocity_m_s: &[f32],
    params: DryDepositionStepParams,
) -> Result<Vec<f32>, GpuDryDepositionError> {
    if deposition_velocity_m_s.len() != particles.particle_count() {
        return Err(GpuDryDepositionError::LengthMismatch {
            field: "deposition_velocity_m_s",
            expected: particles.particle_count(),
            actual: deposition_velocity_m_s.len(),
        });
    }

    let io = DryDepositionIoBuffers::from_velocity(ctx, deposition_velocity_m_s)?;
    dispatch_dry_deposition_probability_gpu(ctx, particles, &io, params)?;
    io.download_probabilities(ctx).await
}

/// Workflow helper that updates a CPU particle store via GPU when available.
///
/// Returns:
/// - `Ok(Some(probabilities))` when a GPU adapter exists and the kernel runs.
/// - `Ok(None)` when no GPU adapter is available (graceful skip).
pub async fn apply_dry_deposition_step_workflow(
    particles: &mut ParticleStore,
    deposition_velocity_m_s: &[f32],
    params: DryDepositionStepParams,
) -> Result<Option<Vec<f32>>, GpuDryDepositionWorkflowError> {
    let ctx = match GpuContext::new().await {
        Ok(ctx) => ctx,
        Err(GpuError::NoAdapter) => return Ok(None),
        Err(err) => return Err(err.into()),
    };

    let gpu_particles = ParticleBuffers::from_store(&ctx, particles);
    let probabilities =
        apply_dry_deposition_step_gpu(&ctx, &gpu_particles, deposition_velocity_m_s, params)
            .await?;

    let updated_particles = gpu_particles.download_particles(&ctx).await?;
    particles.as_mut_slice().copy_from_slice(&updated_particles);
    particles.recount_active();

    Ok(Some(probabilities))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::gpu::GpuError;
    use crate::particles::{ParticleInit, MAX_SPECIES};
    use crate::physics::{dry_deposition_probability_step, in_dry_deposition_layer};

    fn particle_at(x: f32, y: f32, z: f32, mass0: f32) -> Particle {
        let cell_x = x.floor() as i32;
        let cell_y = y.floor() as i32;
        let mut mass = [0.0; MAX_SPECIES];
        mass[0] = mass0;
        mass[1] = 0.5 * mass0;
        Particle::new(&ParticleInit {
            cell_x,
            cell_y,
            pos_x: x - cell_x as f32,
            pos_y: y - cell_y as f32,
            pos_z: z,
            mass,
            release_point: 0,
            class: 0,
            time: 0,
        })
    }

    #[test]
    fn gpu_dry_deposition_probabilities_match_cpu_reference() {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let mut particles = vec![
            particle_at(1.2, 2.3, 2.0, 1.0),   // in layer
            particle_at(0.5, 0.25, 30.0, 2.0), // out of layer for href=15
            particle_at(3.1, 1.1, 0.2, 3.0),   // in layer
            particle_at(2.0, 4.0, 10.0, 4.0),  // in layer but inactive
        ];
        particles[3].deactivate();

        let deposition_velocity_m_s = vec![0.01, 0.05, 0.0, 0.2];
        let params = DryDepositionStepParams {
            dt_seconds: 60.0,
            reference_height_m: HREF,
        };

        let expected_prob: Vec<f32> = particles
            .iter()
            .zip(deposition_velocity_m_s.iter())
            .map(|(particle, vdep)| {
                if particle.is_active()
                    && in_dry_deposition_layer(particle, params.reference_height_m)
                {
                    dry_deposition_probability_step(
                        *vdep,
                        params.dt_seconds,
                        params.reference_height_m,
                    )
                } else {
                    0.0
                }
            })
            .collect();

        let expected_mass0: Vec<f32> = particles
            .iter()
            .zip(expected_prob.iter())
            .map(|(particle, prob)| particle.mass[0] * (1.0 - prob))
            .collect();

        let expected_mass1: Vec<f32> = particles
            .iter()
            .zip(expected_prob.iter())
            .map(|(particle, prob)| particle.mass[1] * (1.0 - prob))
            .collect();

        let particle_buffers = ParticleBuffers::from_particles(&ctx, &particles);
        let probabilities = pollster::block_on(apply_dry_deposition_step_gpu(
            &ctx,
            &particle_buffers,
            &deposition_velocity_m_s,
            params,
        ))
        .expect("gpu dry deposition succeeds");

        assert_eq!(probabilities.len(), expected_prob.len());
        for (gpu, cpu) in probabilities.iter().zip(expected_prob.iter()) {
            assert_relative_eq!(*gpu, *cpu, epsilon = 1.0e-6, max_relative = 1.0e-6);
        }

        let updated = pollster::block_on(particle_buffers.download_particles(&ctx))
            .expect("particle readback succeeds");
        for ((particle, m0), m1) in updated
            .iter()
            .zip(expected_mass0.iter())
            .zip(expected_mass1.iter())
        {
            assert_relative_eq!(
                particle.mass[0],
                *m0,
                epsilon = 1.0e-6,
                max_relative = 1.0e-6
            );
            assert_relative_eq!(
                particle.mass[1],
                *m1,
                epsilon = 1.0e-6,
                max_relative = 1.0e-6
            );
        }
    }

    #[test]
    fn workflow_api_gracefully_skips_without_adapter() {
        let mut store = ParticleStore::with_capacity(2);
        let p0 = particle_at(0.5, 0.5, 1.0, 1.0);
        let p1 = particle_at(1.5, 1.5, 2.0, 1.0);
        store.add(p0).expect("slot 0 available");
        store.add(p1).expect("slot 1 available");

        let initial_mass: Vec<f32> = store.as_slice().iter().map(|p| p.mass[0]).collect();
        let velocities = vec![0.01, 0.02];
        let result = pollster::block_on(apply_dry_deposition_step_workflow(
            &mut store,
            &velocities,
            DryDepositionStepParams {
                dt_seconds: 30.0,
                reference_height_m: HREF,
            },
        ))
        .expect("workflow api should not fail on missing adapter");

        match result {
            Some(probabilities) => {
                assert_eq!(probabilities.len(), 2);
            }
            None => {
                let mass_after_skip: Vec<f32> =
                    store.as_slice().iter().map(|p| p.mass[0]).collect();
                assert_eq!(mass_after_skip, initial_mass);
            }
        }
    }
}
