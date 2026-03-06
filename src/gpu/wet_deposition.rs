//! GPU wet deposition probability dispatch (D-04).
//!
//! Ported from FLEXPART wet-deposition mass-loss update in `wetdepo.f90`:
//! `p = grfraction * (1 - exp(-wetscav * |dt|))`.
//!
//! This kernel computes per-particle wet-deposition probability and applies the
//! corresponding survival factor to all species masses:
//! `mass_new = mass_old * (1 - p)`.
//!
//! MVP assumption:
//! - D-03 CPU logic supplies per-particle `wetscav` and `grfraction` inputs.
//! - This GPU stage applies only the shared wetdepo mass-loss contract, which
//!   keeps parity for probability/mass attenuation while deferring full
//!   below-cloud/in-cloud coefficient branching to upstream CPU code.
//!
//! Buffer contract:
//! - binding 0: `array<Particle>` read-write storage buffer.
//! - binding 1: `array<f32>` scavenging coefficient (`wetscav`) per particle [1/s].
//! - binding 2: `array<f32>` precipitating fraction (`grfraction`) per particle [-].
//! - binding 3: `array<f32>` wet-deposition probability output per particle [-].
//! - binding 4: uniform params `(particle_count, dt_seconds, pad0, pad1)`.

use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::particles::ParticleStore;

use super::{
    download_buffer_typed, render_shader_with_workgroup_size, runtime_workgroup_size,
    GpuBufferError, GpuContext, GpuError, ParticleBuffers, WorkgroupKernel,
};

const SHADER_TEMPLATE: &str = include_str!("../shaders/wet_deposition.wgsl");

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct WetDepositionDispatchParamsRaw {
    particle_count: u32,
    dt_seconds: f32,
    _pad0: f32,
    _pad1: f32,
}

/// Wet deposition dispatch parameters.
#[derive(Debug, Clone, Copy)]
pub struct WetDepositionStepParams {
    /// Timestep duration [s]. Absolute value is used in the exponential term.
    pub dt_seconds: f32,
}

impl Default for WetDepositionStepParams {
    fn default() -> Self {
        Self { dt_seconds: 0.0 }
    }
}

/// Reusable wet-deposition dispatch kernel objects.
pub struct WetDepositionDispatchKernel {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline: wgpu::ComputePipeline,
    workgroup_size_x: u32,
}

impl WetDepositionDispatchKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let workgroup_size_x = runtime_workgroup_size(ctx, WorkgroupKernel::WetDeposition);
        let shader_source = render_shader_with_workgroup_size(SHADER_TEMPLATE, workgroup_size_x);
        let shader = ctx.load_shader("wet_deposition_shader", &shader_source);
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("wet_deposition_bgl"),
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
        let pipeline = ctx.create_compute_pipeline(
            "wet_deposition_pipeline",
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

/// Typed IO buffers for wet deposition dispatch.
pub struct WetDepositionIoBuffers {
    pub scavenging_coefficient_s_inv: wgpu::Buffer,
    pub precipitating_fraction: wgpu::Buffer,
    pub wet_deposition_probability: wgpu::Buffer,
    particle_count: usize,
}

impl WetDepositionIoBuffers {
    pub fn from_inputs(
        ctx: &GpuContext,
        scavenging_coefficient_s_inv: &[f32],
        precipitating_fraction: &[f32],
    ) -> Result<Self, GpuWetDepositionError> {
        if scavenging_coefficient_s_inv.len() != precipitating_fraction.len() {
            return Err(GpuWetDepositionError::LengthMismatch {
                field: "precipitating_fraction",
                expected: scavenging_coefficient_s_inv.len(),
                actual: precipitating_fraction.len(),
            });
        }

        let particle_count = scavenging_coefficient_s_inv.len();
        let scavenging_buffer = create_storage_input_buffer(
            &ctx.device,
            "wet_dep_scavenging_coefficient_s_inv",
            scavenging_coefficient_s_inv,
        );
        let precipitating_fraction_buffer = create_storage_input_buffer(
            &ctx.device,
            "wet_dep_precipitating_fraction",
            precipitating_fraction,
        );

        let probability_byte_len =
            checked_byte_len::<f32>(particle_count, "wet_deposition_probability")?;
        let probability_size = if probability_byte_len == 0 {
            4
        } else {
            u64::try_from(probability_byte_len).map_err(|_| {
                GpuWetDepositionError::SizeOverflow {
                    field: "wet_deposition_probability",
                }
            })?
        };
        let probability_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wet_dep_probability"),
            size: probability_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            scavenging_coefficient_s_inv: scavenging_buffer,
            precipitating_fraction: precipitating_fraction_buffer,
            wet_deposition_probability: probability_buffer,
            particle_count,
        })
    }

    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.particle_count
    }

    pub fn upload_scavenging_coefficient(
        &self,
        ctx: &GpuContext,
        scavenging_coefficient_s_inv: &[f32],
    ) -> Result<(), GpuWetDepositionError> {
        if scavenging_coefficient_s_inv.len() != self.particle_count {
            return Err(GpuWetDepositionError::LengthMismatch {
                field: "scavenging_coefficient_s_inv",
                expected: self.particle_count,
                actual: scavenging_coefficient_s_inv.len(),
            });
        }
        if scavenging_coefficient_s_inv.is_empty() {
            return Ok(());
        }
        ctx.queue.write_buffer(
            &self.scavenging_coefficient_s_inv,
            0,
            bytemuck::cast_slice(scavenging_coefficient_s_inv),
        );
        Ok(())
    }

    pub fn upload_precipitating_fraction(
        &self,
        ctx: &GpuContext,
        precipitating_fraction: &[f32],
    ) -> Result<(), GpuWetDepositionError> {
        if precipitating_fraction.len() != self.particle_count {
            return Err(GpuWetDepositionError::LengthMismatch {
                field: "precipitating_fraction",
                expected: self.particle_count,
                actual: precipitating_fraction.len(),
            });
        }
        if precipitating_fraction.is_empty() {
            return Ok(());
        }
        ctx.queue.write_buffer(
            &self.precipitating_fraction,
            0,
            bytemuck::cast_slice(precipitating_fraction),
        );
        Ok(())
    }

    pub async fn download_probabilities(
        &self,
        ctx: &GpuContext,
    ) -> Result<Vec<f32>, GpuWetDepositionError> {
        if self.particle_count == 0 {
            return Ok(Vec::new());
        }
        download_buffer_typed::<f32>(
            ctx,
            &self.wet_deposition_probability,
            self.particle_count,
            "wet_deposition_probability",
        )
        .await
        .map_err(Into::into)
    }
}

fn create_storage_input_buffer(device: &wgpu::Device, label: &str, data: &[f32]) -> wgpu::Buffer {
    if data.is_empty() {
        return device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

/// Errors returned by GPU wet deposition dispatch.
#[derive(Debug, Error)]
pub enum GpuWetDepositionError {
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
    #[error("invalid dt_seconds for wet deposition: {dt_seconds}")]
    InvalidTimeStep { dt_seconds: f32 },
    #[error("buffer operation failed: {0}")]
    Buffer(#[from] GpuBufferError),
}

/// Errors for the higher-level wet-deposition particle workflow helper.
#[derive(Debug, Error)]
pub enum GpuWetDepositionWorkflowError {
    #[error("gpu initialization failed: {0}")]
    Gpu(#[from] GpuError),
    #[error("wet deposition dispatch failed: {0}")]
    Deposition(#[from] GpuWetDepositionError),
    #[error("particle buffer readback failed: {0}")]
    Buffer(#[from] GpuBufferError),
}

fn usize_to_u32(value: usize, field: &'static str) -> Result<u32, GpuWetDepositionError> {
    u32::try_from(value).map_err(|_| GpuWetDepositionError::ValueTooLarge { field, value })
}

fn checked_byte_len<T>(len: usize, field: &'static str) -> Result<usize, GpuWetDepositionError> {
    len.checked_mul(size_of::<T>())
        .ok_or(GpuWetDepositionError::SizeOverflow { field })
}

/// Dispatch wet deposition probability computation and in-place mass attenuation.
pub fn dispatch_wet_deposition_probability_gpu(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    io: &WetDepositionIoBuffers,
    params: WetDepositionStepParams,
) -> Result<(), GpuWetDepositionError> {
    let kernel = WetDepositionDispatchKernel::new(ctx);
    dispatch_wet_deposition_probability_gpu_with_kernel(ctx, particles, io, params, &kernel)
}

/// Dispatch wet deposition using a reusable prepared kernel.
pub fn dispatch_wet_deposition_probability_gpu_with_kernel(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    io: &WetDepositionIoBuffers,
    params: WetDepositionStepParams,
    kernel: &WetDepositionDispatchKernel,
) -> Result<(), GpuWetDepositionError> {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wet_deposition_encoder"),
        });
    encode_wet_deposition_probability_gpu_with_kernel(
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

/// Encode wet deposition dispatch into a caller-provided command encoder.
pub fn encode_wet_deposition_probability_gpu_with_kernel(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    io: &WetDepositionIoBuffers,
    params: WetDepositionStepParams,
    kernel: &WetDepositionDispatchKernel,
    encoder: &mut wgpu::CommandEncoder,
) -> Result<(), GpuWetDepositionError> {
    let particle_count = particles.particle_count();
    if particle_count == 0 {
        return Ok(());
    }
    if io.particle_count != particle_count {
        return Err(GpuWetDepositionError::LengthMismatch {
            field: "wet_deposition_io_buffers",
            expected: particle_count,
            actual: io.particle_count,
        });
    }
    if !params.dt_seconds.is_finite() {
        return Err(GpuWetDepositionError::InvalidTimeStep {
            dt_seconds: params.dt_seconds,
        });
    }

    let raw_params = WetDepositionDispatchParamsRaw {
        particle_count: usize_to_u32(particle_count, "particle_count")?,
        dt_seconds: params.dt_seconds,
        _pad0: 0.0,
        _pad1: 0.0,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wet_deposition_params"),
            contents: bytemuck::bytes_of(&raw_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wet_deposition_bg"),
        layout: &kernel.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: particles.particle_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: io.scavenging_coefficient_s_inv.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: io.precipitating_fraction.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: io.wet_deposition_probability.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wet_deposition_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&kernel.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        super::dispatch_1d(&mut cpass, raw_params.particle_count, kernel.workgroup_size_x);
    }
    Ok(())
}

/// Minimal callable wet-deposition API for the GPU particle update workflow.
///
/// This convenience helper:
/// 1. creates typed scavenging/fraction/probability IO buffers,
/// 2. dispatches the WGSL wet-deposition kernel,
/// 3. returns per-particle wet-deposition probability output.
///
/// Particle masses are attenuated in place on the GPU particle buffer.
pub async fn apply_wet_deposition_step_gpu(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    scavenging_coefficient_s_inv: &[f32],
    precipitating_fraction: &[f32],
    params: WetDepositionStepParams,
) -> Result<Vec<f32>, GpuWetDepositionError> {
    if scavenging_coefficient_s_inv.len() != particles.particle_count() {
        return Err(GpuWetDepositionError::LengthMismatch {
            field: "scavenging_coefficient_s_inv",
            expected: particles.particle_count(),
            actual: scavenging_coefficient_s_inv.len(),
        });
    }
    if precipitating_fraction.len() != particles.particle_count() {
        return Err(GpuWetDepositionError::LengthMismatch {
            field: "precipitating_fraction",
            expected: particles.particle_count(),
            actual: precipitating_fraction.len(),
        });
    }

    let io = WetDepositionIoBuffers::from_inputs(
        ctx,
        scavenging_coefficient_s_inv,
        precipitating_fraction,
    )?;
    dispatch_wet_deposition_probability_gpu(ctx, particles, &io, params)?;
    io.download_probabilities(ctx).await
}

/// Workflow helper that updates a CPU particle store via GPU when available.
///
/// Returns:
/// - `Ok(Some(probabilities))` when a GPU adapter exists and the kernel runs.
/// - `Ok(None)` when no GPU adapter is available (graceful skip).
pub async fn apply_wet_deposition_step_workflow(
    particles: &mut ParticleStore,
    scavenging_coefficient_s_inv: &[f32],
    precipitating_fraction: &[f32],
    params: WetDepositionStepParams,
) -> Result<Option<Vec<f32>>, GpuWetDepositionWorkflowError> {
    let ctx = match GpuContext::new().await {
        Ok(ctx) => ctx,
        Err(GpuError::NoAdapter) => return Ok(None),
        Err(err) => return Err(err.into()),
    };

    let gpu_particles = ParticleBuffers::from_store(&ctx, particles);
    let probabilities = apply_wet_deposition_step_gpu(
        &ctx,
        &gpu_particles,
        scavenging_coefficient_s_inv,
        precipitating_fraction,
        params,
    )
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
    use crate::particles::{Particle, ParticleInit, MAX_SPECIES};
    use crate::physics::{wet_scavenging_probability_step, WetScavengingStep};

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
    fn gpu_wet_deposition_probabilities_match_cpu_reference() {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let mut particles = vec![
            particle_at(1.2, 2.3, 2.0, 1.0),
            particle_at(0.5, 0.25, 30.0, 2.0),
            particle_at(3.1, 1.1, 0.2, 3.0),
            particle_at(2.0, 4.0, 10.0, 4.0),
            particle_at(2.2, 5.4, 100.0, 5.0),
        ];
        particles[3].deactivate();

        let scavenging_coefficient_s_inv = vec![0.02, 0.0, 0.15, 0.03, -1.0];
        let precipitating_fraction = vec![0.5, 1.0, 1.2, 0.7, 0.9];
        let params = WetDepositionStepParams { dt_seconds: 60.0 };

        let expected_prob: Vec<f32> = particles
            .iter()
            .zip(
                scavenging_coefficient_s_inv
                    .iter()
                    .zip(precipitating_fraction.iter()),
            )
            .map(|(particle, (&lambda, &fraction))| {
                if particle.is_active() {
                    wet_scavenging_probability_step(WetScavengingStep {
                        scavenging_coefficient_s_inv: lambda,
                        dt_seconds: params.dt_seconds,
                        precipitating_fraction: fraction,
                    })
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
        let probabilities = pollster::block_on(apply_wet_deposition_step_gpu(
            &ctx,
            &particle_buffers,
            &scavenging_coefficient_s_inv,
            &precipitating_fraction,
            params,
        ))
        .expect("gpu wet deposition succeeds");

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
        let scavenging = vec![0.01, 0.02];
        let precipitating_fraction = vec![0.3, 0.7];
        let result = pollster::block_on(apply_wet_deposition_step_workflow(
            &mut store,
            &scavenging,
            &precipitating_fraction,
            WetDepositionStepParams { dt_seconds: 30.0 },
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
