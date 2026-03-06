//! GPU Hanna turbulence parameter dispatch (H-02).
//!
//! Dispatches a WGSL kernel that computes per-particle Hanna (1982) turbulence
//! parameters using local PBL fields:
//! - `sigma_u`, `sigma_v`, `sigma_w`
//! - `d(sigma_w)/dz`, `d(sigma_w^2)/dz`
//! - `TL_u`, `TL_v`, `TL_w`
//!
//! The branch logic and formulas mirror `src/physics/hanna.rs` (H-01), itself
//! ported from FLEXPART `hanna.f90`.

use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::pbl::HannaParams;

use super::{
    download_buffer_typed, render_shader_with_workgroup_size, runtime_workgroup_size,
    GpuBufferError, GpuContext, ParticleBuffers, PblBuffers, WorkgroupKernel,
};

const SHADER_TEMPLATE: &str = include_str!("../shaders/hanna_params.wgsl");

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct HannaDispatchParams {
    pbl_nx: u32,
    pbl_ny: u32,
    particle_count: u32,
    _pad0: u32,
}

/// Errors returned while running the GPU Hanna kernel.
#[derive(Debug, Error)]
pub enum GpuHannaError {
    #[error("PBL grid dimensions must be non-zero, got {shape:?}")]
    ZeroPblShape { shape: (usize, usize) },
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
    #[error("buffer operation failed: {0}")]
    Buffer(#[from] GpuBufferError),
}

fn usize_to_u32(value: usize, field: &'static str) -> Result<u32, GpuHannaError> {
    u32::try_from(value).map_err(|_| GpuHannaError::ValueTooLarge { field, value })
}

fn checked_byte_len<T>(len: usize, field: &'static str) -> Result<usize, GpuHannaError> {
    len.checked_mul(size_of::<T>())
        .ok_or(GpuHannaError::SizeOverflow { field })
}

/// Reusable GPU output buffer holding one [`HannaParams`] entry per particle slot.
pub struct HannaParamsOutputBuffer {
    pub buffer: wgpu::Buffer,
    particle_count: usize,
}

impl HannaParamsOutputBuffer {
    pub fn new(ctx: &GpuContext, particle_count: usize) -> Result<Self, GpuHannaError> {
        let output_bytes = checked_byte_len::<HannaParams>(particle_count, "hanna_params_outputs")?;
        let output_size = if output_bytes == 0 {
            4
        } else {
            u64::try_from(output_bytes).map_err(|_| GpuHannaError::SizeOverflow {
                field: "hanna_params_outputs",
            })?
        };
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hanna_params_outputs"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            buffer,
            particle_count,
        })
    }

    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.particle_count
    }
}

/// Reusable Hanna dispatch kernel objects (bind group layout + compute pipeline).
pub struct HannaDispatchKernel {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline: wgpu::ComputePipeline,
    workgroup_size_x: u32,
}

impl HannaDispatchKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let workgroup_size_x = runtime_workgroup_size(ctx, WorkgroupKernel::HannaParams);
        let shader_source = render_shader_with_workgroup_size(SHADER_TEMPLATE, workgroup_size_x);
        let shader = ctx.load_shader("hanna_params_shader", &shader_source);
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("hanna_params_bgl"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
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
            "hanna_params_pipeline",
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

/// Compute per-particle Hanna turbulence parameters on the GPU.
///
/// Input contract:
/// - particle horizontal position comes from `Particle` (`cell + frac`)
/// - particle vertical coordinate uses `Particle::pos_z`
/// - PBL fields come from [`PblBuffers`] flattened as row-major `(nx, ny)`
///
/// Inactive particles (`flags & 1 == 0`) return zeroed [`HannaParams`].
pub async fn compute_hanna_params_gpu(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    pbl: &PblBuffers,
) -> Result<Vec<HannaParams>, GpuHannaError> {
    let particle_count = particles.particle_count();
    if particle_count == 0 {
        return Ok(Vec::new());
    }

    let output = HannaParamsOutputBuffer::new(ctx, particle_count)?;
    dispatch_hanna_params_gpu(ctx, particles, pbl, &output)?;
    download_buffer_typed::<HannaParams>(
        ctx,
        &output.buffer,
        particle_count,
        "hanna_params_outputs",
    )
    .await
    .map_err(Into::into)
}

/// Dispatch Hanna parameter computation into a caller-provided GPU output buffer.
///
/// This API is intended for fully GPU-resident workflows where downstream
/// kernels consume Hanna outputs directly without host readback.
pub fn dispatch_hanna_params_gpu(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    pbl: &PblBuffers,
    output: &HannaParamsOutputBuffer,
) -> Result<(), GpuHannaError> {
    let kernel = HannaDispatchKernel::new(ctx);
    dispatch_hanna_params_gpu_with_kernel(ctx, particles, pbl, output, &kernel)
}

/// Dispatch Hanna parameters using a reusable prepared kernel.
pub fn dispatch_hanna_params_gpu_with_kernel(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    pbl: &PblBuffers,
    output: &HannaParamsOutputBuffer,
    kernel: &HannaDispatchKernel,
) -> Result<(), GpuHannaError> {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hanna_params_encoder"),
        });
    encode_hanna_params_gpu_with_kernel(ctx, particles, pbl, output, kernel, &mut encoder)?;
    ctx.queue.submit(Some(encoder.finish()));
    Ok(())
}

/// Encode Hanna parameter dispatch into a caller-provided command encoder.
pub fn encode_hanna_params_gpu_with_kernel(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    pbl: &PblBuffers,
    output: &HannaParamsOutputBuffer,
    kernel: &HannaDispatchKernel,
    encoder: &mut wgpu::CommandEncoder,
) -> Result<(), GpuHannaError> {
    let particle_count = particles.particle_count();
    if particle_count == 0 {
        return Ok(());
    }
    if output.particle_count != particle_count {
        return Err(GpuHannaError::LengthMismatch {
            field: "hanna_params_outputs",
            expected: particle_count,
            actual: output.particle_count,
        });
    }

    let (pbl_nx, pbl_ny) = pbl.shape;
    if pbl_nx == 0 || pbl_ny == 0 {
        return Err(GpuHannaError::ZeroPblShape { shape: pbl.shape });
    }

    let params = HannaDispatchParams {
        pbl_nx: usize_to_u32(pbl_nx, "pbl_nx")?,
        pbl_ny: usize_to_u32(pbl_ny, "pbl_ny")?,
        particle_count: usize_to_u32(particle_count, "particle_count")?,
        _pad0: 0,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hanna_params_dispatch_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hanna_params_bg"),
        layout: &kernel.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: particles.particle_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pbl.ustar.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: pbl.wstar.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: pbl.hmix.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: pbl.oli.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: output.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hanna_params_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&kernel.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        super::dispatch_1d(&mut cpass, params.particle_count, kernel.workgroup_size_x);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use bytemuck::Zeroable;

    use super::*;
    use crate::gpu::GpuError;
    use crate::particles::{Particle, ParticleInit, ParticleStore, MAX_SPECIES};
    use crate::pbl::PblState;
    use crate::physics::{compute_hanna_params, obukhov_length_from_inverse, HannaInputs};

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

    fn expected_from_cpu(pbl: &PblState, particle: &Particle) -> HannaParams {
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

    fn assert_scalar_close(actual: f32, expected: f32) {
        if expected.is_infinite() {
            assert!(actual.is_infinite());
            assert_eq!(actual.is_sign_positive(), expected.is_sign_positive());
        } else {
            assert_relative_eq!(actual, expected, epsilon = 5.0e-5, max_relative = 5.0e-5);
        }
    }

    fn assert_hanna_close(actual: &HannaParams, expected: &HannaParams) {
        assert_scalar_close(actual.ust, expected.ust);
        assert_scalar_close(actual.wst, expected.wst);
        assert_scalar_close(actual.ol, expected.ol);
        assert_scalar_close(actual.h, expected.h);
        assert_scalar_close(actual.zeta, expected.zeta);
        assert_scalar_close(actual.sigu, expected.sigu);
        assert_scalar_close(actual.sigv, expected.sigv);
        assert_scalar_close(actual.sigw, expected.sigw);
        assert_scalar_close(actual.dsigwdz, expected.dsigwdz);
        assert_scalar_close(actual.dsigw2dz, expected.dsigw2dz);
        assert_scalar_close(actual.tlu, expected.tlu);
        assert_scalar_close(actual.tlv, expected.tlv);
        assert_scalar_close(actual.tlw, expected.tlw);
    }

    #[test]
    fn gpu_hanna_matches_cpu_reference_for_stability_regimes() {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let mut pbl = PblState::new(3, 1);
        // Unstable (L = -50 m)
        pbl.ustar[[0, 0]] = 0.30;
        pbl.wstar[[0, 0]] = 1.80;
        pbl.hmix[[0, 0]] = 900.0;
        pbl.oli[[0, 0]] = -0.02;
        // Neutral (|1/L| < 1e-5 => L = +inf)
        pbl.ustar[[1, 0]] = 0.25;
        pbl.wstar[[1, 0]] = 0.0;
        pbl.hmix[[1, 0]] = 800.0;
        pbl.oli[[1, 0]] = 0.0;
        // Stable (L = +100 m)
        pbl.ustar[[2, 0]] = 0.35;
        pbl.wstar[[2, 0]] = 0.0;
        pbl.hmix[[2, 0]] = 700.0;
        pbl.oli[[2, 0]] = 0.01;

        let mut store = ParticleStore::with_capacity(4);
        store
            .add(make_particle(0.2, 0.0, 120.0))
            .expect("slot 0 available");
        store
            .add(make_particle(1.4, 0.0, 80.0))
            .expect("slot 1 available");
        store
            .add(make_particle(2.0, 0.0, 200.0))
            .expect("slot 2 available");
        // Slot 3 stays zeroed/inactive by design.

        let pbl_buffers = PblBuffers::from_state(&ctx, &pbl).expect("pbl upload succeeds");
        let particle_buffers = ParticleBuffers::from_store(&ctx, &store);

        let actual = pollster::block_on(compute_hanna_params_gpu(
            &ctx,
            &particle_buffers,
            &pbl_buffers,
        ))
        .expect("gpu hanna dispatch succeeds");

        assert_eq!(actual.len(), 4);
        let particles = store.as_slice();
        for idx in 0..3 {
            let expected = expected_from_cpu(&pbl, &particles[idx]);
            assert_hanna_close(&actual[idx], &expected);
        }

        let inactive_expected = HannaParams::zeroed();
        assert_eq!(actual[3].ust, inactive_expected.ust);
        assert_eq!(actual[3].wst, inactive_expected.wst);
        assert_eq!(actual[3].ol, inactive_expected.ol);
        assert_eq!(actual[3].h, inactive_expected.h);
        assert_eq!(actual[3].zeta, inactive_expected.zeta);
        assert_eq!(actual[3].sigu, inactive_expected.sigu);
        assert_eq!(actual[3].sigv, inactive_expected.sigv);
        assert_eq!(actual[3].sigw, inactive_expected.sigw);
        assert_eq!(actual[3].dsigwdz, inactive_expected.dsigwdz);
        assert_eq!(actual[3].dsigw2dz, inactive_expected.dsigw2dz);
        assert_eq!(actual[3].tlu, inactive_expected.tlu);
        assert_eq!(actual[3].tlv, inactive_expected.tlv);
        assert_eq!(actual[3].tlw, inactive_expected.tlw);
    }
}
