//! Fused Hanna + Langevin GPU dispatch.
//!
//! Computes Hanna PBL turbulence parameters inline from PBL grid fields and
//! applies the Langevin turbulent velocity update in a single dispatch,
//! eliminating the intermediate HannaParams buffer and one dispatch barrier.
//!
//! This is the **default production path** for turbulence. The separated
//! Hanna → Langevin path (`hanna.rs` + `langevin.rs`) is used only when
//! `FLEXPART_GPU_VALIDATION=1` for scientific debugging.

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::physics::{philox_counter_add, LangevinStep, PhiloxCounter, PhiloxKey};

use super::{
    render_shader_with_workgroup_size, runtime_workgroup_size, GpuContext, ParticleBuffers,
    PblBuffers, WorkgroupKernel,
};

const SHADER_TEMPLATE: &str = include_str!("../shaders/langevin_fused.wgsl");

/// Reusable fused Hanna+Langevin dispatch kernel objects.
pub struct LangevinFusedDispatchKernel {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline: wgpu::ComputePipeline,
    workgroup_size_x: u32,
}

impl LangevinFusedDispatchKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let workgroup_size_x = runtime_workgroup_size(ctx, WorkgroupKernel::Langevin);
        let shader_source = render_shader_with_workgroup_size(SHADER_TEMPLATE, workgroup_size_x);
        let shader = ctx.load_shader("langevin_fused_shader", &shader_source);

        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("langevin_fused_bgl"),
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
                        storage_ro(1), // pbl_ustar
                        storage_ro(2), // pbl_wstar
                        storage_ro(3), // pbl_hmix
                        storage_ro(4), // pbl_oli
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
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
            "langevin_fused_pipeline",
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
struct FusedLangevinParams {
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
    pbl_nx: u32,
    pbl_ny: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[derive(Debug, Error)]
pub enum GpuLangevinFusedError {
    #[error("invalid dt_seconds: {dt_seconds}")]
    InvalidTimeStep { dt_seconds: f32 },
    #[error("PBL grid dimensions must be non-zero, got {shape:?}")]
    ZeroPblShape { shape: (usize, usize) },
    #[error("value for {field} does not fit in u32: {value}")]
    ValueTooLarge { field: &'static str, value: usize },
}

fn usize_to_u32(value: usize, field: &'static str) -> Result<u32, GpuLangevinFusedError> {
    u32::try_from(value).map_err(|_| GpuLangevinFusedError::ValueTooLarge { field, value })
}

/// Encode the fused Hanna+Langevin dispatch into an existing command encoder.
#[allow(clippy::too_many_arguments)]
pub fn encode_langevin_fused_gpu(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    pbl: &PblBuffers,
    step: LangevinStep,
    key: PhiloxKey,
    base_counter: PhiloxCounter,
    kernel: &LangevinFusedDispatchKernel,
    encoder: &mut wgpu::CommandEncoder,
) -> Result<PhiloxCounter, GpuLangevinFusedError> {
    let particle_count = particles.particle_count();
    if particle_count == 0 {
        return Ok(base_counter);
    }
    if !step.dt_seconds.is_finite() || step.dt_seconds <= 0.0 {
        return Err(GpuLangevinFusedError::InvalidTimeStep {
            dt_seconds: step.dt_seconds,
        });
    }
    let (pbl_nx, pbl_ny) = pbl.shape;
    if pbl_nx == 0 || pbl_ny == 0 {
        return Err(GpuLangevinFusedError::ZeroPblShape {
            shape: pbl.shape,
        });
    }

    let pc_u32 = usize_to_u32(particle_count, "particle_count")?;

    let params = FusedLangevinParams {
        key0: key[0],
        key1: key[1],
        counter0: base_counter[0],
        counter1: base_counter[1],
        counter2: base_counter[2],
        counter3: base_counter[3],
        particle_count: pc_u32,
        n_substeps: step.n_substeps,
        dt_seconds: step.dt_seconds,
        rho_grad_over_rho: step.rho_grad_over_rho,
        min_height_m: step.min_height_m,
        pbl_nx: usize_to_u32(pbl_nx, "pbl_nx")?,
        pbl_ny: usize_to_u32(pbl_ny, "pbl_ny")?,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("langevin_fused_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("langevin_fused_bg"),
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
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("langevin_fused_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&kernel.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        super::dispatch_1d(&mut cpass, pc_u32, kernel.workgroup_size_x);
    }

    let blocks_per_particle: u64 = if step.n_substeps <= 2 { 1 } else { 2 };
    Ok(philox_counter_add(
        base_counter,
        (particle_count as u64) * blocks_per_particle,
    ))
}
