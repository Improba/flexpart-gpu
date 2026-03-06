//! PBL boundary reflection kernel (post-advection).
//!
//! Implements reflecting boundaries at z=0 and z=h following FLEXPART Fortran's
//! well-mixed criterion (Thomson 1987). Particles that cross the ground or PBL
//! top are reflected back, and their turbulent vertical velocity is reversed.

use bytemuck::{Pod, Zeroable};
use thiserror::Error;

use super::{GpuContext, HannaParamsOutputBuffer, ParticleBuffers};
use crate::gpu::workgroup::runtime_workgroup_size;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ReflectionParams {
    particle_count: u32,
    min_height_m: f32,
    _pad0: u32,
    _pad1: u32,
}

#[derive(Debug, Error)]
pub enum GpuPblReflectionError {
    #[error("particle_count is zero")]
    ZeroParticles,
}

/// Reusable dispatch kernel for PBL reflection.
pub struct PblReflectionDispatchKernel {
    pipeline: wgpu::ComputePipeline,
    workgroup_size_x: u32,
}

impl PblReflectionDispatchKernel {
    pub fn new(ctx: &GpuContext) -> Self {
        let shader_source = include_str!("../shaders/pbl_reflection.wgsl");
        let workgroup_size_x =
            runtime_workgroup_size(ctx, crate::gpu::workgroup::WorkgroupKernel::PblReflection);
        let source = shader_source.replace(
            "@workgroup_size(64)",
            &format!("@workgroup_size({workgroup_size_x})"),
        );
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("pbl_reflection_shader"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pbl_reflection_pipeline"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        Self {
            pipeline,
            workgroup_size_x,
        }
    }
}

/// Encode PBL reflection into a command encoder (for batched submission).
pub fn encode_pbl_reflection_gpu_with_kernel(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    hanna_output: &HannaParamsOutputBuffer,
    kernel: &PblReflectionDispatchKernel,
    encoder: &mut wgpu::CommandEncoder,
) -> Result<(), GpuPblReflectionError> {
    let particle_count = particles.particle_count();
    if particle_count == 0 {
        return Err(GpuPblReflectionError::ZeroParticles);
    }

    let raw = ReflectionParams {
        particle_count: particle_count as u32,
        min_height_m: 0.0,
        _pad0: 0,
        _pad1: 0,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbl_reflection_params"),
            contents: bytemuck::bytes_of(&raw),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group_layout = kernel.pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pbl_reflection_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: particles.particle_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: hanna_output.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pbl_reflection_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&kernel.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        super::dispatch_1d(&mut cpass, raw.particle_count, kernel.workgroup_size_x);
    }

    Ok(())
}

use wgpu::util::DeviceExt;
