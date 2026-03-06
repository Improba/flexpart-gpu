//! GPU convective mixing dispatch (C-03).
//!
//! Applies a precomputed convective redistribution matrix to particles by moving
//! each active particle to the expected destination height for its source layer.
//!
//! Matrix generation remains CPU-side (`physics::convection`), mirroring the
//! FLEXPART split between convection diagnostics and matrix application.

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::particles::ParticleStore;
use crate::physics::ConvectiveRedistributionMatrix;

use super::{GpuBufferError, GpuContext, GpuError, ParticleBuffers};

const WORKGROUP_SIZE_X: u32 = 64;
const SHADER_SOURCE: &str = include_str!("../shaders/convective_mixing.wgsl");

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ConvectiveMixingDispatchParams {
    particle_count: u32,
    level_count: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Errors returned by GPU convective mixing dispatch.
#[derive(Debug, Error)]
pub enum GpuConvectionError {
    #[error("value for {field} does not fit in u32: {value}")]
    ValueTooLarge { field: &'static str, value: usize },
    #[error("invalid level interfaces length: expected {expected}, got {actual}")]
    InvalidLevelInterfaces { expected: usize, actual: usize },
    #[error("invalid level count for convection dispatch: {level_count}")]
    InvalidLevelCount { level_count: usize },
    #[error("matrix level count mismatch: matrix={matrix}, interfaces={interfaces}")]
    LevelCountMismatch { matrix: usize, interfaces: usize },
    #[error("gpu buffer operation failed: {0}")]
    Buffer(#[from] GpuBufferError),
}

/// Errors for the higher-level convective workflow helper.
#[derive(Debug, Error)]
pub enum GpuConvectionWorkflowError {
    #[error("gpu initialization failed: {0}")]
    Gpu(#[from] GpuError),
    #[error("convective mixing dispatch failed: {0}")]
    Convection(#[from] GpuConvectionError),
    #[error("particle buffer readback failed: {0}")]
    Buffer(#[from] GpuBufferError),
}

fn usize_to_u32(value: usize, field: &'static str) -> Result<u32, GpuConvectionError> {
    u32::try_from(value).map_err(|_| GpuConvectionError::ValueTooLarge { field, value })
}

fn level_centers_from_interfaces(level_interfaces_m: &[f32]) -> Vec<f32> {
    level_interfaces_m
        .windows(2)
        .map(|pair| 0.5 * (pair[0] + pair[1]))
        .collect()
}

/// Dispatch the WGSL convective mixing kernel.
pub fn dispatch_convective_mixing_gpu(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    level_interfaces_m: &[f32],
    matrix: &ConvectiveRedistributionMatrix,
) -> Result<(), GpuConvectionError> {
    let level_count = matrix.level_count();
    if level_count == 0 {
        return Err(GpuConvectionError::InvalidLevelCount { level_count });
    }
    if level_count > i32::MAX as usize {
        return Err(GpuConvectionError::InvalidLevelCount { level_count });
    }
    if level_interfaces_m.len() != level_count + 1 {
        return Err(GpuConvectionError::InvalidLevelInterfaces {
            expected: level_count + 1,
            actual: level_interfaces_m.len(),
        });
    }
    if level_interfaces_m.len().saturating_sub(1) != level_count {
        return Err(GpuConvectionError::LevelCountMismatch {
            matrix: level_count,
            interfaces: level_interfaces_m.len().saturating_sub(1),
        });
    }

    let particle_count = particles.particle_count();
    if particle_count == 0 {
        return Ok(());
    }

    let level_centers_m = level_centers_from_interfaces(level_interfaces_m);
    let shader = ctx.load_shader("convective_mixing_shader", SHADER_SOURCE);
    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("convective_mixing_bgl"),
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let pipeline = ctx.create_compute_pipeline(
        "convective_mixing_pipeline",
        &shader,
        "main",
        &[&bind_group_layout],
    );

    let matrix_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("convective_matrix_column_major"),
            contents: bytemuck::cast_slice(matrix.coefficients_column_major()),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let level_interfaces_buffer =
        ctx.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("convective_level_interfaces"),
                contents: bytemuck::cast_slice(level_interfaces_m),
                usage: wgpu::BufferUsages::STORAGE,
            });
    let level_centers_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("convective_level_centers"),
            contents: bytemuck::cast_slice(&level_centers_m),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let params = ConvectiveMixingDispatchParams {
        particle_count: usize_to_u32(particle_count, "particle_count")?,
        level_count: usize_to_u32(level_count, "level_count")?,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("convective_mixing_dispatch_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("convective_mixing_bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: particles.particle_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: matrix_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: level_interfaces_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: level_centers_buffer.as_entire_binding(),
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
            label: Some("convective_mixing_encoder"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("convective_mixing_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        super::dispatch_1d(&mut cpass, params.particle_count, WORKGROUP_SIZE_X);
    }
    ctx.queue.submit(Some(encoder.finish()));
    let _ = ctx.device.poll(wgpu::Maintain::Wait);
    Ok(())
}

/// Workflow helper that updates a CPU particle store via GPU when available.
///
/// Returns:
/// - `Ok(Some(new_heights))` when a GPU adapter exists and the kernel runs.
/// - `Ok(None)` when no GPU adapter is available.
pub async fn apply_convective_mixing_step_workflow(
    particles: &mut ParticleStore,
    level_interfaces_m: &[f32],
    matrix: &ConvectiveRedistributionMatrix,
) -> Result<Option<Vec<f32>>, GpuConvectionWorkflowError> {
    let ctx = match GpuContext::new().await {
        Ok(ctx) => ctx,
        Err(GpuError::NoAdapter) => return Ok(None),
        Err(err) => return Err(err.into()),
    };

    let gpu_particles = ParticleBuffers::from_store(&ctx, particles);
    dispatch_convective_mixing_gpu(&ctx, &gpu_particles, level_interfaces_m, matrix)?;
    let updated_particles = gpu_particles.download_particles(&ctx).await?;
    particles.as_mut_slice().copy_from_slice(&updated_particles);
    particles.recount_active();
    let heights = particles
        .as_slice()
        .iter()
        .map(|particle| particle.pos_z)
        .collect();
    Ok(Some(heights))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::gpu::GpuError;
    use crate::particles::{Particle, ParticleInit, MAX_SPECIES};
    use crate::physics::{
        apply_convective_mixing_to_particles_cpu, build_simplified_convection_chain,
        SimplifiedEmanuelInputs,
    };

    fn particle_at(z_m: f32, mass0: f32) -> Particle {
        let mut mass = [0.0_f32; MAX_SPECIES];
        mass[0] = mass0;
        Particle::new(&ParticleInit {
            cell_x: 0,
            cell_y: 0,
            pos_x: 0.5,
            pos_y: 0.5,
            pos_z: z_m,
            mass,
            release_point: 0,
            class: 0,
            time: 0,
        })
    }

    #[test]
    fn gpu_convective_mixing_matches_cpu_reference() {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let inputs = SimplifiedEmanuelInputs {
            level_interfaces_m: vec![0.0, 400.0, 1_000.0, 2_000.0, 3_500.0, 5_000.0],
            convective_precip_mm_h: 12.0,
            convective_velocity_scale_m_s: 2.8,
            boundary_layer_height_m: 900.0,
            cape_override_j_kg: Some(900.0),
        };
        let (column, matrix) =
            build_simplified_convection_chain(inputs, 600.0).expect("convection chain builds");

        let mut gpu_input = vec![
            particle_at(150.0, 1.0),
            particle_at(700.0, 2.0),
            particle_at(1_700.0, 3.0),
            particle_at(3_200.0, 4.0),
        ];
        gpu_input[3].deactivate();
        let mut expected = gpu_input.clone();
        apply_convective_mixing_to_particles_cpu(
            &mut expected,
            &matrix,
            &column.level_interfaces_m,
            &column.level_centers_m,
        )
        .expect("cpu convection apply succeeds");

        let particle_buffers = ParticleBuffers::from_particles(&ctx, &gpu_input);
        dispatch_convective_mixing_gpu(
            &ctx,
            &particle_buffers,
            &column.level_interfaces_m,
            &matrix,
        )
        .expect("gpu convection dispatch succeeds");
        let actual = pollster::block_on(particle_buffers.download_particles(&ctx))
            .expect("particle readback succeeds");

        assert_eq!(actual.len(), expected.len());
        for (gpu, cpu) in actual.iter().zip(expected.iter()) {
            assert_eq!(gpu.flags, cpu.flags);
            assert_eq!(gpu.cell_x, cpu.cell_x);
            assert_eq!(gpu.cell_y, cpu.cell_y);
            assert_relative_eq!(gpu.pos_x, cpu.pos_x, epsilon = 1.0e-7);
            assert_relative_eq!(gpu.pos_y, cpu.pos_y, epsilon = 1.0e-7);
            assert_relative_eq!(
                gpu.pos_z,
                cpu.pos_z,
                epsilon = 5.0e-5,
                max_relative = 5.0e-5
            );
            assert_relative_eq!(
                gpu.mass[0],
                cpu.mass[0],
                epsilon = 1.0e-7,
                max_relative = 1.0e-7
            );
        }
    }

    #[test]
    fn workflow_api_gracefully_skips_without_adapter() {
        let inputs = SimplifiedEmanuelInputs {
            level_interfaces_m: vec![0.0, 500.0, 1_500.0, 3_000.0],
            convective_precip_mm_h: 8.0,
            convective_velocity_scale_m_s: 2.0,
            boundary_layer_height_m: 800.0,
            cape_override_j_kg: Some(600.0),
        };
        let (column, matrix) =
            build_simplified_convection_chain(inputs, 300.0).expect("convection chain builds");

        let mut store = ParticleStore::with_capacity(2);
        store.add(particle_at(100.0, 1.0)).expect("slot 0");
        store.add(particle_at(900.0, 1.0)).expect("slot 1");
        let baseline: Vec<f32> = store
            .as_slice()
            .iter()
            .map(|particle| particle.pos_z)
            .collect();

        let result = pollster::block_on(apply_convective_mixing_step_workflow(
            &mut store,
            &column.level_interfaces_m,
            &matrix,
        ))
        .expect("workflow call should succeed");

        match result {
            Some(new_heights) => {
                assert_eq!(new_heights.len(), 2);
            }
            None => {
                let after: Vec<f32> = store
                    .as_slice()
                    .iter()
                    .map(|particle| particle.pos_z)
                    .collect();
                assert_eq!(after, baseline);
            }
        }
    }
}
