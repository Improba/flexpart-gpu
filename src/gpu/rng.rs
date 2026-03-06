//! GPU Philox RNG dispatch utilities (A-05).
//!
//! This module provides a WGSL-backed Philox4x32-10 generator for deterministic
//! per-thread random numbers. Each invocation produces one block of 4 uniforms.

use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::physics::rng::{PhiloxCounter, PhiloxKey};

use super::{download_buffer_typed, GpuBufferError, GpuContext, GpuError};

const WORKGROUP_SIZE_X: u32 = 64;
const SHADER_SOURCE: &str = include_str!("../shaders/philox_rng.wgsl");

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PhiloxParams {
    key0: u32,
    key1: u32,
    counter0: u32,
    counter1: u32,
    counter2: u32,
    counter3: u32,
    sample_count: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PhiloxUniformBlockRaw {
    values: [f32; 4],
}

/// One Philox output block represented as 4 uniforms in `[0, 1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhiloxUniformBlock {
    pub values: [f32; 4],
}

/// Errors returned by GPU Philox random generation.
#[derive(Debug, Error)]
pub enum GpuPhiloxError {
    #[error("sample_count does not fit in u32: {0}")]
    SampleCountTooLarge(usize),
    #[error("byte-size overflow while preparing output buffer")]
    SizeOverflow,
    #[error("GPU error: {0}")]
    Gpu(#[from] GpuError),
    #[error("buffer operation failed: {0}")]
    Buffer(#[from] GpuBufferError),
}

fn checked_output_size(sample_count: usize) -> Result<u64, GpuPhiloxError> {
    let byte_len = sample_count
        .checked_mul(size_of::<PhiloxUniformBlockRaw>())
        .ok_or(GpuPhiloxError::SizeOverflow)?;
    u64::try_from(byte_len).map_err(|_| GpuPhiloxError::SizeOverflow)
}

/// Generate `sample_count` Philox blocks on GPU.
///
/// Each block corresponds to counter `base_counter + thread_index`.
pub async fn sample_philox_uniform4_gpu(
    ctx: &GpuContext,
    key: PhiloxKey,
    base_counter: PhiloxCounter,
    sample_count: usize,
) -> Result<Vec<PhiloxUniformBlock>, GpuPhiloxError> {
    if sample_count == 0 {
        return Ok(Vec::new());
    }

    let sample_count_u32 = u32::try_from(sample_count)
        .map_err(|_| GpuPhiloxError::SampleCountTooLarge(sample_count))?;
    let output_size = checked_output_size(sample_count)?;

    let params = PhiloxParams {
        key0: key[0],
        key1: key[1],
        counter0: base_counter[0],
        counter1: base_counter[1],
        counter2: base_counter[2],
        counter3: base_counter[3],
        sample_count: sample_count_u32,
        pad0: 0,
    };

    let shader = ctx.load_shader("philox_rng_shader", SHADER_SOURCE);
    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("philox_rng_bgl"),
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let pipeline = ctx.create_compute_pipeline(
        "philox_rng_pipeline",
        &shader,
        "main",
        &[&bind_group_layout],
    );

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("philox_rng_outputs"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("philox_rng_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("philox_rng_bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("philox_rng_encoder"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("philox_rng_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        super::dispatch_1d(&mut cpass, sample_count_u32, WORKGROUP_SIZE_X);
    }
    ctx.queue.submit(Some(encoder.finish()));

    let raw = download_buffer_typed::<PhiloxUniformBlockRaw>(
        ctx,
        &output_buffer,
        sample_count,
        "philox_rng_outputs",
    )
    .await?;

    Ok(raw
        .into_iter()
        .map(|entry| PhiloxUniformBlock {
            values: entry.values,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::physics::rng::{philox4x32_uniforms, philox_counter_add};

    #[test]
    fn gpu_philox_matches_cpu_reference_blocks() {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let key = [0xDECA_FBAD, 0x1234_5678];
        let base_counter = [0, 1, 2, 3];
        let sample_count = 8_usize;

        let gpu_blocks = pollster::block_on(sample_philox_uniform4_gpu(
            &ctx,
            key,
            base_counter,
            sample_count,
        ))
        .expect("gpu philox dispatch succeeds");

        assert_eq!(gpu_blocks.len(), sample_count);

        for (idx, block) in gpu_blocks.iter().enumerate() {
            let counter = philox_counter_add(base_counter, idx as u64);
            let expected = philox4x32_uniforms(counter, key);
            for lane in 0..4 {
                assert_relative_eq!(block.values[lane], expected[lane], epsilon = 1.0e-7);
            }
        }
    }
}
