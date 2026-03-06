//! GPU trilinear wind interpolation (A-02).
//!
//! This module dispatches a WGSL compute kernel that interpolates `(u, v, w)`
//! at arbitrary fractional `(x, y, z)` query points.
//!
//! Buffer contract:
//! - Wind fields are flattened as ndarray row-major `(nx, ny, nz)`:
//!   `flat = ((ix * ny) + iy) * nz + iz`
//! - Queries are `vec4<f32>` storage elements: `(x, y, z, pad)`
//! - Outputs are `vec4<f32>` storage elements: `(u, v, w, pad)`
//! - Uniform params are `u32x4`: `(nx, ny, nz, query_count)`

use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::physics::WindVector;

use super::{download_buffer_typed, GpuBufferError, GpuContext, WindBuffers};

const WORKGROUP_SIZE_X: u32 = 64;
const SHADER_SOURCE: &str = include_str!("../shaders/wind_trilinear_interp.wgsl");

/// GPU input query in fractional grid-index space.
///
/// Coordinates are clamped to `[0, n-1]` per axis in the same way as the CPU
/// reference `interpolate_wind_trilinear`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WindInterpolationQuery {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub pad0: f32,
}

impl WindInterpolationQuery {
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z, pad0: 0.0 }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct WindInterpolationOutputRaw {
    u: f32,
    v: f32,
    w: f32,
    _pad0: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct InterpolationParams {
    nx: u32,
    ny: u32,
    nz: u32,
    query_count: u32,
}

/// Errors returned by GPU trilinear wind interpolation dispatch.
#[derive(Debug, Error)]
pub enum GpuWindInterpolationError {
    #[error("wind grid dimensions must be non-zero, got {shape:?}")]
    ZeroShape { shape: (usize, usize, usize) },
    #[error("value for {field} does not fit in u32: {value}")]
    ValueTooLarge { field: &'static str, value: usize },
    #[error("byte-size overflow while preparing {field}")]
    SizeOverflow { field: &'static str },
    #[error("buffer operation failed: {0}")]
    Buffer(#[from] GpuBufferError),
}

fn usize_to_u32(value: usize, field: &'static str) -> Result<u32, GpuWindInterpolationError> {
    u32::try_from(value).map_err(|_| GpuWindInterpolationError::ValueTooLarge { field, value })
}

fn checked_byte_len<T>(
    len: usize,
    field: &'static str,
) -> Result<usize, GpuWindInterpolationError> {
    len.checked_mul(size_of::<T>())
        .ok_or(GpuWindInterpolationError::SizeOverflow { field })
}

/// Run the WGSL trilinear interpolation kernel for all `queries`.
///
/// Inputs and outputs follow the explicit layout documented at module level.
pub async fn interpolate_wind_trilinear_gpu(
    ctx: &GpuContext,
    wind: &WindBuffers,
    queries: &[WindInterpolationQuery],
) -> Result<Vec<WindVector>, GpuWindInterpolationError> {
    if queries.is_empty() {
        return Ok(Vec::new());
    }

    let (nx, ny, nz) = wind.shape;
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(GpuWindInterpolationError::ZeroShape { shape: wind.shape });
    }

    let params = InterpolationParams {
        nx: usize_to_u32(nx, "nx")?,
        ny: usize_to_u32(ny, "ny")?,
        nz: usize_to_u32(nz, "nz")?,
        query_count: usize_to_u32(queries.len(), "query_count")?,
    };

    let shader = ctx.load_shader("wind_trilinear_interp_shader", SHADER_SOURCE);
    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("wind_trilinear_interp_bgl"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
        "wind_trilinear_interp_pipeline",
        &shader,
        "main",
        &[&bind_group_layout],
    );

    let query_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wind_trilinear_interp_queries"),
            contents: bytemuck::cast_slice(queries),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let output_len = queries.len();
    let output_bytes = checked_byte_len::<WindInterpolationOutputRaw>(
        output_len,
        "wind_trilinear_interp_outputs",
    )?;
    let output_size =
        u64::try_from(output_bytes).map_err(|_| GpuWindInterpolationError::SizeOverflow {
            field: "wind_trilinear_interp_outputs",
        })?;
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wind_trilinear_interp_outputs"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wind_trilinear_interp_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wind_trilinear_interp_bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wind.u_ms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wind.v_ms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wind.w_ms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: query_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wind_trilinear_interp_encoder"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wind_trilinear_interp_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        super::dispatch_1d(&mut cpass, params.query_count, WORKGROUP_SIZE_X);
    }
    ctx.queue.submit(Some(encoder.finish()));

    let raw = download_buffer_typed::<WindInterpolationOutputRaw>(
        ctx,
        &output_buffer,
        output_len,
        "wind_trilinear_interp_outputs",
    )
    .await?;

    Ok(raw
        .into_iter()
        .map(|value| WindVector {
            u: value.u,
            v: value.v,
            w: value.w,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::gpu::GpuError;
    use crate::physics::interpolate_wind_trilinear;
    use crate::wind::WindField3D;

    fn deterministic_test_field(nx: usize, ny: usize, nz: usize) -> WindField3D {
        let mut field = WindField3D::zeros(nx, ny, nz);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = i as f32;
                    let y = j as f32;
                    let z = k as f32;
                    field.u_ms[[i, j, k]] = 1.5 * x * x - 0.25 * y + 2.0 * z + 0.125 * x * y;
                    field.v_ms[[i, j, k]] = -0.75 * x + 0.5 * y * y - 0.3 * z + 0.2 * y * z;
                    field.w_ms[[i, j, k]] = 0.1 * x - 1.2 * y + 0.9 * z * z - 0.05 * x * z;
                }
            }
        }
        field
    }

    #[test]
    fn gpu_trilinear_matches_cpu_reference() {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let field = deterministic_test_field(6, 5, 4);
        let wind_buffers = WindBuffers::from_field(&ctx, &field).expect("wind upload succeeds");

        let queries = vec![
            WindInterpolationQuery::new(0.25, 0.5, 0.75),
            WindInterpolationQuery::new(1.9, 2.2, 1.1),
            WindInterpolationQuery::new(4.7, 3.0, 2.6),
            WindInterpolationQuery::new(-2.0, 9.0, 99.0),
            WindInterpolationQuery::new(5.0, 4.0, 3.0),
        ];

        let expected: Vec<WindVector> = queries
            .iter()
            .map(|query| interpolate_wind_trilinear(&field, query.x, query.y, query.z))
            .collect();

        let actual = pollster::block_on(interpolate_wind_trilinear_gpu(
            &ctx,
            &wind_buffers,
            &queries,
        ))
        .expect("gpu interpolation succeeds");

        assert_eq!(actual.len(), expected.len());
        for (gpu, cpu) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(gpu.u, cpu.u, epsilon = 1.0e-5);
            assert_relative_eq!(gpu.v, cpu.v, epsilon = 1.0e-5);
            assert_relative_eq!(gpu.w, cpu.w, epsilon = 1.0e-5);
        }
    }
}
