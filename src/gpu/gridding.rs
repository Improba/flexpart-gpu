//! GPU concentration gridding dispatch (I-02).
//!
//! Ported from FLEXPART particle-to-grid accumulation workflow in
//! `conccalc.f90`. This module maps active particles to Eulerian grid cells
//! and atomically accumulates:
//! - particle count per cell `[-]`
//! - selected species mass per cell `[kg]` (stored as scaled `u32` atomics).
//!
//! MVP assumptions:
//! - nearest-cell mapping: `(ix, iy) = (cell_x, cell_y)`,
//!   `iz = floor(pos_z)`, each clamped to domain bounds.
//! - concentration is represented as accumulated species mass per cell (no
//!   cell-volume normalization yet).
//! - mass atomics use deterministic fixed-point quantization with
//!   `mass_scale` (`scaled = round(mass_kg * mass_scale)`).

use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::particles::{Particle, MAX_SPECIES};

use super::{
    download_buffer_typed, render_shader_with_workgroup_size, runtime_workgroup_size,
    GpuBufferError, GpuContext, ParticleBuffers, WorkgroupKernel,
};

const SHADER_TEMPLATE: &str = include_str!("../shaders/concentration_gridding.wgsl");

/// 3-D concentration-grid shape `(nx, ny, nz)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConcentrationGridShape {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

impl ConcentrationGridShape {
    /// Number of cells in the flattened row-major layout.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    fn validate_non_zero(&self) -> Result<(), GpuConcentrationGriddingError> {
        if self.nx == 0 || self.ny == 0 || self.nz == 0 {
            return Err(GpuConcentrationGriddingError::ZeroShape {
                shape: (self.nx, self.ny, self.nz),
            });
        }
        Ok(())
    }

    fn flatten_index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        ((ix * self.ny) + iy) * self.nz + iz
    }
}

/// Deterministic gridding outputs downloaded from GPU buffers.
#[derive(Debug, Clone)]
pub struct ConcentrationGridOutput {
    /// Grid shape used for flattening/unflattening.
    pub shape: ConcentrationGridShape,
    /// Number of active particles accumulated into each cell `[-]`.
    pub particle_count_per_cell: Vec<u32>,
    /// Accumulated species mass per cell `[kg]`.
    pub concentration_mass_kg: Vec<f32>,
}

impl ConcentrationGridOutput {
    /// Return particle count and mass for one cell.
    #[must_use]
    pub fn cell_values(&self, ix: usize, iy: usize, iz: usize) -> (u32, f32) {
        let flat = self.shape.flatten_index(ix, iy, iz);
        (
            self.particle_count_per_cell[flat],
            self.concentration_mass_kg[flat],
        )
    }
}

/// Maximum number of output height levels supported in the gridding shader.
pub const MAX_OUTPUT_LEVELS: usize = 16;

/// User-facing concentration gridding options.
#[derive(Debug, Clone, Copy)]
pub struct ConcentrationGriddingParams {
    /// Mass slot in [`Particle::mass`] to accumulate.
    pub species_index: usize,
    /// Deterministic fixed-point scale for mass atomics:
    /// `scaled_mass = round(mass_kg * mass_scale)`.
    pub mass_scale: f32,
    /// Output height levels in meters for vertical binning.
    /// When `outheights[0] > 0`, the shader maps physical height (pos_z in meters)
    /// to the appropriate level via linear scan. When all zeros, the shader falls
    /// back to `floor(pos_z)` treating pos_z as a fractional level index.
    pub outheights: [f32; MAX_OUTPUT_LEVELS],
}

impl Default for ConcentrationGriddingParams {
    fn default() -> Self {
        Self {
            species_index: 0,
            mass_scale: 1_000_000.0,
            outheights: [0.0; MAX_OUTPUT_LEVELS],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ConcentrationGriddingParamsRaw {
    nx: u32,
    ny: u32,
    nz: u32,
    particle_count: u32,
    species_index: u32,
    _pad0: u32,
    _pad1: u32,
    mass_scale: f32,
    outheights: [f32; MAX_OUTPUT_LEVELS],
}

/// Output buffers for concentration gridding accumulation.
pub struct ConcentrationGridIoBuffers {
    pub particle_count_per_cell: wgpu::Buffer,
    pub concentration_mass_scaled: wgpu::Buffer,
    shape: ConcentrationGridShape,
    cell_count: usize,
}

impl ConcentrationGridIoBuffers {
    /// Allocate grid output buffers for the provided shape.
    pub fn from_shape(
        ctx: &GpuContext,
        shape: ConcentrationGridShape,
    ) -> Result<Self, GpuConcentrationGriddingError> {
        shape.validate_non_zero()?;
        let cell_count = shape.cell_count();
        let bytes_len = checked_byte_len::<u32>(cell_count, "concentration_grid")?;
        let size =
            u64::try_from(bytes_len).map_err(|_| GpuConcentrationGriddingError::SizeOverflow {
                field: "concentration_grid",
            })?;

        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let particle_count_per_cell = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("concentration_grid_particle_count"),
            size,
            usage,
            mapped_at_creation: false,
        });
        let concentration_mass_scaled = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("concentration_grid_mass_scaled"),
            size,
            usage,
            mapped_at_creation: false,
        });

        Ok(Self {
            particle_count_per_cell,
            concentration_mass_scaled,
            shape,
            cell_count,
        })
    }

    #[must_use]
    pub fn shape(&self) -> ConcentrationGridShape {
        self.shape
    }

    /// Zero all grid outputs before the next dispatch.
    pub fn clear(&self, ctx: &GpuContext) {
        if self.cell_count == 0 {
            return;
        }
        let zeros = vec![0_u32; self.cell_count];
        let bytes = bytemuck::cast_slice(&zeros);
        ctx.queue
            .write_buffer(&self.particle_count_per_cell, 0, bytes);
        ctx.queue
            .write_buffer(&self.concentration_mass_scaled, 0, bytes);
    }

    /// Download gridding results and convert scaled mass to kilograms.
    pub async fn download_output(
        &self,
        ctx: &GpuContext,
        mass_scale: f32,
    ) -> Result<ConcentrationGridOutput, GpuConcentrationGriddingError> {
        let counts = download_buffer_typed::<u32>(
            ctx,
            &self.particle_count_per_cell,
            self.cell_count,
            "concentration_particle_count",
        )
        .await?;
        let mass_scaled = download_buffer_typed::<u32>(
            ctx,
            &self.concentration_mass_scaled,
            self.cell_count,
            "concentration_mass_scaled",
        )
        .await?;
        let concentration_mass_kg = mass_scaled
            .into_iter()
            .map(|value| (value as f32) / mass_scale)
            .collect();
        Ok(ConcentrationGridOutput {
            shape: self.shape,
            particle_count_per_cell: counts,
            concentration_mass_kg,
        })
    }
}

/// Errors returned by concentration gridding dispatch.
#[derive(Debug, Error)]
pub enum GpuConcentrationGriddingError {
    #[error("concentration grid shape must be non-zero, got {shape:?}")]
    ZeroShape { shape: (usize, usize, usize) },
    #[error("species_index out of range: {species_index} (MAX_SPECIES={max_species})")]
    InvalidSpeciesIndex {
        species_index: usize,
        max_species: usize,
    },
    #[error("invalid mass_scale for concentration gridding: {mass_scale}")]
    InvalidMassScale { mass_scale: f32 },
    #[error("value for {field} does not fit in u32: {value}")]
    ValueTooLarge { field: &'static str, value: usize },
    #[error("byte-size overflow while preparing {field}")]
    SizeOverflow { field: &'static str },
    #[error("buffer operation failed: {0}")]
    Buffer(#[from] GpuBufferError),
}

fn usize_to_u32(value: usize, field: &'static str) -> Result<u32, GpuConcentrationGriddingError> {
    u32::try_from(value).map_err(|_| GpuConcentrationGriddingError::ValueTooLarge { field, value })
}

fn checked_byte_len<T>(
    len: usize,
    field: &'static str,
) -> Result<usize, GpuConcentrationGriddingError> {
    len.checked_mul(size_of::<T>())
        .ok_or(GpuConcentrationGriddingError::SizeOverflow { field })
}

/// Dispatch GPU concentration accumulation for all particle slots.
pub fn dispatch_concentration_gridding_gpu(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    outputs: &ConcentrationGridIoBuffers,
    params: ConcentrationGriddingParams,
) -> Result<(), GpuConcentrationGriddingError> {
    outputs.shape.validate_non_zero()?;
    if params.species_index >= MAX_SPECIES {
        return Err(GpuConcentrationGriddingError::InvalidSpeciesIndex {
            species_index: params.species_index,
            max_species: MAX_SPECIES,
        });
    }
    if !params.mass_scale.is_finite() || params.mass_scale <= 0.0 {
        return Err(GpuConcentrationGriddingError::InvalidMassScale {
            mass_scale: params.mass_scale,
        });
    }
    if params.outheights[0] == 0.0 && outputs.shape.nz > 1 {
        log::warn!(
            "concentration gridding: outheights are all zeros with nz={} — \
             the shader will treat pos_z as a level index via floor(pos_z). \
             If pos_z is in meters, set outheights to the physical level heights.",
            outputs.shape.nz,
        );
    }

    outputs.clear(ctx);
    let particle_count = particles.particle_count();
    if particle_count == 0 {
        return Ok(());
    }

    debug_assert_eq!(Particle::GPU_SIZE, 96);

    let raw = ConcentrationGriddingParamsRaw {
        nx: usize_to_u32(outputs.shape.nx, "nx")?,
        ny: usize_to_u32(outputs.shape.ny, "ny")?,
        nz: usize_to_u32(outputs.shape.nz, "nz")?,
        particle_count: usize_to_u32(particle_count, "particle_count")?,
        species_index: usize_to_u32(params.species_index, "species_index")?,
        _pad0: 0,
        _pad1: 0,
        mass_scale: params.mass_scale,
        outheights: params.outheights,
    };
    let workgroup_size_x = runtime_workgroup_size(ctx, WorkgroupKernel::ConcentrationGridding);

    let shader_source = render_shader_with_workgroup_size(SHADER_TEMPLATE, workgroup_size_x);
    let shader = ctx.load_shader("concentration_gridding_shader", &shader_source);
    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("concentration_gridding_bgl"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
        "concentration_gridding_pipeline",
        &shader,
        "main",
        &[&bind_group_layout],
    );

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("concentration_gridding_params"),
            contents: bytemuck::bytes_of(&raw),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("concentration_gridding_bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: particles.particle_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: outputs.particle_count_per_cell.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: outputs.concentration_mass_scaled.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("concentration_gridding_encoder"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("concentration_gridding_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        super::dispatch_1d(&mut cpass, raw.particle_count, workgroup_size_x);
    }
    ctx.queue.submit(Some(encoder.finish()));
    let _ = ctx.device.poll(wgpu::Maintain::Wait);
    Ok(())
}

/// One-shot helper that allocates outputs, dispatches gridding, and downloads
/// deterministic host data.
pub async fn accumulate_concentration_grid_gpu(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    shape: ConcentrationGridShape,
    params: ConcentrationGriddingParams,
) -> Result<ConcentrationGridOutput, GpuConcentrationGriddingError> {
    let outputs = ConcentrationGridIoBuffers::from_shape(ctx, shape)?;
    dispatch_concentration_gridding_gpu(ctx, particles, &outputs, params)?;
    outputs.download_output(ctx, params.mass_scale).await
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::gpu::GpuError;
    use crate::particles::{Particle, ParticleInit};

    fn particle_for_cell(cell_x: i32, cell_y: i32, z: f32, mass_species0_kg: f32) -> Particle {
        let mut mass = [0.0; MAX_SPECIES];
        mass[0] = mass_species0_kg;
        Particle::new(&ParticleInit {
            cell_x,
            cell_y,
            pos_x: 0.25,
            pos_y: 0.75,
            pos_z: z,
            mass,
            release_point: 0,
            class: 0,
            time: 0,
        })
    }

    #[test]
    fn gpu_concentration_gridding_simple_particle_pattern_matches_expected_cells() {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let mut p3_inactive = particle_for_cell(0, 2, 0.0, 3.0);
        p3_inactive.deactivate();
        let particles = vec![
            particle_for_cell(1, 0, 0.2, 1.5),
            particle_for_cell(1, 0, 0.8, 0.5),
            particle_for_cell(2, 1, 1.2, 2.0),
            p3_inactive,
            particle_for_cell(3, 2, 1.99, 4.25),
        ];

        let particle_buffers = ParticleBuffers::from_particles(&ctx, &particles);
        let shape = ConcentrationGridShape {
            nx: 4,
            ny: 3,
            nz: 2,
        };
        let params = ConcentrationGriddingParams {
            species_index: 0,
            mass_scale: 1_000.0,
            outheights: [0.0; MAX_OUTPUT_LEVELS],
        };

        let output = pollster::block_on(accumulate_concentration_grid_gpu(
            &ctx,
            &particle_buffers,
            shape,
            params,
        ))
        .expect("gridding succeeds");

        for ix in 0..shape.nx {
            for iy in 0..shape.ny {
                for iz in 0..shape.nz {
                    let (count, mass_kg) = output.cell_values(ix, iy, iz);
                    let (expected_count, expected_mass) = match (ix, iy, iz) {
                        (1, 0, 0) => (2_u32, 2.0_f32),
                        (2, 1, 1) => (1_u32, 2.0_f32),
                        (3, 2, 1) => (1_u32, 4.25_f32),
                        _ => (0_u32, 0.0_f32),
                    };
                    assert_eq!(
                        count, expected_count,
                        "unexpected count at ({ix},{iy},{iz})"
                    );
                    assert_relative_eq!(mass_kg, expected_mass, epsilon = 1.0e-6);
                }
            }
        }
    }
}
