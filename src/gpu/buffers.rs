//! GPU buffer abstractions for particle, wind, and PBL data.
//!
//! These types provide a practical baseline for upcoming compute kernels:
//! - typed storage buffer creation
//! - CPU -> GPU uploads
//! - GPU -> CPU readback via staging + async `map_async`
//! - ndarray flattening/shape validation for meteorological fields

use std::mem::size_of;
use std::sync::mpsc;

use bytemuck::Pod;
use ndarray::{Array2, Array3};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::particles::{Particle, ParticleStore};
use crate::pbl::PblState;
use crate::wind::WindField3D;

use super::GpuContext;

/// Errors for GPU buffer creation, transfer, and readback.
#[derive(Debug, Error)]
pub enum GpuBufferError {
    #[error("shape mismatch for {field}: expected {expected}, got {actual}")]
    ShapeMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },
    #[error("length mismatch for {field}: expected {expected}, got {actual}")]
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("byte-size overflow while handling {field}")]
    SizeOverflow { field: &'static str },
    #[error("buffer copy size must be multiple of 4 bytes, got {byte_len}")]
    InvalidCopySize { byte_len: usize },
    #[error("buffer map failed: {0}")]
    BufferMap(#[from] wgpu::BufferAsyncError),
    #[error("failed to build ndarray from downloaded data: {0}")]
    NdarrayShape(#[from] ndarray::ShapeError),
    #[error("map callback channel closed unexpectedly")]
    MapChannelClosed,
    #[error("index out of bounds for {field}: index {index}, len {len}")]
    IndexOutOfBounds {
        field: &'static str,
        index: usize,
        len: usize,
    },
}

fn storage_rw_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
}

fn checked_byte_len<T>(len: usize, field: &'static str) -> Result<usize, GpuBufferError> {
    len.checked_mul(size_of::<T>())
        .ok_or(GpuBufferError::SizeOverflow { field })
}

fn create_storage_buffer_from_pod<T: Pod>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
) -> wgpu::Buffer {
    let usage = storage_rw_usage();
    if data.is_empty() {
        // Keep a tiny allocation to avoid backend issues around zero-sized buffers.
        return device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: 4,
            usage,
            mapped_at_creation: false,
        });
    }

    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage,
    })
}

fn write_exact_pod_slice<T: Pod>(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    field: &'static str,
    expected_len: usize,
    data: &[T],
) -> Result<(), GpuBufferError> {
    if data.len() != expected_len {
        return Err(GpuBufferError::LengthMismatch {
            field,
            expected: expected_len,
            actual: data.len(),
        });
    }
    if data.is_empty() {
        return Ok(());
    }

    queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
    Ok(())
}

/// Download raw bytes from `source` using an internal MAP_READ staging buffer.
pub async fn download_buffer_bytes(
    ctx: &GpuContext,
    source: &wgpu::Buffer,
    byte_len: usize,
) -> Result<Vec<u8>, GpuBufferError> {
    if byte_len == 0 {
        return Ok(Vec::new());
    }
    if byte_len % 4 != 0 {
        return Err(GpuBufferError::InvalidCopySize { byte_len });
    }

    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_readback_staging"),
        size: byte_len as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_readback_encoder"),
        });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, byte_len as u64);
    ctx.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    let _ = ctx.device.poll(wgpu::Maintain::Wait);

    let map_result = rx.recv().map_err(|_| GpuBufferError::MapChannelClosed)?;
    map_result?;

    let data = slice.get_mapped_range().to_vec();
    staging.unmap();
    Ok(data)
}

/// Download a typed vector from a GPU buffer via staging + async map.
pub async fn download_buffer_typed<T: Pod>(
    ctx: &GpuContext,
    source: &wgpu::Buffer,
    len: usize,
    field: &'static str,
) -> Result<Vec<T>, GpuBufferError> {
    let byte_len = checked_byte_len::<T>(len, field)?;
    let bytes = download_buffer_bytes(ctx, source, byte_len).await?;
    Ok(bytemuck::cast_slice::<u8, T>(&bytes).to_vec())
}

/// Flattened CPU-side representation of all 3-D wind fields.
#[derive(Debug, Clone)]
pub struct WindHostData {
    pub shape: (usize, usize, usize),
    pub u_ms: Vec<f32>,
    pub v_ms: Vec<f32>,
    pub w_ms: Vec<f32>,
    pub temperature_k: Vec<f32>,
    pub specific_humidity: Vec<f32>,
    pub pressure_pa: Vec<f32>,
    pub air_density_kg_m3: Vec<f32>,
    pub density_gradient_kg_m2: Vec<f32>,
}

impl WindHostData {
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.shape.0 * self.shape.1 * self.shape.2
    }

    #[must_use]
    pub fn bytes_per_field(&self) -> usize {
        self.cell_count() * size_of::<f32>()
    }

    pub fn to_wind_field(&self) -> Result<WindField3D, GpuBufferError> {
        let shape = self.shape;
        let expected = self.cell_count();
        for (field, len) in [
            ("u_ms", self.u_ms.len()),
            ("v_ms", self.v_ms.len()),
            ("w_ms", self.w_ms.len()),
            ("temperature_k", self.temperature_k.len()),
            ("specific_humidity", self.specific_humidity.len()),
            ("pressure_pa", self.pressure_pa.len()),
            ("air_density_kg_m3", self.air_density_kg_m3.len()),
            ("density_gradient_kg_m2", self.density_gradient_kg_m2.len()),
        ] {
            if len != expected {
                return Err(GpuBufferError::LengthMismatch {
                    field,
                    expected,
                    actual: len,
                });
            }
        }

        Ok(WindField3D {
            u_ms: Array3::from_shape_vec(shape, self.u_ms.clone())?,
            v_ms: Array3::from_shape_vec(shape, self.v_ms.clone())?,
            w_ms: Array3::from_shape_vec(shape, self.w_ms.clone())?,
            temperature_k: Array3::from_shape_vec(shape, self.temperature_k.clone())?,
            specific_humidity: Array3::from_shape_vec(shape, self.specific_humidity.clone())?,
            pressure_pa: Array3::from_shape_vec(shape, self.pressure_pa.clone())?,
            air_density_kg_m3: Array3::from_shape_vec(shape, self.air_density_kg_m3.clone())?,
            density_gradient_kg_m2: Array3::from_shape_vec(
                shape,
                self.density_gradient_kg_m2.clone(),
            )?,
        })
    }
}

fn flatten_array3_f32(array: &Array3<f32>) -> Vec<f32> {
    array.iter().copied().collect()
}

fn flatten_array2_f32(array: &Array2<f32>) -> Vec<f32> {
    array.iter().copied().collect()
}

impl TryFrom<&WindField3D> for WindHostData {
    type Error = GpuBufferError;

    fn try_from(value: &WindField3D) -> Result<Self, Self::Error> {
        let expected = value.u_ms.shape();
        let expected_tuple = (expected[0], expected[1], expected[2]);

        for (field, actual) in [
            ("v_ms", value.v_ms.shape()),
            ("w_ms", value.w_ms.shape()),
            ("temperature_k", value.temperature_k.shape()),
            ("specific_humidity", value.specific_humidity.shape()),
            ("pressure_pa", value.pressure_pa.shape()),
            ("air_density_kg_m3", value.air_density_kg_m3.shape()),
            (
                "density_gradient_kg_m2",
                value.density_gradient_kg_m2.shape(),
            ),
        ] {
            if actual != expected {
                return Err(GpuBufferError::ShapeMismatch {
                    field,
                    expected: format!("{expected:?}"),
                    actual: format!("{actual:?}"),
                });
            }
        }

        Ok(Self {
            shape: expected_tuple,
            u_ms: flatten_array3_f32(&value.u_ms),
            v_ms: flatten_array3_f32(&value.v_ms),
            w_ms: flatten_array3_f32(&value.w_ms),
            temperature_k: flatten_array3_f32(&value.temperature_k),
            specific_humidity: flatten_array3_f32(&value.specific_humidity),
            pressure_pa: flatten_array3_f32(&value.pressure_pa),
            air_density_kg_m3: flatten_array3_f32(&value.air_density_kg_m3),
            density_gradient_kg_m2: flatten_array3_f32(&value.density_gradient_kg_m2),
        })
    }
}

/// Flattened CPU-side representation of all gridded PBL fields.
#[derive(Debug, Clone)]
pub struct PblHostData {
    pub shape: (usize, usize),
    pub ustar: Vec<f32>,
    pub wstar: Vec<f32>,
    pub hmix: Vec<f32>,
    pub oli: Vec<f32>,
    pub sshf: Vec<f32>,
    pub ssr: Vec<f32>,
    pub surfstr: Vec<f32>,
}

impl PblHostData {
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.shape.0 * self.shape.1
    }

    #[must_use]
    pub fn bytes_per_field(&self) -> usize {
        self.cell_count() * size_of::<f32>()
    }

    pub fn to_pbl_state(&self) -> Result<PblState, GpuBufferError> {
        let shape = self.shape;
        let expected = self.cell_count();
        for (field, len) in [
            ("ustar", self.ustar.len()),
            ("wstar", self.wstar.len()),
            ("hmix", self.hmix.len()),
            ("oli", self.oli.len()),
            ("sshf", self.sshf.len()),
            ("ssr", self.ssr.len()),
            ("surfstr", self.surfstr.len()),
        ] {
            if len != expected {
                return Err(GpuBufferError::LengthMismatch {
                    field,
                    expected,
                    actual: len,
                });
            }
        }

        Ok(PblState {
            ustar: Array2::from_shape_vec(shape, self.ustar.clone())?,
            wstar: Array2::from_shape_vec(shape, self.wstar.clone())?,
            hmix: Array2::from_shape_vec(shape, self.hmix.clone())?,
            oli: Array2::from_shape_vec(shape, self.oli.clone())?,
            sshf: Array2::from_shape_vec(shape, self.sshf.clone())?,
            ssr: Array2::from_shape_vec(shape, self.ssr.clone())?,
            surfstr: Array2::from_shape_vec(shape, self.surfstr.clone())?,
        })
    }
}

impl TryFrom<&PblState> for PblHostData {
    type Error = GpuBufferError;

    fn try_from(value: &PblState) -> Result<Self, Self::Error> {
        let expected = value.ustar.shape();
        let expected_tuple = (expected[0], expected[1]);

        for (field, actual) in [
            ("wstar", value.wstar.shape()),
            ("hmix", value.hmix.shape()),
            ("oli", value.oli.shape()),
            ("sshf", value.sshf.shape()),
            ("ssr", value.ssr.shape()),
            ("surfstr", value.surfstr.shape()),
        ] {
            if actual != expected {
                return Err(GpuBufferError::ShapeMismatch {
                    field,
                    expected: format!("{expected:?}"),
                    actual: format!("{actual:?}"),
                });
            }
        }

        Ok(Self {
            shape: expected_tuple,
            ustar: flatten_array2_f32(&value.ustar),
            wstar: flatten_array2_f32(&value.wstar),
            hmix: flatten_array2_f32(&value.hmix),
            oli: flatten_array2_f32(&value.oli),
            sshf: flatten_array2_f32(&value.sshf),
            ssr: flatten_array2_f32(&value.ssr),
            surfstr: flatten_array2_f32(&value.surfstr),
        })
    }
}

/// GPU storage buffer for particle data.
///
/// Supports an adjustable dispatch count for compaction (O-07): after
/// active-particle compaction packs active particles into contiguous
/// leading slots, [`set_dispatch_count`](Self::set_dispatch_count)
/// narrows subsequent dispatches to only those slots. Buffer I/O
/// (upload/download) always operates on the full
/// [`capacity`](Self::capacity).
pub struct ParticleBuffers {
    pub particle_buffer: wgpu::Buffer,
    buffer_capacity: usize,
    dispatch_count: usize,
}

impl ParticleBuffers {
    /// Create particle buffers from a full particle store.
    #[must_use]
    pub fn from_store(ctx: &GpuContext, store: &ParticleStore) -> Self {
        Self::from_particles(ctx, store.as_slice())
    }

    /// Create particle buffers from a particle slice.
    #[must_use]
    pub fn from_particles(ctx: &GpuContext, particles: &[Particle]) -> Self {
        let particle_buffer =
            create_storage_buffer_from_pod(&ctx.device, "particles_storage", particles);
        Self {
            particle_buffer,
            buffer_capacity: particles.len(),
            dispatch_count: particles.len(),
        }
    }

    /// Number of particles to dispatch (may be reduced by compaction).
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.dispatch_count
    }

    /// Total allocated particle slots in the GPU buffer.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.buffer_capacity
    }

    /// Narrow the dispatch count after compaction packs active particles
    /// into the first `count` contiguous slots.
    ///
    /// # Panics
    ///
    /// Panics if `count > capacity()`.
    pub fn set_dispatch_count(&mut self, count: usize) {
        assert!(
            count <= self.buffer_capacity,
            "dispatch count {count} exceeds buffer capacity {}",
            self.buffer_capacity,
        );
        self.dispatch_count = count;
    }

    /// Reset dispatch count to the full buffer capacity.
    pub fn reset_dispatch_count(&mut self) {
        self.dispatch_count = self.buffer_capacity;
    }

    pub fn upload_particles(
        &self,
        ctx: &GpuContext,
        particles: &[Particle],
    ) -> Result<(), GpuBufferError> {
        write_exact_pod_slice(
            &ctx.queue,
            &self.particle_buffer,
            "particles",
            self.buffer_capacity,
            particles,
        )
    }

    pub fn upload_store(
        &self,
        ctx: &GpuContext,
        store: &ParticleStore,
    ) -> Result<(), GpuBufferError> {
        self.upload_particles(ctx, store.as_slice())
    }

    /// Upload only selected particle slots.
    ///
    /// Useful for incremental source-term injection: the host updates new
    /// particles in [`ParticleStore`] and uploads only touched indices.
    pub fn upload_particle_slots(
        &self,
        ctx: &GpuContext,
        store: &ParticleStore,
        indices: &[usize],
    ) -> Result<(), GpuBufferError> {
        for &index in indices {
            let particle = store.get(index).ok_or(GpuBufferError::IndexOutOfBounds {
                field: "particles",
                index,
                len: self.buffer_capacity,
            })?;
            if index >= self.buffer_capacity {
                return Err(GpuBufferError::IndexOutOfBounds {
                    field: "particles",
                    index,
                    len: self.buffer_capacity,
                });
            }
            let byte_offset = checked_byte_len::<Particle>(index, "particles_slot_offset")?
                as wgpu::BufferAddress;
            ctx.queue.write_buffer(
                &self.particle_buffer,
                byte_offset,
                bytemuck::bytes_of(particle),
            );
        }
        Ok(())
    }

    /// Download all particle slots (full buffer capacity).
    pub async fn download_particles(
        &self,
        ctx: &GpuContext,
    ) -> Result<Vec<Particle>, GpuBufferError> {
        download_buffer_typed::<Particle>(
            ctx,
            &self.particle_buffer,
            self.buffer_capacity,
            "particles",
        )
        .await
    }
}

/// GPU storage buffers for all 3-D wind fields.
pub struct WindBuffers {
    pub shape: (usize, usize, usize),
    pub u_ms: wgpu::Buffer,
    pub v_ms: wgpu::Buffer,
    pub w_ms: wgpu::Buffer,
    pub temperature_k: wgpu::Buffer,
    pub specific_humidity: wgpu::Buffer,
    pub pressure_pa: wgpu::Buffer,
    pub air_density_kg_m3: wgpu::Buffer,
    pub density_gradient_kg_m2: wgpu::Buffer,
    pub sampled_wind_uvw: Option<WindSampledTexture3d>,
    cell_count: usize,
}

/// Optional packed `(u, v, w, pad)` sampled 3-D texture for wind lookup.
///
/// The texture uses `Rgba32Float` and is laid out as `(z, y, x)` in texture
/// axes so that host flatten order `((x * ny) + y) * nz + z` stays contiguous.
pub struct WindSampledTexture3d {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

impl WindBuffers {
    pub fn from_field(ctx: &GpuContext, wind: &WindField3D) -> Result<Self, GpuBufferError> {
        let host = WindHostData::try_from(wind)?;
        Ok(Self::from_host_data(ctx, &host))
    }

    #[must_use]
    pub fn from_host_data(ctx: &GpuContext, host: &WindHostData) -> Self {
        let sampled_wind_uvw = create_sampled_wind_texture_3d(ctx, host);
        Self {
            shape: host.shape,
            u_ms: create_storage_buffer_from_pod(&ctx.device, "wind_u_ms", &host.u_ms),
            v_ms: create_storage_buffer_from_pod(&ctx.device, "wind_v_ms", &host.v_ms),
            w_ms: create_storage_buffer_from_pod(&ctx.device, "wind_w_ms", &host.w_ms),
            temperature_k: create_storage_buffer_from_pod(
                &ctx.device,
                "wind_temperature_k",
                &host.temperature_k,
            ),
            specific_humidity: create_storage_buffer_from_pod(
                &ctx.device,
                "wind_specific_humidity",
                &host.specific_humidity,
            ),
            pressure_pa: create_storage_buffer_from_pod(
                &ctx.device,
                "wind_pressure_pa",
                &host.pressure_pa,
            ),
            air_density_kg_m3: create_storage_buffer_from_pod(
                &ctx.device,
                "wind_air_density_kg_m3",
                &host.air_density_kg_m3,
            ),
            density_gradient_kg_m2: create_storage_buffer_from_pod(
                &ctx.device,
                "wind_density_gradient_kg_m2",
                &host.density_gradient_kg_m2,
            ),
            sampled_wind_uvw,
            cell_count: host.cell_count(),
        }
    }

    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.cell_count
    }

    pub fn upload_field(&self, ctx: &GpuContext, wind: &WindField3D) -> Result<(), GpuBufferError> {
        let host = WindHostData::try_from(wind)?;
        if host.shape != self.shape {
            return Err(GpuBufferError::ShapeMismatch {
                field: "wind_3d",
                expected: format!("{:?}", self.shape),
                actual: format!("{:?}", host.shape),
            });
        }

        write_exact_pod_slice(
            &ctx.queue,
            &self.u_ms,
            "wind_u_ms",
            self.cell_count,
            &host.u_ms,
        )?;
        write_exact_pod_slice(
            &ctx.queue,
            &self.v_ms,
            "wind_v_ms",
            self.cell_count,
            &host.v_ms,
        )?;
        write_exact_pod_slice(
            &ctx.queue,
            &self.w_ms,
            "wind_w_ms",
            self.cell_count,
            &host.w_ms,
        )?;
        write_exact_pod_slice(
            &ctx.queue,
            &self.temperature_k,
            "wind_temperature_k",
            self.cell_count,
            &host.temperature_k,
        )?;
        write_exact_pod_slice(
            &ctx.queue,
            &self.specific_humidity,
            "wind_specific_humidity",
            self.cell_count,
            &host.specific_humidity,
        )?;
        write_exact_pod_slice(
            &ctx.queue,
            &self.pressure_pa,
            "wind_pressure_pa",
            self.cell_count,
            &host.pressure_pa,
        )?;
        write_exact_pod_slice(
            &ctx.queue,
            &self.air_density_kg_m3,
            "wind_air_density_kg_m3",
            self.cell_count,
            &host.air_density_kg_m3,
        )?;
        write_exact_pod_slice(
            &ctx.queue,
            &self.density_gradient_kg_m2,
            "wind_density_gradient_kg_m2",
            self.cell_count,
            &host.density_gradient_kg_m2,
        )?;
        if let Some(texture) = &self.sampled_wind_uvw {
            upload_sampled_wind_texture_3d(ctx, texture, &host);
        }
        Ok(())
    }

    pub async fn download_host_data(
        &self,
        ctx: &GpuContext,
    ) -> Result<WindHostData, GpuBufferError> {
        Ok(WindHostData {
            shape: self.shape,
            u_ms: download_buffer_typed(ctx, &self.u_ms, self.cell_count, "wind_u_ms").await?,
            v_ms: download_buffer_typed(ctx, &self.v_ms, self.cell_count, "wind_v_ms").await?,
            w_ms: download_buffer_typed(ctx, &self.w_ms, self.cell_count, "wind_w_ms").await?,
            temperature_k: download_buffer_typed(
                ctx,
                &self.temperature_k,
                self.cell_count,
                "wind_temperature_k",
            )
            .await?,
            specific_humidity: download_buffer_typed(
                ctx,
                &self.specific_humidity,
                self.cell_count,
                "wind_specific_humidity",
            )
            .await?,
            pressure_pa: download_buffer_typed(
                ctx,
                &self.pressure_pa,
                self.cell_count,
                "wind_pressure_pa",
            )
            .await?,
            air_density_kg_m3: download_buffer_typed(
                ctx,
                &self.air_density_kg_m3,
                self.cell_count,
                "wind_air_density_kg_m3",
            )
            .await?,
            density_gradient_kg_m2: download_buffer_typed(
                ctx,
                &self.density_gradient_kg_m2,
                self.cell_count,
                "wind_density_gradient_kg_m2",
            )
            .await?,
        })
    }

    pub async fn download_field(&self, ctx: &GpuContext) -> Result<WindField3D, GpuBufferError> {
        self.download_host_data(ctx).await?.to_wind_field()
    }
}

fn create_sampled_wind_texture_3d(
    ctx: &GpuContext,
    host: &WindHostData,
) -> Option<WindSampledTexture3d> {
    if !ctx.supports_wind_texture_sampling() {
        return None;
    }
    let (nx, ny, nz) = host.shape;
    let nx = u32::try_from(nx).ok()?;
    let ny = u32::try_from(ny).ok()?;
    let nz = u32::try_from(nz).ok()?;
    let max_dim = ctx.device.limits().max_texture_dimension_3d;
    if nx > max_dim || ny > max_dim || nz > max_dim {
        return None;
    }

    let texture_extent = wgpu::Extent3d {
        // Memory flatten order is (x, y, z) with z contiguous; map that to (width, height, depth) = (z, y, x).
        width: nz,
        height: ny,
        depth_or_array_layers: nx,
    };
    let descriptor = wgpu::TextureDescriptor {
        label: Some("wind_sampled_uvw_texture_3d"),
        size: texture_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };

    let packed = pack_sampled_wind_texture_rgba(host);
    let texture = ctx.device.create_texture_with_data(
        &ctx.queue,
        &descriptor,
        wgpu::util::TextureDataOrder::LayerMajor,
        bytemuck::cast_slice(&packed),
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("wind_sampled_uvw_texture_3d_view"),
        dimension: Some(wgpu::TextureViewDimension::D3),
        ..Default::default()
    });
    Some(WindSampledTexture3d { texture, view })
}

fn upload_sampled_wind_texture_3d(
    ctx: &GpuContext,
    texture: &WindSampledTexture3d,
    host: &WindHostData,
) {
    let (nx, ny, nz) = host.shape;
    let extent = wgpu::Extent3d {
        width: u32::try_from(nz).expect("wind nz fits u32 when sampled texture exists"),
        height: u32::try_from(ny).expect("wind ny fits u32 when sampled texture exists"),
        depth_or_array_layers: u32::try_from(nx)
            .expect("wind nx fits u32 when sampled texture exists"),
    };
    let packed = pack_sampled_wind_texture_rgba(host);
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&packed),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(16 * extent.width),
            rows_per_image: Some(extent.height),
        },
        extent,
    );
}

fn pack_sampled_wind_texture_rgba(host: &WindHostData) -> Vec<f32> {
    let mut packed = Vec::with_capacity(host.cell_count() * 4);
    for index in 0..host.cell_count() {
        packed.push(host.u_ms[index]);
        packed.push(host.v_ms[index]);
        packed.push(host.w_ms[index]);
        packed.push(0.0);
    }
    packed
}

/// GPU storage buffers holding two wind field snapshots for dual-time
/// temporal interpolation on the GPU (Tier 1.2 optimization).
///
/// Instead of the CPU computing `(1-alpha)*t0 + alpha*t1` and uploading the
/// result each step, the two bracket endpoints are uploaded once and the GPU
/// blends them inline using `alpha`.
pub struct DualWindBuffers {
    pub shape: (usize, usize, usize),
    pub u_ms_t0: wgpu::Buffer,
    pub v_ms_t0: wgpu::Buffer,
    pub w_ms_t0: wgpu::Buffer,
    pub u_ms_t1: wgpu::Buffer,
    pub v_ms_t1: wgpu::Buffer,
    pub w_ms_t1: wgpu::Buffer,
    pub sampled_wind_uvw_t0: Option<WindSampledTexture3d>,
    pub sampled_wind_uvw_t1: Option<WindSampledTexture3d>,
    cell_count: usize,
}

impl DualWindBuffers {
    /// Create dual-time wind buffers from two wind field snapshots.
    ///
    /// # Errors
    ///
    /// Returns [`GpuBufferError::ShapeMismatch`] if `wind_t0` and `wind_t1`
    /// have different grid dimensions.
    pub fn from_fields(
        ctx: &GpuContext,
        wind_t0: &WindField3D,
        wind_t1: &WindField3D,
    ) -> Result<Self, GpuBufferError> {
        let host_t0 = WindHostData::try_from(wind_t0)?;
        let host_t1 = WindHostData::try_from(wind_t1)?;
        if host_t0.shape != host_t1.shape {
            return Err(GpuBufferError::ShapeMismatch {
                field: "dual_wind_t1",
                expected: format!("{:?}", host_t0.shape),
                actual: format!("{:?}", host_t1.shape),
            });
        }
        Ok(Self::from_host_data_pair(ctx, &host_t0, &host_t1))
    }

    /// Create dual-time wind buffers from two pre-flattened host snapshots.
    #[must_use]
    pub fn from_host_data_pair(
        ctx: &GpuContext,
        host_t0: &WindHostData,
        host_t1: &WindHostData,
    ) -> Self {
        debug_assert_eq!(host_t0.shape, host_t1.shape);
        let sampled_t0 = create_sampled_wind_texture_3d(ctx, host_t0);
        let sampled_t1 = create_sampled_wind_texture_3d(ctx, host_t1);
        Self {
            shape: host_t0.shape,
            u_ms_t0: create_storage_buffer_from_pod(&ctx.device, "dual_wind_u_t0", &host_t0.u_ms),
            v_ms_t0: create_storage_buffer_from_pod(&ctx.device, "dual_wind_v_t0", &host_t0.v_ms),
            w_ms_t0: create_storage_buffer_from_pod(&ctx.device, "dual_wind_w_t0", &host_t0.w_ms),
            u_ms_t1: create_storage_buffer_from_pod(&ctx.device, "dual_wind_u_t1", &host_t1.u_ms),
            v_ms_t1: create_storage_buffer_from_pod(&ctx.device, "dual_wind_v_t1", &host_t1.v_ms),
            w_ms_t1: create_storage_buffer_from_pod(&ctx.device, "dual_wind_w_t1", &host_t1.w_ms),
            sampled_wind_uvw_t0: sampled_t0,
            sampled_wind_uvw_t1: sampled_t1,
            cell_count: host_t0.cell_count(),
        }
    }

    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.cell_count
    }

    /// Upload replacement wind data for the t0 bracket endpoint.
    ///
    /// # Errors
    ///
    /// Returns [`GpuBufferError`] on shape mismatch or length mismatch.
    pub fn upload_t0(
        &self,
        ctx: &GpuContext,
        wind: &WindField3D,
    ) -> Result<(), GpuBufferError> {
        let host = WindHostData::try_from(wind)?;
        self.upload_t0_host(ctx, &host)
    }

    /// Upload replacement wind data for the t1 bracket endpoint.
    ///
    /// # Errors
    ///
    /// Returns [`GpuBufferError`] on shape mismatch or length mismatch.
    pub fn upload_t1(
        &self,
        ctx: &GpuContext,
        wind: &WindField3D,
    ) -> Result<(), GpuBufferError> {
        let host = WindHostData::try_from(wind)?;
        self.upload_t1_host(ctx, &host)
    }

    /// Upload pre-flattened host data for the t0 endpoint.
    ///
    /// # Errors
    ///
    /// Returns [`GpuBufferError`] on shape mismatch or length mismatch.
    pub fn upload_t0_host(
        &self,
        ctx: &GpuContext,
        host: &WindHostData,
    ) -> Result<(), GpuBufferError> {
        if host.shape != self.shape {
            return Err(GpuBufferError::ShapeMismatch {
                field: "dual_wind_t0",
                expected: format!("{:?}", self.shape),
                actual: format!("{:?}", host.shape),
            });
        }
        write_exact_pod_slice(&ctx.queue, &self.u_ms_t0, "dual_wind_u_t0", self.cell_count, &host.u_ms)?;
        write_exact_pod_slice(&ctx.queue, &self.v_ms_t0, "dual_wind_v_t0", self.cell_count, &host.v_ms)?;
        write_exact_pod_slice(&ctx.queue, &self.w_ms_t0, "dual_wind_w_t0", self.cell_count, &host.w_ms)?;
        if let Some(texture) = &self.sampled_wind_uvw_t0 {
            upload_sampled_wind_texture_3d(ctx, texture, host);
        }
        Ok(())
    }

    /// Upload pre-flattened host data for the t1 endpoint.
    ///
    /// # Errors
    ///
    /// Returns [`GpuBufferError`] on shape mismatch or length mismatch.
    pub fn upload_t1_host(
        &self,
        ctx: &GpuContext,
        host: &WindHostData,
    ) -> Result<(), GpuBufferError> {
        if host.shape != self.shape {
            return Err(GpuBufferError::ShapeMismatch {
                field: "dual_wind_t1",
                expected: format!("{:?}", self.shape),
                actual: format!("{:?}", host.shape),
            });
        }
        write_exact_pod_slice(&ctx.queue, &self.u_ms_t1, "dual_wind_u_t1", self.cell_count, &host.u_ms)?;
        write_exact_pod_slice(&ctx.queue, &self.v_ms_t1, "dual_wind_v_t1", self.cell_count, &host.v_ms)?;
        write_exact_pod_slice(&ctx.queue, &self.w_ms_t1, "dual_wind_w_t1", self.cell_count, &host.w_ms)?;
        if let Some(texture) = &self.sampled_wind_uvw_t1 {
            upload_sampled_wind_texture_3d(ctx, texture, host);
        }
        Ok(())
    }

    /// Whether both t0 and t1 have sampled 3-D wind textures available.
    #[must_use]
    pub fn has_sampled_textures(&self) -> bool {
        self.sampled_wind_uvw_t0.is_some() && self.sampled_wind_uvw_t1.is_some()
    }
}

/// GPU storage buffers for all gridded PBL fields.
pub struct PblBuffers {
    pub shape: (usize, usize),
    pub ustar: wgpu::Buffer,
    pub wstar: wgpu::Buffer,
    pub hmix: wgpu::Buffer,
    pub oli: wgpu::Buffer,
    pub sshf: wgpu::Buffer,
    pub ssr: wgpu::Buffer,
    pub surfstr: wgpu::Buffer,
    cell_count: usize,
}

impl PblBuffers {
    pub fn from_state(ctx: &GpuContext, pbl: &PblState) -> Result<Self, GpuBufferError> {
        let host = PblHostData::try_from(pbl)?;
        Ok(Self::from_host_data(ctx, &host))
    }

    #[must_use]
    pub fn from_host_data(ctx: &GpuContext, host: &PblHostData) -> Self {
        Self {
            shape: host.shape,
            ustar: create_storage_buffer_from_pod(&ctx.device, "pbl_ustar", &host.ustar),
            wstar: create_storage_buffer_from_pod(&ctx.device, "pbl_wstar", &host.wstar),
            hmix: create_storage_buffer_from_pod(&ctx.device, "pbl_hmix", &host.hmix),
            oli: create_storage_buffer_from_pod(&ctx.device, "pbl_oli", &host.oli),
            sshf: create_storage_buffer_from_pod(&ctx.device, "pbl_sshf", &host.sshf),
            ssr: create_storage_buffer_from_pod(&ctx.device, "pbl_ssr", &host.ssr),
            surfstr: create_storage_buffer_from_pod(&ctx.device, "pbl_surfstr", &host.surfstr),
            cell_count: host.cell_count(),
        }
    }

    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.cell_count
    }

    pub fn upload_state(&self, ctx: &GpuContext, pbl: &PblState) -> Result<(), GpuBufferError> {
        let host = PblHostData::try_from(pbl)?;
        if host.shape != self.shape {
            return Err(GpuBufferError::ShapeMismatch {
                field: "pbl_2d",
                expected: format!("{:?}", self.shape),
                actual: format!("{:?}", host.shape),
            });
        }

        write_exact_pod_slice(
            &ctx.queue,
            &self.ustar,
            "pbl_ustar",
            self.cell_count,
            &host.ustar,
        )?;
        write_exact_pod_slice(
            &ctx.queue,
            &self.wstar,
            "pbl_wstar",
            self.cell_count,
            &host.wstar,
        )?;
        write_exact_pod_slice(
            &ctx.queue,
            &self.hmix,
            "pbl_hmix",
            self.cell_count,
            &host.hmix,
        )?;
        write_exact_pod_slice(&ctx.queue, &self.oli, "pbl_oli", self.cell_count, &host.oli)?;
        write_exact_pod_slice(
            &ctx.queue,
            &self.sshf,
            "pbl_sshf",
            self.cell_count,
            &host.sshf,
        )?;
        write_exact_pod_slice(&ctx.queue, &self.ssr, "pbl_ssr", self.cell_count, &host.ssr)?;
        write_exact_pod_slice(
            &ctx.queue,
            &self.surfstr,
            "pbl_surfstr",
            self.cell_count,
            &host.surfstr,
        )?;
        Ok(())
    }

    pub async fn download_host_data(
        &self,
        ctx: &GpuContext,
    ) -> Result<PblHostData, GpuBufferError> {
        Ok(PblHostData {
            shape: self.shape,
            ustar: download_buffer_typed(ctx, &self.ustar, self.cell_count, "pbl_ustar").await?,
            wstar: download_buffer_typed(ctx, &self.wstar, self.cell_count, "pbl_wstar").await?,
            hmix: download_buffer_typed(ctx, &self.hmix, self.cell_count, "pbl_hmix").await?,
            oli: download_buffer_typed(ctx, &self.oli, self.cell_count, "pbl_oli").await?,
            sshf: download_buffer_typed(ctx, &self.sshf, self.cell_count, "pbl_sshf").await?,
            ssr: download_buffer_typed(ctx, &self.ssr, self.cell_count, "pbl_ssr").await?,
            surfstr: download_buffer_typed(ctx, &self.surfstr, self.cell_count, "pbl_surfstr")
                .await?,
        })
    }

    pub async fn download_state(&self, ctx: &GpuContext) -> Result<PblState, GpuBufferError> {
        self.download_host_data(ctx).await?.to_pbl_state()
    }
}

/// Unified manager bundling all baseline simulation buffers.
pub struct GpuBufferManager {
    pub particles: ParticleBuffers,
    pub wind: WindBuffers,
    pub pbl: PblBuffers,
}

impl GpuBufferManager {
    pub fn new(
        ctx: &GpuContext,
        particles: &ParticleStore,
        wind: &WindField3D,
        pbl: &PblState,
    ) -> Result<Self, GpuBufferError> {
        Ok(Self {
            particles: ParticleBuffers::from_store(ctx, particles),
            wind: WindBuffers::from_field(ctx, wind)?,
            pbl: PblBuffers::from_state(ctx, pbl)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::gpu::GpuError;
    use crate::particles::{ParticleInit, MAX_SPECIES};

    #[test]
    fn wind_host_data_flatten_order_is_stable() {
        let mut wind = WindField3D::zeros(2, 2, 2);
        wind.u_ms = Array3::from_shape_vec((2, 2, 2), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .expect("shape");
        wind.v_ms = wind.u_ms.clone();
        wind.w_ms = wind.u_ms.clone();
        wind.temperature_k = wind.u_ms.clone();
        wind.specific_humidity = wind.u_ms.clone();
        wind.pressure_pa = wind.u_ms.clone();
        wind.air_density_kg_m3 = wind.u_ms.clone();
        wind.density_gradient_kg_m2 = wind.u_ms.clone();

        let host = WindHostData::try_from(&wind).expect("valid shapes");
        assert_eq!(host.shape, (2, 2, 2));
        assert_eq!(host.u_ms, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(host.bytes_per_field(), 8 * size_of::<f32>());
    }

    #[test]
    fn pbl_host_data_flatten_order_is_stable() {
        let mut pbl = PblState::new(2, 3);
        pbl.ustar = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        pbl.wstar = pbl.ustar.clone();
        pbl.hmix = pbl.ustar.clone();
        pbl.oli = pbl.ustar.clone();
        pbl.sshf = pbl.ustar.clone();
        pbl.ssr = pbl.ustar.clone();
        pbl.surfstr = pbl.ustar.clone();

        let host = PblHostData::try_from(&pbl).expect("valid shapes");
        assert_eq!(host.shape, (2, 3));
        assert_eq!(host.ustar, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(host.bytes_per_field(), 6 * size_of::<f32>());
    }

    #[test]
    fn particle_buffer_byte_size_matches_struct_size() {
        let store = ParticleStore::with_capacity(16);
        let bytes =
            checked_byte_len::<Particle>(store.capacity(), "particles").expect("no overflow");
        assert_eq!(bytes, 16 * Particle::GPU_SIZE);
    }

    #[test]
    fn gpu_particle_roundtrip_skips_without_adapter() {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => return,
            Err(e) => panic!("unexpected GPU init error: {e}"),
        };

        let mut store = ParticleStore::with_capacity(2);
        let p0 = Particle::new(&ParticleInit {
            cell_x: 1,
            cell_y: 2,
            pos_x: 0.25,
            pos_y: 0.5,
            pos_z: 100.0,
            mass: [1.0, 2.0, 0.0, 0.0],
            release_point: 3,
            class: 4,
            time: 10,
        });
        let p1 = Particle::new(&ParticleInit {
            cell_x: -5,
            cell_y: 8,
            pos_x: 0.75,
            pos_y: 0.125,
            pos_z: 250.0,
            mass: [3.0, 0.0, 0.0, 0.0],
            release_point: 1,
            class: 2,
            time: 20,
        });
        store.add(p0).expect("slot 0 available");
        store.add(p1).expect("slot 1 available");
        assert_eq!(MAX_SPECIES, 4);

        let buffers = ParticleBuffers::from_store(&ctx, &store);
        let downloaded =
            pollster::block_on(buffers.download_particles(&ctx)).expect("readback succeeds");

        assert_eq!(downloaded.len(), 2);
        assert_eq!(downloaded[0].cell_x, 1);
        assert_eq!(downloaded[1].cell_y, 8);
        assert!((downloaded[0].mass[1] - 2.0).abs() < f32::EPSILON);
    }
}
