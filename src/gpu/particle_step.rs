//! GPU mega-kernel dispatch: complete particle physics step in a single dispatch.
//!
//! Fuses advection (Petterssen predictor-corrector with dual-wind temporal
//! interpolation), Hanna PBL turbulence, Langevin turbulent velocity update
//! (with sub-stepping + PBL reflection), dry deposition, and wet deposition
//! into one compute pass.
//!
//! This is the **default production path**. The separated multi-dispatch path
//! is used only when `FLEXPART_GPU_VALIDATION=1` is set for scientific
//! debugging.
//!
//! **Binding count constraint**: the mega-kernel requires 9 storage buffers per
//! shader stage (1 read-write + 8 read-only), exceeding the WebGPU default
//! minimum of 8. The [`GpuContext`] must request the adapter's actual
//! `max_storage_buffers_per_shader_stage` limit. All modern discrete GPUs
//! (NVIDIA, AMD, Intel) support ≥16; integrated/mobile GPUs typically ≥12.
//! If the adapter only supports 8, the driver will exit with an error message.

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::physics::{philox_counter_add, LangevinStep, PhiloxCounter, PhiloxKey, VelocityToGridScale};

use super::{
    render_shader_with_workgroup_size, runtime_workgroup_size, GpuContext,
    ParticleBuffers, PblBuffers, WorkgroupKernel,
};
use super::buffers::DualWindBuffers;

const SHADER_TEMPLATE: &str = include_str!("../shaders/particle_step.wgsl");

/// Minimum number of storage buffers per shader stage required by the mega-kernel.
const REQUIRED_STORAGE_BUFFERS: u32 = 9;

// ---------------------------------------------------------------------------
// Packed buffer helpers
// ---------------------------------------------------------------------------

/// PBL fields packed into a single GPU buffer (struct-of-arrays layout).
///
/// Layout: `ustar[nx*ny] ++ wstar[nx*ny] ++ hmix[nx*ny] ++ oli[nx*ny]`.
/// The shader indexes into each sub-array using the PBL grid dimensions
/// from the uniform params.
pub struct PackedPblBuffer {
    pub buffer: wgpu::Buffer,
    pub shape: (usize, usize),
}

impl PackedPblBuffer {
    /// Pre-allocate a persistent packed PBL buffer for the given grid shape.
    #[must_use]
    pub fn with_shape(ctx: &GpuContext, shape: (usize, usize)) -> Self {
        let cell_count = shape.0 * shape.1;
        let total_bytes = (cell_count * 4 * std::mem::size_of::<f32>()) as u64;
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("packed_pbl_persistent"),
            size: total_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { buffer, shape }
    }

    /// Pack PBL fields from host-side arrays into a single storage buffer.
    #[must_use]
    pub fn from_host_arrays(
        ctx: &GpuContext,
        shape: (usize, usize),
        ustar: &[f32],
        wstar: &[f32],
        hmix: &[f32],
        oli: &[f32],
    ) -> Self {
        let cell_count = shape.0 * shape.1;
        debug_assert_eq!(ustar.len(), cell_count);
        debug_assert_eq!(wstar.len(), cell_count);
        debug_assert_eq!(hmix.len(), cell_count);
        debug_assert_eq!(oli.len(), cell_count);

        let mut packed = Vec::with_capacity(cell_count * 4);
        packed.extend_from_slice(ustar);
        packed.extend_from_slice(wstar);
        packed.extend_from_slice(hmix);
        packed.extend_from_slice(oli);

        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("packed_pbl"),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE,
            });
        Self { buffer, shape }
    }

    /// Copy PBL fields from separate GPU buffers into this persistent packed
    /// buffer. Records copy commands into the encoder; no new buffer allocation.
    pub fn copy_from_pbl(&self, pbl: &PblBuffers, encoder: &mut wgpu::CommandEncoder) {
        let cell_count = pbl.cell_count();
        let component_bytes = (cell_count * std::mem::size_of::<f32>()) as u64;

        encoder.copy_buffer_to_buffer(&pbl.ustar, 0, &self.buffer, 0, component_bytes);
        encoder.copy_buffer_to_buffer(&pbl.wstar, 0, &self.buffer, component_bytes, component_bytes);
        encoder.copy_buffer_to_buffer(&pbl.hmix, 0, &self.buffer, 2 * component_bytes, component_bytes);
        encoder.copy_buffer_to_buffer(&pbl.oli, 0, &self.buffer, 3 * component_bytes, component_bytes);
    }

    /// Allocate-and-pack (legacy helper for standalone dispatch).
    #[must_use]
    pub fn pack_on_gpu(
        ctx: &GpuContext,
        pbl: &PblBuffers,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Self {
        let packed = Self::with_shape(ctx, pbl.shape);
        packed.copy_from_pbl(pbl, encoder);
        packed
    }
}

/// Per-particle deposition parameters packed into a single GPU buffer.
///
/// Layout: `vdep[n] ++ scav_coeff[n] ++ precip_frac[n]` where `n = particle_count`.
pub struct PackedDepositionBuffer {
    pub buffer: wgpu::Buffer,
    pub particle_count: usize,
}

impl PackedDepositionBuffer {
    /// Pre-allocate a persistent packed deposition buffer (zeroed).
    #[must_use]
    pub fn with_capacity(ctx: &GpuContext, particle_count: usize) -> Self {
        let total_bytes = particle_count * 3 * std::mem::size_of::<f32>();
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("packed_deposition_persistent"),
            size: total_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        buffer.slice(..).get_mapped_range_mut().fill(0);
        buffer.unmap();
        Self { buffer, particle_count }
    }

    /// Write deposition arrays into this persistent buffer via queue writes.
    pub fn write_arrays(
        &self,
        ctx: &GpuContext,
        deposition_velocity_m_s: &[f32],
        scavenging_coefficient_s_inv: &[f32],
        precipitating_fraction: &[f32],
    ) {
        let n = self.particle_count;
        debug_assert_eq!(deposition_velocity_m_s.len(), n);
        debug_assert_eq!(scavenging_coefficient_s_inv.len(), n);
        debug_assert_eq!(precipitating_fraction.len(), n);

        let stride = (n * std::mem::size_of::<f32>()) as u64;
        ctx.queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(deposition_velocity_m_s));
        ctx.queue.write_buffer(&self.buffer, stride, bytemuck::cast_slice(scavenging_coefficient_s_inv));
        ctx.queue.write_buffer(&self.buffer, 2 * stride, bytemuck::cast_slice(precipitating_fraction));
    }

    /// Write all zeros (no deposition) into this persistent buffer.
    pub fn write_zeros(&self, ctx: &GpuContext) {
        let total_bytes = self.particle_count * 3 * std::mem::size_of::<f32>();
        let zeros = vec![0u8; total_bytes];
        ctx.queue.write_buffer(&self.buffer, 0, &zeros);
    }

    /// Pack three per-particle deposition arrays into one storage buffer
    /// (allocating variant, for standalone use).
    #[must_use]
    pub fn from_arrays(
        ctx: &GpuContext,
        deposition_velocity_m_s: &[f32],
        scavenging_coefficient_s_inv: &[f32],
        precipitating_fraction: &[f32],
    ) -> Self {
        let particle_count = deposition_velocity_m_s.len();
        debug_assert_eq!(scavenging_coefficient_s_inv.len(), particle_count);
        debug_assert_eq!(precipitating_fraction.len(), particle_count);

        let mut packed = Vec::with_capacity(particle_count * 3);
        packed.extend_from_slice(deposition_velocity_m_s);
        packed.extend_from_slice(scavenging_coefficient_s_inv);
        packed.extend_from_slice(precipitating_fraction);

        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("packed_deposition"),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE,
            });
        Self {
            buffer,
            particle_count,
        }
    }

    /// Create a zero-deposition buffer (no mass loss for any particle).
    #[must_use]
    pub fn zeros(ctx: &GpuContext, particle_count: usize) -> Self {
        let packed = vec![0.0f32; particle_count * 3];
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("packed_deposition_zeros"),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE,
            });
        Self {
            buffer,
            particle_count,
        }
    }
}

// ---------------------------------------------------------------------------
// Input parameters
// ---------------------------------------------------------------------------

/// Combined input parameters for the mega-kernel particle step.
///
/// Groups advection, Langevin, and deposition parameters that are passed
/// as the GPU uniform buffer.
pub struct ParticleStepInput {
    /// Timestep duration [s] (shared by advection, Langevin, and deposition).
    pub dt_seconds: f32,
    /// Temporal interpolation weight for dual-wind fields: 0.0 = fully t0, 1.0 = fully t1.
    pub alpha: f32,
    /// Grid-scale conversion factors and level heights.
    pub scale: VelocityToGridScale,
    /// Langevin turbulence parameters (`n_substeps`, `rho_grad`, `min_height`).
    pub langevin: LangevinStep,
    /// Dry-deposition reference height `href` [m].
    pub reference_height_m: f32,
}

// ---------------------------------------------------------------------------
// GPU uniform struct (must match WGSL ParticleStepParams byte-for-byte)
// ---------------------------------------------------------------------------

/// GPU uniform for the `particle_step` mega-kernel.
///
/// 160 bytes total. The `level_heights` array starts at offset 96 (16-byte
/// aligned for `array<vec4<f32>, 4>` in WGSL).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ParticleStepParams {
    nx: u32,
    ny: u32,
    nz: u32,
    particle_count: u32,
    dt_seconds: f32,
    x_scale: f32,
    y_scale: f32,
    z_scale: f32,
    alpha: f32,
    key0: u32,
    key1: u32,
    counter0: u32,
    counter1: u32,
    counter2: u32,
    counter3: u32,
    n_substeps: u32,
    rho_grad_over_rho: f32,
    min_height_m: f32,
    pbl_nx: u32,
    pbl_ny: u32,
    reference_height_m: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    level_heights: [f32; 16],
}

// ---------------------------------------------------------------------------
// Dispatch kernel (pipeline + bind group layout)
// ---------------------------------------------------------------------------

/// Reusable mega-kernel dispatch kernel (bind group layout + compute pipeline).
///
/// Created once and reused for all dispatches within a simulation run.
pub struct ParticleStepDispatchKernel {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline: wgpu::ComputePipeline,
    workgroup_size_x: u32,
}

impl ParticleStepDispatchKernel {
    /// Create the mega-kernel pipeline.
    ///
    /// # Panics
    ///
    /// Panics (via wgpu validation) if the device was created with
    /// `max_storage_buffers_per_shader_stage < 9`. The caller should
    /// check [`supports_mega_kernel`] before constructing this kernel.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let workgroup_size_x = runtime_workgroup_size(ctx, WorkgroupKernel::ParticleStep);
        let shader_source = render_shader_with_workgroup_size(SHADER_TEMPLATE, workgroup_size_x);
        let shader = ctx.load_shader("particle_step_shader", &shader_source);

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
                    label: Some("particle_step_bgl"),
                    entries: &[
                        // 0: particles (read_write)
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
                        storage_ro(1),  // wind_u_t0
                        storage_ro(2),  // wind_v_t0
                        storage_ro(3),  // wind_w_t0
                        storage_ro(4),  // wind_u_t1
                        storage_ro(5),  // wind_v_t1
                        storage_ro(6),  // wind_w_t1
                        storage_ro(7),  // packed_pbl
                        // 8: uniform params
                        wgpu::BindGroupLayoutEntry {
                            binding: 8,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        storage_ro(9),  // packed_deposition
                    ],
                });

        let pipeline = ctx.create_compute_pipeline(
            "particle_step_pipeline",
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

/// Check whether the device supports the mega-kernel (≥ 9 storage buffers per stage).
#[must_use]
pub fn supports_mega_kernel(ctx: &GpuContext) -> bool {
    ctx.device.limits().max_storage_buffers_per_shader_stage >= REQUIRED_STORAGE_BUFFERS
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by the mega-kernel particle step dispatch.
#[derive(Debug, Error)]
pub enum GpuParticleStepError {
    #[error("invalid dt_seconds: {dt_seconds}")]
    InvalidTimeStep { dt_seconds: f32 },
    #[error("PBL grid dimensions must be non-zero, got {shape:?}")]
    ZeroPblShape { shape: (usize, usize) },
    #[error("wind grid dimensions must be non-zero, got {shape:?}")]
    ZeroWindShape { shape: (usize, usize, usize) },
    #[error("value for {field} does not fit in u32: {value}")]
    ValueTooLarge { field: &'static str, value: usize },
    #[error(
        "device supports only {actual} storage buffers per stage, mega-kernel requires {REQUIRED_STORAGE_BUFFERS}"
    )]
    InsufficientStorageBuffers { actual: u32 },
}

fn usize_to_u32(value: usize, field: &'static str) -> Result<u32, GpuParticleStepError> {
    u32::try_from(value).map_err(|_| GpuParticleStepError::ValueTooLarge { field, value })
}

// ---------------------------------------------------------------------------
// Encode helpers
// ---------------------------------------------------------------------------

fn build_params(
    input: &ParticleStepInput,
    particle_count: u32,
    wind_shape: (usize, usize, usize),
    pbl_shape: (usize, usize),
    key: PhiloxKey,
    base_counter: PhiloxCounter,
) -> Result<ParticleStepParams, GpuParticleStepError> {
    Ok(ParticleStepParams {
        nx: usize_to_u32(wind_shape.0, "nx")?,
        ny: usize_to_u32(wind_shape.1, "ny")?,
        nz: usize_to_u32(wind_shape.2, "nz")?,
        particle_count,
        dt_seconds: input.dt_seconds,
        x_scale: input.scale.x_grid_per_meter,
        y_scale: input.scale.y_grid_per_meter,
        z_scale: input.scale.z_grid_per_meter,
        alpha: input.alpha,
        key0: key[0],
        key1: key[1],
        counter0: base_counter[0],
        counter1: base_counter[1],
        counter2: base_counter[2],
        counter3: base_counter[3],
        n_substeps: input.langevin.n_substeps,
        rho_grad_over_rho: input.langevin.rho_grad_over_rho,
        min_height_m: input.langevin.min_height_m,
        pbl_nx: usize_to_u32(pbl_shape.0, "pbl_nx")?,
        pbl_ny: usize_to_u32(pbl_shape.1, "pbl_ny")?,
        reference_height_m: input.reference_height_m,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
        level_heights: input.scale.level_heights_m,
    })
}

#[allow(clippy::too_many_arguments)]
fn create_bind_group<'a>(
    ctx: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    particles: &'a ParticleBuffers,
    dual_wind: &'a DualWindBuffers,
    packed_pbl: &'a PackedPblBuffer,
    params_buffer: &'a wgpu::Buffer,
    packed_deposition: &'a PackedDepositionBuffer,
) -> wgpu::BindGroup {
    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("particle_step_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: particles.particle_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dual_wind.u_ms_t0.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dual_wind.v_ms_t0.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dual_wind.w_ms_t0.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: dual_wind.u_ms_t1.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: dual_wind.v_ms_t1.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: dual_wind.w_ms_t1.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: packed_pbl.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: packed_deposition.buffer.as_entire_binding() },
        ],
    })
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// Encode the mega-kernel particle step into a caller-provided command encoder.
///
/// Performs advection, Hanna turbulence, Langevin velocity update, dry
/// deposition, and wet deposition in a single GPU dispatch.
///
/// Returns the next Philox counter for deterministic RNG chaining.
///
/// # Errors
///
/// Returns errors for invalid time step, zero grid dimensions, or values
/// that don't fit in `u32`.
/// Persistent GPU resources for the mega-kernel, pre-allocated once to avoid
/// per-step buffer and bind group creation overhead.
pub struct ParticleStepResources {
    pub packed_pbl: PackedPblBuffer,
    pub packed_deposition: PackedDepositionBuffer,
    params_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    wind_shape: (usize, usize, usize),
}

impl ParticleStepResources {
    /// Pre-allocate all persistent mega-kernel resources and build the bind group.
    #[must_use]
    pub fn new(
        ctx: &GpuContext,
        particles: &ParticleBuffers,
        dual_wind: &DualWindBuffers,
        pbl_shape: (usize, usize),
        particle_capacity: usize,
        kernel: &ParticleStepDispatchKernel,
    ) -> Self {
        let packed_pbl = PackedPblBuffer::with_shape(ctx, pbl_shape);
        let packed_deposition = PackedDepositionBuffer::with_capacity(ctx, particle_capacity);

        let params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_step_params_persistent"),
            size: std::mem::size_of::<ParticleStepParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = create_bind_group(
            ctx,
            &kernel.bind_group_layout,
            particles,
            dual_wind,
            &packed_pbl,
            &params_buffer,
            &packed_deposition,
        );

        Self {
            packed_pbl,
            packed_deposition,
            params_buffer,
            bind_group,
            wind_shape: dual_wind.shape,
        }
    }
}

/// Encode the mega-kernel using persistent pre-allocated resources. The params
/// uniform is updated via `queue.write_buffer` and PBL data via encoder copies;
/// no buffer or bind group allocation occurs.
#[allow(clippy::too_many_arguments)]
pub fn encode_particle_step_gpu_persistent(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    resources: &ParticleStepResources,
    input: &ParticleStepInput,
    key: PhiloxKey,
    base_counter: PhiloxCounter,
    kernel: &ParticleStepDispatchKernel,
    encoder: &mut wgpu::CommandEncoder,
) -> Result<PhiloxCounter, GpuParticleStepError> {
    let particle_count = particles.particle_count();
    if particle_count == 0 {
        return Ok(base_counter);
    }
    if !input.dt_seconds.is_finite() || input.dt_seconds <= 0.0 {
        return Err(GpuParticleStepError::InvalidTimeStep {
            dt_seconds: input.dt_seconds,
        });
    }

    let pc_u32 = usize_to_u32(particle_count, "particle_count")?;
    let params = build_params(input, pc_u32, resources.wind_shape, resources.packed_pbl.shape, key, base_counter)?;
    ctx.queue.write_buffer(&resources.params_buffer, 0, bytemuck::bytes_of(&params));

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particle_step_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&kernel.pipeline);
        cpass.set_bind_group(0, &resources.bind_group, &[]);
        super::dispatch_1d(&mut cpass, pc_u32, kernel.workgroup_size_x);
    }

    let blocks_per_particle: u64 = if input.langevin.n_substeps <= 2 { 1 } else { 2 };
    Ok(philox_counter_add(
        base_counter,
        (particle_count as u64) * blocks_per_particle,
    ))
}

/// Encode the mega-kernel, allocating fresh buffers and bind group each call.
/// This is the original per-step allocating path; prefer
/// [`encode_particle_step_gpu_persistent`] in the time loop.
#[allow(clippy::too_many_arguments)]
pub fn encode_particle_step_gpu(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    dual_wind: &DualWindBuffers,
    packed_pbl: &PackedPblBuffer,
    packed_deposition: &PackedDepositionBuffer,
    input: &ParticleStepInput,
    key: PhiloxKey,
    base_counter: PhiloxCounter,
    kernel: &ParticleStepDispatchKernel,
    encoder: &mut wgpu::CommandEncoder,
) -> Result<PhiloxCounter, GpuParticleStepError> {
    let particle_count = particles.particle_count();
    if particle_count == 0 {
        return Ok(base_counter);
    }
    if !input.dt_seconds.is_finite() || input.dt_seconds <= 0.0 {
        return Err(GpuParticleStepError::InvalidTimeStep {
            dt_seconds: input.dt_seconds,
        });
    }
    if packed_pbl.shape.0 == 0 || packed_pbl.shape.1 == 0 {
        return Err(GpuParticleStepError::ZeroPblShape {
            shape: packed_pbl.shape,
        });
    }
    if dual_wind.shape.0 == 0 || dual_wind.shape.1 == 0 || dual_wind.shape.2 == 0 {
        return Err(GpuParticleStepError::ZeroWindShape {
            shape: dual_wind.shape,
        });
    }

    let pc_u32 = usize_to_u32(particle_count, "particle_count")?;
    let params = build_params(input, pc_u32, dual_wind.shape, packed_pbl.shape, key, base_counter)?;

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particle_step_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = create_bind_group(
        ctx, &kernel.bind_group_layout,
        particles, dual_wind, packed_pbl, &params_buffer, packed_deposition,
    );

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particle_step_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&kernel.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        super::dispatch_1d(&mut cpass, pc_u32, kernel.workgroup_size_x);
    }

    let blocks_per_particle: u64 = if input.langevin.n_substeps <= 2 { 1 } else { 2 };
    Ok(philox_counter_add(
        base_counter,
        (particle_count as u64) * blocks_per_particle,
    ))
}

/// Dispatch the mega-kernel as a standalone GPU submission (submit + wait).
///
/// Convenience wrapper around [`encode_particle_step_gpu`] that creates its
/// own command encoder, packs PBL buffers via GPU copy, submits, and polls
/// to completion.
///
/// # Errors
///
/// Propagates errors from [`encode_particle_step_gpu`].
#[allow(clippy::too_many_arguments)]
pub fn dispatch_particle_step_gpu(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    dual_wind: &DualWindBuffers,
    pbl: &PblBuffers,
    packed_deposition: &PackedDepositionBuffer,
    input: &ParticleStepInput,
    key: PhiloxKey,
    base_counter: PhiloxCounter,
) -> Result<PhiloxCounter, GpuParticleStepError> {
    let kernel = ParticleStepDispatchKernel::new(ctx);
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("particle_step_encoder"),
        });

    let packed_pbl = PackedPblBuffer::pack_on_gpu(ctx, pbl, &mut encoder);

    let next_counter = encode_particle_step_gpu(
        ctx,
        particles,
        dual_wind,
        &packed_pbl,
        packed_deposition,
        input,
        key,
        base_counter,
        &kernel,
        &mut encoder,
    )?;

    ctx.queue.submit(Some(encoder.finish()));
    let _ = ctx.device.poll(wgpu::Maintain::Wait);
    Ok(next_counter)
}
