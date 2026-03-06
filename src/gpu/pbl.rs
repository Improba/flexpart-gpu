//! GPU PBL diagnostics dispatch (S-02 / Tier 2.1).
//!
//! Dispatches a WGSL compute kernel that computes gridded PBL parameters from
//! interpolated surface meteorological fields. Each workgroup item processes
//! one grid cell independently (embarrassingly parallel).
//!
//! Ported from `src/io/pbl_params.rs` (`compute_pbl_parameters_from_met`).
//! Reference: `calcpar.f90`, `obukhov.f90` (FLEXPART 10.4).
//!
//! The GPU output writes directly into [`PblBuffers`] so that downstream
//! shaders (Hanna, Langevin) consume it without any layout conversion.

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::io::pbl_params::PblComputationOptions;
use crate::wind::SurfaceFields;

use super::{
    render_shader_with_workgroup_size, runtime_workgroup_size, GpuContext, PblBuffers,
    WorkgroupKernel,
};

const SHADER_TEMPLATE: &str = include_str!("../shaders/pbl_diagnostics.wgsl");

// ---------------------------------------------------------------------------
// GPU-side data structures (must match WGSL layout exactly)
// ---------------------------------------------------------------------------

/// Packed surface cell input for the PBL diagnostics shader.
///
/// One entry per grid cell, 48 bytes (12 × f32). Must match the WGSL
/// `SurfaceCellInput` struct layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SurfaceCellInputGpu {
    surface_pressure_pa: f32,
    temperature_2m_k: f32,
    u10_ms: f32,
    v10_ms: f32,
    surface_stress_n_m2: f32,
    sensible_heat_flux_w_m2: f32,
    solar_radiation_w_m2: f32,
    mixing_height_m: f32,
    friction_velocity_ms: f32,
    inv_obukhov_length_per_m: f32,
    _pad0: f32,
    _pad1: f32,
}

const _: () = assert!(
    std::mem::size_of::<SurfaceCellInputGpu>() == 48,
    "SurfaceCellInputGpu must be 48 bytes to match WGSL layout"
);

/// Dispatch parameters uniform buffer.
///
/// 48 bytes (12 × f32-sized slots), must match WGSL `PblDiagnosticsParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PblDiagnosticsDispatchParams {
    grid_nx: u32,
    grid_ny: u32,
    cell_count: u32,
    _pad0: u32,
    roughness_length_m: f32,
    wind_reference_height_m: f32,
    heat_flux_neutral_threshold_w_m2: f32,
    _pad1: f32,
    hmix_min_m: f32,
    hmix_max_m: f32,
    fallback_mixing_height_m: f32,
    _pad2: f32,
}

const _: () = assert!(
    std::mem::size_of::<PblDiagnosticsDispatchParams>() == 48,
    "PblDiagnosticsDispatchParams must be 48 bytes (3 × vec4)"
);

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned while running the GPU PBL diagnostics kernel.
#[derive(Debug, Error)]
pub enum GpuPblDiagnosticsError {
    #[error("grid dimensions must be non-zero, got {shape:?}")]
    ZeroGridShape { shape: (usize, usize) },
    #[error("shape mismatch: surface buffer {surface:?} vs PBL output {pbl:?}")]
    ShapeMismatch {
        surface: (usize, usize),
        pbl: (usize, usize),
    },
    #[error("value for {field} does not fit in u32: {value}")]
    ValueTooLarge { field: &'static str, value: usize },
}

fn usize_to_u32(value: usize, field: &'static str) -> Result<u32, GpuPblDiagnosticsError> {
    u32::try_from(value).map_err(|_| GpuPblDiagnosticsError::ValueTooLarge { field, value })
}

// ---------------------------------------------------------------------------
// Surface field packing
// ---------------------------------------------------------------------------

fn pack_surface_fields(fields: &SurfaceFields) -> Vec<SurfaceCellInputGpu> {
    let shape = fields.surface_pressure_pa.shape();
    let (nx, ny) = (shape[0], shape[1]);
    let mut packed = Vec::with_capacity(nx * ny);
    for i in 0..nx {
        for j in 0..ny {
            packed.push(SurfaceCellInputGpu {
                surface_pressure_pa: fields.surface_pressure_pa[[i, j]],
                temperature_2m_k: fields.temperature_2m_k[[i, j]],
                u10_ms: fields.u10_ms[[i, j]],
                v10_ms: fields.v10_ms[[i, j]],
                surface_stress_n_m2: fields.surface_stress_n_m2[[i, j]],
                sensible_heat_flux_w_m2: fields.sensible_heat_flux_w_m2[[i, j]],
                solar_radiation_w_m2: fields.solar_radiation_w_m2[[i, j]],
                mixing_height_m: fields.mixing_height_m[[i, j]],
                friction_velocity_ms: fields.friction_velocity_ms[[i, j]],
                inv_obukhov_length_per_m: fields.inv_obukhov_length_per_m[[i, j]],
                _pad0: 0.0,
                _pad1: 0.0,
            });
        }
    }
    packed
}

// ---------------------------------------------------------------------------
// SurfaceFieldBuffer
// ---------------------------------------------------------------------------

/// GPU buffer holding packed surface meteorological fields for PBL computation.
///
/// Created from [`SurfaceFields`] on the CPU side. The packing interleaves all
/// 10 required surface variables into a single struct-of-arrays buffer to stay
/// within the WebGPU `maxStorageBuffersPerShaderStage` limit (8).
pub struct SurfaceFieldBuffer {
    buffer: wgpu::Buffer,
    shape: (usize, usize),
    cell_count: usize,
}

impl SurfaceFieldBuffer {
    /// Pre-allocate a GPU buffer for the given grid shape without uploading data.
    ///
    /// The buffer is zero-initialized and must be filled via [`upload`](Self::upload)
    /// before the first GPU dispatch that reads it.
    #[must_use]
    pub fn with_shape(ctx: &GpuContext, shape: (usize, usize)) -> Self {
        let cell_count = shape.0 * shape.1;
        let byte_len = cell_count * std::mem::size_of::<SurfaceCellInputGpu>();
        let size = if byte_len == 0 { 4 } else { byte_len as u64 };
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbl_surface_input"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            shape,
            cell_count,
        }
    }

    /// Create a GPU buffer from CPU-side surface fields.
    #[must_use]
    pub fn from_surface_fields(ctx: &GpuContext, fields: &SurfaceFields) -> Self {
        let packed = pack_surface_fields(fields);
        let shape_raw = fields.surface_pressure_pa.shape();
        let shape = (shape_raw[0], shape_raw[1]);
        let cell_count = shape.0 * shape.1;

        let buffer = if packed.is_empty() {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbl_surface_input"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        } else {
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("pbl_surface_input"),
                    contents: bytemuck::cast_slice(&packed),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                })
        };

        Self {
            buffer,
            shape,
            cell_count,
        }
    }

    /// Upload updated surface fields to the existing buffer.
    ///
    /// # Errors
    ///
    /// Returns [`GpuPblDiagnosticsError::ShapeMismatch`] if the new fields
    /// have a different grid shape than the buffer was created with.
    pub fn upload(
        &self,
        ctx: &GpuContext,
        fields: &SurfaceFields,
    ) -> Result<(), GpuPblDiagnosticsError> {
        let shape_raw = fields.surface_pressure_pa.shape();
        let new_shape = (shape_raw[0], shape_raw[1]);
        if new_shape != self.shape {
            return Err(GpuPblDiagnosticsError::ShapeMismatch {
                surface: new_shape,
                pbl: self.shape,
            });
        }
        let packed = pack_surface_fields(fields);
        if !packed.is_empty() {
            ctx.queue
                .write_buffer(&self.buffer, 0, bytemuck::cast_slice(&packed));
        }
        Ok(())
    }

    /// Grid shape `(nx, ny)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Total grid cell count.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.cell_count
    }
}

// ---------------------------------------------------------------------------
// PblDiagnosticsDispatchKernel
// ---------------------------------------------------------------------------

/// Reusable PBL diagnostics compute pipeline and bind group layout.
///
/// Create once at initialisation time and reuse across timesteps.
pub struct PblDiagnosticsDispatchKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    workgroup_size_x: u32,
}

impl PblDiagnosticsDispatchKernel {
    /// Build the shader module, bind group layout, and compute pipeline.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let workgroup_size_x = runtime_workgroup_size(ctx, WorkgroupKernel::PblDiagnostics);
        let shader_source = render_shader_with_workgroup_size(SHADER_TEMPLATE, workgroup_size_x);
        let shader = ctx.load_shader("pbl_diagnostics_shader", &shader_source);

        let make_storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("pbl_diagnostics_bgl"),
                    entries: &[
                        make_storage_entry(0, true),  // surface_input
                        make_storage_entry(1, false), // out_ustar
                        make_storage_entry(2, false), // out_wstar
                        make_storage_entry(3, false), // out_hmix
                        make_storage_entry(4, false), // out_oli
                        make_storage_entry(5, false), // out_sshf
                        make_storage_entry(6, false), // out_ssr
                        make_storage_entry(7, false), // out_surfstr
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
                    ],
                });

        let pipeline = ctx.create_compute_pipeline(
            "pbl_diagnostics_pipeline",
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

// ---------------------------------------------------------------------------
// Dispatch functions (layered API matching hanna.rs pattern)
// ---------------------------------------------------------------------------

/// Dispatch PBL diagnostics computation, writing results to existing [`PblBuffers`].
///
/// Creates a temporary kernel internally. For repeated dispatches, prefer
/// [`dispatch_pbl_diagnostics_gpu_with_kernel`] with a pre-built kernel.
///
/// # Errors
///
/// Returns an error if grid shapes are zero or mismatched.
pub fn dispatch_pbl_diagnostics_gpu(
    ctx: &GpuContext,
    surface: &SurfaceFieldBuffer,
    output: &PblBuffers,
    options: &PblComputationOptions,
) -> Result<(), GpuPblDiagnosticsError> {
    let kernel = PblDiagnosticsDispatchKernel::new(ctx);
    dispatch_pbl_diagnostics_gpu_with_kernel(ctx, surface, output, options, &kernel)
}

/// Dispatch PBL diagnostics using a pre-built kernel.
///
/// # Errors
///
/// Returns an error if grid shapes are zero or mismatched.
pub fn dispatch_pbl_diagnostics_gpu_with_kernel(
    ctx: &GpuContext,
    surface: &SurfaceFieldBuffer,
    output: &PblBuffers,
    options: &PblComputationOptions,
    kernel: &PblDiagnosticsDispatchKernel,
) -> Result<(), GpuPblDiagnosticsError> {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pbl_diagnostics_encoder"),
        });
    encode_pbl_diagnostics_gpu_with_kernel(ctx, surface, output, options, kernel, &mut encoder)?;
    ctx.queue.submit(Some(encoder.finish()));
    Ok(())
}

/// Encode PBL diagnostics dispatch into a caller-provided command encoder.
///
/// This is the lowest-level API, intended for batching multiple dispatches
/// into a single command buffer submission (e.g. PBL → Hanna → Langevin).
///
/// # Errors
///
/// Returns an error if grid shapes are zero or mismatched.
pub fn encode_pbl_diagnostics_gpu_with_kernel(
    ctx: &GpuContext,
    surface: &SurfaceFieldBuffer,
    output: &PblBuffers,
    options: &PblComputationOptions,
    kernel: &PblDiagnosticsDispatchKernel,
    encoder: &mut wgpu::CommandEncoder,
) -> Result<(), GpuPblDiagnosticsError> {
    let (nx, ny) = surface.shape;
    if nx == 0 || ny == 0 {
        return Err(GpuPblDiagnosticsError::ZeroGridShape { shape: (nx, ny) });
    }
    if output.shape != surface.shape {
        return Err(GpuPblDiagnosticsError::ShapeMismatch {
            surface: surface.shape,
            pbl: output.shape,
        });
    }

    let cell_count = surface.cell_count;

    let dispatch_params = PblDiagnosticsDispatchParams {
        grid_nx: usize_to_u32(nx, "grid_nx")?,
        grid_ny: usize_to_u32(ny, "grid_ny")?,
        cell_count: usize_to_u32(cell_count, "cell_count")?,
        _pad0: 0,
        roughness_length_m: options.roughness_length_m,
        wind_reference_height_m: options.wind_reference_height_m,
        heat_flux_neutral_threshold_w_m2: options.heat_flux_neutral_threshold_w_m2,
        _pad1: 0.0,
        hmix_min_m: options.hmix_min_m,
        hmix_max_m: options.hmix_max_m,
        fallback_mixing_height_m: options.fallback_mixing_height_m,
        _pad2: 0.0,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbl_diagnostics_dispatch_params"),
            contents: bytemuck::bytes_of(&dispatch_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pbl_diagnostics_bg"),
        layout: &kernel.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: surface.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.ustar.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.wstar.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output.hmix.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output.oli.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: output.sshf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: output.ssr.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: output.surfstr.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pbl_diagnostics_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&kernel.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        super::dispatch_1d(&mut cpass, dispatch_params.cell_count, kernel.workgroup_size_x);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::gpu::GpuError;
    use crate::io::pbl_params::{
        compute_pbl_parameters_from_met, PblComputationOptions, PblMetInputGrids,
    };
    use crate::pbl::PblState;
    use crate::wind::SurfaceFields;

    fn make_test_surface_fields() -> SurfaceFields {
        let mut surface = SurfaceFields::zeros(3, 2);

        // Cell (0,0): unstable — positive sensible heat flux, valid hmix
        surface.surface_pressure_pa[[0, 0]] = 101_325.0;
        surface.temperature_2m_k[[0, 0]] = 300.0;
        surface.u10_ms[[0, 0]] = 4.0;
        surface.v10_ms[[0, 0]] = 1.0;
        surface.surface_stress_n_m2[[0, 0]] = 0.4;
        surface.sensible_heat_flux_w_m2[[0, 0]] = 120.0;
        surface.solar_radiation_w_m2[[0, 0]] = 350.0;
        surface.mixing_height_m[[0, 0]] = 1500.0;

        // Cell (1,0): stable — negative sensible heat flux, low hmix
        surface.surface_pressure_pa[[1, 0]] = 100_900.0;
        surface.temperature_2m_k[[1, 0]] = 295.0;
        surface.u10_ms[[1, 0]] = 2.0;
        surface.v10_ms[[1, 0]] = 1.0;
        surface.surface_stress_n_m2[[1, 0]] = 0.2;
        surface.sensible_heat_flux_w_m2[[1, 0]] = -80.0;
        surface.solar_radiation_w_m2[[1, 0]] = 50.0;
        surface.mixing_height_m[[1, 0]] = 50.0;

        // Cell (2,0): neutral — zero heat flux, hmix at max
        surface.surface_pressure_pa[[2, 0]] = 101_100.0;
        surface.temperature_2m_k[[2, 0]] = 299.0;
        surface.u10_ms[[2, 0]] = 3.0;
        surface.v10_ms[[2, 0]] = 0.0;
        surface.sensible_heat_flux_w_m2[[2, 0]] = 0.0;
        surface.solar_radiation_w_m2[[2, 0]] = 100.0;
        surface.mixing_height_m[[2, 0]] = 6000.0;

        // Cell (0,1): fallback hmix (provided = 0)
        surface.surface_pressure_pa[[0, 1]] = 101_000.0;
        surface.temperature_2m_k[[0, 1]] = 301.0;
        surface.u10_ms[[0, 1]] = 1.0;
        surface.v10_ms[[0, 1]] = 1.0;
        surface.sensible_heat_flux_w_m2[[0, 1]] = 0.0;
        surface.solar_radiation_w_m2[[0, 1]] = 100.0;
        surface.mixing_height_m[[0, 1]] = 0.0;

        // Cell (1,1): met-provided u* and 1/L
        surface.surface_pressure_pa[[1, 1]] = 101_300.0;
        surface.temperature_2m_k[[1, 1]] = 298.0;
        surface.u10_ms[[1, 1]] = 5.0;
        surface.v10_ms[[1, 1]] = 2.0;
        surface.surface_stress_n_m2[[1, 1]] = 0.3;
        surface.sensible_heat_flux_w_m2[[1, 1]] = 50.0;
        surface.solar_radiation_w_m2[[1, 1]] = 200.0;
        surface.mixing_height_m[[1, 1]] = 1200.0;
        surface.friction_velocity_ms[[1, 1]] = 0.42;
        surface.inv_obukhov_length_per_m[[1, 1]] = -0.005;

        // Cell (2,1): zero wind (edge case for log-law fallback)
        surface.surface_pressure_pa[[2, 1]] = 101_000.0;
        surface.temperature_2m_k[[2, 1]] = 300.0;
        surface.u10_ms[[2, 1]] = 0.0;
        surface.v10_ms[[2, 1]] = 0.0;
        surface.sensible_heat_flux_w_m2[[2, 1]] = 10.0;
        surface.solar_radiation_w_m2[[2, 1]] = 50.0;
        surface.mixing_height_m[[2, 1]] = 500.0;

        surface
    }

    #[test]
    fn gpu_pbl_diagnostics_matches_cpu_reference() {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let surface = make_test_surface_fields();
        let options = PblComputationOptions::default();

        // CPU reference (without profile points, matching GPU path)
        let cpu_result = compute_pbl_parameters_from_met(
            PblMetInputGrids {
                surface: &surface,
                profile: None,
            },
            options,
        )
        .expect("CPU PBL computation succeeds");

        // GPU path
        let surface_buf = SurfaceFieldBuffer::from_surface_fields(&ctx, &surface);
        let pbl_output = PblBuffers::from_state(&ctx, &PblState::new(3, 2))
            .expect("PBL buffer creation succeeds");

        dispatch_pbl_diagnostics_gpu(&ctx, &surface_buf, &pbl_output, &options)
            .expect("GPU PBL dispatch succeeds");

        let gpu_pbl = pollster::block_on(pbl_output.download_state(&ctx))
            .expect("PBL readback succeeds");

        let (nx, ny) = (3_usize, 2_usize);
        for i in 0..nx {
            for j in 0..ny {
                let label = format!("cell ({i},{j})");
                assert_relative_eq!(
                    gpu_pbl.ustar[[i, j]],
                    cpu_result.pbl_state.ustar[[i, j]],
                    epsilon = 1.0e-4,
                    max_relative = 1.0e-4,
                );
                assert_relative_eq!(
                    gpu_pbl.wstar[[i, j]],
                    cpu_result.pbl_state.wstar[[i, j]],
                    epsilon = 1.0e-4,
                    max_relative = 1.0e-4,
                );
                assert_relative_eq!(
                    gpu_pbl.hmix[[i, j]],
                    cpu_result.pbl_state.hmix[[i, j]],
                    epsilon = 1.0e-4,
                    max_relative = 1.0e-4,
                );
                assert_relative_eq!(
                    gpu_pbl.oli[[i, j]],
                    cpu_result.pbl_state.oli[[i, j]],
                    epsilon = 1.0e-4,
                    max_relative = 1.0e-4,
                );
                assert_relative_eq!(
                    gpu_pbl.sshf[[i, j]],
                    cpu_result.pbl_state.sshf[[i, j]],
                    epsilon = 1.0e-4,
                    max_relative = 1.0e-4,
                );
                assert_relative_eq!(
                    gpu_pbl.ssr[[i, j]],
                    cpu_result.pbl_state.ssr[[i, j]],
                    epsilon = 1.0e-4,
                    max_relative = 1.0e-4,
                );
                assert_relative_eq!(
                    gpu_pbl.surfstr[[i, j]],
                    cpu_result.pbl_state.surfstr[[i, j]],
                    epsilon = 1.0e-4,
                    max_relative = 1.0e-4,
                );

                let _ = label;
            }
        }
    }

    #[test]
    fn surface_field_buffer_rejects_shape_mismatch_on_upload() {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let surface_3x2 = SurfaceFields::zeros(3, 2);
        let surface_2x2 = SurfaceFields::zeros(2, 2);

        let buf = SurfaceFieldBuffer::from_surface_fields(&ctx, &surface_3x2);
        let err = buf
            .upload(&ctx, &surface_2x2)
            .expect_err("shape mismatch must be rejected");

        assert!(matches!(
            err,
            GpuPblDiagnosticsError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn dispatch_rejects_shape_mismatch() {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let surface = SurfaceFields::zeros(3, 2);
        let pbl_wrong_shape = PblState::new(2, 2);

        let surface_buf = SurfaceFieldBuffer::from_surface_fields(&ctx, &surface);
        let pbl_buf = PblBuffers::from_state(&ctx, &pbl_wrong_shape)
            .expect("PBL buffer creation succeeds");

        let err = dispatch_pbl_diagnostics_gpu(
            &ctx,
            &surface_buf,
            &pbl_buf,
            &PblComputationOptions::default(),
        )
        .expect_err("shape mismatch must be rejected");

        assert!(matches!(
            err,
            GpuPblDiagnosticsError::ShapeMismatch { .. }
        ));
    }
}
