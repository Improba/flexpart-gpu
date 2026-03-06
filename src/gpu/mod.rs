/// GPU device management and compute pipeline infrastructure using wgpu.
///
/// This module handles WebGPU device initialization, buffer management,
/// and compute shader dispatch. All GPU operations go through `GpuContext`.
use thiserror::Error;

pub mod advection;
pub mod buffers;
pub mod cbl;
pub mod convection;
pub mod deposition;
pub mod gridding;
pub mod hanna;
pub mod interpolation;
pub mod langevin;
pub mod pbl_reflection;
pub mod preflight;
pub mod rng;
pub mod wet_deposition;
pub mod workgroup;
pub use advection::{
    advect_particles_gpu, advect_particles_gpu_with_sampling,
    dispatch_advection_gpu_with_sampling_and_kernel, encode_advection_gpu_with_kernel,
    resolve_wind_sampling_path, AdvectionBufferDispatchKernel, AdvectionDispatchKernel,
    AdvectionTextureDispatchKernel, GpuAdvectionError, WindSamplingOptions, WindSamplingPath,
};
pub use buffers::{
    download_buffer_bytes, download_buffer_typed, GpuBufferError, GpuBufferManager,
    ParticleBuffers, PblBuffers, PblHostData, WindBuffers, WindHostData,
};
pub use cbl::{
    sample_cbl_vertical_velocity_gpu, sample_cbl_vertical_velocity_workflow, CblSamplingInput,
    CblSamplingOutput, GpuCblError, GpuCblWorkflowError,
};
pub use convection::{
    apply_convective_mixing_step_workflow, dispatch_convective_mixing_gpu, GpuConvectionError,
    GpuConvectionWorkflowError,
};
pub use deposition::{
    apply_dry_deposition_step_gpu, apply_dry_deposition_step_workflow,
    dispatch_dry_deposition_probability_gpu, dispatch_dry_deposition_probability_gpu_with_kernel,
    encode_dry_deposition_probability_gpu_with_kernel, DryDepositionDispatchKernel,
    DryDepositionIoBuffers, DryDepositionStepParams, GpuDryDepositionError,
    GpuDryDepositionWorkflowError,
};
pub use gridding::{
    accumulate_concentration_grid_gpu, dispatch_concentration_gridding_gpu,
    ConcentrationGridIoBuffers, ConcentrationGridOutput, ConcentrationGridShape,
    ConcentrationGriddingParams, GpuConcentrationGriddingError, MAX_OUTPUT_LEVELS,
};
pub use hanna::{
    compute_hanna_params_gpu, dispatch_hanna_params_gpu, dispatch_hanna_params_gpu_with_kernel,
    encode_hanna_params_gpu_with_kernel, GpuHannaError, HannaDispatchKernel,
    HannaParamsOutputBuffer,
};
pub use interpolation::{
    interpolate_wind_trilinear_gpu, GpuWindInterpolationError, WindInterpolationQuery,
};
pub use langevin::{
    encode_update_particles_turbulence_langevin_gpu_with_hanna_buffer_and_kernel,
    update_particles_turbulence_langevin_gpu,
    update_particles_turbulence_langevin_gpu_with_hanna_buffer,
    update_particles_turbulence_langevin_gpu_with_hanna_buffer_and_kernel, GpuLangevinError,
    LangevinDispatchKernel,
};
pub use pbl_reflection::{
    encode_pbl_reflection_gpu_with_kernel, GpuPblReflectionError, PblReflectionDispatchKernel,
};
pub use preflight::{
    normalize_backend_selector, run_preflight, DeviceLimitsSummary, GpuPreflightError,
    GpuPreflightOptions, GpuPreflightReport,
};
pub use rng::{sample_philox_uniform4_gpu, GpuPhiloxError, PhiloxUniformBlock};
pub use wet_deposition::{
    apply_wet_deposition_step_gpu, apply_wet_deposition_step_workflow,
    dispatch_wet_deposition_probability_gpu, dispatch_wet_deposition_probability_gpu_with_kernel,
    encode_wet_deposition_probability_gpu_with_kernel, GpuWetDepositionError,
    GpuWetDepositionWorkflowError, WetDepositionDispatchKernel, WetDepositionIoBuffers,
    WetDepositionStepParams,
};
pub use workgroup::{
    auto_tune_key_kernels, default_autotune_cache_path, render_shader_with_workgroup_size,
    runtime_workgroup_size, save_autotune_report_default, ScopedWorkgroupOverride,
    WorkgroupAutoTuneError, WorkgroupAutoTuneOptions, WorkgroupAutoTuneReport, WorkgroupKernel,
    WorkgroupSizeConfig,
};

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("no suitable GPU adapter found (try setting WGPU_BACKEND)")]
    NoAdapter,
    #[error("failed to request GPU device: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),
    #[error("shader compilation failed: {message}")]
    ShaderCompilation { message: String },
    #[error("buffer operation failed: {0}")]
    BufferMap(#[from] wgpu::BufferAsyncError),
}

/// Maximum workgroups per dimension (WebGPU spec minimum guarantee).
pub const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// Encode a 1-D dispatch that spills into the Y dimension when the number of
/// workgroups exceeds [`MAX_WORKGROUPS_PER_DIM`].
///
/// Shaders must compute their global index as:
/// ```text
/// let idx = gid.y * (nwg.x * WG_SIZE) + gid.x;
/// ```
/// where `nwg` comes from `@builtin(num_workgroups)`.
pub fn dispatch_1d(cpass: &mut wgpu::ComputePass<'_>, total_items: u32, workgroup_size: u32) {
    let groups = total_items.div_ceil(workgroup_size);
    if groups <= MAX_WORKGROUPS_PER_DIM {
        cpass.dispatch_workgroups(groups, 1, 1);
    } else {
        let groups_y = groups.div_ceil(MAX_WORKGROUPS_PER_DIM);
        cpass.dispatch_workgroups(MAX_WORKGROUPS_PER_DIM, groups_y, 1);
    }
}

/// Central GPU context holding the wgpu device, queue, and adapter info.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Initialize the GPU: request adapter, device, and queue.
    ///
    /// Respects `WGPU_BACKEND` env var for backend selection.
    /// Returns `GpuError::NoAdapter` if no suitable GPU is found.
    pub async fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        log::info!(
            "wgpu adapter: {} ({:?}, {:?})",
            adapter_info.name,
            adapter_info.backend,
            adapter_info.device_type
        );

        let optional_features = wgpu::Features::FLOAT32_FILTERABLE;
        let requested_features = adapter.features() & optional_features;
        let mut required_limits = wgpu::Limits::default();
        required_limits.max_buffer_size = adapter.limits().max_buffer_size;
        required_limits.max_storage_buffer_binding_size =
            adapter.limits().max_storage_buffer_binding_size;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("flexpart-gpu"),
                    required_features: requested_features,
                    required_limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;

        Ok(Self {
            device,
            queue,
            adapter_info,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.adapter_info.name
    }

    pub fn backend(&self) -> wgpu::Backend {
        self.adapter_info.backend
    }

    /// Returns true when a filterable float 3-D sampled texture path is usable.
    #[must_use]
    pub fn supports_wind_texture_sampling(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE)
    }

    /// Load a WGSL shader from source and create a `ShaderModule`.
    pub fn load_shader(&self, label: &str, source: &str) -> wgpu::ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
    }

    /// Create a compute pipeline from a shader module and entry point.
    pub fn create_compute_pipeline(
        &self,
        label: &str,
        shader: &wgpu::ShaderModule,
        entry_point: &str,
        bind_group_layouts: &[&wgpu::BindGroupLayout],
    ) -> wgpu::ComputePipeline {
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{label}_layout")),
                bind_group_layouts,
                push_constant_ranges: &[],
            });

        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_creation() {
        // Skip if no GPU available (CI environment)
        let result = pollster::block_on(GpuContext::new());
        match result {
            Ok(ctx) => {
                assert!(!ctx.device_name().is_empty());
                println!("GPU: {} ({:?})", ctx.device_name(), ctx.backend());
            }
            Err(GpuError::NoAdapter) => {
                eprintln!("No GPU adapter found — skipping test");
            }
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }
}
