//! GPU runtime preflight checks for container/dev/CI environments.
//!
//! This module validates that wgpu can discover an adapter, request a device,
//! and execute a tiny compute workload.

use std::sync::mpsc;

use thiserror::Error;
use wgpu::util::DeviceExt;

const SMOKE_SENTINEL_VALUE: u32 = 0x00C0_FFEE;

/// Runtime options for the GPU preflight probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuPreflightOptions {
    /// Optional backend override (`vulkan`, `metal`, `dx12`, `gl`, `webgpu`, or `auto`).
    pub backend_override: Option<String>,
    /// If true, run the tiny compute smoke test.
    pub run_smoke_test: bool,
}

impl Default for GpuPreflightOptions {
    fn default() -> Self {
        Self {
            backend_override: None,
            run_smoke_test: true,
        }
    }
}

/// Compact subset of useful `wgpu::Limits` values for diagnostics.
#[derive(Debug, Clone)]
pub struct DeviceLimitsSummary {
    pub max_bind_groups: u32,
    pub max_storage_buffers_per_shader_stage: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_buffer_size: u64,
}

impl From<wgpu::Limits> for DeviceLimitsSummary {
    fn from(value: wgpu::Limits) -> Self {
        Self {
            max_bind_groups: value.max_bind_groups,
            max_storage_buffers_per_shader_stage: value.max_storage_buffers_per_shader_stage,
            max_compute_invocations_per_workgroup: value.max_compute_invocations_per_workgroup,
            max_compute_workgroup_size_x: value.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: value.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: value.max_compute_workgroup_size_z,
            max_compute_workgroups_per_dimension: value.max_compute_workgroups_per_dimension,
            max_buffer_size: value.max_buffer_size,
        }
    }
}

/// Structured report produced by the preflight run.
#[derive(Debug, Clone)]
pub struct GpuPreflightReport {
    pub requested_backend: String,
    pub adapter_name: String,
    pub adapter_backend: wgpu::Backend,
    pub adapter_type: wgpu::DeviceType,
    pub vendor_id: u32,
    pub device_id: u32,
    pub driver: String,
    pub driver_info: String,
    pub limits: DeviceLimitsSummary,
    pub supports_wind_texture_sampling: bool,
    pub smoke_test_value: Option<u32>,
}

#[derive(Debug, Error)]
pub enum GpuPreflightError {
    #[error("invalid backend selector `{selector}`: {reason}")]
    InvalidBackendSelector { selector: String, reason: String },
    #[error("no suitable GPU adapter found (requested backend: {requested_backend})")]
    NoAdapter { requested_backend: String },
    #[error("failed to request GPU device: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),
    #[error("buffer map operation failed: {0}")]
    BufferMap(#[from] wgpu::BufferAsyncError),
    #[error("buffer map callback channel closed unexpectedly")]
    MapChannelClosed,
    #[error("compute smoke test mismatch: expected 0x{expected:08x}, got 0x{actual:08x}")]
    SmokeTestMismatch { expected: u32, actual: u32 },
}

/// Parse and normalize a backend selector (`auto` or comma-separated backend list).
pub fn normalize_backend_selector(selector: &str) -> Result<String, GpuPreflightError> {
    let trimmed = selector.trim().to_lowercase();
    if trimmed.is_empty() {
        return Err(GpuPreflightError::InvalidBackendSelector {
            selector: selector.to_string(),
            reason: "value cannot be empty".to_string(),
        });
    }

    if trimmed == "auto" || trimmed == "all" {
        return Ok("auto".to_string());
    }

    let mut normalized_tokens: Vec<String> = Vec::new();
    for token in trimmed.split(',') {
        let token = token.trim();
        let canonical = match token {
            "vulkan" => "vulkan",
            "metal" => "metal",
            "dx12" | "d3d12" => "dx12",
            "gl" | "opengl" => "gl",
            "webgpu" | "browser_webgpu" => "webgpu",
            "" => {
                return Err(GpuPreflightError::InvalidBackendSelector {
                    selector: selector.to_string(),
                    reason: "contains an empty backend token".to_string(),
                });
            }
            _ => {
                return Err(GpuPreflightError::InvalidBackendSelector {
                    selector: selector.to_string(),
                    reason: format!(
                        "unsupported backend `{token}` (allowed: auto,vulkan,metal,dx12,gl,webgpu)"
                    ),
                });
            }
        };

        if !normalized_tokens.iter().any(|value| value == canonical) {
            normalized_tokens.push(canonical.to_string());
        }
    }

    if normalized_tokens.is_empty() {
        return Err(GpuPreflightError::InvalidBackendSelector {
            selector: selector.to_string(),
            reason: "no usable backend tokens".to_string(),
        });
    }

    Ok(normalized_tokens.join(","))
}

fn resolve_requested_backend(options: &GpuPreflightOptions) -> Result<String, GpuPreflightError> {
    if let Some(cli_value) = options.backend_override.as_deref() {
        return normalize_backend_selector(cli_value);
    }
    if let Ok(env_value) = std::env::var("WGPU_BACKEND") {
        return normalize_backend_selector(&env_value);
    }
    Ok("auto".to_string())
}

/// Run GPU preflight:
/// 1. discover adapter and create device via wgpu
/// 2. collect adapter/device diagnostics
/// 3. run tiny compute smoke test (optional)
pub async fn run_preflight(
    options: GpuPreflightOptions,
) -> Result<GpuPreflightReport, GpuPreflightError> {
    let requested_backend = resolve_requested_backend(&options)?;
    if options.backend_override.is_some() {
        std::env::set_var("WGPU_BACKEND", &requested_backend);
    }

    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or_else(|| GpuPreflightError::NoAdapter {
            requested_backend: requested_backend.clone(),
        })?;

    let info = adapter.get_info();
    let optional_features = wgpu::Features::FLOAT32_FILTERABLE;
    let requested_features = adapter.features() & optional_features;
    let mut required_limits = wgpu::Limits::default();
    required_limits.max_buffer_size = adapter.limits().max_buffer_size;
    required_limits.max_storage_buffer_binding_size =
        adapter.limits().max_storage_buffer_binding_size;
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gpu-preflight-device"),
                required_features: requested_features,
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await?;

    let smoke_test_value = if options.run_smoke_test {
        Some(run_compute_smoke_test(&device, &queue).await?)
    } else {
        None
    };

    Ok(GpuPreflightReport {
        requested_backend,
        adapter_name: info.name,
        adapter_backend: info.backend,
        adapter_type: info.device_type,
        vendor_id: info.vendor,
        device_id: info.device,
        driver: info.driver,
        driver_info: info.driver_info,
        limits: DeviceLimitsSummary::from(device.limits()),
        supports_wind_texture_sampling: device
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE),
        smoke_test_value,
    })
}

async fn run_compute_smoke_test(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<u32, GpuPreflightError> {
    let output_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu_preflight_output_buffer"),
        contents: bytemuck::cast_slice(&[0_u32]),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_preflight_staging_buffer"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gpu_preflight_shader"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
@group(0) @binding(0)
var<storage, read_write> output_data: array<u32>;

@compute @workgroup_size(1)
fn main() {
    output_data[0] = 0x00C0FFEEu;
}
"#
            .into(),
        ),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gpu_preflight_bind_group_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu_preflight_bind_group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: output_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gpu_preflight_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gpu_preflight_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu_preflight_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_preflight_compute_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, 4);
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::Maintain::Wait);

    let slice = staging_buffer.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::Maintain::Wait);

    let map_result = rx.recv().map_err(|_| GpuPreflightError::MapChannelClosed)?;
    map_result?;

    let mapped = slice.get_mapped_range();
    let mut raw = [0_u8; 4];
    raw.copy_from_slice(&mapped[..4]);
    drop(mapped);
    staging_buffer.unmap();

    let value = u32::from_le_bytes(raw);
    if value != SMOKE_SENTINEL_VALUE {
        return Err(GpuPreflightError::SmokeTestMismatch {
            expected: SMOKE_SENTINEL_VALUE,
            actual: value,
        });
    }

    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_selector_normalizes_auto() {
        assert_eq!(
            normalize_backend_selector("  ALL ").expect("all is accepted"),
            "auto"
        );
        assert_eq!(
            normalize_backend_selector("auto").expect("auto is accepted"),
            "auto"
        );
    }

    #[test]
    fn test_backend_selector_normalizes_and_deduplicates() {
        assert_eq!(
            normalize_backend_selector("vulkan, gl, vulkan").expect("valid selector"),
            "vulkan,gl"
        );
    }

    #[test]
    fn test_backend_selector_rejects_unknown_token() {
        let error = normalize_backend_selector("cuda").expect_err("cuda is not a wgpu backend");
        let rendered = error.to_string();
        assert!(
            rendered.contains("unsupported backend"),
            "unexpected message: {rendered}"
        );
    }

    #[test]
    fn test_preflight_smoke_runs_if_adapter_available() {
        let instance = wgpu::Instance::default();
        let has_adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
            .is_some();

        if !has_adapter {
            return;
        }

        let report = pollster::block_on(run_preflight(GpuPreflightOptions::default()))
            .expect("preflight should succeed when adapter exists");
        assert!(!report.adapter_name.is_empty());
        assert_eq!(
            report.smoke_test_value,
            Some(SMOKE_SENTINEL_VALUE),
            "smoke test should write expected sentinel value"
        );
    }
}
