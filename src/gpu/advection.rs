//! GPU particle advection dispatch (A-04).
//!
//! This module runs a WGSL compute kernel that updates all particle slots and
//! advects only active particles (`flags & 1 != 0`) using the deterministic
//! Euler + Petterssen predictor/corrector scheme from the CPU reference
//! (`physics::advection`).
//!
//! Buffer contract:
//! - binding 0: `array<Particle>` read-write storage buffer (96-byte `repr(C)`
//!   layout from `particles::Particle`)
//! - binding 1: flattened `u` wind field (`f32[cell_count]`, read-only)
//! - binding 2: flattened `v` wind field (`f32[cell_count]`, read-only)
//! - binding 3: flattened `w` wind field (`f32[cell_count]`, read-only)
//! - binding 4: uniform parameters
//!   `(nx, ny, nz, particle_count, dt_seconds, x_scale, y_scale, z_scale)`

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::particles::Particle;
use crate::physics::VelocityToGridScale;

use super::{
    render_shader_with_workgroup_size, runtime_workgroup_size, GpuContext, ParticleBuffers,
    WindBuffers, WorkgroupKernel,
};

const SHADER_SOURCE_BUFFER: &str = include_str!("../shaders/advection.wgsl");
const SHADER_SOURCE_TEXTURE: &str = include_str!("../shaders/advection_texture.wgsl");
const TEXTURE_SAMPLING_ENV: &str = "FLEXPART_GPU_WIND_TEXTURE_SAMPLING";

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AdvectionParams {
    nx: u32,
    ny: u32,
    nz: u32,
    particle_count: u32,
    dt_seconds: f32,
    x_scale: f32,
    y_scale: f32,
    z_scale: f32,
    level_heights: [f32; 16],
}

/// Wind-field sampling backend used by the advection kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindSamplingPath {
    /// Existing storage-buffer path with explicit trilinear interpolation.
    BufferStorage,
    /// Sampled 3-D texture path using hardware linear filtering.
    SampledTexture3d,
}

/// Runtime options controlling wind sampling backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindSamplingOptions {
    /// Opt-in toggle for texture sampling (`false` keeps buffer path).
    pub enable_texture_sampling: bool,
}

impl WindSamplingOptions {
    /// Parse runtime opt-in from `FLEXPART_GPU_WIND_TEXTURE_SAMPLING`.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            enable_texture_sampling: parse_env_bool(TEXTURE_SAMPLING_ENV),
        }
    }
}

impl Default for WindSamplingOptions {
    fn default() -> Self {
        Self::from_env()
    }
}

/// Errors returned by GPU advection dispatch.
#[derive(Debug, Error)]
pub enum GpuAdvectionError {
    #[error("wind grid dimensions must be non-zero, got {shape:?}")]
    ZeroShape { shape: (usize, usize, usize) },
    #[error("value for {field} does not fit in u32: {value}")]
    ValueTooLarge { field: &'static str, value: usize },
    #[error("invalid dt_seconds for advection: {dt_seconds}")]
    InvalidTimeStep { dt_seconds: f32 },
    #[error("advection kernel sampling path mismatch: kernel={kernel:?}, runtime={runtime:?}")]
    KernelSamplingPathMismatch {
        kernel: WindSamplingPath,
        runtime: WindSamplingPath,
    },
}

fn usize_to_u32(value: usize, field: &'static str) -> Result<u32, GpuAdvectionError> {
    u32::try_from(value).map_err(|_| GpuAdvectionError::ValueTooLarge { field, value })
}

/// Reusable advection dispatch kernel for storage-buffer wind sampling.
pub struct AdvectionBufferDispatchKernel {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline: wgpu::ComputePipeline,
    workgroup_size_x: u32,
}

impl AdvectionBufferDispatchKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let workgroup_size_x = runtime_workgroup_size(ctx, WorkgroupKernel::Advection);
        let shader_source =
            render_shader_with_workgroup_size(SHADER_SOURCE_BUFFER, workgroup_size_x);
        let shader = ctx.load_shader("advection_shader_buffer", &shader_source);
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("advection_bgl"),
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
            "advection_pipeline",
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

/// Reusable advection dispatch kernel for sampled 3-D texture wind sampling.
pub struct AdvectionTextureDispatchKernel {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline: wgpu::ComputePipeline,
    pub sampler: wgpu::Sampler,
    workgroup_size_x: u32,
}

impl AdvectionTextureDispatchKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let workgroup_size_x = runtime_workgroup_size(ctx, WorkgroupKernel::Advection);
        let shader_source =
            render_shader_with_workgroup_size(SHADER_SOURCE_TEXTURE, workgroup_size_x);
        let shader = ctx.load_shader("advection_shader_texture", &shader_source);
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("advection_texture_bgl"),
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
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D3,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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
            "advection_texture_pipeline",
            &shader,
            "main",
            &[&bind_group_layout],
        );
        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("advection_texture_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        Self {
            bind_group_layout,
            pipeline,
            sampler,
            workgroup_size_x,
        }
    }
}

/// Reusable advection dispatch kernel variant by wind sampling path.
pub enum AdvectionDispatchKernel {
    Buffer(AdvectionBufferDispatchKernel),
    Texture(AdvectionTextureDispatchKernel),
}

impl AdvectionDispatchKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext, path: WindSamplingPath) -> Self {
        match path {
            WindSamplingPath::BufferStorage => {
                Self::Buffer(AdvectionBufferDispatchKernel::new(ctx))
            }
            WindSamplingPath::SampledTexture3d => {
                Self::Texture(AdvectionTextureDispatchKernel::new(ctx))
            }
        }
    }

    #[must_use]
    pub fn sampling_path(&self) -> WindSamplingPath {
        match self {
            Self::Buffer(_) => WindSamplingPath::BufferStorage,
            Self::Texture(_) => WindSamplingPath::SampledTexture3d,
        }
    }
}

/// Dispatch the WGSL advection kernel for all particle slots.
///
/// Active particles are advanced with Euler + Petterssen correction; inactive
/// particles are left unchanged.
///
/// Wind sampling path defaults to storage buffers. Set
/// `FLEXPART_GPU_WIND_TEXTURE_SAMPLING=1` to opt in to sampled-texture wind
/// lookup when runtime capabilities and wind texture resources are available.
/// Otherwise, this function falls back to the buffer path.
pub fn advect_particles_gpu(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    wind: &WindBuffers,
    dt_seconds: f32,
    velocity_scale: VelocityToGridScale,
) -> Result<(), GpuAdvectionError> {
    advect_particles_gpu_with_sampling(
        ctx,
        particles,
        wind,
        dt_seconds,
        velocity_scale,
        WindSamplingOptions::default(),
    )
}

/// Dispatch advection using an explicit wind sampling configuration.
pub fn advect_particles_gpu_with_sampling(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    wind: &WindBuffers,
    dt_seconds: f32,
    velocity_scale: VelocityToGridScale,
    options: WindSamplingOptions,
) -> Result<(), GpuAdvectionError> {
    let selected_path = resolve_wind_sampling_path(ctx, wind, options);
    let kernel = AdvectionDispatchKernel::new(ctx, selected_path);
    dispatch_advection_gpu_with_sampling_and_kernel(
        ctx,
        particles,
        wind,
        dt_seconds,
        velocity_scale,
        options,
        &kernel,
    )
}

/// Dispatch advection with a reusable prepared kernel (submits command buffer).
pub fn dispatch_advection_gpu_with_sampling_and_kernel(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    wind: &WindBuffers,
    dt_seconds: f32,
    velocity_scale: VelocityToGridScale,
    options: WindSamplingOptions,
    kernel: &AdvectionDispatchKernel,
) -> Result<(), GpuAdvectionError> {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("advection_encoder"),
        });
    encode_advection_gpu_with_kernel(
        ctx,
        particles,
        wind,
        dt_seconds,
        velocity_scale,
        options,
        kernel,
        &mut encoder,
    )?;
    ctx.queue.submit(Some(encoder.finish()));
    let _ = ctx.device.poll(wgpu::Maintain::Wait);
    Ok(())
}

/// Encode advection dispatch into a caller-provided command encoder.
pub fn encode_advection_gpu_with_kernel(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    wind: &WindBuffers,
    dt_seconds: f32,
    velocity_scale: VelocityToGridScale,
    options: WindSamplingOptions,
    kernel: &AdvectionDispatchKernel,
    encoder: &mut wgpu::CommandEncoder,
) -> Result<(), GpuAdvectionError> {
    if particles.particle_count() == 0 {
        return Ok(());
    }
    if !dt_seconds.is_finite() {
        return Err(GpuAdvectionError::InvalidTimeStep { dt_seconds });
    }

    let (nx, ny, nz) = wind.shape;
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(GpuAdvectionError::ZeroShape { shape: wind.shape });
    }

    debug_assert_eq!(Particle::GPU_SIZE, 96);

    let params = AdvectionParams {
        nx: usize_to_u32(nx, "nx")?,
        ny: usize_to_u32(ny, "ny")?,
        nz: usize_to_u32(nz, "nz")?,
        particle_count: usize_to_u32(particles.particle_count(), "particle_count")?,
        dt_seconds,
        x_scale: velocity_scale.x_grid_per_meter,
        y_scale: velocity_scale.y_grid_per_meter,
        z_scale: velocity_scale.z_grid_per_meter,
        level_heights: velocity_scale.level_heights_m,
    };
    let selected_path = resolve_wind_sampling_path(ctx, wind, options);
    if kernel.sampling_path() != selected_path {
        return Err(GpuAdvectionError::KernelSamplingPathMismatch {
            kernel: kernel.sampling_path(),
            runtime: selected_path,
        });
    }

    match kernel {
        AdvectionDispatchKernel::Buffer(buffer_kernel) => {
            let params_buffer = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("advection_params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("advection_bg"),
                layout: &buffer_kernel.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: particles.particle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wind.u_ms.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wind.v_ms.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wind.w_ms.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("advection_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&buffer_kernel.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            super::dispatch_1d(&mut cpass, params.particle_count, buffer_kernel.workgroup_size_x);
        }
        AdvectionDispatchKernel::Texture(texture_kernel) => {
            let sampled = wind
                .sampled_wind_uvw
                .as_ref()
                .expect("selection guarantees sampled texture availability");
            let params_buffer = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("advection_texture_params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("advection_texture_bg"),
                layout: &texture_kernel.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: particles.particle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&sampled.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&texture_kernel.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("advection_texture_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&texture_kernel.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            super::dispatch_1d(&mut cpass, params.particle_count, texture_kernel.workgroup_size_x);
        }
    }
    Ok(())
}

#[must_use]
pub fn resolve_wind_sampling_path(
    ctx: &GpuContext,
    wind: &WindBuffers,
    options: WindSamplingOptions,
) -> WindSamplingPath {
    let texture_ready = ctx.supports_wind_texture_sampling() && wind.sampled_wind_uvw.is_some();
    if options.enable_texture_sampling && texture_ready {
        WindSamplingPath::SampledTexture3d
    } else {
        WindSamplingPath::BufferStorage
    }
}

fn parse_env_bool(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| parse_bool_flag(&value))
}

fn parse_bool_flag(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::gpu::GpuError;
    use crate::particles::{ParticleInit, MAX_SPECIES};
    use crate::physics::advect_particles_cpu;
    use crate::wind::{linear_shear_wind_field, uniform_wind_field, WindField3D, WindFieldGrid};
    use ndarray::Array1;

    fn deterministic_field(nx: usize, ny: usize, nz: usize) -> WindField3D {
        let mut field = WindField3D::zeros(nx, ny, nz);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = i as f32;
                    let y = j as f32;
                    let z = k as f32;
                    field.u_ms[[i, j, k]] = 0.6 * x * x - 0.2 * y + 0.3 * z + 0.1 * x * y;
                    field.v_ms[[i, j, k]] = -0.4 * x + 0.5 * y * y - 0.15 * z + 0.08 * y * z;
                    field.w_ms[[i, j, k]] = 0.2 * x - 0.1 * y + 0.25 * z * z - 0.03 * x * z;
                }
            }
        }
        field
    }

    fn particle_at(x: f32, y: f32, z: f32) -> Particle {
        let cell_x = x.floor() as i32;
        let cell_y = y.floor() as i32;
        Particle::new(&ParticleInit {
            cell_x,
            cell_y,
            pos_x: x - cell_x as f32,
            pos_y: y - cell_y as f32,
            pos_z: z,
            mass: [0.0; MAX_SPECIES],
            release_point: 0,
            class: 0,
            time: 0,
        })
    }

    fn sample_particles() -> Vec<Particle> {
        let mut inactive = particle_at(1.2, 2.2, 1.3);
        inactive.deactivate();
        inactive.vel_u = 11.0;
        inactive.vel_v = -7.5;
        inactive.vel_w = 2.25;

        vec![
            particle_at(0.1, 0.1, 0.2),
            particle_at(2.75, 1.25, 3.4),
            particle_at(5.8, 4.9, 1.1),
            particle_at(6.0, 5.0, 4.0),
            inactive,
        ]
    }

    fn uniform_field(nx: usize, ny: usize, nz: usize, u: f32, v: f32, w: f32) -> WindField3D {
        let grid = WindFieldGrid::new(
            nx,
            ny,
            nz,
            nz,
            nz,
            1.0,
            1.0,
            0.0,
            0.0,
            Array1::from_vec((0..nz).map(|k| k as f32).collect()),
        );
        uniform_wind_field(&grid, u, v, w)
    }

    fn linear_shear_field(
        nx: usize,
        ny: usize,
        nz: usize,
        u_reference_ms: f32,
        v_reference_ms: f32,
        w_reference_ms: f32,
        du_dz_per_s: f32,
        dv_dz_per_s: f32,
        dw_dz_per_s: f32,
        reference_height_m: f32,
    ) -> WindField3D {
        let grid = WindFieldGrid::new(
            nx,
            ny,
            nz,
            nz,
            nz,
            1.0,
            1.0,
            0.0,
            0.0,
            Array1::from_vec((0..nz).map(|k| k as f32).collect()),
        );
        linear_shear_wind_field(
            &grid,
            u_reference_ms,
            v_reference_ms,
            w_reference_ms,
            du_dz_per_s,
            dv_dz_per_s,
            dw_dz_per_s,
            reference_height_m,
        )
    }

    fn assert_particle_close(expected: &Particle, actual: &Particle) {
        assert_eq!(actual.cell_x, expected.cell_x);
        assert_eq!(actual.cell_y, expected.cell_y);
        assert_eq!(actual.flags, expected.flags);
        assert_eq!(actual.time, expected.time);
        assert_eq!(actual.timestep, expected.timestep);
        assert_eq!(actual.time_mem, expected.time_mem);
        assert_eq!(actual.time_split, expected.time_split);
        assert_eq!(actual.release_point, expected.release_point);
        assert_eq!(actual.class, expected.class);
        assert_eq!(actual.cbt, expected.cbt);
        assert_eq!(actual.pad0, expected.pad0);

        assert_relative_eq!(actual.pos_x, expected.pos_x, epsilon = 1.0e-5);
        assert_relative_eq!(actual.pos_y, expected.pos_y, epsilon = 1.0e-5);
        assert_relative_eq!(actual.pos_z, expected.pos_z, epsilon = 1.0e-5);
        assert_relative_eq!(actual.vel_u, expected.vel_u, epsilon = 1.0e-5);
        assert_relative_eq!(actual.vel_v, expected.vel_v, epsilon = 1.0e-5);
        assert_relative_eq!(actual.vel_w, expected.vel_w, epsilon = 1.0e-5);
        assert_relative_eq!(actual.turb_u, expected.turb_u, epsilon = 1.0e-6);
        assert_relative_eq!(actual.turb_v, expected.turb_v, epsilon = 1.0e-6);
        assert_relative_eq!(actual.turb_w, expected.turb_w, epsilon = 1.0e-6);
        for (lhs, rhs) in actual.mass.iter().zip(expected.mass.iter()) {
            assert_relative_eq!(*lhs, *rhs, epsilon = 1.0e-6);
        }
    }

    #[test]
    fn gpu_advection_uniform_wind_matches_analytical_displacement() {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => {
                eprintln!("No GPU adapter found — skipping uniform-wind analytical advection test");
                return;
            }
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        // A-07 assumption: advection currently uses grid-coordinate space.
        // With IDENTITY scaling, analytical motion is x=x0+u*t, y=y0+v*t, z=z0+w*t.
        let dt = 3.0_f32;
        let u = 0.7_f32;
        let v = -0.2_f32;
        let w = 0.1_f32;
        let tolerance = 2.0e-5_f64;

        let field = uniform_field(48, 48, 48, u, v, w);
        let wind_buffers = WindBuffers::from_field(&ctx, &field).expect("wind upload succeeds");
        let particles = vec![
            particle_at(4.2, 7.4, 5.0),
            particle_at(10.5, 12.25, 8.6),
            particle_at(19.875, 16.125, 10.4),
            particle_at(27.0, 3.5, 2.0),
        ];
        let initial_coords: Vec<(f64, f64, f64)> = particles
            .iter()
            .map(|p| (p.grid_x(), p.grid_y(), f64::from(p.pos_z)))
            .collect();

        let particle_buffers = ParticleBuffers::from_particles(&ctx, &particles);
        advect_particles_gpu(
            &ctx,
            &particle_buffers,
            &wind_buffers,
            dt,
            VelocityToGridScale::IDENTITY,
        )
        .expect("gpu advection succeeds");

        let advected = pollster::block_on(particle_buffers.download_particles(&ctx))
            .expect("particle readback succeeds");
        assert_eq!(advected.len(), initial_coords.len());

        for (particle, (x0, y0, z0)) in advected.iter().zip(initial_coords.iter()) {
            assert_relative_eq!(
                particle.grid_x(),
                x0 + f64::from(u * dt),
                epsilon = tolerance
            );
            assert_relative_eq!(
                particle.grid_y(),
                y0 + f64::from(v * dt),
                epsilon = tolerance
            );
            assert_relative_eq!(
                f64::from(particle.pos_z),
                z0 + f64::from(w * dt),
                epsilon = tolerance
            );
            assert_relative_eq!(particle.vel_u, u, epsilon = 1.0e-6);
            assert_relative_eq!(particle.vel_v, v, epsilon = 1.0e-6);
            assert_relative_eq!(particle.vel_w, w, epsilon = 1.0e-6);
        }
    }

    #[test]
    fn gpu_advection_matches_cpu_reference_identity_scale() {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let field = deterministic_field(7, 6, 5);
        let wind_buffers = WindBuffers::from_field(&ctx, &field).expect("wind upload succeeds");

        let gpu_input = sample_particles();
        let mut expected = gpu_input.clone();
        advect_particles_cpu(&mut expected, &field, 0.75, VelocityToGridScale::IDENTITY);

        let particle_buffers = ParticleBuffers::from_particles(&ctx, &gpu_input);
        advect_particles_gpu(
            &ctx,
            &particle_buffers,
            &wind_buffers,
            0.75,
            VelocityToGridScale::IDENTITY,
        )
        .expect("gpu advection succeeds");

        let actual = pollster::block_on(particle_buffers.download_particles(&ctx))
            .expect("particle readback succeeds");
        assert_eq!(actual.len(), expected.len());
        for (cpu, gpu) in expected.iter().zip(actual.iter()) {
            assert_particle_close(cpu, gpu);
        }
    }

    #[test]
    fn gpu_advection_matches_cpu_reference_scaled_velocity() {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let field = deterministic_field(7, 6, 5);
        let wind_buffers = WindBuffers::from_field(&ctx, &field).expect("wind upload succeeds");

        let scale = VelocityToGridScale {
            x_grid_per_meter: 0.5,
            y_grid_per_meter: 0.25,
            z_grid_per_meter: 0.8,
            ..VelocityToGridScale::IDENTITY
        };
        let gpu_input = sample_particles();
        let mut expected = gpu_input.clone();
        advect_particles_cpu(&mut expected, &field, 1.25, scale);

        let particle_buffers = ParticleBuffers::from_particles(&ctx, &gpu_input);
        advect_particles_gpu(&ctx, &particle_buffers, &wind_buffers, 1.25, scale)
            .expect("gpu advection succeeds");

        let actual = pollster::block_on(particle_buffers.download_particles(&ctx))
            .expect("particle readback succeeds");
        assert_eq!(actual.len(), expected.len());
        for (cpu, gpu) in expected.iter().zip(actual.iter()) {
            assert_particle_close(cpu, gpu);
        }
    }

    #[test]
    fn gpu_advection_linear_shear_matches_cpu_reference() {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => {
                eprintln!("No GPU adapter found — skipping linear-shear parity test");
                return;
            }
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let field = linear_shear_field(
            64, 64, 64, 1.1,   // u at z_ref [m/s]
            -0.45, // v at z_ref [m/s]
            0.25,  // w at z_ref [m/s]
            0.05,  // du/dz [1/s]
            -0.02, // dv/dz [1/s]
            0.0,   // dw/dz [1/s]
            0.0,   // z_ref [m]
        );
        let wind_buffers = WindBuffers::from_field(&ctx, &field).expect("wind upload succeeds");

        let gpu_input = vec![
            particle_at(5.2, 4.1, 2.5),
            particle_at(7.9, 3.4, 4.7),
            particle_at(9.15, 8.2, 6.3),
            particle_at(2.4, 10.0, 1.8),
        ];
        let mut expected = gpu_input.clone();
        advect_particles_cpu(&mut expected, &field, 0.4, VelocityToGridScale::IDENTITY);

        let particle_buffers = ParticleBuffers::from_particles(&ctx, &gpu_input);
        advect_particles_gpu(
            &ctx,
            &particle_buffers,
            &wind_buffers,
            0.4,
            VelocityToGridScale::IDENTITY,
        )
        .expect("gpu advection succeeds");

        let actual = pollster::block_on(particle_buffers.download_particles(&ctx))
            .expect("particle readback succeeds");
        assert_eq!(actual.len(), expected.len());
        for (cpu, gpu) in expected.iter().zip(actual.iter()) {
            assert_particle_close(cpu, gpu);
        }
    }

    #[test]
    fn gpu_advection_texture_sampling_matches_buffer_sampling() {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };
        if !ctx.supports_wind_texture_sampling() {
            return;
        }

        let field = deterministic_field(7, 6, 5);
        let wind_buffers = WindBuffers::from_field(&ctx, &field).expect("wind upload succeeds");
        if wind_buffers.sampled_wind_uvw.is_none() {
            return;
        }

        let input = sample_particles();
        let buffer_particles = ParticleBuffers::from_particles(&ctx, &input);
        let texture_particles = ParticleBuffers::from_particles(&ctx, &input);
        let dt = 0.75;
        let scale = VelocityToGridScale::IDENTITY;

        advect_particles_gpu_with_sampling(
            &ctx,
            &buffer_particles,
            &wind_buffers,
            dt,
            scale,
            WindSamplingOptions {
                enable_texture_sampling: false,
            },
        )
        .expect("buffer-path advection succeeds");
        advect_particles_gpu_with_sampling(
            &ctx,
            &texture_particles,
            &wind_buffers,
            dt,
            scale,
            WindSamplingOptions {
                enable_texture_sampling: true,
            },
        )
        .expect("texture-path advection succeeds");

        let expected = pollster::block_on(buffer_particles.download_particles(&ctx))
            .expect("buffer-path readback succeeds");
        let actual = pollster::block_on(texture_particles.download_particles(&ctx))
            .expect("texture-path readback succeeds");
        assert_eq!(actual.len(), expected.len());
        // Hardware texture filtering can differ slightly from explicit buffer
        // interpolation due to implementation-dependent interpolation precision.
        let epsilon = 2.0e-2;
        for (buffer, texture) in expected.iter().zip(actual.iter()) {
            assert_eq!(texture.cell_x, buffer.cell_x);
            assert_eq!(texture.cell_y, buffer.cell_y);
            assert_eq!(texture.flags, buffer.flags);
            assert_relative_eq!(texture.pos_x, buffer.pos_x, epsilon = epsilon);
            assert_relative_eq!(texture.pos_y, buffer.pos_y, epsilon = epsilon);
            assert_relative_eq!(texture.pos_z, buffer.pos_z, epsilon = epsilon);
            assert_relative_eq!(texture.vel_u, buffer.vel_u, epsilon = epsilon);
            assert_relative_eq!(texture.vel_v, buffer.vel_v, epsilon = epsilon);
            assert_relative_eq!(texture.vel_w, buffer.vel_w, epsilon = epsilon);
        }
    }

    #[test]
    fn bool_feature_flag_parser_accepts_common_truthy_values() {
        assert!(parse_bool_flag("1"));
        assert!(parse_bool_flag("true"));
        assert!(parse_bool_flag("YES"));
        assert!(parse_bool_flag(" on "));
        assert!(!parse_bool_flag("0"));
        assert!(!parse_bool_flag("false"));
    }
}
