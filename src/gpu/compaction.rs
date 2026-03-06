//! GPU active particle compaction via parallel prefix-sum (Tier 3.1).
//!
//! Stream compaction identifies active particles (bit 0 of `flags`) and builds
//! a contiguous index buffer mapping compacted positions to original particle
//! indices. This enables subsequent dispatches to operate only on active
//! particles, avoiding wasted GPU threads on deposited or domain-exited slots.
//!
//! ## Algorithm (four-pass GPU prefix sum + gather)
//!
//! 1. **Local prefix sum**: per-workgroup Hillis-Steele inclusive scan of
//!    active flags, producing per-thread exclusive prefixes and per-workgroup
//!    totals.
//! 2. **Scan workgroup sums**: single-workgroup chunked prefix sum over the
//!    workgroup totals, yielding global offsets. Scales to millions of
//!    particles (`WG_SIZE²` workgroups per single dispatch).
//! 3. **Scatter compact**: each active thread writes its original index to
//!    `compacted_indices[global_offset + local_prefix]`.
//! 4. **Gather reorder** *(optional)*: copies active particles into contiguous
//!    leading slots in a staging buffer, clears the tail, and copies back to
//!    the main particle buffer. Enables dispatch-count narrowing.

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::particles::Particle;

use super::{
    download_buffer_typed, render_shader_with_workgroup_size, GpuBufferError, GpuContext,
    ParticleBuffers,
};

const SHADER_TEMPLATE: &str = include_str!("../shaders/compaction.wgsl");

/// Fixed workgroup size for the compaction kernels.
///
/// The prefix-sum shared memory is sized to this value. Using 256 matches
/// common GPU warp/wavefront multiples and provides good occupancy.
const COMPACTION_WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CompactionParams {
    particle_count: u32,
    num_workgroups_pass1: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Errors from the GPU compaction pipeline.
#[derive(Debug, Error)]
pub enum GpuCompactionError {
    #[error("particle count {value} does not fit in u32")]
    ParticleCountOverflow { value: usize },
    #[error("byte-size overflow for {field}")]
    SizeOverflow { field: &'static str },
    #[error("buffer operation failed: {0}")]
    Buffer(#[from] GpuBufferError),
}

fn usize_to_u32(value: usize) -> Result<u32, GpuCompactionError> {
    u32::try_from(value).map_err(|_| GpuCompactionError::ParticleCountOverflow { value })
}

/// Compute the GPU buffer size in bytes for `len` elements of type `T`,
/// returning at least 4 bytes to avoid zero-sized buffer issues.
fn checked_buffer_size<T>(len: usize, field: &'static str) -> Result<u64, GpuCompactionError> {
    let bytes = len
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|b| u64::try_from(b).ok())
        .ok_or(GpuCompactionError::SizeOverflow { field })?;
    Ok(bytes.max(4))
}

fn bgl_storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Reusable GPU compute pipelines for the four compaction passes.
///
/// Build once at initialization and reuse across frames. All four passes
/// share the same bind group layout with seven entries (binding 6 is the
/// staging particle buffer used only by the gather/reorder pass).
pub struct CompactionPipelines {
    bind_group_layout: wgpu::BindGroupLayout,
    local_prefix_pipeline: wgpu::ComputePipeline,
    scan_workgroup_sums_pipeline: wgpu::ComputePipeline,
    scatter_compact_pipeline: wgpu::ComputePipeline,
    gather_reorder_pipeline: wgpu::ComputePipeline,
    workgroup_size_x: u32,
}

impl CompactionPipelines {
    /// Build shader module and compile the four compute pipelines.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let workgroup_size_x = COMPACTION_WORKGROUP_SIZE;
        let shader_source =
            render_shader_with_workgroup_size(SHADER_TEMPLATE, workgroup_size_x);
        let shader = ctx.load_shader("compaction_shader", &shader_source);

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("compaction_bgl"),
                    entries: &[
                        bgl_storage_entry(0, true),  // particles (read)
                        bgl_storage_entry(1, false), // local_prefixes
                        bgl_storage_entry(2, false), // workgroup_sums
                        bgl_storage_entry(3, false), // compacted_indices
                        bgl_storage_entry(4, false), // active_count_buf
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
                        bgl_storage_entry(6, false), // staging_particles
                    ],
                });

        let local_prefix_pipeline = ctx.create_compute_pipeline(
            "compaction_local_prefix_pipeline",
            &shader,
            "local_prefix_sum",
            &[&bind_group_layout],
        );
        let scan_workgroup_sums_pipeline = ctx.create_compute_pipeline(
            "compaction_scan_workgroup_pipeline",
            &shader,
            "scan_workgroup_sums",
            &[&bind_group_layout],
        );
        let scatter_compact_pipeline = ctx.create_compute_pipeline(
            "compaction_scatter_pipeline",
            &shader,
            "scatter_compact",
            &[&bind_group_layout],
        );
        let gather_reorder_pipeline = ctx.create_compute_pipeline(
            "compaction_gather_reorder_pipeline",
            &shader,
            "gather_reorder",
            &[&bind_group_layout],
        );

        Self {
            bind_group_layout,
            local_prefix_pipeline,
            scan_workgroup_sums_pipeline,
            scatter_compact_pipeline,
            gather_reorder_pipeline,
            workgroup_size_x,
        }
    }

    /// The workgroup size used by these pipelines.
    #[must_use]
    pub fn workgroup_size(&self) -> u32 {
        self.workgroup_size_x
    }
}

/// Pre-allocated auxiliary GPU buffers for the compaction algorithm.
///
/// Sized at creation time for `particle_capacity` particles and reused
/// across compaction invocations without per-step allocation. Includes a
/// staging particle buffer used by the gather/reorder pass to physically
/// pack active particles into contiguous leading slots.
pub struct CompactionBuffers {
    local_prefixes: wgpu::Buffer,
    workgroup_sums: wgpu::Buffer,
    /// Compacted index output: `compacted_indices[i]` is the original
    /// particle index for the i-th active particle.
    pub compacted_indices: wgpu::Buffer,
    active_count_buf: wgpu::Buffer,
    /// Staging buffer for the gather/reorder pass. Same layout and size as
    /// the main particle buffer. After the gather pass, the caller encodes
    /// a buffer-to-buffer copy back to the particle buffer.
    staging_particles: wgpu::Buffer,
    particle_capacity: usize,
}

impl CompactionBuffers {
    /// Allocate all compaction buffers for up to `particle_capacity` particles.
    ///
    /// # Errors
    ///
    /// Returns [`GpuCompactionError::ParticleCountOverflow`] if `particle_capacity`
    /// exceeds `u32::MAX`, or [`GpuCompactionError::SizeOverflow`] on byte-size overflow.
    pub fn new(ctx: &GpuContext, particle_capacity: usize) -> Result<Self, GpuCompactionError> {
        let capacity_u32 = usize_to_u32(particle_capacity)?;
        let num_workgroups = capacity_u32.div_ceil(COMPACTION_WORKGROUP_SIZE) as usize;

        let storage_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let local_prefixes = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compaction_local_prefixes"),
            size: checked_buffer_size::<u32>(particle_capacity, "local_prefixes")?,
            usage: storage_usage,
            mapped_at_creation: false,
        });

        let workgroup_sums = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compaction_workgroup_sums"),
            size: checked_buffer_size::<u32>(num_workgroups, "workgroup_sums")?,
            usage: storage_usage,
            mapped_at_creation: false,
        });

        let compacted_indices = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compaction_compacted_indices"),
            size: checked_buffer_size::<u32>(particle_capacity, "compacted_indices")?,
            usage: storage_usage,
            mapped_at_creation: false,
        });

        let active_count_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compaction_active_count"),
            size: 4,
            usage: storage_usage,
            mapped_at_creation: false,
        });

        let staging_particles = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compaction_staging_particles"),
            size: checked_buffer_size::<Particle>(particle_capacity, "staging_particles")?,
            usage: storage_usage,
            mapped_at_creation: false,
        });

        Ok(Self {
            local_prefixes,
            workgroup_sums,
            compacted_indices,
            active_count_buf,
            staging_particles,
            particle_capacity,
        })
    }

    /// Maximum particle count these buffers were allocated for.
    #[must_use]
    pub fn particle_capacity(&self) -> usize {
        self.particle_capacity
    }

    /// Download the compacted index buffer for the first `active_count` entries.
    ///
    /// # Errors
    ///
    /// Returns [`GpuBufferError`] on readback failure.
    pub async fn download_compacted_indices(
        &self,
        ctx: &GpuContext,
        active_count: u32,
    ) -> Result<Vec<u32>, GpuBufferError> {
        if active_count == 0 {
            return Ok(Vec::new());
        }
        download_buffer_typed::<u32>(
            ctx,
            &self.compacted_indices,
            active_count as usize,
            "compacted_indices",
        )
        .await
    }

    /// Download the active particle count written by pass 2.
    ///
    /// # Errors
    ///
    /// Returns [`GpuBufferError`] on readback failure.
    pub async fn download_active_count(
        &self,
        ctx: &GpuContext,
    ) -> Result<u32, GpuBufferError> {
        let data = download_buffer_typed::<u32>(
            ctx,
            &self.active_count_buf,
            1,
            "compaction_active_count",
        )
        .await?;
        Ok(data[0])
    }
}

/// Result of a GPU particle compaction.
pub struct CompactionResult {
    /// Number of active particles found.
    pub active_count: u32,
}

/// Run the full three-pass GPU compaction and read back the active count.
///
/// After this call, `buffers.compacted_indices` contains the mapping
/// from compacted index to original particle index for the first
/// `result.active_count` entries.
///
/// # Errors
///
/// Returns [`GpuCompactionError`] on dispatch or readback failure.
pub async fn compact_active_particles(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    buffers: &CompactionBuffers,
    pipelines: &CompactionPipelines,
) -> Result<CompactionResult, GpuCompactionError> {
    let particle_count = particles.particle_count();
    if particle_count == 0 {
        return Ok(CompactionResult { active_count: 0 });
    }

    encode_and_submit_compaction(ctx, particles, buffers, pipelines)?;

    let active_counts = download_buffer_typed::<u32>(
        ctx,
        &buffers.active_count_buf,
        1,
        "compaction_active_count",
    )
    .await?;

    Ok(CompactionResult {
        active_count: active_counts[0],
    })
}

/// Create the compaction bind group with all seven entries.
fn create_compaction_bind_group(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    buffers: &CompactionBuffers,
    pipelines: &CompactionPipelines,
    params_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compaction_bg"),
        layout: &pipelines.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: particles.particle_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffers.local_prefixes.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffers.workgroup_sums.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffers.compacted_indices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buffers.active_count_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: buffers.staging_particles.as_entire_binding(),
            },
        ],
    })
}

/// Encode all three compaction passes into a caller-provided command encoder.
///
/// After encoding, the caller must submit the encoder and read back
/// `buffers.active_count_buf[0]` to learn the active particle count.
///
/// # Errors
///
/// Returns [`GpuCompactionError::ParticleCountOverflow`] if the particle
/// count does not fit in `u32`.
pub fn encode_compaction(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    buffers: &CompactionBuffers,
    pipelines: &CompactionPipelines,
    encoder: &mut wgpu::CommandEncoder,
) -> Result<(), GpuCompactionError> {
    let particle_count = usize_to_u32(particles.particle_count())?;
    if particle_count == 0 {
        return Ok(());
    }

    let num_workgroups = particle_count.div_ceil(pipelines.workgroup_size_x);

    let params = CompactionParams {
        particle_count,
        num_workgroups_pass1: num_workgroups,
        _pad0: 0,
        _pad1: 0,
    };

    let params_buffer =
        ctx.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("compaction_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

    let bind_group =
        create_compaction_bind_group(ctx, particles, buffers, pipelines, &params_buffer);

    encode_three_passes(pipelines, &bind_group, particle_count, encoder);
    Ok(())
}

/// Encode all four compaction passes plus a buffer-to-buffer copy that
/// physically reorders the particle buffer so active particles occupy
/// contiguous leading slots.
///
/// Uses `particles.capacity()` for all four passes to ensure the gather
/// pass clears inactive tail slots. After submission, read back active
/// count via [`CompactionBuffers::download_active_count`].
///
/// # Errors
///
/// Returns [`GpuCompactionError::ParticleCountOverflow`] if the particle
/// capacity does not fit in `u32`.
pub fn encode_compaction_with_reorder(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    buffers: &CompactionBuffers,
    pipelines: &CompactionPipelines,
    encoder: &mut wgpu::CommandEncoder,
) -> Result<(), GpuCompactionError> {
    let capacity = usize_to_u32(particles.capacity())?;
    if capacity == 0 {
        return Ok(());
    }

    let num_workgroups = capacity.div_ceil(pipelines.workgroup_size_x);

    let params = CompactionParams {
        particle_count: capacity,
        num_workgroups_pass1: num_workgroups,
        _pad0: 0,
        _pad1: 0,
    };

    let params_buffer =
        ctx.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("compaction_reorder_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

    let bind_group =
        create_compaction_bind_group(ctx, particles, buffers, pipelines, &params_buffer);

    encode_three_passes(pipelines, &bind_group, capacity, encoder);

    // Pass 4: gather active particles into staging buffer, clear tail.
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compaction_pass4_gather_reorder"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipelines.gather_reorder_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        super::dispatch_1d(&mut cpass, capacity, pipelines.workgroup_size_x);
    }

    // Copy staging → particles. The full capacity is copied so that the
    // particle buffer reflects both the compacted active prefix and the
    // deactivated tail.
    let byte_size = u64::from(capacity) * std::mem::size_of::<Particle>() as u64;
    encoder.copy_buffer_to_buffer(
        &buffers.staging_particles,
        0,
        &particles.particle_buffer,
        0,
        byte_size,
    );

    Ok(())
}

fn encode_three_passes(
    pipelines: &CompactionPipelines,
    bind_group: &wgpu::BindGroup,
    particle_count: u32,
    encoder: &mut wgpu::CommandEncoder,
) {
    // Pass 1: local prefix sums + per-workgroup totals
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compaction_pass1_local_prefix"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipelines.local_prefix_pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        super::dispatch_1d(&mut cpass, particle_count, pipelines.workgroup_size_x);
    }

    // Pass 2: scan workgroup sums into global offsets (single workgroup)
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compaction_pass2_scan_sums"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipelines.scan_workgroup_sums_pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    // Pass 3: scatter active particle indices to compacted positions
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compaction_pass3_scatter"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipelines.scatter_compact_pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        super::dispatch_1d(&mut cpass, particle_count, pipelines.workgroup_size_x);
    }
}

fn encode_and_submit_compaction(
    ctx: &GpuContext,
    particles: &ParticleBuffers,
    buffers: &CompactionBuffers,
    pipelines: &CompactionPipelines,
) -> Result<(), GpuCompactionError> {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compaction_encoder"),
        });

    encode_compaction(ctx, particles, buffers, pipelines, &mut encoder)?;

    ctx.queue.submit(Some(encoder.finish()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuError;
    use crate::particles::{Particle, ParticleInit};

    fn make_test_particles(
        count: usize,
        inactive_pattern: impl Fn(usize) -> bool,
    ) -> Vec<Particle> {
        (0..count)
            .map(|i| {
                let mut p = Particle::new(&ParticleInit {
                    cell_x: i32::try_from(i).expect("test index fits i32"),
                    cell_y: 0,
                    pos_x: 0.0,
                    pos_y: 0.0,
                    pos_z: 100.0,
                    mass: [1.0, 0.0, 0.0, 0.0],
                    release_point: 0,
                    class: 0,
                    time: 0,
                });
                if inactive_pattern(i) {
                    p.deactivate();
                }
                p
            })
            .collect()
    }

    fn run_compaction_test(particles: &[Particle]) -> Option<(CompactionResult, Vec<u32>)> {
        let ctx = match pollster::block_on(GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter) => return None,
            Err(e) => panic!("unexpected GPU error: {e}"),
        };

        let particle_buffers = ParticleBuffers::from_particles(&ctx, particles);
        let pipelines = CompactionPipelines::new(&ctx);
        let buffers =
            CompactionBuffers::new(&ctx, particles.len()).expect("buffer creation succeeds");

        let result = pollster::block_on(compact_active_particles(
            &ctx,
            &particle_buffers,
            &buffers,
            &pipelines,
        ))
        .expect("compaction succeeds");

        let indices = pollster::block_on(
            buffers.download_compacted_indices(&ctx, result.active_count),
        )
        .expect("index download succeeds");

        Some((result, indices))
    }

    #[test]
    fn test_compaction_every_third_inactive() {
        let particles = make_test_particles(1000, |i| i % 3 == 0);
        let expected_active: Vec<usize> = (0..1000).filter(|i| i % 3 != 0).collect();

        let Some((result, indices)) = run_compaction_test(&particles) else {
            return;
        };

        assert_eq!(
            result.active_count as usize,
            expected_active.len(),
            "active_count should match expected"
        );

        for &idx in &indices {
            assert!(
                particles[idx as usize].is_active(),
                "compacted index {idx} should point to an active particle"
            );
        }

        let mut index_set: std::collections::HashSet<u32> = indices.iter().copied().collect();
        for &expected_idx in &expected_active {
            assert!(
                index_set.remove(&u32::try_from(expected_idx).unwrap()),
                "active particle at index {expected_idx} should appear in compacted output"
            );
        }
        assert!(index_set.is_empty(), "no extra indices should be present");
    }

    #[test]
    fn test_compaction_all_active() {
        let particles = make_test_particles(512, |_| false);

        let Some((result, indices)) = run_compaction_test(&particles) else {
            return;
        };

        assert_eq!(result.active_count, 512);
        for (compacted_idx, &original_idx) in indices.iter().enumerate() {
            assert_eq!(
                original_idx,
                u32::try_from(compacted_idx).unwrap(),
                "all-active should produce identity mapping"
            );
        }
    }

    #[test]
    fn test_compaction_all_inactive() {
        let particles = make_test_particles(256, |_| true);

        let Some((result, indices)) = run_compaction_test(&particles) else {
            return;
        };

        assert_eq!(result.active_count, 0);
        assert!(indices.is_empty());
    }

    #[test]
    fn test_compaction_single_active() {
        let particles = make_test_particles(100, |i| i != 42);

        let Some((result, indices)) = run_compaction_test(&particles) else {
            return;
        };

        assert_eq!(result.active_count, 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 42);
    }

    #[test]
    fn test_compaction_non_power_of_two_count() {
        let particles = make_test_particles(300, |i| i % 5 == 0);
        let expected_active_count = (0..300).filter(|i| i % 5 != 0).count();

        let Some((result, indices)) = run_compaction_test(&particles) else {
            return;
        };

        assert_eq!(result.active_count as usize, expected_active_count);
        assert_eq!(indices.len(), expected_active_count);
        for &idx in &indices {
            assert!(particles[idx as usize].is_active());
        }
    }

    #[test]
    fn test_compaction_multi_workgroup_coverage() {
        let particles = make_test_particles(1000, |i| i % 2 == 0);
        let expected_active_count = 500;

        let Some((result, indices)) = run_compaction_test(&particles) else {
            return;
        };

        assert_eq!(result.active_count, expected_active_count);
        assert_eq!(indices.len(), expected_active_count as usize);

        let mut seen = vec![false; 1000];
        for &idx in &indices {
            let i = idx as usize;
            assert!(particles[i].is_active());
            assert!(!seen[i], "duplicate index {i}");
            seen[i] = true;
        }

        for (i, particle) in particles.iter().enumerate() {
            if particle.is_active() {
                assert!(seen[i], "missing active particle {i}");
            }
        }
    }
}
