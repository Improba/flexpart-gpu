//! Particle data structures for the Lagrangian dispersion model.
//!
//! Ported from `com_mod.f90` (`xtra1`, `ytra1`, `ztra1`, `xmass1`, etc.)
//! Each particle carries position, velocity, mass per species, and metadata.
//!
//! The [`Particle`] struct uses `#[repr(C)]` layout and implements
//! `bytemuck::Pod + bytemuck::Zeroable` for direct GPU buffer upload.
//! All floats are `f32` — Monte Carlo convergence dominates over float
//! precision (see `AGENTS.md`, WGSL conventions).

use bytemuck::Zeroable;
use thiserror::Error;

/// Maximum number of chemical species tracked per particle.
///
/// Matches FLEXPART's `maxspec` but capped at 4 for GPU struct alignment.
/// Increase if needed, but keep the total struct size a multiple of 16 bytes.
pub const MAX_SPECIES: usize = 4;

/// Bit flag: particle is active in the simulation.
pub const FLAG_ACTIVE: u32 = 1;

/// Sentinel Morton key used for inactive particle slots.
///
/// Active particles always receive keys in the 63-bit Morton range for up to
/// 21 bits per axis; `u64::MAX` therefore safely sorts inactive slots last.
pub const INACTIVE_MORTON_KEY: u64 = u64::MAX;

/// Initial parameters for creating a new [`Particle`].
///
/// Groups the many constructor arguments into a single struct to avoid
/// a long parameter list. All velocity/turbulence fields start at zero.
pub struct ParticleInit {
    /// Reference grid cell x-index.
    pub cell_x: i32,
    /// Reference grid cell y-index.
    pub cell_y: i32,
    /// Fractional x-position within the grid cell [0..1).
    pub pos_x: f32,
    /// Fractional y-position within the grid cell [0..1).
    pub pos_y: f32,
    /// Height above ground [m].
    pub pos_z: f32,
    /// Mass per species [kg]; unused slots should be zero.
    pub mass: [f32; MAX_SPECIES],
    /// Release point index (`npoint`).
    pub release_point: i32,
    /// Particle class (`nclass`).
    pub class: i32,
    /// Particle time [s] (`itra1`).
    pub time: i32,
}

/// A single Lagrangian particle.
///
/// Ported from `com_mod.f90:675–695`. Combines the Fortran's separate
/// per-particle arrays (`xtra1`, `ytra1`, `ztra1`, `xmass1`, `itra1`,
/// `npoint`, `nclass`, `idt`, `itramem`, `itrasplit`, `uap`, `ucp`,
/// `uzp`, `us`, `vs`, `ws`, `cbt`) into a single contiguous struct
/// suitable for GPU storage buffers.
///
/// ## Position Encoding
///
/// To maintain `f32` precision over large domains, horizontal position is
/// split into integer grid cell indices ([`cell_x`][Particle::cell_x],
/// [`cell_y`][Particle::cell_y]) and fractional offsets
/// ([`pos_x`][Particle::pos_x], [`pos_y`][Particle::pos_y]) within that
/// cell. The absolute grid coordinate is reconstructed as
/// `cell_x as f32 + pos_x`. This avoids catastrophic cancellation when
/// subtracting nearby large coordinates on the GPU.
///
/// ## GPU Layout
///
/// The struct is exactly 96 bytes (6 × 16), `#[repr(C)]`, and
/// `Pod + Zeroable`. An all-zero particle is inactive (`flags == 0`).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    // ── Position ──────────────────────────────────────────────────────
    /// Fractional x-position within the reference grid cell [grid units, 0..1).
    /// Ported from `com_mod.f90:680` (`xtra1`, split for f32 precision).
    pub pos_x: f32,

    /// Fractional y-position within the reference grid cell [grid units, 0..1).
    /// Ported from `com_mod.f90:680` (`ytra1`, split for f32 precision).
    pub pos_y: f32,

    /// Height above ground or model surface [m].
    /// Ported from `com_mod.f90:681` (`ztra1`).
    pub pos_z: f32,

    /// Reference grid cell x-index.
    /// Combined with `pos_x` to reconstruct the full grid coordinate.
    pub cell_x: i32,

    /// Reference grid cell y-index.
    /// Combined with `pos_y` to reconstruct the full grid coordinate.
    pub cell_y: i32,

    /// Bit flags — bit 0 = active (see [`FLAG_ACTIVE`]).
    pub flags: u32,

    // ── Species mass ─────────────────────────────────────────────────
    /// Mass of each chemical species carried by this particle [kg].
    /// Ported from `com_mod.f90:682` (`xmass1`). Up to [`MAX_SPECIES`]
    /// species; unused slots must be zero.
    pub mass: [f32; MAX_SPECIES],

    // ── Velocity memory ──────────────────────────────────────────────
    /// Remembered u-wind component from previous time step [m/s].
    /// Ported from `com_mod.f90:694` (`uap`).
    pub vel_u: f32,

    /// Remembered v-wind component from previous time step [m/s].
    /// Ported from `com_mod.f90:694` (`ucp`).
    pub vel_v: f32,

    /// Remembered w-wind component from previous time step [m/s].
    /// Ported from `com_mod.f90:694` (`uzp`).
    pub vel_w: f32,

    // ── Turbulent fluctuations ───────────────────────────────────────
    /// Turbulent u-velocity fluctuation [m/s].
    /// Ported from `com_mod.f90:694` (`us`).
    pub turb_u: f32,

    /// Turbulent v-velocity fluctuation [m/s].
    /// Ported from `com_mod.f90:694` (`vs`).
    pub turb_v: f32,

    /// Turbulent w-velocity fluctuation [m/s].
    /// Ported from `com_mod.f90:694` (`ws`).
    pub turb_w: f32,

    // ── Temporal fields ──────────────────────────────────────────────
    /// Absolute simulation time of this particle [s].
    /// Ported from `com_mod.f90:678` (`itra1`).
    pub time: i32,

    /// Integration time step for this particle [s].
    /// Ported from `com_mod.f90:678` (`idt`).
    pub timestep: i32,

    /// Memorized release time [s].
    /// Ported from `com_mod.f90:678` (`itramem`).
    pub time_mem: i32,

    /// Next particle-splitting time [s].
    /// Ported from `com_mod.f90:678` (`itrasplit`).
    pub time_split: i32,

    // ── Metadata ─────────────────────────────────────────────────────
    /// Release point index.
    /// Ported from `com_mod.f90:678` (`npoint`).
    pub release_point: i32,

    /// Particle class (one of `nclassunc` classes).
    /// Ported from `com_mod.f90:678` (`nclass`).
    pub class: i32,

    /// Convective boundary layer (CBL) flag.
    /// Ported from `com_mod.f90:695` (`cbt`). Widened from `i16` to `i32`
    /// for GPU-friendly alignment.
    pub cbt: i32,

    /// Explicit padding to bring struct size to 96 bytes (multiple of 16).
    pub pad0: u32,
}

impl Particle {
    /// Size of the struct in bytes, as seen by the GPU.
    pub const GPU_SIZE: usize = std::mem::size_of::<Self>();

    /// Create a new active particle from the given initial parameters.
    ///
    /// Velocity, turbulence, and time-step fields are initialized to zero.
    /// The particle is marked active via [`FLAG_ACTIVE`].
    #[must_use]
    pub fn new(init: &ParticleInit) -> Self {
        Self {
            pos_x: init.pos_x,
            pos_y: init.pos_y,
            pos_z: init.pos_z,
            cell_x: init.cell_x,
            cell_y: init.cell_y,
            flags: FLAG_ACTIVE,
            mass: init.mass,
            vel_u: 0.0,
            vel_v: 0.0,
            vel_w: 0.0,
            turb_u: 0.0,
            turb_v: 0.0,
            turb_w: 0.0,
            time: init.time,
            timestep: 0,
            time_mem: init.time,
            time_split: 0,
            release_point: init.release_point,
            class: init.class,
            cbt: 0,
            pad0: 0,
        }
    }

    /// Whether this particle is active in the simulation.
    #[inline]
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.flags & FLAG_ACTIVE != 0
    }

    /// Mark this particle as inactive (e.g., deposited or left domain).
    #[inline]
    pub fn deactivate(&mut self) {
        self.flags &= !FLAG_ACTIVE;
    }

    /// Reconstruct the absolute grid x-coordinate.
    #[inline]
    #[must_use]
    pub fn grid_x(&self) -> f64 {
        f64::from(self.cell_x) + f64::from(self.pos_x)
    }

    /// Reconstruct the absolute grid y-coordinate.
    #[inline]
    #[must_use]
    pub fn grid_y(&self) -> f64 {
        f64::from(self.cell_y) + f64::from(self.pos_y)
    }
}

// ── Errors ───────────────────────────────────────────────────────────────

/// Errors that can occur when manipulating particles.
#[derive(Error, Debug)]
pub enum ParticleError {
    /// Attempted to add a particle when the store has no free slots.
    #[error("particle store is full (capacity: {capacity})")]
    StoreFull {
        /// The maximum number of particles in this store.
        capacity: usize,
    },
}

/// Errors produced by particle spatial-key and reordering operations.
#[derive(Error, Debug)]
pub enum ParticleSortError {
    /// `bits_per_axis` must fit in 64-bit Morton interleaving (3 * bits <= 63).
    #[error("invalid bits_per_axis {value} (must be in 1..=21)")]
    InvalidBitsPerAxis { value: u8 },
    /// Reorder map cannot be applied to a slice of mismatched length.
    #[error("cannot reorder `{field}`: expected length {expected}, got {actual}")]
    ReorderLengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
}

/// Spatial bounds used to quantize particle positions before Morton encoding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialSortBounds {
    /// Minimum absolute grid x coordinate.
    pub x_min: f64,
    /// Maximum absolute grid x coordinate.
    pub x_max: f64,
    /// Minimum absolute grid y coordinate.
    pub y_min: f64,
    /// Maximum absolute grid y coordinate.
    pub y_max: f64,
    /// Minimum height [m].
    pub z_min: f64,
    /// Maximum height [m].
    pub z_max: f64,
}

impl SpatialSortBounds {
    /// Derive bounds from active particles in the slice.
    ///
    /// Returns `None` when no active particles are present.
    #[must_use]
    pub fn from_active_particles(particles: &[Particle]) -> Option<Self> {
        let mut bounds: Option<Self> = None;
        for particle in particles.iter().filter(|particle| particle.is_active()) {
            let x = particle.grid_x();
            let y = particle.grid_y();
            let z = f64::from(particle.pos_z);
            bounds = Some(match bounds {
                None => Self {
                    x_min: x,
                    x_max: x,
                    y_min: y,
                    y_max: y,
                    z_min: z,
                    z_max: z,
                },
                Some(current) => Self {
                    x_min: current.x_min.min(x),
                    x_max: current.x_max.max(x),
                    y_min: current.y_min.min(y),
                    y_max: current.y_max.max(y),
                    z_min: current.z_min.min(z),
                    z_max: current.z_max.max(z),
                },
            });
        }
        bounds
    }
}

/// Policy used to select quantization bounds for Morton sorting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpatialSortBoundsMode {
    /// Use explicit user-provided bounds.
    Explicit(SpatialSortBounds),
    /// Compute bounds from active particle extents each sort call.
    ActiveParticleExtent,
}

/// Options controlling particle spatial reordering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleSpatialSortOptions {
    /// Number of quantization bits per axis (1..=21).
    pub bits_per_axis: u8,
    /// Bound-selection strategy used before key generation.
    pub bounds_mode: SpatialSortBoundsMode,
    /// Whether to produce `old_to_new` inverse mapping in the returned map.
    pub include_inverse_map: bool,
}

impl Default for ParticleSpatialSortOptions {
    fn default() -> Self {
        Self {
            bits_per_axis: 10,
            bounds_mode: SpatialSortBoundsMode::ActiveParticleExtent,
            include_inverse_map: false,
        }
    }
}

/// Reordering map returned by spatial sorting.
///
/// `new_to_old[new_index] = old_index` always exists. The optional
/// `old_to_new[old_index] = new_index` is generated when requested.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParticleReorderMap {
    pub new_to_old: Vec<usize>,
    pub old_to_new: Option<Vec<usize>>,
}

impl ParticleReorderMap {
    /// Returns true when the mapping leaves order unchanged.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.new_to_old
            .iter()
            .enumerate()
            .all(|(new_index, old_index)| new_index == *old_index)
    }

    /// Optional inverse map (`old_index -> new_index`).
    #[must_use]
    pub fn old_to_new(&self) -> Option<&[usize]> {
        self.old_to_new.as_deref()
    }

    /// Reorder an external array using this map.
    ///
    /// Useful when companion per-particle arrays (ages, diagnostics, etc.)
    /// must follow the same ordering as the particle store.
    ///
    /// # Errors
    ///
    /// Returns [`ParticleSortError::ReorderLengthMismatch`] if `values.len()`
    /// differs from the particle-store length represented by this map.
    pub fn reorder_slice<T: Copy>(
        &self,
        values: &[T],
        field: &'static str,
    ) -> Result<Vec<T>, ParticleSortError> {
        if values.len() != self.new_to_old.len() {
            return Err(ParticleSortError::ReorderLengthMismatch {
                field,
                expected: self.new_to_old.len(),
                actual: values.len(),
            });
        }
        Ok(self
            .new_to_old
            .iter()
            .map(|&old_index| values[old_index])
            .collect())
    }
}

fn identity_reorder_map(len: usize, include_inverse_map: bool) -> ParticleReorderMap {
    let new_to_old: Vec<usize> = (0..len).collect();
    let old_to_new = include_inverse_map.then(|| (0..len).collect());
    ParticleReorderMap {
        new_to_old,
        old_to_new,
    }
}

fn validate_bits_per_axis(bits_per_axis: u8) -> Result<(), ParticleSortError> {
    if !(1..=21).contains(&bits_per_axis) {
        return Err(ParticleSortError::InvalidBitsPerAxis {
            value: bits_per_axis,
        });
    }
    Ok(())
}

fn quantize_axis(value: f64, axis_min: f64, axis_max: f64, max_bin: u32) -> u32 {
    if !value.is_finite() || !axis_min.is_finite() || !axis_max.is_finite() {
        return 0;
    }
    let span = axis_max - axis_min;
    if span <= f64::EPSILON {
        return 0;
    }
    let normalized = ((value - axis_min) / span).clamp(0.0, 1.0);
    (normalized * f64::from(max_bin)).floor() as u32
}

fn split_by_2_zeros_64(value: u32) -> u64 {
    let mut x = u64::from(value & 0x001F_FFFF);
    x = (x | (x << 32)) & 0x001F_0000_0000_FFFF;
    x = (x | (x << 16)) & 0x001F_0000_FF00_00FF;
    x = (x | (x << 8)) & 0x100F_00F0_0F00_F00F;
    x = (x | (x << 4)) & 0x10C3_0C30_C30C_30C3;
    x = (x | (x << 2)) & 0x1249_2492_4924_9249;
    x
}

/// Encode three quantized 21-bit coordinates into one Morton key.
#[must_use]
pub fn morton_encode_3d_21bit(x: u32, y: u32, z: u32) -> u64 {
    split_by_2_zeros_64(x) | (split_by_2_zeros_64(y) << 1) | (split_by_2_zeros_64(z) << 2)
}

/// Compute a Morton key for a particle position.
///
/// Coordinates are quantized to `bits_per_axis` bins within `bounds`.
///
/// # Errors
///
/// Returns [`ParticleSortError::InvalidBitsPerAxis`] when `bits_per_axis` is
/// outside `1..=21`.
pub fn morton_key_from_position(
    x: f64,
    y: f64,
    z: f64,
    bounds: SpatialSortBounds,
    bits_per_axis: u8,
) -> Result<u64, ParticleSortError> {
    validate_bits_per_axis(bits_per_axis)?;
    let max_bin = (1_u32 << u32::from(bits_per_axis)) - 1;
    let qx = quantize_axis(x, bounds.x_min, bounds.x_max, max_bin);
    let qy = quantize_axis(y, bounds.y_min, bounds.y_max, max_bin);
    let qz = quantize_axis(z, bounds.z_min, bounds.z_max, max_bin);
    Ok(morton_encode_3d_21bit(qx, qy, qz))
}

// ── Particle Store ───────────────────────────────────────────────────────

/// Container for all particles in the simulation.
///
/// Backed by a contiguous `Vec<Particle>` suitable for bulk upload to a GPU
/// storage buffer via [`as_gpu_bytes`](ParticleStore::as_gpu_bytes).
///
/// Particles are addressed by index. Inactive particles remain in place
/// (flagged via [`FLAG_ACTIVE`]) so that GPU-side indexing stays stable
/// between frames. The [`add`](ParticleStore::add) method reuses the first
/// available inactive slot.
pub struct ParticleStore {
    particles: Vec<Particle>,
    capacity: usize,
    active_count: usize,
    /// Stack of known-free slot indices for O(1) insertion.
    free_slots: Vec<usize>,
}

impl ParticleStore {
    /// Create a new store pre-allocated for `capacity` particles.
    ///
    /// The internal buffer is filled with zeroed (inactive) particles so
    /// that the GPU buffer size is known upfront.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let particles = vec![Particle::zeroed(); capacity];
        let free_slots: Vec<usize> = (0..capacity).rev().collect();
        Self {
            particles,
            capacity,
            active_count: 0,
            free_slots,
        }
    }

    /// Release a particle into the first available inactive slot.
    ///
    /// Returns the slot index on success.
    ///
    /// # Errors
    ///
    /// Returns [`ParticleError::StoreFull`] if every slot is occupied by an
    /// active particle.
    pub fn add(&mut self, particle: Particle) -> Result<usize, ParticleError> {
        if let Some(i) = self.free_slots.pop() {
            self.particles[i] = particle;
            self.active_count += 1;
            return Ok(i);
        }
        Err(ParticleError::StoreFull {
            capacity: self.capacity,
        })
    }

    /// Number of currently active particles.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Total capacity (maximum simultaneous particles).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Immutable slice over all particle slots (active and inactive).
    ///
    /// Suitable for GPU buffer upload via `bytemuck::cast_slice`.
    #[must_use]
    pub fn as_slice(&self) -> &[Particle] {
        &self.particles
    }

    /// Mutable slice over all particle slots.
    ///
    /// After mutating flags directly, call [`recount_active`](Self::recount_active)
    /// to resynchronize the cached count.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [Particle] {
        &mut self.particles
    }

    /// The entire particle buffer as raw bytes, ready for GPU upload.
    #[must_use]
    pub fn as_gpu_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.particles)
    }

    /// Access a particle by slot index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Particle> {
        self.particles.get(index)
    }

    /// Mutably access a particle by slot index.
    #[must_use]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Particle> {
        self.particles.get_mut(index)
    }

    /// Deactivate a particle at `index` and decrement the active count.
    ///
    /// Returns `None` if the index is out of bounds or the particle is
    /// already inactive.
    pub fn deactivate(&mut self, index: usize) -> Option<()> {
        let particle = self.particles.get_mut(index)?;
        if particle.is_active() {
            particle.deactivate();
            self.active_count = self.active_count.saturating_sub(1);
            self.free_slots.push(index);
            Some(())
        } else {
            None
        }
    }

    /// Recount active particles by scanning all slots.
    ///
    /// Call after GPU readback or direct mutation of the particle slice
    /// to resynchronize the cached active count.
    pub fn recount_active(&mut self) {
        self.active_count = self.particles.iter().filter(|p| p.is_active()).count();
    }

    /// Synchronize host-side bookkeeping after GPU compaction without
    /// downloading the full particle buffer.
    ///
    /// After GPU gather/reorder, active particles occupy contiguous slots
    /// `[0..active_count)` and all remaining slots are free. This method
    /// updates `active_count` and rebuilds the free-slot stack accordingly,
    /// keeping the host `ParticleStore` consistent for release-slot
    /// assignment without the cost of a full GPU→CPU readback.
    pub fn reset_after_compaction(&mut self, active_count: usize) {
        self.active_count = active_count;
        self.free_slots.clear();
        self.free_slots
            .extend((active_count..self.capacity).rev());
    }

    /// Compute per-slot Morton keys for current particle positions.
    ///
    /// Active particles use quantized `grid_x/grid_y/pos_z`; inactive slots
    /// receive [`INACTIVE_MORTON_KEY`] so they are sorted last.
    ///
    /// # Errors
    ///
    /// Returns [`ParticleSortError::InvalidBitsPerAxis`] when `bits_per_axis`
    /// is outside `1..=21`.
    pub fn compute_spatial_keys(
        &self,
        bounds: SpatialSortBounds,
        bits_per_axis: u8,
    ) -> Result<Vec<u64>, ParticleSortError> {
        validate_bits_per_axis(bits_per_axis)?;
        self.particles
            .iter()
            .map(|particle| {
                if particle.is_active() {
                    morton_key_from_position(
                        particle.grid_x(),
                        particle.grid_y(),
                        f64::from(particle.pos_z),
                        bounds,
                        bits_per_axis,
                    )
                } else {
                    Ok(INACTIVE_MORTON_KEY)
                }
            })
            .collect()
    }

    /// Stable spatial sort of particle slots by Morton/Z-order key.
    ///
    /// Active particles are sorted by key. Inactive slots are always placed
    /// after active particles and keep relative order (stable behavior).
    ///
    /// # Errors
    ///
    /// Returns [`ParticleSortError::InvalidBitsPerAxis`] when options contain
    /// an invalid quantization bit count.
    pub fn sort_spatially(
        &mut self,
        options: ParticleSpatialSortOptions,
    ) -> Result<ParticleReorderMap, ParticleSortError> {
        let bounds = match options.bounds_mode {
            SpatialSortBoundsMode::Explicit(bounds) => bounds,
            SpatialSortBoundsMode::ActiveParticleExtent => {
                if let Some(active_bounds) =
                    SpatialSortBounds::from_active_particles(&self.particles)
                {
                    active_bounds
                } else {
                    return Ok(identity_reorder_map(
                        self.particles.len(),
                        options.include_inverse_map,
                    ));
                }
            }
        };

        let keys = self.compute_spatial_keys(bounds, options.bits_per_axis)?;
        let mut new_to_old: Vec<usize> = (0..self.particles.len()).collect();
        // Stable sort preserves relative order for equal keys, including
        // inactive particles sharing INACTIVE_MORTON_KEY.
        new_to_old.sort_by_key(|&index| keys[index]);

        let old_to_new = options.include_inverse_map.then(|| {
            let mut inverse = vec![0_usize; new_to_old.len()];
            for (new_index, &old_index) in new_to_old.iter().enumerate() {
                inverse[old_index] = new_index;
            }
            inverse
        });

        if !new_to_old
            .iter()
            .enumerate()
            .all(|(new_index, old_index)| new_index == *old_index)
        {
            let reordered: Vec<Particle> = new_to_old
                .iter()
                .map(|&old_index| self.particles[old_index])
                .collect();
            self.particles = reordered;
        }

        Ok(ParticleReorderMap {
            new_to_old,
            old_to_new,
        })
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_init() -> ParticleInit {
        ParticleInit {
            cell_x: 0,
            cell_y: 0,
            pos_x: 0.0,
            pos_y: 0.0,
            pos_z: 0.0,
            mass: [0.0; MAX_SPECIES],
            release_point: 0,
            class: 0,
            time: 0,
        }
    }

    #[test]
    fn test_particle_gpu_size_is_96_bytes() {
        assert_eq!(Particle::GPU_SIZE, 96);
    }

    #[test]
    fn test_particle_gpu_size_multiple_of_16() {
        assert_eq!(
            Particle::GPU_SIZE % 16,
            0,
            "Particle struct size must be a multiple of 16 for GPU alignment"
        );
    }

    #[test]
    fn test_particle_bytemuck_zeroed_is_inactive() {
        let zeroed: Particle = Particle::zeroed();
        assert!(!zeroed.is_active());
        assert_eq!(zeroed.pos_x, 0.0);
        assert_eq!(zeroed.pos_z, 0.0);
        assert_eq!(zeroed.mass, [0.0; MAX_SPECIES]);
    }

    #[test]
    fn test_particle_bytemuck_roundtrip() {
        let p = Particle::new(&ParticleInit {
            cell_x: 5,
            cell_y: 10,
            pos_x: 0.25,
            pos_y: 0.75,
            pos_z: 500.0,
            mass: [1.0, 2.0, 0.0, 0.0],
            release_point: 3,
            class: 1,
            time: 7200,
        });
        let bytes: &[u8] = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 96);

        let restored: &Particle = bytemuck::from_bytes(bytes);
        assert_eq!(restored.cell_x, 5);
        assert_eq!(restored.cell_y, 10);
        assert!((restored.pos_x - 0.25).abs() < f32::EPSILON);
        assert!((restored.mass[1] - 2.0).abs() < f32::EPSILON);
        assert!(restored.is_active());
    }

    #[test]
    fn test_particle_new_is_active() {
        let p = Particle::new(&sample_init());
        assert!(p.is_active());
    }

    #[test]
    fn test_particle_deactivate() {
        let mut p = Particle::new(&sample_init());
        p.deactivate();
        assert!(!p.is_active());
    }

    #[test]
    fn test_particle_grid_coordinates() {
        let p = Particle::new(&ParticleInit {
            cell_x: 100,
            cell_y: 200,
            pos_x: 0.5,
            pos_y: 0.75,
            pos_z: 0.0,
            mass: [0.0; MAX_SPECIES],
            release_point: 0,
            class: 0,
            time: 0,
        });
        assert!((p.grid_x() - 100.5).abs() < f64::EPSILON);
        assert!((p.grid_y() - 200.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_store_create_and_add() {
        let mut store = ParticleStore::with_capacity(10);
        assert_eq!(store.active_count(), 0);
        assert_eq!(store.capacity(), 10);

        let p = Particle::new(&ParticleInit {
            cell_x: 5,
            cell_y: 10,
            pos_x: 0.5,
            pos_y: 0.3,
            pos_z: 100.0,
            mass: [1.0, 0.0, 0.0, 0.0],
            release_point: 0,
            class: 1,
            time: 0,
        });
        let idx = store.add(p).expect("store should have space");
        assert_eq!(idx, 0);
        assert_eq!(store.active_count(), 1);
        assert!(store.get(0).expect("valid index").is_active());
    }

    #[test]
    fn test_store_full_returns_error() {
        let mut store = ParticleStore::with_capacity(1);
        let p = Particle::new(&sample_init());

        store.add(p).expect("first add succeeds");
        let result = store.add(p);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("particle store is full"));
    }

    #[test]
    fn test_store_reuses_deactivated_slots() {
        let mut store = ParticleStore::with_capacity(2);
        let p = Particle::new(&ParticleInit {
            cell_x: 0,
            cell_y: 0,
            pos_x: 0.1,
            pos_y: 0.2,
            pos_z: 50.0,
            mass: [1.0, 0.0, 0.0, 0.0],
            release_point: 0,
            class: 0,
            time: 0,
        });

        let idx0 = store.add(p).expect("slot 0");
        let _idx1 = store.add(p).expect("slot 1");
        assert_eq!(store.active_count(), 2);

        store.deactivate(idx0).expect("deactivate slot 0");
        assert_eq!(store.active_count(), 1);

        let new_p = Particle::new(&ParticleInit {
            cell_x: 1,
            cell_y: 1,
            pos_x: 0.5,
            pos_y: 0.5,
            pos_z: 200.0,
            mass: [2.0, 0.0, 0.0, 0.0],
            release_point: 1,
            class: 0,
            time: 100,
        });
        let reused = store.add(new_p).expect("reuse slot 0");
        assert_eq!(reused, 0);
        assert_eq!(store.active_count(), 2);
        assert!((store.get(0).expect("valid").mass[0] - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_store_gpu_bytes_length() {
        let store = ParticleStore::with_capacity(100);
        assert_eq!(store.as_gpu_bytes().len(), 100 * Particle::GPU_SIZE);
    }

    #[test]
    fn test_store_recount_after_mutation() {
        let mut store = ParticleStore::with_capacity(5);
        let p = Particle::new(&sample_init());
        store.add(p).expect("add");
        store.add(p).expect("add");
        assert_eq!(store.active_count(), 2);

        store.as_mut_slice()[0].deactivate();
        assert_eq!(store.active_count(), 2, "cached count not yet updated");

        store.recount_active();
        assert_eq!(store.active_count(), 1, "recount reflects mutation");
    }

    #[test]
    fn test_store_deactivate_already_inactive_returns_none() {
        let mut store = ParticleStore::with_capacity(5);
        assert!(
            store.deactivate(0).is_none(),
            "zeroed particle is already inactive"
        );
    }

    fn particle_at(cell_x: i32, cell_y: i32, pos_z: f32) -> Particle {
        Particle::new(&ParticleInit {
            cell_x,
            cell_y,
            pos_x: 0.0,
            pos_y: 0.0,
            pos_z,
            mass: [1.0, 0.0, 0.0, 0.0],
            release_point: 0,
            class: 0,
            time: 0,
        })
    }

    #[test]
    fn test_morton_key_increases_for_x_cells() {
        let bounds = SpatialSortBounds {
            x_min: 0.0,
            x_max: 3.0,
            y_min: 0.0,
            y_max: 1.0,
            z_min: 0.0,
            z_max: 1.0,
        };

        let key0 = morton_key_from_position(0.0, 0.0, 0.0, bounds, 2).expect("key0");
        let key1 = morton_key_from_position(1.0, 0.0, 0.0, bounds, 2).expect("key1");
        let key2 = morton_key_from_position(2.0, 0.0, 0.0, bounds, 2).expect("key2");
        assert!(key0 < key1);
        assert!(key1 < key2);
    }

    #[test]
    fn test_compute_spatial_keys_marks_inactive_with_sentinel() {
        let mut store = ParticleStore::with_capacity(3);
        store.add(particle_at(0, 0, 0.0)).expect("slot 0");
        store.add(particle_at(1, 0, 0.0)).expect("slot 1");
        store.deactivate(1).expect("deactivate slot 1");

        let bounds = SpatialSortBounds {
            x_min: 0.0,
            x_max: 2.0,
            y_min: 0.0,
            y_max: 1.0,
            z_min: 0.0,
            z_max: 1.0,
        };
        let keys = store
            .compute_spatial_keys(bounds, 4)
            .expect("keys should compute");
        assert_ne!(keys[0], INACTIVE_MORTON_KEY);
        assert_eq!(keys[1], INACTIVE_MORTON_KEY);
        assert_eq!(keys[2], INACTIVE_MORTON_KEY);
    }

    #[test]
    fn test_spatial_sort_orders_active_and_keeps_inactive_stable() {
        let mut store = ParticleStore::with_capacity(6);
        store.add(particle_at(3, 0, 0.0)).expect("slot 0");
        store.add(particle_at(1, 0, 0.0)).expect("slot 1");
        store.add(particle_at(2, 0, 0.0)).expect("slot 2");
        store.deactivate(1).expect("slot 1 inactive");

        let map = store
            .sort_spatially(ParticleSpatialSortOptions {
                bits_per_axis: 4,
                bounds_mode: SpatialSortBoundsMode::Explicit(SpatialSortBounds {
                    x_min: 0.0,
                    x_max: 4.0,
                    y_min: 0.0,
                    y_max: 1.0,
                    z_min: 0.0,
                    z_max: 1.0,
                }),
                include_inverse_map: true,
            })
            .expect("sort succeeds");

        assert_eq!(map.new_to_old, vec![2, 0, 1, 3, 4, 5]);
        let inverse = map.old_to_new().expect("inverse map requested");
        assert_eq!(inverse[0], 1);
        assert_eq!(inverse[2], 0);
        assert_eq!(store.active_count(), 2);
        assert_eq!(store.get(0).expect("slot").cell_x, 2);
        assert_eq!(store.get(1).expect("slot").cell_x, 3);
        assert!(!store.get(2).expect("slot").is_active());
        assert!(!store.get(3).expect("slot").is_active());
    }

    #[test]
    fn test_reorder_map_reorders_external_array() {
        let map = ParticleReorderMap {
            new_to_old: vec![2, 0, 1],
            old_to_new: Some(vec![1, 2, 0]),
        };
        let values = vec![10_u32, 20_u32, 30_u32];
        let reordered = map.reorder_slice(&values, "diagnostics").expect("reorder");
        assert_eq!(reordered, vec![30_u32, 10_u32, 20_u32]);
    }
}
