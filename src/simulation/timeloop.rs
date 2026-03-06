//! Forward time-loop orchestration (I-01).
//!
//! This module provides an MVP integration manager equivalent in role to
//! FLEXPART's `timemanager.f90` forward loop:
//! 1. release/update particles (A-06),
//! 2. interpolate met fields to current time (IO-04),
//! 3. derive PBL diagnostics from interpolated surface fields (IO-05),
//! 4. dispatch advection (A-04),
//! 5. dispatch Langevin turbulence (H-04),
//! 6. dispatch dry and wet deposition (D-02, D-04).
//!
//! It also provides a backward-mode MVP (`ldirect = -1` equivalent) used for
//! receptor-driven source attribution (B-01).
//!
//! ## MVP assumptions
//! - CBL-specific turbulence branch is not orchestrated yet.
//! - PBL profile cues are not temporally interpolated in this baseline; IO-05 is
//!   driven from interpolated surface fields only.
//! - Deposition forcing vectors are caller-provided per timestep.
//! - Backward mode currently reverses deterministic advection time direction,
//!   while keeping turbulence and deposition dispatch in forward-sign `dt`.
//!   This simplification is intentional for B-01 MVP and documented in the
//!   backward API docs below.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use thiserror::Error;

use crate::config::ReleaseConfig;
use crate::coords::GridDomain;
use crate::gpu::{
    accumulate_concentration_grid_gpu, encode_advection_dual_wind_gpu_with_kernel,
    encode_compaction_with_reorder, encode_dry_deposition_probability_gpu_with_kernel,
    encode_hanna_params_gpu_with_kernel, encode_langevin_fused_gpu,
    encode_pbl_diagnostics_gpu_with_kernel,
    encode_update_particles_turbulence_langevin_gpu_with_hanna_buffer_and_kernel,
    encode_wet_deposition_probability_gpu_with_kernel,
    AdvectionDispatchKernel, AdvectionDualWindDispatchKernel, CompactionBuffers,
    CompactionPipelines, ConcentrationGridOutput, ConcentrationGridShape,
    ConcentrationGriddingParams, DryDepositionDispatchKernel, DryDepositionIoBuffers,
    DryDepositionStepParams, DualWindBuffers, GpuAdvectionError, GpuBufferError,
    GpuCompactionError, GpuConcentrationGriddingError, GpuContext, GpuDryDepositionError,
    GpuError, GpuHannaError, GpuLangevinError, GpuLangevinFusedError,
    GpuPblDiagnosticsError, GpuPblReflectionError, GpuWetDepositionError,
    HannaDispatchKernel, HannaParamsOutputBuffer,
    LangevinDispatchKernel, LangevinFusedDispatchKernel, ParticleBuffers,
    PblBuffers, PblDiagnosticsDispatchKernel,
    SurfaceFieldBuffer, WetDepositionDispatchKernel, WetDepositionIoBuffers,
    WetDepositionStepParams, WindBuffers, WindSamplingPath,
};
use crate::io::{
    compute_pbl_parameters_from_met, interpolate_surface_fields_linear, Era5GribGridMetadata,
    Era5MvpSnapshot, Grib2ReaderError,
    GribPrefetchHandle, PblComputationOptions, PblMetInputGrids, PblParameterError,
    TemporalInterpolationError, TimeBoundsBehavior,
};
use crate::particles::{ParticleSortError, ParticleSpatialSortOptions, ParticleStore};
use crate::physics::{LangevinStep, PhiloxCounter, PhiloxKey, VelocityToGridScale};
use crate::release::{ReleaseError, ReleaseManager};
use crate::wind::{SurfaceFields, WindField3D};

/// Time-loop configuration for forward integration.
#[derive(Debug, Clone)]
pub struct ForwardTimeLoopConfig {
    /// Inclusive simulation start timestamp (`YYYYMMDDHHMMSS`).
    pub start_timestamp: String,
    /// Inclusive simulation end timestamp (`YYYYMMDDHHMMSS`).
    pub end_timestamp: String,
    /// Integration timestep [s].
    pub timestep_seconds: i64,
    /// Temporal-interpolation behavior outside met bracket.
    pub time_bounds_behavior: TimeBoundsBehavior,
    /// Conversion from wind velocity to grid-coordinate displacement.
    pub velocity_to_grid_scale: VelocityToGridScale,
    /// PBL diagnostic options used by IO-05 computation.
    pub pbl_options: PblComputationOptions,
    /// Dry-deposition reference height [m].
    pub dry_reference_height_m: f32,
    /// Deterministic Philox RNG key for the Langevin update.
    pub philox_key: PhiloxKey,
    /// Initial Philox counter for the first timestep.
    pub initial_philox_counter: PhiloxCounter,
    /// Optional particle spatial sorting for better GPU memory locality (O-01).
    pub spatial_sort: Option<ForwardSpatialSortConfig>,
    /// If true, download the full particle buffer after each timestep to keep
    /// host-side [`ParticleStore`] synchronized in lockstep.
    ///
    /// Set to `false` in performance mode to defer host synchronization.
    pub sync_particle_store_each_step: bool,
    /// If true, download per-particle dry/wet deposition probabilities for each
    /// step report. Set to `false` to avoid per-step probability readbacks.
    pub collect_deposition_probabilities_each_step: bool,
}

/// Runtime settings for optional particle Morton sorting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForwardSpatialSortConfig {
    /// Apply sort every `interval_steps` timesteps (1 = every step).
    pub interval_steps: usize,
    /// Spatial-key and map-generation options.
    pub sort_options: ParticleSpatialSortOptions,
}

impl Default for ForwardTimeLoopConfig {
    fn default() -> Self {
        Self {
            start_timestamp: "20240101000000".to_string(),
            end_timestamp: "20240101000000".to_string(),
            timestep_seconds: 1,
            time_bounds_behavior: TimeBoundsBehavior::Clamp,
            velocity_to_grid_scale: VelocityToGridScale::IDENTITY,
            pbl_options: PblComputationOptions::default(),
            dry_reference_height_m: 15.0,
            philox_key: [0xDECA_FBAD, 0x1234_5678],
            initial_philox_counter: [0, 0, 0, 0],
            spatial_sort: None,
            sync_particle_store_each_step: true,
            collect_deposition_probabilities_each_step: true,
        }
    }
}

/// Direction of simulation time integration (`ldirect` in FLEXPART terms).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeDirection {
    /// Forward integration (`ldirect = +1`).
    Forward,
    /// Backward integration (`ldirect = -1`).
    Backward,
}

impl TimeDirection {
    #[must_use]
    fn advection_dt_seconds(self, timestep_seconds: f32) -> f32 {
        match self {
            Self::Forward => timestep_seconds,
            Self::Backward => -timestep_seconds,
        }
    }
}

/// Receptor release specification for backward mode.
///
/// Particles are released at receptor coordinates and then integrated backward
/// in time to estimate source attribution.
#[derive(Debug, Clone)]
pub struct BackwardReceptorConfig {
    /// Receptor identifier.
    pub name: String,
    /// Receptor longitude [degrees].
    pub lon: f64,
    /// Receptor latitude [degrees].
    pub lat: f64,
    /// Receptor release height [m].
    pub z_m: f64,
    /// Number of particles to release from this receptor.
    pub particle_count: u64,
    /// Total emitted mass [kg] distributed over `particle_count`.
    pub mass_kg: f64,
}

/// Source envelope used to collect backward particles.
///
/// A particle is counted as a "hit" if its position lies within all lon/lat/z
/// bounds after a backward timestep.
#[derive(Debug, Clone)]
pub struct BackwardSourceRegionConfig {
    /// Source-region identifier.
    pub name: String,
    /// Minimum longitude [degrees].
    pub lon_min: f64,
    /// Maximum longitude [degrees].
    pub lon_max: f64,
    /// Minimum latitude [degrees].
    pub lat_min: f64,
    /// Maximum latitude [degrees].
    pub lat_max: f64,
    /// Minimum height [m].
    pub z_min_m: f64,
    /// Maximum height [m].
    pub z_max_m: f64,
}

impl BackwardSourceRegionConfig {
    #[must_use]
    fn contains_particle(&self, particle: &crate::particles::Particle, grid: &GridDomain) -> bool {
        let lon = grid.xlon0 + particle.grid_x() * grid.dx;
        let lat = grid.ylat0 + particle.grid_y() * grid.dy;
        let z = f64::from(particle.pos_z);
        (self.lon_min..=self.lon_max).contains(&lon)
            && (self.lat_min..=self.lat_max).contains(&lat)
            && (self.z_min_m..=self.z_max_m).contains(&z)
    }
}

/// Per-source backward collection summary at one timestep.
#[derive(Debug, Clone, PartialEq)]
pub struct BackwardSourceCollection {
    /// Number of active particles inside the source region.
    pub hit_count: usize,
    /// Total species-0 mass represented by those hits [kg].
    pub total_mass_kg: f32,
}

impl Default for BackwardSourceCollection {
    fn default() -> Self {
        Self {
            hit_count: 0,
            total_mass_kg: 0.0,
        }
    }
}

/// Time-loop configuration for backward (time-reversed) integration.
///
/// ## MVP simplifications
/// - Receptors are emitted once at `start_timestamp` (not continuously).
/// - Advection uses negative `dt`, but turbulence and deposition dispatch are
///   still executed with positive `dt` magnitude for numerical stability and
///   kernel compatibility. Callers can set zero forcing to effectively disable
///   deposition during backward attribution runs.
#[derive(Debug, Clone)]
pub struct BackwardTimeLoopConfig {
    /// Inclusive simulation start timestamp (`YYYYMMDDHHMMSS`), typically the
    /// receptor sampling time and therefore later than `end_timestamp`.
    pub start_timestamp: String,
    /// Inclusive simulation end timestamp (`YYYYMMDDHHMMSS`), typically earlier
    /// than `start_timestamp`.
    pub end_timestamp: String,
    /// Positive timestep magnitude [s].
    pub timestep_seconds: i64,
    /// Temporal-interpolation behavior outside met bracket.
    pub time_bounds_behavior: TimeBoundsBehavior,
    /// Conversion from wind velocity to grid-coordinate displacement.
    pub velocity_to_grid_scale: VelocityToGridScale,
    /// PBL diagnostic options used by IO-05 computation.
    pub pbl_options: PblComputationOptions,
    /// Dry-deposition reference height [m].
    pub dry_reference_height_m: f32,
    /// Deterministic Philox RNG key for the Langevin update.
    pub philox_key: PhiloxKey,
    /// Initial Philox counter for the first timestep.
    pub initial_philox_counter: PhiloxCounter,
    /// Receptors that emit backward particles at `start_timestamp`.
    pub receptors: Vec<BackwardReceptorConfig>,
    /// Source regions where backward particles are collected.
    pub source_regions: Vec<BackwardSourceRegionConfig>,
}

impl Default for BackwardTimeLoopConfig {
    fn default() -> Self {
        Self {
            start_timestamp: "20240101000000".to_string(),
            end_timestamp: "20240101000000".to_string(),
            timestep_seconds: 1,
            time_bounds_behavior: TimeBoundsBehavior::Clamp,
            velocity_to_grid_scale: VelocityToGridScale::IDENTITY,
            pbl_options: PblComputationOptions::default(),
            dry_reference_height_m: 15.0,
            philox_key: [0xDECA_FBAD, 0x1234_5678],
            initial_philox_counter: [0, 0, 0, 0],
            receptors: Vec::new(),
            source_regions: Vec::new(),
        }
    }
}

/// Per-step meteorological bracket used for temporal interpolation (IO-04).
pub struct MetTimeBracket<'a> {
    /// 3-D met field at lower timestamp bound.
    pub wind_t0: &'a WindField3D,
    /// 3-D met field at upper timestamp bound.
    pub wind_t1: &'a WindField3D,
    /// 2-D surface field at lower timestamp bound.
    pub surface_t0: &'a SurfaceFields,
    /// 2-D surface field at upper timestamp bound.
    pub surface_t1: &'a SurfaceFields,
    /// Lower timestamp bound [s since epoch].
    pub time_t0_seconds: i64,
    /// Upper timestamp bound [s since epoch].
    pub time_t1_seconds: i64,
}

/// Per-particle forcing input shape for one timestep.
#[derive(Debug, Clone)]
pub enum ParticleForcingField {
    /// Use one scalar for all particle slots.
    Uniform(f32),
    /// Explicit per-slot values; length must match particle-buffer slot count.
    PerParticle(Vec<f32>),
}

impl Default for ParticleForcingField {
    fn default() -> Self {
        Self::Uniform(0.0)
    }
}

impl ParticleForcingField {
    fn materialize(
        &self,
        expected_len: usize,
        field: &'static str,
    ) -> Result<Vec<f32>, TimeLoopError> {
        match self {
            Self::Uniform(value) => Ok(vec![*value; expected_len]),
            Self::PerParticle(values) => {
                if values.len() != expected_len {
                    return Err(TimeLoopError::ForcingLengthMismatch {
                        field,
                        expected: expected_len,
                        actual: values.len(),
                    });
                }
                Ok(values.clone())
            }
        }
    }

    /// Returns `true` when the forcing is uniformly zero, meaning the
    /// corresponding deposition process has no effect and can be skipped.
    #[must_use]
    fn is_zero(&self) -> bool {
        matches!(self, Self::Uniform(v) if *v == 0.0)
    }
}

/// Timestep forcing fields consumed by deposition and Langevin updates.
#[derive(Debug, Clone)]
pub struct ForwardStepForcing {
    /// Dry deposition velocity `vdep` [m/s] for all slots.
    pub dry_deposition_velocity_m_s: ParticleForcingField,
    /// Wet scavenging coefficient `lambda` [1/s] for all slots.
    pub wet_scavenging_coefficient_s_inv: ParticleForcingField,
    /// Wet precipitating-area fraction [0..1] for all slots.
    pub wet_precipitating_fraction: ParticleForcingField,
    /// Density-gradient term `(1/rho) * d(rho)/dz` [1/m] for Langevin vertical drift.
    pub rho_grad_over_rho: f32,
}

impl Default for ForwardStepForcing {
    fn default() -> Self {
        Self {
            dry_deposition_velocity_m_s: ParticleForcingField::Uniform(0.0),
            wet_scavenging_coefficient_s_inv: ParticleForcingField::Uniform(0.0),
            wet_precipitating_fraction: ParticleForcingField::Uniform(0.0),
            rho_grad_over_rho: 0.0,
        }
    }
}

/// Result of one orchestrated forward timestep.
#[derive(Debug, Clone)]
pub struct ForwardStepReport {
    /// Zero-based step index in this run.
    pub step_index: usize,
    /// Timestamp at which this step was evaluated (`YYYYMMDDHHMMSS`).
    pub timestamp: String,
    /// Simulation time [s since epoch] for this step.
    pub simulation_time_seconds: i64,
    /// Temporal interpolation weight in `[0, 1]` (after clamp/strict handling).
    pub interpolation_alpha: f32,
    /// Number of particles released during this step.
    pub released_count: usize,
    /// Particle slots injected by release manager for this step.
    pub released_slots: Vec<usize>,
    /// Active particle count after GPU readback.
    pub active_particle_count: usize,
    /// Dry-deposition probability per slot (GPU output).
    pub dry_deposition_probability: Vec<f32>,
    /// Wet-deposition probability per slot (GPU output).
    pub wet_deposition_probability: Vec<f32>,
    /// Philox counter after the Langevin update.
    pub next_philox_counter: PhiloxCounter,
    /// Per-section timing breakdown (present only when `FLEXPART_GPU_PROFILE=1`).
    pub timing: Option<StepTimingReport>,
}

/// Per-section wall-clock timing for one forward timestep.
///
/// Populated only when `FLEXPART_GPU_PROFILE=1` is set. Each field records
/// the wall-clock duration of one pipeline stage in milliseconds. Used by
/// Phase 0 profiling to measure the actual CPU/GPU split before optimizing.
#[derive(Debug, Clone)]
pub struct StepTimingReport {
    /// `interpolate_wind_field_linear` (CPU).
    pub wind_interp_ms: f64,
    /// `interpolate_surface_fields_linear` (CPU).
    pub surf_interp_ms: f64,
    /// `compute_pbl_parameters_from_met` (CPU).
    pub pbl_ms: f64,
    /// `ensure_wind_buffers` — upload wind field to GPU.
    pub wind_upload_ms: f64,
    /// `ensure_pbl_buffers` — upload PBL state to GPU.
    pub pbl_upload_ms: f64,
    /// `materialize()` calls for dry/wet deposition forcing vectors (CPU alloc+fill).
    pub forcing_ms: f64,
    /// `ensure_dry_deposition_io_buffers` + `ensure_wet_deposition_io_buffers`.
    pub dep_upload_ms: f64,
    /// Blocking wait for the *previous* step's GPU submission to complete
    /// (O-03 pipeline overlap). Zero on the first step or when no work is
    /// pending.
    pub wait_prev_gpu_ms: f64,
    /// CPU-side command encoding: `create_command_encoder` through `queue.submit()`.
    pub gpu_encode_ms: f64,
    /// GPU execution: `queue.submit()` to `device.poll(Wait)` completion.
    pub gpu_exec_ms: f64,
    /// Compaction encode + submit + readback (O-07). Zero when compaction
    /// is disabled or all particles are active.
    pub compaction_ms: f64,
    /// Wall-clock for the entire `run_timestep` call.
    pub total_ms: f64,
}

impl StepTimingReport {
    fn print_summary(&self, step_index: usize) {
        eprintln!(
            "[profile] step={step_index} \
             wind_interp={:.1}ms surf_interp={:.1}ms pbl={:.1}ms \
             wind_upload={:.1}ms pbl_upload={:.1}ms forcing={:.1}ms \
             dep_upload={:.1}ms wait_prev={:.1}ms \
             gpu_encode={:.1}ms gpu_exec={:.1}ms \
             compaction={:.1}ms \
             total={:.1}ms",
            self.wind_interp_ms,
            self.surf_interp_ms,
            self.pbl_ms,
            self.wind_upload_ms,
            self.pbl_upload_ms,
            self.forcing_ms,
            self.dep_upload_ms,
            self.wait_prev_gpu_ms,
            self.gpu_encode_ms,
            self.gpu_exec_ms,
            self.compaction_ms,
            self.total_ms,
        );
    }
}

fn is_profiling_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("FLEXPART_GPU_PROFILE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Returns `true` when PBL diagnostics should be computed on GPU (default).
///
/// Set `FLEXPART_GPU_PBL_CPU=1` to force the CPU fallback path.
fn is_gpu_pbl_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !std::env::var("FLEXPART_GPU_PBL_CPU")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Returns `true` when the full multi-dispatch validation path is requested.
///
/// Set `FLEXPART_GPU_VALIDATION=1` to use separated Hanna → Langevin
/// dispatches (5 dispatches per step), suitable for debugging and
/// scientific validation. The default production path uses the fused
/// Hanna+Langevin kernel (4 dispatches: advection + fused H+L + dry dep + wet dep).
fn is_validation_mode() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("FLEXPART_GPU_VALIDATION")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Returns `true` when active-particle compaction is enabled (O-07).
///
/// Set `FLEXPART_GPU_COMPACTION=1` to run a prefix-sum compaction + gather
/// after each physics step, packing active particles into contiguous leading
/// buffer slots. Subsequent dispatches then use `active_count` instead of
/// `particle_capacity` for workgroup sizing, avoiding wasted GPU threads
/// when deposition or domain exit deactivates particles.
///
/// Default: OFF (the initial benchmark has all particles active, so
/// compaction would add overhead with zero benefit).
fn is_compaction_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("FLEXPART_GPU_COMPACTION")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}


fn dur_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Result of one orchestrated backward timestep.
#[derive(Debug, Clone, PartialEq)]
pub struct BackwardStepReport {
    /// Zero-based step index in this run.
    pub step_index: usize,
    /// Timestamp at which this step was evaluated (`YYYYMMDDHHMMSS`).
    pub timestamp: String,
    /// Simulation time [s since epoch] for this step.
    pub simulation_time_seconds: i64,
    /// Temporal interpolation weight in `[0, 1]` (after clamp/strict handling).
    pub interpolation_alpha: f32,
    /// Number of particles released from receptors during this step.
    pub released_count: usize,
    /// Particle slots injected for this step.
    pub released_slots: Vec<usize>,
    /// Active particle count after GPU readback.
    pub active_particle_count: usize,
    /// Per-source collection summary after this step.
    pub source_collections: BTreeMap<String, BackwardSourceCollection>,
    /// Philox counter after the Langevin update.
    pub next_philox_counter: PhiloxCounter,
}

/// Errors produced by forward time-loop orchestration.
#[derive(Debug, Error)]
pub enum TimeLoopError {
    #[error("invalid timestamp `{value}`: expected 14 digits YYYYMMDDHHMMSS")]
    InvalidTimestamp { value: String },
    #[error("timestamp out of representable range: seconds={seconds}")]
    TimestampOutOfRange { seconds: i64 },
    #[error("invalid config: start timestamp `{start}` must be <= end timestamp `{end}`")]
    InvalidTimeRange { start: String, end: String },
    #[error("invalid backward config: start timestamp `{start}` must be >= end timestamp `{end}`")]
    InvalidBackwardTimeRange { start: String, end: String },
    #[error("invalid config timestep_seconds: {value} (must be > 0)")]
    InvalidTimestep { value: i64 },
    #[error("invalid spatial sort interval_steps: {value} (must be > 0)")]
    InvalidSpatialSortInterval { value: usize },
    #[error(
        "invalid config dry_reference_height_m: {value} (must be finite and strictly positive)"
    )]
    InvalidDryReferenceHeight { value: f32 },
    #[error("forward time-loop has reached end time; no remaining steps")]
    SimulationComplete,
    #[error("forcing length mismatch for `{field}`: expected {expected}, got {actual}")]
    ForcingLengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("backward config requires at least one receptor release")]
    MissingReceptors,
    #[error(
        "invalid receptor `{name}`: {field}={value} ({reason}); expected physically valid bounds"
    )]
    InvalidReceptor {
        name: String,
        field: &'static str,
        value: f64,
        reason: &'static str,
    },
    #[error("invalid source region `{name}` bounds for `{field}`: min {min} > max {max}")]
    InvalidSourceBounds {
        name: String,
        field: &'static str,
        min: f64,
        max: f64,
    },
    #[error("temporal interpolation failed: {0}")]
    Temporal(#[from] TemporalInterpolationError),
    #[error("PBL parameter computation failed: {0}")]
    Pbl(#[from] PblParameterError),
    #[error("particle release failed: {0}")]
    Release(#[from] ReleaseError),
    #[error("particle spatial sorting failed: {0}")]
    ParticleSort(#[from] ParticleSortError),
    #[error("GPU initialization failed: {0}")]
    Gpu(#[from] GpuError),
    #[error("GPU buffer operation failed: {0}")]
    GpuBuffer(#[from] GpuBufferError),
    #[error("GPU advection dispatch failed: {0}")]
    GpuAdvection(#[from] GpuAdvectionError),
    #[error("GPU Hanna dispatch failed: {0}")]
    GpuHanna(#[from] GpuHannaError),
    #[error("GPU Langevin dispatch failed: {0}")]
    GpuLangevin(#[from] GpuLangevinError),
    #[error("GPU PBL diagnostics dispatch failed: {0}")]
    GpuPblDiagnostics(#[from] GpuPblDiagnosticsError),
    #[error("GPU PBL reflection dispatch failed: {0}")]
    GpuPblReflection(#[from] GpuPblReflectionError),
    #[error("GPU dry deposition dispatch failed: {0}")]
    GpuDryDeposition(#[from] GpuDryDepositionError),
    #[error("GPU wet deposition dispatch failed: {0}")]
    GpuWetDeposition(#[from] GpuWetDepositionError),
    #[error("GPU concentration gridding failed: {0}")]
    GpuConcentrationGridding(#[from] GpuConcentrationGriddingError),
    #[error("GPU compaction failed: {0}")]
    GpuCompaction(#[from] GpuCompactionError),
    #[error("GPU fused Hanna+Langevin dispatch failed: {0}")]
    GpuLangevinFused(#[from] GpuLangevinFusedError),
}

/// Forward-mode integration driver orchestrating per-timestep GPU dispatch.
///
/// All GPU buffers and dispatch kernels are pre-allocated at construction time
/// (O-06) based on the known `particle_capacity` and grid dimensions. This
/// eliminates per-step `ensure_*` lazy-init checks.
pub struct ForwardTimeLoopDriver {
    config: ForwardTimeLoopConfig,
    current_time_seconds: i64,
    end_time_seconds: i64,
    step_index: usize,
    philox_counter: PhiloxCounter,
    release_manager: ReleaseManager,
    particle_store: ParticleStore,
    gpu_context: GpuContext,
    particle_buffers: ParticleBuffers,
    /// Single-wind buffers (legacy path, kept for backward compatibility).
    wind_buffers: Option<WindBuffers>,
    advection_dispatch_kernel: Option<AdvectionDispatchKernel>,
    /// Dual-time wind buffers: t0 and t1 uploaded once per met bracket,
    /// GPU interpolates inline using `alpha` (O-02 / Tier 1.2).
    /// Remains `Option` because the 3-D wind grid shape is only known at
    /// the first met bracket upload.
    dual_wind_buffers: Option<DualWindBuffers>,
    /// Pre-allocated dual-wind advection kernel; sampling path resolved from
    /// GPU capabilities at construction time.
    dual_wind_dispatch_kernel: AdvectionDualWindDispatchKernel,
    /// Met bracket tracking: lower bound [s since epoch].
    current_met_t0_seconds: Option<i64>,
    /// Met bracket tracking: upper bound [s since epoch].
    current_met_t1_seconds: Option<i64>,
    /// PBL diagnostics compute kernel (O-04 GPU PBL path).
    pbl_dispatch_kernel: PblDiagnosticsDispatchKernel,
    /// GPU buffer for packed surface meteorological fields (O-04 GPU PBL path).
    surface_field_buffer: SurfaceFieldBuffer,
    /// Double-buffered PBL state (ping-pong A/B, O-03 pipeline overlap).
    /// While the GPU reads from one slot, the CPU can upload the next
    /// step's PBL to the other slot without a data hazard.
    pbl_buffers: [PblBuffers; 2],
    /// Index into `pbl_buffers` for the current CPU write target (alternates 0 ↔ 1).
    pbl_write_index: usize,
    /// Whether a GPU submission is in-flight and has not yet been waited on.
    /// Set `true` after `queue.submit()`; cleared by `device.poll(Wait)` or
    /// by an async readback that internally polls.
    gpu_submission_pending: bool,
    /// `true` when running the full multi-dispatch validation path
    /// (`FLEXPART_GPU_VALIDATION=1`). Uses separated Hanna → Langevin
    /// dispatches so intermediate buffers can be inspected.
    /// Production path uses the fused Hanna+Langevin kernel instead.
    validation_mode: bool,
    /// Fused Hanna+Langevin dispatch kernel (production path).
    /// `None` in validation mode.
    langevin_fused_dispatch_kernel: Option<LangevinFusedDispatchKernel>,
    /// Intermediate Hanna output buffer (validation path only).
    hanna_params_output: Option<HannaParamsOutputBuffer>,
    /// Separated Hanna kernel (validation path only).
    hanna_dispatch_kernel: Option<HannaDispatchKernel>,
    /// Separated Langevin kernel (validation path only).
    langevin_dispatch_kernel: Option<LangevinDispatchKernel>,
    /// Dry deposition IO buffers (both production and validation paths).
    dry_deposition_io: DryDepositionIoBuffers,
    /// Dry deposition kernel (both production and validation paths).
    dry_deposition_dispatch_kernel: DryDepositionDispatchKernel,
    /// Wet deposition IO buffers (both production and validation paths).
    wet_deposition_io: WetDepositionIoBuffers,
    /// Wet deposition kernel (both production and validation paths).
    wet_deposition_dispatch_kernel: WetDepositionDispatchKernel,
    /// Active GRIB prefetch handle, if a background read is in flight.
    grib_prefetch: Option<GribPrefetchHandle>,
    /// Whether active-particle compaction is enabled (O-07).
    /// Controlled by `FLEXPART_GPU_COMPACTION=1`.
    use_compaction: bool,
    /// Pre-allocated compaction pipelines (prefix-sum + gather/reorder).
    /// `None` when compaction is disabled.
    compaction_pipelines: Option<CompactionPipelines>,
    /// Pre-allocated compaction buffers (sized to `particle_capacity`).
    /// `None` when compaction is disabled.
    compaction_buffers: Option<CompactionBuffers>,
}

impl ForwardTimeLoopDriver {
    /// Create a new forward timeloop driver with empty particle store.
    ///
    /// All GPU buffers and dispatch kernels are pre-allocated here based on
    /// `particle_capacity` and `release_grid` dimensions (O-06). This avoids
    /// per-step lazy-init checks in the hot loop.
    pub async fn new(
        config: ForwardTimeLoopConfig,
        releases: &[ReleaseConfig],
        release_grid: GridDomain,
        particle_capacity: usize,
    ) -> Result<Self, TimeLoopError> {
        let (start_time_seconds, end_time_seconds) = validate_config(&config)?;
        let initial_philox_counter = config.initial_philox_counter;
        let met_grid_shape = (release_grid.nx, release_grid.ny);
        let release_manager = ReleaseManager::new(releases, release_grid)?;
        let particle_store = ParticleStore::with_capacity(particle_capacity);
        let gpu_context = GpuContext::new().await?;
        let particle_buffers = ParticleBuffers::from_store(&gpu_context, &particle_store);

        let validation_mode = is_validation_mode();

        let dual_wind_sampling_path = if gpu_context.supports_wind_texture_sampling() {
            WindSamplingPath::SampledTexture3d
        } else {
            WindSamplingPath::BufferStorage
        };
        let dual_wind_dispatch_kernel =
            AdvectionDualWindDispatchKernel::new(&gpu_context, dual_wind_sampling_path);

        let pbl_dispatch_kernel = PblDiagnosticsDispatchKernel::new(&gpu_context);
        let surface_field_buffer = SurfaceFieldBuffer::with_shape(&gpu_context, met_grid_shape);
        let pbl_placeholder_a =
            crate::pbl::PblState::new(met_grid_shape.0, met_grid_shape.1);
        let pbl_placeholder_b =
            crate::pbl::PblState::new(met_grid_shape.0, met_grid_shape.1);
        let pbl_buffers = [
            PblBuffers::from_state(&gpu_context, &pbl_placeholder_a)?,
            PblBuffers::from_state(&gpu_context, &pbl_placeholder_b)?,
        ];

        let langevin_fused_dispatch_kernel = if validation_mode {
            None
        } else {
            Some(LangevinFusedDispatchKernel::new(&gpu_context))
        };

        let (hanna_params_output, hanna_dispatch_kernel, langevin_dispatch_kernel) =
            if validation_mode {
                (
                    Some(HannaParamsOutputBuffer::new(&gpu_context, particle_capacity)?),
                    Some(HannaDispatchKernel::new(&gpu_context)),
                    Some(LangevinDispatchKernel::new(&gpu_context)),
                )
            } else {
                (None, None, None)
            };

        let zeros = vec![0.0_f32; particle_capacity];
        let dry_deposition_io = DryDepositionIoBuffers::from_velocity(&gpu_context, &zeros)?;
        let dry_deposition_dispatch_kernel = DryDepositionDispatchKernel::new(&gpu_context);
        let wet_deposition_io =
            WetDepositionIoBuffers::from_inputs(&gpu_context, &zeros, &zeros)?;
        let wet_deposition_dispatch_kernel = WetDepositionDispatchKernel::new(&gpu_context);

        let use_compaction = is_compaction_enabled();
        let (compaction_pipelines, compaction_buffers) = if use_compaction {
            let pipelines = CompactionPipelines::new(&gpu_context);
            let buffers = CompactionBuffers::new(&gpu_context, particle_capacity)?;
            (Some(pipelines), Some(buffers))
        } else {
            (None, None)
        };

        Ok(Self {
            config,
            current_time_seconds: start_time_seconds,
            end_time_seconds,
            step_index: 0,
            philox_counter: initial_philox_counter,
            release_manager,
            particle_store,
            gpu_context,
            particle_buffers,
            wind_buffers: None,
            advection_dispatch_kernel: None,
            dual_wind_buffers: None,
            dual_wind_dispatch_kernel,
            current_met_t0_seconds: None,
            current_met_t1_seconds: None,
            pbl_dispatch_kernel,
            surface_field_buffer,
            pbl_buffers,
            pbl_write_index: 0,
            gpu_submission_pending: false,
            validation_mode,
            langevin_fused_dispatch_kernel,
            hanna_params_output,
            hanna_dispatch_kernel,
            langevin_dispatch_kernel,
            dry_deposition_io,
            dry_deposition_dispatch_kernel,
            wet_deposition_io,
            wet_deposition_dispatch_kernel,
            grib_prefetch: None,
            use_compaction,
            compaction_pipelines,
            compaction_buffers,
        })
    }

    /// Returns `true` while at least one timestep remains.
    #[must_use]
    pub fn has_remaining_steps(&self) -> bool {
        self.current_time_seconds <= self.end_time_seconds
    }

    /// Current simulation time [s since epoch].
    #[must_use]
    pub fn current_time_seconds(&self) -> i64 {
        self.current_time_seconds
    }

    /// Read-only access to host-side particle storage.
    #[must_use]
    pub fn particle_store(&self) -> &ParticleStore {
        &self.particle_store
    }

    /// Consume the driver and return the host-side particle storage.
    #[must_use]
    pub fn into_particle_store(self) -> ParticleStore {
        self.particle_store
    }

    /// Start prefetching the next meteorological file in a background thread.
    ///
    /// Call this when the next met file path is known (e.g. based on simulation
    /// time approaching a bracket boundary). The prefetch runs on a dedicated OS
    /// thread in parallel with GPU computation. Any previously active prefetch
    /// is replaced (the orphaned thread will run to completion but its result is
    /// discarded).
    pub fn prefetch_met(
        &mut self,
        path: impl AsRef<Path> + Send + 'static,
        expected_grid: Option<Era5GribGridMetadata>,
    ) {
        self.grib_prefetch = Some(GribPrefetchHandle::start(path, expected_grid));
    }

    /// If a prefetch is active and complete, consume and return the result.
    ///
    /// Returns `None` if no prefetch is active or if the background thread has
    /// not finished yet. On success the internal handle is consumed; calling
    /// this again without a new [`prefetch_met`](Self::prefetch_met) returns
    /// `None`.
    pub fn try_consume_prefetch(&mut self) -> Option<Result<Era5MvpSnapshot, Grib2ReaderError>> {
        if self.grib_prefetch.as_ref().is_some_and(|h| h.is_ready()) {
            Some(self.grib_prefetch.take().expect("checked Some above").await_result())
        } else {
            None
        }
    }

    /// Accumulate current particle state into a concentration grid on GPU.
    ///
    /// If a GPU submission is still pending from the last timestep, call
    /// [`finalize`](Self::finalize) first to ensure particle positions are
    /// up to date.
    pub async fn accumulate_concentration_grid(
        &self,
        shape: ConcentrationGridShape,
        params: ConcentrationGriddingParams,
    ) -> Result<ConcentrationGridOutput, TimeLoopError> {
        accumulate_concentration_grid_gpu(&self.gpu_context, &self.particle_buffers, shape, params)
            .await
            .map_err(Into::into)
    }

    /// Run one orchestrated timestep with CPU/GPU overlap (O-03).
    ///
    /// The method is split into four phases:
    ///
    /// 1. **CPU prep** — surface interpolation, PBL computation, buffer
    ///    staging. Runs concurrently with the *previous* step's in-flight
    ///    GPU work.
    /// 2. **Wait previous GPU** — block until the previous submission
    ///    completes (no-op on the first step).
    /// 3. **Encode + submit** — build the command encoder and call
    ///    `queue.submit()`. Returns immediately; GPU starts asynchronously.
    /// 4. **Optional wait + readback** — if per-step synchronization or
    ///    deposition probability collection is enabled, block until *this*
    ///    step's GPU work finishes and download results.
    ///
    /// PBL buffers use a ping-pong double buffer so that CPU uploads for
    /// step N+1 never race with GPU reads from step N.
    pub async fn run_timestep(
        &mut self,
        met: &MetTimeBracket<'_>,
        forcing: &ForwardStepForcing,
    ) -> Result<ForwardStepReport, TimeLoopError> {
        if !self.has_remaining_steps() {
            return Err(TimeLoopError::SimulationComplete);
        }
        self.apply_spatial_sort_if_enabled()?;

        let profiling = is_profiling_enabled();
        let total_start = profiling.then(Instant::now);

        // ── Phase 1: CPU prep (overlaps with previous GPU submission) ──

        let timestamp = format_timestamp_seconds(self.current_time_seconds)?;
        let release_report = self.release_manager.inject_and_upload_for_time(
            &timestamp,
            &mut self.particle_store,
            &self.particle_buffers,
            &self.gpu_context,
        )?;

        // O-07: After release, widen the dispatch window to cover both the
        // compacted active prefix from the previous step and newly released
        // particles (which the release manager placed at the first free
        // slots immediately after the active prefix).
        if self.use_compaction {
            self.particle_buffers
                .set_dispatch_count(self.particle_store.active_count());
        }

        let interpolation_alpha = interpolation_alpha(
            met.time_t0_seconds,
            met.time_t1_seconds,
            self.current_time_seconds,
            self.config.time_bounds_behavior,
        )?;

        // O-02: upload wind_t0 and wind_t1 once per met bracket.
        let t = profiling.then(Instant::now);
        self.upload_dual_wind_if_bracket_changed(met)?;
        let wind_upload_dur = t.map_or(Duration::ZERO, |t| t.elapsed());
        let wind_interp_dur = Duration::ZERO;

        let t = profiling.then(Instant::now);
        let interpolated_surface = interpolate_surface_fields_linear(
            met.surface_t0,
            met.surface_t1,
            met.time_t0_seconds,
            met.time_t1_seconds,
            self.current_time_seconds,
            self.config.time_bounds_behavior,
        )?;
        let surf_interp_dur = t.map_or(Duration::ZERO, |t| t.elapsed());

        let use_gpu_pbl = is_gpu_pbl_enabled();
        let (pbl_dur, pbl_upload_dur) = if use_gpu_pbl {
            let t = profiling.then(Instant::now);
            self.surface_field_buffer
                .upload(&self.gpu_context, &interpolated_surface)?;
            let dur = t.map_or(Duration::ZERO, |t| t.elapsed());
            (dur, Duration::ZERO)
        } else {
            let t = profiling.then(Instant::now);
            let computed_pbl = compute_pbl_parameters_from_met(
                PblMetInputGrids {
                    surface: &interpolated_surface,
                    profile: None,
                },
                self.config.pbl_options,
            )?;
            let pbl_dur = t.map_or(Duration::ZERO, |t| t.elapsed());

            let t = profiling.then(Instant::now);
            self.pbl_buffers[self.pbl_write_index]
                .upload_state(&self.gpu_context, &computed_pbl.pbl_state)?;
            let pbl_upload_dur = t.map_or(Duration::ZERO, |t| t.elapsed());
            (pbl_dur, pbl_upload_dur)
        };

        let step_dt_seconds = timestep_seconds_f32(self.config.timestep_seconds)?;

        let skip_dry_deposition = forcing.dry_deposition_velocity_m_s.is_zero();
        let skip_wet_deposition = forcing.wet_scavenging_coefficient_s_inv.is_zero()
            && forcing.wet_precipitating_fraction.is_zero();

        let t = profiling.then(Instant::now);
        // Use buffer capacity (not dispatch count) for forcing materialization:
        // deposition I/O buffers were pre-allocated for the full capacity.
        let slot_count = self.particle_buffers.capacity();
        if !skip_dry_deposition {
            let dry_velocity = forcing
                .dry_deposition_velocity_m_s
                .materialize(slot_count, "dry_deposition_velocity_m_s")?;
            self.dry_deposition_io
                .upload_deposition_velocity(&self.gpu_context, &dry_velocity)?;
        }
        if !skip_wet_deposition {
            let wet_scavenging = forcing
                .wet_scavenging_coefficient_s_inv
                .materialize(slot_count, "wet_scavenging_coefficient_s_inv")?;
            let wet_fraction = forcing
                .wet_precipitating_fraction
                .materialize(slot_count, "wet_precipitating_fraction")?;
            self.wet_deposition_io
                .upload_scavenging_coefficient(&self.gpu_context, &wet_scavenging)?;
            self.wet_deposition_io
                .upload_precipitating_fraction(&self.gpu_context, &wet_fraction)?;
        }
        let forcing_dur = t.map_or(Duration::ZERO, |t| t.elapsed());
        let dep_upload_dur = Duration::ZERO;

        // ── Phase 2: Wait for previous GPU submission ──────────────────

        let wait_prev_dur = if self.gpu_submission_pending {
            let t = profiling.then(Instant::now);
            self.gpu_context.device.poll(wgpu::Maintain::Wait);
            self.gpu_submission_pending = false;
            t.map_or(Duration::ZERO, |t| t.elapsed())
        } else {
            Duration::ZERO
        };

        // ── Phase 3: Encode + submit (non-blocking) ───────────────────

        let dry_params = DryDepositionStepParams {
            dt_seconds: step_dt_seconds,
            reference_height_m: self.config.dry_reference_height_m,
        };
        let wet_params = WetDepositionStepParams {
            dt_seconds: step_dt_seconds,
        };
        let t = profiling.then(Instant::now);
        let next_philox_counter = {
            let mut encoder =
                self.gpu_context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("forward_timeloop_step_encoder"),
                    });

            // ── PBL diagnostics (both paths) ────────────────────────
            if use_gpu_pbl {
                encode_pbl_diagnostics_gpu_with_kernel(
                    &self.gpu_context,
                    &self.surface_field_buffer,
                    &self.pbl_buffers[self.pbl_write_index],
                    &self.config.pbl_options,
                    &self.pbl_dispatch_kernel,
                    &mut encoder,
                )?;
            }

            // ── Advection (both paths) ───────────────────────────────
            let dual_wind = self
                .dual_wind_buffers
                .as_ref()
                .expect("dual wind buffers uploaded by upload_dual_wind_if_bracket_changed");
            encode_advection_dual_wind_gpu_with_kernel(
                &self.gpu_context,
                &self.particle_buffers,
                dual_wind,
                interpolation_alpha,
                step_dt_seconds,
                self.config.velocity_to_grid_scale,
                &self.dual_wind_dispatch_kernel,
                &mut encoder,
            )?;

            // ── Hanna + Langevin turbulence ──────────────────────────
            let pbl_buffers = &self.pbl_buffers[self.pbl_write_index];
            let langevin_step = LangevinStep {
                dt_seconds: step_dt_seconds,
                rho_grad_over_rho: forcing.rho_grad_over_rho,
                n_substeps: 4,
                min_height_m: 0.01,
            };

            let next_philox_counter = if !self.validation_mode {
                // ── Production: fused Hanna+Langevin (single dispatch) ──
                encode_langevin_fused_gpu(
                    &self.gpu_context,
                    &self.particle_buffers,
                    pbl_buffers,
                    langevin_step,
                    self.config.philox_key,
                    self.philox_counter,
                    self.langevin_fused_dispatch_kernel
                        .as_ref()
                        .expect("fused kernel allocated in production mode"),
                    &mut encoder,
                )?
            } else {
                // ── Validation: separated Hanna → Langevin dispatches ──
                let hanna_output = self
                    .hanna_params_output
                    .as_ref()
                    .expect("hanna output allocated in validation mode");
                encode_hanna_params_gpu_with_kernel(
                    &self.gpu_context,
                    &self.particle_buffers,
                    pbl_buffers,
                    hanna_output,
                    self.hanna_dispatch_kernel
                        .as_ref()
                        .expect("hanna kernel allocated in validation mode"),
                    &mut encoder,
                )?;
                encode_update_particles_turbulence_langevin_gpu_with_hanna_buffer_and_kernel(
                    &self.gpu_context,
                    &self.particle_buffers,
                    &hanna_output.buffer,
                    hanna_output.particle_count(),
                    langevin_step,
                    self.config.philox_key,
                    self.philox_counter,
                    self.langevin_dispatch_kernel
                        .as_ref()
                        .expect("langevin kernel allocated in validation mode"),
                    &mut encoder,
                )?
            };

            // ── Deposition (both paths) ──────────────────────────────
            if !skip_dry_deposition {
                encode_dry_deposition_probability_gpu_with_kernel(
                    &self.gpu_context,
                    &self.particle_buffers,
                    &self.dry_deposition_io,
                    dry_params,
                    &self.dry_deposition_dispatch_kernel,
                    &mut encoder,
                )?;
            }
            if !skip_wet_deposition {
                encode_wet_deposition_probability_gpu_with_kernel(
                    &self.gpu_context,
                    &self.particle_buffers,
                    &self.wet_deposition_io,
                    wet_params,
                    &self.wet_deposition_dispatch_kernel,
                    &mut encoder,
                )?;
            }

            // O-07: Encode compaction + gather/reorder after all physics
            // passes. Compaction reads particle flags set by deposition and
            // reorders the buffer so active particles are contiguous.
            if self.use_compaction {
                encode_compaction_with_reorder(
                    &self.gpu_context,
                    &self.particle_buffers,
                    self.compaction_buffers.as_ref().expect(
                        "compaction buffers allocated when compaction is enabled",
                    ),
                    self.compaction_pipelines.as_ref().expect(
                        "compaction pipelines allocated when compaction is enabled",
                    ),
                    &mut encoder,
                )?;
            }

            self.gpu_context.queue.submit(Some(encoder.finish()));
            next_philox_counter
        };
        let gpu_encode_dur = t.map_or(Duration::ZERO, |t| t.elapsed());

        self.gpu_submission_pending = true;
        self.pbl_write_index = 1 - self.pbl_write_index;

        // ── Phase 4: Optional wait + readback ──────────────────────────
        // When profiling, we always poll to measure `gpu_exec`. When
        // collecting deposition probabilities or syncing the particle
        // store, the async download methods poll internally.

        let gpu_exec_dur = if profiling {
            let t = Instant::now();
            self.gpu_context.device.poll(wgpu::Maintain::Wait);
            self.gpu_submission_pending = false;
            t.elapsed()
        } else {
            Duration::ZERO
        };

        self.philox_counter = next_philox_counter;

        let dry_probability = if self.config.collect_deposition_probabilities_each_step
            && !skip_dry_deposition
        {
            let result = self
                .dry_deposition_io
                .download_probabilities(&self.gpu_context)
                .await?;
            self.gpu_submission_pending = false;
            result
        } else {
            Vec::new()
        };
        let wet_probability = if self.config.collect_deposition_probabilities_each_step
            && !skip_wet_deposition
        {
            let result = self
                .wet_deposition_io
                .download_probabilities(&self.gpu_context)
                .await?;
            self.gpu_submission_pending = false;
            result
        } else {
            Vec::new()
        };

        let t_compact = profiling.then(Instant::now);
        if self.use_compaction {
            if self.gpu_submission_pending {
                self.gpu_context.device.poll(wgpu::Maintain::Wait);
                self.gpu_submission_pending = false;
            }
            let active_count = self
                .compaction_buffers
                .as_ref()
                .expect("compaction buffers allocated when compaction is enabled")
                .download_active_count(&self.gpu_context)
                .await? as usize;
            self.particle_buffers.set_dispatch_count(active_count);
            self.particle_store.reset_after_compaction(active_count);
        }
        if self.config.sync_particle_store_each_step {
            if self.gpu_submission_pending {
                self.gpu_context.device.poll(wgpu::Maintain::Wait);
                self.gpu_submission_pending = false;
            }
            self.sync_store_from_gpu().await?;
        }
        let compaction_dur = t_compact.map_or(Duration::ZERO, |t| t.elapsed());

        let timing = if profiling {
            let total_dur = total_start.map_or(Duration::ZERO, |t| t.elapsed());
            let report = StepTimingReport {
                wind_interp_ms: dur_ms(wind_interp_dur),
                surf_interp_ms: dur_ms(surf_interp_dur),
                pbl_ms: dur_ms(pbl_dur),
                wind_upload_ms: dur_ms(wind_upload_dur),
                pbl_upload_ms: dur_ms(pbl_upload_dur),
                forcing_ms: dur_ms(forcing_dur),
                dep_upload_ms: dur_ms(dep_upload_dur),
                wait_prev_gpu_ms: dur_ms(wait_prev_dur),
                gpu_encode_ms: dur_ms(gpu_encode_dur),
                gpu_exec_ms: dur_ms(gpu_exec_dur),
                compaction_ms: dur_ms(compaction_dur),
                total_ms: dur_ms(total_dur),
            };
            report.print_summary(self.step_index);
            Some(report)
        } else {
            None
        };

        let report = ForwardStepReport {
            step_index: self.step_index,
            timestamp,
            simulation_time_seconds: self.current_time_seconds,
            interpolation_alpha,
            released_count: release_report.released_count,
            released_slots: release_report.released_slots,
            // In performance mode without per-step host sync, this value reflects
            // the host-side cached count and can lag GPU-side deactivations.
            active_particle_count: self.particle_store.active_count(),
            dry_deposition_probability: dry_probability,
            wet_deposition_probability: wet_probability,
            next_philox_counter: self.philox_counter,
            timing,
        };

        self.step_index += 1;
        self.current_time_seconds = self
            .current_time_seconds
            .saturating_add(self.config.timestep_seconds);

        Ok(report)
    }

    /// Run until `end_timestamp` using fixed forcing and one met bracket.
    ///
    /// Calls [`finalize`](Self::finalize) after the last step to drain any
    /// pending GPU submission.
    pub async fn run_to_end(
        &mut self,
        met: &MetTimeBracket<'_>,
        forcing: &ForwardStepForcing,
    ) -> Result<Vec<ForwardStepReport>, TimeLoopError> {
        let mut reports = Vec::new();
        while self.has_remaining_steps() {
            reports.push(self.run_timestep(met, forcing).await?);
        }
        self.finalize().await?;
        Ok(reports)
    }

    /// Drain the last pending GPU submission (if any).
    ///
    /// Must be called after the final [`run_timestep`](Self::run_timestep)
    /// when the caller drives the loop manually. [`run_to_end`](Self::run_to_end)
    /// calls this automatically. Safe to call multiple times.
    pub async fn finalize(&mut self) -> Result<(), TimeLoopError> {
        if self.gpu_submission_pending {
            self.gpu_context.device.poll(wgpu::Maintain::Wait);
            self.gpu_submission_pending = false;
        }
        Ok(())
    }

    /// Legacy single-wind upload path, kept for backward compatibility testing.
    #[allow(dead_code)]
    fn ensure_wind_buffers(&mut self, wind: &WindField3D) -> Result<(), TimeLoopError> {
        if let Some(buffers) = &self.wind_buffers {
            buffers.upload_field(&self.gpu_context, wind)?;
            return Ok(());
        }
        self.wind_buffers = Some(WindBuffers::from_field(&self.gpu_context, wind)?);
        Ok(())
    }

    /// Legacy single-wind kernel setup, kept for backward compatibility testing.
    #[allow(dead_code)]
    fn ensure_advection_dispatch_kernel(&mut self, sampling_path: WindSamplingPath) {
        let recreate = match self.advection_dispatch_kernel.as_ref() {
            Some(kernel) => kernel.sampling_path() != sampling_path,
            None => true,
        };
        if recreate {
            self.advection_dispatch_kernel = Some(AdvectionDispatchKernel::new(
                &self.gpu_context,
                sampling_path,
            ));
        }
    }

    /// Upload wind_t0 and wind_t1 to dual-time GPU buffers if the met bracket
    /// changed. Creates the buffers on first call (the 3-D wind grid shape is
    /// only known at first met bracket). Skips entirely when the bracket is
    /// unchanged, since only `alpha` varies within a bracket.
    fn upload_dual_wind_if_bracket_changed(
        &mut self,
        met: &MetTimeBracket<'_>,
    ) -> Result<(), TimeLoopError> {
        let bracket_changed = self.current_met_t0_seconds != Some(met.time_t0_seconds)
            || self.current_met_t1_seconds != Some(met.time_t1_seconds);
        if !bracket_changed {
            return Ok(());
        }

        if let Some(buffers) = &self.dual_wind_buffers {
            buffers.upload_t0(&self.gpu_context, met.wind_t0)?;
            buffers.upload_t1(&self.gpu_context, met.wind_t1)?;
        } else {
            self.dual_wind_buffers = Some(DualWindBuffers::from_fields(
                &self.gpu_context,
                met.wind_t0,
                met.wind_t1,
            )?);
        }

        self.current_met_t0_seconds = Some(met.time_t0_seconds);
        self.current_met_t1_seconds = Some(met.time_t1_seconds);
        Ok(())
    }

    async fn sync_store_from_gpu(&mut self) -> Result<(), TimeLoopError> {
        let updated = self
            .particle_buffers
            .download_particles(&self.gpu_context)
            .await?;
        self.particle_store.as_mut_slice().copy_from_slice(&updated);
        self.particle_store.recount_active();
        Ok(())
    }

    fn apply_spatial_sort_if_enabled(&mut self) -> Result<(), TimeLoopError> {
        let Some(sort_config) = self.config.spatial_sort else {
            return Ok(());
        };
        if self.step_index % sort_config.interval_steps != 0 {
            return Ok(());
        }

        let reorder = self
            .particle_store
            .sort_spatially(sort_config.sort_options)?;
        if !reorder.is_identity() {
            self.particle_buffers
                .upload_store(&self.gpu_context, &self.particle_store)?;
        }
        Ok(())
    }
}

/// Backward-mode integration driver (`ldirect = -1`) for receptor attribution.
///
/// Always uses the multi-dispatch path (separated Hanna, Langevin, dry/wet
/// deposition) which is the same as the forward driver's validation mode.
/// This gives full per-stage inspectability for scientific analysis.
///
/// Optimizations applied:
/// - Dual-wind bracket tracking (O-02): `wind_t0`/`t1` uploaded once per met
///   bracket, GPU interpolates inline.
/// - GPU PBL diagnostics (O-04): surface fields uploaded, PBL computed on
///   GPU. Falls back to CPU when `FLEXPART_GPU_PBL_CPU=1`.
/// - Batched command encoder: all dispatches in a single `queue.submit()`.
/// - Zero-forcing skip: deposition dispatches skipped when forcing is zero.
///
/// No pipeline overlap or compaction — backward mode always synchronizes
/// every step for source attribution readback.
pub struct BackwardTimeLoopDriver {
    config: BackwardTimeLoopConfig,
    current_time_seconds: i64,
    end_time_seconds: i64,
    step_index: usize,
    philox_counter: PhiloxCounter,
    release_grid: GridDomain,
    release_manager: ReleaseManager,
    particle_store: ParticleStore,
    gpu_context: GpuContext,
    particle_buffers: ParticleBuffers,
    /// Dual-time wind buffers (O-02): t0 and t1 uploaded once per met bracket.
    /// `None` until the first bracket upload (3-D grid shape unknown before).
    dual_wind_buffers: Option<DualWindBuffers>,
    dual_wind_dispatch_kernel: AdvectionDualWindDispatchKernel,
    current_met_t0_seconds: Option<i64>,
    current_met_t1_seconds: Option<i64>,
    /// GPU PBL diagnostics kernel (O-04).
    pbl_dispatch_kernel: PblDiagnosticsDispatchKernel,
    /// GPU buffer for packed surface meteorological fields (O-04).
    surface_field_buffer: SurfaceFieldBuffer,
    /// Single PBL buffer (no double-buffering needed — backward syncs every step).
    pbl_buffers: PblBuffers,
    hanna_params_output: HannaParamsOutputBuffer,
    hanna_dispatch_kernel: HannaDispatchKernel,
    langevin_dispatch_kernel: LangevinDispatchKernel,
    dry_deposition_io: DryDepositionIoBuffers,
    dry_deposition_dispatch_kernel: DryDepositionDispatchKernel,
    wet_deposition_io: WetDepositionIoBuffers,
    wet_deposition_dispatch_kernel: WetDepositionDispatchKernel,
}

impl BackwardTimeLoopDriver {
    /// Create a new backward timeloop driver with empty particle store.
    ///
    /// All GPU buffers and dispatch kernels are pre-allocated here based on
    /// `particle_capacity` and `release_grid` dimensions, following the
    /// forward driver pattern (O-06).
    pub async fn new(
        config: BackwardTimeLoopConfig,
        release_grid: GridDomain,
        particle_capacity: usize,
    ) -> Result<Self, TimeLoopError> {
        let (start_time_seconds, end_time_seconds) = validate_backward_config(&config)?;
        let initial_philox_counter = config.initial_philox_counter;
        let met_grid_shape = (release_grid.nx, release_grid.ny);
        let release_manager = ReleaseManager::new(
            &build_receptor_release_configs(&config.receptors, &config.start_timestamp),
            release_grid.clone(),
        )?;
        let particle_store = ParticleStore::with_capacity(particle_capacity);
        let gpu_context = GpuContext::new().await?;
        let particle_buffers = ParticleBuffers::from_store(&gpu_context, &particle_store);

        let dual_wind_sampling_path = if gpu_context.supports_wind_texture_sampling() {
            WindSamplingPath::SampledTexture3d
        } else {
            WindSamplingPath::BufferStorage
        };
        let dual_wind_dispatch_kernel =
            AdvectionDualWindDispatchKernel::new(&gpu_context, dual_wind_sampling_path);

        let pbl_dispatch_kernel = PblDiagnosticsDispatchKernel::new(&gpu_context);
        let surface_field_buffer = SurfaceFieldBuffer::with_shape(&gpu_context, met_grid_shape);
        let pbl_placeholder = crate::pbl::PblState::new(met_grid_shape.0, met_grid_shape.1);
        let pbl_buffers = PblBuffers::from_state(&gpu_context, &pbl_placeholder)?;

        let hanna_params_output = HannaParamsOutputBuffer::new(&gpu_context, particle_capacity)?;
        let hanna_dispatch_kernel = HannaDispatchKernel::new(&gpu_context);
        let langevin_dispatch_kernel = LangevinDispatchKernel::new(&gpu_context);

        let zeros = vec![0.0_f32; particle_capacity];
        let dry_deposition_io = DryDepositionIoBuffers::from_velocity(&gpu_context, &zeros)?;
        let dry_deposition_dispatch_kernel = DryDepositionDispatchKernel::new(&gpu_context);
        let wet_deposition_io =
            WetDepositionIoBuffers::from_inputs(&gpu_context, &zeros, &zeros)?;
        let wet_deposition_dispatch_kernel = WetDepositionDispatchKernel::new(&gpu_context);

        Ok(Self {
            config,
            current_time_seconds: start_time_seconds,
            end_time_seconds,
            step_index: 0,
            philox_counter: initial_philox_counter,
            release_grid,
            release_manager,
            particle_store,
            gpu_context,
            particle_buffers,
            dual_wind_buffers: None,
            dual_wind_dispatch_kernel,
            current_met_t0_seconds: None,
            current_met_t1_seconds: None,
            pbl_dispatch_kernel,
            surface_field_buffer,
            pbl_buffers,
            hanna_params_output,
            hanna_dispatch_kernel,
            langevin_dispatch_kernel,
            dry_deposition_io,
            dry_deposition_dispatch_kernel,
            wet_deposition_io,
            wet_deposition_dispatch_kernel,
        })
    }

    /// Returns `true` while at least one timestep remains.
    #[must_use]
    pub fn has_remaining_steps(&self) -> bool {
        self.current_time_seconds >= self.end_time_seconds
    }

    /// Current simulation time [s since epoch].
    #[must_use]
    pub fn current_time_seconds(&self) -> i64 {
        self.current_time_seconds
    }

    /// Read-only access to host-side particle storage.
    #[must_use]
    pub fn particle_store(&self) -> &ParticleStore {
        &self.particle_store
    }

    /// Consume the driver and return host-side particle storage.
    #[must_use]
    pub fn into_particle_store(self) -> ParticleStore {
        self.particle_store
    }

    /// Run one orchestrated backward timestep.
    ///
    /// All GPU dispatches are encoded into a single command encoder and
    /// submitted in one `queue.submit()` call. The driver always waits for
    /// GPU completion and downloads particle state, because source
    /// attribution needs host-side particle data.
    pub async fn run_timestep(
        &mut self,
        met: &MetTimeBracket<'_>,
        forcing: &ForwardStepForcing,
    ) -> Result<BackwardStepReport, TimeLoopError> {
        if !self.has_remaining_steps() {
            return Err(TimeLoopError::SimulationComplete);
        }

        let timestamp = format_timestamp_seconds(self.current_time_seconds)?;
        let release_report = self.release_manager.inject_and_upload_for_time(
            &timestamp,
            &mut self.particle_store,
            &self.particle_buffers,
            &self.gpu_context,
        )?;

        let interpolation_alpha = interpolation_alpha(
            met.time_t0_seconds,
            met.time_t1_seconds,
            self.current_time_seconds,
            self.config.time_bounds_behavior,
        )?;

        // O-02: upload wind_t0/t1 once per met bracket.
        self.upload_dual_wind_if_bracket_changed(met)?;

        let interpolated_surface = interpolate_surface_fields_linear(
            met.surface_t0,
            met.surface_t1,
            met.time_t0_seconds,
            met.time_t1_seconds,
            self.current_time_seconds,
            self.config.time_bounds_behavior,
        )?;

        // O-04: GPU PBL by default; CPU fallback with FLEXPART_GPU_PBL_CPU=1.
        let use_gpu_pbl = is_gpu_pbl_enabled();
        if use_gpu_pbl {
            self.surface_field_buffer
                .upload(&self.gpu_context, &interpolated_surface)?;
        } else {
            let computed_pbl = compute_pbl_parameters_from_met(
                PblMetInputGrids {
                    surface: &interpolated_surface,
                    profile: None,
                },
                self.config.pbl_options,
            )?;
            self.pbl_buffers
                .upload_state(&self.gpu_context, &computed_pbl.pbl_state)?;
        }

        let dt_seconds = timestep_seconds_f32(self.config.timestep_seconds)?;

        let skip_dry_deposition = forcing.dry_deposition_velocity_m_s.is_zero();
        let skip_wet_deposition = forcing.wet_scavenging_coefficient_s_inv.is_zero()
            && forcing.wet_precipitating_fraction.is_zero();

        let slot_count = self.particle_buffers.capacity();
        if !skip_dry_deposition {
            let dry_velocity = forcing
                .dry_deposition_velocity_m_s
                .materialize(slot_count, "dry_deposition_velocity_m_s")?;
            self.dry_deposition_io
                .upload_deposition_velocity(&self.gpu_context, &dry_velocity)?;
        }
        if !skip_wet_deposition {
            let wet_scavenging = forcing
                .wet_scavenging_coefficient_s_inv
                .materialize(slot_count, "wet_scavenging_coefficient_s_inv")?;
            let wet_fraction = forcing
                .wet_precipitating_fraction
                .materialize(slot_count, "wet_precipitating_fraction")?;
            self.wet_deposition_io
                .upload_scavenging_coefficient(&self.gpu_context, &wet_scavenging)?;
            self.wet_deposition_io
                .upload_precipitating_fraction(&self.gpu_context, &wet_fraction)?;
        }

        // ── Encode all GPU dispatches in a single command encoder ──────
        let next_philox_counter = {
            let mut encoder =
                self.gpu_context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("backward_timeloop_step_encoder"),
                    });

            if use_gpu_pbl {
                encode_pbl_diagnostics_gpu_with_kernel(
                    &self.gpu_context,
                    &self.surface_field_buffer,
                    &self.pbl_buffers,
                    &self.config.pbl_options,
                    &self.pbl_dispatch_kernel,
                    &mut encoder,
                )?;
            }

            // Advection: negative dt for backward time direction.
            let dual_wind = self
                .dual_wind_buffers
                .as_ref()
                .expect("dual wind buffers uploaded by upload_dual_wind_if_bracket_changed");
            encode_advection_dual_wind_gpu_with_kernel(
                &self.gpu_context,
                &self.particle_buffers,
                dual_wind,
                interpolation_alpha,
                TimeDirection::Backward.advection_dt_seconds(dt_seconds),
                self.config.velocity_to_grid_scale,
                &self.dual_wind_dispatch_kernel,
                &mut encoder,
            )?;

            // B-01 MVP: turbulence and deposition use positive dt magnitude.
            let langevin_step = LangevinStep {
                dt_seconds,
                rho_grad_over_rho: forcing.rho_grad_over_rho,
                n_substeps: 4,
                min_height_m: 0.01,
            };

            encode_hanna_params_gpu_with_kernel(
                &self.gpu_context,
                &self.particle_buffers,
                &self.pbl_buffers,
                &self.hanna_params_output,
                &self.hanna_dispatch_kernel,
                &mut encoder,
            )?;

            let next_pc =
                encode_update_particles_turbulence_langevin_gpu_with_hanna_buffer_and_kernel(
                    &self.gpu_context,
                    &self.particle_buffers,
                    &self.hanna_params_output.buffer,
                    self.hanna_params_output.particle_count(),
                    langevin_step,
                    self.config.philox_key,
                    self.philox_counter,
                    &self.langevin_dispatch_kernel,
                    &mut encoder,
                )?;

            if !skip_dry_deposition {
                encode_dry_deposition_probability_gpu_with_kernel(
                    &self.gpu_context,
                    &self.particle_buffers,
                    &self.dry_deposition_io,
                    DryDepositionStepParams {
                        dt_seconds,
                        reference_height_m: self.config.dry_reference_height_m,
                    },
                    &self.dry_deposition_dispatch_kernel,
                    &mut encoder,
                )?;
            }

            if !skip_wet_deposition {
                encode_wet_deposition_probability_gpu_with_kernel(
                    &self.gpu_context,
                    &self.particle_buffers,
                    &self.wet_deposition_io,
                    WetDepositionStepParams { dt_seconds },
                    &self.wet_deposition_dispatch_kernel,
                    &mut encoder,
                )?;
            }

            self.gpu_context.queue.submit(Some(encoder.finish()));
            next_pc
        };

        self.gpu_context.device.poll(wgpu::Maintain::Wait);
        self.philox_counter = next_philox_counter;

        self.sync_store_from_gpu().await?;
        let source_collections = collect_source_collections(
            &self.particle_store,
            &self.release_grid,
            &self.config.source_regions,
        );
        let report = BackwardStepReport {
            step_index: self.step_index,
            timestamp,
            simulation_time_seconds: self.current_time_seconds,
            interpolation_alpha,
            released_count: release_report.released_count,
            released_slots: release_report.released_slots,
            active_particle_count: self.particle_store.active_count(),
            source_collections,
            next_philox_counter: self.philox_counter,
        };

        self.step_index += 1;
        self.current_time_seconds = self
            .current_time_seconds
            .saturating_sub(self.config.timestep_seconds);

        Ok(report)
    }

    /// Run until `end_timestamp` using fixed forcing and one met bracket.
    pub async fn run_to_end(
        &mut self,
        met: &MetTimeBracket<'_>,
        forcing: &ForwardStepForcing,
    ) -> Result<Vec<BackwardStepReport>, TimeLoopError> {
        let mut reports = Vec::new();
        while self.has_remaining_steps() {
            reports.push(self.run_timestep(met, forcing).await?);
        }
        Ok(reports)
    }

    /// Upload wind_t0 and wind_t1 to dual-time GPU buffers if the met bracket
    /// changed. Creates buffers on first call. Skips when the bracket is
    /// unchanged (only `alpha` varies within a bracket).
    fn upload_dual_wind_if_bracket_changed(
        &mut self,
        met: &MetTimeBracket<'_>,
    ) -> Result<(), TimeLoopError> {
        let bracket_changed = self.current_met_t0_seconds != Some(met.time_t0_seconds)
            || self.current_met_t1_seconds != Some(met.time_t1_seconds);
        if !bracket_changed {
            return Ok(());
        }

        if let Some(buffers) = &self.dual_wind_buffers {
            buffers.upload_t0(&self.gpu_context, met.wind_t0)?;
            buffers.upload_t1(&self.gpu_context, met.wind_t1)?;
        } else {
            self.dual_wind_buffers = Some(DualWindBuffers::from_fields(
                &self.gpu_context,
                met.wind_t0,
                met.wind_t1,
            )?);
        }

        self.current_met_t0_seconds = Some(met.time_t0_seconds);
        self.current_met_t1_seconds = Some(met.time_t1_seconds);
        Ok(())
    }

    async fn sync_store_from_gpu(&mut self) -> Result<(), TimeLoopError> {
        let updated = self
            .particle_buffers
            .download_particles(&self.gpu_context)
            .await?;
        self.particle_store.as_mut_slice().copy_from_slice(&updated);
        self.particle_store.recount_active();
        Ok(())
    }
}

fn collect_source_collections(
    store: &ParticleStore,
    grid: &GridDomain,
    source_regions: &[BackwardSourceRegionConfig],
) -> BTreeMap<String, BackwardSourceCollection> {
    let mut collections: BTreeMap<String, BackwardSourceCollection> = source_regions
        .iter()
        .map(|source| (source.name.clone(), BackwardSourceCollection::default()))
        .collect();
    if source_regions.is_empty() {
        return collections;
    }

    for particle in store.as_slice() {
        if !particle.is_active() {
            continue;
        }
        for source in source_regions {
            if source.contains_particle(particle, grid) {
                if let Some(collection) = collections.get_mut(&source.name) {
                    collection.hit_count += 1;
                    collection.total_mass_kg += particle.mass[0];
                }
            }
        }
    }

    collections
}

fn build_receptor_release_configs(
    receptors: &[BackwardReceptorConfig],
    release_timestamp: &str,
) -> Vec<ReleaseConfig> {
    receptors
        .iter()
        .map(|receptor| ReleaseConfig {
            name: receptor.name.clone(),
            start_time: release_timestamp.to_string(),
            end_time: release_timestamp.to_string(),
            lon: receptor.lon,
            lat: receptor.lat,
            z_min: receptor.z_m,
            z_max: receptor.z_m,
            mass_kg: receptor.mass_kg,
            particle_count: receptor.particle_count,
            raw: BTreeMap::new(),
        })
        .collect()
}

fn validate_config(config: &ForwardTimeLoopConfig) -> Result<(i64, i64), TimeLoopError> {
    let start_seconds = parse_timestamp_seconds(&config.start_timestamp)?;
    let end_seconds = parse_timestamp_seconds(&config.end_timestamp)?;
    if start_seconds > end_seconds {
        return Err(TimeLoopError::InvalidTimeRange {
            start: config.start_timestamp.to_string(),
            end: config.end_timestamp.to_string(),
        });
    }
    if config.timestep_seconds <= 0 {
        return Err(TimeLoopError::InvalidTimestep {
            value: config.timestep_seconds,
        });
    }
    if !config.dry_reference_height_m.is_finite() || config.dry_reference_height_m <= 0.0 {
        return Err(TimeLoopError::InvalidDryReferenceHeight {
            value: config.dry_reference_height_m,
        });
    }
    if let Some(spatial_sort) = config.spatial_sort {
        if spatial_sort.interval_steps == 0 {
            return Err(TimeLoopError::InvalidSpatialSortInterval { value: 0 });
        }
        let _ = crate::particles::morton_key_from_position(
            0.0,
            0.0,
            0.0,
            crate::particles::SpatialSortBounds {
                x_min: 0.0,
                x_max: 1.0,
                y_min: 0.0,
                y_max: 1.0,
                z_min: 0.0,
                z_max: 1.0,
            },
            spatial_sort.sort_options.bits_per_axis,
        )?;
    }
    let _ = timestep_seconds_f32(config.timestep_seconds)?;
    Ok((start_seconds, end_seconds))
}

fn validate_backward_config(config: &BackwardTimeLoopConfig) -> Result<(i64, i64), TimeLoopError> {
    let start_seconds = parse_timestamp_seconds(&config.start_timestamp)?;
    let end_seconds = parse_timestamp_seconds(&config.end_timestamp)?;
    if start_seconds < end_seconds {
        return Err(TimeLoopError::InvalidBackwardTimeRange {
            start: config.start_timestamp.to_string(),
            end: config.end_timestamp.to_string(),
        });
    }
    if config.timestep_seconds <= 0 {
        return Err(TimeLoopError::InvalidTimestep {
            value: config.timestep_seconds,
        });
    }
    if !config.dry_reference_height_m.is_finite() || config.dry_reference_height_m <= 0.0 {
        return Err(TimeLoopError::InvalidDryReferenceHeight {
            value: config.dry_reference_height_m,
        });
    }
    if config.receptors.is_empty() {
        return Err(TimeLoopError::MissingReceptors);
    }
    for receptor in &config.receptors {
        validate_receptor(receptor)?;
    }
    for source in &config.source_regions {
        validate_source_region(source)?;
    }
    let _ = timestep_seconds_f32(config.timestep_seconds)?;
    Ok((start_seconds, end_seconds))
}

fn validate_receptor(receptor: &BackwardReceptorConfig) -> Result<(), TimeLoopError> {
    if !(-180.0..=180.0).contains(&receptor.lon) {
        return Err(TimeLoopError::InvalidReceptor {
            name: receptor.name.clone(),
            field: "lon",
            value: receptor.lon,
            reason: "must be in [-180, 180]",
        });
    }
    if !(-90.0..=90.0).contains(&receptor.lat) {
        return Err(TimeLoopError::InvalidReceptor {
            name: receptor.name.clone(),
            field: "lat",
            value: receptor.lat,
            reason: "must be in [-90, 90]",
        });
    }
    if !receptor.z_m.is_finite() || receptor.z_m < 0.0 {
        return Err(TimeLoopError::InvalidReceptor {
            name: receptor.name.clone(),
            field: "z_m",
            value: receptor.z_m,
            reason: "must be finite and >= 0",
        });
    }
    if receptor.particle_count == 0 {
        return Err(TimeLoopError::InvalidReceptor {
            name: receptor.name.clone(),
            field: "particle_count",
            value: 0.0,
            reason: "must be > 0",
        });
    }
    if !receptor.mass_kg.is_finite() || receptor.mass_kg <= 0.0 {
        return Err(TimeLoopError::InvalidReceptor {
            name: receptor.name.clone(),
            field: "mass_kg",
            value: receptor.mass_kg,
            reason: "must be finite and > 0",
        });
    }
    Ok(())
}

fn validate_source_region(source: &BackwardSourceRegionConfig) -> Result<(), TimeLoopError> {
    if source.lon_min > source.lon_max {
        return Err(TimeLoopError::InvalidSourceBounds {
            name: source.name.clone(),
            field: "longitude",
            min: source.lon_min,
            max: source.lon_max,
        });
    }
    if source.lat_min > source.lat_max {
        return Err(TimeLoopError::InvalidSourceBounds {
            name: source.name.clone(),
            field: "latitude",
            min: source.lat_min,
            max: source.lat_max,
        });
    }
    if source.z_min_m > source.z_max_m {
        return Err(TimeLoopError::InvalidSourceBounds {
            name: source.name.clone(),
            field: "height",
            min: source.z_min_m,
            max: source.z_max_m,
        });
    }
    Ok(())
}

fn timestep_seconds_f32(value: i64) -> Result<f32, TimeLoopError> {
    let dt = value as f32;
    if !dt.is_finite() || dt <= 0.0 {
        return Err(TimeLoopError::InvalidTimestep { value });
    }
    Ok(dt)
}

fn interpolation_alpha(
    time_t0_seconds: i64,
    time_t1_seconds: i64,
    target_time_seconds: i64,
    bounds_behavior: TimeBoundsBehavior,
) -> Result<f32, TimeLoopError> {
    if time_t1_seconds <= time_t0_seconds {
        return Err(TemporalInterpolationError::InvalidTimeBracket {
            time_t0_seconds,
            time_t1_seconds,
        }
        .into());
    }
    let effective_target = if (time_t0_seconds..=time_t1_seconds).contains(&target_time_seconds) {
        target_time_seconds
    } else {
        match bounds_behavior {
            TimeBoundsBehavior::Strict => {
                return Err(TemporalInterpolationError::TargetOutsideBracket {
                    target_time_seconds,
                    time_t0_seconds,
                    time_t1_seconds,
                }
                .into());
            }
            TimeBoundsBehavior::Clamp => {
                target_time_seconds.clamp(time_t0_seconds, time_t1_seconds)
            }
        }
    };

    let elapsed_seconds = (effective_target - time_t0_seconds) as f64;
    let window_seconds = (time_t1_seconds - time_t0_seconds) as f64;
    Ok((elapsed_seconds / window_seconds) as f32)
}

fn parse_timestamp_seconds(value: &str) -> Result<i64, TimeLoopError> {
    if value.len() != 14 || !value.chars().all(|c| c.is_ascii_digit()) {
        return Err(TimeLoopError::InvalidTimestamp {
            value: value.to_string(),
        });
    }

    let year = value[0..4]
        .parse::<i32>()
        .map_err(|_| TimeLoopError::InvalidTimestamp {
            value: value.to_string(),
        })?;
    let month = value[4..6]
        .parse::<u32>()
        .map_err(|_| TimeLoopError::InvalidTimestamp {
            value: value.to_string(),
        })?;
    let day = value[6..8]
        .parse::<u32>()
        .map_err(|_| TimeLoopError::InvalidTimestamp {
            value: value.to_string(),
        })?;
    let hour = value[8..10]
        .parse::<u32>()
        .map_err(|_| TimeLoopError::InvalidTimestamp {
            value: value.to_string(),
        })?;
    let minute = value[10..12]
        .parse::<u32>()
        .map_err(|_| TimeLoopError::InvalidTimestamp {
            value: value.to_string(),
        })?;
    let second = value[12..14]
        .parse::<u32>()
        .map_err(|_| TimeLoopError::InvalidTimestamp {
            value: value.to_string(),
        })?;

    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return Err(TimeLoopError::InvalidTimestamp {
            value: value.to_string(),
        });
    }

    let days = days_from_civil(year, month, day);
    Ok(days * 86_400 + i64::from(hour) * 3_600 + i64::from(minute) * 60 + i64::from(second))
}

fn format_timestamp_seconds(seconds: i64) -> Result<String, TimeLoopError> {
    let days = seconds.div_euclid(86_400);
    let sod = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    if !(0..=9999).contains(&year) {
        return Err(TimeLoopError::TimestampOutOfRange { seconds });
    }

    let hour = sod / 3_600;
    let minute = (sod % 3_600) / 60;
    let second = sod % 60;
    Ok(format!(
        "{year:04}{month:02}{day:02}{hour:02}{minute:02}{second:02}"
    ))
}

/// Gregorian civil date to days since Unix epoch (1970-01-01).
fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let mut y = i64::from(year);
    let m = i64::from(month);
    let d = i64::from(day);

    if m <= 2 {
        y -= 1;
    }
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = m + if m > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Inverse of [`days_from_civil`], returns `(year, month, day)`.
fn civil_from_days(days_since_epoch: i64) -> (i32, u32, u32) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    if m <= 2 {
        y += 1;
    }
    (y as i32, m as u32, d as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_parse_and_format_roundtrip() {
        let ts = "20240101123456";
        let seconds = parse_timestamp_seconds(ts).expect("timestamp parses");
        let roundtrip = format_timestamp_seconds(seconds).expect("timestamp formats");
        assert_eq!(roundtrip, ts);
    }

    #[test]
    fn interpolation_alpha_matches_expected_midpoint() {
        let alpha = interpolation_alpha(100, 200, 150, TimeBoundsBehavior::Strict)
            .expect("midpoint alpha computes");
        assert!((alpha - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn backward_config_rejects_inverted_range() {
        let config = BackwardTimeLoopConfig {
            start_timestamp: "20240101000000".to_string(),
            end_timestamp: "20240101000001".to_string(),
            receptors: vec![BackwardReceptorConfig {
                name: "r1".to_string(),
                lon: 0.0,
                lat: 0.0,
                z_m: 10.0,
                particle_count: 1,
                mass_kg: 1.0,
            }],
            ..BackwardTimeLoopConfig::default()
        };
        let err = validate_backward_config(&config).expect_err("range should be rejected");
        assert!(matches!(
            err,
            TimeLoopError::InvalidBackwardTimeRange { .. }
        ));
    }

    #[test]
    fn source_collection_counts_active_hits() {
        use crate::particles::{Particle, ParticleInit, ParticleStore, MAX_SPECIES};

        let mut mass = [0.0_f32; MAX_SPECIES];
        mass[0] = 1.5;
        let mut store = ParticleStore::with_capacity(2);
        let slot = store
            .add(Particle::new(&ParticleInit {
                cell_x: 9,
                cell_y: 4,
                pos_x: 0.0,
                pos_y: 0.0,
                pos_z: 2.0,
                mass,
                release_point: 0,
                class: 0,
                time: 0,
            }))
            .expect("store has free slot");
        assert_eq!(slot, 0);

        let grid = GridDomain {
            xlon0: 0.0,
            ylat0: 0.0,
            dx: 1.0,
            dy: 1.0,
            nx: 64,
            ny: 64,
        };
        let source = BackwardSourceRegionConfig {
            name: "src".to_string(),
            lon_min: 8.5,
            lon_max: 9.5,
            lat_min: 3.5,
            lat_max: 4.5,
            z_min_m: 0.0,
            z_max_m: 10.0,
        };
        let collections = collect_source_collections(&store, &grid, &[source]);
        let result = collections.get("src").expect("source collection exists");
        assert_eq!(result.hit_count, 1);
        assert!((result.total_mass_kg - 1.5).abs() < 1.0e-6);
    }
}
