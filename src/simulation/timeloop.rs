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

use thiserror::Error;

use crate::config::ReleaseConfig;
use crate::coords::GridDomain;
use crate::gpu::{
    accumulate_concentration_grid_gpu, advect_particles_gpu, apply_dry_deposition_step_gpu,
    apply_wet_deposition_step_gpu, compute_hanna_params_gpu, encode_advection_gpu_with_kernel,
    encode_dry_deposition_probability_gpu_with_kernel, encode_hanna_params_gpu_with_kernel,
    encode_update_particles_turbulence_langevin_gpu_with_hanna_buffer_and_kernel,
    encode_wet_deposition_probability_gpu_with_kernel, update_particles_turbulence_langevin_gpu,
    AdvectionDispatchKernel, ConcentrationGridOutput, ConcentrationGridShape,
    ConcentrationGriddingParams, DryDepositionDispatchKernel, DryDepositionIoBuffers,
    DryDepositionStepParams, GpuAdvectionError, GpuBufferError, GpuConcentrationGriddingError,
    GpuContext, GpuDryDepositionError, GpuError, GpuHannaError, GpuLangevinError,
    GpuPblReflectionError, GpuWetDepositionError, HannaDispatchKernel, HannaParamsOutputBuffer,
    LangevinDispatchKernel, ParticleBuffers, PblBuffers,
    WetDepositionDispatchKernel, WetDepositionIoBuffers, WetDepositionStepParams, WindBuffers,
    WindSamplingOptions, WindSamplingPath,
};
use crate::io::{
    compute_pbl_parameters_from_met, interpolate_surface_fields_linear,
    interpolate_wind_field_linear, PblComputationOptions, PblMetInputGrids, PblParameterError,
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
    #[error("GPU PBL reflection dispatch failed: {0}")]
    GpuPblReflection(#[from] GpuPblReflectionError),
    #[error("GPU dry deposition dispatch failed: {0}")]
    GpuDryDeposition(#[from] GpuDryDepositionError),
    #[error("GPU wet deposition dispatch failed: {0}")]
    GpuWetDeposition(#[from] GpuWetDepositionError),
    #[error("GPU concentration gridding failed: {0}")]
    GpuConcentrationGridding(#[from] GpuConcentrationGriddingError),
}

/// Forward-mode integration driver orchestrating per-timestep GPU dispatch.
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
    wind_buffers: Option<WindBuffers>,
    advection_dispatch_kernel: Option<AdvectionDispatchKernel>,
    pbl_buffers: Option<PblBuffers>,
    hanna_params_output: Option<HannaParamsOutputBuffer>,
    hanna_dispatch_kernel: Option<HannaDispatchKernel>,
    langevin_dispatch_kernel: Option<LangevinDispatchKernel>,
    dry_deposition_io: Option<DryDepositionIoBuffers>,
    dry_deposition_dispatch_kernel: Option<DryDepositionDispatchKernel>,
    wet_deposition_io: Option<WetDepositionIoBuffers>,
    wet_deposition_dispatch_kernel: Option<WetDepositionDispatchKernel>,
}

impl ForwardTimeLoopDriver {
    /// Create a new forward timeloop driver with empty particle store.
    pub async fn new(
        config: ForwardTimeLoopConfig,
        releases: &[ReleaseConfig],
        release_grid: GridDomain,
        particle_capacity: usize,
    ) -> Result<Self, TimeLoopError> {
        let (start_time_seconds, end_time_seconds) = validate_config(&config)?;
        let initial_philox_counter = config.initial_philox_counter;
        let release_manager = ReleaseManager::new(releases, release_grid)?;
        let particle_store = ParticleStore::with_capacity(particle_capacity);
        let gpu_context = GpuContext::new().await?;
        let particle_buffers = ParticleBuffers::from_store(&gpu_context, &particle_store);

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
            pbl_buffers: None,
            hanna_params_output: None,
            hanna_dispatch_kernel: None,
            langevin_dispatch_kernel: None,
            dry_deposition_io: None,
            dry_deposition_dispatch_kernel: None,
            wet_deposition_io: None,
            wet_deposition_dispatch_kernel: None,
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

    /// Accumulate current particle state into a concentration grid on GPU.
    pub async fn accumulate_concentration_grid(
        &self,
        shape: ConcentrationGridShape,
        params: ConcentrationGriddingParams,
    ) -> Result<ConcentrationGridOutput, TimeLoopError> {
        accumulate_concentration_grid_gpu(&self.gpu_context, &self.particle_buffers, shape, params)
            .await
            .map_err(Into::into)
    }

    /// Run one orchestrated timestep.
    pub async fn run_timestep(
        &mut self,
        met: &MetTimeBracket<'_>,
        forcing: &ForwardStepForcing,
    ) -> Result<ForwardStepReport, TimeLoopError> {
        if !self.has_remaining_steps() {
            return Err(TimeLoopError::SimulationComplete);
        }
        self.apply_spatial_sort_if_enabled()?;

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

        let interpolated_wind = interpolate_wind_field_linear(
            met.wind_t0,
            met.wind_t1,
            met.time_t0_seconds,
            met.time_t1_seconds,
            self.current_time_seconds,
            self.config.time_bounds_behavior,
        )?;
        let interpolated_surface = interpolate_surface_fields_linear(
            met.surface_t0,
            met.surface_t1,
            met.time_t0_seconds,
            met.time_t1_seconds,
            self.current_time_seconds,
            self.config.time_bounds_behavior,
        )?;

        let computed_pbl = compute_pbl_parameters_from_met(
            PblMetInputGrids {
                surface: &interpolated_surface,
                profile: None,
            },
            self.config.pbl_options,
        )?;

        self.ensure_wind_buffers(&interpolated_wind)?;
        self.ensure_pbl_buffers(&computed_pbl.pbl_state)?;
        let step_dt_seconds = timestep_seconds_f32(self.config.timestep_seconds)?;
        let advection_options = WindSamplingOptions::default();
        let advection_sampling_path = {
            let wind_buffers = self
                .wind_buffers
                .as_ref()
                .expect("wind buffers are initialized by ensure_wind_buffers");
            crate::gpu::resolve_wind_sampling_path(
                &self.gpu_context,
                wind_buffers,
                advection_options,
            )
        };
        self.ensure_advection_dispatch_kernel(advection_sampling_path);

        self.ensure_hanna_params_output_buffer(self.particle_buffers.particle_count())?;
        self.ensure_hanna_dispatch_kernel();
        self.ensure_langevin_dispatch_kernel();

        let slot_count = self.particle_buffers.particle_count();
        let dry_velocity = forcing
            .dry_deposition_velocity_m_s
            .materialize(slot_count, "dry_deposition_velocity_m_s")?;
        let wet_scavenging = forcing
            .wet_scavenging_coefficient_s_inv
            .materialize(slot_count, "wet_scavenging_coefficient_s_inv")?;
        let wet_fraction = forcing
            .wet_precipitating_fraction
            .materialize(slot_count, "wet_precipitating_fraction")?;

        self.ensure_dry_deposition_io_buffers(&dry_velocity)?;
        self.ensure_wet_deposition_io_buffers(&wet_scavenging, &wet_fraction)?;
        self.ensure_dry_deposition_dispatch_kernel();
        self.ensure_wet_deposition_dispatch_kernel();

        let dry_params = DryDepositionStepParams {
            dt_seconds: step_dt_seconds,
            reference_height_m: self.config.dry_reference_height_m,
        };
        let wet_params = WetDepositionStepParams {
            dt_seconds: step_dt_seconds,
        };
        let next_philox_counter = {
            let mut encoder =
                self.gpu_context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("forward_timeloop_step_encoder"),
                    });
            let wind_buffers = self
                .wind_buffers
                .as_ref()
                .expect("wind buffers are initialized by ensure_wind_buffers");
            let advection_kernel = self
                .advection_dispatch_kernel
                .as_ref()
                .expect("advection dispatch kernel is initialized");
            encode_advection_gpu_with_kernel(
                &self.gpu_context,
                &self.particle_buffers,
                wind_buffers,
                step_dt_seconds,
                self.config.velocity_to_grid_scale,
                advection_options,
                advection_kernel,
                &mut encoder,
            )?;

            // PBL reflection is now integrated into the Langevin sub-stepping
            // kernel (n_substeps >= 1) to properly reflect at z=0 and z=hmix
            // between each vertical turbulence sub-step.

            let pbl_buffers = self
                .pbl_buffers
                .as_ref()
                .expect("pbl buffers are initialized by ensure_pbl_buffers");
            let hanna_output = self
                .hanna_params_output
                .as_ref()
                .expect("hanna output buffer is initialized");
            let hanna_kernel = self
                .hanna_dispatch_kernel
                .as_ref()
                .expect("hanna dispatch kernel is initialized");
            encode_hanna_params_gpu_with_kernel(
                &self.gpu_context,
                &self.particle_buffers,
                pbl_buffers,
                hanna_output,
                hanna_kernel,
                &mut encoder,
            )?;

            let langevin_kernel = self
                .langevin_dispatch_kernel
                .as_ref()
                .expect("langevin dispatch kernel is initialized");
            let next_philox_counter =
                encode_update_particles_turbulence_langevin_gpu_with_hanna_buffer_and_kernel(
                    &self.gpu_context,
                    &self.particle_buffers,
                    &hanna_output.buffer,
                    hanna_output.particle_count(),
                    LangevinStep {
                        dt_seconds: step_dt_seconds,
                        rho_grad_over_rho: forcing.rho_grad_over_rho,
                        n_substeps: 4,
                        min_height_m: 0.01,
                    },
                    self.config.philox_key,
                    self.philox_counter,
                    langevin_kernel,
                    &mut encoder,
                )?;

            let dry_io = self
                .dry_deposition_io
                .as_ref()
                .expect("dry deposition IO is initialized");
            let dry_kernel = self
                .dry_deposition_dispatch_kernel
                .as_ref()
                .expect("dry deposition dispatch kernel is initialized");
            encode_dry_deposition_probability_gpu_with_kernel(
                &self.gpu_context,
                &self.particle_buffers,
                dry_io,
                dry_params,
                dry_kernel,
                &mut encoder,
            )?;

            let wet_io = self
                .wet_deposition_io
                .as_ref()
                .expect("wet deposition IO is initialized");
            let wet_kernel = self
                .wet_deposition_dispatch_kernel
                .as_ref()
                .expect("wet deposition dispatch kernel is initialized");
            encode_wet_deposition_probability_gpu_with_kernel(
                &self.gpu_context,
                &self.particle_buffers,
                wet_io,
                wet_params,
                wet_kernel,
                &mut encoder,
            )?;

            self.gpu_context.queue.submit(Some(encoder.finish()));
            next_philox_counter
        };
        self.philox_counter = next_philox_counter;

        let dry_probability = if self.config.collect_deposition_probabilities_each_step {
            let dry_io = self
                .dry_deposition_io
                .as_ref()
                .expect("dry deposition IO is initialized");
            dry_io.download_probabilities(&self.gpu_context).await?
        } else {
            Vec::new()
        };
        let wet_probability = if self.config.collect_deposition_probabilities_each_step {
            let wet_io = self
                .wet_deposition_io
                .as_ref()
                .expect("wet deposition IO is initialized");
            wet_io.download_probabilities(&self.gpu_context).await?
        } else {
            Vec::new()
        };

        if self.config.sync_particle_store_each_step {
            self.sync_store_from_gpu().await?;
        }
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
        };

        self.step_index += 1;
        self.current_time_seconds = self
            .current_time_seconds
            .saturating_add(self.config.timestep_seconds);

        Ok(report)
    }

    /// Run until `end_timestamp` using fixed forcing and one met bracket.
    pub async fn run_to_end(
        &mut self,
        met: &MetTimeBracket<'_>,
        forcing: &ForwardStepForcing,
    ) -> Result<Vec<ForwardStepReport>, TimeLoopError> {
        let mut reports = Vec::new();
        while self.has_remaining_steps() {
            reports.push(self.run_timestep(met, forcing).await?);
        }
        Ok(reports)
    }

    fn ensure_wind_buffers(&mut self, wind: &WindField3D) -> Result<(), TimeLoopError> {
        if let Some(buffers) = &self.wind_buffers {
            buffers.upload_field(&self.gpu_context, wind)?;
            return Ok(());
        }
        self.wind_buffers = Some(WindBuffers::from_field(&self.gpu_context, wind)?);
        Ok(())
    }

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

    fn ensure_pbl_buffers(
        &mut self,
        pbl_state: &crate::pbl::PblState,
    ) -> Result<(), TimeLoopError> {
        if let Some(buffers) = &self.pbl_buffers {
            buffers.upload_state(&self.gpu_context, pbl_state)?;
            return Ok(());
        }
        self.pbl_buffers = Some(PblBuffers::from_state(&self.gpu_context, pbl_state)?);
        Ok(())
    }

    fn ensure_hanna_params_output_buffer(
        &mut self,
        particle_count: usize,
    ) -> Result<(), TimeLoopError> {
        if particle_count == 0 {
            return Ok(());
        }
        if let Some(buffer) = &self.hanna_params_output {
            if buffer.particle_count() == particle_count {
                return Ok(());
            }
        }
        self.hanna_params_output = Some(HannaParamsOutputBuffer::new(
            &self.gpu_context,
            particle_count,
        )?);
        Ok(())
    }

    fn ensure_hanna_dispatch_kernel(&mut self) {
        if self.hanna_dispatch_kernel.is_none() {
            self.hanna_dispatch_kernel = Some(HannaDispatchKernel::new(&self.gpu_context));
        }
    }

    fn ensure_langevin_dispatch_kernel(&mut self) {
        if self.langevin_dispatch_kernel.is_none() {
            self.langevin_dispatch_kernel = Some(LangevinDispatchKernel::new(&self.gpu_context));
        }
    }

    fn ensure_dry_deposition_io_buffers(
        &mut self,
        deposition_velocity_m_s: &[f32],
    ) -> Result<(), TimeLoopError> {
        if let Some(io) = &self.dry_deposition_io {
            if io.particle_count() == deposition_velocity_m_s.len() {
                io.upload_deposition_velocity(&self.gpu_context, deposition_velocity_m_s)?;
                return Ok(());
            }
        }
        self.dry_deposition_io = Some(DryDepositionIoBuffers::from_velocity(
            &self.gpu_context,
            deposition_velocity_m_s,
        )?);
        Ok(())
    }

    fn ensure_dry_deposition_dispatch_kernel(&mut self) {
        if self.dry_deposition_dispatch_kernel.is_none() {
            self.dry_deposition_dispatch_kernel =
                Some(DryDepositionDispatchKernel::new(&self.gpu_context));
        }
    }

    fn ensure_wet_deposition_io_buffers(
        &mut self,
        scavenging_coefficient_s_inv: &[f32],
        precipitating_fraction: &[f32],
    ) -> Result<(), TimeLoopError> {
        if let Some(io) = &self.wet_deposition_io {
            if io.particle_count() == scavenging_coefficient_s_inv.len()
                && io.particle_count() == precipitating_fraction.len()
            {
                io.upload_scavenging_coefficient(&self.gpu_context, scavenging_coefficient_s_inv)?;
                io.upload_precipitating_fraction(&self.gpu_context, precipitating_fraction)?;
                return Ok(());
            }
        }
        self.wet_deposition_io = Some(WetDepositionIoBuffers::from_inputs(
            &self.gpu_context,
            scavenging_coefficient_s_inv,
            precipitating_fraction,
        )?);
        Ok(())
    }

    fn ensure_wet_deposition_dispatch_kernel(&mut self) {
        if self.wet_deposition_dispatch_kernel.is_none() {
            self.wet_deposition_dispatch_kernel =
                Some(WetDepositionDispatchKernel::new(&self.gpu_context));
        }
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
    wind_buffers: Option<WindBuffers>,
    pbl_buffers: Option<PblBuffers>,
}

impl BackwardTimeLoopDriver {
    /// Create a new backward timeloop driver with empty particle store.
    pub async fn new(
        config: BackwardTimeLoopConfig,
        release_grid: GridDomain,
        particle_capacity: usize,
    ) -> Result<Self, TimeLoopError> {
        let (start_time_seconds, end_time_seconds) = validate_backward_config(&config)?;
        let initial_philox_counter = config.initial_philox_counter;
        let release_manager = ReleaseManager::new(
            &build_receptor_release_configs(&config.receptors, &config.start_timestamp),
            release_grid.clone(),
        )?;
        let particle_store = ParticleStore::with_capacity(particle_capacity);
        let gpu_context = GpuContext::new().await?;
        let particle_buffers = ParticleBuffers::from_store(&gpu_context, &particle_store);

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
            wind_buffers: None,
            pbl_buffers: None,
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

        let interpolated_wind = interpolate_wind_field_linear(
            met.wind_t0,
            met.wind_t1,
            met.time_t0_seconds,
            met.time_t1_seconds,
            self.current_time_seconds,
            self.config.time_bounds_behavior,
        )?;
        let interpolated_surface = interpolate_surface_fields_linear(
            met.surface_t0,
            met.surface_t1,
            met.time_t0_seconds,
            met.time_t1_seconds,
            self.current_time_seconds,
            self.config.time_bounds_behavior,
        )?;

        let computed_pbl = compute_pbl_parameters_from_met(
            PblMetInputGrids {
                surface: &interpolated_surface,
                profile: None,
            },
            self.config.pbl_options,
        )?;

        self.ensure_wind_buffers(&interpolated_wind)?;
        self.ensure_pbl_buffers(&computed_pbl.pbl_state)?;

        let dt_seconds = timestep_seconds_f32(self.config.timestep_seconds)?;
        let wind_buffers = self
            .wind_buffers
            .as_ref()
            .expect("wind buffers are initialized by ensure_wind_buffers");
        advect_particles_gpu(
            &self.gpu_context,
            &self.particle_buffers,
            wind_buffers,
            TimeDirection::Backward.advection_dt_seconds(dt_seconds),
            self.config.velocity_to_grid_scale,
        )?;

        // B-01 MVP: keep turbulence/deposition kernels on positive dt magnitude.
        let pbl_buffers = self
            .pbl_buffers
            .as_ref()
            .expect("pbl buffers are initialized by ensure_pbl_buffers");
        let hanna_params =
            compute_hanna_params_gpu(&self.gpu_context, &self.particle_buffers, pbl_buffers)
                .await?;
        self.philox_counter = update_particles_turbulence_langevin_gpu(
            &self.gpu_context,
            &self.particle_buffers,
            &hanna_params,
            LangevinStep {
                dt_seconds,
                rho_grad_over_rho: forcing.rho_grad_over_rho,
                n_substeps: 4,
                min_height_m: 0.01,
            },
            self.config.philox_key,
            self.philox_counter,
        )?;

        let slot_count = self.particle_buffers.particle_count();
        let dry_velocity = forcing
            .dry_deposition_velocity_m_s
            .materialize(slot_count, "dry_deposition_velocity_m_s")?;
        let wet_scavenging = forcing
            .wet_scavenging_coefficient_s_inv
            .materialize(slot_count, "wet_scavenging_coefficient_s_inv")?;
        let wet_fraction = forcing
            .wet_precipitating_fraction
            .materialize(slot_count, "wet_precipitating_fraction")?;

        let _ = apply_dry_deposition_step_gpu(
            &self.gpu_context,
            &self.particle_buffers,
            &dry_velocity,
            DryDepositionStepParams {
                dt_seconds,
                reference_height_m: self.config.dry_reference_height_m,
            },
        )
        .await?;
        let _ = apply_wet_deposition_step_gpu(
            &self.gpu_context,
            &self.particle_buffers,
            &wet_scavenging,
            &wet_fraction,
            WetDepositionStepParams { dt_seconds },
        )
        .await?;

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

    fn ensure_wind_buffers(&mut self, wind: &WindField3D) -> Result<(), TimeLoopError> {
        if let Some(buffers) = &self.wind_buffers {
            buffers.upload_field(&self.gpu_context, wind)?;
            return Ok(());
        }
        self.wind_buffers = Some(WindBuffers::from_field(&self.gpu_context, wind)?);
        Ok(())
    }

    fn ensure_pbl_buffers(
        &mut self,
        pbl_state: &crate::pbl::PblState,
    ) -> Result<(), TimeLoopError> {
        if let Some(buffers) = &self.pbl_buffers {
            buffers.upload_state(&self.gpu_context, pbl_state)?;
            return Ok(());
        }
        self.pbl_buffers = Some(PblBuffers::from_state(&self.gpu_context, pbl_state)?);
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
