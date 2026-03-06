//! ETEX-style validation harness (I-04).
//!
//! This module provides an end-to-end validation scaffold that can:
//! - run a synthetic ETEX-style scenario through the current pipeline,
//! - compute comparison metrics against a reference field set,
//! - emit structured artifacts for future real-dataset integration.

use std::fs;
use std::path::{Path, PathBuf};

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::config::{ConfigError, SimulationConfig};
use crate::coords::GridDomain;
use crate::gpu::{
    ConcentrationGridOutput, ConcentrationGridShape, ConcentrationGriddingParams, GpuError,
    MAX_OUTPUT_LEVELS,
};
use crate::io::TimeBoundsBehavior;
use crate::physics::VelocityToGridScale;
use crate::simulation::{
    ForwardStepForcing, ForwardStepReport, ForwardTimeLoopConfig, ForwardTimeLoopDriver,
    MetTimeBracket, TimeLoopError,
};
use crate::wind::{uniform_wind_field, SurfaceFields, WindFieldGrid};

/// Error type returned by ETEX validation harness operations.
#[derive(Debug, Error)]
pub enum EtexValidationError {
    #[error("failed to read fixture file `{path}`: {source}")]
    ReadFixture {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse fixture JSON `{path}`: {source}")]
    ParseFixture {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("fixture is invalid: {message}")]
    InvalidFixture { message: String },
    #[error("configuration loading failed: {0}")]
    Config(#[from] ConfigError),
    #[error("time loop execution failed: {0}")]
    TimeLoop(#[from] TimeLoopError),
    #[error("gpu adapter unavailable")]
    GpuUnavailable,
    #[error("failed to create output directory `{path}`: {source}")]
    CreateOutputDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to write artifact `{path}`: {source}")]
    WriteArtifact {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to serialize artifact `{path}`: {source}")]
    SerializeArtifact {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("field shape mismatch for `{field}`: expected values length {expected}, got {actual}")]
    FieldShapeMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("comparison requires same shape for `{field}`: candidate={candidate_shape:?}, reference={reference_shape:?}")]
    ComparisonShapeMismatch {
        field: &'static str,
        candidate_shape: Vec<usize>,
        reference_shape: Vec<usize>,
    },
    #[error("missing fixture-only case in fixture JSON")]
    MissingFixtureOnlyCase,
}

/// Field container with an explicit shape and flattened row-major values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationField {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

impl ValidationField {
    fn expected_len(&self) -> usize {
        self.shape
            .iter()
            .copied()
            .fold(1usize, |acc, dim| acc.saturating_mul(dim))
    }

    fn validate(&self, field: &'static str) -> Result<(), EtexValidationError> {
        let expected = self.expected_len();
        if expected != self.values.len() {
            return Err(EtexValidationError::FieldShapeMismatch {
                field,
                expected,
                actual: self.values.len(),
            });
        }
        Ok(())
    }
}

/// Candidate and reference fields used for validation metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationFieldSet {
    pub concentration_mass_kg: ValidationField,
    pub dry_deposition_kg_m2: ValidationField,
    pub wet_deposition_kg_m2: ValidationField,
}

impl ValidationFieldSet {
    fn validate(&self) -> Result<(), EtexValidationError> {
        self.concentration_mass_kg
            .validate("concentration_mass_kg")?;
        self.dry_deposition_kg_m2.validate("dry_deposition_kg_m2")?;
        self.wet_deposition_kg_m2.validate("wet_deposition_kg_m2")?;
        Ok(())
    }
}

/// Comparison metrics for one field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationMetrics {
    pub sample_count: usize,
    pub rmse: f64,
    pub bias: f64,
    pub mae: f64,
    pub correlation: Option<f64>,
    pub correlation_note: Option<String>,
}

/// Metrics collection for all ETEX validation fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationMetricsSet {
    pub concentration_mass_kg: ValidationMetrics,
    pub dry_deposition_kg_m2: ValidationMetrics,
    pub wet_deposition_kg_m2: ValidationMetrics,
}

/// Pipeline trace entry useful for debugging deterministic runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EtexPipelineTraceEntry {
    pub step_index: usize,
    pub timestamp: String,
    pub interpolation_alpha: f32,
    pub released_count: usize,
    pub active_particle_count: usize,
}

/// Run mode supported by the ETEX validation harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EtexRunMode {
    PipelineSynthetic,
    FixtureOnly,
}

/// Summary report emitted after a validation run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EtexValidationReport {
    pub scenario_id: String,
    pub run_mode: EtexRunMode,
    pub dataset_kind: String,
    pub dataset_path: Option<String>,
    pub notes: Vec<String>,
    pub metrics: ValidationMetricsSet,
    pub pipeline_trace: Vec<EtexPipelineTraceEntry>,
}

/// In-memory outcome produced by one harness run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EtexValidationOutcome {
    pub candidate: ValidationFieldSet,
    pub reference: ValidationFieldSet,
    pub report: EtexValidationReport,
}

/// Artifact paths emitted by [`write_outcome_artifacts`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EtexValidationArtifacts {
    pub output_dir: PathBuf,
    pub candidate_path: PathBuf,
    pub reference_path: PathBuf,
    pub report_path: PathBuf,
}

/// Shape/deposition options for concentration and deposition outputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EtexOutputScaffold {
    pub concentration_shape: [usize; 3],
    pub gridding_species_index: usize,
    pub gridding_mass_scale: f32,
    pub dry_fraction_of_column_mass: f32,
    pub wet_fraction_of_column_mass: f32,
}

impl Default for EtexOutputScaffold {
    fn default() -> Self {
        Self {
            concentration_shape: [12, 12, 6],
            gridding_species_index: 0,
            gridding_mass_scale: 1_000_000.0,
            dry_fraction_of_column_mass: 0.03,
            wet_fraction_of_column_mass: 0.06,
        }
    }
}

/// Synthetic met/deposition forcing used when real ETEX input is unavailable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EtexSyntheticForcing {
    pub wind_t0_ms: [f32; 3],
    pub wind_t1_ms: [f32; 3],
    pub surface_pressure_pa: f32,
    pub temperature_2m_k: f32,
    pub dewpoint_2m_k: f32,
    pub friction_velocity_ms: f32,
    pub convective_velocity_scale_ms: f32,
    pub mixing_height_m: f32,
    pub inv_obukhov_length_per_m: f32,
    pub surface_stress_n_m2: f32,
    pub sensible_heat_flux_w_m2: f32,
    pub solar_radiation_w_m2: f32,
    pub dry_deposition_velocity_m_s: f32,
    pub wet_scavenging_coefficient_s_inv: f32,
    pub wet_precipitating_fraction: f32,
    pub rho_grad_over_rho: f32,
}

impl Default for EtexSyntheticForcing {
    fn default() -> Self {
        Self {
            wind_t0_ms: [0.2, 0.1, 0.0],
            wind_t1_ms: [0.8, 0.2, 0.02],
            surface_pressure_pa: 101_325.0,
            temperature_2m_k: 288.0,
            dewpoint_2m_k: 284.0,
            friction_velocity_ms: 0.35,
            convective_velocity_scale_ms: 0.1,
            mixing_height_m: 1_200.0,
            inv_obukhov_length_per_m: 0.0,
            surface_stress_n_m2: 0.2,
            sensible_heat_flux_w_m2: 8.0,
            solar_radiation_w_m2: 180.0,
            dry_deposition_velocity_m_s: 0.004,
            wet_scavenging_coefficient_s_inv: 6.0e-5,
            wet_precipitating_fraction: 0.2,
            rho_grad_over_rho: 0.0,
        }
    }
}

/// How reference fields are provided while real ETEX oracle is pending.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EtexReferenceMode {
    /// Build synthetic reference fields from current candidate outputs.
    DerivedFromCandidate,
    /// Placeholder mode indicating an external ETEX/FLEXPART dataset should be used.
    ExternalDatasetPlaceholder,
}

/// Affine transform used for scaffolded reference fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldTransform {
    pub scale: f32,
    pub bias: f32,
}

impl Default for FieldTransform {
    fn default() -> Self {
        Self {
            scale: 1.0,
            bias: 0.0,
        }
    }
}

/// Reference builder configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EtexReferenceConfig {
    pub mode: EtexReferenceMode,
    pub dataset_path: Option<String>,
    pub concentration_mass_kg: FieldTransform,
    pub dry_deposition_kg_m2: FieldTransform,
    pub wet_deposition_kg_m2: FieldTransform,
    pub notes: Vec<String>,
    pub explicit_reference: Option<ValidationFieldSet>,
}

impl Default for EtexReferenceConfig {
    fn default() -> Self {
        Self {
            mode: EtexReferenceMode::DerivedFromCandidate,
            dataset_path: None,
            concentration_mass_kg: FieldTransform {
                scale: 1.03,
                bias: 2.0e-5,
            },
            dry_deposition_kg_m2: FieldTransform {
                scale: 0.97,
                bias: 1.0e-6,
            },
            wet_deposition_kg_m2: FieldTransform {
                scale: 1.01,
                bias: -5.0e-7,
            },
            notes: vec![
                "Placeholder reference generated from candidate fields.".to_string(),
                "Replace with FLEXPART Fortran ETEX oracle when dataset is available.".to_string(),
            ],
            explicit_reference: None,
        }
    }
}

/// Optional deterministic case to exercise fixture-only validation mode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixtureOnlyCase {
    pub candidate: ValidationFieldSet,
    pub reference: ValidationFieldSet,
    pub notes: Vec<String>,
}

/// Pipeline inputs used for synthetic end-to-end ETEX-style runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EtexPipelineInputConfig {
    pub config_dir: String,
    pub timestep_seconds: Option<i64>,
    pub particle_capacity: Option<usize>,
}

impl Default for EtexPipelineInputConfig {
    fn default() -> Self {
        Self {
            config_dir: "config".to_string(),
            timestep_seconds: Some(1),
            particle_capacity: Some(2048),
        }
    }
}

/// Root fixture schema for ETEX validation harness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EtexValidationFixture {
    pub scenario_id: String,
    pub description: String,
    pub pipeline: EtexPipelineInputConfig,
    pub synthetic_forcing: EtexSyntheticForcing,
    pub output_scaffold: EtexOutputScaffold,
    pub reference: EtexReferenceConfig,
    pub fixture_only_case: Option<FixtureOnlyCase>,
    pub notes: Vec<String>,
}

/// ETEX validation harness carrying fixture context.
pub struct EtexValidationHarness {
    fixture: EtexValidationFixture,
    fixture_root: PathBuf,
}

impl EtexValidationHarness {
    /// Construct a harness from an already-loaded fixture.
    #[must_use]
    pub fn new(fixture: EtexValidationFixture, fixture_root: PathBuf) -> Self {
        Self {
            fixture,
            fixture_root,
        }
    }

    /// Load harness fixture from JSON file.
    pub fn load_from_json(path: &Path) -> Result<Self, EtexValidationError> {
        let content =
            fs::read_to_string(path).map_err(|source| EtexValidationError::ReadFixture {
                path: path.to_path_buf(),
                source,
            })?;
        let fixture =
            serde_json::from_str::<EtexValidationFixture>(&content).map_err(|source| {
                EtexValidationError::ParseFixture {
                    path: path.to_path_buf(),
                    source,
                }
            })?;
        let fixture_root = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Ok(Self::new(fixture, fixture_root))
    }

    /// Access loaded fixture definition.
    #[must_use]
    pub fn fixture(&self) -> &EtexValidationFixture {
        &self.fixture
    }

    /// Execute deterministic fixture-only metrics flow.
    pub fn run_fixture_only(&self) -> Result<EtexValidationOutcome, EtexValidationError> {
        let Some(case) = &self.fixture.fixture_only_case else {
            return Err(EtexValidationError::MissingFixtureOnlyCase);
        };

        case.candidate.validate()?;
        case.reference.validate()?;
        let metrics = compute_metrics_set(&case.candidate, &case.reference)?;
        let mut notes = self.fixture.notes.clone();
        notes.extend(case.notes.clone());
        notes.extend(self.fixture.reference.notes.clone());

        let report = EtexValidationReport {
            scenario_id: self.fixture.scenario_id.clone(),
            run_mode: EtexRunMode::FixtureOnly,
            dataset_kind: "fixture_scaffold".to_string(),
            dataset_path: self.fixture.reference.dataset_path.clone(),
            notes,
            metrics,
            pipeline_trace: Vec::new(),
        };
        Ok(EtexValidationOutcome {
            candidate: case.candidate.clone(),
            reference: case.reference.clone(),
            report,
        })
    }

    /// Execute synthetic ETEX-style scenario through current time-loop pipeline.
    pub fn run_pipeline_synthetic(&self) -> Result<EtexValidationOutcome, EtexValidationError> {
        let candidate_and_trace = self.build_candidate_from_pipeline()?;
        let reference = self.build_reference_fields(&candidate_and_trace.0)?;
        let metrics = compute_metrics_set(&candidate_and_trace.0, &reference)?;

        let mut notes = self.fixture.notes.clone();
        notes.extend(self.fixture.reference.notes.clone());
        notes.push(
            "Synthetic forcing scaffold was used (real ETEX meteorology/reference pending)."
                .to_string(),
        );

        let report = EtexValidationReport {
            scenario_id: self.fixture.scenario_id.clone(),
            run_mode: EtexRunMode::PipelineSynthetic,
            dataset_kind: match self.fixture.reference.mode {
                EtexReferenceMode::DerivedFromCandidate => "fixture_scaffold".to_string(),
                EtexReferenceMode::ExternalDatasetPlaceholder => {
                    "external_dataset_placeholder".to_string()
                }
            },
            dataset_path: self.fixture.reference.dataset_path.clone(),
            notes,
            metrics,
            pipeline_trace: candidate_and_trace.1,
        };

        Ok(EtexValidationOutcome {
            candidate: candidate_and_trace.0,
            reference,
            report,
        })
    }

    fn build_candidate_from_pipeline(
        &self,
    ) -> Result<(ValidationFieldSet, Vec<EtexPipelineTraceEntry>), EtexValidationError> {
        let config_path = self.fixture_root.join(&self.fixture.pipeline.config_dir);
        let simulation_config = SimulationConfig::load(&config_path)?;
        let outgrid = &simulation_config.outgrid;

        let release_grid = GridDomain {
            xlon0: outgrid.lon_min,
            ylat0: outgrid.lat_min,
            dx: outgrid.dx,
            dy: outgrid.dy,
            nx: outgrid.nx as usize,
            ny: outgrid.ny as usize,
        };
        let particle_capacity = self.fixture.pipeline.particle_capacity.unwrap_or_else(|| {
            let released = simulation_config
                .releases
                .iter()
                .fold(0usize, |acc, release| {
                    acc.saturating_add(usize::try_from(release.particle_count).unwrap_or(0))
                });
            released.saturating_add(512)
        });

        let timestep_seconds = self
            .fixture
            .pipeline
            .timestep_seconds
            .or_else(|| {
                simulation_config
                    .command
                    .sync_interval_seconds
                    .and_then(|value| i64::try_from(value).ok())
            })
            .unwrap_or(1);
        if timestep_seconds <= 0 {
            return Err(EtexValidationError::InvalidFixture {
                message: format!("pipeline.timestep_seconds must be > 0, got {timestep_seconds}"),
            });
        }

        let time_loop_config = ForwardTimeLoopConfig {
            start_timestamp: simulation_config.command.start_time.clone(),
            end_timestamp: simulation_config.command.end_time.clone(),
            timestep_seconds,
            time_bounds_behavior: TimeBoundsBehavior::Clamp,
            velocity_to_grid_scale: VelocityToGridScale::IDENTITY,
            ..ForwardTimeLoopConfig::default()
        };

        let mut driver = match pollster::block_on(ForwardTimeLoopDriver::new(
            time_loop_config,
            &simulation_config.releases,
            release_grid,
            particle_capacity,
        )) {
            Ok(driver) => driver,
            Err(TimeLoopError::Gpu(GpuError::NoAdapter)) => {
                return Err(EtexValidationError::GpuUnavailable);
            }
            Err(err) => return Err(EtexValidationError::TimeLoop(err)),
        };

        let [conc_nx, conc_ny, conc_nz] = self.fixture.output_scaffold.concentration_shape;
        let wind_grid = WindFieldGrid::new(
            conc_nx,
            conc_ny,
            conc_nz,
            conc_nz,
            conc_nz,
            outgrid.dx as f32,
            outgrid.dy as f32,
            outgrid.lon_min as f32,
            outgrid.lat_min as f32,
            Array1::from_iter((0..conc_nz).map(|k| k as f32)),
        );
        let forcing = &self.fixture.synthetic_forcing;
        let wind_t0 = uniform_wind_field(
            &wind_grid,
            forcing.wind_t0_ms[0],
            forcing.wind_t0_ms[1],
            forcing.wind_t0_ms[2],
        );
        let wind_t1 = uniform_wind_field(
            &wind_grid,
            forcing.wind_t1_ms[0],
            forcing.wind_t1_ms[1],
            forcing.wind_t1_ms[2],
        );
        let surface_t0 = synthetic_surface_fields(conc_nx, conc_ny, forcing);
        let surface_t1 = synthetic_surface_fields(conc_nx, conc_ny, forcing);

        let start_seconds = parse_timestamp_seconds(&simulation_config.command.start_time)
            .map_err(|message| EtexValidationError::InvalidFixture { message })?;
        let end_seconds = parse_timestamp_seconds(&simulation_config.command.end_time)
            .map_err(|message| EtexValidationError::InvalidFixture { message })?;
        let upper_time = end_seconds.max(start_seconds.saturating_add(1));
        let met = MetTimeBracket {
            wind_t0: &wind_t0,
            wind_t1: &wind_t1,
            surface_t0: &surface_t0,
            surface_t1: &surface_t1,
            time_t0_seconds: start_seconds,
            time_t1_seconds: upper_time,
        };

        let step_forcing = ForwardStepForcing {
            dry_deposition_velocity_m_s: crate::simulation::ParticleForcingField::Uniform(
                forcing.dry_deposition_velocity_m_s,
            ),
            wet_scavenging_coefficient_s_inv: crate::simulation::ParticleForcingField::Uniform(
                forcing.wet_scavenging_coefficient_s_inv,
            ),
            wet_precipitating_fraction: crate::simulation::ParticleForcingField::Uniform(
                forcing.wet_precipitating_fraction,
            ),
            rho_grad_over_rho: forcing.rho_grad_over_rho,
        };

        let reports = pollster::block_on(driver.run_to_end(&met, &step_forcing))?;
        let concentration_output = pollster::block_on(driver.accumulate_concentration_grid(
            ConcentrationGridShape {
                nx: conc_nx,
                ny: conc_ny,
                nz: conc_nz,
            },
            ConcentrationGriddingParams {
                species_index: self.fixture.output_scaffold.gridding_species_index,
                mass_scale: self.fixture.output_scaffold.gridding_mass_scale,
                outheights: [0.0; MAX_OUTPUT_LEVELS],
            },
        ))?;

        let candidate = convert_candidate_fields(
            concentration_output,
            self.fixture.output_scaffold.dry_fraction_of_column_mass,
            self.fixture.output_scaffold.wet_fraction_of_column_mass,
        );
        candidate.validate()?;

        let trace = reports
            .iter()
            .map(trace_from_step_report)
            .collect::<Vec<_>>();
        Ok((candidate, trace))
    }

    fn build_reference_fields(
        &self,
        candidate: &ValidationFieldSet,
    ) -> Result<ValidationFieldSet, EtexValidationError> {
        if let Some(explicit) = &self.fixture.reference.explicit_reference {
            explicit.validate()?;
            return Ok(explicit.clone());
        }

        match self.fixture.reference.mode {
            EtexReferenceMode::DerivedFromCandidate => {
                let build = &self.fixture.reference;
                let reference = ValidationFieldSet {
                    concentration_mass_kg: transformed_field(
                        &candidate.concentration_mass_kg,
                        &build.concentration_mass_kg,
                    ),
                    dry_deposition_kg_m2: transformed_field(
                        &candidate.dry_deposition_kg_m2,
                        &build.dry_deposition_kg_m2,
                    ),
                    wet_deposition_kg_m2: transformed_field(
                        &candidate.wet_deposition_kg_m2,
                        &build.wet_deposition_kg_m2,
                    ),
                };
                reference.validate()?;
                Ok(reference)
            }
            EtexReferenceMode::ExternalDatasetPlaceholder => Err(EtexValidationError::InvalidFixture {
                message: "reference.mode is `external_dataset_placeholder` but no explicit_reference is provided".to_string(),
            }),
        }
    }
}

fn synthetic_surface_fields(nx: usize, ny: usize, forcing: &EtexSyntheticForcing) -> SurfaceFields {
    let mut surface = SurfaceFields::zeros(nx, ny);
    surface
        .surface_pressure_pa
        .fill(forcing.surface_pressure_pa);
    surface.u10_ms.fill(forcing.wind_t0_ms[0]);
    surface.v10_ms.fill(forcing.wind_t0_ms[1]);
    surface.temperature_2m_k.fill(forcing.temperature_2m_k);
    surface.dewpoint_2m_k.fill(forcing.dewpoint_2m_k);
    surface.precip_large_scale_mm_h.fill(0.0);
    surface.precip_convective_mm_h.fill(0.0);
    surface
        .sensible_heat_flux_w_m2
        .fill(forcing.sensible_heat_flux_w_m2);
    surface
        .solar_radiation_w_m2
        .fill(forcing.solar_radiation_w_m2);
    surface
        .surface_stress_n_m2
        .fill(forcing.surface_stress_n_m2);
    surface
        .friction_velocity_ms
        .fill(forcing.friction_velocity_ms);
    surface
        .convective_velocity_scale_ms
        .fill(forcing.convective_velocity_scale_ms);
    surface.mixing_height_m.fill(forcing.mixing_height_m);
    surface.tropopause_height_m.fill(10_000.0);
    surface
        .inv_obukhov_length_per_m
        .fill(forcing.inv_obukhov_length_per_m);
    surface
}

fn trace_from_step_report(report: &ForwardStepReport) -> EtexPipelineTraceEntry {
    EtexPipelineTraceEntry {
        step_index: report.step_index,
        timestamp: report.timestamp.clone(),
        interpolation_alpha: report.interpolation_alpha,
        released_count: report.released_count,
        active_particle_count: report.active_particle_count,
    }
}

fn convert_candidate_fields(
    concentration_output: ConcentrationGridOutput,
    dry_fraction_of_column_mass: f32,
    wet_fraction_of_column_mass: f32,
) -> ValidationFieldSet {
    let shape = concentration_output.shape;
    let mut dry = vec![0.0_f32; shape.nx * shape.ny];
    let mut wet = vec![0.0_f32; shape.nx * shape.ny];

    for ix in 0..shape.nx {
        for iy in 0..shape.ny {
            let mut column_sum = 0.0_f32;
            for iz in 0..shape.nz {
                let flat = ((ix * shape.ny) + iy) * shape.nz + iz;
                column_sum += concentration_output.concentration_mass_kg[flat];
            }
            let flat2d = ix * shape.ny + iy;
            dry[flat2d] = column_sum * dry_fraction_of_column_mass;
            wet[flat2d] = column_sum * wet_fraction_of_column_mass;
        }
    }

    ValidationFieldSet {
        concentration_mass_kg: ValidationField {
            shape: vec![shape.nx, shape.ny, shape.nz],
            values: concentration_output.concentration_mass_kg,
        },
        dry_deposition_kg_m2: ValidationField {
            shape: vec![shape.nx, shape.ny],
            values: dry,
        },
        wet_deposition_kg_m2: ValidationField {
            shape: vec![shape.nx, shape.ny],
            values: wet,
        },
    }
}

fn transformed_field(field: &ValidationField, transform: &FieldTransform) -> ValidationField {
    ValidationField {
        shape: field.shape.clone(),
        values: field
            .values
            .iter()
            .map(|value| *value * transform.scale + transform.bias)
            .collect(),
    }
}

fn compute_metrics_set(
    candidate: &ValidationFieldSet,
    reference: &ValidationFieldSet,
) -> Result<ValidationMetricsSet, EtexValidationError> {
    Ok(ValidationMetricsSet {
        concentration_mass_kg: compute_metrics_for_field(
            "concentration_mass_kg",
            &candidate.concentration_mass_kg,
            &reference.concentration_mass_kg,
        )?,
        dry_deposition_kg_m2: compute_metrics_for_field(
            "dry_deposition_kg_m2",
            &candidate.dry_deposition_kg_m2,
            &reference.dry_deposition_kg_m2,
        )?,
        wet_deposition_kg_m2: compute_metrics_for_field(
            "wet_deposition_kg_m2",
            &candidate.wet_deposition_kg_m2,
            &reference.wet_deposition_kg_m2,
        )?,
    })
}

fn compute_metrics_for_field(
    field: &'static str,
    candidate: &ValidationField,
    reference: &ValidationField,
) -> Result<ValidationMetrics, EtexValidationError> {
    candidate.validate(field)?;
    reference.validate(field)?;
    if candidate.shape != reference.shape {
        return Err(EtexValidationError::ComparisonShapeMismatch {
            field,
            candidate_shape: candidate.shape.clone(),
            reference_shape: reference.shape.clone(),
        });
    }

    let count = candidate.values.len();
    if count == 0 {
        return Ok(ValidationMetrics {
            sample_count: 0,
            rmse: 0.0,
            bias: 0.0,
            mae: 0.0,
            correlation: None,
            correlation_note: Some("empty field".to_string()),
        });
    }

    let mut sum_diff = 0.0_f64;
    let mut sum_abs_diff = 0.0_f64;
    let mut sum_sq_diff = 0.0_f64;
    let mut sum_candidate = 0.0_f64;
    let mut sum_reference = 0.0_f64;
    for (c, r) in candidate.values.iter().zip(&reference.values) {
        let c64 = f64::from(*c);
        let r64 = f64::from(*r);
        let diff = c64 - r64;
        sum_diff += diff;
        sum_abs_diff += diff.abs();
        sum_sq_diff += diff * diff;
        sum_candidate += c64;
        sum_reference += r64;
    }
    let count_f = count as f64;
    let bias = sum_diff / count_f;
    let mae = sum_abs_diff / count_f;
    let rmse = (sum_sq_diff / count_f).sqrt();

    let mean_candidate = sum_candidate / count_f;
    let mean_reference = sum_reference / count_f;
    let mut cov = 0.0_f64;
    let mut var_candidate = 0.0_f64;
    let mut var_reference = 0.0_f64;
    for (c, r) in candidate.values.iter().zip(&reference.values) {
        let dc = f64::from(*c) - mean_candidate;
        let dr = f64::from(*r) - mean_reference;
        cov += dc * dr;
        var_candidate += dc * dc;
        var_reference += dr * dr;
    }

    let correlation = if var_candidate > 0.0 && var_reference > 0.0 {
        Some(cov / (var_candidate.sqrt() * var_reference.sqrt()))
    } else {
        None
    };
    let correlation_note = if correlation.is_none() {
        Some("zero variance in candidate or reference".to_string())
    } else {
        None
    };

    Ok(ValidationMetrics {
        sample_count: count,
        rmse,
        bias,
        mae,
        correlation,
        correlation_note,
    })
}

/// Write candidate/reference/report JSON artifacts.
pub fn write_outcome_artifacts(
    output_dir: &Path,
    outcome: &EtexValidationOutcome,
) -> Result<EtexValidationArtifacts, EtexValidationError> {
    fs::create_dir_all(output_dir).map_err(|source| EtexValidationError::CreateOutputDir {
        path: output_dir.to_path_buf(),
        source,
    })?;

    let candidate_path = output_dir.join("candidate_fields.json");
    let reference_path = output_dir.join("reference_fields.json");
    let report_path = output_dir.join("validation_report.json");

    write_json_file(&candidate_path, &outcome.candidate)?;
    write_json_file(&reference_path, &outcome.reference)?;
    write_json_file(&report_path, &outcome.report)?;

    Ok(EtexValidationArtifacts {
        output_dir: output_dir.to_path_buf(),
        candidate_path,
        reference_path,
        report_path,
    })
}

fn write_json_file<T: Serialize>(path: &Path, value: &T) -> Result<(), EtexValidationError> {
    let json = serde_json::to_string_pretty(value).map_err(|source| {
        EtexValidationError::SerializeArtifact {
            path: path.to_path_buf(),
            source,
        }
    })?;
    fs::write(path, json).map_err(|source| EtexValidationError::WriteArtifact {
        path: path.to_path_buf(),
        source,
    })
}

fn parse_timestamp_seconds(value: &str) -> Result<i64, String> {
    if value.len() != 14 || !value.chars().all(|c| c.is_ascii_digit()) {
        return Err(format!(
            "invalid timestamp `{value}` (expected YYYYMMDDHHMMSS)"
        ));
    }

    let year = value[0..4]
        .parse::<i32>()
        .map_err(|_| format!("invalid year in timestamp `{value}`"))?;
    let month = value[4..6]
        .parse::<u32>()
        .map_err(|_| format!("invalid month in timestamp `{value}`"))?;
    let day = value[6..8]
        .parse::<u32>()
        .map_err(|_| format!("invalid day in timestamp `{value}`"))?;
    let hour = value[8..10]
        .parse::<u32>()
        .map_err(|_| format!("invalid hour in timestamp `{value}`"))?;
    let minute = value[10..12]
        .parse::<u32>()
        .map_err(|_| format!("invalid minute in timestamp `{value}`"))?;
    let second = value[12..14]
        .parse::<u32>()
        .map_err(|_| format!("invalid second in timestamp `{value}`"))?;
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return Err(format!(
            "invalid date/time components in timestamp `{value}`"
        ));
    }
    let days = days_from_civil(year, month, day);
    Ok(days * 86_400 + i64::from(hour) * 3_600 + i64::from(minute) * 60 + i64::from(second))
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_match_expected_values_for_small_vectors() {
        let candidate = ValidationField {
            shape: vec![4],
            values: vec![1.0, 2.0, 3.0, 4.0],
        };
        let reference = ValidationField {
            shape: vec![4],
            values: vec![0.0, 2.0, 2.0, 6.0],
        };
        let metrics =
            compute_metrics_for_field("sample", &candidate, &reference).expect("metrics compute");
        assert_eq!(metrics.sample_count, 4);
        assert!((metrics.bias - 0.0).abs() < 1.0e-12);
        assert!((metrics.mae - 1.0).abs() < 1.0e-12);
        assert!((metrics.rmse - (1.5_f64).sqrt()).abs() < 1.0e-12);
        assert!(metrics.correlation.is_some());
    }

    #[test]
    fn parse_timestamp_seconds_rejects_bad_inputs() {
        assert!(parse_timestamp_seconds("20240101").is_err());
        assert!(parse_timestamp_seconds("20241301000000").is_err());
        assert!(parse_timestamp_seconds("20240230000000").is_ok());
    }
}
