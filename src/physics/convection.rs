//! CPU reference convection chain (C-01, C-02).
//!
//! Ported in simplified form from FLEXPART convection routines:
//! - `convect43c.f90` (Emanuel-style deep convection trigger and mass-flux idea)
//! - `calcmatrix.f90` (vertical redistribution matrix construction)
//!
//! ## Simplified assumptions (MVP)
//! - CAPE is approximated from available convective diagnostics
//!   (`w*`, convective precipitation), unless explicitly overridden.
//! - Cloud base and top are derived from boundary-layer height and CAPE proxy.
//! - Redistribution uses a deterministic, column-stochastic mixing matrix.
//! - Particle application uses expected destination height (mean of matrix column)
//!   instead of stochastic particle splitting.

use thiserror::Error;

use crate::particles::Particle;

const CAPE_FROM_WSTAR_FACTOR: f32 = 350.0;
const CAPE_FROM_PRECIP_FACTOR: f32 = 35.0;
const CAPE_ACTIVATION_THRESHOLD_J_KG: f32 = 5.0;
const PRECIP_ACTIVATION_THRESHOLD_MM_H: f32 = 0.05;
const MIN_TIMESCALE_S: f32 = 300.0;
const MAX_TIMESCALE_S: f32 = 3_600.0;
const MASS_FLUX_SCALE: f32 = 0.02;
const CONVECTIVE_DEPTH_OFFSET_M: f32 = 1_500.0;
const CAPE_DEPTH_FACTOR_M_PER_SQRT_JKG: f32 = 20.0;
const COLUMN_SUM_TOLERANCE: f32 = 1.0e-4;

/// Errors produced by simplified convection/matrix routines.
#[derive(Debug, Error)]
pub enum ConvectionError {
    #[error("invalid level interfaces: need at least 2 values, got {len}")]
    TooFewLevels { len: usize },
    #[error(
        "invalid level interfaces: values must be finite and strictly increasing at index {index}"
    )]
    NonMonotonicInterfaces { index: usize },
    #[error(
        "invalid boundary_layer_height_m: {value} (must be finite and non-negative for convection)"
    )]
    InvalidBoundaryLayerHeight { value: f32 },
    #[error("invalid timestep for convection matrix: {dt_seconds}")]
    InvalidTimeStep { dt_seconds: f32 },
    #[error("profile length mismatch: expected {expected}, got {actual}")]
    ProfileLengthMismatch { expected: usize, actual: usize },
    #[error(
        "matrix length mismatch: expected {expected} for level_count={level_count}, got {actual}"
    )]
    MatrixLengthMismatch {
        level_count: usize,
        expected: usize,
        actual: usize,
    },
    #[error("source level out of bounds: {source_level} (level_count={level_count})")]
    SourceLevelOutOfBounds {
        source_level: usize,
        level_count: usize,
    },
    #[error(
        "matrix is not column-stochastic at source_level={source_level}: column_sum={column_sum}"
    )]
    NonStochasticColumn {
        source_level: usize,
        column_sum: f32,
    },
}

/// Inputs used by the simplified Emanuel convection reference.
#[derive(Debug, Clone)]
pub struct SimplifiedEmanuelInputs {
    /// Vertical level interfaces [m], length = `n_levels + 1`.
    pub level_interfaces_m: Vec<f32>,
    /// Convective precipitation [mm/h].
    pub convective_precip_mm_h: f32,
    /// Convective velocity scale `w*` [m/s].
    pub convective_velocity_scale_m_s: f32,
    /// Boundary-layer height [m].
    pub boundary_layer_height_m: f32,
    /// Optional CAPE override [J/kg] for deterministic validation setups.
    pub cape_override_j_kg: Option<f32>,
}

impl SimplifiedEmanuelInputs {
    fn validate(&self) -> Result<(), ConvectionError> {
        if self.level_interfaces_m.len() < 2 {
            return Err(ConvectionError::TooFewLevels {
                len: self.level_interfaces_m.len(),
            });
        }

        for (index, pair) in self.level_interfaces_m.windows(2).enumerate() {
            let lower = pair[0];
            let upper = pair[1];
            if !lower.is_finite() || !upper.is_finite() || upper <= lower {
                return Err(ConvectionError::NonMonotonicInterfaces { index });
            }
        }

        if !self.boundary_layer_height_m.is_finite() || self.boundary_layer_height_m < 0.0 {
            return Err(ConvectionError::InvalidBoundaryLayerHeight {
                value: self.boundary_layer_height_m,
            });
        }

        Ok(())
    }
}

/// Simplified Emanuel-like convective column diagnostics.
#[derive(Debug, Clone)]
pub struct SimplifiedEmanuelColumn {
    /// Vertical level interfaces [m], length = `n_levels + 1`.
    pub level_interfaces_m: Vec<f32>,
    /// Vertical level centers [m], length = `n_levels`.
    pub level_centers_m: Vec<f32>,
    /// Whether deep convection is active for this column.
    pub is_active: bool,
    /// CAPE proxy used by this simplified scheme [J/kg].
    pub cape_j_kg: f32,
    /// Diagnosed cloud-base level index.
    pub cloud_base_level: usize,
    /// Diagnosed cloud-top level index.
    pub cloud_top_level: usize,
    /// Updraft mass flux profile [kg m^-2 s^-1], length = `n_levels`.
    pub mass_flux_profile_kg_m2_s: Vec<f32>,
    /// Normalized detrainment target distribution (sum = 1 for active levels).
    pub detrainment_weights: Vec<f32>,
    /// Effective convective timescale [s].
    pub convective_timescale_s: f32,
}

impl SimplifiedEmanuelColumn {
    /// Number of vertical layers in this column.
    #[must_use]
    pub fn level_count(&self) -> usize {
        self.level_centers_m.len()
    }
}

/// Convective redistribution matrix `M(i,j)` in column-major storage.
///
/// `coefficients_column_major[source_level * n_levels + destination_level]`.
#[derive(Debug, Clone)]
pub struct ConvectiveRedistributionMatrix {
    level_count: usize,
    coefficients_column_major: Vec<f32>,
    /// Mixing fraction used for active levels.
    pub mixing_fraction: f32,
    /// Active cloud-base level represented by this matrix.
    pub cloud_base_level: usize,
    /// Active cloud-top level represented by this matrix.
    pub cloud_top_level: usize,
}

impl ConvectiveRedistributionMatrix {
    /// Build a matrix from precomputed column-major coefficients.
    pub fn from_column_major(
        level_count: usize,
        coefficients_column_major: Vec<f32>,
    ) -> Result<Self, ConvectionError> {
        let expected = level_count
            .checked_mul(level_count)
            .unwrap_or(usize::MAX.saturating_sub(1));
        if coefficients_column_major.len() != expected {
            return Err(ConvectionError::MatrixLengthMismatch {
                level_count,
                expected,
                actual: coefficients_column_major.len(),
            });
        }

        let matrix = Self {
            level_count,
            coefficients_column_major,
            mixing_fraction: 0.0,
            cloud_base_level: 0,
            cloud_top_level: level_count.saturating_sub(1),
        };
        matrix.validate_column_stochasticity()?;
        Ok(matrix)
    }

    /// Number of vertical levels represented by the matrix.
    #[must_use]
    pub fn level_count(&self) -> usize {
        self.level_count
    }

    /// Raw matrix coefficients in column-major order.
    #[must_use]
    pub fn coefficients_column_major(&self) -> &[f32] {
        &self.coefficients_column_major
    }

    /// Matrix entry at `(destination_level, source_level)`.
    #[must_use]
    pub fn coefficient(&self, destination_level: usize, source_level: usize) -> f32 {
        self.coefficients_column_major[source_level * self.level_count + destination_level]
    }

    fn validate_column_stochasticity(&self) -> Result<(), ConvectionError> {
        for source_level in 0..self.level_count {
            let mut column_sum = 0.0_f32;
            for destination_level in 0..self.level_count {
                let value = self.coefficient(destination_level, source_level);
                if value < -COLUMN_SUM_TOLERANCE || !value.is_finite() {
                    return Err(ConvectionError::NonStochasticColumn {
                        source_level,
                        column_sum: value,
                    });
                }
                column_sum += value;
            }
            if (column_sum - 1.0).abs() > COLUMN_SUM_TOLERANCE {
                return Err(ConvectionError::NonStochasticColumn {
                    source_level,
                    column_sum,
                });
            }
        }
        Ok(())
    }
}

/// Build simplified Emanuel convective diagnostics for one vertical column.
pub fn compute_simplified_emanuel_column(
    inputs: SimplifiedEmanuelInputs,
) -> Result<SimplifiedEmanuelColumn, ConvectionError> {
    inputs.validate()?;

    let n_levels = inputs.level_interfaces_m.len() - 1;
    let level_centers_m: Vec<f32> = inputs
        .level_interfaces_m
        .windows(2)
        .map(|pair| 0.5 * (pair[0] + pair[1]))
        .collect();

    let cape_j_kg = inputs.cape_override_j_kg.unwrap_or_else(|| {
        let wstar = inputs.convective_velocity_scale_m_s.max(0.0);
        let precip = inputs.convective_precip_mm_h.max(0.0);
        CAPE_FROM_WSTAR_FACTOR * wstar * wstar + CAPE_FROM_PRECIP_FACTOR * precip
    });

    let active = cape_j_kg >= CAPE_ACTIVATION_THRESHOLD_J_KG
        && inputs.convective_precip_mm_h >= PRECIP_ACTIVATION_THRESHOLD_MM_H;

    let mut mass_flux_profile_kg_m2_s = vec![0.0_f32; n_levels];
    let mut detrainment_weights = vec![0.0_f32; n_levels];

    if !active {
        return Ok(SimplifiedEmanuelColumn {
            level_interfaces_m: inputs.level_interfaces_m,
            level_centers_m,
            is_active: false,
            cape_j_kg: cape_j_kg.max(0.0),
            cloud_base_level: 0,
            cloud_top_level: 0,
            mass_flux_profile_kg_m2_s,
            detrainment_weights,
            convective_timescale_s: MAX_TIMESCALE_S,
        });
    }

    let column_top = *inputs.level_interfaces_m.last().unwrap_or(&0.0);
    let base_height_m = (0.6 * inputs.boundary_layer_height_m)
        .max(inputs.level_interfaces_m[0])
        .min(column_top);
    let cloud_base_level = level_index_from_height(&inputs.level_interfaces_m, base_height_m);

    let convective_depth_m =
        CONVECTIVE_DEPTH_OFFSET_M + CAPE_DEPTH_FACTOR_M_PER_SQRT_JKG * cape_j_kg.sqrt();
    let top_height_m = (base_height_m + convective_depth_m)
        .max(base_height_m)
        .min(column_top);
    let mut cloud_top_level = level_index_from_height(&inputs.level_interfaces_m, top_height_m);
    if cloud_top_level < cloud_base_level {
        cloud_top_level = cloud_base_level;
    }

    let span_levels = cloud_top_level.saturating_sub(cloud_base_level);
    let peak_mass_flux =
        MASS_FLUX_SCALE * cape_j_kg.sqrt() * (1.0 + 0.1 * inputs.convective_precip_mm_h.max(0.0));
    let mut detrainment_sum = 0.0_f32;
    for level in cloud_base_level..=cloud_top_level {
        let normalized = if span_levels == 0 {
            0.5
        } else {
            (level - cloud_base_level) as f32 / span_levels as f32
        };
        let bell = (4.0 * normalized * (1.0 - normalized)).max(0.0);
        let shape = 0.05 + bell;
        let flux = peak_mass_flux * shape;
        mass_flux_profile_kg_m2_s[level] = flux;
        detrainment_weights[level] = flux;
        detrainment_sum += flux;
    }

    if detrainment_sum > 0.0 {
        for weight in &mut detrainment_weights {
            *weight /= detrainment_sum;
        }
    } else {
        let active_count = cloud_top_level - cloud_base_level + 1;
        for level in cloud_base_level..=cloud_top_level {
            detrainment_weights[level] = 1.0 / active_count as f32;
        }
    }

    let intensity =
        (cape_j_kg / 1_200.0 + inputs.convective_precip_mm_h.max(0.0) / 20.0).clamp(0.0, 1.0);
    let convective_timescale_s = MAX_TIMESCALE_S - intensity * (MAX_TIMESCALE_S - MIN_TIMESCALE_S);

    Ok(SimplifiedEmanuelColumn {
        level_interfaces_m: inputs.level_interfaces_m,
        level_centers_m,
        is_active: true,
        cape_j_kg,
        cloud_base_level,
        cloud_top_level,
        mass_flux_profile_kg_m2_s,
        detrainment_weights,
        convective_timescale_s,
    })
}

/// Build a deterministic convective redistribution matrix from column diagnostics.
pub fn build_convective_redistribution_matrix(
    column: &SimplifiedEmanuelColumn,
    dt_seconds: f32,
) -> Result<ConvectiveRedistributionMatrix, ConvectionError> {
    if !dt_seconds.is_finite() || dt_seconds <= 0.0 {
        return Err(ConvectionError::InvalidTimeStep { dt_seconds });
    }

    let n_levels = column.level_count();
    let mut coefficients = vec![0.0_f32; n_levels * n_levels];
    for level in 0..n_levels {
        coefficients[level * n_levels + level] = 1.0;
    }

    if !column.is_active {
        return ConvectiveRedistributionMatrix::from_column_major(n_levels, coefficients);
    }

    let timescale = column
        .convective_timescale_s
        .clamp(MIN_TIMESCALE_S, MAX_TIMESCALE_S);
    let cape_factor = (column.cape_j_kg / (column.cape_j_kg + 100.0)).clamp(0.0, 1.0);
    let mixing_fraction = ((1.0 - (-dt_seconds / timescale).exp()) * cape_factor).clamp(0.0, 0.95);

    for source_level in column.cloud_base_level..=column.cloud_top_level {
        let base_index = source_level * n_levels;
        for destination_level in 0..n_levels {
            let target = column.detrainment_weights[destination_level];
            let mut value = mixing_fraction * target;
            if destination_level == source_level {
                value += 1.0 - mixing_fraction;
            }
            coefficients[base_index + destination_level] = value;
        }
    }

    let mut matrix = ConvectiveRedistributionMatrix::from_column_major(n_levels, coefficients)?;
    matrix.mixing_fraction = mixing_fraction;
    matrix.cloud_base_level = column.cloud_base_level;
    matrix.cloud_top_level = column.cloud_top_level;
    Ok(matrix)
}

/// Apply a convective redistribution matrix to a vertical scalar profile.
///
/// Computes `destination = M * source`.
pub fn apply_redistribution_matrix(
    matrix: &ConvectiveRedistributionMatrix,
    source_profile: &[f32],
) -> Result<Vec<f32>, ConvectionError> {
    if source_profile.len() != matrix.level_count {
        return Err(ConvectionError::ProfileLengthMismatch {
            expected: matrix.level_count,
            actual: source_profile.len(),
        });
    }

    let mut destination = vec![0.0_f32; matrix.level_count];
    for (source_level, source_mass) in source_profile.iter().copied().enumerate() {
        let column_start = source_level * matrix.level_count;
        for (destination_level, value) in destination.iter_mut().enumerate() {
            *value +=
                matrix.coefficients_column_major[column_start + destination_level] * source_mass;
        }
    }
    Ok(destination)
}

/// Compute expected destination height from one source level and matrix column.
pub fn expected_destination_height_m(
    matrix: &ConvectiveRedistributionMatrix,
    level_centers_m: &[f32],
    source_level: usize,
) -> Result<f32, ConvectionError> {
    if level_centers_m.len() != matrix.level_count {
        return Err(ConvectionError::ProfileLengthMismatch {
            expected: matrix.level_count,
            actual: level_centers_m.len(),
        });
    }
    if source_level >= matrix.level_count {
        return Err(ConvectionError::SourceLevelOutOfBounds {
            source_level,
            level_count: matrix.level_count,
        });
    }

    let mut weighted_height = 0.0_f32;
    let mut sum = 0.0_f32;
    for destination_level in 0..matrix.level_count {
        let weight = matrix.coefficient(destination_level, source_level).max(0.0);
        weighted_height += weight * level_centers_m[destination_level];
        sum += weight;
    }
    if sum > 0.0 {
        Ok(weighted_height / sum)
    } else {
        Ok(level_centers_m[source_level])
    }
}

/// Apply deterministic convective mixing to particles on CPU.
///
/// Active particles are moved to expected destination heights derived from matrix
/// columns; masses are conserved exactly.
pub fn apply_convective_mixing_to_particles_cpu(
    particles: &mut [Particle],
    matrix: &ConvectiveRedistributionMatrix,
    level_interfaces_m: &[f32],
    level_centers_m: &[f32],
) -> Result<(), ConvectionError> {
    if level_interfaces_m.len() != matrix.level_count + 1 {
        return Err(ConvectionError::ProfileLengthMismatch {
            expected: matrix.level_count + 1,
            actual: level_interfaces_m.len(),
        });
    }
    if level_centers_m.len() != matrix.level_count {
        return Err(ConvectionError::ProfileLengthMismatch {
            expected: matrix.level_count,
            actual: level_centers_m.len(),
        });
    }

    let z_min = level_interfaces_m[0];
    let z_max = *level_interfaces_m.last().unwrap_or(&z_min);
    for particle in particles.iter_mut().filter(|particle| particle.is_active()) {
        let source_level = level_index_from_height(level_interfaces_m, particle.pos_z);
        let destination_height =
            expected_destination_height_m(matrix, level_centers_m, source_level)?;
        particle.pos_z = destination_height.clamp(z_min, z_max);
        particle.cbt = i32::try_from(source_level).unwrap_or(i32::MAX);
    }
    Ok(())
}

/// Build the full simplified convection chain in one call.
pub fn build_simplified_convection_chain(
    inputs: SimplifiedEmanuelInputs,
    dt_seconds: f32,
) -> Result<(SimplifiedEmanuelColumn, ConvectiveRedistributionMatrix), ConvectionError> {
    let column = compute_simplified_emanuel_column(inputs)?;
    let matrix = build_convective_redistribution_matrix(&column, dt_seconds)?;
    Ok((column, matrix))
}

/// Locate vertical layer index for a given particle height.
#[must_use]
pub fn level_index_from_height(level_interfaces_m: &[f32], height_m: f32) -> usize {
    let level_count = level_interfaces_m.len().saturating_sub(1);
    if level_count == 0 {
        return 0;
    }

    let clamped_height = height_m
        .max(level_interfaces_m[0])
        .min(level_interfaces_m[level_count]);
    for level in 0..level_count {
        let lower = level_interfaces_m[level];
        let upper = level_interfaces_m[level + 1];
        if clamped_height >= lower && clamped_height < upper {
            return level;
        }
    }
    level_count - 1
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::particles::{Particle, ParticleInit, MAX_SPECIES};

    fn sample_inputs() -> SimplifiedEmanuelInputs {
        SimplifiedEmanuelInputs {
            level_interfaces_m: vec![0.0, 400.0, 1_000.0, 2_000.0, 3_500.0, 5_000.0],
            convective_precip_mm_h: 10.0,
            convective_velocity_scale_m_s: 2.5,
            boundary_layer_height_m: 900.0,
            cape_override_j_kg: None,
        }
    }

    #[test]
    fn simplified_emanuel_returns_inactive_column_for_weak_forcing() {
        let inputs = SimplifiedEmanuelInputs {
            convective_precip_mm_h: 0.0,
            convective_velocity_scale_m_s: 0.0,
            ..sample_inputs()
        };
        let column = compute_simplified_emanuel_column(inputs).expect("column should build");
        assert!(!column.is_active);
        assert!(column
            .mass_flux_profile_kg_m2_s
            .iter()
            .all(|value| *value == 0.0));
        assert!(column.detrainment_weights.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn redistribution_matrix_is_column_stochastic() {
        let (column, matrix) =
            build_simplified_convection_chain(sample_inputs(), 600.0).expect("chain should build");
        assert!(column.is_active);
        assert!(matrix.mixing_fraction > 0.0);
        for source_level in 0..matrix.level_count() {
            let mut sum = 0.0_f32;
            for destination_level in 0..matrix.level_count() {
                let value = matrix.coefficient(destination_level, source_level);
                assert!(value >= -1.0e-6);
                sum += value;
            }
            assert_relative_eq!(sum, 1.0, epsilon = 1.0e-5, max_relative = 1.0e-5);
        }
    }

    #[test]
    fn redistribution_application_conserves_total_mass() {
        let (_column, matrix) =
            build_simplified_convection_chain(sample_inputs(), 600.0).expect("chain should build");
        let source_profile = vec![12.0, 5.0, 2.0, 0.8, 0.1];
        let destination =
            apply_redistribution_matrix(&matrix, &source_profile).expect("matrix apply succeeds");
        let source_total: f32 = source_profile.iter().sum();
        let destination_total: f32 = destination.iter().sum();
        assert_relative_eq!(
            destination_total,
            source_total,
            epsilon = 1.0e-5,
            max_relative = 1.0e-5
        );
    }

    #[test]
    fn flexpart_style_reference_behavior_lifts_bottom_heavy_profile() {
        let (_column, matrix) =
            build_simplified_convection_chain(sample_inputs(), 900.0).expect("chain should build");
        let source_profile = vec![20.0, 5.0, 1.0, 0.2, 0.1];
        let destination =
            apply_redistribution_matrix(&matrix, &source_profile).expect("matrix apply succeeds");
        assert!(destination[2] > source_profile[2]);
        assert!(destination[3] > source_profile[3]);
    }

    #[test]
    fn cpu_particle_mixing_moves_particles_to_expected_height() {
        let (column, matrix) =
            build_simplified_convection_chain(sample_inputs(), 600.0).expect("chain should build");

        let mut mass = [0.0_f32; MAX_SPECIES];
        mass[0] = 1.0;
        let mut particles = vec![
            Particle::new(&ParticleInit {
                cell_x: 0,
                cell_y: 0,
                pos_x: 0.2,
                pos_y: 0.3,
                pos_z: 200.0,
                mass,
                release_point: 0,
                class: 0,
                time: 0,
            }),
            Particle::new(&ParticleInit {
                cell_x: 0,
                cell_y: 0,
                pos_x: 0.4,
                pos_y: 0.7,
                pos_z: 1_400.0,
                mass,
                release_point: 0,
                class: 0,
                time: 0,
            }),
        ];
        let source_level_0 =
            level_index_from_height(&column.level_interfaces_m, particles[0].pos_z);
        let expected_0 =
            expected_destination_height_m(&matrix, &column.level_centers_m, source_level_0)
                .expect("expected height should compute");

        apply_convective_mixing_to_particles_cpu(
            &mut particles,
            &matrix,
            &column.level_interfaces_m,
            &column.level_centers_m,
        )
        .expect("particle mixing succeeds");

        assert_relative_eq!(
            particles[0].pos_z,
            expected_0,
            epsilon = 1.0e-6,
            max_relative = 1.0e-6
        );
    }
}
