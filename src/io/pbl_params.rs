//! CPU PBL parameter computation from meteorological surface fields.
//!
//! Port target and rationale:
//! - `calcpar.f90` (surface-derived PBL scalars used by FLEXPART turbulence)
//! - `obukhov.f90` (Monin-Obukhov length diagnostics)
//! - `richardson.f90` and `pbl_profile.f90` (stability helpers / profile cues)
//!
//! This is an MVP implementation for IO-05 with explicit simplifications:
//! - 2-m air temperature is used as a proxy for near-surface potential
//!   temperature in Richardson diagnostics.
//! - If a valid `hmix` is provided by met input, it is used (and clipped to
//!   model bounds). If missing/invalid, a deterministic bulk-Richardson-based
//!   fallback estimate is used.
//! - If provided friction velocity (`u*`) or inverse Obukhov (`1/L`) are valid,
//!   they are used. Otherwise they are derived from stress/10 m wind and
//!   sensible heat flux.

use ndarray::Array2;
use thiserror::Error;

use crate::constants::{CPA, GA, HMIX_MAX, HMIX_MIN, R_AIR, VON_KARMAN};
use crate::pbl::{PblState, StabilityClass};
use crate::wind::SurfaceFields;

const DEFAULT_ROUGHNESS_LENGTH_M: f32 = 0.1;
const DEFAULT_WIND_REFERENCE_HEIGHT_M: f32 = 10.0;
const DEFAULT_HEAT_FLUX_NEUTRAL_THRESHOLD_W_M2: f32 = 1.0;
const DEFAULT_BULK_RI_CRITICAL: f32 = 0.25;
const DEFAULT_MIN_SHEAR_SQUARED_M2_S2: f32 = 0.25;
const DEFAULT_FALLBACK_HMIX_M: f32 = 800.0;
const USTAR_MIN_M_S: f32 = 1.0e-4;
const NEUTRAL_OLI_THRESHOLD_PER_M: f32 = 1.0e-5;

/// Tunable options for PBL parameter computation.
#[derive(Clone, Copy, Debug)]
pub struct PblComputationOptions {
    /// Roughness length `z0` used when deriving `u*` from 10 m wind [m].
    pub roughness_length_m: f32,
    /// Wind reference height used with the log-law [m].
    pub wind_reference_height_m: f32,
    /// |H| threshold below which conditions are treated as neutral [W/m²].
    pub heat_flux_neutral_threshold_w_m2: f32,
    /// Critical bulk Richardson number for stable/neutral transition [-].
    pub bulk_richardson_critical: f32,
    /// Floor for shear term in Richardson denominator [(m/s)²].
    pub min_shear_squared_m2_s2: f32,
    /// Fallback mixing height if no valid `hmix` and no profile cue [m].
    pub fallback_mixing_height_m: f32,
    /// Minimum allowed mixing height [m].
    pub hmix_min_m: f32,
    /// Maximum allowed mixing height [m].
    pub hmix_max_m: f32,
}

impl Default for PblComputationOptions {
    fn default() -> Self {
        Self {
            roughness_length_m: DEFAULT_ROUGHNESS_LENGTH_M,
            wind_reference_height_m: DEFAULT_WIND_REFERENCE_HEIGHT_M,
            heat_flux_neutral_threshold_w_m2: DEFAULT_HEAT_FLUX_NEUTRAL_THRESHOLD_W_M2,
            bulk_richardson_critical: DEFAULT_BULK_RI_CRITICAL,
            min_shear_squared_m2_s2: DEFAULT_MIN_SHEAR_SQUARED_M2_S2,
            fallback_mixing_height_m: DEFAULT_FALLBACK_HMIX_M,
            hmix_min_m: HMIX_MIN,
            hmix_max_m: HMIX_MAX,
        }
    }
}

/// Optional profile cue used to derive a bulk Richardson diagnostic.
#[derive(Clone, Copy, Debug)]
pub struct PblBulkProfilePoint {
    /// Height of the profile point above ground [m].
    pub height_m: f32,
    /// Air temperature at the profile point [K].
    pub temperature_k: f32,
    /// U wind at the profile point [m/s].
    pub wind_u_m_s: f32,
    /// V wind at the profile point [m/s].
    pub wind_v_m_s: f32,
}

/// Single-cell meteorological input for PBL diagnostics.
#[derive(Clone, Copy, Debug)]
pub struct PblCellInput {
    /// Surface pressure [Pa].
    pub surface_pressure_pa: f32,
    /// 2 m air temperature [K].
    pub temperature_2m_k: f32,
    /// 10 m U wind component [m/s].
    pub wind_u_10m_m_s: f32,
    /// 10 m V wind component [m/s].
    pub wind_v_10m_m_s: f32,
    /// Surface stress magnitude [N/m²].
    pub surface_stress_n_m2: f32,
    /// Surface sensible heat flux [W/m²] (positive upward).
    pub sensible_heat_flux_w_m2: f32,
    /// Surface solar radiation [W/m²].
    pub solar_radiation_w_m2: f32,
    /// Meteorological `hmix` candidate [m] (<= 0 means unavailable).
    pub provided_mixing_height_m: f32,
    /// Optional met-provided friction velocity `u*` [m/s].
    pub provided_friction_velocity_m_s: Option<f32>,
    /// Optional met-provided inverse Obukhov length `1/L` [1/m].
    pub provided_inverse_obukhov_per_m: Option<f32>,
    /// Optional profile cue for bulk Richardson support.
    pub profile_point: Option<PblBulkProfilePoint>,
}

/// Single-cell computed PBL parameters.
#[derive(Clone, Copy, Debug)]
pub struct PblCellOutput {
    /// Friction velocity `u*` [m/s].
    pub friction_velocity_m_s: f32,
    /// Obukhov length `L` [m].
    pub obukhov_length_m: f32,
    /// Inverse Obukhov length `1/L` [1/m].
    pub inverse_obukhov_length_per_m: f32,
    /// Mixing height [m].
    pub mixing_height_m: f32,
    /// Convective velocity scale `w*` [m/s].
    pub convective_velocity_scale_m_s: f32,
    /// Stability class inferred from Obukhov / Richardson cues.
    pub stability_class: StabilityClass,
    /// Bulk Richardson number [-] when profile input is available.
    pub bulk_richardson_number: Option<f32>,
}

/// Input object for gridded PBL computation.
#[derive(Clone, Copy, Debug)]
pub struct PblMetInputGrids<'a> {
    /// Required 2-D meteorological surface fields.
    pub surface: &'a SurfaceFields,
    /// Optional 2-D profile cue for Richardson diagnostics.
    pub profile: Option<&'a PblProfileInputs>,
}

/// Optional 2-D profile cue arrays used in bulk Richardson diagnostics.
#[derive(Debug, Clone)]
pub struct PblProfileInputs {
    /// Height above ground [m].
    pub height_m: Array2<f32>,
    /// Air temperature [K].
    pub temperature_k: Array2<f32>,
    /// U wind [m/s].
    pub wind_u_m_s: Array2<f32>,
    /// V wind [m/s].
    pub wind_v_m_s: Array2<f32>,
}

/// Grid-level output that is directly compatible with [`PblState`].
#[derive(Debug, Clone)]
pub struct ComputedPblFields {
    /// Main PBL state consumed by turbulence/deposition code paths.
    pub pbl_state: PblState,
    /// Obukhov length `L` [m].
    pub obukhov_length_m: Array2<f32>,
    /// Stability classification per grid cell.
    pub stability_class: Array2<StabilityClass>,
    /// Bulk Richardson number [-], `Some` when profile cues were provided.
    /// Cells without a valid diagnostic contain `NaN`.
    pub bulk_richardson_number: Option<Array2<f32>>,
}

impl ComputedPblFields {
    /// Stability class at grid point `(i, j)`.
    #[must_use]
    pub fn stability_at(&self, i: usize, j: usize) -> StabilityClass {
        self.stability_class[[i, j]]
    }
}

/// Errors for gridded PBL parameter computation.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PblParameterError {
    #[error("shape mismatch for `{field}`: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        field: &'static str,
        expected: (usize, usize),
        actual: (usize, usize),
    },
}

/// Inputs for friction velocity estimation.
#[derive(Clone, Copy, Debug)]
pub struct FrictionVelocityInput {
    /// Surface stress magnitude [N/m²].
    pub surface_stress_n_m2: f32,
    /// Air density [kg/m³].
    pub air_density_kg_m3: f32,
    /// 10 m U wind [m/s].
    pub wind_u_10m_m_s: f32,
    /// 10 m V wind [m/s].
    pub wind_v_10m_m_s: f32,
    /// Roughness length [m].
    pub roughness_length_m: f32,
    /// Wind reference height [m].
    pub wind_reference_height_m: f32,
}

/// Inputs for Obukhov length estimation from surface fluxes.
#[derive(Clone, Copy, Debug)]
pub struct ObukhovInput {
    /// Friction velocity `u*` [m/s].
    pub friction_velocity_m_s: f32,
    /// Near-surface air temperature [K].
    pub near_surface_temperature_k: f32,
    /// Air density [kg/m³].
    pub air_density_kg_m3: f32,
    /// Surface sensible heat flux [W/m²], positive upward.
    pub sensible_heat_flux_w_m2: f32,
    /// Neutrality threshold on |H| [W/m²].
    pub heat_flux_neutral_threshold_w_m2: f32,
}

/// Inputs for gradient Richardson number.
#[derive(Clone, Copy, Debug)]
pub struct GradientRichardsonInput {
    /// Reference potential temperature [K].
    pub reference_potential_temperature_k: f32,
    /// Vertical potential-temperature gradient dθ/dz [K/m].
    pub potential_temperature_gradient_k_m: f32,
    /// Vertical shear du/dz [1/s].
    pub wind_u_gradient_s_inv: f32,
    /// Vertical shear dv/dz [1/s].
    pub wind_v_gradient_s_inv: f32,
    /// Shear floor in denominator [(1/s)²].
    pub min_shear_squared_s_inv2: f32,
}

/// Inputs for bulk Richardson number.
#[derive(Clone, Copy, Debug)]
pub struct BulkRichardsonInput {
    /// Surface potential temperature [K].
    pub potential_temperature_surface_k: f32,
    /// Potential temperature at height `z` [K].
    pub potential_temperature_at_z_k: f32,
    /// Surface U wind [m/s].
    pub wind_u_surface_m_s: f32,
    /// Surface V wind [m/s].
    pub wind_v_surface_m_s: f32,
    /// U wind at height `z` [m/s].
    pub wind_u_at_z_m_s: f32,
    /// V wind at height `z` [m/s].
    pub wind_v_at_z_m_s: f32,
    /// Height separation `z` [m].
    pub height_m: f32,
    /// Shear floor in denominator [(m/s)²].
    pub min_shear_squared_m2_s2: f32,
}

/// Estimate friction velocity `u*` [m/s] from stress or 10 m wind.
///
/// Precedence:
/// 1. If valid stress and density are available: `u* = sqrt(tau / rho)`.
/// 2. Else neutral log-law estimate from 10 m wind.
#[must_use]
pub fn estimate_friction_velocity_m_s(input: FrictionVelocityInput) -> f32 {
    let rho = sanitize_positive(input.air_density_kg_m3, 1.225);
    let stress = sanitize_non_negative(input.surface_stress_n_m2, 0.0);
    if stress > 0.0 {
        return (stress / rho).sqrt().max(USTAR_MIN_M_S);
    }

    let z0 = sanitize_positive(input.roughness_length_m, DEFAULT_ROUGHNESS_LENGTH_M);
    let z_ref = sanitize_positive(
        input.wind_reference_height_m,
        DEFAULT_WIND_REFERENCE_HEIGHT_M,
    )
    .max(1.01 * z0);
    let wind_speed = input.wind_u_10m_m_s.hypot(input.wind_v_10m_m_s).max(0.0);
    if wind_speed <= 0.0 {
        return USTAR_MIN_M_S;
    }

    let ustar = VON_KARMAN * wind_speed / (z_ref / z0).ln().max(1.0e-6);
    sanitize_positive(ustar, USTAR_MIN_M_S).max(USTAR_MIN_M_S)
}

/// Estimate Obukhov length `L` [m] from sensible heat flux and `u*`.
///
/// Formula:
/// `L = -(rho * cp * T * u*^3) / (kappa * g * H)`, with `H` positive upward.
#[must_use]
pub fn obukhov_length_from_surface_flux_m(input: ObukhovInput) -> f32 {
    let heat_flux = sanitize_finite(input.sensible_heat_flux_w_m2, 0.0);
    if heat_flux.abs() < input.heat_flux_neutral_threshold_w_m2.max(0.0) {
        return f32::INFINITY;
    }

    let ustar = sanitize_positive(input.friction_velocity_m_s, USTAR_MIN_M_S).max(USTAR_MIN_M_S);
    let temperature_k = sanitize_positive(input.near_surface_temperature_k, 300.0);
    let air_density = sanitize_positive(input.air_density_kg_m3, 1.225);

    let numerator = -(air_density * CPA * temperature_k * ustar.powi(3));
    let denominator = VON_KARMAN * GA * heat_flux;
    if denominator.abs() < 1.0e-9 {
        return f32::INFINITY;
    }

    let obukhov = numerator / denominator;
    if !obukhov.is_finite() || obukhov.abs() > 1.0e9 {
        f32::INFINITY
    } else {
        obukhov
    }
}

/// Convert Obukhov length `L` [m] to inverse Obukhov `1/L` [1/m].
#[must_use]
pub fn inverse_obukhov_length_per_m(obukhov_length_m: f32) -> f32 {
    if !obukhov_length_m.is_finite() || obukhov_length_m.abs() > 1.0e5 {
        0.0
    } else {
        1.0 / obukhov_length_m
    }
}

/// Compute gradient Richardson number.
#[must_use]
pub fn gradient_richardson_number(input: GradientRichardsonInput) -> f32 {
    let theta_ref = sanitize_positive(input.reference_potential_temperature_k, 300.0);
    let shear_sq = (input.wind_u_gradient_s_inv.powi(2) + input.wind_v_gradient_s_inv.powi(2))
        .max(input.min_shear_squared_s_inv2.max(1.0e-12));
    GA / theta_ref * input.potential_temperature_gradient_k_m / shear_sq
}

/// Compute bulk Richardson number.
#[must_use]
pub fn bulk_richardson_number(input: BulkRichardsonInput) -> f32 {
    let theta_ref = sanitize_positive(input.potential_temperature_surface_k, 300.0);
    let delta_theta = input.potential_temperature_at_z_k - input.potential_temperature_surface_k;
    let delta_u = input.wind_u_at_z_m_s - input.wind_u_surface_m_s;
    let delta_v = input.wind_v_at_z_m_s - input.wind_v_surface_m_s;
    let shear_sq =
        (delta_u.powi(2) + delta_v.powi(2)).max(input.min_shear_squared_m2_s2.max(1.0e-8));
    let dz = sanitize_positive(input.height_m, 10.0);
    GA / theta_ref * delta_theta * dz / shear_sq
}

/// Compute all IO-05 cell-level PBL parameters from meteorological inputs.
#[must_use]
pub fn compute_pbl_cell_parameters(
    input: PblCellInput,
    options: PblComputationOptions,
) -> PblCellOutput {
    let temperature_k = sanitize_positive(input.temperature_2m_k, 300.0);
    let pressure_pa = sanitize_positive(input.surface_pressure_pa, 101_325.0);
    let air_density = pressure_pa / (R_AIR * temperature_k);

    let estimated_ustar = estimate_friction_velocity_m_s(FrictionVelocityInput {
        surface_stress_n_m2: input.surface_stress_n_m2,
        air_density_kg_m3: air_density,
        wind_u_10m_m_s: input.wind_u_10m_m_s,
        wind_v_10m_m_s: input.wind_v_10m_m_s,
        roughness_length_m: options.roughness_length_m,
        wind_reference_height_m: options.wind_reference_height_m,
    });

    let friction_velocity_m_s = input
        .provided_friction_velocity_m_s
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(estimated_ustar)
        .max(USTAR_MIN_M_S);

    let computed_obukhov_m = obukhov_length_from_surface_flux_m(ObukhovInput {
        friction_velocity_m_s,
        near_surface_temperature_k: temperature_k,
        air_density_kg_m3: air_density,
        sensible_heat_flux_w_m2: input.sensible_heat_flux_w_m2,
        heat_flux_neutral_threshold_w_m2: options.heat_flux_neutral_threshold_w_m2,
    });

    let inverse_obukhov_length_per_m = input
        .provided_inverse_obukhov_per_m
        .filter(|value| value.is_finite() && value.abs() >= NEUTRAL_OLI_THRESHOLD_PER_M)
        .unwrap_or_else(|| inverse_obukhov_length_per_m(computed_obukhov_m));

    let obukhov_length_m = if inverse_obukhov_length_per_m.abs() < NEUTRAL_OLI_THRESHOLD_PER_M {
        f32::INFINITY
    } else {
        1.0 / inverse_obukhov_length_per_m
    };

    let bulk_richardson_number = input.profile_point.map(|profile| {
        bulk_richardson_number(BulkRichardsonInput {
            potential_temperature_surface_k: temperature_k,
            potential_temperature_at_z_k: profile.temperature_k,
            wind_u_surface_m_s: input.wind_u_10m_m_s,
            wind_v_surface_m_s: input.wind_v_10m_m_s,
            wind_u_at_z_m_s: profile.wind_u_m_s,
            wind_v_at_z_m_s: profile.wind_v_m_s,
            height_m: profile.height_m,
            min_shear_squared_m2_s2: options.min_shear_squared_m2_s2,
        })
    });

    let mixing_height_m =
        if input.provided_mixing_height_m.is_finite() && input.provided_mixing_height_m > 0.0 {
            clamp_mixing_height_m(input.provided_mixing_height_m, options)
        } else if let Some(bulk_ri) = bulk_richardson_number {
            estimate_mixing_height_from_bulk_richardson_m(bulk_ri, options)
        } else {
            clamp_mixing_height_m(options.fallback_mixing_height_m, options)
        };

    let convective_velocity_scale_m_s = compute_convective_velocity_scale_m_s(
        sanitize_finite(input.sensible_heat_flux_w_m2, 0.0),
        air_density,
        temperature_k,
        mixing_height_m,
    );

    let stability_from_oli = StabilityClass::from_inverse_obukhov(inverse_obukhov_length_per_m);
    let stability_class = if stability_from_oli == StabilityClass::Neutral {
        bulk_richardson_number
            .map(|ri| stability_from_richardson(ri, options.bulk_richardson_critical))
            .unwrap_or(StabilityClass::Neutral)
    } else {
        stability_from_oli
    };

    PblCellOutput {
        friction_velocity_m_s,
        obukhov_length_m,
        inverse_obukhov_length_per_m,
        mixing_height_m,
        convective_velocity_scale_m_s,
        stability_class,
        bulk_richardson_number,
    }
}

/// Compute gridded PBL fields (`PblState`-compatible) from met inputs.
///
/// # Errors
/// Returns [`PblParameterError::ShapeMismatch`] if optional profile arrays do
/// not match the surface grid shape.
pub fn compute_pbl_parameters_from_met(
    inputs: PblMetInputGrids<'_>,
    options: PblComputationOptions,
) -> Result<ComputedPblFields, PblParameterError> {
    let grid_shape = shape2(&inputs.surface.surface_pressure_pa);
    validate_surface_shape_consistency(inputs.surface, grid_shape)?;
    if let Some(profile) = inputs.profile {
        validate_profile_shape(profile, grid_shape)?;
    }

    let (nx, ny) = grid_shape;
    let mut pbl_state = PblState::new(nx, ny);
    let mut obukhov_length_m = Array2::zeros((nx, ny));
    let mut stability_class = Array2::from_elem((nx, ny), StabilityClass::Neutral);
    let mut bulk_richardson_number = inputs
        .profile
        .map(|_| Array2::from_elem((nx, ny), f32::NAN));

    for i in 0..nx {
        for j in 0..ny {
            let profile_point = inputs.profile.map(|profile| PblBulkProfilePoint {
                height_m: profile.height_m[[i, j]],
                temperature_k: profile.temperature_k[[i, j]],
                wind_u_m_s: profile.wind_u_m_s[[i, j]],
                wind_v_m_s: profile.wind_v_m_s[[i, j]],
            });

            let cell = compute_pbl_cell_parameters(
                PblCellInput {
                    surface_pressure_pa: inputs.surface.surface_pressure_pa[[i, j]],
                    temperature_2m_k: inputs.surface.temperature_2m_k[[i, j]],
                    wind_u_10m_m_s: inputs.surface.u10_ms[[i, j]],
                    wind_v_10m_m_s: inputs.surface.v10_ms[[i, j]],
                    surface_stress_n_m2: inputs.surface.surface_stress_n_m2[[i, j]],
                    sensible_heat_flux_w_m2: inputs.surface.sensible_heat_flux_w_m2[[i, j]],
                    solar_radiation_w_m2: inputs.surface.solar_radiation_w_m2[[i, j]],
                    provided_mixing_height_m: inputs.surface.mixing_height_m[[i, j]],
                    provided_friction_velocity_m_s: finite_positive_option(
                        inputs.surface.friction_velocity_ms[[i, j]],
                    ),
                    provided_inverse_obukhov_per_m: finite_option(
                        inputs.surface.inv_obukhov_length_per_m[[i, j]],
                    ),
                    profile_point,
                },
                options,
            );

            pbl_state.ustar[[i, j]] = cell.friction_velocity_m_s;
            pbl_state.wstar[[i, j]] = cell.convective_velocity_scale_m_s;
            pbl_state.hmix[[i, j]] = cell.mixing_height_m;
            pbl_state.oli[[i, j]] = cell.inverse_obukhov_length_per_m;
            pbl_state.sshf[[i, j]] =
                sanitize_finite(inputs.surface.sensible_heat_flux_w_m2[[i, j]], 0.0);
            pbl_state.ssr[[i, j]] =
                sanitize_non_negative(inputs.surface.solar_radiation_w_m2[[i, j]], 0.0);
            pbl_state.surfstr[[i, j]] =
                sanitize_non_negative(inputs.surface.surface_stress_n_m2[[i, j]], 0.0);

            obukhov_length_m[[i, j]] = cell.obukhov_length_m;
            stability_class[[i, j]] = cell.stability_class;
            if let Some(grid) = bulk_richardson_number.as_mut() {
                grid[[i, j]] = cell.bulk_richardson_number.unwrap_or(f32::NAN);
            }
        }
    }

    Ok(ComputedPblFields {
        pbl_state,
        obukhov_length_m,
        stability_class,
        bulk_richardson_number,
    })
}

#[must_use]
fn compute_convective_velocity_scale_m_s(
    sensible_heat_flux_w_m2: f32,
    air_density_kg_m3: f32,
    temperature_k: f32,
    mixing_height_m: f32,
) -> f32 {
    if sensible_heat_flux_w_m2 <= 0.0 {
        return 0.0;
    }
    let density = sanitize_positive(air_density_kg_m3, 1.225);
    let temp = sanitize_positive(temperature_k, 300.0);
    let hmix = sanitize_positive(mixing_height_m, HMIX_MIN);
    let buoyancy_flux_m2_s3 = sensible_heat_flux_w_m2 / (density * CPA) * GA / temp;
    if buoyancy_flux_m2_s3 <= 0.0 || !buoyancy_flux_m2_s3.is_finite() {
        return 0.0;
    }
    (buoyancy_flux_m2_s3 * hmix).powf(1.0 / 3.0).max(0.0)
}

#[must_use]
fn estimate_mixing_height_from_bulk_richardson_m(
    bulk_richardson_number: f32,
    options: PblComputationOptions,
) -> f32 {
    if !bulk_richardson_number.is_finite() {
        return clamp_mixing_height_m(options.fallback_mixing_height_m, options);
    }

    if bulk_richardson_number <= 0.0 {
        return options.hmix_max_m;
    }

    let span = (options.hmix_max_m - options.hmix_min_m).max(0.0);
    let ri_ratio = bulk_richardson_number / options.bulk_richardson_critical.max(1.0e-6);
    let decay = 1.0 / (1.0 + 4.0 * ri_ratio);
    clamp_mixing_height_m(options.hmix_min_m + decay * span, options)
}

#[must_use]
fn stability_from_richardson(ri_bulk: f32, ri_critical: f32) -> StabilityClass {
    if ri_bulk > ri_critical {
        StabilityClass::Stable
    } else if ri_bulk < -0.02 {
        StabilityClass::Unstable
    } else {
        StabilityClass::Neutral
    }
}

fn validate_surface_shape_consistency(
    surface: &SurfaceFields,
    expected: (usize, usize),
) -> Result<(), PblParameterError> {
    validate_shape("u10_ms", shape2(&surface.u10_ms), expected)?;
    validate_shape("v10_ms", shape2(&surface.v10_ms), expected)?;
    validate_shape(
        "temperature_2m_k",
        shape2(&surface.temperature_2m_k),
        expected,
    )?;
    validate_shape(
        "sensible_heat_flux_w_m2",
        shape2(&surface.sensible_heat_flux_w_m2),
        expected,
    )?;
    validate_shape(
        "solar_radiation_w_m2",
        shape2(&surface.solar_radiation_w_m2),
        expected,
    )?;
    validate_shape(
        "surface_stress_n_m2",
        shape2(&surface.surface_stress_n_m2),
        expected,
    )?;
    validate_shape(
        "friction_velocity_ms",
        shape2(&surface.friction_velocity_ms),
        expected,
    )?;
    validate_shape(
        "mixing_height_m",
        shape2(&surface.mixing_height_m),
        expected,
    )?;
    validate_shape(
        "inv_obukhov_length_per_m",
        shape2(&surface.inv_obukhov_length_per_m),
        expected,
    )?;
    Ok(())
}

fn validate_profile_shape(
    profile: &PblProfileInputs,
    expected: (usize, usize),
) -> Result<(), PblParameterError> {
    validate_shape("profile.height_m", shape2(&profile.height_m), expected)?;
    validate_shape(
        "profile.temperature_k",
        shape2(&profile.temperature_k),
        expected,
    )?;
    validate_shape("profile.wind_u_m_s", shape2(&profile.wind_u_m_s), expected)?;
    validate_shape("profile.wind_v_m_s", shape2(&profile.wind_v_m_s), expected)?;
    Ok(())
}

fn validate_shape(
    field: &'static str,
    actual: (usize, usize),
    expected: (usize, usize),
) -> Result<(), PblParameterError> {
    if actual == expected {
        Ok(())
    } else {
        Err(PblParameterError::ShapeMismatch {
            field,
            expected,
            actual,
        })
    }
}

#[must_use]
fn shape2(array: &Array2<f32>) -> (usize, usize) {
    let shape = array.shape();
    (shape[0], shape[1])
}

#[must_use]
fn clamp_mixing_height_m(value: f32, options: PblComputationOptions) -> f32 {
    value
        .clamp(options.hmix_min_m, options.hmix_max_m)
        .max(options.hmix_min_m)
}

#[must_use]
fn sanitize_positive(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

#[must_use]
fn sanitize_non_negative(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        fallback
    }
}

#[must_use]
fn sanitize_finite(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

#[must_use]
fn finite_option(value: f32) -> Option<f32> {
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

#[must_use]
fn finite_positive_option(value: f32) -> Option<f32> {
    if value.is_finite() && value > 0.0 {
        Some(value)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::constants::VON_KARMAN;

    #[test]
    fn friction_velocity_uses_surface_stress_when_available() {
        let ustar = estimate_friction_velocity_m_s(FrictionVelocityInput {
            surface_stress_n_m2: 0.48,
            air_density_kg_m3: 1.2,
            wind_u_10m_m_s: 0.0,
            wind_v_10m_m_s: 0.0,
            roughness_length_m: 0.1,
            wind_reference_height_m: 10.0,
        });

        assert_abs_diff_eq!(ustar, (0.4_f32).sqrt(), epsilon = 1.0e-6);
    }

    #[test]
    fn friction_velocity_falls_back_to_log_law_wind_estimate() {
        let ustar = estimate_friction_velocity_m_s(FrictionVelocityInput {
            surface_stress_n_m2: 0.0,
            air_density_kg_m3: 1.2,
            wind_u_10m_m_s: 5.0,
            wind_v_10m_m_s: 0.0,
            roughness_length_m: 0.1,
            wind_reference_height_m: 10.0,
        });

        let expected = VON_KARMAN * 5.0 / (10.0_f32 / 0.1).ln();
        assert_abs_diff_eq!(ustar, expected, epsilon = 1.0e-6);
    }

    #[test]
    fn obukhov_length_sign_matches_flux_regime() {
        let unstable = obukhov_length_from_surface_flux_m(ObukhovInput {
            friction_velocity_m_s: 0.4,
            near_surface_temperature_k: 300.0,
            air_density_kg_m3: 1.2,
            sensible_heat_flux_w_m2: 150.0,
            heat_flux_neutral_threshold_w_m2: 1.0,
        });
        let stable = obukhov_length_from_surface_flux_m(ObukhovInput {
            friction_velocity_m_s: 0.4,
            near_surface_temperature_k: 300.0,
            air_density_kg_m3: 1.2,
            sensible_heat_flux_w_m2: -150.0,
            heat_flux_neutral_threshold_w_m2: 1.0,
        });
        let neutral = obukhov_length_from_surface_flux_m(ObukhovInput {
            friction_velocity_m_s: 0.4,
            near_surface_temperature_k: 300.0,
            air_density_kg_m3: 1.2,
            sensible_heat_flux_w_m2: 0.0,
            heat_flux_neutral_threshold_w_m2: 1.0,
        });

        assert!(unstable < 0.0);
        assert!(stable > 0.0);
        assert!(neutral.is_infinite());
        assert_eq!(inverse_obukhov_length_per_m(neutral), 0.0);
    }

    #[test]
    fn richardson_helpers_capture_stable_and_unstable_behavior() {
        let bulk_stable = bulk_richardson_number(BulkRichardsonInput {
            potential_temperature_surface_k: 300.0,
            potential_temperature_at_z_k: 303.0,
            wind_u_surface_m_s: 2.0,
            wind_v_surface_m_s: 1.0,
            wind_u_at_z_m_s: 2.5,
            wind_v_at_z_m_s: 1.4,
            height_m: 100.0,
            min_shear_squared_m2_s2: 0.01,
        });
        let bulk_unstable = bulk_richardson_number(BulkRichardsonInput {
            potential_temperature_surface_k: 300.0,
            potential_temperature_at_z_k: 298.0,
            wind_u_surface_m_s: 2.0,
            wind_v_surface_m_s: 1.0,
            wind_u_at_z_m_s: 2.5,
            wind_v_at_z_m_s: 1.4,
            height_m: 100.0,
            min_shear_squared_m2_s2: 0.01,
        });

        let gradient_stable = gradient_richardson_number(GradientRichardsonInput {
            reference_potential_temperature_k: 300.0,
            potential_temperature_gradient_k_m: 0.02,
            wind_u_gradient_s_inv: 0.05,
            wind_v_gradient_s_inv: 0.03,
            min_shear_squared_s_inv2: 1.0e-4,
        });

        assert!(bulk_stable > 0.25);
        assert!(bulk_unstable < 0.0);
        assert!(gradient_stable > 0.0);
    }

    #[test]
    fn cell_level_stability_and_hmix_transition_with_richardson_fallback() {
        let options = PblComputationOptions::default();

        let stable = compute_pbl_cell_parameters(
            PblCellInput {
                surface_pressure_pa: 101_000.0,
                temperature_2m_k: 300.0,
                wind_u_10m_m_s: 2.0,
                wind_v_10m_m_s: 1.0,
                surface_stress_n_m2: 0.0,
                sensible_heat_flux_w_m2: 0.0,
                solar_radiation_w_m2: 100.0,
                provided_mixing_height_m: 0.0,
                provided_friction_velocity_m_s: None,
                provided_inverse_obukhov_per_m: None,
                profile_point: Some(PblBulkProfilePoint {
                    height_m: 150.0,
                    temperature_k: 304.0,
                    wind_u_m_s: 2.1,
                    wind_v_m_s: 1.1,
                }),
            },
            options,
        );

        let unstable = compute_pbl_cell_parameters(
            PblCellInput {
                surface_pressure_pa: 101_000.0,
                temperature_2m_k: 300.0,
                wind_u_10m_m_s: 2.0,
                wind_v_10m_m_s: 1.0,
                surface_stress_n_m2: 0.0,
                sensible_heat_flux_w_m2: 0.0,
                solar_radiation_w_m2: 100.0,
                provided_mixing_height_m: 0.0,
                provided_friction_velocity_m_s: None,
                provided_inverse_obukhov_per_m: None,
                profile_point: Some(PblBulkProfilePoint {
                    height_m: 150.0,
                    temperature_k: 298.0,
                    wind_u_m_s: 2.1,
                    wind_v_m_s: 1.1,
                }),
            },
            options,
        );

        assert_eq!(stable.stability_class, StabilityClass::Stable);
        assert_eq!(unstable.stability_class, StabilityClass::Unstable);
        assert!(stable.mixing_height_m < unstable.mixing_height_m);
        assert_eq!(stable.inverse_obukhov_length_per_m, 0.0);
        assert_eq!(unstable.inverse_obukhov_length_per_m, 0.0);
    }

    #[test]
    fn gridded_computation_populates_pbl_state_and_regime_classes() {
        let mut surface = SurfaceFields::zeros(2, 2);

        // (0,0): unstable from positive sensible heat flux.
        surface.surface_pressure_pa[[0, 0]] = 101_325.0;
        surface.temperature_2m_k[[0, 0]] = 300.0;
        surface.u10_ms[[0, 0]] = 4.0;
        surface.v10_ms[[0, 0]] = 1.0;
        surface.surface_stress_n_m2[[0, 0]] = 0.4;
        surface.sensible_heat_flux_w_m2[[0, 0]] = 120.0;
        surface.solar_radiation_w_m2[[0, 0]] = 350.0;
        surface.mixing_height_m[[0, 0]] = 1500.0;

        // (1,0): stable from negative sensible heat flux, hmix clipped to min.
        surface.surface_pressure_pa[[1, 0]] = 100_900.0;
        surface.temperature_2m_k[[1, 0]] = 295.0;
        surface.u10_ms[[1, 0]] = 2.0;
        surface.v10_ms[[1, 0]] = 1.0;
        surface.surface_stress_n_m2[[1, 0]] = 0.2;
        surface.sensible_heat_flux_w_m2[[1, 0]] = -80.0;
        surface.solar_radiation_w_m2[[1, 0]] = 50.0;
        surface.mixing_height_m[[1, 0]] = 50.0;

        // (0,1): provided hmix clipped to max.
        surface.surface_pressure_pa[[0, 1]] = 101_100.0;
        surface.temperature_2m_k[[0, 1]] = 299.0;
        surface.u10_ms[[0, 1]] = 3.0;
        surface.v10_ms[[0, 1]] = 0.0;
        surface.surface_stress_n_m2[[0, 1]] = 0.0;
        surface.sensible_heat_flux_w_m2[[0, 1]] = 0.0;
        surface.solar_radiation_w_m2[[0, 1]] = 100.0;
        surface.mixing_height_m[[0, 1]] = 6000.0;

        // (1,1): no hmix provided -> fallback from Richardson (unstable profile).
        surface.surface_pressure_pa[[1, 1]] = 101_000.0;
        surface.temperature_2m_k[[1, 1]] = 301.0;
        surface.u10_ms[[1, 1]] = 1.0;
        surface.v10_ms[[1, 1]] = 1.0;
        surface.surface_stress_n_m2[[1, 1]] = 0.0;
        surface.sensible_heat_flux_w_m2[[1, 1]] = 0.0;
        surface.solar_radiation_w_m2[[1, 1]] = 100.0;
        surface.mixing_height_m[[1, 1]] = 0.0;

        let profile = PblProfileInputs {
            height_m: Array2::from_elem((2, 2), 120.0),
            temperature_k: Array2::from_shape_vec((2, 2), vec![302.0, 303.0, 299.0, 298.0])
                .expect("shape is valid"),
            wind_u_m_s: Array2::from_elem((2, 2), 2.0),
            wind_v_m_s: Array2::from_elem((2, 2), 1.0),
        };

        let output = compute_pbl_parameters_from_met(
            PblMetInputGrids {
                surface: &surface,
                profile: Some(&profile),
            },
            PblComputationOptions::default(),
        )
        .expect("grid inputs are shape-consistent");

        assert_eq!(output.pbl_state.shape(), (2, 2));
        assert_eq!(output.stability_at(0, 0), StabilityClass::Unstable);
        assert_eq!(output.stability_at(1, 0), StabilityClass::Stable);
        assert_eq!(output.stability_at(1, 1), StabilityClass::Unstable);
        assert_abs_diff_eq!(output.pbl_state.hmix[[1, 0]], HMIX_MIN, epsilon = 1.0e-6);
        assert_abs_diff_eq!(output.pbl_state.hmix[[0, 1]], HMIX_MAX, epsilon = 1.0e-6);
        assert!(output.pbl_state.hmix[[1, 1]] >= HMIX_MIN);
        assert!(output.pbl_state.hmix[[1, 1]] <= HMIX_MAX);
        assert_eq!(output.pbl_state.sshf[[0, 0]], 120.0);
        assert_eq!(output.pbl_state.ssr[[0, 0]], 350.0);
        assert_eq!(output.pbl_state.surfstr[[0, 0]], 0.4);
        assert!(output.bulk_richardson_number.is_some());
    }

    #[test]
    fn gridded_computation_reports_profile_shape_mismatch() {
        let surface = SurfaceFields::zeros(2, 2);
        let profile = PblProfileInputs {
            height_m: Array2::from_elem((1, 2), 100.0),
            temperature_k: Array2::from_elem((2, 2), 300.0),
            wind_u_m_s: Array2::from_elem((2, 2), 2.0),
            wind_v_m_s: Array2::from_elem((2, 2), 1.0),
        };

        let error = compute_pbl_parameters_from_met(
            PblMetInputGrids {
                surface: &surface,
                profile: Some(&profile),
            },
            PblComputationOptions::default(),
        )
        .expect_err("shape mismatch must be reported");

        assert!(matches!(
            error,
            PblParameterError::ShapeMismatch {
                field: "profile.height_m",
                expected: (2, 2),
                actual: (1, 2),
            }
        ));
    }
}
