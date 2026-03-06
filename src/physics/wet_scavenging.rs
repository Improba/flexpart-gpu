//! CPU reference wet scavenging (below-cloud + in-cloud).
//!
//! Ported from FLEXPART:
//! - `wetdepo.f90` (mass-loss update over one deposition step)
//! - `get_wetscav.f90` (scavenging coefficients and pathway branching)
//!
//! MVP assumptions in this module:
//! - negative/invalid species coefficients are treated as disabled (`0`);
//! - the cloud-water fallback uses FLEXPART's precipitation-only parameterization
//!   (`1e6 * 2e-7 * prec^0.36`) when no explicit cloud-water input is available;
//! - per-step wet loss is represented as an effective probability
//!   `p = precipitating_fraction * (1 - exp(-lambda * |dt|))`.

use crate::constants::R_AIR;

const MIN_TOTAL_PRECIP_MM_H: f32 = 0.01;
const MIN_PRECIPITATING_FRACTION: f32 = 0.05;
const IN_CLOUD_RATIO_DEFAULT: f32 = 6.2;
const PRECIP_MM_H_TO_M_S: f32 = 1.0 / 3.6e6;
const BELOW_CLOUD_RAIN_SNOW_SPLIT_K: f32 = 273.0;
const ICE_ONLY_TEMPERATURE_K: f32 = 253.0;
const AEROSOL_DIAMETER_CAP_UM: f32 = 10.0;
const MIN_CLOUD_WATER_CONTENT: f32 = 1.0e-12;
const EPSILON: f32 = 1.0e-12;

const LARGE_SCALE_AREA_FRACTIONS: [f32; 5] = [0.5, 0.65, 0.8, 0.9, 0.95];
const CONVECTIVE_AREA_FRACTIONS: [f32; 5] = [0.4, 0.55, 0.7, 0.8, 0.9];

// Laakso et al. (2003) polynomial constants for rain below-cloud scavenging.
const BELOW_CLOUD_RAIN_POLY: [f32; 6] = [
    274.35758,
    332_839.59273,
    226_656.57259,
    58_005.9134,
    6_588.38582,
    0.244_984,
];

// Kyrö et al. (2009) polynomial constants for snow below-cloud scavenging.
const BELOW_CLOUD_SNOW_POLY: [f32; 6] = [22.7, 0.0, 0.0, 1321.0, 381.0, 0.0];

/// Cloud regime used by FLEXPART wet scavenging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloudScavengingRegime {
    /// No scavenging (no cloud/precip interaction for this particle).
    None,
    /// In-cloud scavenging.
    InCloud,
    /// Below-cloud scavenging.
    BelowCloud,
}

/// Precipitation and cloud-cover inputs used to derive sub-grid wet-scavenging forcing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WetPrecipitationInputs {
    /// Large-scale precipitation rate [mm/h].
    pub large_scale_precip_mm_h: f32,
    /// Convective precipitation rate [mm/h].
    pub convective_precip_mm_h: f32,
    /// Total cloud cover fraction [-].
    pub cloud_cover_fraction: f32,
}

/// Derived precipitation state used by both wet-scavenging pathways.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WetPrecipitationState {
    /// Total precipitation rate [mm/h].
    pub total_precip_mm_h: f32,
    /// Fraction of the grid cell assumed to precipitate [-].
    pub precipitating_fraction: f32,
    /// Precipitation rate inside precipitating sub-grid area [mm/h].
    pub subgrid_precip_mm_h: f32,
}

impl WetPrecipitationState {
    /// Whether this state triggers wet scavenging.
    #[must_use]
    pub fn is_active(self) -> bool {
        self.total_precip_mm_h >= MIN_TOTAL_PRECIP_MM_H
            && self.precipitating_fraction > 0.0
            && self.subgrid_precip_mm_h > 0.0
    }
}

/// Gas parameters for below-cloud scavenging (`weta_gas`, `wetb_gas`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GasBelowCloudParams {
    /// Multiplicative coefficient `A` in `A * prec^B`.
    pub coefficient_a: f32,
    /// Exponent `B` in `A * prec^B`.
    pub exponent_b: f32,
}

/// Aerosol parameters for below-cloud scavenging.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AerosolBelowCloudParams {
    /// Mean particle diameter [um] (`dquer` in FLEXPART).
    pub particle_diameter_um: f32,
    /// Rain scavenging efficiency multiplier (`crain_aero`).
    pub rain_efficiency: f32,
    /// Snow scavenging efficiency multiplier (`csnow_aero`).
    pub snow_efficiency: f32,
}

/// Inputs for below-cloud scavenging calculations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BelowCloudInputs {
    /// Precipitation rate inside precipitating sub-grid area [mm/h].
    pub subgrid_precip_mm_h: f32,
    /// Local air temperature [K].
    pub air_temperature_k: f32,
}

/// Gas parameters for in-cloud scavenging.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GasInCloudParams {
    /// Henry coefficient (`henry` in FLEXPART).
    pub henry_coefficient: f32,
}

/// Aerosol activation parameters for in-cloud scavenging.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AerosolInCloudParams {
    /// Activated fraction in liquid cloud water (`ccn_aero`).
    pub ccn_activation_fraction: f32,
    /// Activated fraction in ice cloud water (`in_aero`).
    pub ice_activation_fraction: f32,
}

/// Inputs for in-cloud scavenging calculations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InCloudInputs {
    /// Precipitation rate inside precipitating sub-grid area [mm/h].
    pub subgrid_precip_mm_h: f32,
    /// Local air temperature [K].
    pub air_temperature_k: f32,
    /// Total cloud cover fraction [-].
    pub cloud_cover_fraction: f32,
    /// Fraction of the grid cell where precipitation occurs [-].
    pub precipitating_fraction: f32,
    /// Grid-cell mean cloud water + cloud ice content.
    ///
    /// If present, this is scaled by `precipitating_fraction / cloud_cover_fraction`
    /// like FLEXPART's `ctwc * grfraction / cc`.
    /// If absent, a precipitation-based fallback parameterization is used.
    pub cloud_water_content: Option<f32>,
    /// In-cloud ratio constant (`incloud_ratio` in `par_mod.f90`).
    pub in_cloud_ratio: f32,
}

impl Default for InCloudInputs {
    fn default() -> Self {
        Self {
            subgrid_precip_mm_h: 0.0,
            air_temperature_k: BELOW_CLOUD_RAIN_SNOW_SPLIT_K,
            cloud_cover_fraction: 0.0,
            precipitating_fraction: 0.0,
            cloud_water_content: None,
            in_cloud_ratio: IN_CLOUD_RATIO_DEFAULT,
        }
    }
}

/// Parameters for one wet-scavenging mass-loss step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WetScavengingStep {
    /// Scavenging coefficient [1/s].
    pub scavenging_coefficient_s_inv: f32,
    /// Model integration step [s].
    pub dt_seconds: f32,
    /// Fraction of area where precipitation occurs [-].
    pub precipitating_fraction: f32,
}

/// Compute precipitating area fraction and sub-grid precipitation from meteo inputs.
///
/// Ported from `get_wetscav.f90` (`lfr/cfr` table and `grfraction` logic).
#[must_use]
pub fn wet_precipitation_state(inputs: WetPrecipitationInputs) -> WetPrecipitationState {
    let lsp = sanitize_non_negative(inputs.large_scale_precip_mm_h);
    let convp = sanitize_non_negative(inputs.convective_precip_mm_h);
    let total = lsp + convp;
    if total < MIN_TOTAL_PRECIP_MM_H {
        return WetPrecipitationState {
            total_precip_mm_h: total,
            precipitating_fraction: 0.0,
            subgrid_precip_mm_h: 0.0,
        };
    }

    let cloud_cover = inputs.cloud_cover_fraction.clamp(0.0, 1.0);
    let lsp_idx = precipitation_class_index(lsp);
    let conv_idx = precipitation_class_index(convp);
    let weighted_area = (lsp * LARGE_SCALE_AREA_FRACTIONS[lsp_idx]
        + convp * CONVECTIVE_AREA_FRACTIONS[conv_idx])
        / total.max(EPSILON);
    let area_fraction = (cloud_cover * weighted_area)
        .max(MIN_PRECIPITATING_FRACTION)
        .clamp(MIN_PRECIPITATING_FRACTION, 1.0);
    let subgrid_precip = total / area_fraction.max(EPSILON);

    WetPrecipitationState {
        total_precip_mm_h: total,
        precipitating_fraction: area_fraction,
        subgrid_precip_mm_h: sanitize_non_negative(subgrid_precip),
    }
}

/// Below-cloud gas scavenging coefficient [1/s].
///
/// Ported from `get_wetscav.f90`: `wetscav = weta_gas * prec^wetb_gas`.
#[must_use]
pub fn below_cloud_scavenging_coefficient_gas(
    inputs: BelowCloudInputs,
    params: GasBelowCloudParams,
) -> f32 {
    let precip = sanitize_non_negative(inputs.subgrid_precip_mm_h);
    if precip <= 0.0 {
        return 0.0;
    }

    let a = sanitize_non_negative(params.coefficient_a);
    let b = sanitize_non_negative(params.exponent_b);
    sanitize_non_negative(a * precip.powf(b))
}

/// Below-cloud aerosol scavenging coefficient [1/s].
///
/// Ported from `get_wetscav.f90`:
/// - rain branch (Laakso et al. 2003 polynomial),
/// - snow branch (Kyrö et al. 2009 polynomial).
#[must_use]
pub fn below_cloud_scavenging_coefficient_aerosol(
    inputs: BelowCloudInputs,
    params: AerosolBelowCloudParams,
) -> f32 {
    let precip = sanitize_non_negative(inputs.subgrid_precip_mm_h);
    if precip <= 0.0 {
        return 0.0;
    }

    let diameter_um = params.particle_diameter_um;
    if !diameter_um.is_finite() || diameter_um <= 0.0 {
        return 0.0;
    }
    let diameter_m = diameter_um.min(AEROSOL_DIAMETER_CAP_UM) * 1.0e-6;
    let log_d = diameter_m.log10();
    if !log_d.is_finite() || log_d.abs() < EPSILON {
        return 0.0;
    }

    if inputs.air_temperature_k >= BELOW_CLOUD_RAIN_SNOW_SPLIT_K {
        return aerosol_polynomial_scavenging(
            sanitize_non_negative(params.rain_efficiency),
            log_d,
            precip,
            &BELOW_CLOUD_RAIN_POLY,
        );
    }
    aerosol_polynomial_scavenging(
        sanitize_non_negative(params.snow_efficiency),
        log_d,
        precip,
        &BELOW_CLOUD_SNOW_POLY,
    )
}

/// In-cloud aerosol scavenging coefficient [1/s].
///
/// Ported from `get_wetscav.f90`:
/// `S_i = frac_act / cl`, then
/// `wetscav = incloud_ratio * S_i * (prec / 3.6e6)`.
#[must_use]
pub fn in_cloud_scavenging_coefficient_aerosol(
    inputs: InCloudInputs,
    params: AerosolInCloudParams,
) -> f32 {
    let precip = sanitize_non_negative(inputs.subgrid_precip_mm_h);
    if precip <= 0.0 {
        return 0.0;
    }

    let ccn = sanitize_non_negative(params.ccn_activation_fraction);
    let ice = sanitize_non_negative(params.ice_activation_fraction);
    if ccn <= 0.0 && ice <= 0.0 {
        return 0.0;
    }

    let (liquid_fraction, ice_fraction) = cloud_phase_fractions(inputs.air_temperature_k);
    let activated_fraction = liquid_fraction * ccn + ice_fraction * ice;
    if activated_fraction <= 0.0 {
        return 0.0;
    }

    let cloud_water = effective_cloud_water_content(inputs);
    if cloud_water <= 0.0 {
        return 0.0;
    }

    let scavenging_strength = activated_fraction / cloud_water;
    let ratio = sanitize_non_negative(inputs.in_cloud_ratio);
    sanitize_non_negative(ratio * scavenging_strength * precip * PRECIP_MM_H_TO_M_S)
}

/// In-cloud gas scavenging coefficient [1/s].
///
/// Ported from `get_wetscav.f90`:
/// `cle = (1-cl)/(henry*(R_air/3500)*T) + cl`, `S_i = 1/cle`,
/// then `wetscav = incloud_ratio * S_i * (prec / 3.6e6)`.
#[must_use]
pub fn in_cloud_scavenging_coefficient_gas(inputs: InCloudInputs, params: GasInCloudParams) -> f32 {
    let precip = sanitize_non_negative(inputs.subgrid_precip_mm_h);
    if precip <= 0.0 {
        return 0.0;
    }

    let henry = sanitize_non_negative(params.henry_coefficient);
    if henry <= 0.0 {
        return 0.0;
    }

    let cloud_water = effective_cloud_water_content(inputs);
    if cloud_water <= 0.0 {
        return 0.0;
    }

    let temperature_k = inputs.air_temperature_k.max(200.0);
    let partition = (1.0 - cloud_water) / (henry * (R_AIR / 3500.0) * temperature_k) + cloud_water;
    if partition <= 0.0 || !partition.is_finite() {
        return 0.0;
    }

    let scavenging_strength = 1.0 / partition;
    let ratio = sanitize_non_negative(inputs.in_cloud_ratio);
    sanitize_non_negative(ratio * scavenging_strength * precip * PRECIP_MM_H_TO_M_S)
}

/// Compute wet-scavenging coefficient [1/s] for one cloud regime.
#[must_use]
pub fn wet_scavenging_coefficient(
    regime: CloudScavengingRegime,
    below_cloud_gas: Option<(BelowCloudInputs, GasBelowCloudParams)>,
    below_cloud_aerosol: Option<(BelowCloudInputs, AerosolBelowCloudParams)>,
    in_cloud_gas: Option<(InCloudInputs, GasInCloudParams)>,
    in_cloud_aerosol: Option<(InCloudInputs, AerosolInCloudParams)>,
) -> f32 {
    match regime {
        CloudScavengingRegime::None => 0.0,
        CloudScavengingRegime::BelowCloud => {
            if let Some((inputs, params)) = below_cloud_gas {
                return below_cloud_scavenging_coefficient_gas(inputs, params);
            }
            if let Some((inputs, params)) = below_cloud_aerosol {
                return below_cloud_scavenging_coefficient_aerosol(inputs, params);
            }
            0.0
        }
        CloudScavengingRegime::InCloud => {
            if let Some((inputs, params)) = in_cloud_gas {
                return in_cloud_scavenging_coefficient_gas(inputs, params);
            }
            if let Some((inputs, params)) = in_cloud_aerosol {
                return in_cloud_scavenging_coefficient_aerosol(inputs, params);
            }
            0.0
        }
    }
}

/// Effective wet-scavenging probability for one integration step.
///
/// Matches `wetdepo.f90` mass-loss form:
/// `p = grfraction * (1 - exp(-wetscav * |dt|))`.
#[must_use]
pub fn wet_scavenging_probability_step(step: WetScavengingStep) -> f32 {
    let lambda = sanitize_non_negative(step.scavenging_coefficient_s_inv);
    let dt = step.dt_seconds.abs();
    let precip_fraction = step.precipitating_fraction.clamp(0.0, 1.0);
    if lambda <= 0.0 || dt <= 0.0 || precip_fraction <= 0.0 {
        return 0.0;
    }
    let local_probability = 1.0 - (-lambda * dt).exp();
    (local_probability * precip_fraction).clamp(0.0, 1.0)
}

/// Wet-deposited mass [kg] for one step.
#[must_use]
pub fn wet_scavenged_mass_loss_kg(initial_mass_kg: f32, step: WetScavengingStep) -> f32 {
    let mass = sanitize_non_negative(initial_mass_kg);
    let probability = wet_scavenging_probability_step(step);
    (mass * probability).clamp(0.0, mass)
}

/// Apply one wet-scavenging step to a species mass.
///
/// Returns `(remaining_mass_kg, wet_deposited_mass_kg)`.
#[must_use]
pub fn apply_wet_scavenging_mass_step(initial_mass_kg: f32, step: WetScavengingStep) -> (f32, f32) {
    let deposited = wet_scavenged_mass_loss_kg(initial_mass_kg, step);
    let remaining = (sanitize_non_negative(initial_mass_kg) - deposited).max(0.0);
    (remaining, deposited)
}

#[must_use]
fn sanitize_non_negative(value: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

#[must_use]
fn precipitation_class_index(precip_mm_h: f32) -> usize {
    if precip_mm_h > 20.0 {
        4
    } else if precip_mm_h > 8.0 {
        3
    } else if precip_mm_h > 3.0 {
        2
    } else if precip_mm_h > 1.0 {
        1
    } else {
        0
    }
}

#[must_use]
fn aerosol_polynomial_scavenging(
    efficiency: f32,
    log_d: f32,
    precip_mm_h: f32,
    poly: &[f32; 6],
) -> f32 {
    if efficiency <= 0.0 {
        return 0.0;
    }
    let exponent = poly[0]
        + poly[1] * log_d.powi(-4)
        + poly[2] * log_d.powi(-3)
        + poly[3] * log_d.powi(-2)
        + poly[4] * log_d.powi(-1)
        + poly[5] * precip_mm_h.sqrt();
    sanitize_non_negative(efficiency * 10.0_f32.powf(exponent))
}

#[must_use]
fn cloud_phase_fractions(air_temperature_k: f32) -> (f32, f32) {
    if air_temperature_k <= ICE_ONLY_TEMPERATURE_K {
        return (0.0, 1.0);
    }
    if air_temperature_k >= BELOW_CLOUD_RAIN_SNOW_SPLIT_K {
        return (1.0, 0.0);
    }

    let ice_fraction = ((air_temperature_k - BELOW_CLOUD_RAIN_SNOW_SPLIT_K)
        / (BELOW_CLOUD_RAIN_SNOW_SPLIT_K - ICE_ONLY_TEMPERATURE_K))
        .powi(2)
        .clamp(0.0, 1.0);
    let liquid_fraction = (1.0 - ice_fraction).clamp(0.0, 1.0);
    (liquid_fraction, ice_fraction)
}

#[must_use]
fn effective_cloud_water_content(inputs: InCloudInputs) -> f32 {
    if let Some(cloud_water) = inputs.cloud_water_content {
        let cloud_cover = inputs.cloud_cover_fraction.max(EPSILON);
        let precip_fraction = inputs.precipitating_fraction.max(0.0);
        return sanitize_non_negative(cloud_water * precip_fraction / cloud_cover)
            .max(MIN_CLOUD_WATER_CONTENT);
    }
    // FLEXPART fallback: precipitation-only parameterization.
    (1.0e6 * 2.0e-7 * sanitize_non_negative(inputs.subgrid_precip_mm_h).powf(0.36))
        .max(MIN_CLOUD_WATER_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precipitation_state_is_inactive_below_threshold() {
        let state = wet_precipitation_state(WetPrecipitationInputs {
            large_scale_precip_mm_h: 0.004,
            convective_precip_mm_h: 0.003,
            cloud_cover_fraction: 0.8,
        });
        assert!(!state.is_active());
        assert_eq!(state.subgrid_precip_mm_h, 0.0);
        assert_eq!(state.precipitating_fraction, 0.0);
    }

    #[test]
    fn precipitating_fraction_increases_with_cloud_cover() {
        let low_cloud = wet_precipitation_state(WetPrecipitationInputs {
            large_scale_precip_mm_h: 4.0,
            convective_precip_mm_h: 2.0,
            cloud_cover_fraction: 0.3,
        });
        let high_cloud = wet_precipitation_state(WetPrecipitationInputs {
            large_scale_precip_mm_h: 4.0,
            convective_precip_mm_h: 2.0,
            cloud_cover_fraction: 0.9,
        });
        assert!(high_cloud.precipitating_fraction > low_cloud.precipitating_fraction);
        assert!(high_cloud.subgrid_precip_mm_h < low_cloud.subgrid_precip_mm_h);
    }

    #[test]
    fn below_cloud_gas_scavenging_increases_with_precip() {
        let params = GasBelowCloudParams {
            coefficient_a: 1.0e-5,
            exponent_b: 0.7,
        };
        let low = below_cloud_scavenging_coefficient_gas(
            BelowCloudInputs {
                subgrid_precip_mm_h: 0.5,
                air_temperature_k: 280.0,
            },
            params,
        );
        let high = below_cloud_scavenging_coefficient_gas(
            BelowCloudInputs {
                subgrid_precip_mm_h: 5.0,
                air_temperature_k: 280.0,
            },
            params,
        );
        assert!(high > low);
        assert!(low.is_finite() && high.is_finite());
    }

    #[test]
    fn below_cloud_aerosol_branches_by_temperature() {
        let params = AerosolBelowCloudParams {
            particle_diameter_um: 1.0,
            rain_efficiency: 0.6,
            snow_efficiency: 0.5,
        };
        let rain = below_cloud_scavenging_coefficient_aerosol(
            BelowCloudInputs {
                subgrid_precip_mm_h: 2.0,
                air_temperature_k: 276.0,
            },
            params,
        );
        let snow = below_cloud_scavenging_coefficient_aerosol(
            BelowCloudInputs {
                subgrid_precip_mm_h: 2.0,
                air_temperature_k: 268.0,
            },
            params,
        );
        assert!(rain >= 0.0);
        assert!(snow >= 0.0);
        assert!(rain.is_finite());
        assert!(snow.is_finite());
    }

    #[test]
    fn in_cloud_aerosol_scavenging_increases_with_precip_and_activation() {
        let low = in_cloud_scavenging_coefficient_aerosol(
            InCloudInputs {
                subgrid_precip_mm_h: 1.0,
                air_temperature_k: 265.0,
                cloud_cover_fraction: 0.8,
                precipitating_fraction: 0.4,
                cloud_water_content: Some(0.2),
                in_cloud_ratio: IN_CLOUD_RATIO_DEFAULT,
            },
            AerosolInCloudParams {
                ccn_activation_fraction: 0.2,
                ice_activation_fraction: 0.2,
            },
        );
        let high = in_cloud_scavenging_coefficient_aerosol(
            InCloudInputs {
                subgrid_precip_mm_h: 6.0,
                air_temperature_k: 265.0,
                cloud_cover_fraction: 0.8,
                precipitating_fraction: 0.4,
                cloud_water_content: Some(0.2),
                in_cloud_ratio: IN_CLOUD_RATIO_DEFAULT,
            },
            AerosolInCloudParams {
                ccn_activation_fraction: 0.8,
                ice_activation_fraction: 0.8,
            },
        );
        assert!(high > low);
    }

    #[test]
    fn in_cloud_gas_scavenging_increases_with_henry() {
        let inputs = InCloudInputs {
            subgrid_precip_mm_h: 4.0,
            air_temperature_k: 278.0,
            cloud_cover_fraction: 0.7,
            precipitating_fraction: 0.5,
            cloud_water_content: Some(0.3),
            in_cloud_ratio: IN_CLOUD_RATIO_DEFAULT,
        };
        let low = in_cloud_scavenging_coefficient_gas(
            inputs,
            GasInCloudParams {
                henry_coefficient: 1.0e2,
            },
        );
        let high = in_cloud_scavenging_coefficient_gas(
            inputs,
            GasInCloudParams {
                henry_coefficient: 1.0e4,
            },
        );
        assert!(high > low);
    }

    #[test]
    fn wet_scavenging_probability_and_mass_loss_are_bounded() {
        let step = WetScavengingStep {
            scavenging_coefficient_s_inv: 0.02,
            dt_seconds: 60.0,
            precipitating_fraction: 0.6,
        };
        let probability = wet_scavenging_probability_step(step);
        assert!((0.0..=1.0).contains(&probability));

        let initial_mass = 2.5;
        let loss = wet_scavenged_mass_loss_kg(initial_mass, step);
        assert!((0.0..=initial_mass).contains(&loss));

        let (remaining, deposited) = apply_wet_scavenging_mass_step(initial_mass, step);
        assert!((remaining + deposited - initial_mass).abs() <= 1.0e-6);
    }

    #[test]
    fn wet_scavenging_probability_is_monotonic_in_lambda_dt_and_fraction() {
        let base = wet_scavenging_probability_step(WetScavengingStep {
            scavenging_coefficient_s_inv: 0.01,
            dt_seconds: 30.0,
            precipitating_fraction: 0.2,
        });
        let stronger_lambda = wet_scavenging_probability_step(WetScavengingStep {
            scavenging_coefficient_s_inv: 0.02,
            dt_seconds: 30.0,
            precipitating_fraction: 0.2,
        });
        let longer_dt = wet_scavenging_probability_step(WetScavengingStep {
            scavenging_coefficient_s_inv: 0.01,
            dt_seconds: 60.0,
            precipitating_fraction: 0.2,
        });
        let larger_fraction = wet_scavenging_probability_step(WetScavengingStep {
            scavenging_coefficient_s_inv: 0.01,
            dt_seconds: 30.0,
            precipitating_fraction: 0.8,
        });

        assert!(stronger_lambda > base);
        assert!(longer_dt > base);
        assert!(larger_fraction > base);
    }

    #[test]
    fn edge_inputs_do_not_produce_nan_or_negative_results() {
        let precip = wet_precipitation_state(WetPrecipitationInputs {
            large_scale_precip_mm_h: -1.0,
            convective_precip_mm_h: f32::NAN,
            cloud_cover_fraction: 5.0,
        });
        assert!(precip.total_precip_mm_h >= 0.0);
        assert!(precip.precipitating_fraction >= 0.0);
        assert!(precip.subgrid_precip_mm_h >= 0.0);

        let coeff = below_cloud_scavenging_coefficient_aerosol(
            BelowCloudInputs {
                subgrid_precip_mm_h: f32::INFINITY,
                air_temperature_k: 260.0,
            },
            AerosolBelowCloudParams {
                particle_diameter_um: -2.0,
                rain_efficiency: -1.0,
                snow_efficiency: -1.0,
            },
        );
        assert!(coeff >= 0.0);
        assert!(coeff.is_finite());
    }
}
