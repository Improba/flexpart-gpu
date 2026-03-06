//! CPU reference dry deposition utilities (resistance model).
//!
//! Ported from FLEXPART:
//! - `getvdep.f90` (gas deposition and resistance combination `ra + rb + rc`)
//! - `partdep.f90` (particle deposition with settling + deposition-layer resistance)
//! - `advance.f90` (deposition probability accumulation term)
//!
//! This module provides an MVP CPU reference with explicit assumptions:
//! - land-use and particle-bin fractions are normalized internally if needed;
//! - gas surface resistance `rc` is provided by caller (land-use/season logic is external);
//! - aerodynamic resistance uses a standard Monin-Obukhov log-law approximation.

use crate::constants::{GA, HREF, VON_KARMAN};
use crate::particles::Particle;
use crate::pbl::PblState;
use crate::physics::hanna::obukhov_length_from_inverse;

const EPSILON: f32 = 1.0e-5;
const USTAR_MIN: f32 = 1.0e-4;
const ROUGHNESS_MIN_M: f32 = 1.0e-5;
const REF_HEIGHT_MIN_M: f32 = 1.0;
const HREF_MIN_M: f32 = 0.1;
const VDEP_MAX_M_S: f32 = 9.999;
const AIR_GAS_CONSTANT: f32 = 287.0;

/// Meteorological inputs needed for dry deposition resistance terms.
#[derive(Clone, Copy, Debug)]
pub struct DryDepMeteoInputs {
    /// Friction velocity `u*` [m/s].
    pub friction_velocity_m_s: f32,
    /// Obukhov length `L` [m]. Use `+inf` for neutral conditions.
    pub obukhov_length_m: f32,
    /// Air temperature [K] (near-surface).
    pub air_temperature_k: f32,
    /// Surface pressure [Pa].
    pub pressure_pa: f32,
    /// Reference height for aerodynamic resistance [m].
    pub reference_height_m: f32,
}

impl DryDepMeteoInputs {
    /// Build dry-deposition inputs from one PBL grid point and met scalars.
    #[must_use]
    pub fn from_pbl_state(
        pbl: &PblState,
        i: usize,
        j: usize,
        air_temperature_k: f32,
        pressure_pa: f32,
    ) -> Self {
        let oli = pbl.oli[[i, j]];
        Self {
            friction_velocity_m_s: pbl.ustar[[i, j]],
            obukhov_length_m: obukhov_length_from_inverse(oli),
            air_temperature_k,
            pressure_pa,
            reference_height_m: HREF,
        }
    }
}

/// Land-use class contribution for gas deposition.
#[derive(Clone, Copy, Debug)]
pub struct LandUseResistance {
    /// Fractional coverage of this land-use class [-].
    pub fraction: f32,
    /// Roughness length `z0` [m].
    pub roughness_length_m: f32,
    /// Surface/canopy resistance `rc` [s/m].
    pub surface_resistance_s_m: f32,
}

/// Gas-species deposition input.
#[derive(Clone, Copy, Debug)]
pub struct GasSpeciesDepositionInput {
    /// Relative diffusivity to water vapor (`reldiff` in FLEXPART).
    /// Values <= 0 are treated as "not gas-parameterized".
    pub relative_diffusivity_to_h2o: f32,
    /// Optional constant fallback dry deposition velocity [m/s].
    pub constant_dry_velocity_m_s: Option<f32>,
}

/// Particle-bin properties used by `partdep`-style deposition.
#[derive(Clone, Copy, Debug)]
pub struct ParticleBinDepositionInput {
    /// Mass fraction of this diameter bin [-].
    pub mass_fraction: f32,
    /// Schmidt number exponent term (`Sc^(2/3)`) for this bin [-].
    pub schmidt_two_thirds: f32,
    /// Gravitational settling velocity [m/s].
    pub settling_velocity_m_s: f32,
}

/// Particle-species deposition input.
#[derive(Debug)]
pub struct ParticleSpeciesDepositionInput<'a> {
    /// Particle material density [kg/m^3]. Must be > 0 for particle pathway.
    pub density_kg_m3: f32,
    /// Diameter-bin data for this species.
    pub bins: &'a [ParticleBinDepositionInput],
}

/// Compute molecular diffusivity of water vapor [m^2/s].
///
/// Ported from `getvdep.f90`:
/// `diffh2o = 2.11e-5 * (T/273.15)^1.94 * (101325 / p)`.
#[must_use]
pub fn water_vapor_diffusivity_m2_s(air_temperature_k: f32, pressure_pa: f32) -> f32 {
    let t = air_temperature_k.max(150.0);
    let p = pressure_pa.max(1.0);
    let diff = 2.11e-5 * (t / 273.15).powf(1.94) * (101_325.0 / p);
    sanitize_non_negative(diff, 2.11e-5)
}

/// Compute dynamic viscosity of air [kg/(m·s)].
///
/// Ported from `getvdep.f90` piecewise polynomial in Celsius.
#[must_use]
pub fn dynamic_viscosity_air_kg_m_s(air_temperature_k: f32) -> f32 {
    let tc = air_temperature_k - 273.15;
    let dynamic = if tc < 0.0 {
        (1.718 + 0.0049 * tc - 1.2e-05 * tc * tc) * 1.0e-5
    } else {
        (1.718 + 0.0049 * tc) * 1.0e-5
    };
    sanitize_non_negative(dynamic, 1.8e-5)
}

/// Compute kinematic viscosity of air [m^2/s].
#[must_use]
pub fn kinematic_viscosity_air_m2_s(air_temperature_k: f32, pressure_pa: f32) -> f32 {
    let dynamic = dynamic_viscosity_air_kg_m_s(air_temperature_k);
    let density = pressure_pa.max(1.0) / (AIR_GAS_CONSTANT * air_temperature_k.max(150.0));
    sanitize_non_negative(dynamic / density.max(1.0e-6), 1.5e-5)
}

/// Compute aerodynamic resistance `ra` [s/m].
///
/// This is the CPU resistance-model approximation equivalent to FLEXPART's
/// `raerod(...)` call used in `getvdep.f90`.
#[must_use]
pub fn aerodynamic_resistance_s_m(inputs: DryDepMeteoInputs, roughness_length_m: f32) -> f32 {
    let ustar = inputs.friction_velocity_m_s.max(USTAR_MIN);
    let z0 = roughness_length_m.max(ROUGHNESS_MIN_M);
    let z_ref = inputs
        .reference_height_m
        .max(REF_HEIGHT_MIN_M)
        .max(1.01 * z0);
    let inv_l = inverse_obukhov(inputs.obukhov_length_m);

    let psi_ref = stability_correction_heat(z_ref * inv_l);
    let psi_z0 = stability_correction_heat(z0 * inv_l);
    let log_term = (z_ref / z0).ln();
    let numerator = (log_term - psi_ref + psi_z0).max(EPSILON);
    let ra = numerator / (VON_KARMAN * ustar);
    sanitize_non_negative(ra, 1.0)
}

/// Compute gas quasi-laminar resistance `rb` [s/m].
///
/// Uses `rb = 2/(k*u*) * Sc^(2/3)` with `Sc = nu / D_species` and
/// `D_species = relative_diffusivity_to_h2o * D_h2o`.
#[must_use]
pub fn gas_quasi_laminar_resistance_s_m(
    meteo: DryDepMeteoInputs,
    relative_diffusivity_to_h2o: f32,
) -> f32 {
    if relative_diffusivity_to_h2o <= 0.0 || !relative_diffusivity_to_h2o.is_finite() {
        return f32::INFINITY;
    }

    let ustar = meteo.friction_velocity_m_s.max(USTAR_MIN);
    let nu = kinematic_viscosity_air_m2_s(meteo.air_temperature_k, meteo.pressure_pa);
    let diff_h2o = water_vapor_diffusivity_m2_s(meteo.air_temperature_k, meteo.pressure_pa);
    let diff_species = (relative_diffusivity_to_h2o * diff_h2o).max(1.0e-12);
    let schmidt_number = (nu / diff_species).max(EPSILON);
    let rb = 2.0 / (VON_KARMAN * ustar) * schmidt_number.powf(2.0 / 3.0);
    sanitize_non_negative(rb, 100.0)
}

/// Compute dry deposition velocity from resistance sum `ra + rb + rc`.
///
/// Ported from `getvdep.f90`:
/// `vd = 1 / (ra + rb + rc)` for positive denominator.
#[must_use]
pub fn dry_deposition_velocity_from_resistances_m_s(
    aerodynamic_resistance_s_m: f32,
    quasi_laminar_resistance_s_m: f32,
    surface_resistance_s_m: f32,
) -> f32 {
    let ra = aerodynamic_resistance_s_m.max(0.0);
    let rb = quasi_laminar_resistance_s_m.max(0.0);
    let rc = surface_resistance_s_m.max(0.0);
    let denom = ra + rb + rc;
    if !denom.is_finite() {
        return 0.0;
    }
    if denom <= 0.0 {
        return VDEP_MAX_M_S;
    }
    (1.0 / denom).clamp(0.0, VDEP_MAX_M_S)
}

/// Compute weighted aerodynamic resistance across land-use classes.
#[must_use]
pub fn weighted_aerodynamic_resistance_s_m(
    meteo: DryDepMeteoInputs,
    land_use: &[LandUseResistance],
) -> f32 {
    let mut weighted_sum = 0.0;
    let mut total_fraction = 0.0;

    for class in land_use {
        if class.fraction > EPSILON {
            let ra = aerodynamic_resistance_s_m(meteo, class.roughness_length_m);
            weighted_sum += class.fraction * ra;
            total_fraction += class.fraction;
        }
    }

    if total_fraction <= EPSILON {
        return aerodynamic_resistance_s_m(meteo, 0.1);
    }

    sanitize_non_negative(weighted_sum / total_fraction, 100.0)
}

/// Compute gas dry-deposition velocity [m/s], weighted by land-use fractions.
///
/// Follows `getvdep.f90` logic:
/// - `rb` computed once for species;
/// - `ra` and `rc` per land-use class;
/// - weighted sum over classes.
#[must_use]
pub fn gas_dry_deposition_velocity_m_s(
    meteo: DryDepMeteoInputs,
    species: GasSpeciesDepositionInput,
    land_use: &[LandUseResistance],
) -> f32 {
    if species.relative_diffusivity_to_h2o <= 0.0 {
        return species
            .constant_dry_velocity_m_s
            .unwrap_or(0.0)
            .clamp(0.0, VDEP_MAX_M_S);
    }

    let rb = gas_quasi_laminar_resistance_s_m(meteo, species.relative_diffusivity_to_h2o);
    if !rb.is_finite() {
        return 0.0;
    }

    let mut weighted_sum = 0.0;
    let mut total_fraction = 0.0;
    for class in land_use {
        if class.fraction > EPSILON {
            let ra = aerodynamic_resistance_s_m(meteo, class.roughness_length_m);
            let vd =
                dry_deposition_velocity_from_resistances_m_s(ra, rb, class.surface_resistance_s_m);
            weighted_sum += class.fraction * vd;
            total_fraction += class.fraction;
        }
    }

    if total_fraction <= EPSILON {
        return 0.0;
    }

    (weighted_sum / total_fraction).clamp(0.0, VDEP_MAX_M_S)
}

/// Compute particle dry-deposition velocity [m/s] for one species.
///
/// Ported from `partdep.f90`:
/// - settling + resistance interaction:
///   `vdep_bin = vset + 1/(ra + rdp + ra*rdp*vset)`.
#[must_use]
pub fn particle_dry_deposition_velocity_m_s(
    species: &ParticleSpeciesDepositionInput<'_>,
    aerodynamic_resistance_s_m: f32,
    friction_velocity_m_s: f32,
    kinematic_viscosity_m2_s: f32,
) -> f32 {
    if species.density_kg_m3 <= 0.0 {
        return 0.0;
    }

    let ustar = friction_velocity_m_s.max(0.0);
    let ra = aerodynamic_resistance_s_m.max(0.0);
    let nu = kinematic_viscosity_m2_s.max(1.0e-12);
    let mut total_fraction = 0.0;
    let mut weighted_vdep = 0.0;

    for bin in species.bins {
        if bin.mass_fraction <= 0.0 {
            continue;
        }
        let vdep_bin = if ustar > EPSILON {
            let stokes = (bin.settling_velocity_m_s / GA) * (ustar * ustar) / nu;
            let alpha = if stokes.abs() <= 1.0e-12 {
                f32::NEG_INFINITY
            } else {
                -3.0 / stokes
            };

            let schmi = bin.schmidt_two_thirds.max(EPSILON);
            let rdp = if alpha <= EPSILON.log10() {
                1.0 / (schmi * ustar)
            } else {
                1.0 / ((schmi + 10.0_f32.powf(alpha)) * ustar)
            };

            let denom = ra + rdp + ra * rdp * bin.settling_velocity_m_s;
            let resistance_term = if denom > EPSILON && denom.is_finite() {
                1.0 / denom
            } else {
                0.0
            };
            (bin.settling_velocity_m_s + resistance_term).max(0.0)
        } else {
            bin.settling_velocity_m_s.max(0.0)
        };

        weighted_vdep += bin.mass_fraction * vdep_bin;
        total_fraction += bin.mass_fraction;
    }

    if total_fraction <= EPSILON {
        return 0.0;
    }
    (weighted_vdep / total_fraction).clamp(0.0, VDEP_MAX_M_S)
}

/// Returns true when particle is in the near-surface deposition layer.
///
/// FLEXPART computes dry deposition only for `z < 2*href`.
#[must_use]
pub fn in_dry_deposition_layer(particle: &Particle, reference_height_m: f32) -> bool {
    let href = reference_height_m.max(HREF_MIN_M);
    particle.pos_z.is_finite() && particle.pos_z < 2.0 * href
}

/// One-step dry-deposition probability from velocity [m/s].
///
/// Uses the same exponential form as `advance.f90`:
/// `p = 1 - exp(-vdep * |dt| / (2*href))`.
#[must_use]
pub fn dry_deposition_probability_step(
    deposition_velocity_m_s: f32,
    dt_seconds: f32,
    reference_height_m: f32,
) -> f32 {
    let decay =
        dry_deposition_decay_factor(deposition_velocity_m_s, dt_seconds, reference_height_m);
    (1.0 - decay).clamp(0.0, 1.0)
}

/// Accumulate dry-deposition probability over multiple substeps.
///
/// Ported from `advance.f90` update:
/// `prob_new = 1 + (prob_old - 1) * exp(-vdep*|dt|/(2*href))`.
#[must_use]
pub fn accumulate_dry_deposition_probability(
    previous_probability: f32,
    deposition_velocity_m_s: f32,
    dt_seconds: f32,
    reference_height_m: f32,
) -> f32 {
    let prev = previous_probability.clamp(0.0, 1.0);
    let decay =
        dry_deposition_decay_factor(deposition_velocity_m_s, dt_seconds, reference_height_m);
    (1.0 + (prev - 1.0) * decay).clamp(0.0, 1.0)
}

#[must_use]
fn dry_deposition_decay_factor(
    deposition_velocity_m_s: f32,
    dt_seconds: f32,
    reference_height_m: f32,
) -> f32 {
    let href = reference_height_m.max(HREF_MIN_M);
    let vdep = deposition_velocity_m_s.max(0.0);
    let exponent = -vdep * dt_seconds.abs() / (2.0 * href);
    exponent.exp().clamp(0.0, 1.0)
}

#[must_use]
fn inverse_obukhov(obukhov_length_m: f32) -> f32 {
    if !obukhov_length_m.is_finite() || obukhov_length_m.abs() < 1.0 {
        0.0
    } else {
        1.0 / obukhov_length_m
    }
}

#[must_use]
fn stability_correction_heat(zeta: f32) -> f32 {
    if !zeta.is_finite() || zeta.abs() < 1.0e-6 {
        0.0
    } else if zeta > 0.0 {
        -5.0 * zeta
    } else {
        let x = (1.0 - 16.0 * zeta).max(1.0).powf(0.25);
        2.0 * ((1.0 + x * x) * 0.5).ln()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_meteo() -> DryDepMeteoInputs {
        DryDepMeteoInputs {
            friction_velocity_m_s: 0.35,
            obukhov_length_m: -120.0,
            air_temperature_k: 290.0,
            pressure_pa: 101_325.0,
            reference_height_m: HREF,
        }
    }

    #[test]
    fn dry_deposition_velocity_is_non_negative_and_bounded() {
        for ra in [0.0_f32, 1.0, 10.0, 100.0, 1_000.0] {
            for rb in [0.0_f32, 5.0, 20.0, 500.0] {
                for rc in [0.0_f32, 10.0, 100.0, 1_000.0] {
                    let vd = dry_deposition_velocity_from_resistances_m_s(ra, rb, rc);
                    assert!(vd.is_finite());
                    assert!((0.0..=VDEP_MAX_M_S).contains(&vd));
                }
            }
        }
    }

    #[test]
    fn larger_aerodynamic_resistance_reduces_vdep() {
        let rb = 20.0;
        let rc = 80.0;
        let low_ra = dry_deposition_velocity_from_resistances_m_s(30.0, rb, rc);
        let high_ra = dry_deposition_velocity_from_resistances_m_s(200.0, rb, rc);
        assert!(high_ra < low_ra);
    }

    #[test]
    fn gas_weighted_vdep_is_stable_and_bounded() {
        let meteo = sample_meteo();
        let species = GasSpeciesDepositionInput {
            relative_diffusivity_to_h2o: 0.8,
            constant_dry_velocity_m_s: None,
        };
        let land_use = [
            LandUseResistance {
                fraction: 0.6,
                roughness_length_m: 0.1,
                surface_resistance_s_m: 120.0,
            },
            LandUseResistance {
                fraction: 0.4,
                roughness_length_m: 1.0,
                surface_resistance_s_m: 60.0,
            },
        ];

        let vd = gas_dry_deposition_velocity_m_s(meteo, species, &land_use);
        assert!(vd.is_finite());
        assert!((0.0..=VDEP_MAX_M_S).contains(&vd));
    }

    #[test]
    fn particle_vdep_is_stable_and_bounded() {
        let species = ParticleSpeciesDepositionInput {
            density_kg_m3: 1_500.0,
            bins: &[
                ParticleBinDepositionInput {
                    mass_fraction: 0.3,
                    schmidt_two_thirds: 0.8,
                    settling_velocity_m_s: 0.001,
                },
                ParticleBinDepositionInput {
                    mass_fraction: 0.7,
                    schmidt_two_thirds: 1.5,
                    settling_velocity_m_s: 0.02,
                },
            ],
        };

        let vd = particle_dry_deposition_velocity_m_s(&species, 120.0, 0.3, 1.5e-5);
        assert!(vd.is_finite());
        assert!((0.0..=VDEP_MAX_M_S).contains(&vd));
    }

    #[test]
    fn probability_helpers_are_bounded_and_monotonic() {
        let step_low = dry_deposition_probability_step(0.002, 60.0, HREF);
        let step_high = dry_deposition_probability_step(0.02, 60.0, HREF);
        assert!(step_high > step_low);
        assert!((0.0..=1.0).contains(&step_low));
        assert!((0.0..=1.0).contains(&step_high));

        let mut p = 0.0;
        for _ in 0..12 {
            p = accumulate_dry_deposition_probability(p, 0.01, 30.0, HREF);
            assert!(p.is_finite());
            assert!((0.0..=1.0).contains(&p));
        }
        assert!(p > 0.0);
    }

    #[test]
    fn no_nan_or_inf_on_edge_inputs() {
        let meteo_cases = [
            DryDepMeteoInputs {
                friction_velocity_m_s: 0.0,
                obukhov_length_m: f32::INFINITY,
                air_temperature_k: 250.0,
                pressure_pa: 80_000.0,
                reference_height_m: 15.0,
            },
            DryDepMeteoInputs {
                friction_velocity_m_s: 1.2,
                obukhov_length_m: -10.0,
                air_temperature_k: 320.0,
                pressure_pa: 110_000.0,
                reference_height_m: 30.0,
            },
        ];

        for meteo in meteo_cases {
            let ra = aerodynamic_resistance_s_m(meteo, 0.001);
            let rb = gas_quasi_laminar_resistance_s_m(meteo, 0.5);
            let vd = dry_deposition_velocity_from_resistances_m_s(ra, rb, 200.0);

            assert!(ra.is_finite());
            assert!(rb.is_finite());
            assert!(vd.is_finite());
        }
    }
}
