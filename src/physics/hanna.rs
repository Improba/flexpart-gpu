//! CPU reference for Hanna turbulence parameters.
//!
//! Ported from FLEXPART `hanna.f90` (routine `hanna`, lines 38-107):
//! - Neutral regime: `sigma_u`, `sigma_v`, `sigma_w`, `tlu`, `tlv`, `tlw`
//! - Unstable regime: same quantities with convective scaling
//! - Stable regime: same quantities with stable scaling
//!
//! The implementation follows Hanna (1982) parameterizations used by FLEXPART.
//! Mapping notes:
//! - Fortran `sigu/sigv/sigw` -> [`HannaParams::sigu`], [`HannaParams::sigv`], [`HannaParams::sigw`]
//! - Fortran `tlu/tlv/tlw` -> [`HannaParams::tlu`], [`HannaParams::tlv`], [`HannaParams::tlw`]
//! - Fortran `dsigwdz` -> [`HannaParams::dsigwdz`]

use crate::pbl::{HannaParams, PblState, StabilityClass};

const USTAR_MIN: f32 = 1.0e-4;
const TLU_MIN: f32 = 10.0;
const TLV_MIN: f32 = 10.0;
const TLW_MIN: f32 = 30.0;
const SIGMA_FLOOR: f32 = 1.0e-2;
const H_MIN: f32 = 1.0;

/// Inputs required for Hanna turbulence parameterization at one particle position.
#[derive(Clone, Copy, Debug)]
pub struct HannaInputs {
    /// Friction velocity u* [m/s].
    pub ust: f32,
    /// Convective velocity scale w* [m/s].
    pub wst: f32,
    /// Obukhov length L [m].
    pub ol: f32,
    /// Mixing height h [m].
    pub h: f32,
    /// Particle height above ground z [m].
    pub z: f32,
}

/// Compute Hanna turbulence parameters for one particle location.
///
/// This is the CPU reference implementation for task H-01 and mirrors
/// `flexpart/src/hanna.f90`.
///
/// Assumptions for robust/deterministic behavior:
/// - `z` is clamped to `[0, h]` (the Fortran routine is designed for in-PBL usage).
/// - `h` is lower-bounded by 1 m to avoid divisions by zero.
/// - very small `|1/L|` from gridded met fields can be represented by `L = +∞`.
#[allow(clippy::similar_names)]
#[must_use]
pub fn compute_hanna_params(inputs: HannaInputs) -> HannaParams {
    let ust = inputs.ust.max(USTAR_MIN);
    let wst = inputs.wst.max(0.0);
    let h = inputs.h.max(H_MIN);
    let z = inputs.z.clamp(0.0, h);
    let ol = inputs.ol;
    let zeta = (z / h).clamp(0.0, 1.0);

    let mut sigu;
    let mut sigv;
    let mut sigw;
    let mut dsigwdz;
    let mut tlu;
    let mut tlv;
    let mut tlw;

    // Fortran branch ordering:
    // 1) neutral if h/abs(ol) < 1
    // 2) unstable if ol < 0
    // 3) stable otherwise
    if !ol.is_finite() || ol == 0.0 || h / ol.abs() < 1.0 {
        let corr = z / ust;
        sigu = SIGMA_FLOOR + 2.0 * ust * (-3.0e-4 * corr).exp();
        sigw = 1.3 * ust * (-2.0e-4 * corr).exp();
        dsigwdz = -2.0e-4 * sigw;
        sigw += SIGMA_FLOOR;
        sigv = sigw;

        tlu = 0.5 * z / sigw / (1.0 + 1.5e-3 * corr);
        tlv = tlu;
        tlw = tlu;
    } else if ol < 0.0 {
        sigu = SIGMA_FLOOR + ust * (12.0 - 0.5 * h / ol).powf(1.0 / 3.0);
        sigv = sigu;

        let sigw_sq = 1.2 * wst * wst * (1.0 - 0.9 * zeta) * zeta.powf(2.0 / 3.0)
            + (1.8 - 1.4 * zeta) * ust * ust;
        sigw = sigw_sq.sqrt() + SIGMA_FLOOR;

        dsigwdz = 0.5 / sigw / h
            * (-1.4 * ust * ust
                + wst
                    * wst
                    * (0.8 * zeta.max(1.0e-3).powf(-1.0 / 3.0) - 1.8 * zeta.powf(2.0 / 3.0)));

        tlu = 0.15 * h / sigu;
        tlv = tlu;
        if z < ol.abs() {
            tlw = 0.1 * z / (sigw * (0.55 - 0.38 * (z / ol).abs()));
        } else if zeta < 0.1 {
            tlw = 0.59 * z / sigw;
        } else {
            tlw = 0.15 * h / sigw * (1.0 - (-5.0 * zeta).exp());
        }
    } else {
        sigu = SIGMA_FLOOR + 2.0 * ust * (1.0 - zeta);
        sigv = SIGMA_FLOOR + 1.3 * ust * (1.0 - zeta);
        sigw = sigv;
        dsigwdz = -1.3 * ust / h;

        tlu = 0.15 * h / sigu * zeta.sqrt();
        tlv = 0.467 * tlu;
        tlw = 0.1 * h / sigw * zeta.powf(0.8);
    }

    tlu = tlu.max(TLU_MIN);
    tlv = tlv.max(TLV_MIN);
    tlw = tlw.max(TLW_MIN);

    if dsigwdz == 0.0 {
        dsigwdz = 1.0e-10;
    }

    // Keep output numerically safe for downstream Langevin integration.
    if !sigu.is_finite() || sigu <= 0.0 {
        sigu = SIGMA_FLOOR;
    }
    if !sigv.is_finite() || sigv <= 0.0 {
        sigv = SIGMA_FLOOR;
    }
    if !sigw.is_finite() || sigw <= 0.0 {
        sigw = SIGMA_FLOOR;
    }
    if !tlu.is_finite() {
        tlu = TLU_MIN;
    }
    if !tlv.is_finite() {
        tlv = TLV_MIN;
    }
    if !tlw.is_finite() {
        tlw = TLW_MIN;
    }
    if !dsigwdz.is_finite() {
        dsigwdz = 1.0e-10;
    }

    let dsigw_sq_dz = 2.0 * sigw * dsigwdz;

    HannaParams {
        ust,
        wst,
        ol,
        h,
        zeta,
        sigu,
        sigv,
        sigw,
        dsigwdz,
        dsigw2dz: dsigw_sq_dz,
        tlu,
        tlv,
        tlw,
        _pad: [0.0; 3],
    }
}

/// Convert inverse Obukhov length `1/L` to Obukhov length `L`.
///
/// Neutral conditions (`|1/L|` very small) are mapped to `+∞`.
#[must_use]
pub fn obukhov_length_from_inverse(oli: f32) -> f32 {
    match StabilityClass::from_inverse_obukhov(oli) {
        StabilityClass::Neutral => f32::INFINITY,
        _ => 1.0 / oli,
    }
}

/// Convenience API: compute Hanna parameters from the gridded PBL state.
#[must_use]
pub fn compute_hanna_params_from_pbl(
    pbl: &PblState,
    i: usize,
    j: usize,
    particle_height_m: f32,
) -> HannaParams {
    let oli = pbl.oli[[i, j]];
    let ol = obukhov_length_from_inverse(oli);

    compute_hanna_params(HannaInputs {
        ust: pbl.ustar[[i, j]],
        wst: pbl.wstar[[i, j]],
        ol,
        h: pbl.hmix[[i, j]],
        z: particle_height_m,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbl::PblState;
    use approx::assert_abs_diff_eq;

    #[derive(Clone, Copy)]
    struct SigmaTableCase {
        name: &'static str,
        inputs: HannaInputs,
        expected_sigu_over_ust: f32,
        expected_sigw_over_ust: f32,
        relative_tolerance: f32,
    }

    fn assert_table_style_ratio_close(
        case_name: &str,
        quantity: &str,
        actual: f32,
        expected: f32,
        relative_tolerance: f32,
    ) {
        let absolute_tolerance = expected.abs() * relative_tolerance;
        let lower = expected - absolute_tolerance;
        let upper = expected + absolute_tolerance;
        assert!(
            (lower..=upper).contains(&actual),
            "{case_name}: {quantity}={actual:.4} not within table-style range [{lower:.4}, {upper:.4}] around expected {expected:.4}",
        );
    }

    #[test]
    fn test_hanna_neutral_sigmas_decrease_with_height() {
        let near_surface = compute_hanna_params(HannaInputs {
            ust: 0.4,
            wst: 0.0,
            ol: 2_000.0, // neutral by h/|L| < 1
            h: 1_000.0,
            z: 20.0,
        });
        let aloft = compute_hanna_params(HannaInputs {
            ust: 0.4,
            wst: 0.0,
            ol: 2_000.0,
            h: 1_000.0,
            z: 600.0,
        });

        assert!(aloft.sigu < near_surface.sigu);
        assert!(aloft.sigw < near_surface.sigw);
    }

    #[test]
    fn test_hanna_stable_sigma_u_decreases_with_height() {
        let low = compute_hanna_params(HannaInputs {
            ust: 0.35,
            wst: 0.0,
            ol: 100.0, // stable by sign and h/|L| >= 1
            h: 1_000.0,
            z: 50.0,
        });
        let high = compute_hanna_params(HannaInputs {
            ust: 0.35,
            wst: 0.0,
            ol: 100.0,
            h: 1_000.0,
            z: 500.0,
        });

        assert!(high.sigu < low.sigu);
    }

    #[test]
    fn test_hanna_unstable_vs_stable_vertical_sigma_difference() {
        let stable = compute_hanna_params(HannaInputs {
            ust: 0.3,
            wst: 1.5,
            ol: 120.0,
            h: 1_000.0,
            z: 100.0,
        });
        let unstable = compute_hanna_params(HannaInputs {
            ust: 0.3,
            wst: 1.5,
            ol: -120.0,
            h: 1_000.0,
            z: 100.0,
        });

        assert!(unstable.sigw > stable.sigw);
        assert!(unstable.tlu >= stable.tlu);
    }

    #[test]
    fn test_hanna_no_nan_or_inf_for_realistic_ranges() {
        let ustars = [0.05_f32, 0.2, 0.6];
        let wstars = [0.0_f32, 0.8, 2.5];
        let ols = [-500.0_f32, -80.0, 80.0, 500.0, 2.0e6];
        let hmix = [100.0_f32, 800.0, 2_500.0];
        let heights = [0.0_f32, 5.0, 50.0, 200.0, 700.0, 2_000.0];

        for ust in ustars {
            for wst in wstars {
                for ol in ols {
                    for h in hmix {
                        for z in heights {
                            let p = compute_hanna_params(HannaInputs { ust, wst, ol, h, z });
                            let scalars = [
                                p.ust, p.wst, p.ol, p.h, p.zeta, p.sigu, p.sigv, p.sigw, p.dsigwdz,
                                p.dsigw2dz, p.tlu, p.tlv, p.tlw,
                            ];
                            for value in scalars {
                                assert!(
                                    value.is_finite(),
                                    "non-finite output for ust={ust}, wst={wst}, ol={ol}, h={h}, z={z}",
                                );
                            }
                            assert!(p.sigu > 0.0);
                            assert!(p.sigv > 0.0);
                            assert!(p.sigw > 0.0);
                            assert!(p.tlu >= TLU_MIN);
                            assert!(p.tlv >= TLV_MIN);
                            assert!(p.tlw >= TLW_MIN);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_hanna_from_pbl_matches_direct_call() {
        let mut pbl = PblState::new(1, 1);
        pbl.ustar[[0, 0]] = 0.25;
        pbl.wstar[[0, 0]] = 1.8;
        pbl.hmix[[0, 0]] = 900.0;
        pbl.oli[[0, 0]] = -0.02; // L = -50 m

        let from_grid = compute_hanna_params_from_pbl(&pbl, 0, 0, 120.0);
        let direct = compute_hanna_params(HannaInputs {
            ust: 0.25,
            wst: 1.8,
            ol: -50.0,
            h: 900.0,
            z: 120.0,
        });

        assert_abs_diff_eq!(from_grid.ust, direct.ust, epsilon = 1.0e-6);
        assert_abs_diff_eq!(from_grid.wst, direct.wst, epsilon = 1.0e-6);
        assert_abs_diff_eq!(from_grid.ol, direct.ol, epsilon = 1.0e-6);
        assert_abs_diff_eq!(from_grid.h, direct.h, epsilon = 1.0e-6);
        assert_abs_diff_eq!(from_grid.zeta, direct.zeta, epsilon = 1.0e-6);
        assert_abs_diff_eq!(from_grid.sigu, direct.sigu, epsilon = 1.0e-6);
        assert_abs_diff_eq!(from_grid.sigv, direct.sigv, epsilon = 1.0e-6);
        assert_abs_diff_eq!(from_grid.sigw, direct.sigw, epsilon = 1.0e-6);
        assert_abs_diff_eq!(from_grid.dsigwdz, direct.dsigwdz, epsilon = 1.0e-6);
        assert_abs_diff_eq!(from_grid.dsigw2dz, direct.dsigw2dz, epsilon = 1.0e-6);
        assert_abs_diff_eq!(from_grid.tlu, direct.tlu, epsilon = 1.0e-6);
        assert_abs_diff_eq!(from_grid.tlv, direct.tlv, epsilon = 1.0e-6);
        assert_abs_diff_eq!(from_grid.tlw, direct.tlw, epsilon = 1.0e-6);
    }

    #[test]
    fn test_hanna_stable_sigma_matches_table_style_reference() {
        let stable_cases = [
            SigmaTableCase {
                name: "stable_low_z_over_h",
                inputs: HannaInputs {
                    ust: 0.30,
                    wst: 0.0,
                    ol: 120.0,
                    h: 1_000.0,
                    z: 100.0,
                },
                expected_sigu_over_ust: 1.83,
                expected_sigw_over_ust: 1.20,
                relative_tolerance: 0.20,
            },
            SigmaTableCase {
                name: "stable_mid_z_over_h",
                inputs: HannaInputs {
                    ust: 0.28,
                    wst: 0.0,
                    ol: 150.0,
                    h: 1_000.0,
                    z: 500.0,
                },
                expected_sigu_over_ust: 1.04,
                expected_sigw_over_ust: 0.66,
                relative_tolerance: 0.22,
            },
        ];

        for case in stable_cases {
            let params = compute_hanna_params(case.inputs);
            let sigu_over_ust = params.sigu / params.ust;
            let sigw_over_ust = params.sigw / params.ust;

            assert_table_style_ratio_close(
                case.name,
                "sigma_u/u*",
                sigu_over_ust,
                case.expected_sigu_over_ust,
                case.relative_tolerance,
            );
            assert_table_style_ratio_close(
                case.name,
                "sigma_w/u*",
                sigw_over_ust,
                case.expected_sigw_over_ust,
                case.relative_tolerance,
            );
            assert_abs_diff_eq!(params.sigv, params.sigw, epsilon = 1.0e-6);
        }
    }

    #[test]
    fn test_hanna_neutral_sigma_matches_table_style_reference() {
        let neutral_cases = [
            SigmaTableCase {
                name: "neutral_low_z_over_h",
                inputs: HannaInputs {
                    ust: 0.30,
                    wst: 0.0,
                    ol: 5_000.0,
                    h: 1_000.0,
                    z: 50.0,
                },
                expected_sigu_over_ust: 1.93,
                expected_sigw_over_ust: 1.29,
                relative_tolerance: 0.16,
            },
            SigmaTableCase {
                name: "neutral_aloft",
                inputs: HannaInputs {
                    ust: 0.32,
                    wst: 0.0,
                    ol: 5_000.0,
                    h: 1_000.0,
                    z: 600.0,
                },
                expected_sigu_over_ust: 1.17,
                expected_sigw_over_ust: 0.93,
                relative_tolerance: 0.20,
            },
        ];

        for case in neutral_cases {
            let params = compute_hanna_params(case.inputs);
            let sigu_over_ust = params.sigu / params.ust;
            let sigw_over_ust = params.sigw / params.ust;

            assert_table_style_ratio_close(
                case.name,
                "sigma_u/u*",
                sigu_over_ust,
                case.expected_sigu_over_ust,
                case.relative_tolerance,
            );
            assert_table_style_ratio_close(
                case.name,
                "sigma_w/u*",
                sigw_over_ust,
                case.expected_sigw_over_ust,
                case.relative_tolerance,
            );
            assert_abs_diff_eq!(params.sigv, params.sigw, epsilon = 1.0e-6);
        }
    }

    #[test]
    fn test_hanna_unstable_sigma_matches_table_style_reference() {
        let unstable_cases = [
            SigmaTableCase {
                name: "unstable_low_z_over_h",
                inputs: HannaInputs {
                    ust: 0.30,
                    wst: 2.0,
                    ol: -80.0,
                    h: 1_000.0,
                    z: 100.0,
                },
                expected_sigu_over_ust: 2.66,
                expected_sigw_over_ust: 3.51,
                relative_tolerance: 0.22,
            },
            SigmaTableCase {
                name: "unstable_mid_z_over_h",
                inputs: HannaInputs {
                    ust: 0.30,
                    wst: 2.0,
                    ol: -80.0,
                    h: 1_000.0,
                    z: 600.0,
                },
                expected_sigu_over_ust: 2.66,
                expected_sigw_over_ust: 4.32,
                relative_tolerance: 0.22,
            },
        ];

        for case in unstable_cases {
            let params = compute_hanna_params(case.inputs);
            let sigu_over_ust = params.sigu / params.ust;
            let sigw_over_ust = params.sigw / params.ust;

            assert_table_style_ratio_close(
                case.name,
                "sigma_u/u*",
                sigu_over_ust,
                case.expected_sigu_over_ust,
                case.relative_tolerance,
            );
            assert_table_style_ratio_close(
                case.name,
                "sigma_w/u*",
                sigw_over_ust,
                case.expected_sigw_over_ust,
                case.relative_tolerance,
            );
            assert_abs_diff_eq!(params.sigu, params.sigv, epsilon = 1.0e-6);
            assert!(params.sigw > params.sigu);
        }
    }
}
