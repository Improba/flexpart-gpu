//! CPU reference CBL skewed PDF utilities (H-05).
//!
//! Ported as a scientific MVP from FLEXPART `cbl.f90` and
//! `re_initialize_particle.f90`:
//! - bi-Gaussian reconstruction of vertical velocity PDF from moments
//! - deterministic sampling helpers for turbulence-step integration
//!
//! Scientific assumptions in this MVP:
//! - focus on PDF moments + sampling path used by CBL branching
//! - no full `Phi/Q/ath` drift closure from `cbl.f90` yet
//! - numerically robust fallbacks for near-neutral / near-zero skewness
//!   are used to avoid singular bi-Gaussian parameterization.

use crate::constants::PI;

use super::langevin::box_muller_normals;
use super::rng::PhiloxRng;

const FLUARW_FACTOR: f32 = 2.0 / 3.0;
const MIN_H_M: f32 = 1.0;
const MIN_SIGMA_W_M_S: f32 = 1.0e-5;
const SKEW_EPS: f32 = 1.0e-6;
const W3_EPS: f32 = 1.0e-6;

/// Inputs for CBL skewed-PDF construction at one particle location.
#[derive(Clone, Copy, Debug)]
pub struct CblPdfInputs {
    /// Particle height above ground z \[m\].
    pub z_m: f32,
    /// Boundary-layer height h \[m\].
    pub h_m: f32,
    /// Convective velocity scale w* \[m/s\].
    pub wstar_m_s: f32,
    /// Vertical velocity standard deviation `sigma_w` \[m/s\].
    pub sigma_w_m_s: f32,
    /// Obukhov length L \[m\].
    pub obukhov_length_m: f32,
}

/// Moments that define the CBL vertical velocity PDF.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CblPdfMoments {
    /// Second raw moment E[w'^2] \[(m/s)^2\].
    pub second_moment_w2: f32,
    /// Third raw moment E[w'^3] \[(m/s)^3\].
    pub third_moment_w3: f32,
    /// Dimensionless skewness E[w'^3] / E[w'^2]^(3/2).
    pub skewness: f32,
}

/// One Gaussian component in the bi-Gaussian CBL PDF.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CblGaussianComponent {
    /// Mixture weight in \[0, 1\].
    pub weight: f32,
    /// Component mean \[m/s\].
    pub mean_m_s: f32,
    /// Component standard deviation \[m/s\].
    pub sigma_m_s: f32,
}

/// Bi-Gaussian CBL PDF representation (updraft + downdraft modes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CblBiGaussianPdf {
    /// Updraft branch component.
    pub updraft: CblGaussianComponent,
    /// Downdraft branch component.
    pub downdraft: CblGaussianComponent,
    /// Target moments inferred from CBL closure.
    pub target_moments: CblPdfMoments,
    /// Transition factor from neutral to convective formulation \[0, 1\].
    pub transition: f32,
    /// Relative particle height z/h in \[0, 1\].
    pub z_over_h: f32,
}

/// CBL branch used for sign-constrained reinjection sampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CblBranch {
    Updraft,
    Downdraft,
}

#[inline]
fn time_direction_scale(time_direction: i32) -> f32 {
    if time_direction >= 0 {
        1.0
    } else {
        -1.0
    }
}

#[inline]
fn normal_pdf(value: f32, mean: f32, sigma: f32) -> f32 {
    let sigma = sigma.max(MIN_SIGMA_W_M_S);
    let standardized = (value - mean) / sigma;
    (0.398_942_3 / sigma) * (-0.5 * standardized * standardized).exp()
}

#[inline]
fn sanitize_inputs(inputs: CblPdfInputs) -> (f32, f32, f32, f32, f32) {
    let h_m = inputs.h_m.max(MIN_H_M);
    let z_m = inputs.z_m.clamp(0.0, h_m);
    let z_over_h = z_m / h_m;
    let wstar_m_s = inputs.wstar_m_s.max(0.0);
    let sigma_w_m_s = inputs.sigma_w_m_s.max(MIN_SIGMA_W_M_S);
    (z_m, h_m, z_over_h, wstar_m_s, sigma_w_m_s)
}

/// Stability transition from `cbl.f90` (Cassiani-style blend).
///
/// Returns a value in `[0, 1]` that damps CBL skewness in weakly convective
/// conditions.
#[must_use]
pub fn cbl_transition_factor(h_m: f32, obukhov_length_m: f32) -> f32 {
    if !h_m.is_finite() || h_m <= 0.0 || !obukhov_length_m.is_finite() || obukhov_length_m == 0.0 {
        return 1.0;
    }

    let ratio = -h_m / obukhov_length_m;
    let transition = if ratio < 15.0 {
        (((ratio + 10.0) / 10.0) * PI).sin() * 0.5 + 0.5
    } else {
        1.0
    };
    transition.clamp(0.0, 1.0)
}

/// Compute CBL target moments (`w2`, `w3`, skewness) for one particle location.
#[must_use]
pub fn compute_cbl_moments(inputs: CblPdfInputs) -> CblPdfMoments {
    let (_z_m, h_m, z_over_h, wstar_m_s, sigma_w_m_s) = sanitize_inputs(inputs);
    let transition = cbl_transition_factor(h_m, inputs.obukhov_length_m);

    let second_moment_w2 = sigma_w_m_s * sigma_w_m_s;
    let shape = 1.2 * z_over_h * (1.0 - z_over_h).powf(1.5) + W3_EPS;
    let third_moment_w3 = shape * wstar_m_s.powi(3) * transition;
    let skewness = third_moment_w3 / second_moment_w2.powf(1.5);

    CblPdfMoments {
        second_moment_w2,
        third_moment_w3,
        skewness,
    }
}

/// Build a bi-Gaussian PDF that matches CBL moments.
///
/// This is a robust version of the FLEXPART CBL parameterization:
/// near-zero skewness falls back to a symmetric Gaussian split.
#[allow(clippy::similar_names)]
#[must_use]
pub fn compute_cbl_bigaussian_pdf(inputs: CblPdfInputs) -> CblBiGaussianPdf {
    let (_z_m, h_m, z_over_h, _wstar_m_s, sigma_w_m_s) = sanitize_inputs(inputs);
    let target_moments = compute_cbl_moments(inputs);
    let skew = target_moments.skewness;
    let transition = cbl_transition_factor(h_m, inputs.obukhov_length_m);

    if !skew.is_finite() || skew.abs() <= SKEW_EPS {
        return CblBiGaussianPdf {
            updraft: CblGaussianComponent {
                weight: 0.5,
                mean_m_s: 0.0,
                sigma_m_s: sigma_w_m_s,
            },
            downdraft: CblGaussianComponent {
                weight: 0.5,
                mean_m_s: 0.0,
                sigma_m_s: sigma_w_m_s,
            },
            target_moments,
            transition,
            z_over_h,
        };
    }

    let fluarw = FLUARW_FACTOR * skew.cbrt();
    let fluarw2 = fluarw * fluarw;

    if !fluarw.is_finite() || fluarw.abs() <= SKEW_EPS {
        return CblBiGaussianPdf {
            updraft: CblGaussianComponent {
                weight: 0.5,
                mean_m_s: 0.0,
                sigma_m_s: sigma_w_m_s,
            },
            downdraft: CblGaussianComponent {
                weight: 0.5,
                mean_m_s: 0.0,
                sigma_m_s: sigma_w_m_s,
            },
            target_moments,
            transition,
            z_over_h,
        };
    }

    let skew2 = skew * skew;
    let numerator_r = (1.0 + fluarw2).powi(3) * skew2;
    let denominator_r = (3.0 + fluarw2).powi(2) * fluarw2;
    let rluarw = (numerator_r / denominator_r.max(SKEW_EPS)).max(0.0);
    let xluarw = rluarw.sqrt();

    let mut aluarw = 0.5 * (1.0 - xluarw / (4.0 + rluarw).sqrt());
    aluarw = aluarw.clamp(1.0e-4, 1.0 - 1.0e-4);
    let bluarw = 1.0 - aluarw;

    let sigmawa = sigma_w_m_s * (bluarw / (aluarw * (1.0 + fluarw2))).sqrt();
    let sigmawb = sigma_w_m_s * (aluarw / (bluarw * (1.0 + fluarw2))).sqrt();
    let wa = fluarw * sigmawa;
    let wb = fluarw * sigmawb;

    CblBiGaussianPdf {
        updraft: CblGaussianComponent {
            weight: aluarw,
            mean_m_s: wa,
            sigma_m_s: sigmawa.max(MIN_SIGMA_W_M_S),
        },
        downdraft: CblGaussianComponent {
            weight: bluarw,
            mean_m_s: -wb,
            sigma_m_s: sigmawb.max(MIN_SIGMA_W_M_S),
        },
        target_moments,
        transition,
        z_over_h,
    }
}

/// Reconstruct moments implied by a bi-Gaussian PDF.
#[must_use]
pub fn reconstruct_cbl_moments(pdf: &CblBiGaussianPdf) -> CblPdfMoments {
    let w_a = pdf.updraft.weight;
    let w_b = pdf.downdraft.weight;
    let mu_a = pdf.updraft.mean_m_s;
    let mu_b = pdf.downdraft.mean_m_s;
    let s2_a = pdf.updraft.sigma_m_s * pdf.updraft.sigma_m_s;
    let s2_b = pdf.downdraft.sigma_m_s * pdf.downdraft.sigma_m_s;

    let second_moment_w2 = w_a * (s2_a + mu_a * mu_a) + w_b * (s2_b + mu_b * mu_b);
    let third_moment_w3 =
        w_a * (mu_a.powi(3) + 3.0 * mu_a * s2_a) + w_b * (mu_b.powi(3) + 3.0 * mu_b * s2_b);
    let skewness = if second_moment_w2 > 0.0 {
        third_moment_w3 / second_moment_w2.powf(1.5)
    } else {
        0.0
    };

    CblPdfMoments {
        second_moment_w2,
        third_moment_w3,
        skewness,
    }
}

impl CblBiGaussianPdf {
    /// Probability density p(w') of the bi-Gaussian CBL model.
    #[must_use]
    pub fn probability_density(&self, w_prime_m_s: f32) -> f32 {
        self.updraft.weight * normal_pdf(w_prime_m_s, self.updraft.mean_m_s, self.updraft.sigma_m_s)
            + self.downdraft.weight
                * normal_pdf(
                    w_prime_m_s,
                    self.downdraft.mean_m_s,
                    self.downdraft.sigma_m_s,
                )
    }
}

/// Sample one vertical velocity perturbation from the CBL bi-Gaussian PDF.
///
/// Inputs are deterministic uniforms in `[0, 1)`.
#[must_use]
pub fn sample_cbl_vertical_velocity(
    pdf: &CblBiGaussianPdf,
    branch_uniform: f32,
    gaussian_uniform0: f32,
    gaussian_uniform1: f32,
) -> f32 {
    let (eta, _) = box_muller_normals(gaussian_uniform0, gaussian_uniform1);
    if branch_uniform < pdf.updraft.weight {
        pdf.updraft.mean_m_s + pdf.updraft.sigma_m_s * eta
    } else {
        pdf.downdraft.mean_m_s + pdf.downdraft.sigma_m_s * eta
    }
}

/// Sample one vertical velocity perturbation from Philox RNG.
#[must_use]
pub fn sample_cbl_vertical_velocity_with_rng(pdf: &CblBiGaussianPdf, rng: &mut PhiloxRng) -> f32 {
    let [u0, u1, u2, _u3] = rng.next_uniform4();
    sample_cbl_vertical_velocity(pdf, u0, u1, u2)
}

/// Infer CBL branch sign from previous velocity and time direction.
///
/// Matches the branching criterion in `re_initialize_particle.f90`.
#[must_use]
pub fn infer_cbl_branch(previous_w_prime_m_s: f32, time_direction: i32) -> CblBranch {
    let signed = previous_w_prime_m_s.signum() * time_direction_scale(time_direction);
    if signed >= 0.0 {
        CblBranch::Updraft
    } else {
        CblBranch::Downdraft
    }
}

/// Sign-constrained branch sample for CBL reinjection.
///
/// This utility is intended for turbulence-step fallback paths where a sampled
/// velocity must preserve updraft (`>0`) or downdraft (`<0`) branch semantics.
#[must_use]
pub fn sample_cbl_branch_constrained_velocity(
    pdf: &CblBiGaussianPdf,
    branch: CblBranch,
    gaussian_uniform0: f32,
    gaussian_uniform1: f32,
    time_direction: i32,
) -> f32 {
    let (eta, _) = box_muller_normals(gaussian_uniform0, gaussian_uniform1);
    let speed = match branch {
        CblBranch::Updraft => (pdf.updraft.mean_m_s + pdf.updraft.sigma_m_s * eta).abs(),
        CblBranch::Downdraft => {
            -(pdf.downdraft.mean_m_s.abs() + pdf.downdraft.sigma_m_s * eta.abs())
        }
    };
    speed * time_direction_scale(time_direction)
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    use super::*;

    #[test]
    fn cbl_bigaussian_reconstructs_target_moments() {
        let inputs = CblPdfInputs {
            z_m: 180.0,
            h_m: 1200.0,
            wstar_m_s: 2.2,
            sigma_w_m_s: 1.1,
            obukhov_length_m: -80.0,
        };

        let pdf = compute_cbl_bigaussian_pdf(inputs);
        let reconstructed = reconstruct_cbl_moments(&pdf);

        assert!(pdf.updraft.weight > 0.0 && pdf.updraft.weight < 1.0);
        assert!(pdf.downdraft.weight > 0.0 && pdf.downdraft.weight < 1.0);
        assert!(pdf.updraft.sigma_m_s > 0.0);
        assert!(pdf.downdraft.sigma_m_s > 0.0);

        assert_relative_eq!(
            reconstructed.second_moment_w2,
            pdf.target_moments.second_moment_w2,
            epsilon = 2.0e-4,
            max_relative = 2.0e-4
        );
        assert_relative_eq!(
            reconstructed.third_moment_w3,
            pdf.target_moments.third_moment_w3,
            epsilon = 5.0e-4,
            max_relative = 5.0e-4
        );
    }

    #[test]
    fn cbl_sampling_is_deterministic_and_matches_moments() {
        let inputs = CblPdfInputs {
            z_m: 200.0,
            h_m: 1400.0,
            wstar_m_s: 2.4,
            sigma_w_m_s: 1.2,
            obukhov_length_m: -90.0,
        };
        let pdf = compute_cbl_bigaussian_pdf(inputs);

        let key = [0xCAFEBABE, 0x12345678];
        let counter = [7, 11, 13, 17];
        let mut rng_a = PhiloxRng::new(key, counter);
        let mut rng_b = PhiloxRng::new(key, counter);

        let sample_count = 100_000usize;
        let mut sum_a = 0.0_f64;
        let mut sum_sq_a = 0.0_f64;
        let mut sum_cu_a = 0.0_f64;

        for _ in 0..sample_count {
            let w_a = sample_cbl_vertical_velocity_with_rng(&pdf, &mut rng_a);
            let w_b = sample_cbl_vertical_velocity_with_rng(&pdf, &mut rng_b);
            assert_eq!(w_a.to_bits(), w_b.to_bits());

            let w = f64::from(w_a);
            sum_a += w;
            sum_sq_a += w * w;
            sum_cu_a += w * w * w;
        }

        let inv_n = 1.0_f64 / sample_count as f64;
        let sample_mean = sum_a * inv_n;
        let sample_w2 = sum_sq_a * inv_n;
        let sample_w3 = sum_cu_a * inv_n;

        assert_abs_diff_eq!(sample_mean, 0.0, epsilon = 0.03);
        assert_relative_eq!(
            sample_w2 as f32,
            pdf.target_moments.second_moment_w2,
            epsilon = 0.05,
            max_relative = 0.05
        );
        assert_relative_eq!(
            sample_w3 as f32,
            pdf.target_moments.third_moment_w3,
            epsilon = 0.12,
            max_relative = 0.12
        );
    }

    #[test]
    fn cbl_neutral_or_zero_skew_falls_back_to_symmetric_pdf() {
        let inputs = CblPdfInputs {
            z_m: 100.0,
            h_m: 1000.0,
            wstar_m_s: 0.0,
            sigma_w_m_s: 0.9,
            obukhov_length_m: 1.0e6,
        };

        let pdf = compute_cbl_bigaussian_pdf(inputs);
        assert_abs_diff_eq!(pdf.updraft.weight, 0.5, epsilon = 1.0e-6);
        assert_abs_diff_eq!(pdf.downdraft.weight, 0.5, epsilon = 1.0e-6);
        assert_abs_diff_eq!(pdf.updraft.mean_m_s, 0.0, epsilon = 1.0e-6);
        assert_abs_diff_eq!(pdf.downdraft.mean_m_s, 0.0, epsilon = 1.0e-6);
        assert_abs_diff_eq!(pdf.updraft.sigma_m_s, 0.9, epsilon = 1.0e-6);
        assert_abs_diff_eq!(pdf.downdraft.sigma_m_s, 0.9, epsilon = 1.0e-6);
    }

    #[test]
    fn cbl_branch_constrained_sampling_preserves_sign() {
        let inputs = CblPdfInputs {
            z_m: 300.0,
            h_m: 1200.0,
            wstar_m_s: 1.8,
            sigma_w_m_s: 1.0,
            obukhov_length_m: -120.0,
        };
        let pdf = compute_cbl_bigaussian_pdf(inputs);

        let up = sample_cbl_branch_constrained_velocity(&pdf, CblBranch::Updraft, 0.2, 0.4, 1);
        let down = sample_cbl_branch_constrained_velocity(&pdf, CblBranch::Downdraft, 0.2, 0.4, 1);
        let up_backward =
            sample_cbl_branch_constrained_velocity(&pdf, CblBranch::Updraft, 0.2, 0.4, -1);

        assert!(up >= 0.0);
        assert!(down <= 0.0);
        assert!(up_backward <= 0.0);
    }

    #[test]
    fn cbl_pdf_probability_is_positive() {
        let pdf = compute_cbl_bigaussian_pdf(CblPdfInputs {
            z_m: 150.0,
            h_m: 1000.0,
            wstar_m_s: 2.0,
            sigma_w_m_s: 1.0,
            obukhov_length_m: -70.0,
        });

        for w in [-5.0_f32, -2.0, -1.0, 0.0, 1.0, 3.0, 5.0] {
            assert!(pdf.probability_density(w) >= 0.0);
        }
    }
}
