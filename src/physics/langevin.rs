//! CPU reference Langevin turbulence integration (H-03).
//!
//! Ported from FLEXPART `advance.f90` around lines 300-500 (stochastic
//! turbulent velocity update for `up`, `vp`, `wp`, without CBL-specific branch).
//! This module updates per-particle turbulent velocity memory
//! (`Particle::turb_u`, `Particle::turb_v`, `Particle::turb_w`) from:
//! - Hanna turbulence parameters (`HannaParams`)
//! - integration step settings (`LangevinStep`)
//! - standard-normal random forcing (`LangevinNoise`)
//!
//! The update follows FLEXPART's split logic:
//! - small-step approximation for `dt / TL < 0.5`
//! - exact Ornstein-Uhlenbeck form otherwise
//!
//! The RNG source is deterministic Philox (`physics::rng`, task A-05).

use std::f32::consts::PI;

use thiserror::Error;

use crate::particles::Particle;
use crate::pbl::HannaParams;
use crate::physics::PhiloxRng;

const BOX_MULLER_U_MIN: f32 = f32::MIN_POSITIVE;
const BOX_MULLER_U_MAX: f32 = 1.0 - f32::EPSILON;

/// Integration settings for one Langevin turbulence update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LangevinStep {
    /// Time step `dt` in seconds.
    pub dt_seconds: f32,
    /// Local density-gradient correction term `(1/rho) * d(rho)/dz` [1/m].
    ///
    /// In FLEXPART this corresponds to `rhoaux = rhograd / rhoa`.
    pub rho_grad_over_rho: f32,
    /// Number of vertical turbulence sub-steps per timestep (Fortran `ifine`).
    ///
    /// - `0`: legacy mode — update turb_u/v/w only, no vertical displacement
    ///   or PBL reflection (backward-compatible with old pipeline).
    /// - `1..=4`: sub-step the vertical Langevin update with
    ///   `dt_sub = dt / n_substeps`, applying displacement and PBL reflection
    ///   between each sub-step. Horizontal turbulence uses full `dt`.
    ///
    /// Fortran default is `ifine = 4`.
    pub n_substeps: u32,
    /// Minimum height [m] for PBL ground reflection (typically 0.01 m).
    /// Only used when `n_substeps >= 1`.
    pub min_height_m: f32,
}

impl LangevinStep {
    /// Create a step with legacy behavior (no vertical sub-stepping).
    #[must_use]
    pub fn legacy(dt_seconds: f32, rho_grad_over_rho: f32) -> Self {
        Self {
            dt_seconds,
            rho_grad_over_rho,
            n_substeps: 0,
            min_height_m: 0.01,
        }
    }
}

/// Standard-normal random forcing (`N(0, 1)`) for one particle update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LangevinNoise {
    /// Forcing for turbulent u component.
    pub eta_u: f32,
    /// Forcing for turbulent v component.
    pub eta_v: f32,
    /// Forcing for turbulent w component.
    pub eta_w: f32,
}

/// Turbulent velocity fluctuation after one update [m/s].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurbulentVelocity {
    pub u: f32,
    pub v: f32,
    pub w: f32,
}

/// Errors for CPU Langevin turbulence updates.
#[derive(Debug, Error)]
pub enum LangevinError {
    #[error("invalid dt_seconds for Langevin update: {dt_seconds}")]
    InvalidTimeStep { dt_seconds: f32 },
    #[error("mismatched input lengths: particles={particles}, params={params}, noise={noise}")]
    MismatchedInputLengths {
        particles: usize,
        params: usize,
        noise: usize,
    },
}

fn validate_step(step: LangevinStep) -> Result<(), LangevinError> {
    if !step.dt_seconds.is_finite() || step.dt_seconds <= 0.0 {
        return Err(LangevinError::InvalidTimeStep {
            dt_seconds: step.dt_seconds,
        });
    }
    Ok(())
}

#[inline]
fn sanitize_positive(value: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

#[inline]
fn update_horizontal_component(previous: f32, sigma: f32, tl: f32, dt: f32, eta: f32) -> f32 {
    let sigma = sanitize_positive(sigma);
    let tl = sanitize_positive(tl);
    if tl == 0.0 {
        return 0.0;
    }

    let ratio = dt / tl;
    let next = if ratio < 0.5 {
        (1.0 - ratio) * previous + eta * sigma * (2.0 * ratio).max(0.0).sqrt()
    } else {
        let corr = (-ratio).exp();
        corr * previous + eta * sigma * (1.0 - corr * corr).max(0.0).sqrt()
    };

    if next.is_finite() {
        next
    } else {
        0.0
    }
}

#[inline]
fn update_vertical_component(
    previous: f32,
    params: &HannaParams,
    dt: f32,
    rho_grad_over_rho: f32,
    eta: f32,
) -> f32 {
    let sigma_w = sanitize_positive(params.sigw);
    let tlw = sanitize_positive(params.tlw);
    if tlw == 0.0 {
        return 0.0;
    }

    // FLEXPART drift term in the "turbswitch" branch:
    // sigw * (d(sigw)/dz + (1/rho) d(rho)/dz * sigw)
    let drift = sigma_w * (params.dsigwdz + rho_grad_over_rho * sigma_w);
    let ratio = dt / tlw;
    let next = if ratio < 0.5 {
        (1.0 - ratio) * previous + eta * sigma_w * (2.0 * ratio).max(0.0).sqrt() + dt * drift
    } else {
        let corr = (-ratio).exp();
        corr * previous
            + eta * sigma_w * (1.0 - corr * corr).max(0.0).sqrt()
            + tlw * (1.0 - corr) * drift
    };

    if next.is_finite() {
        next
    } else {
        0.0
    }
}

/// Convert two uniforms in `[0, 1)` to two standard normals using Box-Muller.
///
/// The clamping keeps behavior stable for edge values near 0.
#[must_use]
pub fn box_muller_normals(u0: f32, u1: f32) -> (f32, f32) {
    let radius_uniform = u0.clamp(BOX_MULLER_U_MIN, BOX_MULLER_U_MAX);
    let angle_uniform = u1.clamp(0.0, BOX_MULLER_U_MAX);
    let radius = (-2.0 * radius_uniform.ln()).sqrt();
    let theta = 2.0 * PI * angle_uniform;
    (radius * theta.cos(), radius * theta.sin())
}

/// Sample one deterministic `LangevinNoise` tuple from Philox uniforms.
///
/// This consumes one Philox block (`next_uniform4`) and transforms it into
/// three standard-normal values.
#[must_use]
pub fn sample_langevin_noise(rng: &mut PhiloxRng) -> LangevinNoise {
    let [u0, u1, u2, u3] = rng.next_uniform4();
    let (eta_u, eta_v) = box_muller_normals(u0, u1);
    let (eta_w, _) = box_muller_normals(u2, u3);
    LangevinNoise {
        eta_u,
        eta_v,
        eta_w,
    }
}

/// Update one particle turbulent velocity (`turb_u/v/w`) with explicit noise.
///
/// Returns the updated turbulent velocity.
///
/// # Errors
///
/// Returns [`LangevinError::InvalidTimeStep`] when `dt_seconds` is not finite
/// or not strictly positive.
pub fn update_particle_turbulence_langevin_cpu(
    particle: &mut Particle,
    params: &HannaParams,
    step: LangevinStep,
    noise: LangevinNoise,
) -> Result<TurbulentVelocity, LangevinError> {
    validate_step(step)?;

    let next_u = update_horizontal_component(
        particle.turb_u,
        params.sigu,
        params.tlu,
        step.dt_seconds,
        noise.eta_u,
    );
    let next_v = update_horizontal_component(
        particle.turb_v,
        params.sigv,
        params.tlv,
        step.dt_seconds,
        noise.eta_v,
    );
    let next_w = update_vertical_component(
        particle.turb_w,
        params,
        step.dt_seconds,
        step.rho_grad_over_rho,
        noise.eta_w,
    );

    particle.turb_u = next_u;
    particle.turb_v = next_v;
    particle.turb_w = next_w;

    Ok(TurbulentVelocity {
        u: next_u,
        v: next_v,
        w: next_w,
    })
}

/// Update one particle turbulent velocity using deterministic Philox noise.
///
/// Returns the updated turbulent velocity.
///
/// # Errors
///
/// Returns [`LangevinError::InvalidTimeStep`] when `dt_seconds` is not finite
/// or not strictly positive.
pub fn update_particle_turbulence_langevin_with_rng_cpu(
    particle: &mut Particle,
    params: &HannaParams,
    step: LangevinStep,
    rng: &mut PhiloxRng,
) -> Result<TurbulentVelocity, LangevinError> {
    let noise = sample_langevin_noise(rng);
    update_particle_turbulence_langevin_cpu(particle, params, step, noise)
}

/// Batch update with explicit per-particle noise.
///
/// Inactive particles are skipped, but a noise entry is still expected for
/// every slot to keep index-stable deterministic mapping.
///
/// # Errors
///
/// Returns:
/// - [`LangevinError::InvalidTimeStep`] when `dt_seconds` is invalid
/// - [`LangevinError::MismatchedInputLengths`] when slice lengths differ
pub fn update_particles_turbulence_langevin_cpu(
    particles: &mut [Particle],
    params: &[HannaParams],
    step: LangevinStep,
    noise: &[LangevinNoise],
) -> Result<(), LangevinError> {
    validate_step(step)?;
    if particles.len() != params.len() || particles.len() != noise.len() {
        return Err(LangevinError::MismatchedInputLengths {
            particles: particles.len(),
            params: params.len(),
            noise: noise.len(),
        });
    }

    for ((particle, hanna), forcing) in particles.iter_mut().zip(params).zip(noise) {
        if particle.is_active() {
            let _ = update_particle_turbulence_langevin_cpu(particle, hanna, step, *forcing)?;
        }
    }
    Ok(())
}

/// Batch update using deterministic Philox noise generation.
///
/// One `LangevinNoise` tuple is consumed per particle slot.
///
/// # Errors
///
/// Returns:
/// - [`LangevinError::InvalidTimeStep`] when `dt_seconds` is invalid
/// - [`LangevinError::MismatchedInputLengths`] when `particles` and `params`
///   lengths differ
pub fn update_particles_turbulence_langevin_with_rng_cpu(
    particles: &mut [Particle],
    params: &[HannaParams],
    step: LangevinStep,
    rng: &mut PhiloxRng,
) -> Result<(), LangevinError> {
    validate_step(step)?;
    if particles.len() != params.len() {
        return Err(LangevinError::MismatchedInputLengths {
            particles: particles.len(),
            params: params.len(),
            noise: particles.len(),
        });
    }

    for (particle, hanna) in particles.iter_mut().zip(params) {
        let forcing = sample_langevin_noise(rng);
        if particle.is_active() {
            let _ = update_particle_turbulence_langevin_cpu(particle, hanna, step, forcing)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use bytemuck::Zeroable;

    use super::*;
    use crate::particles::{ParticleInit, MAX_SPECIES};
    use crate::physics::compute_hanna_params;
    use crate::physics::HannaInputs;

    fn particle_with_turbulence(turb_u: f32, turb_v: f32, turb_w: f32) -> Particle {
        let mut particle = Particle::new(&ParticleInit {
            cell_x: 0,
            cell_y: 0,
            pos_x: 0.25,
            pos_y: 0.75,
            pos_z: 100.0,
            mass: [0.0; MAX_SPECIES],
            release_point: 0,
            class: 0,
            time: 0,
        });
        particle.turb_u = turb_u;
        particle.turb_v = turb_v;
        particle.turb_w = turb_w;
        particle
    }

    #[test]
    fn zero_turbulence_yields_stable_deterministic_behavior() {
        let mut particle = particle_with_turbulence(2.0, -1.0, 0.5);
        let mut params = HannaParams::zeroed();
        params.tlu = 50.0;
        params.tlv = 50.0;
        params.tlw = 50.0;

        let step = LangevinStep::legacy(10.0, 0.0);
        let noise = LangevinNoise {
            eta_u: 0.0,
            eta_v: 0.0,
            eta_w: 0.0,
        };
        let updated = update_particle_turbulence_langevin_cpu(&mut particle, &params, step, noise)
            .expect("step should be valid");

        assert!((updated.u - 1.6).abs() < 1.0e-6);
        assert!((updated.v + 0.8).abs() < 1.0e-6);
        assert!((updated.w - 0.4).abs() < 1.0e-6);
    }

    #[test]
    fn finite_dt_produces_bounded_updates() {
        let mut particle = particle_with_turbulence(12.0, -9.0, 6.0);
        let params = compute_hanna_params(HannaInputs {
            ust: 0.3,
            wst: 1.2,
            ol: -100.0,
            h: 900.0,
            z: 120.0,
        });
        let step = LangevinStep::legacy(180.0, 5.0e-4);
        let noise = LangevinNoise {
            eta_u: 0.5,
            eta_v: -0.3,
            eta_w: 0.2,
        };

        let updated = update_particle_turbulence_langevin_cpu(&mut particle, &params, step, noise)
            .expect("step should be valid");

        assert!(updated.u.is_finite());
        assert!(updated.v.is_finite());
        assert!(updated.w.is_finite());
        assert!(updated.u.abs() < 100.0);
        assert!(updated.v.abs() < 100.0);
        assert!(updated.w.abs() < 100.0);
    }

    #[test]
    fn reproducible_with_fixed_philox_seed_and_counter() {
        let mut particles_a = vec![
            particle_with_turbulence(0.5, -0.25, 0.75),
            particle_with_turbulence(-1.0, 0.3, -0.1),
        ];
        let mut particles_b = particles_a.clone();
        let params = vec![
            compute_hanna_params(HannaInputs {
                ust: 0.25,
                wst: 1.4,
                ol: -80.0,
                h: 800.0,
                z: 80.0,
            }),
            compute_hanna_params(HannaInputs {
                ust: 0.35,
                wst: 0.0,
                ol: 120.0,
                h: 700.0,
                z: 50.0,
            }),
        ];
        let step = LangevinStep::legacy(30.0, 2.0e-4);

        let key = [0xDECA_FBAD, 0x1234_5678];
        let counter = [0xA5A5_0001, 0xFACE_B00C, 7, 9];
        let mut rng_a = PhiloxRng::new(key, counter);
        let mut rng_b = PhiloxRng::new(key, counter);

        update_particles_turbulence_langevin_with_rng_cpu(
            &mut particles_a,
            &params,
            step,
            &mut rng_a,
        )
        .expect("update succeeds");
        update_particles_turbulence_langevin_with_rng_cpu(
            &mut particles_b,
            &params,
            step,
            &mut rng_b,
        )
        .expect("update succeeds");

        for (lhs, rhs) in particles_a.iter().zip(particles_b.iter()) {
            assert_eq!(lhs.turb_u.to_bits(), rhs.turb_u.to_bits());
            assert_eq!(lhs.turb_v.to_bits(), rhs.turb_v.to_bits());
            assert_eq!(lhs.turb_w.to_bits(), rhs.turb_w.to_bits());
        }
    }
}
