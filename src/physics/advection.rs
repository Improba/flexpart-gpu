//! Deterministic CPU advection reference (Euler + Petterssen correction).
//!
//! Ported from the deterministic advection part of `../flexpart/src/advance.f90`,
//! especially the predictor/corrector flow around lines 817-923.
//!
//! This implementation currently operates in **grid-coordinate space**:
//! - particle horizontal coordinates are fractional grid indices;
//! - `pos_z` is treated as vertical grid index for interpolation;
//! - interpolated wind is converted to coordinate displacement using an explicit
//!   scale factor (`VelocityToGridScale`).
//!
//! If metric conversion factors are not yet available, use
//! [`VelocityToGridScale::IDENTITY`] so wind is interpreted as grid-units/s.

use crate::particles::Particle;
use crate::physics::interpolation::{interpolate_wind_trilinear, WindVector};
use crate::wind::WindField3D;

/// Maximum number of vertical levels carried in the advection uniform.
pub const MAX_VERTICAL_LEVELS: usize = 16;

/// Converts wind velocity components to coordinate-space speed.
///
/// The interpolator returns wind components from the met field (typically m/s).
/// This scale maps those to advection-coordinate units/s:
///
/// - `x_grid_per_meter`: multiply `u` by this to get `dx/dt` in grid-x units/s
/// - `y_grid_per_meter`: multiply `v` by this to get `dy/dt` in grid-y units/s
/// - `z_grid_per_meter`: multiply `w` by this to get `dz/dt` in vertical units/s
///   (1.0 when `pos_z` is in meters and `w` is m/s)
/// - `level_heights_m`: heights of each wind-grid level [m].  When non-zero the
///   shader converts `pos_z` (meters) → fractional grid level for trilinear
///   interpolation and clamps `pos_z` to `[0, max_height]`.  When all zeros the
///   shader falls back to treating `pos_z` as a grid-level index (legacy mode).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityToGridScale {
    pub x_grid_per_meter: f32,
    pub y_grid_per_meter: f32,
    pub z_grid_per_meter: f32,
    pub level_heights_m: [f32; MAX_VERTICAL_LEVELS],
}

impl VelocityToGridScale {
    /// Identity conversion: treat wind as already in grid-units/s (legacy mode).
    pub const IDENTITY: Self = Self {
        x_grid_per_meter: 1.0,
        y_grid_per_meter: 1.0,
        z_grid_per_meter: 1.0,
        level_heights_m: [0.0; MAX_VERTICAL_LEVELS],
    };
}

impl Default for VelocityToGridScale {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Advect one particle with a pure Euler step (predictor only).
///
/// Returns the wind used for this step (`u`, `v`, `w` in field units).
#[must_use]
pub fn advect_particle_cpu_euler(
    particle: &mut Particle,
    field: &WindField3D,
    dt_seconds: f32,
    velocity_scale: VelocityToGridScale,
) -> WindVector {
    let (shape_x, shape_y, shape_z) = field.shape();
    assert!(
        shape_x > 0 && shape_y > 0 && shape_z > 0,
        "wind field dimensions must be non-zero"
    );

    let (x0, y0, z0) = particle_coords(particle);
    let wind_at_start = interpolate_wind_trilinear(field, x0, y0, z0);
    let (sx, sy, sz) = scale_velocity(wind_at_start, velocity_scale);

    let x1 = clamp_axis(x0 + sx * dt_seconds, shape_x);
    let y1 = clamp_axis(y0 + sy * dt_seconds, shape_y);
    let z1 = clamp_axis(z0 + sz * dt_seconds, shape_z);

    set_particle_coords(particle, x1, y1, z1, shape_x, shape_y);
    particle.vel_u = wind_at_start.u;
    particle.vel_v = wind_at_start.v;
    particle.vel_w = wind_at_start.w;
    wind_at_start
}

/// Advect one particle with Euler predictor + Petterssen correction.
///
/// Algorithm:
/// 1. interpolate wind at current point (`v0`)
/// 2. Euler-predict new coordinate
/// 3. interpolate wind at predicted point (`v1`)
/// 4. apply correction with mean velocity `0.5 * (v0 + v1)`
///
/// Returns the corrected (mean) wind used for displacement.
#[must_use]
pub fn advect_particle_cpu(
    particle: &mut Particle,
    field: &WindField3D,
    dt_seconds: f32,
    velocity_scale: VelocityToGridScale,
) -> WindVector {
    let (shape_x, shape_y, shape_z) = field.shape();
    assert!(
        shape_x > 0 && shape_y > 0 && shape_z > 0,
        "wind field dimensions must be non-zero"
    );

    let (x0, y0, z0) = particle_coords(particle);
    let wind_start = interpolate_wind_trilinear(field, x0, y0, z0);
    let (sx0, sy0, sz0) = scale_velocity(wind_start, velocity_scale);

    let predicted_x = clamp_axis(x0 + sx0 * dt_seconds, shape_x);
    let predicted_y = clamp_axis(y0 + sy0 * dt_seconds, shape_y);
    let predicted_z = clamp_axis(z0 + sz0 * dt_seconds, shape_z);

    let wind_predicted = interpolate_wind_trilinear(field, predicted_x, predicted_y, predicted_z);
    let corrected = WindVector {
        u: 0.5 * (wind_start.u + wind_predicted.u),
        v: 0.5 * (wind_start.v + wind_predicted.v),
        w: 0.5 * (wind_start.w + wind_predicted.w),
    };
    let (sx, sy, sz) = scale_velocity(corrected, velocity_scale);

    let x1 = clamp_axis(x0 + sx * dt_seconds, shape_x);
    let y1 = clamp_axis(y0 + sy * dt_seconds, shape_y);
    let z1 = clamp_axis(z0 + sz * dt_seconds, shape_z);

    set_particle_coords(particle, x1, y1, z1, shape_x, shape_y);
    particle.vel_u = corrected.u;
    particle.vel_v = corrected.v;
    particle.vel_w = corrected.w;
    corrected
}

/// Batch advection using Euler predictor + Petterssen correction.
///
/// Inactive particles are skipped.
pub fn advect_particles_cpu(
    particles: &mut [Particle],
    field: &WindField3D,
    dt_seconds: f32,
    velocity_scale: VelocityToGridScale,
) {
    for particle in particles {
        if particle.is_active() {
            let _ = advect_particle_cpu(particle, field, dt_seconds, velocity_scale);
        }
    }
}

#[inline]
fn scale_velocity(wind: WindVector, scale: VelocityToGridScale) -> (f32, f32, f32) {
    (
        wind.u * scale.x_grid_per_meter,
        wind.v * scale.y_grid_per_meter,
        wind.w * scale.z_grid_per_meter,
    )
}

#[inline]
fn particle_coords(particle: &Particle) -> (f32, f32, f32) {
    (
        particle.cell_x as f32 + particle.pos_x,
        particle.cell_y as f32 + particle.pos_y,
        particle.pos_z,
    )
}

#[allow(clippy::cast_precision_loss)]
#[inline]
fn clamp_axis(coord: f32, axis_len: usize) -> f32 {
    coord.clamp(0.0, (axis_len - 1) as f32)
}

fn set_particle_coords(particle: &mut Particle, x: f32, y: f32, z: f32, nx: usize, ny: usize) {
    let (cell_x, pos_x) = split_horizontal_coord(x, nx);
    let (cell_y, pos_y) = split_horizontal_coord(y, ny);
    particle.cell_x = cell_x;
    particle.cell_y = cell_y;
    particle.pos_x = pos_x;
    particle.pos_y = pos_y;
    particle.pos_z = z;
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn split_horizontal_coord(coord: f32, axis_len: usize) -> (i32, f32) {
    let clamped = clamp_axis(coord, axis_len);
    let mut cell = clamped.floor() as usize;
    let mut frac = clamped - (cell as f32);
    if cell >= axis_len - 1 {
        cell = axis_len - 1;
        frac = 0.0;
    }
    (cell as i32, frac)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particles::{ParticleInit, MAX_SPECIES};
    use crate::wind::{linear_shear_wind_field, uniform_wind_field, WindFieldGrid};
    use ndarray::Array1;

    fn particle_at(x: f32, y: f32, z: f32) -> Particle {
        let cell_x = x.floor() as i32;
        let cell_y = y.floor() as i32;
        Particle::new(&ParticleInit {
            cell_x,
            cell_y,
            pos_x: x - cell_x as f32,
            pos_y: y - cell_y as f32,
            pos_z: z,
            mass: [0.0; MAX_SPECIES],
            release_point: 0,
            class: 0,
            time: 0,
        })
    }

    fn uniform_field(nx: usize, ny: usize, nz: usize, u: f32, v: f32, w: f32) -> WindField3D {
        let grid = WindFieldGrid::new(
            nx,
            ny,
            nz,
            nz,
            nz,
            1.0,
            1.0,
            0.0,
            0.0,
            Array1::from_vec((0..nz).map(|k| k as f32).collect()),
        );
        uniform_wind_field(&grid, u, v, w)
    }

    fn linear_x_shear_field(nx: usize, ny: usize, nz: usize, a: f32) -> WindField3D {
        let mut field = WindField3D::zeros(nx, ny, nz);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = i as f32;
                    field.u_ms[[i, j, k]] = a * x;
                    field.v_ms[[i, j, k]] = 0.0;
                    field.w_ms[[i, j, k]] = 0.0;
                }
            }
        }
        field
    }

    fn synthetic_grid(nx: usize, ny: usize, nz: usize) -> WindFieldGrid {
        let heights_m = Array1::from_iter((0..nz).map(|k| k as f32));
        WindFieldGrid::new(nx, ny, nz, nz, nz, 1.0, 1.0, 0.0, 0.0, heights_m)
    }

    fn analytical_linear_shear_trajectory(
        x0: f32,
        y0: f32,
        z0: f32,
        total_time_s: f32,
        u_reference_ms: f32,
        v_reference_ms: f32,
        w_constant_ms: f32,
        du_dz_per_s: f32,
        dv_dz_per_s: f32,
        reference_height_m: f32,
    ) -> (f32, f32, f32) {
        let z_expected = z0 + w_constant_ms * total_time_s;
        let u_at_t0 = u_reference_ms + du_dz_per_s * (z0 - reference_height_m);
        let v_at_t0 = v_reference_ms + dv_dz_per_s * (z0 - reference_height_m);
        let x_expected =
            x0 + u_at_t0 * total_time_s + 0.5 * du_dz_per_s * w_constant_ms * total_time_s.powi(2);
        let y_expected =
            y0 + v_at_t0 * total_time_s + 0.5 * dv_dz_per_s * w_constant_ms * total_time_s.powi(2);
        (x_expected, y_expected, z_expected)
    }

    #[test]
    fn advection_uniform_wind_matches_analytical_solution_for_deterministic_particles() {
        // A-07 assumption: advection runs in grid-coordinate mode.
        // With IDENTITY scaling, u/v/w are interpreted directly as grid-units/s.
        let dt = 2.75_f32;
        let u = 0.8_f32;
        let v = -0.35_f32;
        let w = 0.15_f32;
        let tolerance = 1.0e-5_f64;
        let field = uniform_field(64, 64, 64, u, v, w);

        let initial_positions = [
            (5.25_f32, 8.5_f32, 6.25_f32),
            (11.0_f32, 13.125_f32, 9.75_f32),
            (23.9_f32, 17.4_f32, 12.1_f32),
            (31.375_f32, 4.2_f32, 3.0_f32),
        ];

        for (x0, y0, z0) in initial_positions {
            let mut particle = particle_at(x0, y0, z0);
            let advected_wind =
                advect_particle_cpu(&mut particle, &field, dt, VelocityToGridScale::IDENTITY);

            let x_expected = f64::from(x0 + u * dt);
            let y_expected = f64::from(y0 + v * dt);
            let z_expected = f64::from(z0 + w * dt);

            assert!(
                (particle.grid_x() - x_expected).abs() <= tolerance,
                "x mismatch for x0={x0}: got {}, expected {x_expected}",
                particle.grid_x()
            );
            assert!(
                (particle.grid_y() - y_expected).abs() <= tolerance,
                "y mismatch for y0={y0}: got {}, expected {y_expected}",
                particle.grid_y()
            );
            assert!(
                (f64::from(particle.pos_z) - z_expected).abs() <= tolerance,
                "z mismatch for z0={z0}: got {}, expected {z_expected}",
                particle.pos_z
            );
            assert!((advected_wind.u - u).abs() <= f32::EPSILON);
            assert!((advected_wind.v - v).abs() <= f32::EPSILON);
            assert!((advected_wind.w - w).abs() <= f32::EPSILON);
            assert!((particle.vel_u - u).abs() <= f32::EPSILON);
            assert!((particle.vel_v - v).abs() <= f32::EPSILON);
            assert!((particle.vel_w - w).abs() <= f32::EPSILON);
        }
    }

    #[test]
    fn petterssen_linear_shear_is_more_accurate_than_euler() {
        let a = 0.2_f32;
        let dt = 1.0_f32;
        let x0 = 2.5_f32;
        let field = linear_x_shear_field(64, 4, 4, a);

        let mut euler_particle = particle_at(x0, 1.0, 1.0);
        let mut petterssen_particle = particle_at(x0, 1.0, 1.0);

        let _ = advect_particle_cpu_euler(
            &mut euler_particle,
            &field,
            dt,
            VelocityToGridScale::IDENTITY,
        );
        let _ = advect_particle_cpu(
            &mut petterssen_particle,
            &field,
            dt,
            VelocityToGridScale::IDENTITY,
        );

        let exact = x0 * (a * dt).exp();
        let euler_err = ((euler_particle.grid_x() as f32) - exact).abs();
        let petterssen_err = ((petterssen_particle.grid_x() as f32) - exact).abs();

        assert!(
            petterssen_err < euler_err,
            "Petterssen should improve accuracy: Euler err={euler_err}, Petterssen err={petterssen_err}"
        );
    }

    #[test]
    fn advection_clamps_out_of_domain_position_stably() {
        let field = uniform_field(3, 3, 3, 10.0, -10.0, 10.0);
        let mut particle = particle_at(1.8, 0.2, 1.0);

        let _ = advect_particle_cpu(&mut particle, &field, 1.0, VelocityToGridScale::IDENTITY);

        assert_eq!(particle.cell_x, 2);
        assert_eq!(particle.pos_x, 0.0);
        assert_eq!(particle.cell_y, 0);
        assert_eq!(particle.pos_y, 0.0);
        assert_eq!(particle.pos_z, 2.0);
        assert!((0.0..1.0).contains(&particle.pos_x) || particle.pos_x == 0.0);
        assert!((0.0..1.0).contains(&particle.pos_y) || particle.pos_y == 0.0);
    }

    #[test]
    fn petterssen_linear_shear_matches_analytical_single_step() {
        let grid = synthetic_grid(96, 96, 96);
        let u_reference_ms = 1.2_f32;
        let v_reference_ms = -0.35_f32;
        let w_constant_ms = 0.2_f32;
        let du_dz_per_s = 0.04_f32;
        let dv_dz_per_s = -0.015_f32;
        let reference_height_m = 0.0_f32;
        let field = linear_shear_wind_field(
            &grid,
            u_reference_ms,
            v_reference_ms,
            w_constant_ms,
            du_dz_per_s,
            dv_dz_per_s,
            0.0,
            reference_height_m,
        );

        let x0 = 10.25_f32;
        let y0 = 11.75_f32;
        let z0 = 5.4_f32;
        let dt_seconds = 0.6_f32;
        let mut particle = particle_at(x0, y0, z0);

        let _ = advect_particle_cpu(
            &mut particle,
            &field,
            dt_seconds,
            VelocityToGridScale::IDENTITY,
        );

        let (x_expected, y_expected, z_expected) = analytical_linear_shear_trajectory(
            x0,
            y0,
            z0,
            dt_seconds,
            u_reference_ms,
            v_reference_ms,
            w_constant_ms,
            du_dz_per_s,
            dv_dz_per_s,
            reference_height_m,
        );

        let abs_tol = 1.0e-5_f32;
        assert!(((particle.grid_x() as f32) - x_expected).abs() <= abs_tol);
        assert!(((particle.grid_y() as f32) - y_expected).abs() <= abs_tol);
        assert!((particle.pos_z - z_expected).abs() <= abs_tol);
    }

    #[test]
    fn petterssen_linear_shear_matches_analytical_multi_step() {
        let grid = synthetic_grid(128, 128, 128);
        let u_reference_ms = 0.8_f32;
        let v_reference_ms = 0.5_f32;
        let w_constant_ms = 0.3_f32;
        let du_dz_per_s = 0.03_f32;
        let dv_dz_per_s = -0.02_f32;
        let reference_height_m = 0.0_f32;
        let field = linear_shear_wind_field(
            &grid,
            u_reference_ms,
            v_reference_ms,
            w_constant_ms,
            du_dz_per_s,
            dv_dz_per_s,
            0.0,
            reference_height_m,
        );

        let x0 = 8.125_f32;
        let y0 = 9.625_f32;
        let z0 = 6.5_f32;
        let dt_seconds = 0.25_f32;
        let n_steps = 20_u32;
        let mut particle = particle_at(x0, y0, z0);
        for _ in 0..n_steps {
            let _ = advect_particle_cpu(
                &mut particle,
                &field,
                dt_seconds,
                VelocityToGridScale::IDENTITY,
            );
        }

        let total_time_s = dt_seconds * (n_steps as f32);
        let (x_expected, y_expected, z_expected) = analytical_linear_shear_trajectory(
            x0,
            y0,
            z0,
            total_time_s,
            u_reference_ms,
            v_reference_ms,
            w_constant_ms,
            du_dz_per_s,
            dv_dz_per_s,
            reference_height_m,
        );

        let abs_tol = 2.0e-4_f32;
        assert!(((particle.grid_x() as f32) - x_expected).abs() <= abs_tol);
        assert!(((particle.grid_y() as f32) - y_expected).abs() <= abs_tol);
        assert!((particle.pos_z - z_expected).abs() <= abs_tol);
    }
}
