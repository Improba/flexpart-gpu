//! CPU reference wind interpolation utilities.
//!
//! Ported from the spatial interpolation stages in
//! `../flexpart/src/interpol_wind.f90` (bilinear horizontal + linear vertical,
//! equivalent to trilinear interpolation for one time level), especially lines
//! 49-94 and 172-179.
//!
//! This module is intentionally deterministic and CPU-only so it can serve as
//! the ground truth for future GPU parity tests.

use crate::wind::WindField3D;
use ndarray::Array3;

/// Interpolated wind vector at a single fractional grid coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WindVector {
    /// East-west wind component [m/s].
    pub u: f32,
    /// North-south wind component [m/s].
    pub v: f32,
    /// Vertical wind component [m/s].
    pub w: f32,
}

/// Trilinearly interpolate `u`, `v`, and `w` from a [`WindField3D`].
///
/// Coordinates `x`, `y`, and `z` are fractional grid indices in array space.
/// This reference implementation clamps each coordinate to the valid domain
/// `[0, n - 1]`, so out-of-bounds requests are mapped to the nearest boundary.
///
/// Fortran reference:
/// `../flexpart/src/interpol_wind.f90` (horizontal interpolation and vertical
/// interpolation in one time level).
#[must_use]
pub fn interpolate_wind_trilinear(field: &WindField3D, x: f32, y: f32, z: f32) -> WindVector {
    WindVector {
        u: trilinear_interpolate_clamped(&field.u_ms, x, y, z),
        v: trilinear_interpolate_clamped(&field.v_ms, x, y, z),
        w: trilinear_interpolate_clamped(&field.w_ms, x, y, z),
    }
}

fn trilinear_interpolate_clamped(values: &Array3<f32>, x: f32, y: f32, z: f32) -> f32 {
    let (nx, ny, nz) = values.dim();
    assert!(
        nx > 0 && ny > 0 && nz > 0,
        "wind field dimensions must be non-zero"
    );

    let (x0, x1, tx) = clamped_axis_bracket(x, nx);
    let (y0, y1, ty) = clamped_axis_bracket(y, ny);
    let (z0, z1, tz) = clamped_axis_bracket(z, nz);

    let c000 = values[[x0, y0, z0]];
    let c100 = values[[x1, y0, z0]];
    let c010 = values[[x0, y1, z0]];
    let c110 = values[[x1, y1, z0]];
    let c001 = values[[x0, y0, z1]];
    let c101 = values[[x1, y0, z1]];
    let c011 = values[[x0, y1, z1]];
    let c111 = values[[x1, y1, z1]];

    let c00 = lerp(c000, c100, tx);
    let c10 = lerp(c010, c110, tx);
    let c01 = lerp(c001, c101, tx);
    let c11 = lerp(c011, c111, tx);

    let c0 = lerp(c00, c10, ty);
    let c1 = lerp(c01, c11, ty);

    lerp(c0, c1, tz)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a.mul_add(1.0 - t, b * t)
}

#[allow(clippy::cast_precision_loss)]
fn clamped_axis_bracket(coord: f32, len: usize) -> (usize, usize, f32) {
    let max_index = (len - 1) as f32;
    let clamped = coord.clamp(0.0, max_index);
    let lower_f = clamped.floor();
    let lower = lower_f as usize;
    let upper = (lower + 1).min(len - 1);
    let frac = clamped - lower_f;
    (lower, upper, frac)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corner_field() -> WindField3D {
        let mut field = WindField3D::zeros(2, 2, 2);
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let base = (i as f32) + 10.0 * (j as f32) + 100.0 * (k as f32);
                    field.u_ms[[i, j, k]] = base;
                    field.v_ms[[i, j, k]] = base + 1.0;
                    field.w_ms[[i, j, k]] = -base;
                }
            }
        }
        field
    }

    fn linear_field(nx: usize, ny: usize, nz: usize) -> WindField3D {
        let mut field = WindField3D::zeros(nx, ny, nz);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = i as f32;
                    let y = j as f32;
                    let z = k as f32;
                    field.u_ms[[i, j, k]] = 2.0 * x + 3.0 * y - 0.5 * z + 7.0;
                    field.v_ms[[i, j, k]] = -1.25 * x + 0.75 * y + 2.5 * z - 4.0;
                    field.w_ms[[i, j, k]] = 0.5 * x - 2.0 * y + 4.0 * z + 1.5;
                }
            }
        }
        field
    }

    #[test]
    fn trilinear_exact_corner_values() {
        let field = corner_field();
        let wind = interpolate_wind_trilinear(&field, 1.0, 0.0, 1.0);
        assert_eq!(wind.u, 101.0);
        assert_eq!(wind.v, 102.0);
        assert_eq!(wind.w, -101.0);
    }

    #[test]
    fn trilinear_cell_center_is_corner_average() {
        let field = corner_field();
        let wind = interpolate_wind_trilinear(&field, 0.5, 0.5, 0.5);

        let mut u_sum = 0.0;
        let mut v_sum = 0.0;
        let mut w_sum = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    u_sum += field.u_ms[[i, j, k]];
                    v_sum += field.v_ms[[i, j, k]];
                    w_sum += field.w_ms[[i, j, k]];
                }
            }
        }

        assert!((wind.u - (u_sum / 8.0)).abs() < 1e-6);
        assert!((wind.v - (v_sum / 8.0)).abs() < 1e-6);
        assert!((wind.w - (w_sum / 8.0)).abs() < 1e-6);
    }

    #[test]
    fn trilinear_linear_field_is_exact() {
        let field = linear_field(4, 3, 5);
        let (x, y, z) = (2.2, 1.4, 3.1);

        let wind = interpolate_wind_trilinear(&field, x, y, z);

        let expected_u = 2.0 * x + 3.0 * y - 0.5 * z + 7.0;
        let expected_v = -1.25 * x + 0.75 * y + 2.5 * z - 4.0;
        let expected_w = 0.5 * x - 2.0 * y + 4.0 * z + 1.5;

        assert!((wind.u - expected_u).abs() < 1e-5);
        assert!((wind.v - expected_v).abs() < 1e-5);
        assert!((wind.w - expected_w).abs() < 1e-5);
    }

    #[test]
    fn trilinear_out_of_bounds_coordinates_are_clamped() {
        let field = corner_field();
        let wind = interpolate_wind_trilinear(&field, -8.0, 99.0, 5.0);

        assert_eq!(wind.u, field.u_ms[[0, 1, 1]]);
        assert_eq!(wind.v, field.v_ms[[0, 1, 1]]);
        assert_eq!(wind.w, field.w_ms[[0, 1, 1]]);
    }
}
