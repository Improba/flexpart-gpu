//! Temporal interpolation between meteorological snapshots (IO-04).
//!
//! Fortran reference:
//! `../flexpart/src/getfields.f90` (memory bracketing and interval handling used
//! by `timemanager.f90` to keep the simulation time between two met fields).
//!
//! This module provides deterministic linear interpolation for:
//! - 3-D wind/state fields ([`WindField3D`])
//! - 2-D surface/PBL-driving fields ([`SurfaceFields`])
//!
//! The API supports strict in-bracket validation and optional clamping behavior.

use ndarray::{Array2, Array3, Zip};
use thiserror::Error;

use crate::wind::{SurfaceFields, WindField3D};

/// Behavior when the target time lies outside `[time_t0_seconds, time_t1_seconds]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeBoundsBehavior {
    /// Reject out-of-range target times with an explicit error.
    Strict,
    /// Clamp out-of-range target times to the nearest endpoint.
    Clamp,
}

/// Errors returned by temporal interpolation utilities.
#[derive(Debug, Error)]
pub enum TemporalInterpolationError {
    #[error(
        "invalid interpolation bracket: time_t0_seconds ({time_t0_seconds}) must be strictly smaller than time_t1_seconds ({time_t1_seconds})"
    )]
    InvalidTimeBracket {
        time_t0_seconds: i64,
        time_t1_seconds: i64,
    },
    #[error(
        "target time {target_time_seconds} is outside interpolation bracket [{time_t0_seconds}, {time_t1_seconds}]"
    )]
    TargetOutsideBracket {
        target_time_seconds: i64,
        time_t0_seconds: i64,
        time_t1_seconds: i64,
    },
    #[error("3-D field shape mismatch for `{field_name}`: expected {expected:?}, got {actual:?}")]
    ShapeMismatch3D {
        field_name: &'static str,
        expected: (usize, usize, usize),
        actual: (usize, usize, usize),
    },
    #[error("2-D field shape mismatch for `{field_name}`: expected {expected:?}, got {actual:?}")]
    ShapeMismatch2D {
        field_name: &'static str,
        expected: (usize, usize),
        actual: (usize, usize),
    },
}

/// Linearly interpolate a [`WindField3D`] between two timestamps.
///
/// The interpolation factor is:
/// `alpha = (target_time_seconds - time_t0_seconds) / (time_t1_seconds - time_t0_seconds)`.
///
/// Endpoint behavior:
/// - `alpha == 0` returns an exact clone of `field_t0`
/// - `alpha == 1` returns an exact clone of `field_t1`
pub fn interpolate_wind_field_linear(
    field_t0: &WindField3D,
    field_t1: &WindField3D,
    time_t0_seconds: i64,
    time_t1_seconds: i64,
    target_time_seconds: i64,
    bounds_behavior: TimeBoundsBehavior,
) -> Result<WindField3D, TemporalInterpolationError> {
    validate_wind_field_shapes(field_t0, field_t1)?;
    let alpha = interpolation_factor(
        time_t0_seconds,
        time_t1_seconds,
        target_time_seconds,
        bounds_behavior,
    )?;

    if alpha <= 0.0 {
        return Ok(field_t0.clone());
    }
    if alpha >= 1.0 {
        return Ok(field_t1.clone());
    }

    Ok(WindField3D {
        u_ms: interpolate_array3_linear(&field_t0.u_ms, &field_t1.u_ms, alpha),
        v_ms: interpolate_array3_linear(&field_t0.v_ms, &field_t1.v_ms, alpha),
        w_ms: interpolate_array3_linear(&field_t0.w_ms, &field_t1.w_ms, alpha),
        temperature_k: interpolate_array3_linear(
            &field_t0.temperature_k,
            &field_t1.temperature_k,
            alpha,
        ),
        specific_humidity: interpolate_array3_linear(
            &field_t0.specific_humidity,
            &field_t1.specific_humidity,
            alpha,
        ),
        pressure_pa: interpolate_array3_linear(&field_t0.pressure_pa, &field_t1.pressure_pa, alpha),
        air_density_kg_m3: interpolate_array3_linear(
            &field_t0.air_density_kg_m3,
            &field_t1.air_density_kg_m3,
            alpha,
        ),
        density_gradient_kg_m2: interpolate_array3_linear(
            &field_t0.density_gradient_kg_m2,
            &field_t1.density_gradient_kg_m2,
            alpha,
        ),
    })
}

/// Linearly interpolate [`SurfaceFields`] between two timestamps.
///
/// This includes near-surface winds and PBL-driving diagnostics such as `hmix`,
/// `ustar`, and `oli`.
pub fn interpolate_surface_fields_linear(
    surface_t0: &SurfaceFields,
    surface_t1: &SurfaceFields,
    time_t0_seconds: i64,
    time_t1_seconds: i64,
    target_time_seconds: i64,
    bounds_behavior: TimeBoundsBehavior,
) -> Result<SurfaceFields, TemporalInterpolationError> {
    validate_surface_field_shapes(surface_t0, surface_t1)?;
    let alpha = interpolation_factor(
        time_t0_seconds,
        time_t1_seconds,
        target_time_seconds,
        bounds_behavior,
    )?;

    if alpha <= 0.0 {
        return Ok(surface_t0.clone());
    }
    if alpha >= 1.0 {
        return Ok(surface_t1.clone());
    }

    Ok(SurfaceFields {
        surface_pressure_pa: interpolate_array2_linear(
            &surface_t0.surface_pressure_pa,
            &surface_t1.surface_pressure_pa,
            alpha,
        ),
        u10_ms: interpolate_array2_linear(&surface_t0.u10_ms, &surface_t1.u10_ms, alpha),
        v10_ms: interpolate_array2_linear(&surface_t0.v10_ms, &surface_t1.v10_ms, alpha),
        temperature_2m_k: interpolate_array2_linear(
            &surface_t0.temperature_2m_k,
            &surface_t1.temperature_2m_k,
            alpha,
        ),
        dewpoint_2m_k: interpolate_array2_linear(
            &surface_t0.dewpoint_2m_k,
            &surface_t1.dewpoint_2m_k,
            alpha,
        ),
        precip_large_scale_mm_h: interpolate_array2_linear(
            &surface_t0.precip_large_scale_mm_h,
            &surface_t1.precip_large_scale_mm_h,
            alpha,
        ),
        precip_convective_mm_h: interpolate_array2_linear(
            &surface_t0.precip_convective_mm_h,
            &surface_t1.precip_convective_mm_h,
            alpha,
        ),
        sensible_heat_flux_w_m2: interpolate_array2_linear(
            &surface_t0.sensible_heat_flux_w_m2,
            &surface_t1.sensible_heat_flux_w_m2,
            alpha,
        ),
        solar_radiation_w_m2: interpolate_array2_linear(
            &surface_t0.solar_radiation_w_m2,
            &surface_t1.solar_radiation_w_m2,
            alpha,
        ),
        surface_stress_n_m2: interpolate_array2_linear(
            &surface_t0.surface_stress_n_m2,
            &surface_t1.surface_stress_n_m2,
            alpha,
        ),
        friction_velocity_ms: interpolate_array2_linear(
            &surface_t0.friction_velocity_ms,
            &surface_t1.friction_velocity_ms,
            alpha,
        ),
        convective_velocity_scale_ms: interpolate_array2_linear(
            &surface_t0.convective_velocity_scale_ms,
            &surface_t1.convective_velocity_scale_ms,
            alpha,
        ),
        mixing_height_m: interpolate_array2_linear(
            &surface_t0.mixing_height_m,
            &surface_t1.mixing_height_m,
            alpha,
        ),
        tropopause_height_m: interpolate_array2_linear(
            &surface_t0.tropopause_height_m,
            &surface_t1.tropopause_height_m,
            alpha,
        ),
        inv_obukhov_length_per_m: interpolate_array2_linear(
            &surface_t0.inv_obukhov_length_per_m,
            &surface_t1.inv_obukhov_length_per_m,
            alpha,
        ),
    })
}

fn interpolation_factor(
    time_t0_seconds: i64,
    time_t1_seconds: i64,
    target_time_seconds: i64,
    bounds_behavior: TimeBoundsBehavior,
) -> Result<f32, TemporalInterpolationError> {
    if time_t1_seconds <= time_t0_seconds {
        return Err(TemporalInterpolationError::InvalidTimeBracket {
            time_t0_seconds,
            time_t1_seconds,
        });
    }

    let in_range = target_time_seconds >= time_t0_seconds && target_time_seconds <= time_t1_seconds;
    let effective_target = if in_range {
        target_time_seconds
    } else {
        match bounds_behavior {
            TimeBoundsBehavior::Strict => {
                return Err(TemporalInterpolationError::TargetOutsideBracket {
                    target_time_seconds,
                    time_t0_seconds,
                    time_t1_seconds,
                });
            }
            TimeBoundsBehavior::Clamp => {
                target_time_seconds.clamp(time_t0_seconds, time_t1_seconds)
            }
        }
    };

    let window_seconds = (time_t1_seconds - time_t0_seconds) as f64;
    let elapsed_seconds = (effective_target - time_t0_seconds) as f64;
    Ok((elapsed_seconds / window_seconds) as f32)
}

fn validate_wind_field_shapes(
    field_t0: &WindField3D,
    field_t1: &WindField3D,
) -> Result<(), TemporalInterpolationError> {
    ensure_shape_match_3d("u_ms", &field_t0.u_ms, &field_t1.u_ms)?;
    ensure_shape_match_3d("v_ms", &field_t0.v_ms, &field_t1.v_ms)?;
    ensure_shape_match_3d("w_ms", &field_t0.w_ms, &field_t1.w_ms)?;
    ensure_shape_match_3d(
        "temperature_k",
        &field_t0.temperature_k,
        &field_t1.temperature_k,
    )?;
    ensure_shape_match_3d(
        "specific_humidity",
        &field_t0.specific_humidity,
        &field_t1.specific_humidity,
    )?;
    ensure_shape_match_3d("pressure_pa", &field_t0.pressure_pa, &field_t1.pressure_pa)?;
    ensure_shape_match_3d(
        "air_density_kg_m3",
        &field_t0.air_density_kg_m3,
        &field_t1.air_density_kg_m3,
    )?;
    ensure_shape_match_3d(
        "density_gradient_kg_m2",
        &field_t0.density_gradient_kg_m2,
        &field_t1.density_gradient_kg_m2,
    )?;
    Ok(())
}

fn validate_surface_field_shapes(
    surface_t0: &SurfaceFields,
    surface_t1: &SurfaceFields,
) -> Result<(), TemporalInterpolationError> {
    ensure_shape_match_2d(
        "surface_pressure_pa",
        &surface_t0.surface_pressure_pa,
        &surface_t1.surface_pressure_pa,
    )?;
    ensure_shape_match_2d("u10_ms", &surface_t0.u10_ms, &surface_t1.u10_ms)?;
    ensure_shape_match_2d("v10_ms", &surface_t0.v10_ms, &surface_t1.v10_ms)?;
    ensure_shape_match_2d(
        "temperature_2m_k",
        &surface_t0.temperature_2m_k,
        &surface_t1.temperature_2m_k,
    )?;
    ensure_shape_match_2d(
        "dewpoint_2m_k",
        &surface_t0.dewpoint_2m_k,
        &surface_t1.dewpoint_2m_k,
    )?;
    ensure_shape_match_2d(
        "precip_large_scale_mm_h",
        &surface_t0.precip_large_scale_mm_h,
        &surface_t1.precip_large_scale_mm_h,
    )?;
    ensure_shape_match_2d(
        "precip_convective_mm_h",
        &surface_t0.precip_convective_mm_h,
        &surface_t1.precip_convective_mm_h,
    )?;
    ensure_shape_match_2d(
        "sensible_heat_flux_w_m2",
        &surface_t0.sensible_heat_flux_w_m2,
        &surface_t1.sensible_heat_flux_w_m2,
    )?;
    ensure_shape_match_2d(
        "solar_radiation_w_m2",
        &surface_t0.solar_radiation_w_m2,
        &surface_t1.solar_radiation_w_m2,
    )?;
    ensure_shape_match_2d(
        "surface_stress_n_m2",
        &surface_t0.surface_stress_n_m2,
        &surface_t1.surface_stress_n_m2,
    )?;
    ensure_shape_match_2d(
        "friction_velocity_ms",
        &surface_t0.friction_velocity_ms,
        &surface_t1.friction_velocity_ms,
    )?;
    ensure_shape_match_2d(
        "convective_velocity_scale_ms",
        &surface_t0.convective_velocity_scale_ms,
        &surface_t1.convective_velocity_scale_ms,
    )?;
    ensure_shape_match_2d(
        "mixing_height_m",
        &surface_t0.mixing_height_m,
        &surface_t1.mixing_height_m,
    )?;
    ensure_shape_match_2d(
        "tropopause_height_m",
        &surface_t0.tropopause_height_m,
        &surface_t1.tropopause_height_m,
    )?;
    ensure_shape_match_2d(
        "inv_obukhov_length_per_m",
        &surface_t0.inv_obukhov_length_per_m,
        &surface_t1.inv_obukhov_length_per_m,
    )?;
    Ok(())
}

fn ensure_shape_match_3d(
    field_name: &'static str,
    left: &Array3<f32>,
    right: &Array3<f32>,
) -> Result<(), TemporalInterpolationError> {
    let expected = left.dim();
    let actual = right.dim();
    if expected != actual {
        return Err(TemporalInterpolationError::ShapeMismatch3D {
            field_name,
            expected,
            actual,
        });
    }
    Ok(())
}

fn ensure_shape_match_2d(
    field_name: &'static str,
    left: &Array2<f32>,
    right: &Array2<f32>,
) -> Result<(), TemporalInterpolationError> {
    let expected = left.dim();
    let actual = right.dim();
    if expected != actual {
        return Err(TemporalInterpolationError::ShapeMismatch2D {
            field_name,
            expected,
            actual,
        });
    }
    Ok(())
}

fn interpolate_array3_linear(
    field_t0: &Array3<f32>,
    field_t1: &Array3<f32>,
    alpha: f32,
) -> Array3<f32> {
    let mut out = field_t0.clone();
    Zip::from(&mut out)
        .and(field_t0)
        .and(field_t1)
        .for_each(|value, start, end| *value = lerp(*start, *end, alpha));
    out
}

fn interpolate_array2_linear(
    field_t0: &Array2<f32>,
    field_t1: &Array2<f32>,
    alpha: f32,
) -> Array2<f32> {
    let mut out = field_t0.clone();
    Zip::from(&mut out)
        .and(field_t0)
        .and(field_t1)
        .for_each(|value, start, end| *value = lerp(*start, *end, alpha));
    out
}

#[inline]
fn lerp(start: f32, end: f32, alpha: f32) -> f32 {
    start.mul_add(1.0 - alpha, end * alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn wind_field_with_offset(nx: usize, ny: usize, nz: usize, offset: f32) -> WindField3D {
        let mut field = WindField3D::zeros(nx, ny, nz);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let base = i as f32 + 10.0 * (j as f32) + 100.0 * (k as f32);
                    field.u_ms[[i, j, k]] = base + offset;
                    field.v_ms[[i, j, k]] = 2.0 * base + offset;
                    field.w_ms[[i, j, k]] = 3.0 * base + offset;
                    field.temperature_k[[i, j, k]] = 4.0 * base + offset;
                    field.specific_humidity[[i, j, k]] = 5.0 * base + offset;
                    field.pressure_pa[[i, j, k]] = 6.0 * base + offset;
                    field.air_density_kg_m3[[i, j, k]] = 7.0 * base + offset;
                    field.density_gradient_kg_m2[[i, j, k]] = 8.0 * base + offset;
                }
            }
        }
        field
    }

    fn surface_fields_with_offset(nx: usize, ny: usize, offset: f32) -> SurfaceFields {
        let mut surface = SurfaceFields::zeros(nx, ny);
        for i in 0..nx {
            for j in 0..ny {
                let base = i as f32 + 10.0 * (j as f32);
                surface.surface_pressure_pa[[i, j]] = base + offset;
                surface.u10_ms[[i, j]] = 2.0 * base + offset;
                surface.v10_ms[[i, j]] = 3.0 * base + offset;
                surface.temperature_2m_k[[i, j]] = 4.0 * base + offset;
                surface.dewpoint_2m_k[[i, j]] = 5.0 * base + offset;
                surface.precip_large_scale_mm_h[[i, j]] = 6.0 * base + offset;
                surface.precip_convective_mm_h[[i, j]] = 7.0 * base + offset;
                surface.sensible_heat_flux_w_m2[[i, j]] = 8.0 * base + offset;
                surface.solar_radiation_w_m2[[i, j]] = 9.0 * base + offset;
                surface.surface_stress_n_m2[[i, j]] = 10.0 * base + offset;
                surface.friction_velocity_ms[[i, j]] = 11.0 * base + offset;
                surface.convective_velocity_scale_ms[[i, j]] = 12.0 * base + offset;
                surface.mixing_height_m[[i, j]] = 13.0 * base + offset;
                surface.tropopause_height_m[[i, j]] = 14.0 * base + offset;
                surface.inv_obukhov_length_per_m[[i, j]] = 15.0 * base + offset;
            }
        }
        surface
    }

    #[test]
    fn wind_interpolation_returns_exact_endpoints() {
        let field_t0 = wind_field_with_offset(2, 2, 2, 10.0);
        let field_t1 = wind_field_with_offset(2, 2, 2, 14.0);

        let at_t0 = interpolate_wind_field_linear(
            &field_t0,
            &field_t1,
            100,
            200,
            100,
            TimeBoundsBehavior::Strict,
        )
        .expect("t0 interpolation should succeed");
        let at_t1 = interpolate_wind_field_linear(
            &field_t0,
            &field_t1,
            100,
            200,
            200,
            TimeBoundsBehavior::Strict,
        )
        .expect("t1 interpolation should succeed");

        assert_eq!(at_t0.u_ms, field_t0.u_ms);
        assert_eq!(at_t0.pressure_pa, field_t0.pressure_pa);
        assert_eq!(at_t1.u_ms, field_t1.u_ms);
        assert_eq!(at_t1.pressure_pa, field_t1.pressure_pa);
    }

    #[test]
    fn wind_interpolation_midpoint_is_linear_average() {
        let field_t0 = wind_field_with_offset(2, 2, 2, 10.0);
        let field_t1 = wind_field_with_offset(2, 2, 2, 14.0);
        let mid = interpolate_wind_field_linear(
            &field_t0,
            &field_t1,
            100,
            200,
            150,
            TimeBoundsBehavior::Strict,
        )
        .expect("midpoint interpolation should succeed");

        assert_relative_eq!(
            mid.u_ms[[1, 1, 1]],
            field_t0.u_ms[[1, 1, 1]] + 2.0,
            epsilon = 1.0e-6
        );
        assert_relative_eq!(
            mid.temperature_k[[0, 1, 0]],
            field_t0.temperature_k[[0, 1, 0]] + 2.0,
            epsilon = 1.0e-6
        );
        assert_relative_eq!(
            mid.air_density_kg_m3[[1, 0, 1]],
            field_t0.air_density_kg_m3[[1, 0, 1]] + 2.0,
            epsilon = 1.0e-6
        );
    }

    #[test]
    fn wind_interpolation_rejects_out_of_bounds_target_in_strict_mode() {
        let field_t0 = wind_field_with_offset(1, 1, 1, 0.0);
        let field_t1 = wind_field_with_offset(1, 1, 1, 2.0);
        let error = interpolate_wind_field_linear(
            &field_t0,
            &field_t1,
            0,
            10,
            11,
            TimeBoundsBehavior::Strict,
        )
        .expect_err("strict mode should reject out-of-bracket target");

        assert!(matches!(
            error,
            TemporalInterpolationError::TargetOutsideBracket { .. }
        ));
    }

    #[test]
    fn wind_interpolation_clamps_out_of_bounds_target_when_requested() {
        let field_t0 = wind_field_with_offset(1, 1, 1, 0.0);
        let field_t1 = wind_field_with_offset(1, 1, 1, 2.0);
        let clamped_low = interpolate_wind_field_linear(
            &field_t0,
            &field_t1,
            0,
            10,
            -5,
            TimeBoundsBehavior::Clamp,
        )
        .expect("clamped low-end interpolation should succeed");
        let clamped_high = interpolate_wind_field_linear(
            &field_t0,
            &field_t1,
            0,
            10,
            99,
            TimeBoundsBehavior::Clamp,
        )
        .expect("clamped high-end interpolation should succeed");

        assert_eq!(clamped_low.u_ms, field_t0.u_ms);
        assert_eq!(clamped_high.u_ms, field_t1.u_ms);
    }

    #[test]
    fn surface_interpolation_returns_exact_endpoints_and_midpoint() {
        let surface_t0 = surface_fields_with_offset(2, 2, 20.0);
        let surface_t1 = surface_fields_with_offset(2, 2, 24.0);

        let at_t0 = interpolate_surface_fields_linear(
            &surface_t0,
            &surface_t1,
            1_000,
            1_060,
            1_000,
            TimeBoundsBehavior::Strict,
        )
        .expect("surface t0 interpolation should succeed");
        let mid = interpolate_surface_fields_linear(
            &surface_t0,
            &surface_t1,
            1_000,
            1_060,
            1_030,
            TimeBoundsBehavior::Strict,
        )
        .expect("surface midpoint interpolation should succeed");

        assert_eq!(at_t0.u10_ms, surface_t0.u10_ms);
        assert_eq!(at_t0.mixing_height_m, surface_t0.mixing_height_m);
        assert_relative_eq!(
            mid.u10_ms[[1, 1]],
            surface_t0.u10_ms[[1, 1]] + 2.0,
            epsilon = 1.0e-6
        );
        assert_relative_eq!(
            mid.mixing_height_m[[0, 1]],
            surface_t0.mixing_height_m[[0, 1]] + 2.0,
            epsilon = 1.0e-6
        );
    }

    #[test]
    fn interpolation_rejects_invalid_time_bracket() {
        let field_t0 = wind_field_with_offset(1, 1, 1, 0.0);
        let field_t1 = wind_field_with_offset(1, 1, 1, 1.0);

        let error = interpolate_wind_field_linear(
            &field_t0,
            &field_t1,
            100,
            100,
            100,
            TimeBoundsBehavior::Strict,
        )
        .expect_err("equal timestamps should fail");

        assert!(matches!(
            error,
            TemporalInterpolationError::InvalidTimeBracket { .. }
        ));
    }
}
