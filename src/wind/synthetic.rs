//! Synthetic analytical wind fields for validation and regression tests.
//!
//! These generators provide closed-form wind patterns for advection validation
//! tasks `A-07` (uniform flow) and `A-08` (linear shear flow), plus a
//! physically interpretable vortex case for interpolation/advection stress tests.
//! The resulting fields are deterministic and independent from meteorological I/O.

use super::{WindField3D, WindFieldGrid};
use crate::constants::{PI180, R_EARTH};

/// Background scalar state used to fill non-velocity fields in synthetic data.
///
/// All values are uniform in space and use explicit SI units.
#[derive(Debug, Clone, Copy)]
pub struct SyntheticFieldBackground {
    /// Air temperature [K].
    pub temperature_k: f32,
    /// Specific humidity [kg/kg].
    pub specific_humidity_kg_kg: f32,
    /// Air pressure [Pa].
    pub pressure_pa: f32,
    /// Air density [kg/m^3].
    pub air_density_kg_m3: f32,
    /// Vertical air density gradient [kg/m^2].
    pub density_gradient_kg_m2: f32,
}

impl Default for SyntheticFieldBackground {
    fn default() -> Self {
        Self {
            temperature_k: 288.0,
            specific_humidity_kg_kg: 0.005,
            pressure_pa: 101_325.0,
            air_density_kg_m3: 1.225,
            density_gradient_kg_m2: 0.0,
        }
    }
}

/// Parameters describing a Rankine vortex.
///
/// The tangential speed profile is:
/// - solid-body rotation inside `core_radius_m`
/// - inverse-radius decay outside `core_radius_m`
#[derive(Debug, Clone, Copy)]
pub struct RankineVortexConfig {
    /// Vortex center longitude [degrees].
    pub center_lon_deg: f32,
    /// Vortex center latitude [degrees].
    pub center_lat_deg: f32,
    /// Vortex core radius [m].
    pub core_radius_m: f32,
    /// Maximum tangential speed at the core edge [m/s].
    pub max_tangential_speed_ms: f32,
    /// If `true`, clockwise rotation; otherwise counter-clockwise.
    pub clockwise: bool,
    /// Uniform background U component [m/s].
    pub background_u_ms: f32,
    /// Uniform background V component [m/s].
    pub background_v_ms: f32,
    /// Uniform vertical velocity W [m/s].
    pub vertical_velocity_ms: f32,
}

impl Default for RankineVortexConfig {
    fn default() -> Self {
        Self {
            center_lon_deg: 0.0,
            center_lat_deg: 0.0,
            core_radius_m: 5_000.0,
            max_tangential_speed_ms: 20.0,
            clockwise: false,
            background_u_ms: 0.0,
            background_v_ms: 0.0,
            vertical_velocity_ms: 0.0,
        }
    }
}

/// Generate a spatially uniform wind field.
///
/// Velocity components are constant everywhere:
/// `u(x,y,z) = u_ms`, `v(x,y,z) = v_ms`, `w(x,y,z) = w_ms`.
#[must_use]
pub fn uniform_wind_field(grid: &WindFieldGrid, u_ms: f32, v_ms: f32, w_ms: f32) -> WindField3D {
    let mut field = initialize_synthetic_field(grid);
    field.u_ms.fill(u_ms);
    field.v_ms.fill(v_ms);
    field.w_ms.fill(w_ms);
    field
}

/// Generate a linear shear wind profile with altitude.
///
/// For each component:
/// `component(z) = component_reference_ms + dcomponent_dz_per_s * (z - reference_height_m)`.
///
/// Derivatives are expressed as `(m/s) / m = s^-1`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn linear_shear_wind_field(
    grid: &WindFieldGrid,
    u_reference_ms: f32,
    v_reference_ms: f32,
    w_reference_ms: f32,
    du_dz_per_s: f32,
    dv_dz_per_s: f32,
    dw_dz_per_s: f32,
    reference_height_m: f32,
) -> WindField3D {
    let mut field = initialize_synthetic_field(grid);

    for (k, height_m) in grid.heights_m.indexed_iter() {
        let delta_z_m = *height_m - reference_height_m;
        let u_level_ms = u_reference_ms + du_dz_per_s * delta_z_m;
        let v_level_ms = v_reference_ms + dv_dz_per_s * delta_z_m;
        let w_level_ms = w_reference_ms + dw_dz_per_s * delta_z_m;

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                field.u_ms[[i, j, k]] = u_level_ms;
                field.v_ms[[i, j, k]] = v_level_ms;
                field.w_ms[[i, j, k]] = w_level_ms;
            }
        }
    }

    field
}

/// Generate a Rankine vortex wind field on the horizontal grid.
///
/// Horizontal velocities are computed from a tangential profile around
/// `center_lon_deg, center_lat_deg` and are identical for all vertical levels.
#[must_use]
pub fn rankine_vortex_wind_field(grid: &WindFieldGrid, config: RankineVortexConfig) -> WindField3D {
    assert!(
        config.core_radius_m > 0.0,
        "core_radius_m must be positive (got {})",
        config.core_radius_m
    );
    assert!(
        config.max_tangential_speed_ms >= 0.0,
        "max_tangential_speed_ms must be non-negative (got {})",
        config.max_tangential_speed_ms
    );

    let mut field = initialize_synthetic_field(grid);
    let meters_per_deg_lat_m = R_EARTH * PI180;
    let meters_per_deg_lon_m = R_EARTH * PI180 * (config.center_lat_deg * PI180).cos();
    let rotation_sign = if config.clockwise { -1.0 } else { 1.0 };

    for i in 0..grid.nx {
        let lon_deg = grid.xlon0 + (i as f32) * grid.dx_deg;
        for j in 0..grid.ny {
            let lat_deg = grid.ylat0 + (j as f32) * grid.dy_deg;
            let dx_m = (lon_deg - config.center_lon_deg) * meters_per_deg_lon_m;
            let dy_m = (lat_deg - config.center_lat_deg) * meters_per_deg_lat_m;
            let radius_m = dx_m.hypot(dy_m);

            let (u_ms, v_ms) = if radius_m > 0.0 {
                let tangential_speed_ms = rankine_tangential_speed_ms(
                    radius_m,
                    config.core_radius_m,
                    config.max_tangential_speed_ms,
                );
                let unit_tangent_x = -dy_m / radius_m;
                let unit_tangent_y = dx_m / radius_m;
                (
                    config.background_u_ms + rotation_sign * tangential_speed_ms * unit_tangent_x,
                    config.background_v_ms + rotation_sign * tangential_speed_ms * unit_tangent_y,
                )
            } else {
                (config.background_u_ms, config.background_v_ms)
            };

            for k in 0..grid.nz {
                field.u_ms[[i, j, k]] = u_ms;
                field.v_ms[[i, j, k]] = v_ms;
                field.w_ms[[i, j, k]] = config.vertical_velocity_ms;
            }
        }
    }

    field
}

fn rankine_tangential_speed_ms(
    radius_m: f32,
    core_radius_m: f32,
    max_tangential_speed_ms: f32,
) -> f32 {
    if radius_m <= core_radius_m {
        max_tangential_speed_ms * (radius_m / core_radius_m)
    } else {
        max_tangential_speed_ms * (core_radius_m / radius_m)
    }
}

fn initialize_synthetic_field(grid: &WindFieldGrid) -> WindField3D {
    let mut field = WindField3D::zeros(grid.nx, grid.ny, grid.nz);
    let background = SyntheticFieldBackground::default();
    field.temperature_k.fill(background.temperature_k);
    field
        .specific_humidity
        .fill(background.specific_humidity_kg_kg);
    field.pressure_pa.fill(background.pressure_pa);
    field.air_density_kg_m3.fill(background.air_density_kg_m3);
    field
        .density_gradient_kg_m2
        .fill(background.density_gradient_kg_m2);
    field
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    fn make_test_grid(nx: usize, ny: usize, heights_m: Vec<f32>) -> WindFieldGrid {
        let nz = heights_m.len();
        WindFieldGrid::new(
            nx,
            ny,
            nz,
            nz,
            nz,
            0.01,
            0.01,
            -0.04,
            -0.04,
            Array1::from_vec(heights_m),
        )
    }

    #[test]
    fn uniform_field_is_constant_everywhere() {
        let grid = make_test_grid(4, 3, vec![0.0, 100.0, 250.0]);
        let field = uniform_wind_field(&grid, 5.0, -2.5, 0.75);

        for value in &field.u_ms {
            assert_relative_eq!(*value, 5.0, epsilon = 1.0e-6);
        }
        for value in &field.v_ms {
            assert_relative_eq!(*value, -2.5, epsilon = 1.0e-6);
        }
        for value in &field.w_ms {
            assert_relative_eq!(*value, 0.75, epsilon = 1.0e-6);
        }
    }

    #[test]
    fn linear_shear_follows_expected_altitude_profile() {
        let heights_m = vec![0.0, 100.0, 200.0, 500.0];
        let grid = make_test_grid(3, 2, heights_m.clone());
        let field = linear_shear_wind_field(
            &grid, 2.0,    // u at z_ref [m/s]
            -1.0,   // v at z_ref [m/s]
            0.25,   // w at z_ref [m/s]
            0.01,   // du/dz [1/s]
            -0.002, // dv/dz [1/s]
            0.0005, // dw/dz [1/s]
            100.0,  // z_ref [m]
        );

        for (k, height_m) in heights_m.iter().enumerate() {
            let delta_z_m = *height_m - 100.0;
            let expected_u_ms = 2.0 + 0.01 * delta_z_m;
            let expected_v_ms = -1.0 - 0.002 * delta_z_m;
            let expected_w_ms = 0.25 + 0.0005 * delta_z_m;

            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    assert_relative_eq!(field.u_ms[[i, j, k]], expected_u_ms, epsilon = 1.0e-6);
                    assert_relative_eq!(field.v_ms[[i, j, k]], expected_v_ms, epsilon = 1.0e-6);
                    assert_relative_eq!(field.w_ms[[i, j, k]], expected_w_ms, epsilon = 1.0e-6);
                }
            }
        }
    }

    #[test]
    fn rankine_vortex_matches_core_and_outer_radial_behavior() {
        let grid = make_test_grid(9, 9, vec![0.0, 100.0]);
        let config = RankineVortexConfig {
            center_lon_deg: 0.0,
            center_lat_deg: 0.0,
            core_radius_m: 2_500.0,
            max_tangential_speed_ms: 20.0,
            clockwise: false,
            background_u_ms: 0.0,
            background_v_ms: 0.0,
            vertical_velocity_ms: 0.0,
        };
        let field = rankine_vortex_wind_field(&grid, config);

        let center_i = 4;
        let center_j = 4;
        assert_relative_eq!(field.u_ms[[center_i, center_j, 0]], 0.0, epsilon = 1.0e-6);
        assert_relative_eq!(field.v_ms[[center_i, center_j, 0]], 0.0, epsilon = 1.0e-6);

        let speed_at = |i: usize| field.u_ms[[i, center_j, 0]].hypot(field.v_ms[[i, center_j, 0]]);
        let r_step_m = R_EARTH * PI180 * 0.01;
        let r1_m = r_step_m;
        let r2_m = 2.0 * r_step_m;
        let r3_m = 3.0 * r_step_m;

        let speed1_ms = speed_at(5);
        let speed2_ms = speed_at(6);
        let speed3_ms = speed_at(7);

        // Inside the core: v_t / r should be constant (solid-body rotation).
        assert_relative_eq!(speed1_ms / r1_m, speed2_ms / r2_m, epsilon = 1.0e-6);
        assert_relative_eq!(
            speed1_ms / r1_m,
            config.max_tangential_speed_ms / config.core_radius_m,
            epsilon = 1.0e-6
        );

        // Outside the core: v_t * r should be constant (inverse-radius decay).
        assert_relative_eq!(
            speed3_ms * r3_m,
            config.max_tangential_speed_ms * config.core_radius_m,
            epsilon = 1.0e-2
        );
    }
}
