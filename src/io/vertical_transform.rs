//! Hybrid sigma-pressure to geometric-height transform (IO-03 MVP).
//!
//! Port target: FLEXPART `verttransform_ecmwf.f90` (hypsometric integration).
//!
//! Assumptions for this MVP:
//! - Uses `aknew`/`bknew` output-level coefficients from [`VerticalCoordinates`].
//! - Uses level `k=0` as the height reference (`z = 0 m`) for every column.
//! - Produces one geometric height profile per grid column `(x, y)`.
//! - Collapses column-wise heights back to `WindFieldGrid::heights_m` using a
//!   horizontal mean profile for compatibility with the current grid type.
//!
//! The full Fortran routine also handles staggered W-level transforms, slope
//! corrections, and cloud diagnostics. Those remain out of scope for IO-03 MVP.
use ndarray::{Array1, Array2, Array3};
use thiserror::Error;

use crate::constants::{GA, R_AIR};
use crate::wind::{VerticalCoordinates, WindFieldGrid};

const TV_GRADIENT_SWITCH_K: f32 = 0.2;
const VIRTUAL_TEMP_FACTOR: f32 = 0.608;
const HYPSOMETRIC_SCALE_M_PER_K: f32 = R_AIR / GA;

/// Output fields from the hybrid sigma-pressure vertical transform.
#[derive(Debug, Clone)]
pub struct HybridVerticalTransform {
    /// Pressure on transformed model levels [Pa], shape `(nx, ny, nz)`.
    pub pressure_pa: Array3<f32>,
    /// Geometric height on transformed model levels [m], shape `(nx, ny, nz)`.
    pub geometric_height_m: Array3<f32>,
}

/// Errors produced by hybrid sigma-pressure vertical transformations.
#[derive(Debug, Error, PartialEq)]
pub enum VerticalTransformError {
    #[error(
        "invalid vertical coefficient lengths for nz={nz}: aknew={aknew_len}, bknew={bknew_len}"
    )]
    InvalidCoefficientLengths {
        nz: usize,
        aknew_len: usize,
        bknew_len: usize,
    },
    #[error(
        "shape mismatch for `{field}`: expected ({expected_nx}, {expected_ny}), got ({actual_nx}, {actual_ny})"
    )]
    Shape2D {
        field: &'static str,
        expected_nx: usize,
        expected_ny: usize,
        actual_nx: usize,
        actual_ny: usize,
    },
    #[error(
        "shape mismatch for `{field}`: expected ({expected_nx}, {expected_ny}, {expected_nz}), got ({actual_nx}, {actual_ny}, {actual_nz})"
    )]
    Shape3D {
        field: &'static str,
        expected_nx: usize,
        expected_ny: usize,
        expected_nz: usize,
        actual_nx: usize,
        actual_ny: usize,
        actual_nz: usize,
    },
    #[error("invalid thermodynamic state at (i={i}, j={j}, k={k}): T={temperature_k} K, q={specific_humidity}")]
    InvalidThermodynamics {
        i: usize,
        j: usize,
        k: usize,
        temperature_k: f32,
        specific_humidity: f32,
    },
    #[error("non-positive pressure at (i={i}, j={j}, k={k}): {pressure_pa} Pa")]
    NonPositivePressure {
        i: usize,
        j: usize,
        k: usize,
        pressure_pa: f32,
    },
    #[error(
        "pressure must decrease with level index at (i={i}, j={j}, k={k}): p[k-1]={previous_pressure_pa} Pa, p[k]={current_pressure_pa} Pa"
    )]
    NonDecreasingPressure {
        i: usize,
        j: usize,
        k: usize,
        previous_pressure_pa: f32,
        current_pressure_pa: f32,
    },
    #[error(
        "geometric heights must increase with level index at (i={i}, j={j}, k={k}): z[k-1]={previous_height_m} m, z[k]={current_height_m} m"
    )]
    NonMonotonicHeight {
        i: usize,
        j: usize,
        k: usize,
        previous_height_m: f32,
        current_height_m: f32,
    },
}

/// Compute pressure on transformed output levels from hybrid coefficients.
///
/// Formula: `p = A + B * p_surface`.
pub fn compute_hybrid_level_pressure_pa(
    grid: &WindFieldGrid,
    vertical_coordinates: &VerticalCoordinates,
    surface_pressure_pa: &Array2<f32>,
) -> Result<Array3<f32>, VerticalTransformError> {
    ensure_vertical_coefficients_match_grid(grid, vertical_coordinates)?;
    ensure_shape_2d("surface_pressure_pa", surface_pressure_pa, grid.nx, grid.ny)?;

    let mut pressure_pa = Array3::zeros((grid.nx, grid.ny, grid.nz));
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            let surface_pressure = surface_pressure_pa[[i, j]];
            for k in 0..grid.nz {
                let level_pressure = vertical_coordinates.aknew_pa[k]
                    + vertical_coordinates.bknew[k] * surface_pressure;
                if !level_pressure.is_finite() || level_pressure <= 0.0 {
                    return Err(VerticalTransformError::NonPositivePressure {
                        i,
                        j,
                        k,
                        pressure_pa: level_pressure,
                    });
                }
                pressure_pa[[i, j, k]] = level_pressure;
            }
        }
    }
    Ok(pressure_pa)
}

/// Transform hybrid sigma-pressure levels into geometric heights.
///
/// This follows the hypsometric integration branch used in
/// `verttransform_ecmwf.f90` for `uvzlev`, including the `|Tv-Tv_old| > 0.2`
/// switch used in FLEXPART.
///
/// On success, this function also updates `grid.heights_m` with the horizontal
/// mean geometric height profile to keep downstream APIs compatible.
pub fn transform_hybrid_sigma_to_height(
    grid: &mut WindFieldGrid,
    vertical_coordinates: &VerticalCoordinates,
    surface_pressure_pa: &Array2<f32>,
    temperature_k: &Array3<f32>,
    specific_humidity: Option<&Array3<f32>>,
) -> Result<HybridVerticalTransform, VerticalTransformError> {
    ensure_shape_3d("temperature_k", temperature_k, grid.nx, grid.ny, grid.nz)?;
    if let Some(humidity) = specific_humidity {
        ensure_shape_3d("specific_humidity", humidity, grid.nx, grid.ny, grid.nz)?;
    }

    let pressure_pa =
        compute_hybrid_level_pressure_pa(grid, vertical_coordinates, surface_pressure_pa)?;
    let mut geometric_height_m: Array3<f32> = Array3::zeros((grid.nx, grid.ny, grid.nz));

    for i in 0..grid.nx {
        for j in 0..grid.ny {
            let mut previous_pressure_pa = pressure_pa[[i, j, 0]];
            let mut previous_virtual_temperature_k =
                virtual_temperature_k(temperature_k, specific_humidity, i, j, 0)?;

            for k in 1..grid.nz {
                let current_pressure_pa = pressure_pa[[i, j, k]];
                if current_pressure_pa >= previous_pressure_pa {
                    return Err(VerticalTransformError::NonDecreasingPressure {
                        i,
                        j,
                        k,
                        previous_pressure_pa,
                        current_pressure_pa,
                    });
                }

                let current_virtual_temperature_k =
                    virtual_temperature_k(temperature_k, specific_humidity, i, j, k)?;
                let layer_thickness_m = hypsometric_layer_thickness_m(
                    previous_pressure_pa,
                    current_pressure_pa,
                    previous_virtual_temperature_k,
                    current_virtual_temperature_k,
                );

                let previous_height_m = geometric_height_m[[i, j, k - 1]];
                let current_height_m: f32 = previous_height_m + layer_thickness_m;
                if !current_height_m.is_finite() || current_height_m <= previous_height_m {
                    return Err(VerticalTransformError::NonMonotonicHeight {
                        i,
                        j,
                        k,
                        previous_height_m,
                        current_height_m,
                    });
                }

                geometric_height_m[[i, j, k]] = current_height_m;
                previous_pressure_pa = current_pressure_pa;
                previous_virtual_temperature_k = current_virtual_temperature_k;
            }
        }
    }

    grid.heights_m = horizontal_mean_height_profile_m(&geometric_height_m);

    Ok(HybridVerticalTransform {
        pressure_pa,
        geometric_height_m,
    })
}

fn ensure_vertical_coefficients_match_grid(
    grid: &WindFieldGrid,
    vertical_coordinates: &VerticalCoordinates,
) -> Result<(), VerticalTransformError> {
    if vertical_coordinates.aknew_pa.len() != grid.nz || vertical_coordinates.bknew.len() != grid.nz
    {
        return Err(VerticalTransformError::InvalidCoefficientLengths {
            nz: grid.nz,
            aknew_len: vertical_coordinates.aknew_pa.len(),
            bknew_len: vertical_coordinates.bknew.len(),
        });
    }
    Ok(())
}

fn ensure_shape_2d(
    field: &'static str,
    values: &Array2<f32>,
    expected_nx: usize,
    expected_ny: usize,
) -> Result<(), VerticalTransformError> {
    let shape = values.shape();
    let (actual_nx, actual_ny) = (shape[0], shape[1]);
    if actual_nx != expected_nx || actual_ny != expected_ny {
        return Err(VerticalTransformError::Shape2D {
            field,
            expected_nx,
            expected_ny,
            actual_nx,
            actual_ny,
        });
    }
    Ok(())
}

fn ensure_shape_3d(
    field: &'static str,
    values: &Array3<f32>,
    expected_nx: usize,
    expected_ny: usize,
    expected_nz: usize,
) -> Result<(), VerticalTransformError> {
    let shape = values.shape();
    let (actual_nx, actual_ny, actual_nz) = (shape[0], shape[1], shape[2]);
    if actual_nx != expected_nx || actual_ny != expected_ny || actual_nz != expected_nz {
        return Err(VerticalTransformError::Shape3D {
            field,
            expected_nx,
            expected_ny,
            expected_nz,
            actual_nx,
            actual_ny,
            actual_nz,
        });
    }
    Ok(())
}

fn virtual_temperature_k(
    temperature_k: &Array3<f32>,
    specific_humidity: Option<&Array3<f32>>,
    i: usize,
    j: usize,
    k: usize,
) -> Result<f32, VerticalTransformError> {
    let temperature = temperature_k[[i, j, k]];
    let humidity = specific_humidity.map_or(0.0, |values| values[[i, j, k]]);
    if !temperature.is_finite() || temperature <= 0.0 || !humidity.is_finite() {
        return Err(VerticalTransformError::InvalidThermodynamics {
            i,
            j,
            k,
            temperature_k: temperature,
            specific_humidity: humidity,
        });
    }
    let clamped_humidity = humidity.max(0.0);
    Ok(temperature * (1.0 + VIRTUAL_TEMP_FACTOR * clamped_humidity))
}

fn hypsometric_layer_thickness_m(
    previous_pressure_pa: f32,
    current_pressure_pa: f32,
    previous_virtual_temperature_k: f32,
    current_virtual_temperature_k: f32,
) -> f32 {
    let pressure_ratio_ln = (previous_pressure_pa / current_pressure_pa).ln();
    let virtual_temperature_delta = current_virtual_temperature_k - previous_virtual_temperature_k;
    let virtual_temperature_ratio_ln =
        (current_virtual_temperature_k / previous_virtual_temperature_k).ln();

    if virtual_temperature_delta.abs() > TV_GRADIENT_SWITCH_K
        && virtual_temperature_ratio_ln.abs() > f32::EPSILON
    {
        HYPSOMETRIC_SCALE_M_PER_K * pressure_ratio_ln * virtual_temperature_delta
            / virtual_temperature_ratio_ln
    } else {
        HYPSOMETRIC_SCALE_M_PER_K * pressure_ratio_ln * current_virtual_temperature_k
    }
}

fn horizontal_mean_height_profile_m(geometric_height_m: &Array3<f32>) -> Array1<f32> {
    let shape = geometric_height_m.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

    #[allow(clippy::cast_precision_loss)]
    let horizontal_cell_count = (nx * ny) as f32;
    let mut mean_profile_m = Array1::zeros(nz);
    for k in 0..nz {
        let mut sum_m = 0.0_f32;
        for i in 0..nx {
            for j in 0..ny {
                sum_m += geometric_height_m[[i, j, k]];
            }
        }
        mean_profile_m[k] = sum_m / horizontal_cell_count;
    }
    mean_profile_m
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2, Array3};

    use super::{
        compute_hybrid_level_pressure_pa, transform_hybrid_sigma_to_height, VerticalTransformError,
    };
    use crate::constants::{GA, R_AIR};
    use crate::wind::{VerticalCoordinates, WindFieldGrid};

    fn test_vertical_coordinates(aknew_pa: [f32; 3], bknew: [f32; 3]) -> VerticalCoordinates {
        VerticalCoordinates {
            akm_pa: Array1::zeros(3),
            bkm: Array1::zeros(3),
            akz_pa: Array1::zeros(3),
            bkz: Array1::zeros(3),
            aknew_pa: Array1::from_vec(aknew_pa.to_vec()),
            bknew: Array1::from_vec(bknew.to_vec()),
        }
    }

    fn test_grid(nx: usize, ny: usize, nz: usize) -> WindFieldGrid {
        WindFieldGrid::new(nx, ny, nz, nz, nz, 0.25, 0.25, 0.0, 0.0, Array1::zeros(nz))
    }

    #[test]
    fn hybrid_pressure_matches_a_plus_b_ps() {
        let grid = test_grid(2, 1, 3);
        let vertical_coordinates = test_vertical_coordinates([0.0, 0.0, 0.0], [1.0, 0.8, 0.6]);
        let surface_pressure_pa =
            Array2::from_shape_vec((2, 1), vec![100_000.0, 90_000.0]).expect("shape must be valid");

        let pressure_pa =
            compute_hybrid_level_pressure_pa(&grid, &vertical_coordinates, &surface_pressure_pa)
                .expect("pressure transform must succeed");

        assert_abs_diff_eq!(pressure_pa[[0, 0, 0]], 100_000.0, epsilon = 1.0e-5);
        assert_abs_diff_eq!(pressure_pa[[0, 0, 1]], 80_000.0, epsilon = 1.0e-5);
        assert_abs_diff_eq!(pressure_pa[[0, 0, 2]], 60_000.0, epsilon = 1.0e-2);
        assert_abs_diff_eq!(pressure_pa[[1, 0, 0]], 90_000.0, epsilon = 1.0e-5);
        assert_abs_diff_eq!(pressure_pa[[1, 0, 1]], 72_000.0, epsilon = 1.0e-5);
        assert_abs_diff_eq!(pressure_pa[[1, 0, 2]], 54_000.0, epsilon = 1.0e-2);
    }

    #[test]
    fn isothermal_profile_matches_hypsometric_solution() {
        let mut grid = test_grid(1, 1, 3);
        let vertical_coordinates = test_vertical_coordinates([0.0, 0.0, 0.0], [1.0, 0.8, 0.6]);
        let surface_pressure_pa =
            Array2::from_shape_vec((1, 1), vec![100_000.0]).expect("shape must be valid");
        let temperature_k = Array3::from_elem((1, 1, 3), 300.0);
        let specific_humidity = Array3::zeros((1, 1, 3));

        let transformed = transform_hybrid_sigma_to_height(
            &mut grid,
            &vertical_coordinates,
            &surface_pressure_pa,
            &temperature_k,
            Some(&specific_humidity),
        )
        .expect("vertical transform must succeed");

        let scale = R_AIR / GA;
        let expected_z1 = scale * 300.0 * (100_000.0_f32 / 80_000.0_f32).ln();
        let expected_z2 = expected_z1 + scale * 300.0 * (80_000.0_f32 / 60_000.0_f32).ln();

        assert_abs_diff_eq!(
            transformed.geometric_height_m[[0, 0, 0]],
            0.0,
            epsilon = 1.0e-5
        );
        assert_abs_diff_eq!(
            transformed.geometric_height_m[[0, 0, 1]],
            expected_z1,
            epsilon = 1.0e-2
        );
        assert_abs_diff_eq!(
            transformed.geometric_height_m[[0, 0, 2]],
            expected_z2,
            epsilon = 1.0e-2
        );
        assert_abs_diff_eq!(grid.heights_m[2], expected_z2, epsilon = 1.0e-2);
    }

    #[test]
    fn moist_virtual_temperature_increases_layer_height() {
        let mut dry_grid = test_grid(1, 1, 3);
        let mut moist_grid = test_grid(1, 1, 3);
        let vertical_coordinates = test_vertical_coordinates([0.0, 0.0, 0.0], [1.0, 0.8, 0.6]);
        let surface_pressure_pa =
            Array2::from_shape_vec((1, 1), vec![100_000.0]).expect("shape must be valid");
        let temperature_k = Array3::from_elem((1, 1, 3), 290.0);
        let dry_q = Array3::zeros((1, 1, 3));
        let moist_q = Array3::from_elem((1, 1, 3), 0.015);

        let dry = transform_hybrid_sigma_to_height(
            &mut dry_grid,
            &vertical_coordinates,
            &surface_pressure_pa,
            &temperature_k,
            Some(&dry_q),
        )
        .expect("dry transform must succeed");
        let moist = transform_hybrid_sigma_to_height(
            &mut moist_grid,
            &vertical_coordinates,
            &surface_pressure_pa,
            &temperature_k,
            Some(&moist_q),
        )
        .expect("moist transform must succeed");

        assert!(moist.geometric_height_m[[0, 0, 2]] > dry.geometric_height_m[[0, 0, 2]]);
    }

    #[test]
    fn geometric_height_is_monotonic_for_synthetic_profile() {
        let mut grid = test_grid(2, 2, 3);
        let vertical_coordinates = test_vertical_coordinates([0.0, 0.0, 0.0], [1.0, 0.85, 0.7]);
        let surface_pressure_pa =
            Array2::from_shape_vec((2, 2), vec![101_000.0, 99_000.0, 97_000.0, 95_000.0])
                .expect("shape must be valid");
        let mut temperature_k = Array3::zeros((2, 2, 3));
        for i in 0..2 {
            for j in 0..2 {
                temperature_k[[i, j, 0]] = 292.0;
                temperature_k[[i, j, 1]] = 286.0;
                temperature_k[[i, j, 2]] = 280.0;
            }
        }

        let transformed = transform_hybrid_sigma_to_height(
            &mut grid,
            &vertical_coordinates,
            &surface_pressure_pa,
            &temperature_k,
            None,
        )
        .expect("transform must succeed");

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    transformed.geometric_height_m[[i, j, 1]]
                        > transformed.geometric_height_m[[i, j, 0]]
                );
                assert!(
                    transformed.geometric_height_m[[i, j, 2]]
                        > transformed.geometric_height_m[[i, j, 1]]
                );
            }
        }
    }

    #[test]
    fn transform_rejects_non_decreasing_pressure_profile() {
        let mut grid = test_grid(1, 1, 3);
        let vertical_coordinates = test_vertical_coordinates([0.0, 0.0, 0.0], [1.0, 1.1, 0.8]);
        let surface_pressure_pa =
            Array2::from_shape_vec((1, 1), vec![100_000.0]).expect("shape must be valid");
        let temperature_k = Array3::from_elem((1, 1, 3), 290.0);

        let error = transform_hybrid_sigma_to_height(
            &mut grid,
            &vertical_coordinates,
            &surface_pressure_pa,
            &temperature_k,
            None,
        )
        .expect_err("pressure inversion must fail");

        assert!(matches!(
            error,
            VerticalTransformError::NonDecreasingPressure { .. }
        ));
    }
}
