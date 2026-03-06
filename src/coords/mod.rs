//! Coordinate system utilities: lat/lon ↔ grid indices, great-circle distance.
//!
//! Ported from `coordtrafo.f90` and grid parameters in `com_mod.f90`.
//!
//! Key design: particle positions on the GPU use [`RelativePosition`] (integer
//! cell index + f32 fractional offset) to avoid the precision loss that would
//! occur when storing absolute lat/lon as f32.

use crate::constants::{PI180, R_EARTH};
use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Coordinate types
// ---------------------------------------------------------------------------

/// Geographic coordinates in degrees (latitude, longitude).
///
/// Uses f64 for full double-precision on the CPU side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoCoord {
    /// Latitude \[degrees\], positive north, range \[−90, 90\].
    pub lat: f64,
    /// Longitude \[degrees\], positive east.
    pub lon: f64,
}

/// Grid-relative coordinates as fractional indices.
///
/// Corresponds to the transformed coordinates in `coordtrafo.f90`:
/// ```text
///   x = (lon − xlon0) / dx
///   y = (lat − ylat0) / dy
/// ```
///
/// Uses f64 for CPU-side precision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridCoord {
    /// Fractional grid index along the x (longitude) axis.
    pub x: f64,
    /// Fractional grid index along the y (latitude) axis.
    pub y: f64,
}

/// GPU-friendly particle position: integer cell index + f32 fractional offset.
///
/// Avoids the precision loss inherent in storing absolute lat/lon as f32.
/// The full grid position is reconstructed as `cell_x + frac_x` (likewise y).
///
/// Corresponds to the WGSL struct used in compute shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RelativePosition {
    /// Integer grid cell index along x (longitude axis).
    pub cell_x: i32,
    /// Integer grid cell index along y (latitude axis).
    pub cell_y: i32,
    /// Fractional offset within the cell along x, ∈ \[0, 1).
    pub frac_x: f32,
    /// Fractional offset within the cell along y, ∈ \[0, 1).
    pub frac_y: f32,
}

// ---------------------------------------------------------------------------
// Grid domain
// ---------------------------------------------------------------------------

/// Meteorological grid domain definition.
///
/// Mirrors the grid parameters from `com_mod.f90` (lines 294–320):
/// `xlon0`, `ylat0`, `dx`, `dy`, `nx`, `ny`.
#[derive(Clone, Debug)]
pub struct GridDomain {
    /// Longitude of the lower-left grid corner \[degrees\].
    pub xlon0: f64,
    /// Latitude of the lower-left grid corner \[degrees\].
    pub ylat0: f64,
    /// Grid spacing in the longitude direction \[degrees\].
    pub dx: f64,
    /// Grid spacing in the latitude direction \[degrees\].
    pub dy: f64,
    /// Number of grid points along x (longitude).
    pub nx: usize,
    /// Number of grid points along y (latitude).
    pub ny: usize,
}

impl GridDomain {
    /// Check whether a [`GridCoord`] lies inside the domain
    /// \[0, nx−1) × \[0, ny−1).
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // grid dims are always small
    pub fn contains(&self, gc: &GridCoord) -> bool {
        gc.x >= 0.0 && gc.x < (self.nx - 1) as f64 && gc.y >= 0.0 && gc.y < (self.ny - 1) as f64
    }
}

// ---------------------------------------------------------------------------
// Coordinate conversions (ported from coordtrafo.f90)
// ---------------------------------------------------------------------------

/// Convert geographic coordinates to grid-relative indices.
///
/// Implements the transform from `coordtrafo.f90`:
/// ```text
///   x = (lon − xlon0) / dx
///   y = (lat − ylat0) / dy
/// ```
#[must_use]
pub fn geo_to_grid(geo: GeoCoord, grid: &GridDomain) -> GridCoord {
    GridCoord {
        x: (geo.lon - grid.xlon0) / grid.dx,
        y: (geo.lat - grid.ylat0) / grid.dy,
    }
}

/// Convert grid-relative indices back to geographic coordinates.
///
/// Inverse of [`geo_to_grid`]:
/// ```text
///   lon = x · dx + xlon0
///   lat = y · dy + ylat0
/// ```
#[must_use]
pub fn grid_to_geo(grid_coord: GridCoord, grid: &GridDomain) -> GeoCoord {
    GeoCoord {
        lon: grid_coord.x.mul_add(grid.dx, grid.xlon0),
        lat: grid_coord.y.mul_add(grid.dy, grid.ylat0),
    }
}

/// Great-circle distance between two geographic points \[m\].
///
/// Uses the Haversine formula with `R_EARTH` from `par_mod.f90`.
#[must_use]
pub fn distance_meters(a: GeoCoord, b: GeoCoord) -> f32 {
    let pi180 = f64::from(PI180);
    let lat1 = a.lat * pi180;
    let lat2 = b.lat * pi180;
    let dlat = (b.lat - a.lat) * pi180;
    let dlon = (b.lon - a.lon) * pi180;

    let h = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * h.sqrt().atan2((1.0 - h).sqrt());

    #[allow(clippy::cast_possible_truncation)] // intentional f64→f32 for API
    {
        (f64::from(R_EARTH) * c) as f32
    }
}

/// Meters per degree of longitude at a given latitude.
///
/// At the equator ≈ 111 195 m/°; shrinks as cos(lat) toward the poles.
/// Uses `R_EARTH` and `PI180` from `par_mod.f90`.
#[must_use]
pub fn dx_meters_at_latitude(lat_deg: f32) -> f32 {
    R_EARTH * (lat_deg * PI180).cos() * PI180
}

// ---------------------------------------------------------------------------
// RelativePosition ↔ GridCoord
// ---------------------------------------------------------------------------

impl GridCoord {
    /// Split into a GPU-friendly [`RelativePosition`]: integer cell index +
    /// f32 fractional offset.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // intentional f64→i32/f32 narrowing
    pub fn to_relative(&self) -> RelativePosition {
        let fx = self.x.floor();
        let fy = self.y.floor();
        RelativePosition {
            cell_x: fx as i32,
            cell_y: fy as i32,
            frac_x: (self.x - fx) as f32,
            frac_y: (self.y - fy) as f32,
        }
    }
}

impl RelativePosition {
    /// Reconstruct a full-precision [`GridCoord`] from integer cell +
    /// fractional offset.
    #[must_use]
    pub fn to_grid_coord(self) -> GridCoord {
        GridCoord {
            x: f64::from(self.cell_x) + f64::from(self.frac_x),
            y: f64::from(self.cell_y) + f64::from(self.frac_y),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// 1° global grid starting at (−180°E, −90°N).
    fn global_grid() -> GridDomain {
        GridDomain {
            xlon0: -180.0,
            ylat0: -90.0,
            dx: 1.0,
            dy: 1.0,
            nx: 360,
            ny: 181,
        }
    }

    // -- geo ↔ grid ---------------------------------------------------------

    #[test]
    fn geo_to_grid_at_origin() {
        let grid = global_grid();
        let gc = geo_to_grid(
            GeoCoord {
                lat: -90.0,
                lon: -180.0,
            },
            &grid,
        );
        assert_relative_eq!(gc.x, 0.0, epsilon = 1e-12);
        assert_relative_eq!(gc.y, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn geo_to_grid_paris() {
        let grid = global_grid();
        let paris = GeoCoord {
            lat: 48.8566,
            lon: 2.3522,
        };
        let gc = geo_to_grid(paris, &grid);
        assert_relative_eq!(gc.x, 182.3522, epsilon = 1e-10);
        assert_relative_eq!(gc.y, 138.8566, epsilon = 1e-10);
    }

    #[test]
    fn roundtrip_geo_grid_geo() {
        let grid = global_grid();
        let original = GeoCoord {
            lat: 48.8566,
            lon: 2.3522,
        };
        let recovered = grid_to_geo(geo_to_grid(original, &grid), &grid);
        assert_relative_eq!(recovered.lat, original.lat, epsilon = 1e-12);
        assert_relative_eq!(recovered.lon, original.lon, epsilon = 1e-12);
    }

    #[test]
    fn roundtrip_multiple_cities() {
        let grid = global_grid();
        let cities = [
            GeoCoord { lat: 0.0, lon: 0.0 },
            GeoCoord {
                lat: -33.8688,
                lon: 151.2093,
            }, // Sydney
            GeoCoord {
                lat: 35.6762,
                lon: 139.6503,
            }, // Tokyo
            GeoCoord {
                lat: -22.9068,
                lon: -43.1729,
            }, // Rio de Janeiro
            GeoCoord {
                lat: 64.1466,
                lon: -21.9426,
            }, // Reykjavik
        ];
        for &geo in &cities {
            let recovered = grid_to_geo(geo_to_grid(geo, &grid), &grid);
            assert_relative_eq!(recovered.lat, geo.lat, epsilon = 1e-12);
            assert_relative_eq!(recovered.lon, geo.lon, epsilon = 1e-12);
        }
    }

    #[test]
    fn roundtrip_fractional_grid() {
        let grid = GridDomain {
            xlon0: -10.0,
            ylat0: 35.0,
            dx: 0.25,
            dy: 0.25,
            nx: 200,
            ny: 160,
        };
        let geo = GeoCoord {
            lat: 48.123,
            lon: 2.456,
        };
        let recovered = grid_to_geo(geo_to_grid(geo, &grid), &grid);
        assert_relative_eq!(recovered.lat, geo.lat, epsilon = 1e-12);
        assert_relative_eq!(recovered.lon, geo.lon, epsilon = 1e-12);
    }

    // -- distance_meters ----------------------------------------------------

    #[test]
    fn distance_equator_one_degree_lon() {
        let a = GeoCoord { lat: 0.0, lon: 0.0 };
        let b = GeoCoord { lat: 0.0, lon: 1.0 };
        let d = distance_meters(a, b);
        // 1° at equator ≈ 111 195 m (with R_EARTH = 6.371e6)
        assert!(
            (d - 111_195.0).abs() < 100.0,
            "equatorial 1° lon: expected ~111 195 m, got {d}"
        );
    }

    #[test]
    fn distance_equator_one_degree_lat() {
        let a = GeoCoord { lat: 0.0, lon: 0.0 };
        let b = GeoCoord { lat: 1.0, lon: 0.0 };
        let d = distance_meters(a, b);
        assert!(
            (d - 111_195.0).abs() < 100.0,
            "equatorial 1° lat: expected ~111 195 m, got {d}"
        );
    }

    #[test]
    fn distance_same_point_is_zero() {
        let a = GeoCoord {
            lat: 48.0,
            lon: 2.0,
        };
        let d = distance_meters(a, a);
        assert!(d.abs() < 1e-3, "same point should be ~0 m, got {d}");
    }

    #[test]
    fn distance_symmetric() {
        let a = GeoCoord {
            lat: 48.8566,
            lon: 2.3522,
        };
        let b = GeoCoord {
            lat: 40.7128,
            lon: -74.0060,
        };
        let d_ab = distance_meters(a, b);
        let d_ba = distance_meters(b, a);
        assert!(
            (d_ab - d_ba).abs() < 1.0,
            "distance should be symmetric: {d_ab} vs {d_ba}"
        );
    }

    #[test]
    fn distance_paris_new_york() {
        let paris = GeoCoord {
            lat: 48.8566,
            lon: 2.3522,
        };
        let nyc = GeoCoord {
            lat: 40.7128,
            lon: -74.0060,
        };
        let d = distance_meters(paris, nyc);
        // Known great-circle distance ≈ 5 837 km
        assert!(
            (d - 5_837_000.0).abs() < 50_000.0,
            "Paris–NYC: expected ~5 837 km, got {} km",
            d / 1000.0
        );
    }

    // -- dx_meters_at_latitude ----------------------------------------------

    #[test]
    fn dx_meters_equator() {
        let dx = dx_meters_at_latitude(0.0);
        assert!(
            (dx - 111_195.0).abs() < 100.0,
            "equatorial dx: expected ~111 195 m/°, got {dx}"
        );
    }

    #[test]
    fn dx_meters_60n_half_of_equator() {
        let dx_eq = dx_meters_at_latitude(0.0);
        let dx_60 = dx_meters_at_latitude(60.0);
        // cos(60°) = 0.5 exactly
        let ratio = dx_60 / dx_eq;
        assert!(
            (ratio - 0.5).abs() < 0.01,
            "dx ratio 60°N / equator: expected ~0.5, got {ratio}"
        );
    }

    #[test]
    fn dx_meters_pole_near_zero() {
        let dx = dx_meters_at_latitude(90.0);
        assert!(dx.abs() < 1.0, "dx at pole should be ~0, got {dx}");
    }

    #[test]
    fn dx_meters_45n() {
        let dx_eq = dx_meters_at_latitude(0.0);
        let dx_45 = dx_meters_at_latitude(45.0);
        // cos(45°) ≈ 0.7071
        let ratio = dx_45 / dx_eq;
        assert!(
            (ratio - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.01,
            "dx ratio 45°N / equator: expected ~0.707, got {ratio}"
        );
    }

    // -- RelativePosition ---------------------------------------------------

    #[test]
    fn relative_position_size() {
        assert_eq!(std::mem::size_of::<RelativePosition>(), 16);
    }

    #[test]
    fn relative_position_roundtrip() {
        let gc = GridCoord {
            x: 182.3522,
            y: 138.8566,
        };
        let rel = gc.to_relative();
        assert_eq!(rel.cell_x, 182);
        assert_eq!(rel.cell_y, 138);
        assert!((rel.frac_x - 0.3522).abs() < 1e-4);
        assert!((rel.frac_y - 0.8566).abs() < 1e-4);

        let recovered = rel.to_grid_coord();
        // f32 fractional part limits round-trip precision to ~1e-4
        assert!((recovered.x - gc.x).abs() < 1e-3);
        assert!((recovered.y - gc.y).abs() < 1e-3);
    }

    #[test]
    fn relative_position_negative_coords() {
        let gc = GridCoord { x: -0.5, y: -0.3 };
        let rel = gc.to_relative();
        assert_eq!(rel.cell_x, -1);
        assert_eq!(rel.cell_y, -1);

        let recovered = rel.to_grid_coord();
        assert!((recovered.x - gc.x).abs() < 1e-3);
        assert!((recovered.y - gc.y).abs() < 1e-3);
    }

    #[test]
    fn relative_position_exact_integer() {
        let gc = GridCoord { x: 100.0, y: 50.0 };
        let rel = gc.to_relative();
        assert_eq!(rel.cell_x, 100);
        assert_eq!(rel.cell_y, 50);
        assert_eq!(rel.frac_x, 0.0);
        assert_eq!(rel.frac_y, 0.0);
    }

    #[test]
    fn relative_position_bytemuck_roundtrip() {
        let rel = RelativePosition {
            cell_x: 42,
            cell_y: 99,
            frac_x: 0.75,
            frac_y: 0.25,
        };
        let bytes: &[u8] = bytemuck::bytes_of(&rel);
        let recovered: &RelativePosition = bytemuck::from_bytes(bytes);
        assert_eq!(recovered.cell_x, 42);
        assert_eq!(recovered.cell_y, 99);
        assert_eq!(recovered.frac_x, 0.75);
        assert_eq!(recovered.frac_y, 0.25);
    }

    // -- GridDomain::contains -----------------------------------------------

    #[test]
    fn grid_domain_contains() {
        let grid = global_grid();
        assert!(grid.contains(&GridCoord { x: 100.0, y: 90.0 }));
        assert!(grid.contains(&GridCoord { x: 0.0, y: 0.0 }));
        assert!(!grid.contains(&GridCoord { x: 360.0, y: 90.0 }));
        assert!(!grid.contains(&GridCoord { x: -1.0, y: 90.0 }));
        assert!(!grid.contains(&GridCoord { x: 100.0, y: 181.0 }));
    }
}
