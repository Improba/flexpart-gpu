/// Wind field data structures (3D grids: u, v, w, T, qv + grid metadata).
///
/// Ported from `com_mod.f90` (uu, vv, ww, tt, qv arrays and grid parameters)
/// and `par_mod.f90` (grid dimension parameters).
///
/// The Fortran code stores all fields with a trailing `numwfmem=2` dimension
/// for temporal interpolation between two time levels. In this Rust port,
/// [`WindFieldPair`] holds two snapshots explicitly instead.
///
/// All fields use `f32` to match the Fortran `real` (single precision) and
/// to align with GPU f32 compute. CPU-side storage uses `ndarray`; GPU buffer
/// layout is handled separately by the buffer manager (S-07).
///
/// The `synthetic` submodule provides analytical wind generators (uniform flow,
/// linear shear, Rankine vortex) used to validate advection tasks A-07/A-08.
use ndarray::{Array1, Array2, Array3};

pub mod synthetic;

pub use synthetic::{
    linear_shear_wind_field, rankine_vortex_wind_field, uniform_wind_field, RankineVortexConfig,
    SyntheticFieldBackground,
};

// ---------------------------------------------------------------------------
// Grid metadata
// ---------------------------------------------------------------------------

/// Spatial grid definition for the wind field domain.
///
/// Corresponds to Fortran `com_mod.f90` lines 294-320:
/// `nx, ny, nz, dx, dy, xlon0, ylat0, height(nzmax)`.
#[derive(Debug, Clone)]
pub struct WindFieldGrid {
    /// Number of grid points in the x (longitude) direction.
    pub nx: usize,
    /// Number of grid points in the y (latitude) direction.
    pub ny: usize,
    /// Number of vertical levels in transformed coordinates.
    pub nz: usize,
    /// Number of vertical levels for u/v on original eta levels.
    pub nuvz: usize,
    /// Number of vertical levels for w (staggered grid).
    pub nwz: usize,
    /// Grid spacing in x (longitude) direction [degrees].
    pub dx_deg: f32,
    /// Grid spacing in y (latitude) direction [degrees].
    pub dy_deg: f32,
    /// Longitude of the lower-left grid corner [degrees].
    pub xlon0: f32,
    /// Latitude of the lower-left grid corner [degrees].
    pub ylat0: f32,
    /// Heights of each vertical level [m], length = `nz`.
    pub heights_m: Array1<f32>,
}

impl WindFieldGrid {
    /// Create a new grid definition.
    ///
    /// # Panics
    /// Panics if `heights_m.len() != nz`.
    #[must_use]
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        nuvz: usize,
        nwz: usize,
        dx_deg: f32,
        dy_deg: f32,
        xlon0: f32,
        ylat0: f32,
        heights_m: Array1<f32>,
    ) -> Self {
        assert_eq!(
            heights_m.len(),
            nz,
            "heights_m length ({}) must equal nz ({})",
            heights_m.len(),
            nz,
        );
        Self {
            nx,
            ny,
            nz,
            nuvz,
            nwz,
            dx_deg,
            dy_deg,
            xlon0,
            ylat0,
            heights_m,
        }
    }
}

// ---------------------------------------------------------------------------
// Vertical coordinates
// ---------------------------------------------------------------------------

/// Hybrid sigma-pressure coordinate coefficients.
///
/// Corresponds to Fortran `com_mod.f90` lines 328-336:
/// ```text
/// akm(nwzmax), bkm(nwzmax)   — layer boundaries
/// akz(nuvzmax), bkz(nuvzmax) — layer centres
/// aknew(nzmax), bknew(nzmax) — interpolated levels
/// ```
///
/// Pressure at a level is computed as:  p = ak + bk × p_surface.
#[derive(Debug, Clone)]
pub struct VerticalCoordinates {
    /// A-coefficient at layer boundaries [Pa], length = `nwz`.
    pub akm_pa: Array1<f32>,
    /// B-coefficient at layer boundaries [dimensionless], length = `nwz`.
    pub bkm: Array1<f32>,
    /// A-coefficient at layer centres [Pa], length = `nuvz`.
    pub akz_pa: Array1<f32>,
    /// B-coefficient at layer centres [dimensionless], length = `nuvz`.
    pub bkz: Array1<f32>,
    /// A-coefficient at interpolated output levels [Pa], length = `nz`.
    pub aknew_pa: Array1<f32>,
    /// B-coefficient at interpolated output levels [dimensionless], length = `nz`.
    pub bknew: Array1<f32>,
}

// ---------------------------------------------------------------------------
// 3-D meteorological fields (one time level)
// ---------------------------------------------------------------------------

/// One time-level snapshot of 3-D meteorological fields.
///
/// Each array is dimensioned `(nx, ny, nz)` — matching the Fortran layout
/// `(0:nxmax-1, 0:nymax-1, nzmax)` for a single `numwfmem` slot.
///
/// Fortran source: `com_mod.f90` lines 355-400.
#[derive(Debug, Clone)]
pub struct WindField3D {
    /// U-component of wind [m/s].  Fortran: `uu`.
    pub u_ms: Array3<f32>,
    /// V-component of wind [m/s].  Fortran: `vv`.
    pub v_ms: Array3<f32>,
    /// W-component of wind [m/s].  Fortran: `ww`.
    pub w_ms: Array3<f32>,
    /// Temperature [K].  Fortran: `tt`.
    pub temperature_k: Array3<f32>,
    /// Specific humidity [kg/kg].  Fortran: `qv`.
    pub specific_humidity: Array3<f32>,
    /// Air pressure [Pa].  Fortran: `prs`.
    pub pressure_pa: Array3<f32>,
    /// Air density [kg/m³].  Fortran: `rho`.
    pub air_density_kg_m3: Array3<f32>,
    /// Vertical air density gradient [kg/m²].  Fortran: `drhodz`.
    pub density_gradient_kg_m2: Array3<f32>,
}

impl WindField3D {
    /// Allocate a zero-initialised set of 3-D fields for the given grid shape.
    #[must_use]
    pub fn zeros(nx: usize, ny: usize, nz: usize) -> Self {
        let shape = (nx, ny, nz);
        Self {
            u_ms: Array3::zeros(shape),
            v_ms: Array3::zeros(shape),
            w_ms: Array3::zeros(shape),
            temperature_k: Array3::zeros(shape),
            specific_humidity: Array3::zeros(shape),
            pressure_pa: Array3::zeros(shape),
            air_density_kg_m3: Array3::zeros(shape),
            density_gradient_kg_m2: Array3::zeros(shape),
        }
    }

    /// Shape `(nx, ny, nz)` of the contained arrays.
    #[must_use]
    pub fn shape(&self) -> (usize, usize, usize) {
        let s = self.u_ms.shape();
        (s[0], s[1], s[2])
    }
}

// ---------------------------------------------------------------------------
// 2-D surface fields (one time level)
// ---------------------------------------------------------------------------

/// One time-level snapshot of 2-D surface / boundary-layer fields.
///
/// Each array is dimensioned `(nx, ny)` — corresponding to the Fortran
/// `(0:nxmax-1, 0:nymax-1, 1, numwfmem)` layout with the singleton z-dim
/// removed.
///
/// Fortran source: `com_mod.f90` lines 407-450.
#[derive(Debug, Clone)]
pub struct SurfaceFields {
    /// Surface pressure [Pa].  Fortran: `ps`.
    pub surface_pressure_pa: Array2<f32>,
    /// 10-metre U wind component [m/s].  Fortran: `u10`.
    pub u10_ms: Array2<f32>,
    /// 10-metre V wind component [m/s].  Fortran: `v10`.
    pub v10_ms: Array2<f32>,
    /// 2-metre temperature [K].  Fortran: `tt2`.
    pub temperature_2m_k: Array2<f32>,
    /// 2-metre dew-point temperature [K].  Fortran: `td2`.
    pub dewpoint_2m_k: Array2<f32>,
    /// Large-scale precipitation rate [mm/h].  Fortran: `lsprec`.
    pub precip_large_scale_mm_h: Array2<f32>,
    /// Convective precipitation rate [mm/h].  Fortran: `convprec`.
    pub precip_convective_mm_h: Array2<f32>,
    /// Surface sensible heat flux [W/m²].  Fortran: `sshf`.
    pub sensible_heat_flux_w_m2: Array2<f32>,
    /// Surface solar radiation [W/m²].  Fortran: `ssr`.
    pub solar_radiation_w_m2: Array2<f32>,
    /// Surface stress [N/m²].  Fortran: `surfstr`.
    pub surface_stress_n_m2: Array2<f32>,
    /// Friction velocity [m/s].  Fortran: `ustar`.
    pub friction_velocity_ms: Array2<f32>,
    /// Convective velocity scale [m/s].  Fortran: `wstar`.
    pub convective_velocity_scale_ms: Array2<f32>,
    /// Mixing layer height [m].  Fortran: `hmix`.
    pub mixing_height_m: Array2<f32>,
    /// Tropopause height [m].  Fortran: `tropopause`.
    pub tropopause_height_m: Array2<f32>,
    /// Inverse Obukhov length [1/m].  Fortran: `oli`.
    pub inv_obukhov_length_per_m: Array2<f32>,
}

impl SurfaceFields {
    /// Allocate a zero-initialised set of 2-D surface fields.
    #[must_use]
    pub fn zeros(nx: usize, ny: usize) -> Self {
        let shape = (nx, ny);
        Self {
            surface_pressure_pa: Array2::zeros(shape),
            u10_ms: Array2::zeros(shape),
            v10_ms: Array2::zeros(shape),
            temperature_2m_k: Array2::zeros(shape),
            dewpoint_2m_k: Array2::zeros(shape),
            precip_large_scale_mm_h: Array2::zeros(shape),
            precip_convective_mm_h: Array2::zeros(shape),
            sensible_heat_flux_w_m2: Array2::zeros(shape),
            solar_radiation_w_m2: Array2::zeros(shape),
            surface_stress_n_m2: Array2::zeros(shape),
            friction_velocity_ms: Array2::zeros(shape),
            convective_velocity_scale_ms: Array2::zeros(shape),
            mixing_height_m: Array2::zeros(shape),
            tropopause_height_m: Array2::zeros(shape),
            inv_obukhov_length_per_m: Array2::zeros(shape),
        }
    }
}

// ---------------------------------------------------------------------------
// Time-invariant (static) fields
// ---------------------------------------------------------------------------

/// Time-invariant geographical fields.
///
/// Fortran source: `com_mod.f90` lines 340-350.
#[derive(Debug, Clone)]
pub struct StaticFields {
    /// Orography / surface elevation [m].  Fortran: `oro`.
    pub orography_m: Array2<f32>,
    /// Land-sea mask [0..1].  Fortran: `lsm`.
    pub land_sea_mask: Array2<f32>,
}

impl StaticFields {
    /// Allocate zero-initialised static fields.
    #[must_use]
    pub fn zeros(nx: usize, ny: usize) -> Self {
        Self {
            orography_m: Array2::zeros((nx, ny)),
            land_sea_mask: Array2::zeros((nx, ny)),
        }
    }
}

// ---------------------------------------------------------------------------
// Temporal pair for interpolation
// ---------------------------------------------------------------------------

/// Two time levels of wind and surface data for temporal interpolation.
///
/// Replaces the Fortran `numwfmem=2` trailing dimension on every field array.
/// The advection kernel linearly interpolates between `t0` and `t1` based on
/// the current simulation time.
#[derive(Debug, Clone)]
pub struct WindFieldPair {
    /// 3-D fields at the earlier time level.
    pub field_t0: WindField3D,
    /// 3-D fields at the later time level.
    pub field_t1: WindField3D,
    /// 2-D surface fields at the earlier time level.
    pub surface_t0: SurfaceFields,
    /// 2-D surface fields at the later time level.
    pub surface_t1: SurfaceFields,
}

impl WindFieldPair {
    /// Allocate a zero-initialised pair of time levels.
    #[must_use]
    pub fn zeros(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            field_t0: WindField3D::zeros(nx, ny, nz),
            field_t1: WindField3D::zeros(nx, ny, nz),
            surface_t0: SurfaceFields::zeros(nx, ny),
            surface_t1: SurfaceFields::zeros(nx, ny),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn wind_field_grid_construction() {
        let nz = 10;
        let heights = Array1::linspace(0.0, 20_000.0, nz);
        let grid = WindFieldGrid::new(
            361,    // nx — typical 1° global
            181,    // ny
            nz,     // nz
            138,    // nuvz
            138,    // nwz
            1.0,    // dx [deg]
            1.0,    // dy [deg]
            -180.0, // xlon0
            -90.0,  // ylat0
            heights.clone(),
        );

        assert_eq!(grid.nx, 361);
        assert_eq!(grid.ny, 181);
        assert_eq!(grid.nz, nz);
        assert_eq!(grid.heights_m.len(), nz);
        assert!((grid.dx_deg - 1.0).abs() < f32::EPSILON);
        assert!((grid.xlon0 - (-180.0)).abs() < f32::EPSILON);
    }

    #[test]
    #[should_panic(expected = "heights_m length")]
    fn wind_field_grid_rejects_mismatched_heights() {
        let heights = Array1::zeros(5);
        let _ = WindFieldGrid::new(10, 10, 20, 20, 20, 1.0, 1.0, 0.0, 0.0, heights);
    }

    #[test]
    fn wind_field_3d_construction() {
        let (nx, ny, nz) = (36, 18, 10);
        let field = WindField3D::zeros(nx, ny, nz);

        assert_eq!(field.shape(), (nx, ny, nz));
        assert_eq!(field.u_ms.shape(), &[nx, ny, nz]);
        assert_eq!(field.temperature_k[[0, 0, 0]], 0.0);
    }

    #[test]
    fn surface_fields_construction() {
        let (nx, ny) = (36, 18);
        let sf = SurfaceFields::zeros(nx, ny);

        assert_eq!(sf.surface_pressure_pa.shape(), &[nx, ny]);
        assert_eq!(sf.mixing_height_m[[0, 0]], 0.0);
    }

    #[test]
    fn static_fields_construction() {
        let sf = StaticFields::zeros(10, 10);
        assert_eq!(sf.orography_m.shape(), &[10, 10]);
        assert_eq!(sf.land_sea_mask.shape(), &[10, 10]);
    }

    #[test]
    fn wind_field_pair_holds_two_levels() {
        let (nx, ny, nz) = (36, 18, 10);
        let pair = WindFieldPair::zeros(nx, ny, nz);

        assert_eq!(pair.field_t0.shape(), (nx, ny, nz));
        assert_eq!(pair.field_t1.shape(), (nx, ny, nz));
        assert_eq!(pair.surface_t0.surface_pressure_pa.shape(), &[nx, ny]);
        assert_eq!(pair.surface_t1.surface_pressure_pa.shape(), &[nx, ny]);
    }
}
