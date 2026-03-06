//! NetCDF meteorological input adapter (IO-02 MVP).
//!
//! Port scope: alternative to the GRIB2 ERA5 path from `readwind_ecmwf.f90`.
//! This module reads the core 3-D meteorological variables and materializes:
//! - [`crate::wind::WindFieldGrid`]
//! - [`crate::wind::WindField3D`]
//!
//! MVP assumptions:
//! - Supported variable layout is either `(time, level, latitude, longitude)` or
//!   `(level, latitude, longitude)`.
//! - Horizontal coordinates (`latitude`, `longitude`) are strictly monotonic and
//!   approximately uniformly spaced.
//! - Vertical levels are used as placeholder `heights_m` in `WindFieldGrid`.
//! - `pressure_pa` is populated from level values only when
//!   `assume_pressure_levels` is enabled (default). If max |level| <= 2000,
//!   levels are interpreted as hPa and converted to Pa.
//!
//! This keeps IO-02 testable and deterministic while leaving room for richer
//! conventions (CF metadata handling, hybrid levels, staggered dimensions) in a
//! later revision.
#[cfg(feature = "netcdf")]
use std::cmp;
use std::path::{Path, PathBuf};

#[cfg(feature = "netcdf")]
use ndarray::Array1;
#[cfg(feature = "netcdf")]
use ndarray::Array3;
use thiserror::Error;

use crate::wind::{WindField3D, WindFieldGrid};

#[cfg(feature = "netcdf")]
const NETCDF_MVP_VARIABLE_COUNT: usize = 5;
const DEFAULT_PRESSURE_HPA_UPPER_BOUND: f32 = 2_000.0;
#[cfg(feature = "netcdf")]
const COORD_UNIFORM_REL_TOL: f32 = 1.0e-4;

/// Metadata for one NetCDF meteorological snapshot assembled by IO-02.
#[derive(Debug, Clone, PartialEq)]
pub struct NetcdfMvpSnapshotMetadata {
    /// Source file path used to load this snapshot.
    pub source_path: PathBuf,
    /// Selected time index in the NetCDF time dimension (if present).
    pub time_index: usize,
    /// Name of the level coordinate variable.
    pub level_variable: String,
    /// Name of the latitude coordinate variable.
    pub latitude_variable: String,
    /// Name of the longitude coordinate variable.
    pub longitude_variable: String,
    /// Raw level values as read from the NetCDF coordinate variable.
    pub levels: Vec<f32>,
    /// Whether the loader interpreted levels as pressure levels.
    pub assumed_pressure_levels: bool,
    /// Scaling factor applied to level values when filling `pressure_pa`.
    /// Typical values are 100.0 (hPa -> Pa) or 1.0 (already Pa).
    pub pressure_level_scale_to_pa: f32,
}

/// Result object returned by the NetCDF MVP loader.
#[derive(Debug, Clone)]
pub struct NetcdfMvpSnapshot {
    pub grid: WindFieldGrid,
    pub field: WindField3D,
    pub metadata: NetcdfMvpSnapshotMetadata,
}

/// Variable names used by the NetCDF MVP loader for core 3-D fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetcdfMvpVariableNames {
    pub u: String,
    pub v: String,
    pub w: String,
    pub t: String,
    pub q: String,
}

impl Default for NetcdfMvpVariableNames {
    fn default() -> Self {
        Self {
            u: "u".to_string(),
            v: "v".to_string(),
            w: "w".to_string(),
            t: "t".to_string(),
            q: "q".to_string(),
        }
    }
}

/// Options for loading a single meteorological snapshot from NetCDF.
#[derive(Debug, Clone, PartialEq)]
pub struct NetcdfMvpReadOptions {
    /// Names of core 3-D meteorological variables.
    pub variables: NetcdfMvpVariableNames,
    /// Name of the vertical level coordinate variable.
    pub level_variable: String,
    /// Name of the latitude coordinate variable.
    pub latitude_variable: String,
    /// Name of the longitude coordinate variable.
    pub longitude_variable: String,
    /// Time index to select when 4-D variables include a time dimension.
    pub time_index: usize,
    /// If `true`, fill `pressure_pa` from level values.
    pub assume_pressure_levels: bool,
    /// If max absolute level value is below this threshold, levels are assumed
    /// to be in hPa and converted to Pa with a factor of 100.
    pub pressure_hpa_upper_bound: f32,
}

impl Default for NetcdfMvpReadOptions {
    fn default() -> Self {
        Self {
            variables: NetcdfMvpVariableNames::default(),
            level_variable: "level".to_string(),
            latitude_variable: "latitude".to_string(),
            longitude_variable: "longitude".to_string(),
            time_index: 0,
            assume_pressure_levels: true,
            pressure_hpa_upper_bound: DEFAULT_PRESSURE_HPA_UPPER_BOUND,
        }
    }
}

#[derive(Debug, Error)]
pub enum NetcdfReaderError {
    #[error("the `netcdf` cargo feature is disabled; cannot decode NetCDF file `{path}`")]
    FeatureDisabled { path: PathBuf },
    #[error("failed to open NetCDF file `{path}`: {message}")]
    OpenFailed { path: PathBuf, message: String },
    #[error("required variable `{name}` is missing from NetCDF file `{path}`")]
    MissingVariable { path: PathBuf, name: String },
    #[error("unsupported rank for variable `{variable}` in `{path}`: expected 3 or 4 dimensions, got {rank}")]
    UnsupportedVariableRank {
        path: PathBuf,
        variable: String,
        rank: usize,
    },
    #[error("invalid time index {time_index} for variable `{variable}` in `{path}` with time dimension length {time_len}")]
    InvalidTimeIndex {
        path: PathBuf,
        variable: String,
        time_index: usize,
        time_len: usize,
    },
    #[error("shape mismatch for variable `{variable}` in `{path}`: expected ({expected_nx}, {expected_ny}, {expected_nz}, {expected_time_desc}), got ({actual_nx}, {actual_ny}, {actual_nz}, {actual_time_desc})")]
    VariableShapeMismatch {
        path: PathBuf,
        variable: String,
        expected_nx: usize,
        expected_ny: usize,
        expected_nz: usize,
        expected_time_desc: String,
        actual_nx: usize,
        actual_ny: usize,
        actual_nz: usize,
        actual_time_desc: String,
    },
    #[error("invalid value count for variable `{variable}` in `{path}`: expected {expected}, got {actual}")]
    InvalidValueCount {
        path: PathBuf,
        variable: String,
        expected: usize,
        actual: usize,
    },
    #[error("coordinate `{coordinate}` in `{path}` has invalid rank {rank}; expected rank 1")]
    InvalidCoordinateRank {
        path: PathBuf,
        coordinate: String,
        rank: usize,
    },
    #[error("coordinate `{coordinate}` in `{path}` is empty")]
    EmptyCoordinate { path: PathBuf, coordinate: String },
    #[error(
        "coordinate `{coordinate}` in `{path}` contains non-finite value at index {index}: {value}"
    )]
    NonFiniteCoordinate {
        path: PathBuf,
        coordinate: String,
        index: usize,
        value: f32,
    },
    #[error("coordinate `{coordinate}` in `{path}` must be strictly monotonic")]
    NonMonotonicCoordinate { path: PathBuf, coordinate: String },
    #[error("coordinate `{coordinate}` in `{path}` must be approximately uniformly spaced; first delta={first_delta}, delta at index {index}={delta}")]
    NonUniformCoordinateSpacing {
        path: PathBuf,
        coordinate: String,
        first_delta: f32,
        index: usize,
        delta: f32,
    },
    #[error(
        "coordinate `{coordinate}` length mismatch in `{path}`: expected {expected}, got {actual}"
    )]
    CoordinateLengthMismatch {
        path: PathBuf,
        coordinate: String,
        expected: usize,
        actual: usize,
    },
    #[error("invalid reader options: {message}")]
    InvalidOptions { message: String },
    #[error("failed to decode variable `{variable}` in `{path}`: {message}")]
    DecodeFailed {
        path: PathBuf,
        variable: String,
        message: String,
    },
}

#[cfg(feature = "netcdf")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MvpVariable {
    U,
    V,
    W,
    T,
    Q,
}

#[cfg(feature = "netcdf")]
impl MvpVariable {
    fn all() -> [Self; NETCDF_MVP_VARIABLE_COUNT] {
        [Self::U, Self::V, Self::W, Self::T, Self::Q]
    }

    fn name(self, names: &NetcdfMvpVariableNames) -> &str {
        match self {
            Self::U => &names.u,
            Self::V => &names.v,
            Self::W => &names.w,
            Self::T => &names.t,
            Self::Q => &names.q,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::U => "u",
            Self::V => "v",
            Self::W => "w",
            Self::T => "t",
            Self::Q => "q",
        }
    }
}

#[cfg(feature = "netcdf")]
#[derive(Debug, Clone, PartialEq)]
struct DecodedVariable3D {
    nx: usize,
    ny: usize,
    nz: usize,
    time_len: Option<usize>,
    values: Vec<f32>,
}

#[cfg(feature = "netcdf")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FieldShape {
    nx: usize,
    ny: usize,
    nz: usize,
    time_len: Option<usize>,
}

#[cfg(feature = "netcdf")]
#[derive(Debug, Clone, PartialEq)]
struct AxisInfo {
    ascending_values: Vec<f32>,
    spacing_abs: f32,
    reversed: bool,
}

/// Load one core meteorological snapshot from NetCDF with default conventions.
pub fn load_mvp_from_netcdf(path: &Path) -> Result<NetcdfMvpSnapshot, NetcdfReaderError> {
    load_mvp_from_netcdf_with_options(path, &NetcdfMvpReadOptions::default())
}

/// Load one core meteorological snapshot from NetCDF with custom conventions.
pub fn load_mvp_from_netcdf_with_options(
    path: &Path,
    options: &NetcdfMvpReadOptions,
) -> Result<NetcdfMvpSnapshot, NetcdfReaderError> {
    validate_options(options)?;

    #[cfg(feature = "netcdf")]
    {
        let file = netcdf::open(path).map_err(|error| NetcdfReaderError::OpenFailed {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
        build_snapshot_from_file(path, &file, options)
    }

    #[cfg(not(feature = "netcdf"))]
    {
        let _ = options;
        Err(NetcdfReaderError::FeatureDisabled {
            path: path.to_path_buf(),
        })
    }
}

fn validate_options(options: &NetcdfMvpReadOptions) -> Result<(), NetcdfReaderError> {
    let validate_non_empty = |label: &'static str, value: &str| {
        if value.trim().is_empty() {
            Err(NetcdfReaderError::InvalidOptions {
                message: format!("`{label}` cannot be empty"),
            })
        } else {
            Ok(())
        }
    };

    validate_non_empty("variables.u", &options.variables.u)?;
    validate_non_empty("variables.v", &options.variables.v)?;
    validate_non_empty("variables.w", &options.variables.w)?;
    validate_non_empty("variables.t", &options.variables.t)?;
    validate_non_empty("variables.q", &options.variables.q)?;
    validate_non_empty("level_variable", &options.level_variable)?;
    validate_non_empty("latitude_variable", &options.latitude_variable)?;
    validate_non_empty("longitude_variable", &options.longitude_variable)?;
    if !options.pressure_hpa_upper_bound.is_finite() || options.pressure_hpa_upper_bound <= 0.0 {
        return Err(NetcdfReaderError::InvalidOptions {
            message: "`pressure_hpa_upper_bound` must be finite and > 0".to_string(),
        });
    }
    Ok(())
}

#[cfg(feature = "netcdf")]
fn build_snapshot_from_file(
    path: &Path,
    file: &netcdf::File,
    options: &NetcdfMvpReadOptions,
) -> Result<NetcdfMvpSnapshot, NetcdfReaderError> {
    let longitude_values = read_coordinate_values(path, file, &options.longitude_variable)?;
    let latitude_values = read_coordinate_values(path, file, &options.latitude_variable)?;
    let level_values = read_coordinate_values(path, file, &options.level_variable)?;

    let lon_axis = analyze_axis(path, &options.longitude_variable, &longitude_values)?;
    let lat_axis = analyze_axis(path, &options.latitude_variable, &latitude_values)?;

    let mut decoded = Vec::with_capacity(NETCDF_MVP_VARIABLE_COUNT);
    for variable in MvpVariable::all() {
        decoded.push((
            variable,
            read_variable_3d_at_time(
                path,
                file,
                variable.name(&options.variables),
                options.time_index,
            )?,
        ));
    }

    let (first_var, first_decoded) =
        decoded
            .first()
            .ok_or_else(|| NetcdfReaderError::InvalidOptions {
                message: "internal IO-02 invariant: expected at least one variable".to_string(),
            })?;
    let expected = FieldShape {
        nx: first_decoded.nx,
        ny: first_decoded.ny,
        nz: first_decoded.nz,
        time_len: first_decoded.time_len,
    };

    for (variable, entry) in &decoded[1..] {
        let current = FieldShape {
            nx: entry.nx,
            ny: entry.ny,
            nz: entry.nz,
            time_len: entry.time_len,
        };
        if current != expected {
            return Err(NetcdfReaderError::VariableShapeMismatch {
                path: path.to_path_buf(),
                variable: variable.name(&options.variables).to_string(),
                expected_nx: expected.nx,
                expected_ny: expected.ny,
                expected_nz: expected.nz,
                expected_time_desc: match expected.time_len {
                    Some(time_len) => format!("time={time_len}"),
                    None => "no-time".to_string(),
                },
                actual_nx: current.nx,
                actual_ny: current.ny,
                actual_nz: current.nz,
                actual_time_desc: match current.time_len {
                    Some(time_len) => format!("time={time_len}"),
                    None => "no-time".to_string(),
                },
            });
        }
    }

    if lon_axis.ascending_values.len() != expected.nx {
        return Err(NetcdfReaderError::CoordinateLengthMismatch {
            path: path.to_path_buf(),
            coordinate: options.longitude_variable.clone(),
            expected: expected.nx,
            actual: lon_axis.ascending_values.len(),
        });
    }
    if lat_axis.ascending_values.len() != expected.ny {
        return Err(NetcdfReaderError::CoordinateLengthMismatch {
            path: path.to_path_buf(),
            coordinate: options.latitude_variable.clone(),
            expected: expected.ny,
            actual: lat_axis.ascending_values.len(),
        });
    }
    if level_values.len() != expected.nz {
        return Err(NetcdfReaderError::CoordinateLengthMismatch {
            path: path.to_path_buf(),
            coordinate: options.level_variable.clone(),
            expected: expected.nz,
            actual: level_values.len(),
        });
    }

    let mut field = WindField3D::zeros(expected.nx, expected.ny, expected.nz);
    for (variable, entry) in &decoded {
        write_reoriented(
            path,
            variable.label(),
            &entry.values,
            expected.nx,
            expected.ny,
            expected.nz,
            lon_axis.reversed,
            lat_axis.reversed,
            match variable {
                MvpVariable::U => &mut field.u_ms,
                MvpVariable::V => &mut field.v_ms,
                MvpVariable::W => &mut field.w_ms,
                MvpVariable::T => &mut field.temperature_k,
                MvpVariable::Q => &mut field.specific_humidity,
            },
        )?;
    }

    let (assumed_pressure_levels, pressure_level_scale_to_pa) =
        fill_pressure_from_levels(&mut field, &level_values, options);
    let heights_m = Array1::from_vec(level_values.clone());
    let grid = WindFieldGrid::new(
        expected.nx,
        expected.ny,
        expected.nz,
        expected.nz,
        expected.nz,
        lon_axis.spacing_abs,
        lat_axis.spacing_abs,
        lon_axis.ascending_values[0],
        lat_axis.ascending_values[0],
        heights_m,
    );

    Ok(NetcdfMvpSnapshot {
        grid,
        field,
        metadata: NetcdfMvpSnapshotMetadata {
            source_path: path.to_path_buf(),
            time_index: options.time_index,
            level_variable: options.level_variable.clone(),
            latitude_variable: options.latitude_variable.clone(),
            longitude_variable: options.longitude_variable.clone(),
            levels: level_values,
            assumed_pressure_levels,
            pressure_level_scale_to_pa,
        },
    })
}

#[cfg(feature = "netcdf")]
fn read_coordinate_values(
    path: &Path,
    file: &netcdf::File,
    coordinate_name: &str,
) -> Result<Vec<f32>, NetcdfReaderError> {
    let variable =
        file.variable(coordinate_name)
            .ok_or_else(|| NetcdfReaderError::MissingVariable {
                path: path.to_path_buf(),
                name: coordinate_name.to_string(),
            })?;
    let dimensions = variable.dimensions();
    if dimensions.len() != 1 {
        return Err(NetcdfReaderError::InvalidCoordinateRank {
            path: path.to_path_buf(),
            coordinate: coordinate_name.to_string(),
            rank: dimensions.len(),
        });
    }
    let values =
        variable
            .get_values::<f32, _>(..)
            .map_err(|error| NetcdfReaderError::DecodeFailed {
                path: path.to_path_buf(),
                variable: coordinate_name.to_string(),
                message: error.to_string(),
            })?;
    if values.is_empty() {
        return Err(NetcdfReaderError::EmptyCoordinate {
            path: path.to_path_buf(),
            coordinate: coordinate_name.to_string(),
        });
    }

    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(NetcdfReaderError::NonFiniteCoordinate {
                path: path.to_path_buf(),
                coordinate: coordinate_name.to_string(),
                index,
                value,
            });
        }
    }

    Ok(values)
}

#[cfg(feature = "netcdf")]
fn analyze_axis(
    path: &Path,
    axis_name: &str,
    values: &[f32],
) -> Result<AxisInfo, NetcdfReaderError> {
    if values.is_empty() {
        return Err(NetcdfReaderError::EmptyCoordinate {
            path: path.to_path_buf(),
            coordinate: axis_name.to_string(),
        });
    }
    if values.len() == 1 {
        return Ok(AxisInfo {
            ascending_values: vec![values[0]],
            spacing_abs: 0.0,
            reversed: false,
        });
    }

    let first_delta = values[1] - values[0];
    if first_delta.abs() <= f32::EPSILON {
        return Err(NetcdfReaderError::NonMonotonicCoordinate {
            path: path.to_path_buf(),
            coordinate: axis_name.to_string(),
        });
    }
    let direction_sign = first_delta.signum();

    for idx in 2..values.len() {
        let delta = values[idx] - values[idx - 1];
        if delta.abs() <= f32::EPSILON || delta.signum() != direction_sign {
            return Err(NetcdfReaderError::NonMonotonicCoordinate {
                path: path.to_path_buf(),
                coordinate: axis_name.to_string(),
            });
        }

        let scale = cmp::max_by(first_delta.abs(), delta.abs(), |a, b| {
            a.partial_cmp(b).unwrap_or(cmp::Ordering::Equal)
        })
        .max(1.0);
        if (delta - first_delta).abs() > COORD_UNIFORM_REL_TOL * scale {
            return Err(NetcdfReaderError::NonUniformCoordinateSpacing {
                path: path.to_path_buf(),
                coordinate: axis_name.to_string(),
                first_delta,
                index: idx - 1,
                delta,
            });
        }
    }

    let mut ascending = values.to_vec();
    let reversed = first_delta < 0.0;
    if reversed {
        ascending.reverse();
    }
    Ok(AxisInfo {
        ascending_values: ascending,
        spacing_abs: first_delta.abs(),
        reversed,
    })
}

#[cfg(feature = "netcdf")]
fn read_variable_3d_at_time(
    path: &Path,
    file: &netcdf::File,
    variable_name: &str,
    time_index: usize,
) -> Result<DecodedVariable3D, NetcdfReaderError> {
    let variable =
        file.variable(variable_name)
            .ok_or_else(|| NetcdfReaderError::MissingVariable {
                path: path.to_path_buf(),
                name: variable_name.to_string(),
            })?;
    let dimensions = variable.dimensions();
    let (nx, ny, nz, time_len) = match dimensions.len() {
        4 => (
            dimensions[3].len(),
            dimensions[2].len(),
            dimensions[1].len(),
            Some(dimensions[0].len()),
        ),
        3 => (
            dimensions[2].len(),
            dimensions[1].len(),
            dimensions[0].len(),
            None,
        ),
        rank => {
            return Err(NetcdfReaderError::UnsupportedVariableRank {
                path: path.to_path_buf(),
                variable: variable_name.to_string(),
                rank,
            });
        }
    };

    if let Some(time_len) = time_len {
        if time_index >= time_len {
            return Err(NetcdfReaderError::InvalidTimeIndex {
                path: path.to_path_buf(),
                variable: variable_name.to_string(),
                time_index,
                time_len,
            });
        }
    }

    let values_all =
        variable
            .get_values::<f32, _>(..)
            .map_err(|error| NetcdfReaderError::DecodeFailed {
                path: path.to_path_buf(),
                variable: variable_name.to_string(),
                message: error.to_string(),
            })?;
    let elements_per_time = nx * ny * nz;
    let expected = elements_per_time * time_len.unwrap_or(1);
    if values_all.len() != expected {
        return Err(NetcdfReaderError::InvalidValueCount {
            path: path.to_path_buf(),
            variable: variable_name.to_string(),
            expected,
            actual: values_all.len(),
        });
    }

    let values = if let Some(_) = time_len {
        let start = time_index * elements_per_time;
        values_all[start..start + elements_per_time].to_vec()
    } else {
        values_all
    };

    Ok(DecodedVariable3D {
        nx,
        ny,
        nz,
        time_len,
        values,
    })
}

#[cfg(feature = "netcdf")]
#[allow(clippy::too_many_arguments)]
fn write_reoriented(
    path: &Path,
    variable_name: &'static str,
    source_values: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    reverse_lon: bool,
    reverse_lat: bool,
    target: &mut Array3<f32>,
) -> Result<(), NetcdfReaderError> {
    let expected = nx * ny * nz;
    if source_values.len() != expected {
        return Err(NetcdfReaderError::InvalidValueCount {
            path: path.to_path_buf(),
            variable: variable_name.to_string(),
            expected,
            actual: source_values.len(),
        });
    }

    for k in 0..nz {
        for j in 0..ny {
            let src_j = if reverse_lat { ny - 1 - j } else { j };
            for i in 0..nx {
                let src_i = if reverse_lon { nx - 1 - i } else { i };
                let source_index = (k * ny + src_j) * nx + src_i;
                target[[i, j, k]] = source_values[source_index];
            }
        }
    }
    Ok(())
}

#[cfg(feature = "netcdf")]
fn fill_pressure_from_levels(
    field: &mut WindField3D,
    levels: &[f32],
    options: &NetcdfMvpReadOptions,
) -> (bool, f32) {
    if !options.assume_pressure_levels || levels.is_empty() {
        return (false, 0.0);
    }

    let max_abs = levels.iter().fold(
        0.0_f32,
        |acc, &value| if value.abs() > acc { value.abs() } else { acc },
    );
    let scale = if max_abs <= options.pressure_hpa_upper_bound {
        100.0
    } else {
        1.0
    };

    let (nx, ny, nz) = field.shape();
    for k in 0..nz {
        let pressure_level = levels[k] * scale;
        for i in 0..nx {
            for j in 0..ny {
                field.pressure_pa[[i, j, k]] = pressure_level;
            }
        }
    }
    (true, scale)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "netcdf")]
    use std::fs;
    #[cfg(feature = "netcdf")]
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[cfg(not(feature = "netcdf"))]
    #[test]
    fn netcdf_loader_reports_feature_disabled_without_netcdf_feature() {
        let error = load_mvp_from_netcdf(Path::new("/tmp/mock-met.nc"))
            .expect_err("loader should fail without netcdf feature");
        assert!(matches!(error, NetcdfReaderError::FeatureDisabled { .. }));
    }

    #[cfg(feature = "netcdf")]
    fn temp_netcdf_path(test_name: &str) -> PathBuf {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be >= UNIX_EPOCH")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "flexpart-gpu-io02-{test_name}-{}-{ts}.nc",
            std::process::id()
        ))
    }

    #[cfg(feature = "netcdf")]
    fn create_test_file(path: &Path, with_q: bool) {
        let mut file = netcdf::create(path).expect("must create temporary netcdf");
        file.add_dimension("time", 2).expect("time dimension");
        file.add_dimension("level", 2).expect("level dimension");
        file.add_dimension("latitude", 2)
            .expect("latitude dimension");
        file.add_dimension("longitude", 3)
            .expect("longitude dimension");

        let mut lon = file
            .add_variable::<f32>("longitude", &["longitude"])
            .expect("longitude variable");
        lon.put_values(&[1.0, 1.5, 2.0], ..)
            .expect("longitude values");

        // Descending latitude; loader should flip to ascending internally.
        let mut lat = file
            .add_variable::<f32>("latitude", &["latitude"])
            .expect("latitude variable");
        lat.put_values(&[50.0, 49.5], ..).expect("latitude values");

        let mut level = file
            .add_variable::<f32>("level", &["level"])
            .expect("level variable");
        level
            .put_values(&[1000.0, 850.0], ..)
            .expect("level values");

        write_4d_field(&mut file, "u", 1.0);
        write_4d_field(&mut file, "v", 2.0);
        write_4d_field(&mut file, "w", 3.0);
        write_4d_field(&mut file, "t", 4.0);
        if with_q {
            write_4d_field(&mut file, "q", 5.0);
        }
    }

    #[cfg(feature = "netcdf")]
    fn write_4d_field(file: &mut netcdf::FileMut, name: &str, base: f32) {
        let mut var = file
            .add_variable::<f32>(name, &["time", "level", "latitude", "longitude"])
            .expect("must add 4d variable");
        let mut values = Vec::new();
        for t in 0..2 {
            for k in 0..2 {
                for j in 0..2 {
                    for i in 0..3 {
                        let value = base
                            + (t as f32) * 100.0
                            + (k as f32) * 10.0
                            + (j as f32) * 3.0
                            + (i as f32);
                        values.push(value);
                    }
                }
            }
        }
        var.put_values(&values, ..)
            .expect("must write field values");
    }

    #[cfg(feature = "netcdf")]
    #[test]
    fn load_snapshot_from_generated_netcdf_file() {
        let path = temp_netcdf_path("happy-path");
        create_test_file(&path, true);
        let snapshot = load_mvp_from_netcdf(&path).expect("mock netcdf should load");

        assert_eq!(snapshot.grid.nx, 3);
        assert_eq!(snapshot.grid.ny, 2);
        assert_eq!(snapshot.grid.nz, 2);
        assert!((snapshot.grid.dx_deg - 0.5).abs() < 1.0e-6);
        assert!((snapshot.grid.dy_deg - 0.5).abs() < 1.0e-6);
        assert!((snapshot.grid.xlon0 - 1.0).abs() < 1.0e-6);
        assert!((snapshot.grid.ylat0 - 49.5).abs() < 1.0e-6);
        assert_eq!(snapshot.metadata.levels, vec![1000.0, 850.0]);
        assert!(snapshot.metadata.assumed_pressure_levels);
        assert!((snapshot.metadata.pressure_level_scale_to_pa - 100.0).abs() < 1.0e-6);
        assert!((snapshot.field.pressure_pa[[0, 0, 0]] - 100_000.0).abs() < 1.0e-3);
        assert!((snapshot.field.pressure_pa[[0, 0, 1]] - 85_000.0).abs() < 1.0e-3);

        // time_index=0 by default; latitude should be flipped to ascending order.
        // src indices: k=0, src_j=1 (flipped), i=0 -> value = 1 + 0 + 0 + 3 + 0 = 4.
        assert!((snapshot.field.u_ms[[0, 0, 0]] - 4.0).abs() < 1.0e-6);
        // Same location, variable t with base=4 -> value=7.
        assert!((snapshot.field.temperature_k[[0, 0, 0]] - 7.0).abs() < 1.0e-6);

        fs::remove_file(&path).expect("temporary netcdf must be removable");
    }

    #[cfg(feature = "netcdf")]
    #[test]
    fn load_snapshot_fails_when_required_variable_is_missing() {
        let path = temp_netcdf_path("missing-q");
        create_test_file(&path, false);

        let error = load_mvp_from_netcdf(&path).expect_err("missing q must fail");
        assert!(matches!(
            error,
            NetcdfReaderError::MissingVariable { ref name, .. } if name == "q"
        ));

        fs::remove_file(&path).expect("temporary netcdf must be removable");
    }

    #[cfg(feature = "netcdf")]
    #[test]
    fn load_snapshot_fails_when_time_index_is_out_of_bounds() {
        let path = temp_netcdf_path("time-index-oob");
        create_test_file(&path, true);
        let mut options = NetcdfMvpReadOptions::default();
        options.time_index = 9;

        let error = load_mvp_from_netcdf_with_options(&path, &options)
            .expect_err("time index out of range must fail");
        assert!(matches!(error, NetcdfReaderError::InvalidTimeIndex { .. }));

        fs::remove_file(&path).expect("temporary netcdf must be removable");
    }
}
