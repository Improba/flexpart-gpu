//! GRIB2 meteorological input adapter (ERA5 MVP subset).
//!
//! Port target: `readwind_ecmwf.f90` (field extraction stage only).
//! This module intentionally implements a minimal and testable extraction path:
//! - read ERA5 GRIB2 fields `u`, `v`, `w`, `t`, `q`
//! - validate grid/level consistency
//! - materialize `WindFieldGrid` + `WindField3D`
//!
//! For hybrid/sigma inputs, geometric-height conversion is provided by
//! `crate::io::vertical_transform` and can be applied once A/B coefficients and
//! surface pressure fields are available from the input stream.
//! Temporal interpolation between snapshots is provided in `crate::io::temporal`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ndarray::Array1;
use thiserror::Error;

use crate::wind::{WindField3D, WindFieldGrid};

const ERA5_MVP_VARIABLE_COUNT: usize = 5;

#[derive(Debug, Clone, PartialEq)]
pub struct Era5GribGridMetadata {
    pub nx: usize,
    pub ny: usize,
    pub dx_deg: f32,
    pub dy_deg: f32,
    pub xlon0_deg: f32,
    pub ylat0_deg: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Era5MvpSnapshotMetadata {
    pub data_date: i64,
    pub data_time_hhmm: i64,
    pub step_hours: i64,
    pub type_of_level: String,
    pub levels: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Era5GribRecord {
    pub short_name: String,
    pub level: i64,
    pub values: Vec<f32>,
    pub grid: Era5GribGridMetadata,
    pub data_date: i64,
    pub data_time_hhmm: i64,
    pub step_hours: i64,
    pub type_of_level: String,
}

#[derive(Debug, Clone)]
pub struct Era5MvpSnapshot {
    pub grid: WindFieldGrid,
    pub field: WindField3D,
    pub metadata: Era5MvpSnapshotMetadata,
}

#[derive(Debug, Error)]
pub enum Grib2ReaderError {
    #[error("the `eccodes` cargo feature is disabled; cannot decode GRIB2 file `{path}`")]
    FeatureDisabled { path: PathBuf },
    #[error("failed to open GRIB2 file `{path}` with ecCodes: {message}")]
    OpenFailed { path: PathBuf, message: String },
    #[error("failed to decode GRIB2 file `{path}` with ecCodes: {message}")]
    DecodeFailed { path: PathBuf, message: String },
    #[error("no ERA5 MVP records were found in the input")]
    EmptyInput,
    #[error("required ERA5 MVP variable `{variable}` is missing")]
    MissingVariable { variable: &'static str },
    #[error("duplicate record for variable `{variable}` at level `{level}`")]
    DuplicateRecord { variable: String, level: i64 },
    #[error("level set mismatch for variable `{variable}`: expected {expected:?}, got {actual:?}")]
    LevelSetMismatch {
        variable: &'static str,
        expected: Vec<i64>,
        actual: Vec<i64>,
    },
    #[error("missing level `{level}` for variable `{variable}` while assembling 3-D field")]
    MissingLevel { variable: &'static str, level: i64 },
    #[error(
        "grid metadata mismatch for variable `{variable}` at level `{level}`: expected {expected}, got {actual}"
    )]
    InconsistentGrid {
        variable: String,
        level: i64,
        expected: String,
        actual: String,
    },
    #[error(
        "message metadata mismatch for variable `{variable}` at level `{level}`: expected {expected}, got {actual}"
    )]
    InconsistentMetadata {
        variable: String,
        level: i64,
        expected: String,
        actual: String,
    },
    #[error(
        "invalid value count for variable `{variable}` at level `{level}`: expected {expected}, got {actual}"
    )]
    InvalidValueCount {
        variable: String,
        level: i64,
        expected: usize,
        actual: usize,
    },
    #[error("invalid metadata: {message}")]
    InvalidMetadata { message: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Era5Variable {
    U,
    V,
    W,
    T,
    Q,
}

impl Era5Variable {
    fn from_short_name(short_name: &str) -> Option<Self> {
        match short_name {
            "u" => Some(Self::U),
            "v" => Some(Self::V),
            "w" => Some(Self::W),
            "t" => Some(Self::T),
            "q" => Some(Self::Q),
            _ => None,
        }
    }

    fn short_name(self) -> &'static str {
        match self {
            Self::U => "u",
            Self::V => "v",
            Self::W => "w",
            Self::T => "t",
            Self::Q => "q",
        }
    }

    fn all() -> [Self; ERA5_MVP_VARIABLE_COUNT] {
        [Self::U, Self::V, Self::W, Self::T, Self::Q]
    }
}

/// Load ERA5 GRIB2 meteorological fields using ecCodes (if enabled).
///
/// When the `eccodes` Cargo feature is disabled, this function returns
/// [`Grib2ReaderError::FeatureDisabled`]. This keeps binary builds portable on
/// machines where ecCodes linkage is not available.
pub fn load_era5_mvp_from_grib2(path: &Path) -> Result<Era5MvpSnapshot, Grib2ReaderError> {
    let records = read_records_from_grib2(path)?;
    build_era5_mvp_from_records(records)
}

/// Build the ERA5 MVP snapshot from decoded GRIB records.
///
/// This adapter layer is intentionally public so tests and alternative I/O
/// backends can inject mocked records without requiring native ecCodes linkage.
pub fn build_era5_mvp_from_records(
    records: Vec<Era5GribRecord>,
) -> Result<Era5MvpSnapshot, Grib2ReaderError> {
    let mut fields_by_var: BTreeMap<Era5Variable, BTreeMap<i64, Vec<f32>>> = BTreeMap::new();
    let mut reference_grid: Option<Era5GribGridMetadata> = None;
    let mut reference_meta: Option<(i64, i64, i64, String)> = None;

    for record in records {
        let Some(variable) = Era5Variable::from_short_name(&record.short_name) else {
            continue;
        };

        match &reference_grid {
            None => reference_grid = Some(record.grid.clone()),
            Some(grid) if *grid == record.grid => {}
            Some(grid) => {
                return Err(Grib2ReaderError::InconsistentGrid {
                    variable: record.short_name,
                    level: record.level,
                    expected: format!("{grid:?}"),
                    actual: format!("{:?}", record.grid),
                });
            }
        }

        let record_meta = (
            record.data_date,
            record.data_time_hhmm,
            record.step_hours,
            record.type_of_level.clone(),
        );
        match &reference_meta {
            None => reference_meta = Some(record_meta),
            Some(meta) if *meta == record_meta => {}
            Some(meta) => {
                return Err(Grib2ReaderError::InconsistentMetadata {
                    variable: record.short_name,
                    level: record.level,
                    expected: format!("{meta:?}"),
                    actual: format!("{record_meta:?}"),
                });
            }
        }

        let previous = fields_by_var
            .entry(variable)
            .or_default()
            .insert(record.level, record.values);
        if previous.is_some() {
            return Err(Grib2ReaderError::DuplicateRecord {
                variable: variable.short_name().to_string(),
                level: record.level,
            });
        }
    }

    let grid = reference_grid.ok_or(Grib2ReaderError::EmptyInput)?;
    let (data_date, data_time_hhmm, step_hours, type_of_level) =
        reference_meta.ok_or(Grib2ReaderError::EmptyInput)?;

    for variable in Era5Variable::all() {
        if !fields_by_var.contains_key(&variable) {
            return Err(Grib2ReaderError::MissingVariable {
                variable: variable.short_name(),
            });
        }
    }

    let expected_levels = levels_for(&fields_by_var, Era5Variable::U)?;
    for variable in [
        Era5Variable::V,
        Era5Variable::W,
        Era5Variable::T,
        Era5Variable::Q,
    ] {
        let actual_levels = levels_for(&fields_by_var, variable)?;
        if actual_levels != expected_levels {
            return Err(Grib2ReaderError::LevelSetMismatch {
                variable: variable.short_name(),
                expected: expected_levels.clone(),
                actual: actual_levels,
            });
        }
    }

    let nz = expected_levels.len();
    if nz == 0 {
        return Err(Grib2ReaderError::InvalidMetadata {
            message: "no vertical levels detected in ERA5 MVP records".to_string(),
        });
    }

    // Isobaric MVP fallback: keep level IDs as placeholder heights.
    // For hybrid/sigma, call `transform_hybrid_sigma_to_height` with A/B + ps.
    #[allow(clippy::cast_precision_loss)]
    let placeholder_heights_m =
        Array1::from_iter(expected_levels.iter().map(|level| *level as f32));
    let wind_grid = WindFieldGrid::new(
        grid.nx,
        grid.ny,
        nz,
        nz,
        nz,
        grid.dx_deg,
        grid.dy_deg,
        grid.xlon0_deg,
        grid.ylat0_deg,
        placeholder_heights_m,
    );

    let mut wind_field = WindField3D::zeros(grid.nx, grid.ny, nz);
    let level_maps = build_level_map_lookup(&fields_by_var)?;

    for (k, level) in expected_levels.iter().copied().enumerate() {
        write_level(
            &mut wind_field.u_ms,
            level_maps.u.get(&level).map(Vec::as_slice),
            grid.nx,
            grid.ny,
            k,
            "u",
            level,
        )?;
        write_level(
            &mut wind_field.v_ms,
            level_maps.v.get(&level).map(Vec::as_slice),
            grid.nx,
            grid.ny,
            k,
            "v",
            level,
        )?;
        write_level(
            &mut wind_field.w_ms,
            level_maps.w.get(&level).map(Vec::as_slice),
            grid.nx,
            grid.ny,
            k,
            "w",
            level,
        )?;
        write_level(
            &mut wind_field.temperature_k,
            level_maps.t.get(&level).map(Vec::as_slice),
            grid.nx,
            grid.ny,
            k,
            "t",
            level,
        )?;
        write_level(
            &mut wind_field.specific_humidity,
            level_maps.q.get(&level).map(Vec::as_slice),
            grid.nx,
            grid.ny,
            k,
            "q",
            level,
        )?;

        if let Some(pressure_pa) = maybe_pressure_pa(&type_of_level, level) {
            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    wind_field.pressure_pa[[i, j, k]] = pressure_pa;
                }
            }
        }
    }

    Ok(Era5MvpSnapshot {
        grid: wind_grid,
        field: wind_field,
        metadata: Era5MvpSnapshotMetadata {
            data_date,
            data_time_hhmm,
            step_hours,
            type_of_level,
            levels: expected_levels,
        },
    })
}

fn maybe_pressure_pa(type_of_level: &str, level: i64) -> Option<f32> {
    if type_of_level == "isobaricInhPa" {
        #[allow(clippy::cast_precision_loss)]
        let level_hpa = level as f32;
        Some(level_hpa * 100.0)
    } else {
        None
    }
}

fn levels_for(
    fields_by_var: &BTreeMap<Era5Variable, BTreeMap<i64, Vec<f32>>>,
    variable: Era5Variable,
) -> Result<Vec<i64>, Grib2ReaderError> {
    let levels = fields_by_var
        .get(&variable)
        .ok_or(Grib2ReaderError::MissingVariable {
            variable: variable.short_name(),
        })?
        .keys()
        .copied()
        .collect::<Vec<_>>();
    Ok(levels)
}

struct LevelMapLookup<'a> {
    u: &'a BTreeMap<i64, Vec<f32>>,
    v: &'a BTreeMap<i64, Vec<f32>>,
    w: &'a BTreeMap<i64, Vec<f32>>,
    t: &'a BTreeMap<i64, Vec<f32>>,
    q: &'a BTreeMap<i64, Vec<f32>>,
}

fn build_level_map_lookup(
    fields_by_var: &BTreeMap<Era5Variable, BTreeMap<i64, Vec<f32>>>,
) -> Result<LevelMapLookup<'_>, Grib2ReaderError> {
    Ok(LevelMapLookup {
        u: fields_by_var
            .get(&Era5Variable::U)
            .ok_or(Grib2ReaderError::MissingVariable { variable: "u" })?,
        v: fields_by_var
            .get(&Era5Variable::V)
            .ok_or(Grib2ReaderError::MissingVariable { variable: "v" })?,
        w: fields_by_var
            .get(&Era5Variable::W)
            .ok_or(Grib2ReaderError::MissingVariable { variable: "w" })?,
        t: fields_by_var
            .get(&Era5Variable::T)
            .ok_or(Grib2ReaderError::MissingVariable { variable: "t" })?,
        q: fields_by_var
            .get(&Era5Variable::Q)
            .ok_or(Grib2ReaderError::MissingVariable { variable: "q" })?,
    })
}

fn write_level(
    target: &mut ndarray::Array3<f32>,
    values: Option<&[f32]>,
    nx: usize,
    ny: usize,
    k: usize,
    variable: &'static str,
    level: i64,
) -> Result<(), Grib2ReaderError> {
    let values = values.ok_or(Grib2ReaderError::MissingLevel { variable, level })?;
    let expected = nx * ny;
    if values.len() != expected {
        return Err(Grib2ReaderError::InvalidValueCount {
            variable: variable.to_string(),
            level,
            expected,
            actual: values.len(),
        });
    }

    for j in 0..ny {
        for i in 0..nx {
            target[[i, j, k]] = values[j * nx + i];
        }
    }
    Ok(())
}

#[cfg(feature = "eccodes")]
fn read_records_from_grib2(path: &Path) -> Result<Vec<Era5GribRecord>, Grib2ReaderError> {
    use eccodes::{CodesFile, FallibleIterator, KeyRead, ProductKind};

    let mut file = CodesFile::new_from_file(path, ProductKind::GRIB).map_err(|error| {
        Grib2ReaderError::OpenFailed {
            path: path.to_path_buf(),
            message: error.to_string(),
        }
    })?;

    let mut records = Vec::new();
    let mut messages = file.ref_message_iter();
    while let Some(message) = messages
        .next()
        .map_err(|error| Grib2ReaderError::DecodeFailed {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?
    {
        let short_name: String =
            message
                .read_key("shortName")
                .map_err(|error| Grib2ReaderError::DecodeFailed {
                    path: path.to_path_buf(),
                    message: format!("failed reading key `shortName`: {error}"),
                })?;
        if Era5Variable::from_short_name(&short_name).is_none() {
            continue;
        }

        let level: i64 =
            message
                .read_key("level")
                .map_err(|error| Grib2ReaderError::DecodeFailed {
                    path: path.to_path_buf(),
                    message: format!("failed reading key `level`: {error}"),
                })?;
        let values_f64: Vec<f64> =
            message
                .read_key("values")
                .map_err(|error| Grib2ReaderError::DecodeFailed {
                    path: path.to_path_buf(),
                    message: format!("failed reading key `values`: {error}"),
                })?;
        #[allow(clippy::cast_possible_truncation)]
        let values = values_f64.into_iter().map(|value| value as f32).collect();

        let nx_i64: i64 =
            message
                .read_key("Ni")
                .map_err(|error| Grib2ReaderError::DecodeFailed {
                    path: path.to_path_buf(),
                    message: format!("failed reading key `Ni`: {error}"),
                })?;
        let ny_i64: i64 =
            message
                .read_key("Nj")
                .map_err(|error| Grib2ReaderError::DecodeFailed {
                    path: path.to_path_buf(),
                    message: format!("failed reading key `Nj`: {error}"),
                })?;
        let nx = usize::try_from(nx_i64).map_err(|_| Grib2ReaderError::InvalidMetadata {
            message: format!("`Ni` must be non-negative and fit into usize, got {nx_i64}"),
        })?;
        let ny = usize::try_from(ny_i64).map_err(|_| Grib2ReaderError::InvalidMetadata {
            message: format!("`Nj` must be non-negative and fit into usize, got {ny_i64}"),
        })?;

        let dx_deg: f64 = message
            .read_key("iDirectionIncrementInDegrees")
            .map_err(|error| Grib2ReaderError::DecodeFailed {
                path: path.to_path_buf(),
                message: format!("failed reading key `iDirectionIncrementInDegrees`: {error}"),
            })?;
        let dy_deg: f64 = message
            .read_key("jDirectionIncrementInDegrees")
            .map_err(|error| Grib2ReaderError::DecodeFailed {
                path: path.to_path_buf(),
                message: format!("failed reading key `jDirectionIncrementInDegrees`: {error}"),
            })?;
        let xlon0_deg: f64 = message
            .read_key("longitudeOfFirstGridPointInDegrees")
            .map_err(|error| Grib2ReaderError::DecodeFailed {
                path: path.to_path_buf(),
                message: format!(
                    "failed reading key `longitudeOfFirstGridPointInDegrees`: {error}"
                ),
            })?;
        let ylat0_deg: f64 = message
            .read_key("latitudeOfFirstGridPointInDegrees")
            .map_err(|error| Grib2ReaderError::DecodeFailed {
                path: path.to_path_buf(),
                message: format!("failed reading key `latitudeOfFirstGridPointInDegrees`: {error}"),
            })?;

        let data_date: i64 =
            message
                .read_key("dataDate")
                .map_err(|error| Grib2ReaderError::DecodeFailed {
                    path: path.to_path_buf(),
                    message: format!("failed reading key `dataDate`: {error}"),
                })?;
        let data_time_hhmm: i64 =
            message
                .read_key("dataTime")
                .map_err(|error| Grib2ReaderError::DecodeFailed {
                    path: path.to_path_buf(),
                    message: format!("failed reading key `dataTime`: {error}"),
                })?;
        let type_of_level: String =
            message
                .read_key("typeOfLevel")
                .map_err(|error| Grib2ReaderError::DecodeFailed {
                    path: path.to_path_buf(),
                    message: format!("failed reading key `typeOfLevel`: {error}"),
                })?;

        let step_hours: i64 = message
            .read_key("forecastTime")
            .or_else(|_| message.read_key("step"))
            .unwrap_or(0);

        records.push(Era5GribRecord {
            short_name,
            level,
            values,
            grid: Era5GribGridMetadata {
                nx,
                ny,
                #[allow(clippy::cast_possible_truncation)]
                dx_deg: dx_deg as f32,
                #[allow(clippy::cast_possible_truncation)]
                dy_deg: dy_deg as f32,
                #[allow(clippy::cast_possible_truncation)]
                xlon0_deg: xlon0_deg as f32,
                #[allow(clippy::cast_possible_truncation)]
                ylat0_deg: ylat0_deg as f32,
            },
            data_date,
            data_time_hhmm,
            step_hours,
            type_of_level,
        });
    }

    if records.is_empty() {
        return Err(Grib2ReaderError::EmptyInput);
    }
    Ok(records)
}

#[cfg(not(feature = "eccodes"))]
fn read_records_from_grib2(path: &Path) -> Result<Vec<Era5GribRecord>, Grib2ReaderError> {
    Err(Grib2ReaderError::FeatureDisabled {
        path: path.to_path_buf(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(
        short_name: &str,
        level: i64,
        nx: usize,
        ny: usize,
        seed: f32,
    ) -> Era5GribRecord {
        let mut values = Vec::with_capacity(nx * ny);
        for idx in 0..(nx * ny) {
            #[allow(clippy::cast_precision_loss)]
            let idx_f32 = idx as f32;
            values.push(seed + idx_f32);
        }

        Era5GribRecord {
            short_name: short_name.to_string(),
            level,
            values,
            grid: Era5GribGridMetadata {
                nx,
                ny,
                dx_deg: 0.25,
                dy_deg: 0.25,
                xlon0_deg: -12.0,
                ylat0_deg: 35.0,
            },
            data_date: 20260305,
            data_time_hhmm: 0,
            step_hours: 0,
            type_of_level: "isobaricInhPa".to_string(),
        }
    }

    fn full_record_set(nx: usize, ny: usize) -> Vec<Era5GribRecord> {
        let mut records = Vec::new();
        let levels = [850_i64, 1000_i64];
        for level in levels {
            records.push(make_record("u", level, nx, ny, 10.0 + level as f32));
            records.push(make_record("v", level, nx, ny, 20.0 + level as f32));
            records.push(make_record("w", level, nx, ny, 30.0 + level as f32));
            records.push(make_record("t", level, nx, ny, 40.0 + level as f32));
            records.push(make_record("q", level, nx, ny, 50.0 + level as f32));
        }
        records
    }

    #[test]
    fn build_snapshot_from_mocked_records() {
        let records = full_record_set(3, 2);
        let snapshot =
            build_era5_mvp_from_records(records).expect("mocked ERA5 records should assemble");

        assert_eq!(snapshot.grid.nx, 3);
        assert_eq!(snapshot.grid.ny, 2);
        assert_eq!(snapshot.grid.nz, 2);
        assert_eq!(snapshot.field.shape(), (3, 2, 2));
        assert_eq!(snapshot.metadata.type_of_level, "isobaricInhPa");
        assert_eq!(snapshot.metadata.levels, vec![850, 1000]);
        assert_eq!(snapshot.field.pressure_pa[[0, 0, 0]], 85_000.0);
        assert_eq!(snapshot.field.pressure_pa[[0, 0, 1]], 100_000.0);

        assert!((snapshot.field.u_ms[[1, 0, 0]] - 861.0).abs() < 1.0e-6);
        assert!((snapshot.field.v_ms[[2, 1, 1]] - 1_025.0).abs() < 1.0e-6);
        assert!((snapshot.field.temperature_k[[0, 1, 0]] - 893.0).abs() < 1.0e-6);
        assert!((snapshot.field.specific_humidity[[0, 0, 1]] - 1_050.0).abs() < 1.0e-6);
    }

    #[test]
    fn build_snapshot_fails_when_required_variable_is_missing() {
        let mut records = full_record_set(2, 2);
        records.retain(|record| record.short_name != "q");

        let error = build_era5_mvp_from_records(records)
            .expect_err("missing required variable should fail");
        assert!(matches!(
            error,
            Grib2ReaderError::MissingVariable { variable: "q" }
        ));
    }

    #[test]
    fn build_snapshot_fails_when_variable_levels_differ() {
        let mut records = full_record_set(2, 2);
        records.retain(|record| !(record.short_name == "w" && record.level == 850));

        let error = build_era5_mvp_from_records(records)
            .expect_err("inconsistent variable levels should fail");
        assert!(matches!(
            error,
            Grib2ReaderError::LevelSetMismatch { variable: "w", .. }
        ));
    }

    #[cfg(not(feature = "eccodes"))]
    #[test]
    fn grib_file_loader_reports_feature_disabled_without_eccodes() {
        let error = load_era5_mvp_from_grib2(Path::new("/tmp/mock-era5.grib"))
            .expect_err("loader should fail without eccodes feature");
        assert!(matches!(error, Grib2ReaderError::FeatureDisabled { .. }));
    }
}
