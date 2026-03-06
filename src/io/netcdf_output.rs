//! NetCDF concentration/deposition output writer (I-03).
//!
//! Ported from FLEXPART output responsibilities in `concoutput.f90`.
//!
//! MVP assumptions:
//! - one output snapshot per file (single `time` index),
//! - concentration is stored as accumulated species mass per cell `[kg]`,
//! - dry and wet deposition are 2-D horizontal grids `[kg m-2]`.

use std::path::Path;

use thiserror::Error;

use crate::gpu::{ConcentrationGridOutput, ConcentrationGridShape};

/// Horizontal deposition-grid shape `(nx, ny)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepositionGridShape {
    pub nx: usize,
    pub ny: usize,
}

impl DepositionGridShape {
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.nx * self.ny
    }
}

/// Dry/wet deposition outputs on the horizontal output grid.
#[derive(Debug, Clone)]
pub struct DepositionGridOutput {
    /// Horizontal shape used for flattening.
    pub shape: DepositionGridShape,
    /// Dry deposition field `[kg m-2]`, row-major `(x, y)`.
    pub dry_deposition_kg_m2: Vec<f32>,
    /// Wet deposition field `[kg m-2]`, row-major `(x, y)`.
    pub wet_deposition_kg_m2: Vec<f32>,
}

/// Metadata associated with one gridded output snapshot.
#[derive(Debug, Clone, Copy)]
pub struct GriddedOutputMetadata {
    /// Simulation time represented by this snapshot [s since Unix epoch].
    pub simulation_time_seconds: i64,
    /// Species slot index that was accumulated into concentration.
    pub species_index: usize,
}

/// Complete concentration/deposition snapshot written to one NetCDF file.
#[derive(Debug, Clone)]
pub struct GriddedOutputSnapshot {
    pub metadata: GriddedOutputMetadata,
    pub concentration: ConcentrationGridOutput,
    pub deposition: DepositionGridOutput,
}

/// Errors returned by NetCDF output writing.
#[derive(Debug, Error)]
pub enum NetcdfOutputError {
    #[error("concentration shape must be non-zero, got {shape:?}")]
    ZeroConcentrationShape { shape: (usize, usize, usize) },
    #[error("deposition shape must be non-zero, got {shape:?}")]
    ZeroDepositionShape { shape: (usize, usize) },
    #[error("concentration/deposition horizontal shape mismatch: concentration=({cnx},{cny}), deposition=({dnx},{dny})")]
    HorizontalShapeMismatch {
        cnx: usize,
        cny: usize,
        dnx: usize,
        dny: usize,
    },
    #[error("length mismatch for `{field}`: expected {expected}, got {actual}")]
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("species index does not fit i32: {value}")]
    SpeciesIndexTooLarge { value: usize },
    #[error("netcdf operation failed: {0}")]
    Netcdf(#[from] netcdf::Error),
}

fn validate_snapshot(snapshot: &GriddedOutputSnapshot) -> Result<(), NetcdfOutputError> {
    let ConcentrationGridShape { nx, ny, nz } = snapshot.concentration.shape;
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(NetcdfOutputError::ZeroConcentrationShape {
            shape: (nx, ny, nz),
        });
    }
    let dshape = snapshot.deposition.shape;
    if dshape.nx == 0 || dshape.ny == 0 {
        return Err(NetcdfOutputError::ZeroDepositionShape {
            shape: (dshape.nx, dshape.ny),
        });
    }
    if nx != dshape.nx || ny != dshape.ny {
        return Err(NetcdfOutputError::HorizontalShapeMismatch {
            cnx: nx,
            cny: ny,
            dnx: dshape.nx,
            dny: dshape.ny,
        });
    }

    let concentration_cell_count = nx * ny * nz;
    if snapshot.concentration.particle_count_per_cell.len() != concentration_cell_count {
        return Err(NetcdfOutputError::LengthMismatch {
            field: "concentration.particle_count_per_cell",
            expected: concentration_cell_count,
            actual: snapshot.concentration.particle_count_per_cell.len(),
        });
    }
    if snapshot.concentration.concentration_mass_kg.len() != concentration_cell_count {
        return Err(NetcdfOutputError::LengthMismatch {
            field: "concentration.concentration_mass_kg",
            expected: concentration_cell_count,
            actual: snapshot.concentration.concentration_mass_kg.len(),
        });
    }

    let deposition_cell_count = dshape.cell_count();
    if snapshot.deposition.dry_deposition_kg_m2.len() != deposition_cell_count {
        return Err(NetcdfOutputError::LengthMismatch {
            field: "deposition.dry_deposition_kg_m2",
            expected: deposition_cell_count,
            actual: snapshot.deposition.dry_deposition_kg_m2.len(),
        });
    }
    if snapshot.deposition.wet_deposition_kg_m2.len() != deposition_cell_count {
        return Err(NetcdfOutputError::LengthMismatch {
            field: "deposition.wet_deposition_kg_m2",
            expected: deposition_cell_count,
            actual: snapshot.deposition.wet_deposition_kg_m2.len(),
        });
    }
    Ok(())
}

/// Write one deterministic concentration/deposition snapshot to NetCDF.
pub fn write_gridded_output_netcdf(
    path: &Path,
    snapshot: &GriddedOutputSnapshot,
) -> Result<(), NetcdfOutputError> {
    validate_snapshot(snapshot)?;

    let mut file = netcdf::create(path)?;
    file.add_attribute("title", "FLEXPART-GPU concentration/deposition MVP output")?;
    file.add_attribute(
        "source",
        "flexpart-gpu (I-03 NetCDF concentration/deposition writer)",
    )?;
    file.add_attribute("conventions", "CF-1.10")?;
    file.add_attribute("deterministic_fields", "true")?;
    let species_index_i32 = i32::try_from(snapshot.metadata.species_index).map_err(|_| {
        NetcdfOutputError::SpeciesIndexTooLarge {
            value: snapshot.metadata.species_index,
        }
    })?;
    file.add_attribute("species_index", species_index_i32)?;

    let shape = snapshot.concentration.shape;
    file.add_dimension("time", 1)?;
    file.add_dimension("x", shape.nx)?;
    file.add_dimension("y", shape.ny)?;
    file.add_dimension("z", shape.nz)?;

    {
        let mut time_var = file.add_variable::<i64>("time", &["time"])?;
        time_var.put_attribute("units", "seconds since 1970-01-01 00:00:00 UTC")?;
        time_var.put_values(&[snapshot.metadata.simulation_time_seconds], ..)?;
    }
    {
        let x_values: Vec<f32> = (0..shape.nx).map(|x| x as f32).collect();
        let y_values: Vec<f32> = (0..shape.ny).map(|y| y as f32).collect();
        let z_values: Vec<f32> = (0..shape.nz).map(|z| z as f32).collect();

        let mut x_var = file.add_variable::<f32>("x", &["x"])?;
        x_var.put_attribute("axis", "X")?;
        x_var.put_values(&x_values, ..)?;

        let mut y_var = file.add_variable::<f32>("y", &["y"])?;
        y_var.put_attribute("axis", "Y")?;
        y_var.put_values(&y_values, ..)?;

        let mut z_var = file.add_variable::<f32>("z", &["z"])?;
        z_var.put_attribute("axis", "Z")?;
        z_var.put_values(&z_values, ..)?;
    }
    {
        let mut conc_count_var =
            file.add_variable::<u32>("concentration_particle_count", &["time", "x", "y", "z"])?;
        conc_count_var.put_attribute("long_name", "particle count per concentration cell")?;
        conc_count_var.put_attribute("units", "1")?;
        conc_count_var.put_values(
            &snapshot.concentration.particle_count_per_cell,
            (0, .., .., ..),
        )?;
    }
    {
        let mut conc_mass_var =
            file.add_variable::<f32>("concentration_mass_kg", &["time", "x", "y", "z"])?;
        conc_mass_var.put_attribute(
            "long_name",
            "accumulated species mass per concentration cell",
        )?;
        conc_mass_var.put_attribute("units", "kg")?;
        conc_mass_var.put_values(
            &snapshot.concentration.concentration_mass_kg,
            (0, .., .., ..),
        )?;
    }
    {
        let mut dry_var = file.add_variable::<f32>("dry_deposition_kg_m2", &["time", "x", "y"])?;
        dry_var.put_attribute("long_name", "dry deposition per horizontal cell")?;
        dry_var.put_attribute("units", "kg m-2")?;
        dry_var.put_values(&snapshot.deposition.dry_deposition_kg_m2, (0, .., ..))?;
    }
    {
        let mut wet_var = file.add_variable::<f32>("wet_deposition_kg_m2", &["time", "x", "y"])?;
        wet_var.put_attribute("long_name", "wet deposition per horizontal cell")?;
        wet_var.put_attribute("units", "kg m-2")?;
        wet_var.put_values(&snapshot.deposition.wet_deposition_kg_m2, (0, .., ..))?;
    }

    file.sync()?;
    file.close()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn temp_output_path(test_name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before epoch")
            .as_nanos();
        path.push(format!("flexpart_gpu_{test_name}_{nonce}.nc"));
        path
    }

    #[test]
    fn netcdf_output_smoke_writes_expected_metadata_and_shapes() {
        let path = temp_output_path("netcdf_smoke");
        let snapshot = GriddedOutputSnapshot {
            metadata: GriddedOutputMetadata {
                simulation_time_seconds: 1_706_000_000,
                species_index: 0,
            },
            concentration: ConcentrationGridOutput {
                shape: ConcentrationGridShape {
                    nx: 2,
                    ny: 3,
                    nz: 2,
                },
                particle_count_per_cell: vec![1, 0, 2, 3, 0, 0, 4, 1, 0, 0, 0, 2],
                concentration_mass_kg: vec![
                    0.1, 0.0, 0.2, 0.3, 0.0, 0.0, 0.4, 0.1, 0.0, 0.0, 0.0, 0.2,
                ],
            },
            deposition: DepositionGridOutput {
                shape: DepositionGridShape { nx: 2, ny: 3 },
                dry_deposition_kg_m2: vec![0.01, 0.02, 0.03, 0.0, 0.0, 0.1],
                wet_deposition_kg_m2: vec![0.2, 0.1, 0.0, 0.3, 0.2, 0.0],
            },
        };

        write_gridded_output_netcdf(&path, &snapshot).expect("netcdf write succeeds");

        let file = netcdf::open(&path).expect("netcdf read open succeeds");
        assert_eq!(file.dimension("time").expect("time dim").len(), 1);
        assert_eq!(file.dimension("x").expect("x dim").len(), 2);
        assert_eq!(file.dimension("y").expect("y dim").len(), 3);
        assert_eq!(file.dimension("z").expect("z dim").len(), 2);

        let conc_count = file
            .variable("concentration_particle_count")
            .expect("count variable exists");
        let dims: Vec<usize> = conc_count
            .dimensions()
            .iter()
            .map(netcdf::Dimension::len)
            .collect();
        assert_eq!(dims, vec![1, 2, 3, 2]);

        let counts = conc_count
            .get_values::<u32, _>((0, .., .., ..))
            .expect("read count values");
        assert_eq!(counts.len(), 12);
        assert_eq!(counts[0], 1);
        assert_eq!(counts[2], 2);
        assert_eq!(counts[11], 2);

        let conc_mass = file
            .variable("concentration_mass_kg")
            .expect("mass variable exists")
            .get_values::<f32, _>((0, .., .., ..))
            .expect("read mass values");
        assert_eq!(conc_mass.len(), 12);
        assert!((conc_mass[0] - 0.1).abs() < 1.0e-6);
        assert!((conc_mass[11] - 0.2).abs() < 1.0e-6);

        let dry = file
            .variable("dry_deposition_kg_m2")
            .expect("dry variable exists")
            .get_values::<f32, _>((0, .., ..))
            .expect("read dry values");
        assert_eq!(dry.len(), 6);
        assert!((dry[0] - 0.01).abs() < 1.0e-6);
        assert!((dry[5] - 0.1).abs() < 1.0e-6);

        fs::remove_file(&path).expect("cleanup temporary netcdf");
    }
}
