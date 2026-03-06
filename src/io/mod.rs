//! I/O modules for meteorological data and model output.
//!
//! Current MVP coverage:
//! - GRIB2 ERA5 read path for core 3-D fields (`u`, `v`, `w`, `t`, `q`) via
//!   a feature-gated ecCodes adapter.
//! - NetCDF read path for core 3-D fields (`u`, `v`, `w`, `t`, `q`) via
//!   a feature-gated NetCDF adapter.
//! - Conversion into existing wind structures (`WindFieldGrid`, `WindField3D`).
//! - Hybrid sigma-pressure vertical transform into geometric heights (IO-03).
//! - Temporal interpolation between two meteorological snapshots (IO-04).
//! - NetCDF output writer for concentration/deposition grids (I-03).

pub mod grib2;
pub mod grib2_async;
pub mod netcdf;
#[cfg(feature = "netcdf")]
pub mod netcdf_output;
pub mod pbl_params;
pub mod temporal;
pub mod vertical_transform;

pub use grib2::{
    build_era5_mvp_from_records, load_era5_mvp_from_grib2, Era5GribGridMetadata, Era5GribRecord,
    Era5MvpSnapshot, Era5MvpSnapshotMetadata, Grib2ReaderError,
};
pub use grib2_async::{read_era5_snapshot_async, GribPrefetchHandle};
pub use netcdf::{
    load_mvp_from_netcdf, load_mvp_from_netcdf_with_options, NetcdfMvpReadOptions,
    NetcdfMvpSnapshot, NetcdfMvpSnapshotMetadata, NetcdfMvpVariableNames, NetcdfReaderError,
};
#[cfg(feature = "netcdf")]
pub use netcdf_output::{
    write_gridded_output_netcdf, DepositionGridOutput, DepositionGridShape, GriddedOutputMetadata,
    GriddedOutputSnapshot, NetcdfOutputError,
};
pub use pbl_params::{
    bulk_richardson_number, compute_pbl_cell_parameters, compute_pbl_parameters_from_met,
    estimate_friction_velocity_m_s, gradient_richardson_number, inverse_obukhov_length_per_m,
    obukhov_length_from_surface_flux_m, BulkRichardsonInput, ComputedPblFields,
    FrictionVelocityInput, GradientRichardsonInput, ObukhovInput, PblBulkProfilePoint,
    PblCellInput, PblCellOutput, PblComputationOptions, PblMetInputGrids, PblParameterError,
    PblProfileInputs,
};
pub use temporal::{
    interpolate_surface_fields_linear, interpolate_wind_field_linear, TemporalInterpolationError,
    TimeBoundsBehavior,
};
pub use vertical_transform::{
    compute_hybrid_level_pressure_pa, transform_hybrid_sigma_to_height, HybridVerticalTransform,
    VerticalTransformError,
};
