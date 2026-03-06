#!/usr/bin/env python3
"""Download ERA5 data for ETEX-1 from Google ARCO-ERA5 (no CDS account needed).

Uses the Analysis-Ready, Cloud Optimized (ARCO) ERA5 dataset hosted on
Google Cloud Storage. This is freely accessible without authentication.

Downloads only the variables needed for FLEXPART: 3D wind/T/q on pressure
levels + surface fields, for Oct 23-26, 1994 over Europe.

Prerequisites:
    uv pip install xarray gcsfs zarr numpy

Usage:
    python3 scripts/etex/download_era5_gcs.py --output-dir target/etex/era5_raw
"""

import argparse
import json
import os
import sys
import time as time_mod

try:
    import numpy as np
except ImportError:
    print("Missing numpy", file=sys.stderr); sys.exit(1)
try:
    import xarray as xr
except ImportError:
    print("Missing xarray. Install: uv pip install xarray", file=sys.stderr); sys.exit(1)
try:
    import gcsfs
except ImportError:
    print("Missing gcsfs. Install: uv pip install gcsfs", file=sys.stderr); sys.exit(1)


ARCO_ERA5_BUCKET = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

LON_RANGE = slice(-15, 35)
LAT_RANGE = slice(65, 35)
TIME_START = "1994-10-23T00:00"
TIME_END = "1994-10-26T23:00"

PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50]

VARS_3D = [
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "temperature",
    "specific_humidity",
    "geopotential",
]

VARS_SFC = [
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "boundary_layer_height",
    "surface_sensible_heat_flux",
    "surface_net_solar_radiation",
    "mean_eastward_turbulent_surface_stress",
    "mean_northward_turbulent_surface_stress",
    "convective_precipitation",
    "large_scale_precipitation",
    "forecast_surface_roughness",
]


def open_arco_era5():
    fs = gcsfs.GCSFileSystem(token="anon")
    store = gcsfs.GCSMap(ARCO_ERA5_BUCKET, gcs=fs)
    return xr.open_zarr(store, consolidated=True)


def download_variable(ds, var_name: str, output_dir: str, is_3d: bool) -> dict:
    """Download a single variable and save as .npy."""
    if var_name not in ds:
        return None

    var = ds[var_name]
    t0 = time_mod.time()

    try:
        if is_3d and "level" in var.dims:
            subset = var.sel(
                time=slice(TIME_START, TIME_END),
                level=PRESSURE_LEVELS,
                latitude=LAT_RANGE,
                longitude=LON_RANGE,
            )
        else:
            subset = var.sel(
                time=slice(TIME_START, TIME_END),
                latitude=LAT_RANGE,
                longitude=LON_RANGE,
            )
    except Exception as e:
        print(f"  [SKIP] {var_name}: {e}")
        return None

    print(f"  {var_name:45s} {str(subset.shape):25s}", end="", flush=True)
    data = subset.values
    dt = time_mod.time() - t0
    size_mb = data.nbytes / 1e6

    out_path = os.path.join(output_dir, f"{var_name}.npy")
    np.save(out_path, data)
    print(f" {size_mb:6.1f} MB  ({dt:.1f}s)")

    return {
        "name": var_name,
        "dims": list(subset.dims),
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "file": f"{var_name}.npy",
        "size_mb": round(size_mb, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", default="target/etex/era5_raw")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  ERA5 for ETEX-1 via Google ARCO-ERA5 (no CDS account)")
    print("=" * 70)
    print(f"  Period:  {TIME_START} to {TIME_END}")
    print(f"  Domain:  {LAT_RANGE.start}°N-{LAT_RANGE.stop}°N, "
          f"{LON_RANGE.start}°E-{LON_RANGE.stop}°E")
    print(f"  Levels:  {len(PRESSURE_LEVELS)} pressure levels")
    print(f"  3D vars: {len(VARS_3D)}")
    print(f"  Sfc vars: {len(VARS_SFC)}")
    print()

    print("Opening ARCO-ERA5 store...")
    ds = open_arco_era5()
    print(f"  {len(ds.data_vars)} variables available")

    time_ds = ds.sel(time=slice(TIME_START, TIME_END))
    n_times = len(time_ds.time)
    print(f"  {n_times} hourly timesteps selected")

    lat_sub = ds.sel(latitude=LAT_RANGE).latitude.values
    lon_sub = ds.sel(longitude=LON_RANGE).longitude.values
    print(f"  Grid: {len(lat_sub)} lat x {len(lon_sub)} lon")

    metadata = {
        "time_start": TIME_START,
        "time_end": TIME_END,
        "n_timesteps": n_times,
        "timesteps": [str(t) for t in time_ds.time.values[:4]] + ["..."],
        "latitudes_range": [float(lat_sub[-1]), float(lat_sub[0])],
        "longitudes_range": [float(lon_sub[0]), float(lon_sub[-1])],
        "n_lat": len(lat_sub),
        "n_lon": len(lon_sub),
        "pressure_levels_hpa": PRESSURE_LEVELS,
        "variables": [],
    }

    print("\nDownloading 3D fields (pressure levels):")
    for var in VARS_3D:
        info = download_variable(ds, var, args.output_dir, is_3d=True)
        if info:
            metadata["variables"].append(info)

    print("\nDownloading surface fields:")
    for var in VARS_SFC:
        info = download_variable(ds, var, args.output_dir, is_3d=False)
        if info:
            metadata["variables"].append(info)

    np.save(os.path.join(args.output_dir, "latitudes.npy"), lat_sub)
    np.save(os.path.join(args.output_dir, "longitudes.npy"), lon_sub)
    np.save(os.path.join(args.output_dir, "times.npy"),
            np.array([str(t) for t in time_ds.time.values]))

    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    total_mb = sum(v.get("size_mb", 0) for v in metadata["variables"])
    print(f"\nDone: {len(metadata['variables'])} variables, {total_mb:.0f} MB total")
    print(f"Metadata: {meta_path}")
    print(f"\nNext: python3 scripts/etex/prepare_flexpart_input_from_npy.py \\")
    print(f"        --era5-dir {args.output_dir} --output-dir target/etex/meteo")


if __name__ == "__main__":
    main()
