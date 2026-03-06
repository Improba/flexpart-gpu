#!/usr/bin/env python3
"""Download the western part of ERA5 for ETEX (345-360° = -15° to 0°) and merge.

The ARCO-ERA5 Zarr store uses 0-360° longitude. The initial download only
got 0-35°E. This script downloads 345-360° and merges all variables.
"""

import json
import os
import sys
import time as time_mod

import numpy as np

try:
    import xarray as xr
    import gcsfs
except ImportError:
    print("Need: uv pip install xarray gcsfs zarr", file=sys.stderr)
    sys.exit(1)


ARCO_ERA5_BUCKET = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
LON_WEST = slice(345, 360)
LAT_RANGE = slice(65, 35)
TIME_START = "1994-10-23T00:00"
TIME_END = "1994-10-26T23:00"
PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50]

ERA5_DIR = "target/etex/era5_raw"


def main():
    with open(os.path.join(ERA5_DIR, "metadata.json")) as f:
        meta = json.load(f)
    existing_vars = [v["name"] for v in meta["variables"]]

    print(f"Downloading western section (345-360° = -15° to 0°)")
    print(f"Variables to extend: {len(existing_vars)}")

    fs = gcsfs.GCSFileSystem(token="anon")
    store = gcsfs.GCSMap(ARCO_ERA5_BUCKET, gcs=fs)
    ds = xr.open_zarr(store, consolidated=True)

    west_dir = os.path.join(ERA5_DIR, "west")
    os.makedirs(west_dir, exist_ok=True)

    for var_info in meta["variables"]:
        var_name = var_info["name"]
        is_3d = "level" in var_info["dims"]

        if var_name not in ds:
            print(f"  [SKIP] {var_name} not in dataset")
            continue

        var = ds[var_name]
        t0 = time_mod.time()

        try:
            if is_3d:
                west = var.sel(
                    time=slice(TIME_START, TIME_END),
                    level=PRESSURE_LEVELS,
                    latitude=LAT_RANGE,
                    longitude=LON_WEST,
                )
            else:
                west = var.sel(
                    time=slice(TIME_START, TIME_END),
                    latitude=LAT_RANGE,
                    longitude=LON_WEST,
                )
        except Exception as e:
            print(f"  [SKIP] {var_name}: {e}")
            continue

        print(f"  {var_name:45s} {str(west.shape):20s}", end="", flush=True)
        west_data = west.values
        dt = time_mod.time() - t0
        print(f" ({dt:.1f}s)", end="")

        east_path = os.path.join(ERA5_DIR, f"{var_name}.npy")
        east_data = np.load(east_path)

        if is_3d:
            merged = np.concatenate([west_data, east_data], axis=3)
        else:
            merged = np.concatenate([west_data, east_data], axis=2)

        np.save(east_path, merged)
        print(f"  -> merged {merged.shape}")

    west_lons = ds.sel(longitude=LON_WEST).longitude.values
    east_lons = np.load(os.path.join(ERA5_DIR, "longitudes.npy"))
    west_lons_neg = west_lons - 360.0
    merged_lons = np.concatenate([west_lons_neg, east_lons])
    np.save(os.path.join(ERA5_DIR, "longitudes.npy"), merged_lons)
    print(f"\nMerged longitudes: {merged_lons.min():.2f} to {merged_lons.max():.2f} ({len(merged_lons)} pts)")

    meta["longitudes_range"] = [float(merged_lons[0]), float(merged_lons[-1])]
    meta["n_lon"] = len(merged_lons)
    for v in meta["variables"]:
        old_path = os.path.join(ERA5_DIR, v["file"])
        arr = np.load(old_path)
        v["shape"] = list(arr.shape)
        v["size_mb"] = round(arr.nbytes / 1e6, 1)

    with open(os.path.join(ERA5_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    monterfil_lon = -2.0
    monterfil_lat = 48.058
    lats = np.load(os.path.join(ERA5_DIR, "latitudes.npy"))
    print(f"\nMonterfil (-2°E, 48°N) in domain: "
          f"lon={merged_lons.min() <= monterfil_lon <= merged_lons.max()}, "
          f"lat={lats.min() <= monterfil_lat <= lats.max()}")

    print("\nDone. ERA5 data now covers full ETEX domain.")


if __name__ == "__main__":
    main()
