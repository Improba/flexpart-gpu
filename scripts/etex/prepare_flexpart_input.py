#!/usr/bin/env python3
"""Prepare FLEXPART-compatible input files from downloaded ERA5 GRIB data.

Reads ERA5 GRIB2 pressure-level and single-level files, extracts fields
per timestep, and writes FLEXPART-compatible GRIB1 files with the
standard ECMWF naming convention (ENyymmddHH).

Also exports wind/surface fields as JSON for the flexpart-gpu binary.

Prerequisites:
    pip install eccodes-python numpy

Usage:
    python3 scripts/etex/prepare_flexpart_input.py \\
        --era5-dir target/etex/era5_raw \\
        --output-dir target/etex/meteo
"""

import argparse
import json
import os
import sys
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    print("Missing numpy. Install: pip install numpy", file=sys.stderr)
    sys.exit(1)

try:
    import eccodes
except ImportError:
    print("Missing eccodes. Install: pip install eccodes-python", file=sys.stderr)
    sys.exit(1)


PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50]

PARAM_MAP_3D = {
    131: "u",    # U wind
    132: "v",    # V wind
    135: "w",    # omega (Pa/s)
    130: "t",    # temperature
    133: "q",    # specific humidity
    129: "z",    # geopotential
}

PARAM_MAP_SFC = {
    134: "sp",      # surface pressure
    165: "10u",     # 10m U wind
    166: "10v",     # 10m V wind
    167: "2t",      # 2m temperature
    168: "2d",      # 2m dewpoint
    146: "sshf",    # sensible heat flux
    176: "ssr",     # surface solar radiation
    180: "ewss",    # east-west surface stress
    181: "nsss",    # north-south surface stress
    142: "lsp",     # large-scale precipitation
    143: "cp",      # convective precipitation
    159: "blh",     # boundary layer height
}


def read_grib_by_timestep(grib_path: str):
    """Read GRIB file and organize messages by (date, time)."""
    timesteps = defaultdict(list)

    with open(grib_path, "rb") as f:
        while True:
            msgid = eccodes.codes_grib_new_from_file(f)
            if msgid is None:
                break

            try:
                date = eccodes.codes_get(msgid, "dataDate")
                time = eccodes.codes_get(msgid, "dataTime")
                param = eccodes.codes_get(msgid, "indicatorOfParameter", ktype=int)
                level = eccodes.codes_get(msgid, "level", ktype=int)
                nx = eccodes.codes_get(msgid, "Ni", ktype=int)
                ny = eccodes.codes_get(msgid, "Nj", ktype=int)
                values = eccodes.codes_get_values(msgid)

                short_name = eccodes.codes_get(msgid, "shortName")
                level_type = eccodes.codes_get(msgid, "typeOfLevel")

                lat_first = eccodes.codes_get(msgid, "latitudeOfFirstGridPointInDegrees")
                lon_first = eccodes.codes_get(msgid, "longitudeOfFirstGridPointInDegrees")
                lat_last = eccodes.codes_get(msgid, "latitudeOfLastGridPointInDegrees")
                lon_last = eccodes.codes_get(msgid, "longitudeOfLastGridPointInDegrees")
                dx = eccodes.codes_get(msgid, "iDirectionIncrementInDegrees")
                dy = eccodes.codes_get(msgid, "jDirectionIncrementInDegrees")

                timesteps[(date, time)].append({
                    "param": param,
                    "short_name": short_name,
                    "level": level,
                    "level_type": level_type,
                    "nx": nx, "ny": ny,
                    "dx": dx, "dy": dy,
                    "lat_first": lat_first, "lon_first": lon_first,
                    "lat_last": lat_last, "lon_last": lon_last,
                    "values": values.tolist(),
                })
            finally:
                eccodes.codes_release(msgid)

    return dict(timesteps)


def approximate_heights_m(pressure_levels_hpa):
    """Approximate geometric heights from pressure levels using barometric formula."""
    P0 = 1013.25
    T0 = 288.15
    L = 0.0065
    g = 9.80665
    R = 287.05
    heights = []
    for p in pressure_levels_hpa:
        h = (T0 / L) * (1.0 - (p / P0) ** (R * L / g))
        heights.append(round(h, 1))
    return heights


def write_flexpart_grib1(output_path: str, messages_3d: list, messages_sfc: list,
                          date: int, time: int, nx: int, ny: int, nz: int):
    """Write a FLEXPART-compatible GRIB1 file for one timestep."""
    with open(output_path, "wb") as fout:
        for msg in messages_3d:
            msgid = eccodes.codes_grib_new_from_samples("GRIB1")
            eccodes.codes_set(msgid, "centre", 98)
            eccodes.codes_set(msgid, "editionNumber", 1)
            eccodes.codes_set(msgid, "table2Version", 128)
            eccodes.codes_set(msgid, "dataDate", date)
            eccodes.codes_set(msgid, "dataTime", time)
            eccodes.codes_set(msgid, "indicatorOfParameter", msg["param"])
            eccodes.codes_set(msgid, "indicatorOfTypeOfLevel", 100)
            eccodes.codes_set(msgid, "level", msg["level"])
            eccodes.codes_set(msgid, "gridType", "regular_ll")
            eccodes.codes_set(msgid, "Ni", nx)
            eccodes.codes_set(msgid, "Nj", ny)
            eccodes.codes_set(msgid, "iDirectionIncrementInDegrees", msg["dx"])
            eccodes.codes_set(msgid, "jDirectionIncrementInDegrees", msg["dy"])
            eccodes.codes_set(msgid, "latitudeOfFirstGridPointInDegrees", msg["lat_first"])
            eccodes.codes_set(msgid, "longitudeOfFirstGridPointInDegrees", msg["lon_first"])
            eccodes.codes_set(msgid, "latitudeOfLastGridPointInDegrees", msg["lat_last"])
            eccodes.codes_set(msgid, "longitudeOfLastGridPointInDegrees", msg["lon_last"])
            eccodes.codes_set(msgid, "jScansPositively", 0)
            eccodes.codes_set(msgid, "packingType", "grid_simple")
            eccodes.codes_set(msgid, "bitsPerValue", 16)
            eccodes.codes_set_values(msgid, msg["values"])
            eccodes.codes_write(msgid, fout)
            eccodes.codes_release(msgid)

        for msg in messages_sfc:
            msgid = eccodes.codes_grib_new_from_samples("GRIB1")
            eccodes.codes_set(msgid, "centre", 98)
            eccodes.codes_set(msgid, "editionNumber", 1)
            eccodes.codes_set(msgid, "table2Version", 128)
            eccodes.codes_set(msgid, "dataDate", date)
            eccodes.codes_set(msgid, "dataTime", time)
            eccodes.codes_set(msgid, "indicatorOfParameter", msg["param"])
            eccodes.codes_set(msgid, "indicatorOfTypeOfLevel", 1)
            eccodes.codes_set(msgid, "level", 0)
            eccodes.codes_set(msgid, "gridType", "regular_ll")
            eccodes.codes_set(msgid, "Ni", nx)
            eccodes.codes_set(msgid, "Nj", ny)
            eccodes.codes_set(msgid, "iDirectionIncrementInDegrees", msg["dx"])
            eccodes.codes_set(msgid, "jDirectionIncrementInDegrees", msg["dy"])
            eccodes.codes_set(msgid, "latitudeOfFirstGridPointInDegrees", msg["lat_first"])
            eccodes.codes_set(msgid, "longitudeOfFirstGridPointInDegrees", msg["lon_first"])
            eccodes.codes_set(msgid, "latitudeOfLastGridPointInDegrees", msg["lat_last"])
            eccodes.codes_set(msgid, "longitudeOfLastGridPointInDegrees", msg["lon_last"])
            eccodes.codes_set(msgid, "jScansPositively", 0)
            eccodes.codes_set(msgid, "packingType", "grid_simple")
            eccodes.codes_set(msgid, "bitsPerValue", 16)
            eccodes.codes_set_values(msgid, msg["values"])
            eccodes.codes_write(msgid, fout)
            eccodes.codes_release(msgid)


def export_gpu_json(output_path: str, plev_data: dict, sfc_data: dict,
                     grid_info: dict, timesteps_info: list):
    """Export wind/surface fields as JSON for flexpart-gpu consumption."""
    data = {
        "grid": grid_info,
        "pressure_levels_hpa": PRESSURE_LEVELS,
        "heights_m": approximate_heights_m(PRESSURE_LEVELS),
        "timesteps": timesteps_info,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  GPU JSON: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--era5-dir", required=True,
                        help="Directory containing ERA5 GRIB files")
    parser.add_argument("--output-dir", default="target/etex/meteo",
                        help="Output directory for FLEXPART input")
    parser.add_argument("--gpu-json", default=None,
                        help="Path for GPU JSON export (default: <output-dir>/gpu_meteo.json)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plev_path = os.path.join(args.era5_dir, "era5_pressure_levels.grib")
    sfc_path = os.path.join(args.era5_dir, "era5_single_levels.grib")

    for p in [plev_path, sfc_path]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found. Run download_era5.py first.", file=sys.stderr)
            sys.exit(1)

    print("Reading ERA5 pressure-level GRIB...")
    plev_by_time = read_grib_by_timestep(plev_path)
    print(f"  {len(plev_by_time)} timesteps found")

    print("Reading ERA5 single-level GRIB...")
    sfc_by_time = read_grib_by_timestep(sfc_path)
    print(f"  {len(sfc_by_time)} timesteps found")

    all_times = sorted(set(plev_by_time.keys()) & set(sfc_by_time.keys()))
    print(f"\n{len(all_times)} common timesteps: {all_times[0]} -> {all_times[-1]}")

    grid_info = None
    available_lines = []

    for date, time in all_times:
        time_hhmm = time * 100 if time < 100 else time
        hour = time if time < 100 else time // 100
        filename = f"EN{date}{hour:02d}"
        out_path = os.path.join(args.output_dir, filename)

        plev_msgs = plev_by_time.get((date, time), [])
        sfc_msgs = sfc_by_time.get((date, time), [])

        if not plev_msgs:
            print(f"  [WARN] No pressure-level data for {date} {time:04d}")
            continue

        first = plev_msgs[0]
        nx, ny = first["nx"], first["ny"]
        nz = len(PRESSURE_LEVELS)

        if grid_info is None:
            grid_info = {
                "nx": nx, "ny": ny, "nz": nz,
                "dx": first["dx"], "dy": first["dy"],
                "lon_first": first["lon_first"],
                "lat_first": first["lat_first"],
                "lon_last": first["lon_last"],
                "lat_last": first["lat_last"],
            }

        write_flexpart_grib1(out_path, plev_msgs, sfc_msgs, date, time_hhmm, nx, ny, nz)
        available_lines.append(f"{date} {time_hhmm:06d}      {filename}      ON DISC")
        print(f"  {filename}")

    available_path = os.path.join(args.output_dir, "AVAILABLE")
    with open(available_path, "w") as f:
        f.write("DATE      TIME     FILENAME     SPECIFICATIONS\n")
        f.write("YYYYMMDD  HHMMSS\n")
        f.write("________ ________ __________ __________\n")
        for line in available_lines:
            f.write(line + "\n")

    print(f"\nWrote {len(available_lines)} GRIB1 files + AVAILABLE to {args.output_dir}/")

    gpu_json_path = args.gpu_json or os.path.join(args.output_dir, "gpu_meteo.json")
    if grid_info:
        export_gpu_json(gpu_json_path, plev_by_time, sfc_by_time, grid_info,
                        [{"date": d, "time": t} for d, t in all_times])


if __name__ == "__main__":
    main()
