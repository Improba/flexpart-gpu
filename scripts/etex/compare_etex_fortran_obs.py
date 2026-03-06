#!/usr/bin/env python3
"""Compare Fortran FLEXPART ETEX output against station observations.

Reads Fortran grid_conc binary files, interpolates to station locations,
and compares against the parsed ETEX measurements.

Usage:
    python3 scripts/etex/compare_etex_fortran_obs.py \\
        --fortran-output target/etex/fortran_run/output \\
        --measurements target/etex/measurements.json \\
        --output target/etex/etex_validation_report.json
"""

import argparse
import json
import math
import os
import struct
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np


def parse_header_txt(path: str) -> dict:
    """Parse FLEXPART header_txt file to get grid metadata."""
    with open(path) as f:
        lines = [l.strip() for l in f.readlines()]

    info = {}
    for i, line in enumerate(lines):
        if "outlon0" in line and i + 1 < len(lines):
            parts = lines[i + 1].split()
            info["outlon0"] = float(parts[0])
            info["outlat0"] = float(parts[1])
            info["nx"] = int(parts[2])
            info["ny"] = int(parts[3])
            info["dx"] = float(parts[4])
            info["dy"] = float(parts[5])
        if "numzgrid, outheight" in line and i + 1 < len(lines):
            parts = lines[i + 1].split()
            info["nz"] = int(parts[0])
            info["outheights"] = [float(x) for x in parts[1:]]
        if "interval, averaging" in line and i + 1 < len(lines):
            parts = lines[i + 1].split()
            info["interval_s"] = int(parts[0])
            info["averaging_s"] = int(parts[1])
            info["sampling_s"] = int(parts[2])
        if "ibdate" in line and i + 1 < len(lines):
            parts = lines[i + 1].split()
            info["ibdate"] = parts[0]
            info["ibtime"] = parts[1]
    return info


def _read_fortran_record(data: bytes, offset: int, dtype: str = "i") -> tuple:
    """Read one Fortran unformatted sequential record."""
    endian = "<"
    rec_len = struct.unpack_from(f"{endian}i", data, offset)[0]
    offset += 4
    n = rec_len // 4
    if n > 0:
        fmt = f"{endian}{n}i" if dtype == "i" else f"{endian}{n}f"
        vals = list(struct.unpack_from(fmt, data, offset))
    else:
        vals = []
    offset += rec_len + 4
    return vals, offset


def read_grid_conc(path: str, nx: int, ny: int, nz: int) -> np.ndarray:
    """Read a FLEXPART sparse binary grid_conc file.

    Returns array of shape (ny, nx, nz) with concentrations in pg/m³.
    The sparse encoding uses sign alternation: sp_fact starts at -1,
    flips at each run start, so run 0 is positive, run 1 negative, etc.
    Flat index = ix + jy*nx + kz*nx*ny with kz in [1..nz].
    """
    with open(path, "rb") as f:
        data = f.read()

    if len(data) < 20:
        return np.zeros((ny, nx, nz), dtype=np.float32)

    offset = 0

    _ts, offset = _read_fortran_record(data, offset, "i")

    # Wet deposition (sp_count_i, indices, sp_count_r, values)
    _, offset = _read_fortran_record(data, offset, "i")
    _, offset = _read_fortran_record(data, offset, "i")
    _, offset = _read_fortran_record(data, offset, "i")
    _, offset = _read_fortran_record(data, offset, "f")

    # Dry deposition
    _, offset = _read_fortran_record(data, offset, "i")
    _, offset = _read_fortran_record(data, offset, "i")
    _, offset = _read_fortran_record(data, offset, "i")
    _, offset = _read_fortran_record(data, offset, "f")

    # 3D concentration
    _, offset = _read_fortran_record(data, offset, "i")
    indices, offset = _read_fortran_record(data, offset, "i")
    _, offset = _read_fortran_record(data, offset, "i")
    values, offset = _read_fortran_record(data, offset, "f")

    total_size = (nz + 1) * ny * nx
    grid_flat = np.zeros(total_size, dtype=np.float32)

    val_idx = 0
    for run_idx in range(len(indices)):
        run_start = indices[run_idx]
        pos = run_start
        expected_positive = (run_idx % 2 == 0)

        while val_idx < len(values):
            v = values[val_idx]
            if (v > 0) != expected_positive:
                break
            if 0 <= pos < total_size:
                grid_flat[pos] = abs(v)
            pos += 1
            val_idx += 1

    # Reshape: grid_flat[ix + jy*nx + kz*nx*ny] -> grid[jy, ix, kz-1]
    grid = np.zeros((ny, nx, nz), dtype=np.float32)
    for kz in range(1, nz + 1):
        level = grid_flat[kz * nx * ny: (kz + 1) * nx * ny].reshape((ny, nx))
        grid[:, :, kz - 1] = level
    return grid


def interpolate_to_station(grid_2d, outlon0, outlat0, dx, dy, nx, ny, stn_lon, stn_lat):
    """Bilinear interpolation of 2D grid (ny, nx) to station location."""
    ix = (stn_lon - outlon0) / dx
    iy = (stn_lat - outlat0) / dy

    if ix < 0 or ix >= nx - 1 or iy < 0 or iy >= ny - 1:
        return None

    ix0, iy0 = int(ix), int(iy)
    fx, fy = ix - ix0, iy - iy0

    val = ((1 - fx) * (1 - fy) * grid_2d[iy0, ix0] +
           fx * (1 - fy) * grid_2d[iy0, ix0 + 1] +
           (1 - fx) * fy * grid_2d[iy0 + 1, ix0] +
           fx * fy * grid_2d[iy0 + 1, ix0 + 1])
    return float(val)


def compute_metrics(obs: np.ndarray, mod: np.ndarray) -> dict:
    """Statistical comparison metrics."""
    n = len(obs)
    if n == 0:
        return {"n": 0}

    diff = mod - obs
    bias = float(np.mean(diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    obs_max = float(np.max(obs))
    nrmse = rmse / obs_max if obs_max > 0 else float("inf")

    obs_mean, mod_mean = np.mean(obs), np.mean(mod)
    cov = np.mean((obs - obs_mean) * (mod - mod_mean))
    obs_std, mod_std = np.std(obs), np.std(mod)
    corr = float(cov / (obs_std * mod_std)) if obs_std > 0 and mod_std > 0 else 0.0

    obs_pos, mod_pos = obs > 0, mod > 0
    overlap = np.sum(obs_pos & mod_pos)
    union = np.sum(obs_pos | mod_pos)
    fms = float(overlap / union) if union > 0 else 0.0

    return {
        "n": n,
        "bias": round(bias, 2),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "nrmse": round(nrmse, 4),
        "correlation": round(corr, 4),
        "fms": round(fms, 4),
        "obs_mean": round(float(obs_mean), 2),
        "mod_mean": round(float(mod_mean), 2),
        "obs_max": round(float(obs_max), 2),
        "mod_max": round(float(np.max(mod)), 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fortran-output", required=True)
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--output", default="target/etex/etex_validation_report.json")
    args = parser.parse_args()

    header_path = os.path.join(args.fortran_output, "header_txt")
    info = parse_header_txt(header_path)
    print(f"Grid: {info['nx']}x{info['ny']}x{info['nz']}")
    print(f"Origin: ({info['outlon0']}, {info['outlat0']})")
    print(f"Resolution: {info['dx']}° x {info['dy']}°")
    print(f"Heights: {info['outheights']}")

    with open(args.measurements) as f:
        meas_data = json.load(f)

    dates_path = os.path.join(args.fortran_output, "dates")
    with open(dates_path) as f:
        output_dates = [l.strip() for l in f.readlines() if l.strip()]

    print(f"\nOutput timesteps: {len(output_dates)}")
    release_start = datetime(1994, 10, 23, 16, 0, 0)

    all_obs = []
    all_mod = []
    timestep_results = []

    for date_str in output_dates:
        dt = datetime.strptime(date_str, "%Y%m%d%H%M%S")
        hours_after = (dt - release_start).total_seconds() / 3600.0

        conc_file = os.path.join(args.fortran_output, f"grid_conc_{date_str}_001")
        if not os.path.exists(conc_file):
            print(f"  [SKIP] {conc_file} not found")
            continue

        grid = read_grid_conc(conc_file, info["nx"], info["ny"], info["nz"])
        surface_conc = grid[:, :, 0]  # shape (ny, nx) — first vertical level
        total_mass = float(np.sum(grid))

        print(f"\n  {date_str} (t+{hours_after:.0f}h): total_mass={total_mass:.2e}, "
              f"surface max={float(np.max(surface_conc)):.2e}")

        relevant_meas = [m for m in meas_data["measurements"]
                         if abs(m["hours_after_release"] - hours_after) < 2.0]

        obs_vals = []
        mod_vals = []
        stations_compared = 0

        for m in relevant_meas:
            stn_lon, stn_lat = m["lon"], m["lat"]
            model_val = interpolate_to_station(
                surface_conc, info["outlon0"], info["outlat0"],
                info["dx"], info["dy"], info["nx"], info["ny"],
                stn_lon, stn_lat
            )
            if model_val is not None:
                obs_vals.append(m["concentration_pg_m3"])
                mod_vals.append(model_val)
                stations_compared += 1

        print(f"  Stations matched: {stations_compared}/{len(relevant_meas)}")

        if obs_vals:
            obs_arr = np.array(obs_vals)
            mod_arr = np.array(mod_vals)
            metrics = compute_metrics(obs_arr, mod_arr)
            print(f"  Correlation: {metrics.get('correlation', 0):.4f}, "
                  f"RMSE: {metrics.get('rmse', 0):.2f}")
            all_obs.extend(obs_vals)
            all_mod.extend(mod_vals)
            timestep_results.append({
                "datetime": date_str,
                "hours_after_release": hours_after,
                "stations_compared": stations_compared,
                "metrics": metrics,
            })

    print(f"\n{'=' * 60}")
    print(f"  Overall ETEX-1 Validation (Fortran vs Observations)")
    print(f"{'=' * 60}")

    if all_obs:
        obs_all = np.array(all_obs)
        mod_all = np.array(all_mod)
        overall = compute_metrics(obs_all, mod_all)
        print(f"  Total comparisons: {overall['n']}")
        print(f"  Correlation:       {overall['correlation']:.4f}")
        print(f"  RMSE:              {overall['rmse']:.2f} pg/m³")
        print(f"  NRMSE:             {overall['nrmse']:.4f}")
        print(f"  Bias:              {overall['bias']:.2f} pg/m³")
        print(f"  MAE:               {overall['mae']:.2f} pg/m³")
        print(f"  FMS:               {overall['fms']:.4f}")
        print(f"  Obs mean/max:      {overall['obs_mean']:.1f} / {overall['obs_max']:.1f} pg/m³")
        print(f"  Model mean/max:    {overall['mod_mean']:.1f} / {overall['mod_max']:.1f} pg/m³")
    else:
        overall = {"n": 0, "note": "no matching observations in simulation period"}
        print("  No matching observation data in simulation period")

    report = {
        "experiment": "ETEX-1",
        "source": "Fortran FLEXPART 10.4 with ERA5 pressure-level data",
        "grid": info,
        "output_timesteps": len(output_dates),
        "simulation_hours": (datetime.strptime(output_dates[-1], "%Y%m%d%H%M%S")
                             - release_start).total_seconds() / 3600.0,
        "overall_metrics": overall,
        "timestep_results": timestep_results,
        "notes": [
            "ERA5 pressure-level data converted to hybrid-level format for FLEXPART",
            "Simulation crashed at t+18h due to particles leaving limited-area domain",
            "Comparison uses surface-level concentration (layer 0-100m)",
            "Concentrations in pg/m³ (1e12 factor applied by FLEXPART concoutput)",
        ],
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {args.output}")


if __name__ == "__main__":
    main()
