#!/usr/bin/env python3
"""Compare flexpart-gpu ETEX output against station observations.

Reads the GPU JSON output, converts mass-per-cell to concentration (pg/m³),
interpolates to station locations, and compares against ETEX measurements.

Also compares GPU vs Fortran reference if both are available.

Usage:
    python3 scripts/etex/compare_gpu_obs.py \
        --gpu-output target/etex/gpu_output.json \
        --measurements target/etex/measurements.json \
        --fortran-output target/etex/fortran_run/output \
        --output target/etex/etex_full_report.json
"""
import argparse
import json
import math
import os
import struct
from datetime import datetime

import numpy as np

R_EARTH = 6_371_000.0


def cell_volume_m3(ix, iy, dx_deg, dy_deg, xlon0, ylat0, dz_m):
    """Compute grid cell volume at (ix, iy) given lat-dependent width."""
    lat = ylat0 + (iy + 0.5) * dy_deg
    lat_rad = math.radians(lat)
    dx_m = R_EARTH * math.cos(lat_rad) * math.radians(dx_deg)
    dy_m = R_EARTH * math.radians(dy_deg)
    return dx_m * dy_m * dz_m


def gpu_mass_to_concentration(mass_flat, nx, ny, nz, dx_deg, dy_deg, xlon0, ylat0, outheights):
    """Convert mass-per-cell (kg) flat array to concentration grid (pg/m³).

    Returns array of shape (ny, nx) for the surface level.
    """
    mass = np.array(mass_flat, dtype=np.float64)

    # Compute cell volumes for surface level (0 to outheights[0])
    dz_surface = outheights[0]

    conc_2d = np.zeros((ny, nx), dtype=np.float64)
    for iy in range(ny):
        lat = ylat0 + (iy + 0.5) * dy_deg
        lat_rad = math.radians(lat)
        dx_m = R_EARTH * math.cos(lat_rad) * math.radians(dx_deg)
        dy_m = R_EARTH * math.radians(dy_deg)
        vol = dx_m * dy_m * dz_surface

        for ix in range(nx):
            flat = (ix * ny + iy) * nz + 0  # surface level (kz=0)
            if flat < len(mass):
                conc_2d[iy, ix] = mass[flat] * 1e12 / vol  # kg → pg/m³

    return conc_2d


def interpolate_to_station(grid_2d, xlon0, ylat0, dx, dy, nx, ny, stn_lon, stn_lat):
    """Bilinear interpolation of 2D grid (ny, nx) to station location."""
    ix = (stn_lon - xlon0) / dx
    iy = (stn_lat - ylat0) / dy

    if ix < 0 or ix >= nx - 1 or iy < 0 or iy >= ny - 1:
        return None

    ix0, iy0 = int(ix), int(iy)
    fx, fy = ix - ix0, iy - iy0

    val = ((1 - fx) * (1 - fy) * grid_2d[iy0, ix0] +
           fx * (1 - fy) * grid_2d[iy0, ix0 + 1] +
           (1 - fx) * fy * grid_2d[iy0 + 1, ix0] +
           fx * fy * grid_2d[iy0 + 1, ix0 + 1])
    return float(val)


def compute_metrics(obs, mod_vals):
    """Compute comparison metrics."""
    obs = np.array(obs, dtype=np.float64)
    mod_vals = np.array(mod_vals, dtype=np.float64)
    n = len(obs)
    if n == 0:
        return {"n": 0}

    diff = mod_vals - obs
    bias = float(np.mean(diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    obs_max = float(np.max(obs))
    nrmse = rmse / obs_max if obs_max > 0 else float("inf")

    obs_mean, mod_mean = np.mean(obs), np.mean(mod_vals)
    cov = np.mean((obs - obs_mean) * (mod_vals - mod_mean))
    obs_std, mod_std = np.std(obs), np.std(mod_vals)
    corr = float(cov / (obs_std * mod_std)) if obs_std > 0 and mod_std > 0 else 0.0

    obs_pos = obs > 0
    mod_pos = mod_vals > 0
    overlap = np.sum(obs_pos & mod_pos)
    union = np.sum(obs_pos | mod_pos)
    fms = float(overlap / union) if union > 0 else 0.0

    return {
        "n": n,
        "bias": round(bias, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "nrmse": round(nrmse, 6),
        "correlation": round(corr, 4),
        "fms": round(fms, 4),
        "obs_mean": round(float(obs_mean), 4),
        "mod_mean": round(float(mod_mean), 4),
        "obs_max": round(float(obs_max), 4),
        "mod_max": round(float(np.max(mod_vals)), 4),
    }


# --- Fortran reader (reuse from compare_etex_fortran_obs.py) ---

def read_fortran_record(data, offset, dtype="i"):
    rec_len = struct.unpack_from("<i", data, offset)[0]
    offset += 4
    n = rec_len // 4
    if n > 0:
        fmt = f"<{n}i" if dtype == "i" else f"<{n}f"
        vals = list(struct.unpack_from(fmt, data, offset))
    else:
        vals = []
    offset += rec_len + 4
    return vals, offset


def read_fortran_grid_conc(path, nx, ny, nz):
    """Read FLEXPART sparse binary grid_conc → (ny, nx, nz)."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 20:
        return np.zeros((ny, nx, nz), dtype=np.float32)

    offset = 0
    _, offset = read_fortran_record(data, offset, "i")
    for _ in range(2):  # wet + dry
        _, offset = read_fortran_record(data, offset, "i")
        _, offset = read_fortran_record(data, offset, "i")
        _, offset = read_fortran_record(data, offset, "i")
        _, offset = read_fortran_record(data, offset, "f")

    _, offset = read_fortran_record(data, offset, "i")
    indices, offset = read_fortran_record(data, offset, "i")
    _, offset = read_fortran_record(data, offset, "i")
    values, offset = read_fortran_record(data, offset, "f")

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

    grid = np.zeros((ny, nx, nz), dtype=np.float32)
    for kz in range(1, nz + 1):
        level = grid_flat[kz * nx * ny: (kz + 1) * nx * ny].reshape((ny, nx))
        grid[:, :, kz - 1] = level
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu-output", required=True)
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--fortran-output", default=None)
    parser.add_argument("--output", default="target/etex/etex_full_report.json")
    args = parser.parse_args()

    with open(args.gpu_output) as f:
        gpu_data = json.load(f)

    with open(args.measurements) as f:
        meas_data = json.load(f)

    grid = gpu_data["grid"]
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    dx, dy = grid["dx"], grid["dy"]
    xlon0, ylat0 = grid["xlon0"], grid["ylat0"]
    outheights = grid["heights_m"]

    print(f"GPU grid: {nx}×{ny}×{nz}, origin=({xlon0},{ylat0}), dx={dx}°")
    print(f"Output timesteps: {len(gpu_data['timesteps'])}")

    release_start = datetime(1994, 10, 23, 16, 0, 0)

    all_gpu_obs = []
    all_gpu_mod = []
    gpu_timestep_results = []

    for ts in gpu_data["timesteps"]:
        dt_str = ts["datetime"]
        dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
        hours_after = (dt - release_start).total_seconds() / 3600.0

        if hours_after < 0:
            continue

        conc_2d = gpu_mass_to_concentration(
            ts["concentration_mass_kg"], nx, ny, nz,
            dx, dy, xlon0, ylat0, outheights)

        surface_max = float(conc_2d.max())
        surface_nonzero = int(np.count_nonzero(conc_2d))

        relevant_meas = [m for m in meas_data["measurements"]
                         if abs(m["hours_after_release"] - hours_after) < 2.0]

        obs_vals, mod_vals = [], []
        for m in relevant_meas:
            model_val = interpolate_to_station(
                conc_2d, xlon0, ylat0, dx, dy, nx, ny,
                m["lon"], m["lat"])
            if model_val is not None:
                obs_vals.append(m["concentration_pg_m3"])
                mod_vals.append(model_val)

        print(f"  {dt_str} (t+{hours_after:.0f}h): surface_max={surface_max:.2f} pg/m³, "
              f"nonzero={surface_nonzero}, stations={len(obs_vals)}/{len(relevant_meas)}")

        if obs_vals:
            metrics = compute_metrics(obs_vals, mod_vals)
            print(f"    corr={metrics['correlation']:.4f}, rmse={metrics['rmse']:.2f}, "
                  f"bias={metrics['bias']:.2f}")
            all_gpu_obs.extend(obs_vals)
            all_gpu_mod.extend(mod_vals)
            gpu_timestep_results.append({
                "datetime": dt_str,
                "hours_after_release": hours_after,
                "surface_max_pg_m3": surface_max,
                "stations_compared": len(obs_vals),
                "active_particles": ts["active_particles"],
                "metrics": metrics,
            })

    print(f"\n{'='*60}")
    print(f"  GPU vs Observations (Overall)")
    print(f"{'='*60}")
    if all_gpu_obs:
        gpu_overall = compute_metrics(all_gpu_obs, all_gpu_mod)
        for k, v in gpu_overall.items():
            print(f"  {k}: {v}")
    else:
        gpu_overall = {"n": 0}
        print("  No matching observations")

    # --- Fortran comparison (if available) ---
    fortran_results = None
    gpu_vs_fortran = None

    if args.fortran_output and os.path.isdir(args.fortran_output):
        print(f"\n{'='*60}")
        print(f"  Fortran vs Observations")
        print(f"{'='*60}")

        header_path = os.path.join(args.fortran_output, "header_txt")
        dates_path = os.path.join(args.fortran_output, "dates")
        if os.path.exists(header_path) and os.path.exists(dates_path):
            with open(header_path) as f:
                header_lines = [l.strip() for l in f.readlines()]

            f_info = {}
            for i, line in enumerate(header_lines):
                if "outlon0" in line and i + 1 < len(header_lines):
                    p = header_lines[i + 1].split()
                    f_info = {"outlon0": float(p[0]), "outlat0": float(p[1]),
                              "nx": int(p[2]), "ny": int(p[3]),
                              "dx": float(p[4]), "dy": float(p[5])}
                if "numzgrid, outheight" in line and i + 1 < len(header_lines):
                    p = header_lines[i + 1].split()
                    f_info["nz"] = int(p[0])

            with open(dates_path) as f:
                output_dates = [l.strip() for l in f if l.strip()]

            all_f_obs, all_f_mod = [], []
            for date_str in output_dates:
                dt = datetime.strptime(date_str, "%Y%m%d%H%M%S")
                hours_after = (dt - release_start).total_seconds() / 3600.0

                conc_file = os.path.join(args.fortran_output,
                                         f"grid_conc_{date_str}_001")
                if not os.path.exists(conc_file):
                    continue

                grid_f = read_fortran_grid_conc(
                    conc_file, f_info["nx"], f_info["ny"], f_info["nz"])
                surface_f = grid_f[:, :, 0]

                relevant_meas = [m for m in meas_data["measurements"]
                                 if abs(m["hours_after_release"] - hours_after) < 2.0]

                for m in relevant_meas:
                    val = interpolate_to_station(
                        surface_f, f_info["outlon0"], f_info["outlat0"],
                        f_info["dx"], f_info["dy"], f_info["nx"], f_info["ny"],
                        m["lon"], m["lat"])
                    if val is not None:
                        all_f_obs.append(m["concentration_pg_m3"])
                        all_f_mod.append(val)

            if all_f_obs:
                fortran_results = compute_metrics(all_f_obs, all_f_mod)
                for k, v in fortran_results.items():
                    print(f"  {k}: {v}")

            # GPU vs Fortran direct comparison on shared timesteps
            print(f"\n{'='*60}")
            print(f"  GPU vs Fortran (grid-to-grid)")
            print(f"{'='*60}")

            gpu_f_obs, gpu_f_mod = [], []
            for gpu_ts in gpu_data["timesteps"]:
                dt_str = gpu_ts["datetime"]
                conc_file = os.path.join(args.fortran_output,
                                         f"grid_conc_{dt_str}_001")
                if not os.path.exists(conc_file):
                    continue

                gpu_conc = gpu_mass_to_concentration(
                    gpu_ts["concentration_mass_kg"], nx, ny, nz,
                    dx, dy, xlon0, ylat0, outheights)

                grid_f = read_fortran_grid_conc(
                    conc_file, f_info["nx"], f_info["ny"], f_info["nz"])
                surface_f = grid_f[:, :, 0]

                # Compare at station locations (common grid for both)
                relevant_meas = [m for m in meas_data["measurements"]
                                 if abs(m["hours_after_release"] -
                                        (datetime.strptime(dt_str, "%Y%m%d%H%M%S")
                                         - release_start).total_seconds() / 3600.0) < 2.0]
                for m in relevant_meas:
                    g_val = interpolate_to_station(
                        gpu_conc, xlon0, ylat0, dx, dy, nx, ny,
                        m["lon"], m["lat"])
                    f_val = interpolate_to_station(
                        surface_f, f_info["outlon0"], f_info["outlat0"],
                        f_info["dx"], f_info["dy"], f_info["nx"], f_info["ny"],
                        m["lon"], m["lat"])
                    if g_val is not None and f_val is not None:
                        gpu_f_obs.append(f_val)
                        gpu_f_mod.append(g_val)

            if gpu_f_obs:
                gpu_vs_fortran = compute_metrics(gpu_f_obs, gpu_f_mod)
                for k, v in gpu_vs_fortran.items():
                    print(f"  {k}: {v}")

    report = {
        "experiment": "ETEX-1",
        "gpu": {
            "source": "flexpart-gpu with ERA5 0.25° pressure-level data",
            "particle_count": gpu_data.get("final_active_particles", 0),
            "total_steps": gpu_data.get("total_steps", 0),
            "output_timesteps": len(gpu_data["timesteps"]),
            "vs_observations": gpu_overall,
            "timestep_results": gpu_timestep_results,
        },
        "fortran": {
            "source": "Fortran FLEXPART 10.4 with ERA5 → GRIB1 hybrid levels",
            "vs_observations": fortran_results,
        } if fortran_results else None,
        "gpu_vs_fortran": gpu_vs_fortran,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {args.output}")


if __name__ == "__main__":
    main()
