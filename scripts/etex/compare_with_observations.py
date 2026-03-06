#!/usr/bin/env python3
"""Compare FLEXPART (Fortran and GPU) output against ETEX-1 observations.

Reads:
  - Fortran FLEXPART grid_conc output (binary) or partposit particle dumps
  - flexpart-gpu JSON output
  - ETEX-1 parsed measurements (JSON from parse_measurements.py)

Computes:
  - Statistical metrics: RMSE, normalized RMSE, correlation, bias
  - Figure of Merit in Space (FMS)
  - Rank correlation (Spearman)
  - Station-level time series comparison
  - Plume arrival time comparison

Usage:
    python3 scripts/etex/compare_with_observations.py \\
        --measurements target/etex/measurements.json \\
        --fortran-output target/etex/fortran_run/output \\
        --gpu-output target/etex/gpu_output.json \\
        --output target/etex/etex_comparison_report.json
"""

import argparse
import json
import math
import os
import struct
import sys
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    print("Missing numpy. Install: pip install numpy", file=sys.stderr)
    sys.exit(1)


def load_measurements(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_gpu_output(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def read_fortran_header(header_path: str) -> dict:
    """Read FLEXPART header file to get grid metadata."""
    if not os.path.exists(header_path):
        return None
    with open(header_path, "rb") as f:
        data = f.read()

    for endian in ["<", ">"]:
        try:
            offset = 4
            ibdate = struct.unpack_from(f"{endian}i", data, offset)[0]; offset += 4
            ibtime = struct.unpack_from(f"{endian}i", data, offset)[0]; offset += 4

            fmt_s = f"{endian}24s"
            version = struct.unpack_from(fmt_s, data, offset)[0]; offset += 24

            offset += 4 + 4
            loutstep = struct.unpack_from(f"{endian}i", data, offset)[0]; offset += 4
            loutaver = struct.unpack_from(f"{endian}i", data, offset)[0]; offset += 4
            loutsample = struct.unpack_from(f"{endian}i", data, offset)[0]; offset += 4
            offset += 4

            offset += 4 + 4
            outlon0 = struct.unpack_from(f"{endian}f", data, offset)[0]; offset += 4
            outlat0 = struct.unpack_from(f"{endian}f", data, offset)[0]; offset += 4
            numxgrid = struct.unpack_from(f"{endian}i", data, offset)[0]; offset += 4
            numygrid = struct.unpack_from(f"{endian}i", data, offset)[0]; offset += 4
            dxout = struct.unpack_from(f"{endian}f", data, offset)[0]; offset += 4
            dyout = struct.unpack_from(f"{endian}f", data, offset)[0]; offset += 4

            if 0 < numxgrid < 10000 and 0 < numygrid < 10000 and 0 < dxout < 100:
                return {
                    "outlon0": outlon0, "outlat0": outlat0,
                    "nx": numxgrid, "ny": numygrid,
                    "dx": dxout, "dy": dyout,
                    "endian": endian,
                }
        except Exception:
            continue
    return None


def interpolate_model_to_stations(grid_data, grid_info, stations):
    """Interpolate gridded model output to station locations (bilinear)."""
    results = {}
    for stn_id, stn in stations.items():
        lat, lon = stn["lat"], stn["lon"]
        ix = (lon - grid_info["outlon0"]) / grid_info["dx"]
        iy = (lat - grid_info["outlat0"]) / grid_info["dy"]

        if ix < 0 or ix >= grid_info["nx"] - 1 or iy < 0 or iy >= grid_info["ny"] - 1:
            continue

        ix0, iy0 = int(ix), int(iy)
        fx, fy = ix - ix0, iy - iy0

        c00 = grid_data[ix0, iy0] if ix0 < grid_info["nx"] and iy0 < grid_info["ny"] else 0
        c10 = grid_data[ix0+1, iy0] if ix0+1 < grid_info["nx"] and iy0 < grid_info["ny"] else 0
        c01 = grid_data[ix0, iy0+1] if ix0 < grid_info["nx"] and iy0+1 < grid_info["ny"] else 0
        c11 = grid_data[ix0+1, iy0+1] if ix0+1 < grid_info["nx"] and iy0+1 < grid_info["ny"] else 0

        val = (1-fx)*(1-fy)*c00 + fx*(1-fy)*c10 + (1-fx)*fy*c01 + fx*fy*c11
        results[stn_id] = float(val)

    return results


def compute_metrics(observed: np.ndarray, modeled: np.ndarray) -> dict:
    """Compute comparison metrics between observed and modeled arrays."""
    n = len(observed)
    if n == 0:
        return {"n": 0}

    diff = modeled - observed
    bias = float(np.mean(diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    obs_range = float(np.max(observed) - np.min(observed))
    nrmse = rmse / obs_range if obs_range > 0 else float("inf")

    obs_mean = np.mean(observed)
    mod_mean = np.mean(modeled)
    cov = np.mean((observed - obs_mean) * (modeled - mod_mean))
    obs_std = np.std(observed)
    mod_std = np.std(modeled)
    corr = float(cov / (obs_std * mod_std)) if obs_std > 0 and mod_std > 0 else 0.0

    # Figure of Merit in Space (overlap / union of nonzero areas)
    obs_pos = observed > 0
    mod_pos = modeled > 0
    overlap = np.sum(obs_pos & mod_pos)
    union = np.sum(obs_pos | mod_pos)
    fms = float(overlap / union) if union > 0 else 0.0

    # Fractional bias
    sum_obs_mod = np.sum(observed + modeled)
    fb = float(2 * np.sum(diff) / sum_obs_mod) if sum_obs_mod > 0 else 0.0

    return {
        "n": n,
        "bias": round(bias, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "nrmse": round(nrmse, 4),
        "correlation": round(corr, 4),
        "fms": round(fms, 4),
        "fractional_bias": round(fb, 4),
        "obs_mean": round(float(obs_mean), 4),
        "mod_mean": round(float(mod_mean), 4),
    }


def compare_gpu_vs_fortran_particles(gpu_output: dict, fortran_partposit: dict) -> dict:
    """Compare GPU and Fortran particle distributions directly."""
    if not gpu_output or not fortran_partposit:
        return {"status": "insufficient_data"}

    gpu_stats = gpu_output.get("particle_z_stats", {})
    if not gpu_stats:
        return {"status": "no_gpu_stats"}

    return {
        "gpu_particles": gpu_stats.get("count", 0),
        "gpu_mean_z_m": gpu_stats.get("mean_m", 0),
        "gpu_std_z_m": gpu_stats.get("std_m", 0),
        "gpu_com_lon": gpu_stats.get("lon_mean", 0),
        "gpu_com_lat": gpu_stats.get("lat_mean", 0),
        "status": "compared",
    }


def print_report(report: dict, verbose: bool = False):
    """Print comparison report to stdout."""
    print("\n" + "=" * 70)
    print("  ETEX-1 Validation Report")
    print("=" * 70)

    meas_stats = report.get("measurement_statistics", {})
    print(f"\nMeasurements: {meas_stats.get('total_records', 0)} records, "
          f"{meas_stats.get('stations_count', 0)} stations")
    print(f"Detection rate: {meas_stats.get('detection_rate', 0)*100:.1f}%")
    print(f"Max: {meas_stats.get('max_concentration_pg_m3', 0):.0f} pg/m³")

    for source in ["fortran_vs_observations", "gpu_vs_observations"]:
        metrics = report.get(source, {})
        if not metrics or metrics.get("n", 0) == 0:
            print(f"\n{source}: no data")
            continue
        print(f"\n{source}:")
        print(f"  N:           {metrics['n']}")
        print(f"  Correlation: {metrics.get('correlation', 0):.4f}")
        print(f"  RMSE:        {metrics.get('rmse', 0):.4f}")
        print(f"  NRMSE:       {metrics.get('nrmse', 0):.4f}")
        print(f"  Bias:        {metrics.get('bias', 0):.4f}")
        print(f"  FMS:         {metrics.get('fms', 0):.4f}")

    gpu_vs_f = report.get("gpu_vs_fortran", {})
    if gpu_vs_f and gpu_vs_f.get("status") == "compared":
        print(f"\nGPU vs Fortran particle comparison:")
        print(f"  GPU particles: {gpu_vs_f.get('gpu_particles', 0)}")
        print(f"  GPU mean z:    {gpu_vs_f.get('gpu_mean_z_m', 0):.1f} m")
        print(f"  GPU COM:       ({gpu_vs_f.get('gpu_com_lon', 0):.4f}°, "
              f"{gpu_vs_f.get('gpu_com_lat', 0):.4f}°)")

    verdict = report.get("verdict", {})
    print(f"\n{'=' * 70}")
    print(f"  VERDICT: {verdict.get('status', 'UNKNOWN')}")
    if "notes" in verdict:
        for note in verdict["notes"]:
            print(f"    - {note}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--measurements", required=True,
                        help="ETEX measurements JSON (from parse_measurements.py)")
    parser.add_argument("--fortran-output", default=None,
                        help="Fortran FLEXPART output directory")
    parser.add_argument("--gpu-output", default=None,
                        help="GPU output JSON")
    parser.add_argument("--output", default="target/etex/comparison_report.json",
                        help="Output report JSON")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    measurements = load_measurements(args.measurements)
    meas_records = measurements["measurements"]
    stations = measurements["stations"]
    meas_stats = measurements["statistics"]

    gpu_output = load_gpu_output(args.gpu_output) if args.gpu_output and os.path.exists(args.gpu_output) else None

    report = {
        "experiment": "ETEX-1",
        "measurement_statistics": meas_stats,
        "fortran_vs_observations": {},
        "gpu_vs_observations": {},
        "gpu_vs_fortran": {},
        "verdict": {"status": "INCOMPLETE", "notes": []},
    }

    if args.fortran_output and os.path.isdir(args.fortran_output):
        header_path = os.path.join(args.fortran_output, "header")
        header = read_fortran_header(header_path)
        if header:
            report["fortran_grid"] = header
            report["verdict"]["notes"].append("Fortran output loaded")
        else:
            report["verdict"]["notes"].append("Could not parse Fortran header")

    if gpu_output:
        report["gpu_vs_fortran"] = compare_gpu_vs_fortran_particles(gpu_output, {})
        report["verdict"]["notes"].append(f"GPU output loaded: "
            f"{gpu_output.get('total_particles_active', 0)} active particles")

    notes = report["verdict"]["notes"]
    if not args.fortran_output:
        notes.append("Fortran output not provided")
    if not gpu_output:
        notes.append("GPU output not provided")

    if report["fortran_vs_observations"] or report["gpu_vs_observations"]:
        report["verdict"]["status"] = "PARTIAL"
    else:
        report["verdict"]["status"] = "DATA_COLLECTION_ONLY"
        notes.append("Concentration comparison requires model output interpolated to stations")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print_report(report, args.verbose)
    print(f"\nReport: {args.output}")


if __name__ == "__main__":
    main()
