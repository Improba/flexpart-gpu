#!/usr/bin/env python3
"""Clean comparison for Fortran vs GPU using a shared host-side gridding operator.

This script compares final particle snapshots at the same timestamp:
- Fortran: `partposit_end` (or explicit --fortran-partposit)
- GPU: host-exported final particle gridding from fortran-validation JSON

It also reports legacy GPU gridding metrics for reference.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def load_compare_module(repo_root: Path):
    module_path = repo_root / "scripts" / "compare_concentrations.py"
    spec = importlib.util.spec_from_file_location("compare_concentrations", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def horizontal_distance_km(com_a, com_b):
    if not com_a or not com_b:
        return None
    dlon = com_b["lon"] - com_a["lon"]
    dlat = com_b["lat"] - com_a["lat"]
    return ((dlon * 111.0) ** 2 + (dlat * 111.0) ** 2) ** 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to flexpart-gpu repository root (default: current directory)",
    )
    parser.add_argument(
        "--fortran-output",
        required=True,
        help="Path to Fortran output directory (contains partposit_end)",
    )
    parser.add_argument(
        "--gpu-output",
        required=True,
        help="Path to fortran-validation JSON output",
    )
    parser.add_argument(
        "--fortran-partposit",
        default="partposit_end",
        help="Fortran partposit filename relative to --fortran-output",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output JSON report path",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    fortran_output = Path(args.fortran_output).resolve()
    gpu_output = Path(args.gpu_output).resolve()
    fortran_partposit = fortran_output / args.fortran_partposit

    cc = load_compare_module(repo_root)

    with open(gpu_output) as f:
        gpu = json.load(f)

    gi = gpu["grid"]
    nx, ny, nz = gi["nx"], gi["ny"], gi["nz"]
    heights = gi["heights_m"]

    f_lons = f_lats = f_zs = None
    used_endian = None
    for endian in ["<", ">"]:
        lons, lats, zs = cc.read_partposit(str(fortran_partposit), endian)
        if lons is not None and len(lons) > 0:
            f_lons, f_lats, f_zs = lons, lats, zs
            used_endian = endian
            break
    if f_lons is None:
        raise SystemExit(f"Could not parse Fortran particle file: {fortran_partposit}")

    f_grid_flat = cc.grid_particles(
        f_lons,
        f_lats,
        f_zs,
        gi["xlon0"],
        gi["ylat0"],
        gi["dx"],
        gi["dy"],
        nx,
        ny,
        nz,
        heights,
    )

    g_legacy_flat = np.array(gpu["particle_count_per_cell"], dtype=np.float32)
    if "particle_count_per_cell_host_gridding" in gpu:
        g_clean_flat = np.array(gpu["particle_count_per_cell_host_gridding"], dtype=np.float32)
    else:
        g_clean_flat = g_legacy_flat.copy()

    f_norm = f_grid_flat / max(float(f_grid_flat.sum()), 1.0)
    g_legacy_norm = g_legacy_flat / max(float(g_legacy_flat.sum()), 1.0)
    g_clean_norm = g_clean_flat / max(float(g_clean_flat.sum()), 1.0)

    metrics_legacy = cc.compute_metrics(
        f_norm, g_legacy_norm, "Fortran(partposit-hostgrid) vs GPU(legacy-grid)"
    )
    metrics_clean = cc.compute_metrics(
        f_norm, g_clean_norm, "Fortran(partposit-hostgrid) vs GPU(clean-hostgrid)"
    )

    f_grid = f_grid_flat.reshape((nx, ny, nz))
    g_legacy = g_legacy_flat.reshape((nx, ny, nz))
    g_clean = g_clean_flat.reshape((nx, ny, nz))
    com_fortran = cc.compute_center_of_mass(
        f_grid, gi["xlon0"], gi["ylat0"], gi["dx"], gi["dy"], heights
    )
    com_gpu_legacy = cc.compute_center_of_mass(
        g_legacy, gi["xlon0"], gi["ylat0"], gi["dx"], gi["dy"], heights
    )
    com_gpu_clean = cc.compute_center_of_mass(
        g_clean, gi["xlon0"], gi["ylat0"], gi["dx"], gi["dy"], heights
    )

    report = {
        "fortran_particles": int(len(f_lons)),
        "gpu_particles_legacy": int(g_legacy_flat.sum()),
        "gpu_particles_clean": int(g_clean_flat.sum()),
        "used_endian": used_endian,
        "metrics_legacy": metrics_legacy,
        "metrics_clean": metrics_clean,
        "com_fortran": com_fortran,
        "com_gpu_legacy": com_gpu_legacy,
        "com_gpu_clean": com_gpu_clean,
        "horizontal_distance_km_legacy": horizontal_distance_km(com_fortran, com_gpu_legacy),
        "horizontal_distance_km_clean": horizontal_distance_km(com_fortran, com_gpu_clean),
    }

    print("=== Clean Comparison (partposit-aligned) ===")
    print(f"Fortran particles: {report['fortran_particles']}")
    print(f"GPU particles (legacy): {report['gpu_particles_legacy']}")
    print(f"GPU particles (clean):  {report['gpu_particles_clean']}")
    print(f"Correlation legacy: {metrics_legacy['correlation']:.6f}")
    print(f"Correlation clean:  {metrics_clean['correlation']:.6f}")
    print(f"NRMSE legacy: {metrics_legacy['normalized_rmse']:.4f}")
    print(f"NRMSE clean:  {metrics_clean['normalized_rmse']:.4f}")
    print(f"Hdist legacy (km): {report['horizontal_distance_km_legacy']:.2f}")
    print(f"Hdist clean  (km): {report['horizontal_distance_km_clean']:.2f}")

    if args.output_json:
        output_json = Path(args.output_json).resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        def _json_default(obj):
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        with open(output_json, "w") as f:
            json.dump(report, f, indent=2, default=_json_default)
        print(f"Report written to: {output_json}")


if __name__ == "__main__":
    main()
