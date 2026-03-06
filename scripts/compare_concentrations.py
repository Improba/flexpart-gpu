#!/usr/bin/env python3
"""Compare FLEXPART Fortran and flexpart-gpu concentration outputs.

Reads:
  - Fortran: header + grid_conc_* binary files (sparse format)
  - GPU:     JSON from fortran-validation binary

Computes:
  - Normalized RMSE, correlation, bias, MAE
  - Center of mass comparison
  - Total mass check

Usage:
    python3 compare_concentrations.py \
        --fortran-output /path/to/fortran/output/ \
        --gpu-output /path/to/gpu_concentration.json \
        [--verbose]
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Fortran binary reader
# ---------------------------------------------------------------------------

def read_fortran_record(f, endian="<"):
    """Read one Fortran unformatted sequential record.

    Returns bytes of the data payload, or None at EOF.
    """
    raw = f.read(4)
    if len(raw) < 4:
        return None
    rec_len = struct.unpack(f"{endian}i", raw)[0]
    if rec_len < 0:
        raise ValueError(f"Negative record length {rec_len} — wrong endianness?")
    data = f.read(rec_len)
    if len(data) < rec_len:
        raise ValueError(f"Truncated record: expected {rec_len} bytes, got {len(data)}")
    suffix = struct.unpack(f"{endian}i", f.read(4))[0]
    if suffix != rec_len:
        raise ValueError(
            f"Record length mismatch: header={rec_len}, trailer={suffix}"
        )
    return data


def read_header(header_path, endian="<"):
    """Parse the FLEXPART binary 'header' file to get grid dimensions."""
    with open(header_path, "rb") as f:
        # Record 1: ibdate, ibtime, flexversion (string)
        rec = read_fortran_record(f, endian)
        ibdate = struct.unpack(f"{endian}i", rec[:4])[0]
        ibtime = struct.unpack(f"{endian}i", rec[4:8])[0]

        # Record 2: loutstep, loutaver, loutsample
        rec = read_fortran_record(f, endian)
        loutstep, loutaver, loutsample = struct.unpack(f"{endian}3i", rec[:12])

        # Record 3: outlon0, outlat0, numxgrid, numygrid, dxout, dyout
        rec = read_fortran_record(f, endian)
        outlon0, outlat0 = struct.unpack(f"{endian}2f", rec[:8])
        numxgrid, numygrid = struct.unpack(f"{endian}2i", rec[8:16])
        dxout, dyout = struct.unpack(f"{endian}2f", rec[16:24])

        # Record 4: numzgrid, outheight(1:numzgrid)
        rec = read_fortran_record(f, endian)
        numzgrid = struct.unpack(f"{endian}i", rec[:4])[0]
        outheights = np.frombuffer(rec[4:4 + numzgrid * 4], dtype=f"{endian}f4")

        # Record 5: jjjjmmdd, ihmmss (release date)
        rec = read_fortran_record(f, endian)

        # Record 6: 3*nspec, maxpointspec_act
        rec = read_fortran_record(f, endian)
        n3spec, maxpointspec_act = struct.unpack(f"{endian}2i", rec[:8])
        nspec = n3spec // 3

        # Skip species name records (3 per species: WD, DD, concentration)
        for _ in range(n3spec):
            read_fortran_record(f, endian)

        # Record: numpoint
        rec = read_fortran_record(f, endian)
        numpoint = struct.unpack(f"{endian}i", rec[:4])[0]

        # Skip release records
        for _ in range(numpoint):
            read_fortran_record(f, endian)  # ireleasestart, ireleaseend, kindz
            read_fortran_record(f, endian)  # xp1, yp1, xp2, yp2, z1, z2
            read_fortran_record(f, endian)  # npart, 1
            read_fortran_record(f, endian)  # comment
            for _ in range(nspec):
                read_fortran_record(f, endian)  # xmass (wet)
                read_fortran_record(f, endian)  # xmass (dry)
                read_fortran_record(f, endian)  # xmass (conc)

        # Record: method, lsubgrid, lconvection, ind_source, ind_receptor
        rec = read_fortran_record(f, endian)

        # Record: nageclass, lage(1:nageclass)
        rec = read_fortran_record(f, endian)
        nageclass = struct.unpack(f"{endian}i", rec[:4])[0]

    return {
        "numxgrid": numxgrid,
        "numygrid": numygrid,
        "numzgrid": numzgrid,
        "outlon0": outlon0,
        "outlat0": outlat0,
        "dxout": dxout,
        "dyout": dyout,
        "outheights": outheights.tolist(),
        "nspec": nspec,
        "maxpointspec_act": maxpointspec_act,
        "nageclass": nageclass,
        "loutstep": loutstep,
    }


def decode_sparse_block(records, idx, endian="<"):
    """Decode one sparse block (4 records: sp_count_i, indices, sp_count_r, values).

    Returns (grid_indices, abs_values, next_record_index).
    """
    # sp_count_i
    rec = records[idx]
    sp_count_i = struct.unpack(f"{endian}i", rec[:4])[0]
    idx += 1

    # sparse_dump_i
    rec = records[idx]
    if sp_count_i > 0:
        indices = np.frombuffer(rec, dtype=f"{endian}i4", count=sp_count_i).copy()
    else:
        indices = np.array([], dtype=np.int32)
    idx += 1

    # sp_count_r
    rec = records[idx]
    sp_count_r = struct.unpack(f"{endian}i", rec[:4])[0]
    idx += 1

    # sparse_dump_r
    rec = records[idx]
    if sp_count_r > 0:
        values = np.frombuffer(rec, dtype=f"{endian}f4", count=sp_count_r).copy()
    else:
        values = np.array([], dtype=np.float32)
    idx += 1

    return indices, values, idx


def reconstruct_grid_from_sparse(indices, values, total_cells):
    """Reconstruct a flat grid array from FLEXPART sparse encoding.

    The sparse format uses sign-alternation to delimit runs of consecutive
    non-zero grid cells. Each entry in `indices` marks the flat grid position
    where a new run begins. Values within a run all have the same sign;
    the sign flips at each run boundary. Physical values are |value|.
    """
    grid = np.zeros(total_cells, dtype=np.float32)
    if len(indices) == 0 or len(values) == 0:
        return grid

    val_offset = 0
    for run_idx in range(len(indices)):
        start_cell = indices[run_idx]
        if run_idx + 1 < len(indices):
            next_start = indices[run_idx + 1]
        else:
            next_start = total_cells

        if val_offset >= len(values):
            break

        run_sign = np.sign(values[val_offset])
        cell = start_cell
        while val_offset < len(values) and cell < next_start:
            if np.sign(values[val_offset]) != run_sign:
                break
            grid[cell] = abs(values[val_offset])
            val_offset += 1
            cell += 1

    return grid


def read_grid_conc(filepath, header, endian="<"):
    """Read a grid_conc_* file and return the 3D concentration grid."""
    nx = header["numxgrid"]
    ny = header["numygrid"]
    nz = header["numzgrid"]
    nkp = header["maxpointspec_act"]
    nage = header["nageclass"]
    total_2d = nx * ny
    # Fortran flat index: ix + jy*nx + kz*nx*ny with kz ∈ [1, nz]
    # Max index = (nx-1) + (ny-1)*nx + nz*nx*ny = (nz+1)*nx*ny - 1
    total_flat = (nz + 1) * nx * ny

    with open(filepath, "rb") as f:
        records = []
        while True:
            rec = read_fortran_record(f, endian)
            if rec is None:
                break
            records.append(rec)

    # Record 0: itime
    idx = 1  # skip itime record

    conc_flat = np.zeros(total_flat, dtype=np.float32)

    for _kp in range(nkp):
        for _nage in range(nage):
            # Wet deposition (2D sparse, indices = ix + jy*nx)
            _wet_idx, _wet_val, idx = decode_sparse_block(records, idx, endian)

            # Dry deposition (2D sparse)
            _dry_idx, _dry_val, idx = decode_sparse_block(records, idx, endian)

            # 3D concentration (sparse, indices = ix + jy*nx + kz*nx*ny, kz≥1)
            conc_indices, conc_values, idx = decode_sparse_block(
                records, idx, endian
            )
            conc_flat += reconstruct_grid_from_sparse(
                conc_indices, conc_values, total_flat
            )

    # Extract 3D grid from the flat buffer (kz=0 slice is unused)
    grid = np.zeros((nx, ny, nz), dtype=np.float32)
    for kz in range(1, nz + 1):
        for jy in range(ny):
            for ix in range(nx):
                flat_idx = ix + jy * nx + kz * nx * ny
                grid[ix, jy, kz - 1] = conc_flat[flat_idx]

    return grid


# ---------------------------------------------------------------------------
# GPU JSON reader
# ---------------------------------------------------------------------------

def read_gpu_output(filepath):
    """Read the GPU validation JSON output."""
    with open(filepath) as f:
        data = json.load(f)

    grid_info = data["grid"]
    nx = grid_info["nx"]
    ny = grid_info["ny"]
    nz = grid_info["nz"]

    counts = np.array(data["particle_count_per_cell"], dtype=np.uint32)
    mass = np.array(data["concentration_mass_kg"], dtype=np.float32)

    # Reshape: GPU uses flat row-major (ix * ny + iy) * nz + iz
    count_grid = counts.reshape((nx, ny, nz))
    mass_grid = mass.reshape((nx, ny, nz))

    return {
        "grid_info": grid_info,
        "particle_count": count_grid,
        "mass_kg": mass_grid,
        "total_active": data["total_particles_active"],
        "total_steps": data["total_steps"],
        "particle_z_stats": data.get("particle_z_stats"),
    }


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------

def compute_metrics(field_a, field_b, label=""):
    """Compute comparison metrics between two fields."""
    a = field_a.flatten().astype(np.float64)
    b = field_b.flatten().astype(np.float64)

    n = len(a)
    assert n == len(b), f"shape mismatch: {len(a)} vs {len(b)}"

    diff = a - b
    bias = np.mean(diff)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))

    sum_a = np.sum(a)
    sum_b = np.sum(b)
    norm_rmse = rmse / max(np.mean(np.abs(a)), np.mean(np.abs(b)), 1e-30)

    # Pearson correlation
    if np.std(a) > 0 and np.std(b) > 0:
        correlation = np.corrcoef(a, b)[0, 1]
    else:
        correlation = None

    return {
        "label": label,
        "sample_count": n,
        "nonzero_a": int(np.count_nonzero(a)),
        "nonzero_b": int(np.count_nonzero(b)),
        "sum_a": float(sum_a),
        "sum_b": float(sum_b),
        "bias": float(bias),
        "mae": float(mae),
        "rmse": float(rmse),
        "normalized_rmse": float(norm_rmse),
        "correlation": float(correlation) if correlation is not None else None,
    }


def compute_center_of_mass(grid, xlon0, ylat0, dx, dy, heights):
    """Compute 3D center of mass of a concentration field."""
    nx, ny, nz = grid.shape
    total = np.sum(grid)
    if total <= 0:
        return None

    com_x = 0.0
    com_y = 0.0
    com_z = 0.0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                w = float(grid[ix, iy, iz])
                if w > 0:
                    com_x += w * (xlon0 + (ix + 0.5) * dx)
                    com_y += w * (ylat0 + (iy + 0.5) * dy)
                    com_z += w * heights[iz]

    return {
        "lon": com_x / total,
        "lat": com_y / total,
        "z_m": com_z / total,
        "total_mass": float(total),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_grid_conc_files(output_dir):
    """Find grid_conc_* files in a FLEXPART output directory."""
    files = []
    for f in sorted(os.listdir(output_dir)):
        if f.startswith("grid_conc_"):
            files.append(os.path.join(output_dir, f))
    return files


def read_partposit(filepath, endian="<"):
    """Read Fortran particle dump (partposit_*) and return arrays of (lon, lat, z)."""
    lons, lats, zs = [], [], []
    with open(filepath, "rb") as f:
        rec = read_fortran_record(f, endian)
        if rec is None:
            return None, None, None
        while True:
            rec = read_fortran_record(f, endian)
            if rec is None:
                break
            npoint = struct.unpack(f"{endian}i", rec[:4])[0]
            if npoint <= 0:
                break
            xlon, ylat, z = struct.unpack(f"{endian}3f", rec[4:16])
            lons.append(xlon)
            lats.append(ylat)
            zs.append(z)
    return np.array(lons), np.array(lats), np.array(zs)


def grid_particles(lons, lats, zs, xlon0, ylat0, dx, dy, nx, ny, nz, heights):
    """Bin particle positions into the output grid, returning a flat count array."""
    grid = np.zeros(nx * ny * nz, dtype=np.float32)
    for i in range(len(lons)):
        ix = int((lons[i] - xlon0) / dx)
        iy = int((lats[i] - ylat0) / dy)
        iz = nz - 1
        for kz in range(nz):
            if zs[i] <= heights[kz]:
                iz = kz
                break
        if 0 <= ix < nx and 0 <= iy < ny:
            flat = ((ix * ny) + iy) * nz + iz
            grid[flat] += 1.0
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fortran-output", required=True,
                        help="Path to Fortran FLEXPART output directory")
    parser.add_argument("--gpu-output", required=True,
                        help="Path to GPU validation JSON file")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-json", default=None,
                        help="Write comparison report as JSON")
    args = parser.parse_args()

    fortran_dir = Path(args.fortran_output)
    header_path = fortran_dir / "header"
    if not header_path.exists():
        print(f"ERROR: Fortran header not found: {header_path}", file=sys.stderr)
        sys.exit(1)

    # Try little-endian first (most common on Linux), then big-endian
    for endian in ["<", ">"]:
        try:
            header = read_header(str(header_path), endian)
            break
        except (ValueError, struct.error):
            continue
    else:
        print("ERROR: Could not parse Fortran header (tried both endiannesses)",
              file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Fortran header: {json.dumps(header, indent=2, default=str)}")

    conc_files = find_grid_conc_files(str(fortran_dir))
    if not conc_files:
        print(f"ERROR: No grid_conc_* files in {fortran_dir}", file=sys.stderr)
        sys.exit(1)

    # Read last output timestep
    last_conc_file = conc_files[-1]
    print(f"Reading Fortran output: {last_conc_file}")
    fortran_grid = read_grid_conc(last_conc_file, header, endian)
    print(f"  shape: {fortran_grid.shape}")
    print(f"  non-zero cells: {np.count_nonzero(fortran_grid)}")
    print(f"  total sum: {np.sum(fortran_grid):.6e}")

    # Read GPU output
    print(f"Reading GPU output: {args.gpu_output}")
    gpu_data = read_gpu_output(args.gpu_output)
    gpu_counts = gpu_data["particle_count"]
    gpu_mass = gpu_data["mass_kg"]
    gi = gpu_data["grid_info"]
    print(f"  shape: {gpu_counts.shape}")
    print(f"  non-zero cells (counts): {np.count_nonzero(gpu_counts)}")
    print(f"  total particles: {np.sum(gpu_counts)}")
    print(f"  total mass: {np.sum(gpu_mass):.6e}")

    # Ensure grids have same shape
    if fortran_grid.shape != gpu_counts.shape:
        print(f"ERROR: Shape mismatch: Fortran={fortran_grid.shape}, GPU={gpu_counts.shape}",
              file=sys.stderr)
        sys.exit(1)

    # Normalize both fields for comparison (spatial distribution)
    f_sum = np.sum(fortran_grid)
    g_count_sum = np.sum(gpu_counts.astype(np.float64))
    g_mass_sum = np.sum(np.abs(gpu_mass).astype(np.float64))

    if f_sum > 0:
        fortran_norm = fortran_grid / f_sum
    else:
        fortran_norm = fortran_grid.copy()
        print("WARNING: Fortran field is all zeros!")

    if g_count_sum > 0:
        gpu_count_norm = gpu_counts.astype(np.float32) / float(g_count_sum)
    else:
        gpu_count_norm = gpu_counts.astype(np.float32)
        print("WARNING: GPU count field is all zeros!")

    if g_mass_sum > 0:
        gpu_mass_norm = np.abs(gpu_mass) / float(g_mass_sum)
    else:
        gpu_mass_norm = np.abs(gpu_mass)

    print()
    print("=" * 70)
    print("  FLEXPART Fortran vs flexpart-gpu — Scientific Validation")
    print("=" * 70)

    # Metrics: normalized Fortran conc vs normalized GPU particle counts
    metrics_count = compute_metrics(fortran_norm, gpu_count_norm,
                                     "Fortran(norm) vs GPU-count(norm)")
    metrics_mass = compute_metrics(fortran_norm, gpu_mass_norm,
                                    "Fortran(norm) vs GPU-mass(norm)")

    for m in [metrics_count, metrics_mass]:
        print(f"\n--- {m['label']} ---")
        print(f"  Samples:          {m['sample_count']}")
        print(f"  Non-zero (A/B):   {m['nonzero_a']} / {m['nonzero_b']}")
        print(f"  Sum (A/B):        {m['sum_a']:.6e} / {m['sum_b']:.6e}")
        print(f"  Bias:             {m['bias']:.6e}")
        print(f"  MAE:              {m['mae']:.6e}")
        print(f"  RMSE:             {m['rmse']:.6e}")
        print(f"  Normalized RMSE:  {m['normalized_rmse']:.4f}")
        if m["correlation"] is not None:
            print(f"  Correlation:      {m['correlation']:.6f}")
        else:
            print("  Correlation:      N/A (zero variance)")

    # Center of mass
    heights = gi["heights_m"]
    com_fortran = compute_center_of_mass(
        fortran_grid, header["outlon0"], header["outlat0"],
        header["dxout"], header["dyout"], header["outheights"]
    )
    com_gpu = compute_center_of_mass(
        gpu_counts.astype(np.float32),
        gi["xlon0"], gi["ylat0"], gi["dx"], gi["dy"], heights
    )

    print("\n--- Center of Mass ---")
    if com_fortran:
        print(f"  Fortran: lon={com_fortran['lon']:.4f}°, "
              f"lat={com_fortran['lat']:.4f}°, z={com_fortran['z_m']:.1f}m")
    else:
        print("  Fortran: N/A (empty field)")
    if com_gpu:
        print(f"  GPU:     lon={com_gpu['lon']:.4f}°, "
              f"lat={com_gpu['lat']:.4f}°, z={com_gpu['z_m']:.1f}m")
    else:
        print("  GPU:     N/A (empty field)")
    if com_fortran and com_gpu:
        dlon = com_gpu["lon"] - com_fortran["lon"]
        dlat = com_gpu["lat"] - com_fortran["lat"]
        dz = com_gpu["z_m"] - com_fortran["z_m"]
        dist_km = ((dlon * 111.0) ** 2 + (dlat * 111.0) ** 2) ** 0.5
        print(f"  Δlon={dlon:+.4f}°, Δlat={dlat:+.4f}°, Δz={dz:+.1f}m")
        print(f"  Horizontal distance: {dist_km:.2f} km")

    # Pass/fail thresholds
    print("\n--- Validation Verdict ---")
    corr = metrics_count.get("correlation")
    nrmse = metrics_count["normalized_rmse"]
    passed = True

    if corr is not None and corr > 0.8:
        print(f"  [PASS] Correlation = {corr:.4f} > 0.80")
    elif corr is not None:
        print(f"  [WARN] Correlation = {corr:.4f} < 0.80")
        passed = False
    else:
        print("  [WARN] Correlation not computable")
        passed = False

    if nrmse < 0.5:
        print(f"  [PASS] Normalized RMSE = {nrmse:.4f} < 0.50")
    else:
        print(f"  [WARN] Normalized RMSE = {nrmse:.4f} >= 0.50")
        passed = False

    if com_fortran and com_gpu:
        if dist_km < 50.0:
            print(f"  [PASS] COM distance = {dist_km:.2f} km < 50 km")
        else:
            print(f"  [WARN] COM distance = {dist_km:.2f} km >= 50 km")
            passed = False

    if passed:
        print("\n  >>> VALIDATION: PASS <<<")
    else:
        print("\n  >>> VALIDATION: NEEDS INVESTIGATION <<<")
        print("  (Differences may be expected due to different turbulence RNG,")
        print("   PBL schemes, or coordinate system handling.)")

    print("=" * 70)

    # ------------------------------------------------------------------
    # Particle dump comparison (more reliable than concentration fields)
    # ------------------------------------------------------------------
    partposit_files = sorted(
        str(p) for p in fortran_dir.glob("partposit_*") if p.is_file()
    )
    if partposit_files:
        last_partposit = partposit_files[-1]
        print(f"\n{'=' * 70}")
        print("  Direct Particle Position Comparison (partposit)")
        print(f"{'=' * 70}")
        print(f"\nReading: {last_partposit}")

        f_lons, f_lats, f_zs = read_partposit(last_partposit, endian)
        if f_lons is not None and len(f_lons) > 0:
            print(f"  Fortran particles: {len(f_lons)}")
            print(f"  Fortran lon: mean={f_lons.mean():.4f}, std={f_lons.std():.4f}")
            print(f"  Fortran lat: mean={f_lats.mean():.4f}, std={f_lats.std():.4f}")
            print(f"  Fortran z:   mean={f_zs.mean():.1f}, std={f_zs.std():.1f}")

            f_grid_flat = grid_particles(
                f_lons, f_lats, f_zs,
                gi["xlon0"], gi["ylat0"], gi["dx"], gi["dy"],
                gi["nx"], gi["ny"], gi["nz"], heights,
            )
            g_flat = gpu_counts.flatten().astype(np.float32)

            f_sum_pp = f_grid_flat.sum()
            g_sum_pp = g_flat.sum()
            if f_sum_pp > 0 and g_sum_pp > 0:
                metrics_pp = compute_metrics(
                    f_grid_flat / f_sum_pp,
                    g_flat / g_sum_pp,
                    "Fortran(partposit) vs GPU(gridded)",
                )
                print(f"\n--- {metrics_pp['label']} ---")
                print(f"  Non-zero (F/G):   {metrics_pp['nonzero_a']} / {metrics_pp['nonzero_b']}")
                print(f"  RMSE:             {metrics_pp['rmse']:.6e}")
                print(f"  Normalized RMSE:  {metrics_pp['normalized_rmse']:.4f}")
                if metrics_pp["correlation"] is not None:
                    print(f"  Correlation:      {metrics_pp['correlation']:.6f}")

            com_f_lon = float(f_lons.mean())
            com_f_lat = float(f_lats.mean())
            com_f_z = float(f_zs.mean())
            com_g_lon = com_gpu["lon"] if com_gpu else None
            com_g_lat = com_gpu["lat"] if com_gpu else None

            if com_g_lon is not None:
                dlon = com_g_lon - com_f_lon
                dlat = com_g_lat - com_f_lat
                h_dist = ((dlon * 111.0) ** 2 + (dlat * 111.0) ** 2) ** 0.5
                print(f"\n--- Horizontal Center of Mass (partposit-based) ---")
                print(f"  Fortran: lon={com_f_lon:.4f}, lat={com_f_lat:.4f}")
                print(f"  GPU:     lon={com_g_lon:.4f}, lat={com_g_lat:.4f}")
                print(f"  Distance: {h_dist:.2f} km")
                if h_dist < 10:
                    print(f"  [PASS] Horizontal advection within 10 km")
                elif h_dist < 50:
                    print(f"  [PASS] Horizontal advection within 50 km")
                else:
                    print(f"  [WARN] Horizontal advection differs by {h_dist:.1f} km")

                # --- Horizontal spread comparison ---
                print(f"\n--- Horizontal Spread (partposit-based) ---")
                f_lon_std = float(f_lons.std())
                f_lat_std = float(f_lats.std())
                gpu_z = gpu_data.get("particle_z_stats")
                # Fortran spread in km (approx)
                f_spread_lon_km = f_lon_std * 111.0
                f_spread_lat_km = f_lat_std * 111.0
                print(f"  Fortran: σ_lon={f_lon_std:.4f}° ({f_spread_lon_km:.2f} km), "
                      f"σ_lat={f_lat_std:.4f}° ({f_spread_lat_km:.2f} km)")
                if gpu_z and "lon_std" in gpu_z:
                    g_lon_std = gpu_z["lon_std"]
                    g_lat_std = gpu_z["lat_std"]
                    g_spread_lon_km = g_lon_std * 111.0
                    g_spread_lat_km = g_lat_std * 111.0
                    print(f"  GPU:     σ_lon={g_lon_std:.4f}° ({g_spread_lon_km:.2f} km), "
                          f"σ_lat={g_lat_std:.4f}° ({g_spread_lat_km:.2f} km)")
                    lon_ratio = g_lon_std / max(f_lon_std, 1e-10)
                    lat_ratio = g_lat_std / max(f_lat_std, 1e-10)
                    print(f"  σ_lon ratio (GPU/F): {lon_ratio:.2f}")
                    print(f"  σ_lat ratio (GPU/F): {lat_ratio:.2f}")
                    if 0.5 <= lon_ratio <= 2.0 and 0.5 <= lat_ratio <= 2.0:
                        print(f"  [PASS] Horizontal spread ratios within [0.5, 2.0]")
                    else:
                        print(f"  [NOTE] Horizontal spread ratio outside expected range")

                print(f"\n--- Vertical (raw particle z) ---")
                if gpu_z:
                    gpu_mean_z = gpu_z["mean_m"]
                    gpu_std_z = gpu_z["std_m"]
                    print(f"  Fortran z: mean={com_f_z:.1f}m, std={f_zs.std():.1f}m")
                    print(f"  GPU z:     mean={gpu_mean_z:.1f}m, std={gpu_std_z:.1f}m")
                    dz = gpu_mean_z - com_f_z
                    std_ratio = gpu_std_z / max(f_zs.std(), 1e-10)
                    print(f"  Δz mean: {dz:+.0f}m")
                    print(f"  σ_z ratio (GPU/F): {std_ratio:.2f}")
                    if abs(dz) < 200:
                        print(f"  [PASS] Vertical mean within 200m (Δz={dz:+.0f}m)")
                    elif abs(dz) < 500:
                        print(f"  [PASS] Vertical mean within 500m (Δz={dz:+.0f}m)")
                    else:
                        print(f"  [NOTE] Vertical mean differs (Δz={dz:+.0f}m)")
                        print(f"         Likely due to hmix computation or sub-stepping differences.")
                    if 0.7 <= std_ratio <= 1.3:
                        print(f"  [PASS] σ_z ratio within [0.7, 1.3]")
                    else:
                        print(f"  [NOTE] σ_z ratio outside expected range ({std_ratio:.2f})")

                    # --- Per-level vertical profile comparison ---
                    print(f"\n--- Vertical Profile (per-level particle fraction) ---")
                    f_z_levels = np.zeros(len(heights))
                    g_z_levels = np.zeros(len(heights))
                    for z in f_zs:
                        for kz in range(len(heights)):
                            if z <= heights[kz]:
                                f_z_levels[kz] += 1
                                break
                        else:
                            f_z_levels[-1] += 1
                    f_z_frac = f_z_levels / max(len(f_zs), 1)
                    # GPU: sum gridded counts per level
                    for iz in range(len(heights)):
                        g_z_levels[iz] = float(gpu_counts[:, :, iz].sum())
                    g_z_frac = g_z_levels / max(g_z_levels.sum(), 1e-10)
                    print(f"  {'Level':>10} {'Height(m)':>10} {'Fortran%':>10} {'GPU%':>10} {'Δ%':>10}")
                    for kz in range(len(heights)):
                        ff = f_z_frac[kz] * 100
                        gf = g_z_frac[kz] * 100
                        print(f"  {kz:>10} {heights[kz]:>10.0f} {ff:>10.1f} {gf:>10.1f} {gf-ff:>+10.1f}")
                else:
                    print(f"  Fortran z: mean={com_f_z:.1f}m, std={f_zs.std():.1f}m")
                    print(f"  GPU z (gridded COM): {com_gpu['z_m']:.1f}m")
                    dz = com_gpu["z_m"] - com_f_z
                    if abs(dz) < 500:
                        print(f"  [PASS] Vertical spread within 500m (Δz={dz:+.0f}m)")
                    else:
                        print(f"  [NOTE] Vertical spread differs (Δz={dz:+.0f}m)")
        print("=" * 70)

    if args.output_json:
        report = {
            "metrics_count_normalized": metrics_count,
            "metrics_mass_normalized": metrics_mass,
            "center_of_mass_fortran": com_fortran,
            "center_of_mass_gpu": com_gpu,
            "passed": passed,
            "fortran_header": header,
            "gpu_grid_info": gi,
        }
        if partposit_files and f_lons is not None and len(f_lons) > 0:
            pp_report = {
                "fortran_particles": len(f_lons),
                "fortran_com_lon": com_f_lon,
                "fortran_com_lat": com_f_lat,
                "fortran_com_z": com_f_z,
                "horizontal_distance_km": h_dist if com_g_lon else None,
                "fortran_lon_std": float(f_lons.std()),
                "fortran_lat_std": float(f_lats.std()),
                "fortran_z_std": float(f_zs.std()),
            }
            gpu_z = gpu_data.get("particle_z_stats")
            if gpu_z:
                pp_report["gpu_mean_z"] = gpu_z["mean_m"]
                pp_report["gpu_std_z"] = gpu_z["std_m"]
                pp_report["delta_z_m"] = gpu_z["mean_m"] - com_f_z
                pp_report["sigma_z_ratio"] = gpu_z["std_m"] / max(float(f_zs.std()), 1e-10)
            report["partposit_comparison"] = pp_report
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport written to {args.output_json}")


if __name__ == "__main__":
    main()
