#!/usr/bin/env python3
"""Generate synthetic GRIB1 files for FLEXPART Fortran comparison tests.

Creates ECMWF-style GRIB1 files with uniform wind fields on hybrid eta
levels so that FLEXPART can be run without real meteorological data.

Prerequisites:
    apt install python3-eccodes python3-numpy
    OR: pip install eccodes-python numpy

Usage:
    python3 generate_synthetic_grib.py \\
        --output-dir target/comparison/meteo \\
        --nx 32 --ny 32 --nz 12 \\
        --u-wind 0.5 --v-wind -0.3 --w-wind 0.0 \\
        --start-date 20240101 --hours 6
"""

import argparse
import os
import sys

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

# Simplified 12-level hybrid eta coefficients (13 half-level boundaries).
# Pressure at half-level k: p(k) = A(k) + B(k) * ps
# These are a rough approximation of ECMWF L60-like levels spanning
# ~10 hPa (top) to surface.
HALF_LEVEL_A = np.array([
    0.0,        2000.0,     5000.0,    10000.0,
    15000.0,    20000.0,    22000.0,   18000.0,
    12000.0,     6000.0,     2000.0,      500.0,
    0.0,
], dtype=np.float64)

HALF_LEVEL_B = np.array([
    0.0,        0.0,        0.0,        0.0,
    0.0,        0.0,        0.05,       0.15,
    0.30,       0.50,       0.70,       0.85,
    1.0,
], dtype=np.float64)


def write_grib1_message(fout, param_id: int, level_type: int, level: int,
                         nx: int, ny: int, values: np.ndarray,
                         date: int, time: int, nz: int):
    """Write a single GRIB1 message with full ECMWF-compatible metadata."""
    msgid = eccodes.codes_grib_new_from_samples("GRIB1")

    eccodes.codes_set(msgid, "centre", 98)  # ECMWF
    eccodes.codes_set(msgid, "editionNumber", 1)
    eccodes.codes_set(msgid, "table2Version", 128)
    eccodes.codes_set(msgid, "dataDate", date)
    eccodes.codes_set(msgid, "dataTime", time // 100)
    eccodes.codes_set(msgid, "indicatorOfParameter", param_id)
    eccodes.codes_set(msgid, "indicatorOfTypeOfLevel", level_type)
    eccodes.codes_set(msgid, "level", level)

    eccodes.codes_set(msgid, "gridType", "regular_ll")
    eccodes.codes_set(msgid, "Ni", nx)
    eccodes.codes_set(msgid, "Nj", ny)
    eccodes.codes_set(msgid, "iDirectionIncrementInDegrees", 360.0 / nx)
    eccodes.codes_set(msgid, "jDirectionIncrementInDegrees", 180.0 / (ny - 1))
    eccodes.codes_set(msgid, "latitudeOfFirstGridPointInDegrees", 90.0)
    eccodes.codes_set(msgid, "longitudeOfFirstGridPointInDegrees", 0.0)
    eccodes.codes_set(msgid, "latitudeOfLastGridPointInDegrees", -90.0)
    eccodes.codes_set(msgid, "longitudeOfLastGridPointInDegrees",
                      360.0 - 360.0 / nx)
    eccodes.codes_set(msgid, "jScansPositively", 0)

    # Hybrid eta level vertical coordinate (pv array)
    n_half = nz + 1
    pv = np.concatenate([HALF_LEVEL_A[:n_half], HALF_LEVEL_B[:n_half]])
    eccodes.codes_set(msgid, "PVPresent", 1)
    eccodes.codes_set_double_array(msgid, "pv", pv.tolist())

    eccodes.codes_set(msgid, "packingType", "grid_simple")
    eccodes.codes_set(msgid, "bitsPerValue", 16)
    eccodes.codes_set_values(msgid, values.flatten().astype(np.float64))

    eccodes.codes_write(msgid, fout)
    eccodes.codes_release(msgid)


LEVEL_SURFACE = 1
LEVEL_HYBRID = 109

PARAM_U = 131
PARAM_V = 132
PARAM_W = 135
PARAM_T = 130
PARAM_Q = 133
PARAM_SP = 134
PARAM_T2M = 167
PARAM_TD2M = 168
PARAM_U10M = 165
PARAM_V10M = 166
PARAM_SSHF = 146
PARAM_SSR = 176
PARAM_EWSS = 180
PARAM_NSSS = 181
PARAM_LSP = 142
PARAM_CP = 143
PARAM_BLH = 159     # boundary layer height
PARAM_LNSP = 152    # log of surface pressure


def generate_one_timestep(output_path: str, nx: int, ny: int, nz: int,
                           u_wind: float, v_wind: float, w_wind: float,
                           date: int, time: int):
    npoints = nx * ny
    uniform = lambda val: np.full(npoints, val, dtype=np.float64)

    with open(output_path, "wb") as fout:
        for k in range(1, nz + 1):
            write_grib1_message(fout, PARAM_U, LEVEL_HYBRID, k, nx, ny,
                                uniform(u_wind), date, time, nz)
            write_grib1_message(fout, PARAM_V, LEVEL_HYBRID, k, nx, ny,
                                uniform(v_wind), date, time, nz)
            write_grib1_message(fout, PARAM_W, LEVEL_HYBRID, k, nx, ny,
                                uniform(w_wind), date, time, nz)
            temp_k = max(220.0 + 6.5 * (nz - k), 200.0)
            write_grib1_message(fout, PARAM_T, LEVEL_HYBRID, k, nx, ny,
                                uniform(temp_k), date, time, nz)
            qv = 0.001 + 0.009 * (k / nz)
            write_grib1_message(fout, PARAM_Q, LEVEL_HYBRID, k, nx, ny,
                                uniform(qv), date, time, nz)

        # Surface fields
        write_grib1_message(fout, PARAM_SP, LEVEL_SURFACE, 0, nx, ny,
                            uniform(101325.0), date, time, nz)
        write_grib1_message(fout, PARAM_LNSP, LEVEL_SURFACE, 1, nx, ny,
                            uniform(np.log(101325.0)), date, time, nz)
        write_grib1_message(fout, PARAM_T2M, LEVEL_SURFACE, 0, nx, ny,
                            uniform(289.0), date, time, nz)
        write_grib1_message(fout, PARAM_TD2M, LEVEL_SURFACE, 0, nx, ny,
                            uniform(284.0), date, time, nz)
        write_grib1_message(fout, PARAM_U10M, LEVEL_SURFACE, 0, nx, ny,
                            uniform(u_wind), date, time, nz)
        write_grib1_message(fout, PARAM_V10M, LEVEL_SURFACE, 0, nx, ny,
                            uniform(v_wind), date, time, nz)
        write_grib1_message(fout, PARAM_SSHF, LEVEL_SURFACE, 0, nx, ny,
                            uniform(40.0), date, time, nz)
        write_grib1_message(fout, PARAM_SSR, LEVEL_SURFACE, 0, nx, ny,
                            uniform(220.0), date, time, nz)
        write_grib1_message(fout, PARAM_EWSS, LEVEL_SURFACE, 0, nx, ny,
                            uniform(0.1), date, time, nz)
        write_grib1_message(fout, PARAM_NSSS, LEVEL_SURFACE, 0, nx, ny,
                            uniform(0.1), date, time, nz)
        write_grib1_message(fout, PARAM_LSP, LEVEL_SURFACE, 0, nx, ny,
                            uniform(0.0), date, time, nz)
        write_grib1_message(fout, PARAM_CP, LEVEL_SURFACE, 0, nx, ny,
                            uniform(0.0), date, time, nz)
        write_grib1_message(fout, PARAM_BLH, LEVEL_SURFACE, 0, nx, ny,
                            uniform(800.0), date, time, nz)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nz", type=int, default=12)
    parser.add_argument("--u-wind", type=float, default=0.5)
    parser.add_argument("--v-wind", type=float, default=-0.3)
    parser.add_argument("--w-wind", type=float, default=0.0)
    parser.add_argument("--start-date", default="20240101")
    parser.add_argument("--hours", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    date = int(args.start_date)
    for hour in range(0, args.hours + 1, 3):
        time_hhmmss = hour * 10000
        filename = f"EN{args.start_date}{hour:02d}"
        output_path = os.path.join(args.output_dir, filename)

        print(f"  {filename}  (date={date}, time={time_hhmmss:06d})")
        generate_one_timestep(
            output_path, args.nx, args.ny, args.nz,
            args.u_wind, args.v_wind, args.w_wind,
            date, time_hhmmss,
        )

    available_path = os.path.join(args.output_dir, "AVAILABLE")
    with open(available_path, "w") as f:
        f.write("DATE      TIME     FILENAME     SPECIFICATIONS\n")
        f.write("YYYYMMDD  HHMMSS\n")
        f.write("________ ________ __________ __________\n")
        for hour in range(0, args.hours + 1, 3):
            fname = f"EN{args.start_date}{hour:02d}"
            f.write(f"{args.start_date} {hour * 10000:06d}      "
                    f"{fname}      ON DISC\n")

    count = args.hours // 3 + 1
    print(f"\nGenerated {count} GRIB1 files + AVAILABLE in {args.output_dir}/")


if __name__ == "__main__":
    main()
