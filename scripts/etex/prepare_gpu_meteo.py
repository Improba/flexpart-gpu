#!/usr/bin/env python3
"""Pack ERA5 numpy arrays into raw binary files for flexpart-gpu etex-run.

Reads the numpy arrays downloaded by download_era5_gcs.py and writes
per-timestep binary files containing WindField3D + SurfaceFields data
in the exact memory layout expected by the Rust binary.

Layout per .bin file (all little-endian f32, C-contiguous (nx, ny, nz)):
  3D fields (8): u, v, w, T, q, pressure, air_density, density_gradient
  2D fields (15): sp, u10, v10, t2m, td2m, lsp, cp, sshf, ssr,
                  surfstr, ustar, wstar, hmix, tropo, oli

Usage:
    python3 scripts/etex/prepare_gpu_meteo.py \
        --era5-dir target/etex/era5_raw \
        --output-dir target/etex/gpu_meteo
"""
import argparse
import json
import os
import struct

import numpy as np

R_DRY = 287.058  # J/(kg·K) specific gas constant for dry air
G = 9.80665

PRESSURE_LEVELS_HPA = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50]

STD_HEIGHTS_M = [111, 762, 1457, 3013, 4206, 5574, 7185, 9164, 10363, 11784, 13608, 16180, 18442, 20576]


def load_npy(era5_dir, name):
    return np.load(os.path.join(era5_dir, f"{name}.npy"))


def prepare_3d(data_4d, t_idx):
    """Extract one timestep from (time, level, lat, lon) and transpose to (lon, lat, level).

    ERA5 latitudes are north-to-south; we flip to south-to-north (ylat0 = southernmost).
    """
    field = data_4d[t_idx]  # (nz, nlat, nlon)
    field = field[:, ::-1, :]  # flip lat to S→N
    return np.ascontiguousarray(field.transpose(2, 1, 0), dtype=np.float32)  # (nlon, nlat, nz)


def prepare_2d(data_3d, t_idx):
    """Extract one timestep from (time, lat, lon) and transpose to (lon, lat)."""
    field = data_3d[t_idx]  # (nlat, nlon)
    field = field[::-1, :]  # flip lat to S→N
    return np.ascontiguousarray(field.T, dtype=np.float32)  # (nlon, nlat)


def write_timestep(path, wind3d_arrays, surface_arrays):
    """Write all fields for one timestep as concatenated raw f32."""
    with open(path, "wb") as f:
        for arr in wind3d_arrays:
            f.write(arr.tobytes())
        for arr in surface_arrays:
            f.write(arr.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--era5-dir", default="target/etex/era5_raw")
    parser.add_argument("--output-dir", default="target/etex/gpu_meteo")
    parser.add_argument("--interval-hours", type=int, default=3,
                        help="Met bracket interval in hours (default: 3)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading ERA5 numpy arrays...")
    u = load_npy(args.era5_dir, "u_component_of_wind")
    v = load_npy(args.era5_dir, "v_component_of_wind")
    w = load_npy(args.era5_dir, "vertical_velocity")
    t = load_npy(args.era5_dir, "temperature")
    q = load_npy(args.era5_dir, "specific_humidity")
    sp = load_npy(args.era5_dir, "surface_pressure")
    u10 = load_npy(args.era5_dir, "10m_u_component_of_wind")
    v10 = load_npy(args.era5_dir, "10m_v_component_of_wind")
    t2m = load_npy(args.era5_dir, "2m_temperature")
    td2m = load_npy(args.era5_dir, "2m_dewpoint_temperature")
    blh = load_npy(args.era5_dir, "boundary_layer_height")
    sshf = load_npy(args.era5_dir, "surface_sensible_heat_flux")
    ssr = load_npy(args.era5_dir, "surface_net_solar_radiation")
    ewss = load_npy(args.era5_dir, "mean_eastward_turbulent_surface_stress")
    nsss = load_npy(args.era5_dir, "mean_northward_turbulent_surface_stress")
    cp_arr = load_npy(args.era5_dir, "convective_precipitation")
    lsp = load_npy(args.era5_dir, "large_scale_precipitation")

    times = load_npy(args.era5_dir, "times")
    lats = load_npy(args.era5_dir, "latitudes")
    lons = load_npy(args.era5_dir, "longitudes")

    n_times, nz, nlat, nlon = u.shape
    print(f"Grid: {nlon}×{nlat}×{nz}, {n_times} hourly timesteps")
    print(f"Lon: [{lons[0]:.2f}, {lons[-1]:.2f}], Lat: [{lats[-1]:.2f}, {lats[0]:.2f}]")

    # After S→N flip: ylat0 = lats[-1] (southernmost)
    ylat0 = float(lats[-1])
    xlon0 = float(lons[0])
    dx = float(lons[1] - lons[0])
    dy = abs(float(lats[0] - lats[1]))

    pressure_pa_3d = np.array(PRESSURE_LEVELS_HPA, dtype=np.float32) * 100.0

    import calendar
    from datetime import datetime as _dt
    epoch_base = int(calendar.timegm(_dt(1994, 10, 23, 0, 0, 0).timetuple()))

    # Select timesteps at given interval
    step = args.interval_hours
    selected = list(range(0, n_times, step))

    timestep_files = []
    for idx in selected:
        time_str = str(times[idx])[:19]
        hour = idx  # hours since epoch_base
        epoch = epoch_base + hour * 3600

        u_3d = prepare_3d(u, idx)
        v_3d = prepare_3d(v, idx)
        omega_3d = prepare_3d(w, idx)  # Pa/s (negative = ascending)
        t_3d = prepare_3d(t, idx)
        q_3d = prepare_3d(q, idx)

        # Pressure (constant per level, broadcast to 3D)
        p_3d = np.zeros((nlon, nlat, nz), dtype=np.float32)
        for k in range(nz):
            p_3d[:, :, k] = pressure_pa_3d[k]

        # Air density: rho = p / (Rd * Tv), Tv = T * (1 + 0.608*q)
        tv_3d = t_3d * (1.0 + 0.608 * q_3d)
        rho_3d = p_3d / (R_DRY * tv_3d)

        # Convert omega (Pa/s) to vertical velocity w (m/s): w = -omega / (rho * g)
        w_3d = (-omega_3d / (rho_3d * G)).astype(np.float32)

        # Density gradient (vertical, centered differences where possible)
        drhodz = np.zeros_like(rho_3d)
        heights = np.array(STD_HEIGHTS_M, dtype=np.float32)
        for k in range(nz):
            if k == 0:
                dz = heights[1] - heights[0]
                drhodz[:, :, k] = (rho_3d[:, :, 1] - rho_3d[:, :, 0]) / dz
            elif k == nz - 1:
                dz = heights[k] - heights[k - 1]
                drhodz[:, :, k] = (rho_3d[:, :, k] - rho_3d[:, :, k - 1]) / dz
            else:
                dz = heights[k + 1] - heights[k - 1]
                drhodz[:, :, k] = (rho_3d[:, :, k + 1] - rho_3d[:, :, k - 1]) / dz

        # Surface fields
        sp_2d = prepare_2d(sp, idx)
        u10_2d = prepare_2d(u10, idx)
        v10_2d = prepare_2d(v10, idx)
        t2m_2d = prepare_2d(t2m, idx)
        td2m_2d = prepare_2d(td2m, idx)

        # ERA5 accumulated fields → instantaneous rates
        # sshf is in J/m² accumulated over the forecast step (1h for ERA5)
        sshf_2d = prepare_2d(sshf, idx) / 3600.0  # J/m²→W/m²
        ssr_2d = prepare_2d(ssr, idx) / 3600.0

        # Surface stress magnitude from E/W and N/S components
        ewss_2d = prepare_2d(ewss, idx)
        nsss_2d = prepare_2d(nsss, idx)
        surfstr_2d = np.sqrt(ewss_2d**2 + nsss_2d**2).astype(np.float32)

        # Friction velocity: u* = sqrt(|tau| / rho_sfc)
        rho_sfc = sp_2d / (R_DRY * t2m_2d * (1.0 + 0.608 * 0.005))
        ustar_2d = np.sqrt(np.maximum(surfstr_2d, 0.0) / np.maximum(rho_sfc, 0.5)).astype(np.float32)

        # Precipitation: ERA5 accumulated m → mm/h
        lsp_2d = prepare_2d(lsp, idx) * 1000.0 / 1.0  # m→mm per 1h step
        cp_2d = prepare_2d(cp_arr, idx) * 1000.0 / 1.0

        # Convective velocity scale (placeholder)
        wstar_2d = np.zeros((nlon, nlat), dtype=np.float32)

        # Mixing height from BLH
        hmix_2d = np.maximum(prepare_2d(blh, idx), 50.0).astype(np.float32)

        # Tropopause height (fixed)
        tropo_2d = np.full((nlon, nlat), 10000.0, dtype=np.float32)

        # Inverse Obukhov length (placeholder)
        oli_2d = np.zeros((nlon, nlat), dtype=np.float32)

        fname = f"met_{idx:03d}.bin"
        fpath = os.path.join(args.output_dir, fname)

        write_timestep(fpath, [u_3d, v_3d, w_3d, t_3d, q_3d, p_3d, rho_3d, drhodz],
                       [sp_2d, u10_2d, v10_2d, t2m_2d, td2m_2d,
                        lsp_2d, cp_2d, sshf_2d, ssr_2d,
                        surfstr_2d, ustar_2d, wstar_2d,
                        hmix_2d, tropo_2d, oli_2d])

        timestep_files.append({
            "index": idx,
            "datetime": time_str,
            "epoch_seconds": epoch,
            "file": fname,
        })
        print(f"  [{idx:3d}] {time_str} → {fname} "
              f"(u=[{u_3d.min():.1f},{u_3d.max():.1f}], "
              f"hmix=[{hmix_2d.min():.0f},{hmix_2d.max():.0f}])")

    manifest = {
        "nx": int(nlon),
        "ny": int(nlat),
        "nz": int(nz),
        "dx_deg": float(dx),
        "dy_deg": float(dy),
        "xlon0_deg": float(xlon0),
        "ylat0_deg": float(ylat0),
        "heights_m": STD_HEIGHTS_M,
        "pressure_levels_hpa": PRESSURE_LEVELS_HPA,
        "timesteps": timestep_files,
        "release": {
            "name": "ETEX-1",
            "start": "19941023160000",
            "end": "19941024034000",
            "lon": -2.0,
            "lat": 48.058,
            "z_min": 5.0,
            "z_max": 15.0,
            "mass_kg": 340.0,
            "particle_count": 100000,
        },
        "simulation": {
            "start": "19941023160000",
            "end": "19941025160000",
            "dt_seconds": 900,
        },
        "output": {
            "nx": int(nlon),
            "ny": int(nlat),
            "nz": 5,
            "heights_m": [100.0, 500.0, 1000.0, 2000.0, 5000.0],
            "interval_seconds": 10800,
        },
    }

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")
    print(f"Wrote {len(timestep_files)} timestep files")


if __name__ == "__main__":
    main()
