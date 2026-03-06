#!/usr/bin/env python3
"""Download ERA5 meteorological data for the ETEX-1 experiment period.

Downloads ERA5 reanalysis on pressure levels and single levels from the
Copernicus Climate Data Store (CDS) for October 23-26, 1994 over Europe.

Prerequisites:
    1. Create a CDS account: https://cds.climate.copernicus.eu
    2. Accept the ERA5 licence terms on the dataset page
    3. Create ~/.cdsapirc:
         url: https://cds.climate.copernicus.eu/api
         key: <YOUR_PERSONAL_ACCESS_TOKEN>
    4. pip install cdsapi

Usage:
    python3 scripts/etex/download_era5.py --output-dir target/etex/meteo
"""

import argparse
import os
import sys

try:
    import cdsapi
except ImportError:
    print("ERROR: cdsapi not installed. Run: pip install cdsapi", file=sys.stderr)
    print("Then configure ~/.cdsapirc (see script header for instructions).", file=sys.stderr)
    sys.exit(1)

ETEX_AREA = [65, -15, 35, 35]  # [N, W, S, E] - Europe

PRESSURE_LEVELS = [
    "1000", "925", "850", "700", "600",
    "500", "400", "300", "250", "200",
    "150", "100", "70", "50",
]

PRESSURE_VARIABLES_3D = [
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "temperature",
    "specific_humidity",
    "geopotential",
]

SURFACE_VARIABLES = [
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_sensible_heat_flux",
    "surface_net_solar_radiation",
    "eastward_turbulent_surface_stress",
    "northward_turbulent_surface_stress",
    "large_scale_precipitation",
    "convective_precipitation",
    "boundary_layer_height",
    "forecast_surface_roughness",
    "total_column_water_vapour",
    "orography",
]

DAYS = ["23", "24", "25", "26"]
TIMES = [f"{h:02d}:00" for h in range(0, 24, 3)]


def download_pressure_levels(client, output_dir: str):
    """Download 3D fields on pressure levels."""
    out_path = os.path.join(output_dir, "era5_pressure_levels.grib")
    if os.path.exists(out_path):
        print(f"  [SKIP] {out_path} already exists")
        return out_path

    print("  Downloading ERA5 pressure-level data...")
    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": ["reanalysis"],
            "variable": PRESSURE_VARIABLES_3D,
            "pressure_level": PRESSURE_LEVELS,
            "year": ["1994"],
            "month": ["10"],
            "day": DAYS,
            "time": TIMES,
            "area": ETEX_AREA,
            "data_format": "grib",
        },
        out_path,
    )
    print(f"  -> {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return out_path


def download_single_levels(client, output_dir: str):
    """Download surface / single-level fields."""
    out_path = os.path.join(output_dir, "era5_single_levels.grib")
    if os.path.exists(out_path):
        print(f"  [SKIP] {out_path} already exists")
        return out_path

    print("  Downloading ERA5 single-level data...")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": SURFACE_VARIABLES,
            "year": ["1994"],
            "month": ["10"],
            "day": DAYS,
            "time": TIMES,
            "area": ETEX_AREA,
            "data_format": "grib",
        },
        out_path,
    )
    print(f"  -> {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="target/etex/era5_raw",
        help="Directory to store downloaded GRIB files",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"ERA5 download for ETEX-1 (Oct 23-26, 1994)")
    print(f"Output: {args.output_dir}")
    print(f"Area: N={ETEX_AREA[0]}, W={ETEX_AREA[1]}, S={ETEX_AREA[2]}, E={ETEX_AREA[3]}")
    print()

    client = cdsapi.Client()

    plev_path = download_pressure_levels(client, args.output_dir)
    sfc_path = download_single_levels(client, args.output_dir)

    print()
    print("Download complete:")
    print(f"  Pressure levels: {plev_path}")
    print(f"  Single levels:   {sfc_path}")
    print()
    print("Next: python3 scripts/etex/prepare_flexpart_input.py \\")
    print(f"        --era5-dir {args.output_dir} \\")
    print("        --output-dir target/etex/meteo")


if __name__ == "__main__":
    main()
