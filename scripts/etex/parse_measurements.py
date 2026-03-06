#!/usr/bin/env python3
"""Parse ETEX-1 measurement data from DATEM format.

Reads stations.txt and meas-t1.txt from the DATEM archive and outputs
a structured JSON file with station metadata and time-series measurements,
suitable for comparison with FLEXPART output.

ETEX-1 release parameters:
  Location: Monterfil, France (48.058°N, 2.000°W)
  Start:    23 October 1994, 16:00 UTC
  Duration: ~11h40m
  Tracer:   PMCH (perfluoromonomethylcyclohexane)
  Mass:     340 kg

Usage:
    python3 scripts/etex/parse_measurements.py \\
        --data-dir fixtures/etex/data \\
        --output target/etex/measurements.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta


def parse_stations(path: str) -> dict:
    """Parse stations.txt → {stn_id: {lat, lon}}."""
    stations = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("\\") or line.startswith("stn"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    stn_id = int(parts[0])
                    lat = float(parts[1])
                    lon = float(parts[2])
                    stations[stn_id] = {"lat": lat, "lon": lon}
                except ValueError:
                    continue
    return stations


def parse_measurements(path: str) -> list:
    """Parse meas-t1.txt → list of measurement records.

    Format: year mn dy shr dur lat lon conc stn
    - shr:  start hour (HHMM)
    - dur:  sampling duration in minutes
    - conc: concentration in pg/m³ (pico-grams per cubic meter)
    """
    measurements = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("\\") or line.startswith("year"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
                shr = int(parts[3])
                dur = int(parts[4])
                lat = float(parts[5])
                lon = float(parts[6])
                conc = float(parts[7])
                stn = int(parts[8])

                start_hour = shr // 100
                start_min = shr % 100
                start_dt = datetime(year, month, day, start_hour, start_min)
                end_dt = start_dt + timedelta(minutes=dur)

                hours_after_release = (
                    start_dt - datetime(1994, 10, 23, 16, 0)
                ).total_seconds() / 3600.0

                measurements.append({
                    "station": stn,
                    "lat": lat,
                    "lon": lon,
                    "start_time": start_dt.strftime("%Y-%m-%d %H:%M"),
                    "end_time": end_dt.strftime("%Y-%m-%d %H:%M"),
                    "duration_min": dur,
                    "hours_after_release": round(hours_after_release, 2),
                    "concentration_pg_m3": conc,
                })
            except (ValueError, IndexError):
                continue
    return measurements


def compute_statistics(measurements: list) -> dict:
    """Compute summary statistics from the measurements."""
    concs = [m["concentration_pg_m3"] for m in measurements]
    nonzero = [c for c in concs if c > 0]
    by_station = defaultdict(list)
    for m in measurements:
        by_station[m["station"]].append(m["concentration_pg_m3"])

    max_conc = max(concs) if concs else 0
    max_record = next((m for m in measurements if m["concentration_pg_m3"] == max_conc), None)

    return {
        "total_records": len(measurements),
        "stations_count": len(by_station),
        "nonzero_count": len(nonzero),
        "detection_rate": round(len(nonzero) / len(concs), 3) if concs else 0,
        "max_concentration_pg_m3": max_conc,
        "max_station": max_record["station"] if max_record else None,
        "max_time": max_record["start_time"] if max_record else None,
        "mean_nonzero_pg_m3": round(sum(nonzero) / len(nonzero), 2) if nonzero else 0,
        "time_range_hours": round(
            max(m["hours_after_release"] for m in measurements)
            - min(m["hours_after_release"] for m in measurements), 1
        ) if measurements else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default="fixtures/etex/data",
                        help="Directory with stations.txt and meas-t1.txt")
    parser.add_argument("--output", default="target/etex/measurements.json",
                        help="Output JSON path")
    args = parser.parse_args()

    stations_path = os.path.join(args.data_dir, "stations.txt")
    meas_path = os.path.join(args.data_dir, "meas-t1.txt")
    for p in [stations_path, meas_path]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found.", file=sys.stderr)
            sys.exit(1)

    print("Parsing ETEX-1 measurement data...")
    stations = parse_stations(stations_path)
    print(f"  {len(stations)} stations loaded")

    measurements = parse_measurements(meas_path)
    print(f"  {len(measurements)} measurement records parsed")

    stats = compute_statistics(measurements)
    print(f"  Detection rate: {stats['detection_rate']*100:.1f}%")
    print(f"  Max: {stats['max_concentration_pg_m3']:.0f} pg/m³ "
          f"at station {stats['max_station']} ({stats['max_time']})")

    output = {
        "experiment": "ETEX-1",
        "tracer": "PMCH",
        "release": {
            "location": "Monterfil, France",
            "lat": 48.058,
            "lon": -2.000,
            "start": "1994-10-23 16:00 UTC",
            "duration_h": 11.67,
            "mass_kg": 340.0,
        },
        "source": "NOAA ARL DATEM",
        "units": "pg/m3",
        "statistics": stats,
        "stations": {str(k): v for k, v in stations.items()},
        "measurements": measurements,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
