# Quickstart ETEX (first simulation)

This guide shows the simplest way to run an ETEX-style simulation
with `flexpart-gpu`, without modifying the engine source code.

## 1) Prerequisites

- Rust toolchain (`cargo`)
- Python 3 (+ `numpy`, `eccodes`, `cdsapi`)
- Docker + Docker Compose (for Fortran comparison)
- Copernicus CDS account (to download ERA5) with `~/.cdsapirc`

Example `~/.cdsapirc`:

```yaml
url: https://cds.climate.copernicus.eu/api
key: <YOUR_PERSONAL_ACCESS_TOKEN>
```

## 2) Check pipeline status

From the `flexpart-gpu` root:

```bash
scripts/run-etex.sh status
```

This command reports what is missing (ETEX data, ERA5, outputs already
produced, etc.).

## 3) Run the full pipeline (recommended for a first run)

```bash
scripts/run-etex.sh all
```

The script chains:

1. parsing ETEX measurements,
2. downloading ERA5,
3. preparing FLEXPART meteorological files,
4. running the Fortran reference (Docker),
5. running `flexpart-gpu`,
6. comparing against observations,
7. generating the report.

## 4) Run and debug step by step

If `all` fails, run individual steps:

```bash
scripts/run-etex.sh parse
scripts/run-etex.sh download
scripts/run-etex.sh prepare
scripts/run-etex.sh fortran
scripts/run-etex.sh gpu
scripts/run-etex.sh compare
scripts/run-etex.sh report
```

## 5) Scenario configuration files

The reference ETEX scenario is located in:

- `fixtures/etex/real/config/COMMAND`
- `fixtures/etex/real/config/RELEASES`
- `fixtures/etex/real/config/OUTGRID`
- `fixtures/etex/real/config/SPECIES/*`

To create a new simulation, duplicate this folder, edit these files,
and adjust the script (or variables) to point to your new `config`.

## 6) Produced artifacts

The pipeline writes mainly to `target/etex/`, including:

- `gpu_output.json`
- `comparison_report.json`
- `fortran.log`
- `gpu.log`

## 7) Practical notes

- You do not need to write a new C++/Rust program for each case:
  standard usage relies on config files + a launcher script.
- If you change the physics (kernels, timeloop), then yes, recompilation
  is required.
- If CDS credentials are missing, the pipeline may stop at the ERA5
  download step.

## 8) Further reading

- Benchmarks: `docs/benchmarks.md`
- Validation report: `docs/validation-report.md`
- Scientific validation: `docs/benchmarks/benchmark-scientific-validation.md`
