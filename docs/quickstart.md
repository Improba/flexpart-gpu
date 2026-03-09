# Quickstart ETEX (first simulation)

This guide shows the simplest way to run an ETEX-style simulation
with `flexpart-gpu`, without modifying the engine source code.

The default quickstart is **GPU-only** and does **not** require a
sibling `../flexpart` checkout.

## 1) Prerequisites

- Rust toolchain (`cargo`)
- Python 3 (+ `numpy`, `eccodes`, `cdsapi`)
- Copernicus CDS account (to download ERA5) with `~/.cdsapirc`

Optional (only for Fortran comparison):

- Docker + Docker Compose
- sibling Fortran checkout at `../flexpart`

Optional (for GPU execution inside Docker):

- `docker/docker-compose.yml` (default GPU container)
- `docker/docker-compose.nvidia.yml` (NVIDIA overlay)

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

## 3) Run the GPU-only pipeline (recommended for a first run)

```bash
scripts/run-etex.sh all
```

The script chains:

1. parsing ETEX measurements,
2. downloading ERA5,
3. preparing FLEXPART meteorological files,
4. running `flexpart-gpu`,
5. comparing against observations,
6. generating the report.

## 4) Run and debug step by step

If `all` fails, run individual steps:

```bash
scripts/run-etex.sh parse
scripts/run-etex.sh download
scripts/run-etex.sh prepare
scripts/run-etex.sh gpu
scripts/run-etex.sh compare
scripts/run-etex.sh report
```

Optional Fortran step:

```bash
scripts/run-etex.sh fortran
```

Or run everything including Fortran in one go:

```bash
scripts/run-etex.sh all-with-fortran
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
- `gpu.log`

When Fortran comparison is enabled, it also writes:

- `fortran.log`

## 7) Practical notes

- You do not need to write a new C++/Rust program for each case:
  standard usage relies on config files + a launcher script.
- If you change the physics (kernels, timeloop), then yes, recompilation
  is required.
- If CDS credentials are missing, the pipeline may stop at the ERA5
  download step.

## 8) Optional Docker usage (GPU path)

From the `flexpart-gpu` root:

```bash
# Default containerized GPU run
docker compose -f docker/docker-compose.yml run --rm flexpart-gpu bash

# NVIDIA containerized GPU run
docker compose -f docker/docker-compose.yml -f docker/docker-compose.nvidia.yml \
  run --rm flexpart-gpu bash
```

Helper wrappers (recommended):

```bash
scripts/gpu-preflight.sh compose
scripts/gpu-preflight.sh nvidia
```

## 9) Further reading

- Benchmarks: `docs/benchmarks.md`
- Validation report: `docs/validation-report.md`
- Scientific validation: `docs/benchmarks/benchmark-scientific-validation.md`
