# Documentation Index

Complete documentation for the `flexpart-gpu` project.

## Getting Started

| Document | Description |
|----------|-------------|
| [quickstart.md](quickstart.md) | Run your first ETEX simulation (GPU-only quickstart) |
| [development.md](development.md) | Build environment, Docker, testing, contributing |

## Architecture & Design

| Document | Description |
|----------|-------------|
| [architecture.md](architecture.md) | Project structure, CPU/GPU split, data flow |
| [science/](science/README.md) | Scientific foundations (Langevin, Hanna, deposition, coordinates) |
| [science/simulation-flow.md](science/simulation-flow.md) | Step-by-step execution flow of a timestep |

## Validation & Quality

| Document | Description |
|----------|-------------|
| [validation-report.md](validation-report.md) | GPU vs Fortran comparison (synthetic uniform wind) |
| [benchmarks.md](benchmarks.md) | Performance measurement methodology and recipes |
| [scientific-changelog.md](scientific-changelog.md) | Log of physics-affecting changes |

## Benchmark Reports

| Document | Description |
|----------|-------------|
| [benchmarks/benchmark-fortran-vs-gpu-current.md](benchmarks/benchmark-fortran-vs-gpu-current.md) | Canonical Fortran serial vs GPU production baseline (replicated 1M campaign) |
| [benchmarks/benchmark-fortran-vs-gpu-10m-production-20260307.md](benchmarks/benchmark-fortran-vs-gpu-10m-production-20260307.md) | Replicated Fortran serial vs GPU production campaign at 10M particles |
| [benchmarks/benchmark-1m-fortran-vs-gpu0.md](benchmarks/benchmark-1m-fortran-vs-gpu0.md) | Scientific comparison at 1M particles: partposit alignment, COM, vertical profile |
| [benchmarks/benchmark-scientific-validation.md](benchmarks/benchmark-scientific-validation.md) | Physics validation: advection, Langevin, Hanna, aligned module inventory |
