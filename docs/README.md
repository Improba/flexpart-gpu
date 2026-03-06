# Documentation Index

Complete documentation for the `flexpart-gpu` project.

## Getting Started

| Document | Description |
|----------|-------------|
| [quickstart.md](quickstart.md) | Run your first ETEX simulation in 10 minutes |
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

## Planning (internal, work in progress)

| Document | Description |
|----------|-------------|
| [temp/implementation-plan.md](temp/implementation-plan.md) | Full implementation plan, task matrix, phasing |
| [temp/scientific-hardening-plan.md](temp/scientific-hardening-plan.md) | Scientific validation roadmap (Phases A–E) |
| [temp/gpu-optimisation-plan.md](temp/gpu-optimisation-plan.md) | GPU performance optimisation plan |
| [temp/phase-d-etex-report.md](temp/phase-d-etex-report.md) | ETEX-1 real-data validation report |
