# FLEXPART-GPU

Rust/WebGPU port of FLEXPART for scientific exploration, numerical validation, and CPU (Fortran/Rust) vs GPU performance evaluation.

## Project status and relationship to upstream

`flexpart-gpu` is an independent and unofficial project inspired by, and partially derived from, FLEXPART.
It is not affiliated with or endorsed by the official FLEXPART development team.

## Context

This project re-implements FLEXPART components in Rust and WGSL, with a focus on:

- experiment reproducibility,
- output comparison between the legacy implementation and the GPU port,
- performance measurement at significant problem sizes.

## Value beyond raw performance

Beyond speedups, this codebase brings practical value for scientific and
operational workflows:

- **Shorter scenario turnaround**: faster runs make it easier to iterate on
  hypotheses, boundary conditions, and sensitivity studies.
- **More accessible operations**: strong throughput on a single GPU workstation
  can reduce dependence on MPI clusters for many day-to-day campaigns.
- **Reproducibility and auditability**: benchmark protocols, validation reports,
  and comparison scripts are versioned and repeatable.
- **Safer engineering surface**: Rust typing, explicit error handling, and test
  coverage improve maintainability of a complex numerical pipeline.
- **Structured migration path**: side-by-side comparison with legacy outputs
  helps adopt GPU acceleration incrementally instead of requiring a hard switch.

## License and implications

Upstream FLEXPART is distributed under `GPL-3.0-or-later`.
This project is therefore also published under `GPL-3.0-or-later`.

Practical implications for distribution:

- this port must remain under a GPL-compatible license;
- if binaries are distributed, the corresponding source code must be provided;
- upstream copyright/license notices and attributions must be preserved;
- modifications must be clearly identified;
- AI-assisted porting does not waive license obligations.

## Attribution

This project explicitly acknowledges the scientific and software origins of FLEXPART.
Primary credit for the model foundations, methods, and scientific validation goes to the FLEXPART team.

Upstream references:

- FLEXPART home page: <https://www.flexpart.eu/>
- FLEXPART v11 repository: <https://gitlab.phaidra.org/flexpart/flexpart>

Key publication:

- Bakels et al. (2024), Geosci. Model Dev., 17, 7595-7624, <https://doi.org/10.5194/gmd-17-7595-2024>

See `NOTICE.md` for attribution and compliance details.

## Benchmarks

The benchmark methodology and recommended commands are documented in `docs/benchmarks.md`.

## Documentation

Full documentation index: [`docs/`](docs/README.md)

- Quickstart (first simulation): [`docs/quickstart.md`](docs/quickstart.md)
- Architecture & source tree: [`docs/architecture.md`](docs/architecture.md)
- Development guide (build, Docker, test): [`docs/development.md`](docs/development.md)
- Scientific foundations: [`docs/science/`](docs/science/README.md)
- Validation report: [`docs/validation-report.md`](docs/validation-report.md)

## Contact

Sylvain Meylan (Improba)  
Email: sylvain.meylan@improba.fr
