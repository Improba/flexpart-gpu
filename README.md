# FLEXPART-GPU

Rust/WebGPU port of FLEXPART for scientific exploration, numerical validation, and CPU (Fortran/Rust) vs GPU performance evaluation.

## Context

This project re-implements FLEXPART components in Rust and WGSL, with a focus on:

- experiment reproducibility,
- output comparison between the legacy implementation and the GPU port,
- performance measurement at significant problem sizes.

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

See `NOTICE.md` for attribution and compliance details.

## Benchmarks

The benchmark methodology and recommended commands are documented in `docs/benchmarks.md`.

## Documentation

- Scientific foundations of the model: [`docs/science/`](docs/science/README.md)
- ETEX simulation quickstart: [`docs/quickstart.md`](docs/quickstart.md)
- Synthetic validation report: [`docs/validation-report.md`](docs/validation-report.md)
- Detailed ETEX report (work in progress): [`docs/temp/phase-d-etex-report.md`](docs/temp/phase-d-etex-report.md)

## Contact

Sylvain Meylan (Improba)  
Email: `A_RENSEIGNER`
