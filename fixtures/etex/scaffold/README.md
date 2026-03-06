# ETEX Validation Scaffold Fixture

This directory provides an ETEX-style validation scaffold for task I-04.

## What is included

- `validation_fixture.json`: harness inputs for both:
  - `pipeline` mode (synthetic end-to-end run through current timeloop + gridding),
  - `fixture-only` mode (deterministic metrics regression vectors).
- `config/COMMAND`, `config/RELEASES`, `config/OUTGRID`, `config/SPECIES/*`:
  minimal FLEXPART-style config set consumed by `SimulationConfig::load`.

## Placeholder strategy

Real ETEX meteorology and FLEXPART-Fortran oracle outputs are not bundled yet.
For MVP:

- synthetic meteorology forcing is injected via fixture parameters,
- reference fields are generated from candidate fields using affine transforms,
- metrics pipeline (RMSE, bias, MAE, correlation) is fully wired and structured.

When real data is available, switch `reference.mode` and provide
`reference.explicit_reference` (or future NetCDF/reader plug-in) without changing
the harness interfaces.
