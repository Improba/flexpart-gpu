# Benchmark: Fortran serial vs GPU production (10M)

Date: 2026-03-07  
Project: `flexpart-gpu`

## Executive summary

Replicated 10M-particle campaign (production GPU mode):

- Fortran serial mean: `230.74 s`
- GPU production mean (all reps): `23.87 s`
- GPU production mean (warm reps): `23.35 s`

Speedup:

- **9.67x** (conservative, all reps)
- **9.88x** (warm steady-state)

## Scenario

- Shared synthetic validation setup used by:
  - `fortran-validation` (GPU)
  - `../flexpart-fortran-docker/comparison/validate_run` (Fortran)
- Particles: `10,000,000`
- Production mode on GPU: `SYNC_READBACK=0`
- Fortran compile-time particle cap updated to run 10M:
  - `flexpart/src/par_mod.f90`: `maxpart=11000000`

## Raw measured times

GPU production warm-up:

- `24.15 s`

GPU production reps:

- `24.90 s`
- `23.38 s`
- `23.33 s`

Fortran serial warm-up:

- `240.59 s`

Fortran serial reps:

- `241.24 s`
- `223.38 s`
- `227.61 s`

## Aggregated results

| Metric | GPU production | Fortran serial | Speedup (Fortran / GPU) |
|---|---:|---:|---:|
| Mean (all GPU reps) | 23.87 s | 230.74 s | **9.67x** |
| Stddev (all GPU reps) | 0.73 s | 7.62 s | - |
| Mean (GPU warm reps `<24s`) | 23.35 s | 230.74 s | **9.88x** |
| Stddev (GPU warm reps) | 0.03 s | 7.62 s | - |

## Notes

- The first `FORTRAN_REP_2` attempt in the log is incomplete (stopped before a time line).
- Final statistics use complete timings from:
  - `FORTRAN_REP_1`
  - `FORTRAN_REP_2_RETRY`
  - `FORTRAN_REP_3`

## Artifact

- `target/validation/fortran-vs-gpu-prod-10m-20260306-rerun.log`
