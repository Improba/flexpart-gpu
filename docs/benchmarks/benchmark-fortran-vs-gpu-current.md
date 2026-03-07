# Benchmark baseline: Fortran serial vs GPU production (current)

Date: 2026-03-06  
Project: `flexpart-gpu`

## Executive summary

Current replicated baseline at 1M particles:

- Fortran serial mean: `24.55 s`
- GPU production mean (all reps): `2.95 s`
- GPU production mean (warm reps): `2.20 s`

Speedup:

- **8.32x** (conservative, all reps)
- **11.16x** (warm steady-state)

These values replace previous benchmark baselines.

## Additional replicated point (10M)

Latest 10M production-vs-Fortran campaign:

- Fortran serial mean: `230.74 s`
- GPU production mean (all reps): `23.87 s`
- Conservative speedup (all reps): **9.67x**
- Warm steady-state speedup: **9.88x**

Detailed report:

- [`benchmark-fortran-vs-gpu-10m-production-20260307.md`](benchmark-fortran-vs-gpu-10m-production-20260307.md)

## Scenario

- Shared synthetic validation setup used by:
  - `fortran-validation` (GPU)
  - `../flexpart-fortran-docker/comparison/validate_run` (Fortran)
- Particles: `1,000,000`
- Production mode on GPU: `SYNC_READBACK=0`

## Raw measured times

GPU production reps:

- `4.45 s`
- `2.22 s`
- `2.18 s`

Fortran serial reps:

- `23.62 s`
- `25.35 s`
- `24.68 s`

## Aggregated results

| Metric | GPU production | Fortran serial | Speedup (Fortran / GPU) |
|---|---:|---:|---:|
| Mean (all GPU reps) | 2.95 s | 24.55 s | **8.32x** |
| Stddev (all GPU reps) | 1.06 s | 0.71 s | - |
| Mean (GPU warm reps `<3s`) | 2.20 s | 24.55 s | **11.16x** |
| Stddev (GPU warm reps) | 0.02 s | 0.71 s | - |

## Reproducibility

```bash
# GPU production (from flexpart-gpu/)
OUTPUT_PATH=/dev/null PARTICLES=1000000 SYNC_READBACK=0 \
  /usr/bin/time -f 'GPU_PROD REAL_SECONDS=%e' \
  cargo run --release --bin fortran-validation

# Fortran serial (from flexpart-fortran-docker/)
/usr/bin/time -f 'FORTRAN REAL_SECONDS=%e' \
  docker compose run --rm flexpart-fortran bash -lc \
  'cd /workspace/comparison/validate_run && /workspace/flexpart/src/FLEXPART'
```

## Artifact

- `target/validation/fortran-vs-gpu-prod-1m-20260306.log`
