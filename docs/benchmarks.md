# FLEXPART-GPU Benchmark Suite

This project ships a Criterion benchmark suite in `benches/advection.rs` that covers:

- per-stage GPU kernels (advection, Hanna, Langevin, dry/wet deposition, concentration gridding),
- an end-to-end forward time-loop benchmark (`ForwardTimeLoopDriver::run_timestep`),
- CPU reference baselines for the same stages and an end-to-end CPU step benchmark.

## Scenarios

Three named scenarios are always present: `1M`, `10M`, `100M`.

Each scenario runs at `min(target_particles, FLEXPART_BENCH_MAX_PARTICLES)`. When clamped
below target, the label becomes `10M (scaled_250.00K)` etc. **A scaled run does NOT
represent the named scenario** — at low particle counts the GPU dispatch overhead dominates
and the results are meaningless for speedup analysis.

## Determinism

Benchmarks are deterministic by construction:

- fixed synthetic meteorology and surface fields,
- deterministic particle initialization (index-driven hash, no system RNG),
- fixed Philox key/counter for Langevin updates,
- fixed deterministic deposition/scavenging forcing vectors.

Given the same binary, backend, and environment variables, inputs are identical across runs.

---

## Methodology: how to produce usable GPU vs CPU numbers

### Common pitfalls

| Pitfall | Why it invalidates results |
|---------|--------------------------|
| **Running at 1K particles** | GPU dispatch latency (~50–200 µs) dominates; you measure overhead, not compute throughput. |
| **Comparing `base` vs `new` in Criterion** | They represent consecutive runs of the *same* benchmark, not GPU vs CPU. |
| **Mixing readback modes** | GPU with `SYNC_HOST=1` does a full buffer download each step; comparing that to GPU-no-readback conflates transfer cost with compute cost. |
| **Single sample** | Wall-clock variance (thermal throttling, OS scheduling) requires ≥10 samples. |
| **Not warming up the GPU** | First dispatch compiles shaders / allocates; warm-up of ≥2 s is needed. |

### What to measure

Three independent quantities give the full picture:

1. **GPU throughput (fire-and-forget)** — readbacks off, pure pipeline speed.
2. **GPU strict (readbacks on)** — realistic worst-case where host needs data each step.
3. **CPU reference (Rust)** — single-threaded CPU baseline for the same pipeline.

From these:

- **Speedup (fire-and-forget)** = CPU / GPU-no-readback
- **Speedup (strict)** = CPU / GPU-strict
- **Readback overhead** = GPU-strict − GPU-no-readback

### Minimum particle counts

GPU compute becomes dominant over dispatch overhead roughly above 100K particles.
Meaningful speedup numbers require **≥ 1M particles**. The canonical comparison
points are **1M** and **10M**.

### Recommended Criterion parameters

For stable estimates at large particle counts:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `SAMPLE_SIZE` | 20 | Enough for CI/stddev estimates without excessive runtime |
| `WARMUP_SECS` | 3 | Shader compilation + thermal stabilization |
| `MEASUREMENT_SECS` | 10 | Enough iterations even at 10M particles |

---

## Concrete benchmark recipes

### Prerequisite: build in release mode

All commands below invoke `cargo bench`, which implies `--release`. Ensure
the build is up to date before comparing runs — a stale binary invalidates
everything.

### Run A — GPU fire-and-forget (no readback)

Measures pure GPU pipeline throughput at 1M and 10M particles.

```bash
FLEXPART_BENCH_MAX_PARTICLES=10000000 \
FLEXPART_BENCH_SAMPLE_SIZE=20 \
FLEXPART_BENCH_WARMUP_SECS=3 \
FLEXPART_BENCH_MEASUREMENT_SECS=10 \
FLEXPART_BENCH_TIMELOOP_SYNC_HOST=0 \
FLEXPART_BENCH_TIMELOOP_COLLECT_PROBABILITIES=0 \
  cargo bench --bench advection -- 'pipeline_end_to_end_timeloop/forward_step'
```

### Run B — GPU strict (full readbacks each step)

Measures GPU pipeline with host synchronization and probability collection enabled.

```bash
FLEXPART_BENCH_MAX_PARTICLES=10000000 \
FLEXPART_BENCH_SAMPLE_SIZE=20 \
FLEXPART_BENCH_WARMUP_SECS=3 \
FLEXPART_BENCH_MEASUREMENT_SECS=10 \
FLEXPART_BENCH_TIMELOOP_SYNC_HOST=1 \
FLEXPART_BENCH_TIMELOOP_COLLECT_PROBABILITIES=1 \
  cargo bench --bench advection -- 'pipeline_end_to_end_timeloop/forward_step'
```

> **Important**: Run B writes to the same Criterion directory as Run A.  
> Criterion will treat Run A's results as `base` and Run B as `new`, which is
> useful for seeing the delta but **not** a GPU-vs-CPU comparison. To preserve
> both, copy `target/criterion/pipeline_end_to_end_timeloop` after Run A:
>
> ```bash
> cp -r target/criterion/pipeline_end_to_end_timeloop{,_no_readback}
> ```

### Run C — CPU reference baseline

Measures the single-threaded Rust CPU pipeline at the same particle counts.

```bash
FLEXPART_BENCH_MAX_PARTICLES=10000000 \
FLEXPART_BENCH_SAMPLE_SIZE=20 \
FLEXPART_BENCH_WARMUP_SECS=3 \
FLEXPART_BENCH_MEASUREMENT_SECS=10 \
  cargo bench --bench advection -- 'pipeline_end_to_end_timeloop_cpu'
```

### Reading results

Criterion writes JSON estimates to:

```
target/criterion/<group>/<bench_id>/<scenario>/new/estimates.json
```

The `mean.point_estimate` field is in **nanoseconds**. To compare:

```
speedup = cpu_mean_ns / gpu_mean_ns
readback_cost_ns = gpu_strict_mean_ns - gpu_noreadback_mean_ns
readback_overhead_pct = readback_cost_ns / gpu_noreadback_mean_ns * 100
```

### Per-stage breakdown (optional)

To identify which kernel is the bottleneck, run the per-stage benchmarks at
the same particle count:

```bash
FLEXPART_BENCH_MAX_PARTICLES=10000000 \
FLEXPART_BENCH_SAMPLE_SIZE=20 \
FLEXPART_BENCH_WARMUP_SECS=3 \
FLEXPART_BENCH_MEASUREMENT_SECS=10 \
  cargo bench --bench advection -- 'pipeline_stage_'
```

This benches: advection, hanna, langevin, dry_deposition, wet_deposition,
concentration_gridding — each with GPU and CPU variants.

---

## Environment variable reference

| Variable | Default | Description |
|----------|---------|-------------|
| `FLEXPART_BENCH_MAX_PARTICLES` | `1000000` | Caps effective particle count per scenario. |
| `FLEXPART_BENCH_SAMPLE_SIZE` | `10` | Criterion sample size (minimum 10). |
| `FLEXPART_BENCH_WARMUP_SECS` | `1` | Criterion warm-up duration. |
| `FLEXPART_BENCH_MEASUREMENT_SECS` | `3` | Criterion measurement duration. |
| `FLEXPART_BENCH_TIMELOOP_SYNC_HOST` | `0` | `1` = sync host particle store each step (GPU readback). |
| `FLEXPART_BENCH_TIMELOOP_COLLECT_PROBABILITIES` | `0` | `1` = download dry/wet probability vectors each step. |
| `FLEXPART_BENCH_AUTOTUNE` | `0` | `1` = run workgroup auto-tuning before benchmarks. |
| `FLEXPART_BENCH_AUTOTUNE_PARTICLES` | min(MAX, 250k) | Particle count for auto-tuning harness. |
| `FLEXPART_BENCH_AUTOTUNE_CANDIDATES` | — | Comma-separated candidate workgroup sizes. |
| `FLEXPART_WG_AUTOTUNE_DETERMINISTIC` | `1` | Deterministic tie-breaking in auto-tune. |
| `FLEXPART_WG_AUTOTUNE_CACHE` | `~/.cache/flexpart-gpu/…` | Override auto-tune cache path. |
| `FLEXPART_GPU_VALIDATION` | `0` | `1` = use separated Hanna → Langevin dispatches instead of fused production path. |

If no GPU adapter is available, GPU groups emit a placeholder entry; CPU baselines still run.

---

## Fortran comparison benchmarks

In addition to the Criterion micro-benchmarks above, the project includes a
**Fortran FLEXPART vs GPU end-to-end comparison** using the `fortran-validation`
binary and the `compare-fortran.sh` orchestration script. Both engines run on an
identical synthetic scenario; comparison scripts produce statistical metrics and
particle position comparisons.

### Prerequisites

1. Docker with `docker compose` (for the Fortran side).
2. The Fortran Docker image must be built:
   ```bash
   cd ../flexpart-fortran-docker && docker compose build
   ```
3. The Fortran FLEXPART binary must be compiled (done once during setup):
   ```bash
   scripts/compare-fortran.sh compose setup
   ```

### Quick performance benchmark (GPU only)

Measures GPU end-to-end time at 1M particles, no Fortran needed:

```bash
# Production mode (readback off) — measures pure GPU pipeline
OUTPUT_PATH=/dev/null PARTICLES=1000000 SYNC_READBACK=0 \
  /usr/bin/time -f 'REAL_SECONDS=%e' \
  cargo run --release --bin fortran-validation

# Validation mode (readback on) — includes GPU→CPU transfer at each step
OUTPUT_PATH=/dev/null PARTICLES=1000000 SYNC_READBACK=1 \
  /usr/bin/time -f 'REAL_SECONDS=%e' \
  cargo run --release --bin fortran-validation
```

### Quick kernel-level benchmark

Measures per-step GPU compute time with deposition, spatially-varying met:

```bash
PARTICLES=1000000 WARMUP_STEPS=5 MEASURE_STEPS=30 \
  cargo run --release --bin bench-timeloop
```

### Full Fortran vs GPU comparison (performance + science)

Run both engines and produce a scientific comparison:

```bash
# Step 1: Run GPU with readback on (exports particle positions)
OUTPUT_PATH=target/validation/gpu_1m_readback_on.json \
PARTICLES=1000000 SYNC_READBACK=1 \
  /usr/bin/time -f 'REAL_SECONDS=%e' \
  cargo run --release --bin fortran-validation

# Step 2: Set Fortran particle count to match
# Edit ../flexpart-fortran-docker/comparison/validate_run/options/RELEASES
# Set PARTS = 1000000

# Step 3: Run Fortran
cd ../flexpart-fortran-docker
/usr/bin/time -f 'REAL_SECONDS=%e' \
  docker compose run --rm flexpart-fortran bash -c \
  'cd /workspace/comparison/validate_run && /workspace/flexpart/src/FLEXPART'
cd ../flexpart-gpu

# Step 4: Run scientific comparison (partposit-based)
.venv/bin/python scripts/compare_concentrations.py \
  --fortran-output ../flexpart-fortran-docker/comparison/validate_run/output \
  --gpu-output target/validation/gpu_1m_readback_on.json \
  --output-json target/validation/comparison_1m_report.json \
  --verbose

# Step 5: Run clean aligned comparison (same gridding operator)
.venv/bin/python scripts/compare_clean_partposit.py \
  --repo-root . \
  --fortran-output ../flexpart-fortran-docker/comparison/validate_run/output \
  --gpu-output target/validation/gpu_1m_readback_on.json \
  --output-json target/validation/comparison_1m_clean_report.json
```

### Automated Fortran validation (all-in-one)

The `compare-fortran.sh` script automates the full workflow:

```bash
# Full setup + Fortran run + GPU tests
scripts/compare-fortran.sh compose all

# Scientific validation only (builds, runs, compares)
scripts/compare-fortran.sh compose validate
```

### Environment variables for `fortran-validation`

| Variable | Default | Description |
|----------|---------|-------------|
| `PARTICLES` | `10000` | Number of particles to release |
| `OUTPUT_PATH` | `target/validation/gpu_concentration.json` | Output JSON path |
| `SYNC_READBACK` | `1` | `0` = fire-and-forget (production), `1` = full readback each step (validation) |

### Environment variables for `bench-timeloop`

| Variable | Default | Description |
|----------|---------|-------------|
| `PARTICLES` | `1000000` | Number of particles |
| `WARMUP_STEPS` | `3` | Warm-up steps (not measured) |
| `MEASURE_STEPS` | `20` | Steps to measure |
| `TIMESTEP_SECONDS` | `1` | Simulation timestep |

### Reports

Detailed benchmark results are in `docs/temp/`:

- [`benchmark-performance-scaling.md`](temp/benchmark-performance-scaling.md) — particle count scaling (10K–1M), GPU vs Fortran serial, readback on/off, MPI context.
- [`benchmark-1m-fortran-vs-gpu0.md`](temp/benchmark-1m-fortran-vs-gpu0.md) — scientific comparison at 1M particles (clean aligned gridding, COM, correlation, known differences).
- [`benchmark-scientific-validation.md`](temp/benchmark-scientific-validation.md) — physics-focused validation report: advection, vertical/horizontal diffusion, vertical profile analysis, aligned physics inventory.
