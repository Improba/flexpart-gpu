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

If no GPU adapter is available, GPU groups emit a placeholder entry; CPU baselines still run.
