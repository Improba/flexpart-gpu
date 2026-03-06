# Simulation Flow

This document describes the execution sequence of a typical forward simulation,
from launch to output. It maps FLEXPART concepts to the actual Rust/WGSL
modules and shows what runs on CPU vs GPU.

Two execution paths exist (see [architecture.md](../architecture.md) for
rationale): the **fused Hanna+Langevin** path (production default, 4 GPU
dispatches) and the **separated** path (scientific validation, enabled with
`FLEXPART_GPU_VALIDATION=1`, 5 GPU dispatches).

## High-Level Pipeline — Production (Fused Hanna+Langevin)

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INITIALISATION                             │
│  Load config ─▶ Build GPU context ─▶ Load meteorology ─▶ Init      │
│  (COMMAND,       (wgpu adapter,       (ERA5 GRIB/binary,  release   │
│   RELEASES,       device, queue)        time brackets)    manager,  │
│   OUTGRID,                             Async prefetch     particle  │
│   SPECIES)                             (grib2_async.rs)   store     │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      TIME LOOP  (per timestep dt)                   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 1. RELEASE            CPU   release/mod.rs                   │   │
│  │    Inject new particles at scheduled source times            │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│  ┌──────────────────────────────▼───────────────────────────────┐   │
│  │ 2. WIND BRACKETS      CPU→GPU  (once per met bracket change) │   │
│  │    Upload wind_t0 and wind_t1 to GPU. Compute interpolation  │   │
│  │    factor α = (t − t0)/(t1 − t0). GPU interpolates inline.  │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│  ┌──────────────────────────────▼───────────────────────────────┐   │
│  │ 3. SURFACE FIELDS     CPU→GPU                                │   │
│  │    Upload interpolated surface fields for GPU PBL compute    │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│  ┌──────────────────────────────▼───────────────────────────────┐   │
│  │ 4. GPU PBL DIAGNOSTICS  GPU  shaders/pbl_diagnostics.wgsl   │   │
│  │    Compute u*, w*, L, h per grid cell on GPU                 │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│  ┌──────────────────────────────▼───────────────────────────────┐   │
│  │ 5. GPU PHYSICS (4 dispatches, single command encoder)        │   │
│  │    5a. Advection (3D texture dual-wind)                      │   │
│  │    5b. Fused Hanna+Langevin (turbulence + PBL reflection)    │   │
│  │    5c. Dry deposition                                        │   │
│  │    5d. Wet deposition                                        │   │
│  │                                                              │   │
│  │    queue.submit(encoder)                                     │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│  ┌──────────────────────────────▼───────────────────────────────┐   │
│  │ 6. READBACK (optional) GPU→CPU                               │   │
│  │    Download particle buffer (at output times or if sync on)  │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│  ┌──────────────────────────────▼───────────────────────────────┐   │
│  │ 7. OUTPUT GRIDDING    GPU   shaders/concentration_gridding   │   │
│  │    (at output intervals)                                     │   │
│  │    Atomic-add particle masses to concentration grid          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Advance time:  t ← t + dt                                          │
│  Loop until t ≥ t_end                                               │
└─────────────────────────────────────────────────────────────────────┘
```

## High-Level Pipeline — Validation (Separated Dispatches)

When `FLEXPART_GPU_VALIDATION=1` is set, step 5 above is replaced by five
separate dispatches (identical physics, separate shaders for easier debugging
and comparison against Fortran):

```
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │               GPU COMMAND ENCODER  (single submit)           │   │
│  │                                                              │   │
│  │  5a. ADVECTION       GPU   shaders/advection_dual_wind.wgsl │   │
│  │  5b. HANNA PARAMS    GPU   shaders/hanna_params.wgsl        │   │
│  │  5c. LANGEVIN        GPU   shaders/langevin.wgsl            │   │
│  │  5d. DRY DEPOSITION  GPU   shaders/dry_deposition.wgsl      │   │
│  │  5e. WET DEPOSITION  GPU   shaders/wet_deposition.wgsl      │   │
│  │                                                              │   │
│  │  queue.submit(encoder)                                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
```

All other steps (release, wind upload, PBL, readback, gridding) are identical.

## Step-by-Step Detail

### 1. Release (`release/mod.rs`)

The `ReleaseManager` checks whether the current simulation time falls within
any active release window defined in `RELEASES`. If so, new particles are
initialised (position, mass, species) and uploaded to the GPU particle buffer.

**Fortran equivalent:** `releaseparticles` in `timemanager.f90`.

### 2. Wind Bracket Upload (`simulation/timeloop.rs`)

Wind fields (`WindField3D`) are managed as time brackets (t0, t1). When the
simulation time crosses a bracket boundary, new wind data is read (optionally
via async prefetch from `io/grib2_async.rs`) and uploaded to the GPU as two
persistent buffers (`wind_t0`, `wind_t1`). An interpolation factor α is
computed each step and passed as a uniform:

```
α = (t − t0) / (t1 − t0)
```

The GPU performs the temporal interpolation inline during advection:
`wind(t) = (1 − α) · wind_t0 + α · wind_t1`. This avoids re-uploading an
interpolated wind field every timestep.

**Fortran equivalent:** `getfields` → `readwind` + temporal interpolation in
`timemanager.f90`.

### 3. PBL Diagnostics — GPU (`shaders/pbl_diagnostics.wgsl`)

Surface fields (friction velocity, sensible heat flux, surface stress,
2 m temperature) are uploaded to the GPU. A compute shader computes, per
grid cell:

- friction velocity u\*
- convective velocity scale w\*
- Obukhov length L
- mixing height h

The result is written to PBL buffers (separate `ustar`, `wstar`, `hmix`,
`oli` arrays) consumed by the fused Hanna+Langevin kernel (production) or
the separated Hanna kernel (validation).

**Dispatch:** `gpu/pbl.rs`

**Fortran equivalent:** `calcpar` → `obukhov`, `richardson`.

**Fallback:** The CPU-side reference (`io/pbl_params.rs`) still exists and is
used in CPU-only tests.

### 4. Advection — GPU

Each particle is advected by the mean wind using the Petterssen
predictor–corrector scheme:

1. Sample wind at current position → (u₀, v₀, w₀)
2. Predict: x\_pred = x + dt · v₀
3. Sample wind at predicted position → (u₁, v₁, w₁)
4. Correct: x\_final = x + dt · 0.5·(v₀ + v₁)

Wind is trilinearly interpolated in (x, y, level) from the dual-bracket
buffers (`wind_t0`, `wind_t1`) with temporal factor α. On GPUs that support
3D textures, hardware-accelerated texture sampling replaces manual trilinear
interpolation.

Velocities [m/s] are converted to grid displacement via `VelocityToGridScale`.

**Shaders:**
- `shaders/advection_texture_dual_wind.wgsl` (production, 3D texture path)
- `shaders/advection_dual_wind.wgsl` (buffer fallback / validation path)

**Dispatch:** `gpu/advection.rs`

**Fortran equivalent:** `advance.f90` (lines 817–923).

### 5. Hanna Turbulence Parameters + Langevin — GPU

In **production mode**, these two stages are fused into a single dispatch
(`langevin_fused.wgsl`). For each particle, the kernel:

1. Looks up PBL fields (u\*, w\*, L, h) from the PBL grid buffers
2. Computes σ\_u, σ\_v, σ\_w, T\_Lu, T\_Lv, T\_Lw, dσ\_w/dz inline
   (Hanna 1982, identical logic to `hanna_params.wgsl`)
3. Updates turb\_u, turb\_v (horizontal Langevin, full dt)
4. Sub-steps turb\_w with PBL reflection (Thomson 1987)

This eliminates the intermediate HannaParams buffer (~64 MB for 1M particles)
and one dispatch barrier.

In **validation mode**, these are two separate dispatches:
- `hanna_params.wgsl` writes HannaParams to a per-particle buffer
- `langevin.wgsl` reads HannaParams and applies the Langevin update

**Shaders:**
- `shaders/langevin_fused.wgsl` (production)
- `shaders/hanna_params.wgsl` + `shaders/langevin.wgsl` (validation)

**Dispatch:**
- `gpu/langevin_fused.rs` (production)
- `gpu/hanna.rs` + `gpu/langevin.rs` (validation)

**Fortran equivalent:** `hanna.f90` + `advance.f90` (Langevin section) +
`hanna_short.f90`.

### 6. Dry Deposition — GPU

For particles with `pos_z < 2 · h_ref` (h\_ref = 15 m):

```
survival = exp(−v_d · |dt| / (2 · h_ref))
mass ← mass · survival
```

The deposition velocity v\_d is provided per particle by the forcing vector.

**Shaders:** `shaders/dry_deposition.wgsl`

**Dispatch:** `gpu/deposition.rs`

**Fortran equivalent:** `drydepokernel.f90`, `getvdep.f90`.

### 7. Wet Deposition — GPU

```
P_wet = gr_fraction · (1 − exp(−λ · |dt|))
mass ← mass · (1 − P_wet)
```

The scavenging coefficient λ and precipitating fraction are forcing inputs.

**Shaders:** `shaders/wet_deposition.wgsl`

**Dispatch:** `gpu/wet_deposition.rs`

**Fortran equivalent:** `wetdepo.f90`, `wetdepokernel.f90`.

### 8. Host Readback (optional)

If `sync_particle_store_each_step` is enabled, the full particle buffer is
downloaded from GPU to CPU after each step. In production mode readback is
deferred (fire-and-forget) for maximum throughput.

### 9. Concentration Gridding — GPU (`shaders/concentration_gridding.wgsl`)

At output intervals, particle masses are binned into the 3D output grid
(OUTGRID) using atomic integer additions. The grid is then downloaded to CPU
and written to the output file.

**Dispatch:** `gpu/gridding.rs`

**Fortran equivalent:** `conccalc.f90` → `concoutput.f90`.

## GPU Command Encoding

### Production (fused Hanna+Langevin)

Four compute dispatches plus PBL diagnostics, encoded into a single command
encoder with sequential execution:

```
encoder = device.create_command_encoder()
encoder.dispatch(pbl_diagnostics)        // per grid cell
encoder.dispatch(advection)              // Petterssen predictor–corrector
encoder.dispatch(langevin_fused)         // inline Hanna + Langevin + PBL reflection
encoder.dispatch(dry_deposition)         // mass survival
encoder.dispatch(wet_deposition)         // mass survival
queue.submit(encoder)
```

### Validation (separated dispatches)

Five separate compute passes, encoded into a single command encoder:

```
encoder = device.create_command_encoder()
encoder.dispatch(pbl_diagnostics)
encoder.dispatch(advection)
encoder.dispatch(hanna)
encoder.dispatch(langevin)
encoder.dispatch(dry_deposition)
encoder.dispatch(wet_deposition)
queue.submit(encoder)
```

## Comparison with Fortran FLEXPART

| Aspect | Fortran (`timemanager.f90`) | GPU (`simulation/timeloop.rs`) |
|--------|---------------------------|-------------------------------|
| Met I/O | `getfields` reads GRIB each step | Dual-bracket upload (once per met change) + async prefetch |
| Wind interpolation | CPU, per-particle, per-step | GPU-side: `(1−α)·t0 + α·t1` inline during advection |
| PBL | `calcpar` (u\*, L, h) on CPU | `pbl_diagnostics.wgsl` on GPU (per grid cell) |
| Convection | Emanuel scheme (`convmix`) | Not yet implemented |
| Advection | `advance.f90` (per-particle loop) | GPU shader, all particles in parallel |
| Turbulence | `hanna` + Langevin in `advance.f90` | Fused Hanna+Langevin (prod) or two separate passes (validation) |
| Deposition | `wetdepo`, `drydepokernel` | Separate dry + wet dispatches (both paths) |
| Output | `conccalc` + `concoutput` | GPU gridding + CPU download |
| Nested grids | Supported | Not yet implemented |

## Source Files

| Module | Path | CPU/GPU |
|--------|------|---------|
| Time loop orchestration | `src/simulation/timeloop.rs` | CPU (async) |
| Release manager | `src/release/mod.rs` | CPU |
| Met bracket management | `src/io/temporal.rs` | CPU |
| Async GRIB prefetch | `src/io/grib2_async.rs` | CPU (background thread) |
| PBL parameters (CPU ref.) | `src/io/pbl_params.rs` | CPU |
| **Fused Hanna+Langevin (production)** | `src/shaders/langevin_fused.wgsl` | **GPU** |
| **Fused H+L dispatch** | `src/gpu/langevin_fused.rs` | **CPU→GPU** |
| GPU PBL diagnostics | `src/shaders/pbl_diagnostics.wgsl` | GPU |
| GPU PBL dispatch | `src/gpu/pbl.rs` | CPU→GPU |
| Advection kernel | `src/shaders/advection_texture_dual_wind.wgsl` | GPU |
| Advection dispatch | `src/gpu/advection.rs` | CPU→GPU |
| Hanna kernel (validation) | `src/shaders/hanna_params.wgsl` | GPU |
| Hanna dispatch (validation) | `src/gpu/hanna.rs` | CPU→GPU |
| Langevin kernel (validation) | `src/shaders/langevin.wgsl` | GPU |
| Langevin dispatch (validation) | `src/gpu/langevin.rs` | CPU→GPU |
| Dry deposition | `src/shaders/dry_deposition.wgsl` | GPU |
| Dry deposition dispatch | `src/gpu/deposition.rs` | CPU→GPU |
| Wet deposition | `src/shaders/wet_deposition.wgsl` | GPU |
| Wet deposition dispatch | `src/gpu/wet_deposition.rs` | CPU→GPU |
| Active particle compaction | `src/shaders/compaction.wgsl` | GPU |
| Compaction dispatch | `src/gpu/compaction.rs` | CPU→GPU |
| Concentration gridding | `src/shaders/concentration_gridding.wgsl` | GPU |
| Gridding dispatch | `src/gpu/gridding.rs` | CPU→GPU |
| Particle buffer | `src/gpu/buffers.rs` | GPU memory |
| Coordinate transforms | `src/coords/mod.rs` | CPU + GPU |
| ETEX driver (example) | `src/bin/etex-run.rs` | CPU (main) |
