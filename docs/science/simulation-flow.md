# Simulation Flow

This document describes the execution sequence of a typical forward simulation,
from launch to output. It maps FLEXPART concepts to the actual Rust/WGSL
modules and shows what runs on CPU vs GPU.

## High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INITIALISATION                             │
│  Load config ─▶ Build GPU context ─▶ Load meteorology ─▶ Init      │
│  (COMMAND,       (wgpu adapter,       (ERA5 GRIB/binary,  release   │
│   RELEASES,       device, queue)        time brackets)    manager,  │
│   OUTGRID,                                                particle  │
│   SPECIES)                                                store     │
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
│  │ 2. MET INTERPOLATION  CPU   io/temporal.rs                   │   │
│  │    Temporal interpolation of wind and surface fields          │   │
│  │    between two met brackets (t0, t1)                         │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│  ┌──────────────────────────────▼───────────────────────────────┐   │
│  │ 3. PBL DIAGNOSTICS    CPU   io/pbl_params.rs                 │   │
│  │    Derive u*, w*, L, h from interpolated surface fields      │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│  ┌──────────────────────────────▼───────────────────────────────┐   │
│  │               GPU COMMAND ENCODER  (single submit)           │   │
│  │                                                              │   │
│  │  ┌────────────────────────────────────────────────────────┐  │   │
│  │  │ 4. ADVECTION       GPU   shaders/advection.wgsl        │  │   │
│  │  │    Petterssen predictor–corrector on mean wind          │  │   │
│  │  │    Trilinear interpolation, velocity→grid scaling       │  │   │
│  │  └────────────────────────┬───────────────────────────────┘  │   │
│  │  ┌────────────────────────▼───────────────────────────────┐  │   │
│  │  │ 5. HANNA PARAMS    GPU   shaders/hanna_params.wgsl     │  │   │
│  │  │    Per-particle σ_u,v,w and T_Lu,v,w from PBL state    │  │   │
│  │  └────────────────────────┬───────────────────────────────┘  │   │
│  │  ┌────────────────────────▼───────────────────────────────┐  │   │
│  │  │ 6. LANGEVIN        GPU   shaders/langevin.wgsl         │  │   │
│  │  │    Turbulent velocity update (Ornstein–Uhlenbeck)      │  │   │
│  │  │    Vertical sub-stepping (n=4) with PBL reflection     │  │   │
│  │  │    Hanna-short recalc between sub-steps                │  │   │
│  │  │    RNG: Philox4x32-10 → Box–Muller                     │  │   │
│  │  └────────────────────────┬───────────────────────────────┘  │   │
│  │  ┌────────────────────────▼───────────────────────────────┐  │   │
│  │  │ 7. DRY DEPOSITION  GPU   shaders/dry_deposition.wgsl   │  │   │
│  │  │    Mass attenuation for z < 2·h_ref                     │  │   │
│  │  └────────────────────────┬───────────────────────────────┘  │   │
│  │  ┌────────────────────────▼───────────────────────────────┐  │   │
│  │  │ 8. WET DEPOSITION  GPU   shaders/wet_deposition.wgsl   │  │   │
│  │  │    Mass attenuation from precipitation scavenging       │  │   │
│  │  └────────────────────────┘───────────────────────────────┘  │   │
│  │                                                              │   │
│  │  queue.submit(encoder)                                       │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│  ┌──────────────────────────────▼───────────────────────────────┐   │
│  │ 9. READBACK (optional) GPU→CPU                               │   │
│  │    Download particle buffer and deposition probabilities      │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│  ┌──────────────────────────────▼───────────────────────────────┐   │
│  │ 10. OUTPUT GRIDDING   GPU   shaders/concentration_gridding   │   │
│  │     (at output intervals)                                    │   │
│  │     Atomic-add particle masses to concentration grid         │   │
│  └──────────────────────────────┘───────────────────────────────┘   │
│                                                                     │
│  Advance time:  t ← t + dt                                          │
│  Loop until t ≥ t_end                                               │
└─────────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Detail

### 1. Release (`release/mod.rs`)

The `ReleaseManager` checks whether the current simulation time falls within
any active release window defined in `RELEASES`. If so, new particles are
initialised (position, mass, species) and uploaded to the GPU particle buffer.

**Fortran equivalent:** `releaseparticles` in `timemanager.f90`.

### 2. Meteorological Interpolation (`io/temporal.rs`)

Wind fields (`WindField3D`) and surface fields (`SurfaceFields`) are provided
as time brackets (t0, t1). A linear interpolation factor α is computed:

```
α = (t − t0) / (t1 − t0)
field(t) = (1 − α) · field(t0) + α · field(t1)
```

**Fortran equivalent:** `getfields` → `readwind` + temporal interpolation in
`timemanager.f90`.

### 3. PBL Diagnostics (`io/pbl_params.rs`)

From the interpolated surface fields (friction velocity, sensible heat flux,
surface stress, 2 m temperature), the PBL state is derived:

- friction velocity u\*
- convective velocity scale w\*
- Obukhov length L
- mixing height h

These are uploaded to a GPU buffer consumed by the Hanna kernel.

**Fortran equivalent:** `calcpar` → `obukhov`, `richardson`.

### 4. Advection — GPU (`shaders/advection.wgsl`)

Each particle is advected by the mean wind using the Petterssen
predictor–corrector scheme:

1. Sample wind at current position → (u₀, v₀, w₀)
2. Predict: x\_pred = x + dt · v₀
3. Sample wind at predicted position → (u₁, v₁, w₁)
4. Correct: x\_final = x + dt · 0.5·(v₀ + v₁)

Wind is trilinearly interpolated in (x, y, level). Velocities [m/s] are
converted to grid displacement via `VelocityToGridScale`.

**Dispatch:** `gpu/advection.rs` → `AdvectionDispatchKernel`

**Fortran equivalent:** `advance.f90` (lines 817–923).

### 5. Hanna Turbulence Parameters — GPU (`shaders/hanna_params.wgsl`)

For each particle, computes σ\_u, σ\_v, σ\_w, T\_Lu, T\_Lv, T\_Lw, dσ\_w/dz
from its height and the PBL state (u\*, w\*, L, h). The stability regime
(neutral / unstable / stable) is selected automatically.

Output is written to a per-particle `HannaParamsOutputBuffer`.

**Dispatch:** `gpu/hanna.rs` → `HannaDispatchKernel`

**Fortran equivalent:** `hanna.f90` (called from `advance.f90`).

### 6. Langevin Turbulent Diffusion — GPU (`shaders/langevin.wgsl`)

Updates the turbulent velocity fluctuations (turb\_u, turb\_v, turb\_w) using
the Langevin equation, then applies vertical displacement with sub-stepping:

```
for sub in 0..n_substeps:
    update turb_w (Ornstein–Uhlenbeck + drift)
    pos_z += turb_w · dt_sub
    PBL reflection (Thomson 1987)
    recompute Hanna at new z (hanna_short, except last sub-step)
```

Horizontal turbulent velocities are updated once (no sub-stepping needed).

Random variates: Philox4x32-10 counter-based RNG → Box–Muller for Gaussian
samples. The counter is deterministic and advanced per particle per step.

**Dispatch:** `gpu/langevin.rs` → `LangevinDispatchKernel`

**Fortran equivalent:** `advance.f90` (Langevin section) + `hanna_short.f90`.

### 7. Dry Deposition — GPU (`shaders/dry_deposition.wgsl`)

For particles with `pos_z < 2 · h_ref` (h\_ref = 15 m):

```
survival = exp(−v_d · |dt| / (2 · h_ref))
mass ← mass · survival
```

The deposition velocity v\_d is provided per particle by the forcing vector.

**Dispatch:** `gpu/deposition.rs` → `DryDepositionDispatchKernel`

**Fortran equivalent:** `drydepokernel.f90`, `getvdep.f90`.

### 8. Wet Deposition — GPU (`shaders/wet_deposition.wgsl`)

```
P_wet = gr_fraction · (1 − exp(−λ · |dt|))
mass ← mass · (1 − P_wet)
```

The scavenging coefficient λ and precipitating fraction are forcing inputs.

**Dispatch:** `gpu/wet_deposition.rs` → `WetDepositionDispatchKernel`

**Fortran equivalent:** `wetdepo.f90`, `wetdepokernel.f90`.

### 9. Host Readback (optional)

If `sync_particle_store_each_step` is enabled, the full particle buffer is
downloaded from GPU to CPU after each step. In performance mode this is
deferred (fire-and-forget), and only deposition probability vectors are
optionally retrieved.

### 10. Concentration Gridding — GPU (`shaders/concentration_gridding.wgsl`)

At output intervals, particle masses are binned into the 3D output grid
(OUTGRID) using atomic integer additions. The grid is then downloaded to CPU
and written to the output file.

**Dispatch:** `gpu/gridding.rs`

**Fortran equivalent:** `conccalc.f90` → `concoutput.f90`.

## GPU Command Encoding

Steps 4–8 are encoded into a **single `wgpu` command encoder** and submitted
as one batch. This minimises CPU–GPU round-trips: the five compute passes
execute back-to-back on the GPU without intermediate synchronisation.

```
encoder = device.create_command_encoder()
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
| Met I/O | `getfields` reads GRIB on disk | Pre-loaded time brackets, linear interpolation on CPU |
| PBL | `calcpar` (u\*, L, h) | `compute_pbl_parameters_from_met` (CPU) |
| Convection | Emanuel scheme (`convmix`) | Not yet implemented |
| Advection | `advance.f90` (per-particle loop) | GPU shader, all particles in parallel |
| Turbulence | `hanna` + Langevin in `advance.f90` | Two separate GPU passes (Hanna, then Langevin) |
| Deposition | `wetdepo`, `drydepokernel` | Two GPU passes |
| Output | `conccalc` + `concoutput` | GPU gridding + CPU download |
| Nested grids | Supported | Not yet implemented |

## Source Files

| Module | Path | CPU/GPU |
|--------|------|---------|
| Time loop orchestration | `src/simulation/timeloop.rs` | CPU (async) |
| Release manager | `src/release/mod.rs` | CPU |
| Met interpolation | `src/io/temporal.rs` | CPU |
| PBL parameters | `src/io/pbl_params.rs` | CPU |
| Advection kernel | `src/shaders/advection.wgsl` | GPU |
| Advection dispatch | `src/gpu/advection.rs` | CPU→GPU |
| Hanna kernel | `src/shaders/hanna_params.wgsl` | GPU |
| Hanna dispatch | `src/gpu/hanna.rs` | CPU→GPU |
| Langevin kernel | `src/shaders/langevin.wgsl` | GPU |
| Langevin dispatch | `src/gpu/langevin.rs` | CPU→GPU |
| Dry deposition kernel | `src/shaders/dry_deposition.wgsl` | GPU |
| Dry deposition dispatch | `src/gpu/deposition.rs` | CPU→GPU |
| Wet deposition kernel | `src/shaders/wet_deposition.wgsl` | GPU |
| Wet deposition dispatch | `src/gpu/wet_deposition.rs` | CPU→GPU |
| Concentration gridding | `src/shaders/concentration_gridding.wgsl` | GPU |
| Gridding dispatch | `src/gpu/gridding.rs` | CPU→GPU |
| Particle buffer | `src/gpu/buffers.rs` | GPU memory |
| Coordinate transforms | `src/coords/mod.rs` | CPU + GPU |
| ETEX driver (example) | `src/bin/etex-run.rs` | CPU (main) |
