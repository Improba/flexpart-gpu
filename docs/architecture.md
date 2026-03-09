# Architecture

## Design Decision: Standalone Rust Binary

`flexpart-gpu` is a **standalone Rust binary**, not an FFI plugin into the
Fortran code. Rationale:

- Avoids Fortran build complexity and mixed-language debugging.
- Full control over memory layout and GPU buffer management.
- Easier to distribute (single binary for crisis operations).
- The Fortran source serves as the **reference specification**, not as runtime
  code. Same inputs → compare outputs.

## High-Level Structure

```
┌──────────────────────────────────────────────────────────┐
│                    Rust Host (CPU)                        │
│                                                          │
│  ┌────────────┐ ┌────────────┐ ┌───────────┐            │
│  │ Config     │ │ Wind I/O   │ │ Particle  │            │
│  │ & Release  │ │ (GRIB2/    │ │ Release   │            │
│  │ Parser     │ │  NetCDF)   │ │ Manager   │            │
│  └─────┬──────┘ └─────┬──────┘ └─────┬─────┘            │
│        │              │              │                   │
│        ▼              ▼              ▼                   │
│  ┌─────────────────────────────────────────────────┐     │
│  │           Time Loop Manager                     │     │
│  │  for each timestep:                             │     │
│  │    1. Release new particles (if needed)          │     │
│  │    2. Upload wind brackets (once per met change) │     │
│  │    3. Upload surface fields for GPU PBL          │     │
│  │    4. Dispatch GPU physics (4 passes per step)   │     │
│  │    5. Download results (at output times only)    │     │
│  └──────────────────────┬──────────────────────────┘     │
│                         │                                │
│  ┌──────────────────────▼──────────────────────────┐     │
│  │       GPU Dispatch Layer (wgpu)                 │     │
│  │  Buffer management, pipeline caching,           │     │
│  │  command encoding, synchronization              │     │
│  └──────────────────────┬──────────────────────────┘     │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│                   GPU (WebGPU / WGSL)                    │
│                                                          │
│  PRODUCTION PATH (default) ─ 4 dispatches per step:      │
│  ┌─────────────┐ ┌────────────────────┐ ┌────────────┐ │
│  │ Advection   │→│ Fused Hanna+       │→│ Dry dep    │ │
│  │ (3D texture)│ │ Langevin           │ │ + Wet dep  │ │
│  └─────────────┘ └────────────────────┘ └────────────┘ │
│                                                          │
│  VALIDATION PATH (FLEXPART_GPU_VALIDATION=1) ─ 5 disp.: │
│  ┌─────────────┐ ┌─────────────┐ ┌────────────────────┐ │
│  │ Advection   │→│ Hanna PBL   │→│ Langevin           │ │
│  │ (3D texture)│ │ Turbulence  │ │ (+ PBL reflection) │ │
│  └─────────────┘ └─────────────┘ └────────────────────┘ │
│  ┌─────────────┐ ┌─────────────┐                        │
│  │ Dry         │ │ Wet         │                        │
│  │ Deposition  │ │ Deposition  │                        │
│  └─────────────┘ └─────────────┘                        │
│                                                          │
│  SHARED:                                                 │
│  ┌─────────────┐ ┌───────────┐ ┌────────────────────┐   │
│  │ PBL Diag.   │ │ Philox    │ │ Concentration      │   │
│  │ (GPU)       │ │ RNG       │ │ Gridding           │   │
│  └─────────────┘ └───────────┘ └────────────────────┘   │
│                                                          │
│  Buffers: [particles] [wind_t0+t1] [pbl] [output grid]  │
└──────────────────────────────────────────────────────────┘
```

## Source Tree

```
flexpart-gpu/
├── src/
│   ├── lib.rs                    # Library root, constants
│   ├── main.rs                   # Default binary entry point
│   ├── bin/
│   │   ├── etex-run.rs           # ETEX real-data simulation driver
│   │   ├── etex-validation.rs    # ETEX validation harness
│   │   ├── fortran-validation.rs # Synthetic Fortran comparison driver
│   │   ├── bench-timeloop.rs     # Standalone timeloop benchmark
│   │   └── gpu-preflight.rs      # GPU backend detection / smoke test
│   │
│   ├── simulation/
│   │   ├── mod.rs                # Public API
│   │   └── timeloop.rs           # Forward & backward time-loop orchestration
│   │
│   ├── physics/                  # CPU reference implementations
│   │   ├── advection.rs          # Petterssen predictor–corrector
│   │   ├── hanna.rs              # Hanna (1982) turbulence parameters
│   │   ├── langevin.rs           # Langevin equation
│   │   ├── deposition.rs         # Dry deposition (resistance model)
│   │   ├── wet_scavenging.rs     # Wet scavenging
│   │   ├── convection.rs         # Emanuel convection (simplified)
│   │   ├── cbl.rs                # Convective boundary layer profiles
│   │   ├── interpolation.rs      # Trilinear interpolation
│   │   └── rng.rs                # Philox CPU reference
│   │
│   ├── shaders/                  # WGSL GPU compute kernels
│   │   ├── langevin_fused.wgsl   # ★ FUSED Hanna+Langevin (production default)
│   │   │                         #   Inline Hanna PBL + Langevin turbulence
│   │   ├── particle_step.wgsl    # Mega-kernel (abandoned, kept for reference)
│   │   ├── advection.wgsl        # Mean-wind advection (validation path)
│   │   ├── advection_dual_wind.wgsl      # Dual-wind bracket advection
│   │   ├── advection_texture_dual_wind.wgsl  # 3D texture-sampled variant
│   │   ├── advection_texture.wgsl        # 3D texture advection (single wind)
│   │   ├── hanna_params.wgsl     # Hanna σ and T_L computation
│   │   ├── langevin.wgsl         # Langevin + sub-stepping + PBL reflection
│   │   ├── dry_deposition.wgsl   # Dry deposition probability
│   │   ├── wet_deposition.wgsl   # Wet deposition probability
│   │   ├── pbl_diagnostics.wgsl  # GPU PBL parameter computation
│   │   ├── compaction.wgsl       # Active particle compaction (prefix-sum)
│   │   ├── concentration_gridding.wgsl  # Particle → grid accumulation
│   │   ├── pbl_reflection.wgsl   # Standalone PBL reflection (legacy)
│   │   ├── wind_trilinear_interp.wgsl  # GPU wind interpolation helper
│   │   ├── convective_mixing.wgsl
│   │   ├── cbl.wgsl
│   │   └── philox_rng.wgsl       # Counter-based RNG
│   │
│   ├── gpu/                      # GPU dispatch layer (wgpu plumbing)
│   │   ├── mod.rs                # Public GPU API
│   │   ├── langevin_fused.rs     # ★ Fused Hanna+Langevin dispatch (production)
│   │   ├── particle_step.rs      # Mega-kernel dispatch (abandoned, kept for ref)
│   │   ├── buffers.rs            # Particle, wind, PBL buffer management
│   │   ├── advection.rs          # Advection kernel dispatch (+ dual-wind)
│   │   ├── hanna.rs              # Hanna kernel dispatch
│   │   ├── langevin.rs           # Langevin kernel dispatch
│   │   ├── deposition.rs         # Dry deposition dispatch
│   │   ├── wet_deposition.rs     # Wet deposition dispatch
│   │   ├── pbl.rs                # GPU PBL diagnostics dispatch
│   │   ├── compaction.rs         # Active particle compaction dispatch
│   │   ├── gridding.rs           # Concentration gridding dispatch
│   │   ├── pbl_reflection.rs     # PBL reflection dispatch
│   │   ├── convection.rs         # Convective mixing dispatch
│   │   ├── cbl.rs                # CBL dispatch
│   │   ├── interpolation.rs      # Wind interpolation dispatch
│   │   ├── rng.rs                # RNG dispatch
│   │   ├── workgroup.rs          # Workgroup auto-tuning
│   │   └── preflight.rs          # GPU capability detection
│   │
│   ├── io/                       # Meteorological I/O
│   │   ├── mod.rs
│   │   ├── grib2.rs              # GRIB2 reader (eccodes)
│   │   ├── grib2_async.rs        # Async GRIB prefetch (background thread)
│   │   ├── netcdf.rs             # NetCDF reader
│   │   ├── netcdf_output.rs      # NetCDF output writer
│   │   ├── vertical_transform.rs # Hybrid σ-pressure → height
│   │   ├── temporal.rs           # Temporal interpolation between met brackets
│   │   └── pbl_params.rs         # PBL parameter derivation (u*, L, h)
│   │
│   ├── particles/mod.rs          # Particle data structure (96 bytes, repr(C))
│   ├── coords/mod.rs             # Coordinate transforms (lat/lon ↔ grid ↔ m)
│   ├── wind/                     # Wind field structures and synthetic generators
│   │   ├── mod.rs
│   │   └── synthetic.rs
│   ├── pbl/mod.rs                # PBL state structures
│   ├── release/mod.rs            # Particle release manager
│   ├── config/mod.rs             # Configuration file parser
│   └── validation/mod.rs         # Validation metrics (RMSE, bias, correlation)
│
├── tests/                        # Integration and validation tests
│   ├── forward_timeloop.rs
│   ├── backward_timeloop.rs
│   ├── cpu_gpu_comparison.rs
│   ├── convection_chain.rs
│   ├── validation_etex.rs
│   └── integration/
│       ├── mass_conservation.rs
│       ├── physics_validation.rs
│       ├── scientific_invariants.rs
│       ├── deposition_decay.rs
│       └── source_receptor_consistency.rs
│
├── benches/advection.rs          # Criterion benchmarks (GPU + CPU)
│
├── fixtures/etex/                # ETEX test data and config
│   ├── real/config/              # Real ETEX-1 configuration
│   └── scaffold/                 # Synthetic scaffold for CI
│
├── scripts/
│   ├── run-etex.sh               # ETEX pipeline (GPU-only default, optional Fortran)
│   ├── compare-fortran.sh        # Synthetic Fortran comparison
│   ├── gpu-preflight.sh          # GPU backend check
│   └── etex/                     # ETEX helper scripts (ERA5, obs parsing)
│
├── docker/
│   └── Dockerfile.gpu            # GPU build image (Ubuntu + Vulkan + Rust)
│
├── docker-compose.yml            # Default compose (any Vulkan GPU)
└── docker-compose.nvidia.yml     # NVIDIA overlay

# Fortran Docker is in a separate sibling directory:
# ../flexpart-fortran-docker/
#   ├── Dockerfile
#   └── docker-compose.yml
```

## Execution Paths

The project supports two execution paths, selectable via environment variable:

| Path | Activation | Purpose |
|------|-----------|---------|
| **Fused H+L** (default) | Default, no env var needed | Production. 4 dispatches: advection → fused Hanna+Langevin → dry dep → wet dep. |
| **Separated** | `FLEXPART_GPU_VALIDATION=1` | Scientific validation. 5 dispatches: advection → Hanna → Langevin → dry dep → wet dep. |

## CPU / GPU Boundary

- **CPU** handles: config parsing, meteorological I/O (GRIB reading, async
  prefetch), particle release, output writing, and time-loop orchestration.
- **GPU** handles: wind temporal interpolation (dual-wind brackets uploaded
  once per met change, GPU interpolates with α), PBL diagnostics (`u*`, `w*`,
  `L`, `h` computed on GPU from surface fields), all per-particle physics
  (advection, turbulence, deposition), active particle compaction, and
  concentration gridding.

In production mode, four dispatches per timestep are encoded into a single
`wgpu` command encoder: advection, fused Hanna+Langevin, dry deposition, and
wet deposition. The fused kernel eliminates the intermediate HannaParams buffer.
In validation mode, five separate dispatches are encoded into the same encoder. See
[science/simulation-flow.md](science/simulation-flow.md) for the full
execution sequence.

## GPU Backend (wgpu / WebGPU)

The project uses [wgpu](https://wgpu.rs/) which provides a portable abstraction
over Vulkan, Metal, DX12, and OpenGL. Compute shaders are written in WGSL.

| Host GPU | Backend | `WGPU_BACKEND` |
|----------|---------|-----------------|
| NVIDIA (Linux) | Vulkan | `vulkan` (default) |
| AMD (Linux) | Vulkan (RADV) | `vulkan` |
| Intel (Linux) | Vulkan (ANV) | `vulkan` |
| Apple Silicon | Metal | `metal` |
| No GPU (CI) | CPU software | `gl` |

## Docker Environment

All development and execution can happen inside Docker containers for
reproducible builds and portable GPU access.

Two images (in separate projects):

| Image | Location | Purpose |
|-------|----------|---------|
| `flexpart-gpu` | `docker/Dockerfile.gpu` (this project) | Ubuntu 22.04 + Vulkan + Rust + eccodes + netcdf |
| `flexpart-fortran` | `../flexpart-fortran-docker/Dockerfile` (sibling directory) | Fortran FLEXPART for oracle comparison |

Named volumes (`cargo-home`, `cargo-target`) persist the Rust build cache
between container restarts. See [development.md](development.md) for usage.

## Relationship to Fortran FLEXPART

The Fortran FLEXPART codebase (in `../flexpart/`) is the **validation oracle**,
not a runtime dependency. The relationship:

- Same physics equations, same algorithms (Petterssen, Hanna, Langevin, Thomson).
- Independent implementations (Fortran vs Rust+WGSL).
- Validation by comparing outputs on identical inputs.
- Known architectural differences are documented in
  [validation-report.md](validation-report.md).
