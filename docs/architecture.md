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
│  │    1. Upload new wind field (if needed)          │     │
│  │    2. Release new particles (if needed)          │     │
│  │    3. Dispatch GPU kernels                       │     │
│  │    4. Download results (at output times only)    │     │
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
│  ┌─────────────┐ ┌─────────────┐ ┌────────────────────┐ │
│  │ Advection   │→│ Hanna PBL   │→│ Langevin           │ │
│  │ (Petterssen)│ │ Turbulence  │ │ (+ PBL reflection) │ │
│  └─────────────┘ └─────────────┘ └────────────────────┘ │
│  ┌─────────────┐ ┌─────────────┐ ┌────────────────────┐ │
│  │ Dry         │ │ Wet         │ │ Concentration      │ │
│  │ Deposition  │ │ Deposition  │ │ Gridding           │ │
│  └─────────────┘ └─────────────┘ └────────────────────┘ │
│  ┌─────────────┐                                        │
│  │ Philox RNG  │  Buffers: [particles] [wind] [pbl]     │
│  └─────────────┘           [output grid]                 │
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
│   │   ├── advection.wgsl        # Mean-wind advection
│   │   ├── hanna_params.wgsl     # Hanna σ and T_L computation
│   │   ├── langevin.wgsl         # Langevin + sub-stepping + PBL reflection
│   │   ├── dry_deposition.wgsl   # Dry deposition probability
│   │   ├── wet_deposition.wgsl   # Wet deposition probability
│   │   ├── concentration_gridding.wgsl  # Particle → grid accumulation
│   │   ├── pbl_reflection.wgsl   # Standalone PBL reflection (legacy)
│   │   ├── convective_mixing.wgsl
│   │   ├── cbl.wgsl
│   │   └── philox_rng.wgsl       # Counter-based RNG
│   │
│   ├── gpu/                      # GPU dispatch layer (wgpu plumbing)
│   │   ├── mod.rs                # Public GPU API
│   │   ├── buffers.rs            # Particle, wind, PBL buffer management
│   │   ├── advection.rs          # Advection kernel dispatch
│   │   ├── hanna.rs              # Hanna kernel dispatch
│   │   ├── langevin.rs           # Langevin kernel dispatch
│   │   ├── deposition.rs         # Dry deposition dispatch
│   │   ├── wet_deposition.rs     # Wet deposition dispatch
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
│   ├── run-etex.sh               # Complete ETEX validation pipeline
│   ├── compare-fortran.sh        # Synthetic Fortran comparison
│   ├── gpu-preflight.sh          # GPU backend check
│   └── etex/                     # ETEX helper scripts (ERA5, obs parsing)
│
├── docker/
│   ├── Dockerfile.gpu            # GPU build image (Ubuntu + Vulkan + Rust)
│   └── Dockerfile.fortran        # Fortran FLEXPART build image
│
├── docker-compose.yml            # Default compose (any Vulkan GPU)
└── docker-compose.nvidia.yml     # NVIDIA overlay
```

## CPU / GPU Boundary

The boundary is clear:

- **CPU** handles: config parsing, meteorological I/O, temporal interpolation,
  PBL parameter derivation, particle release, output writing, and
  orchestration of the time loop.
- **GPU** handles: all per-particle physics (advection, turbulence, deposition,
  gridding). These are embarrassingly parallel — each particle is independent
  within a timestep.

Steps 4–8 of the time loop are encoded into a **single GPU command encoder**
and submitted as one batch, minimising CPU↔GPU round-trips. See
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

Two images:

| Image | Dockerfile | Purpose |
|-------|-----------|---------|
| `flexpart-gpu` | `docker/Dockerfile.gpu` | Ubuntu 22.04 + Vulkan + Rust + eccodes + netcdf |
| `flexpart-fortran` | `docker/Dockerfile.fortran` | Fortran FLEXPART for oracle comparison |

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
