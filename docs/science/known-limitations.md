# Known Limitations and Open Questions

This document tracks deliberate simplifications, inherited choices, and
not-yet-implemented features in `flexpart-gpu`. Each item includes its
scientific impact, affected files, and — where applicable — what would be
needed to resolve it.

## 1. Neutral σ\_v = σ\_w in the Hanna Parameterisation

**Severity:** low–medium · **Status:** inherited from Fortran FLEXPART

In Hanna (1982), the neutral-stability profiles are:

```
σ_u = σ_v = 2 · u* · exp(-3×10⁻⁴ · z/u*)
σ_w       = 1.3 · u* · exp(-2×10⁻⁴ · z/u*)
```

The code assigns `σ_v = σ_w` instead of `σ_v = σ_u`. This means the
lateral velocity variance is underestimated in neutral conditions (σ\_w < σ\_u
by roughly 35 % at z = 0). The same choice appears in Fortran FLEXPART's
`hanna.f90`, so this may be intentional (possibly to compensate for missing
mesoscale diffusion), but it deviates from the strict Hanna (1982)
formulation.

**Impact:** horizontal dispersion in neutral conditions is narrower than
the published parameterisation predicts. For short-range releases (Seveso
use case), the effect is small because unstable/stable regimes dominate
daytime/nighttime dispersion.

**Files:** `src/shaders/hanna_params.wgsl`, `src/shaders/langevin_fused.wgsl`
(inline `compute_hanna`), `src/physics/hanna.rs`

**To resolve:** cross-check Fortran `hanna.f90` and, if confirmed as
intentional, add a code comment citing the Fortran source. If unintentional,
set `σ_v = σ_u` in the neutral branch (in all three files).

## 2. Additive σ Floor Values

**Severity:** low · **Status:** deliberate

The Hanna kernel adds a constant floor (`SIGMA_FLOOR = 0.01 m/s`) to each
sigma component:

```
σ_w = SIGMA_FLOOR + 1.3 · u* · exp(…)
```

A strict reading of the Hanna (1982) floor would use `max(σ, floor)`
instead. The additive form slightly raises σ everywhere, not only near zero.
The practical difference is negligible (0.01 m/s is well below typical
turbulent velocities), but it is a minor departure from the reference.

**Files:** `src/shaders/hanna_params.wgsl`, `src/shaders/langevin_fused.wgsl`
(inline `compute_hanna`), `src/physics/hanna.rs`

## 3. Convection (Emanuel Scheme) Not Yet Implemented

**Severity:** medium for tropical/deep-convective scenarios ·
**Status:** not yet implemented

The Fortran FLEXPART `convmix` routine (Emanuel convective scheme) is not
yet ported. Particles in deep-convection environments will not be
redistributed vertically by convective mass fluxes.

**Impact:** for the primary use case (European industrial accidents, Seveso
sites), deep convection is rare and the effect on surface concentrations is
small. For tropical or summertime thunderstorm scenarios, vertical mixing
will be underestimated.

**Files:** `src/physics/convection.rs` (stub/simplified version exists)

**To resolve:** port the full Emanuel scheme from `convect.f90` /
`convmix.f90`.

## 4. Nested Output Grids Not Yet Implemented

**Severity:** low · **Status:** not yet implemented

Fortran FLEXPART supports nested grids for higher-resolution output near
the source. `flexpart-gpu` currently supports a single output grid.

**Impact:** users needing high-resolution output near the source must run
with a uniformly fine grid, which increases memory and compute cost.

**To resolve:** extend `concentration_gridding.wgsl` and
`gpu/gridding.rs` to support multiple output grids with different
resolutions and domains.

## 5. Single Deposition Velocity / Scavenging Coefficient per Particle

**Severity:** low · **Status:** design choice

The dry and wet deposition GPU kernels apply a single deposition velocity
(v\_d) and scavenging coefficient (λ) per particle. All four species mass
slots on each particle receive the same survival factor.

This is correct when all species share the same deposition properties, but
will need extension for multi-species simulations where species have
different v\_d or λ values.

**Files:** `src/shaders/dry_deposition.wgsl`, `src/shaders/wet_deposition.wgsl`

**To resolve:** pass per-species v\_d / λ arrays and compute per-slot
survival factors in the shader.

## 6. No Pole or Date Line Handling in Coordinates

**Severity:** low · **Status:** out of scope for current use case

The coordinate system (`src/coords/mod.rs`) uses a regular lat–lon grid
without wrapping at the date line (180°) or special handling at the poles.
`dx_meters_at_latitude(90°)` correctly returns 0, but particles reaching
the pole or crossing the date line are not handled.

**Impact:** none for the target domain (European Seveso sites). Would need
attention for global or polar simulations.

**Files:** `src/coords/mod.rs`

## 7. f32 Precision Throughout

**Severity:** low · **Status:** deliberate

All GPU computation uses `f32`. This is justified in `AGENTS.md`:
Monte Carlo convergence (1/√N) dominates over floating-point precision for
particle counts typical of emergency-response scenarios (10⁴–10⁶ particles).

CPU/GPU parity tests enforce max relative error < 10⁻⁴ between the `f32`
GPU path and the `f32` CPU reference.

For very long simulations (weeks) or very high particle counts, accumulated
rounding in position updates could become significant, but this has not been
observed in ETEX-scale validation.

## 8. Hanna Physics Duplicated Across Shaders

**Severity:** low · **Status:** by design (performance trade-off)

The Hanna turbulence parameterisation logic exists in three locations:

- `src/shaders/hanna_params.wgsl` — standalone Hanna kernel (validation path)
- `src/shaders/langevin_fused.wgsl` — inlined `compute_hanna()` (production path)
- `src/physics/hanna.rs` — CPU reference implementation

The production fused kernel embeds Hanna inline to eliminate an intermediate
buffer and dispatch barrier. Any physics change to the Hanna parameterisation
must be applied to all three files.

**Impact:** risk of silent divergence between production and validation paths
if one file is updated without the others.

**Mitigation:** the `compute_hanna()` function in `langevin_fused.wgsl` is a
verbatim copy of the one in `hanna_params.wgsl` (verified by textual diff).
Integration tests exercise the production path; validation mode exercises the
separated path.

## Summary

| # | Item | Severity | Status |
|---|------|----------|--------|
| 1 | Neutral σ\_v = σ\_w | Low–Medium | Inherited from Fortran |
| 2 | Additive σ floors | Low | Deliberate |
| 3 | No convection (Emanuel) | Medium | Not yet implemented |
| 4 | No nested grids | Low | Not yet implemented |
| 5 | Single v\_d / λ per particle | Low | Design choice |
| 6 | No pole / date line | Low | Out of scope |
| 7 | f32 precision | Low | Deliberate |
| 8 | Hanna physics duplicated across shaders | Low | By design |
