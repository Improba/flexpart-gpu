# Scientific changelog

Track physics-affecting changes to flexpart-gpu. Every PR that modifies
shaders, physics kernels, or advection logic must add an entry here.

## Format

```
### YYYY-MM-DD — <one-line summary>
**Impact**: <physics | numerics | output-only | none>
**Files**: <list of changed shaders / physics modules>
**Validation**: <what was tested, results>
```

---

## Entries

### 2026-03-06 — Fused Hanna+Langevin default production path
**Impact**: none (identical physics, different execution path)
**Files**: `langevin_fused.wgsl`, `gpu/langevin_fused.rs`, `simulation/timeloop.rs`
**Validation**: The production path fuses Hanna PBL turbulence parameterisation
and Langevin velocity update into a single GPU dispatch (`langevin_fused.wgsl`),
eliminating the intermediate HannaParams buffer and one dispatch barrier.
Physics is identical to the separated shaders (verified by textual diff of
every function). The separated Hanna → Langevin path is retained under
`FLEXPART_GPU_VALIDATION=1` for scientific validation.

*Note*: A full mega-kernel approach (advection + Hanna + Langevin + deposition
in one dispatch) was attempted and abandoned due to severe register pressure
on the target GPU. The mega-kernel source remains in `particle_step.wgsl` /
`particle_step.rs` for reference only.

### 2026-03-06 — GPU-side PBL diagnostics
**Impact**: numerics (minor — f32 vs previous CPU f32, same formulas)
**Files**: `pbl_diagnostics.wgsl`, `gpu/pbl.rs`, `simulation/timeloop.rs`
**Validation**: PBL parameters (u*, w*, L, h) now computed on GPU per grid
cell. Same formulas as CPU reference (`io/pbl_params.rs`). CPU reference
retained for tests.

### 2026-03-06 — GPU-side dual-wind temporal interpolation
**Impact**: numerics (minor — interpolation order changed)
**Files**: `advection_dual_wind.wgsl`, `advection_texture_dual_wind.wgsl`,
`gpu/advection.rs`, `simulation/timeloop.rs`
**Validation**: Wind brackets (t0, t1) uploaded once per met change. GPU
performs `(1−α)·t0 + α·t1` inline during advection. Removes per-step CPU
interpolation and wind re-upload. Interpolation result is mathematically
identical (same linear formula).

### 2026-03-06 — Active particle compaction
**Impact**: none (reorders particle buffer, physics unchanged)
**Files**: `compaction.wgsl`, `gpu/compaction.rs`, `simulation/timeloop.rs`
**Validation**: Prefix-sum compaction packs active particles to the front
of the buffer, reducing wasted GPU work on inactive particles.

### 2026-03-06 — hanna_short recalculation between sub-steps
**Impact**: physics
**Files**: `langevin.wgsl`
**Validation**: Fortran comparison shows vertical mean gap reduced from 86m
to 22m (74% improvement). sigma_z ratio = 0.94. Between each vertical
sub-step (except the last), `sigw`, `dsigwdz`, and `tlw` are recalculated
at the particle's new height using the Hanna (1982) profile equations,
exactly matching Fortran's `hanna_short(zt)` call in `advance.f90`.

### 2026-03-06 — Vertical turbulence sub-stepping (ifine=4)
**Impact**: physics
**Files**: `langevin.wgsl`, `advection.wgsl`, `advection_texture.wgsl`,
`gpu/langevin.rs`, `physics/langevin.rs`, `simulation/timeloop.rs`
**Validation**: All 35 tests pass. `physics_validation_advection_turbulence_pbl`
confirms PBL confinement, mass conservation, and advection direction with
n_substeps=4. Vertical turbulence now matches Fortran's `ifine` sub-stepping:
each timestep splits the vertical Langevin update into 4 sub-steps with
`dt_sub = dt/4`, applying displacement and PBL reflection between each.
Advection shader no longer applies `turb_w` to vertical displacement.

### 2026-03-06 — PBL boundary reflection kernel
**Impact**: physics
**Files**: `pbl_reflection.wgsl`, `gpu/pbl_reflection.rs`,
`simulation/timeloop.rs`
**Validation**: Fortran comparison shows GPU particles confined within
[0, BLH] with correct reflection behavior. Vertical mean z gap reduced
from >1000m to 180m vs Fortran.

### 2026-03-06 — Fix has_level_heights() ground-level detection
**Impact**: physics (critical)
**Files**: `advection.wgsl`, `advection_texture.wgsl`,
`concentration_gridding.wgsl`
**Validation**: Fixed bug where `level_heights[0]=0.0` caused fallback to
grid-level indexing. Now checks top-most level. Prevents particle clamping
to [0, nz-1] meters when first wind level is at ground.

### 2026-03-06 — Apply turbulent velocities to advection
**Impact**: physics (critical)
**Files**: `advection.wgsl`, `advection_texture.wgsl`
**Validation**: Particles now disperse correctly. Before this fix, Langevin
computed turb_u/v/w but advection ignored them — dispersion was zero.

### 2026-03-06 — Fix pos_z meters-vs-levels in advection and gridding
**Impact**: physics (critical)
**Files**: `advection.wgsl`, `advection_texture.wgsl`,
`concentration_gridding.wgsl`
**Validation**: `pos_z` (meters) is now correctly converted to fractional
grid level for wind sampling and to output level for gridding. Before,
particles were clamped to z < 3m.

---

## PR checklist (copy into PR description)

```markdown
### Scientific impact

- [ ] Does this PR modify shaders, physics modules, or advection logic?
- [ ] If yes: entry added to `docs/scientific-changelog.md`
- [ ] `cargo test` passes (all integration + scientific invariant tests)
- [ ] `physics_validation_advection_turbulence_pbl` test passes
- [ ] No new linter warnings in modified files
```
