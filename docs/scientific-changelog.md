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
