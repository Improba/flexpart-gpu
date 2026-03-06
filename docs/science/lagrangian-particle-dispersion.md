# Lagrangian Particle Dispersion

## Principle

Instead of solving the advection–diffusion equation on a fixed grid (Eulerian
approach), FLEXPART tracks a large number of computational particles that move
with the mean wind and are perturbed by stochastic turbulent fluctuations. The
concentration field is reconstructed by counting particle masses in output grid
cells.

This Lagrangian approach is naturally suited to:

- point and line sources (no numerical diffusion),
- complex terrain and boundary-layer turbulence,
- backward-time source attribution (receptor → source).

## Governing Stochastic Differential Equation

Each particle's position **x** evolves as:

```
dx_i = (u_i + u'_i) dt
```

where `u_i` is the resolved (mean) wind component and `u'_i` is the turbulent
velocity fluctuation. The turbulent component is modelled by a Langevin
equation (see [turbulent-diffusion.md](turbulent-diffusion.md)).

## Time Integration Structure

The forward time loop processes each timestep `dt` in the following order:

1. **Release** — inject new particles according to user-defined source terms.
2. **Meteorological interpolation** — trilinear interpolation of wind and
   surface fields to each particle's position and time.
3. **PBL diagnostics** — compute boundary-layer parameters (u\*, w\*, L, h)
   from surface fields.
4. **Advection** — Petterssen predictor–corrector step using mean wind
   (see [advection.md](advection.md)).
5. **Hanna parameterisation** — compute per-particle turbulence statistics
   σ\_u, σ\_v, σ\_w and Lagrangian timescales T\_Lu, T\_Lv, T\_Lw.
6. **Langevin step** — update turbulent velocities and apply vertical
   displacement with sub-stepping and PBL reflection
   (see [turbulent-diffusion.md](turbulent-diffusion.md)).
7. **Dry deposition** — mass loss for particles near the surface
   (see [deposition.md](deposition.md)).
8. **Wet deposition** — mass loss from precipitation scavenging.
9. **Concentration gridding** — accumulate surviving particle masses on the
   output grid (see [concentration-gridding.md](concentration-gridding.md)).

### Backward Mode

In backward mode the advection direction is reversed (`-dt`), while turbulence
and deposition use positive `dt`. Particles are emitted from receptor locations
and the resulting field maps the sensitivity to upwind sources (source–receptor
matrix; Seibert & Frank, 2004).

## Particle State Vector

Each particle carries (96 bytes, C-layout):

| Field | Type | Meaning |
|-------|------|---------|
| `pos_x`, `pos_y` | f32 | fractional position within grid cell [0, 1) |
| `pos_z` | f32 | height above ground [m] |
| `cell_x`, `cell_y` | i32 | grid cell indices |
| `flags` | u32 | bit 0 = active |
| `mass[4]` | f32 | mass per species [kg] |
| `vel_u`, `vel_v`, `vel_w` | f32 | mean wind from last step [m/s] |
| `turb_u`, `turb_v`, `turb_w` | f32 | turbulent velocity [m/s] |
| `time` | i32 | simulation time [s] |
| `timestep` | i32 | integration timestep [s] |
| `release_point` | i32 | source index |

The split `cell + frac` representation avoids float32 precision loss at large
grid coordinates.

## References

- Stohl, A. et al. (2005). *Atmos. Chem. Phys.*, 5, 2461–2474.
- Seibert, P. and Frank, A. (2004). Source–receptor matrix calculation with a
  Lagrangian particle dispersion model in backward mode. *Atmos. Chem. Phys.*,
  4, 51–63.
- Flesch, T. K. et al. (1995). Backward-time Lagrangian stochastic dispersion
  models. *Boundary-Layer Meteorol.*, 77, 187–208.
