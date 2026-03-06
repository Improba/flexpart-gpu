# Mean-Wind Advection

## Scheme: Petterssen Predictor–Corrector

FLEXPART uses a two-step scheme attributed to Petterssen (1940), which
provides second-order accuracy without storing intermediate derivatives.

### Algorithm

Given a particle at position **x₀** with timestep `dt`:

1. **Interpolate** wind at current position and time:
   `(u₀, v₀, w₀) = wind(x₀, t)`

2. **Predictor** (Euler forward):
   `x_pred = x₀ + dt · (u₀, v₀, w₀)`

3. **Interpolate** wind at predicted position:
   `(u₁, v₁, w₁) = wind(x_pred, t + dt)`

4. **Corrected wind**:
   `u_corr = 0.5 · (u₀ + u₁)`

5. **Final position**:
   `x₁ = x₀ + dt · u_corr`

This is equivalent to an explicit trapezoidal rule and removes the first-order
bias of a pure Euler step in spatially varying wind fields.

### Velocity-to-Grid Scaling

Wind is in m/s; particle coordinates are in fractional grid cells. The
conversion uses per-particle scale factors:

```
Δcell_x = u [m/s] · x_grid_per_meter · dt
Δcell_y = v [m/s] · y_grid_per_meter · dt
Δz      = w [m/s] · dt                        (vertical stays in metres)
```

where:
- `x_grid_per_meter = 1 / (R_earth · cos(lat) · Δλ · π/180)`
- `y_grid_per_meter = 1 / (R_earth · Δφ · π/180)`

### Wind Interpolation

Trilinear interpolation in (x, y, z):

- **Horizontal**: bilinear on the four surrounding grid points.
- **Vertical**: linear between the two enclosing model levels, after mapping
  `pos_z` [m] to a fractional level index via the `level_heights_m` array.
- **Temporal**: linear between the two bracketing meteorological time steps.

Only the **mean** wind is interpolated here. Turbulent fluctuations are handled
separately by the Langevin equation.

## Source Files

| File | Role |
|------|------|
| `src/shaders/advection_texture_dual_wind.wgsl` | Production — 3D texture-sampled dual-wind advection |
| `src/shaders/advection_dual_wind.wgsl` | Buffer-based dual-wind advection (fallback / validation) |
| `src/shaders/advection.wgsl` | Legacy single-wind advection |
| `src/physics/advection.rs` | CPU reference |
| `src/gpu/advection.rs` | Advection dispatch and buffer setup (both paths) |

## References

- Petterssen, S. (1940). *Weather Analysis and Forecasting*. McGraw-Hill.
- Stohl, A. et al. (2005). *Atmos. Chem. Phys.*, 5, 2461–2474 (Section 2.2).
