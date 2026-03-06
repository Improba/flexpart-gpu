# Concentration Gridding

## Principle

The Lagrangian model carries mass on individual particles. To produce Eulerian
output fields (concentration in kg/m³, deposition in kg/m²), particle masses
are accumulated into a regular output grid at prescribed time intervals.

## Output Grid Definition

The output grid is defined in the `OUTGRID` configuration file:

| Parameter | Meaning |
|-----------|---------|
| `OUTLON0`, `OUTLAT0` | South-west corner (degrees) |
| `NUMXGRID`, `NUMYGRID` | Number of cells in x and y |
| `DXOUT`, `DYOUT` | Cell size (degrees) |
| `OUTHEIGHTS` | Upper boundary of each vertical layer [m] |

## Gridding Algorithm

For each active particle:

1. **Horizontal cell**: `(ix, iy) = (cell_x, cell_y)`, clamped to domain
   bounds.
2. **Vertical level**: linear scan through `outheights` to find the first
   level k such that `pos_z ≤ outheights[k]`.
3. **Mass contribution**: the particle's mass (after deposition losses) is
   added to the grid cell using atomic operations on GPU:
   ```
   mass_scaled = round(max(mass, 0) · mass_scale)
   atomicAdd(concentration[ix, iy, iz], mass_scaled)
   ```
4. **Particle count**: an atomic counter tracks the number of particles per
   cell (useful for diagnostics).

The final concentration in physical units is recovered by dividing by
`mass_scale` and by the cell volume:

```
C [kg/m³] = concentration_mass_scaled / (mass_scale · V_cell)
V_cell    = dx_m · dy_m · dz_m
```

## GPU Implementation Notes

- Atomic additions on GPU use integer arithmetic (scaled mass) to avoid
  floating-point atomics, which are not universally supported in WebGPU.
- The `mass_scale` factor (default 10⁶) controls the precision–range tradeoff
  of the integer representation.

## Source Files

| File | Role |
|------|------|
| `src/shaders/concentration_gridding.wgsl` | GPU gridding kernel |
| `src/gpu/gridding.rs` | GPU dispatch and buffer management |

## References

- Stohl, A. et al. (2005). *Atmos. Chem. Phys.*, 5, 2461–2474 (Section 2.7).
