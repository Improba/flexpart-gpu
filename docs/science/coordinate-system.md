# Coordinate System

## Horizontal Coordinates

### Grid-Relative Representation

Particle positions are stored as **(cell index, fractional offset)**:

```
absolute_x = cell_x + pos_x      (pos_x ∈ [0, 1))
absolute_y = cell_y + pos_y      (pos_y ∈ [0, 1))
```

This split avoids precision loss inherent to single-precision floats at large
grid indices: a float32 has ~7 significant digits, which is insufficient for
sub-metre resolution on a global 0.25° grid (1440 cells), but perfectly
adequate for the fractional part alone.

### Grid ↔ Geographic

```
lon = absolute_x · Δλ + λ₀
lat = absolute_y · Δφ + φ₀
```

where (λ₀, φ₀) is the south-west corner and (Δλ, Δφ) is the grid spacing in
degrees.

### Metres ↔ Degrees

Conversion factors depend on latitude:

```
dx_m = R_earth · cos(lat · π/180) · Δλ · π/180    [m per grid cell in x]
dy_m = R_earth · Δφ · π/180                        [m per grid cell in y]
```

The inverse factors (`x_grid_per_meter`, `y_grid_per_meter`) are used by the
advection kernel to convert wind velocities [m/s] into grid displacement per
timestep.

### Great-Circle Distance

For diagnostics, the Haversine formula is available:

```
a = sin²(Δφ/2) + cos(φ₁)·cos(φ₂)·sin²(Δλ/2)
d = 2 · R_earth · arctan2(√a, √(1−a))
```

## Vertical Coordinate

### Height Above Ground

The primary vertical coordinate is **z [m] above ground level** (`pos_z`).

- In the PBL (z ≤ h): turbulence is parameterised by Hanna (1982) and the
  Langevin equation operates directly in metres.
- Particles are reflected at z = 0 (ground) and z = h (PBL top) by the
  Thomson (1987) scheme.

### Mapping to Model Levels

Meteorological fields are stored on discrete model levels with known heights
`level_heights_m[k]`. The particle height is mapped to a fractional level index
for trilinear wind interpolation:

```
find k such that level_heights_m[k] ≤ pos_z < level_heights_m[k+1]
frac = (pos_z − level_heights_m[k]) / (level_heights_m[k+1] − level_heights_m[k])
level_index = k + frac
```

### Output Levels

The output grid uses independently defined vertical layers (from `OUTHEIGHTS`).
Particles are binned by their `pos_z` into the appropriate output layer.

## Physical Constants

| Constant | Value | Unit |
|----------|-------|------|
| R\_earth | 6.371 × 10⁶ | m |
| g | 9.81 | m/s² |
| κ (von Kármán) | 0.4 | — |
| R\_air | 287.05 | J/(kg·K) |

## Source Files

| File | Role |
|------|------|
| `src/coords/mod.rs` | Coordinate transforms (Rust) |
| `src/shaders/advection.wgsl` | Velocity-to-grid scaling in GPU kernel |
