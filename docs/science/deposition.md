# Deposition

Particles lose mass through two removal processes: **dry deposition** at the
surface and **wet deposition** (scavenging) by precipitation.

## 1. Dry Deposition

### Resistance Model

For gaseous species the deposition velocity is:

```
v_d = 1 / (r_a + r_b + r_c)
```

| Resistance | Formula | Meaning |
|------------|---------|---------|
| r\_a (aerodynamic) | `(ln(z_ref/z₀) − ψ(z_ref) + ψ(z₀)) / (κ · u*)` | Transport through the surface layer (Monin–Obukhov) |
| r\_b (quasi-laminar) | `2/(κ · u*) · Sc^(2/3)` | Molecular diffusion across the laminar sublayer; Sc = ν/D |
| r\_c (surface) | user-provided | Uptake by vegetation, soil, water |

For particulate species, gravitational settling is added:

```
v_dep = v_set + 1 / (r_a + r_dp + r_a · r_dp · v_set)
```

where:
- `v_set` is the Stokes settling velocity,
- `St = (v_set / g) · u*² / ν` is the Stokes number,
- `r_dp = 1 / ((Sc^(2/3) + 10^(−3/St)) · u*)`.

### Deposition Layer

Only particles below `2 · h_ref` (h\_ref = 15 m by default) experience dry
deposition. The probability formulation avoids removing entire particles:

```
survival = exp(−v_d · |dt| / (2 · h_ref))
mass_new = mass_old · survival
P_dry    = 1 − survival
```

## 2. Wet Deposition

### Below-Cloud Scavenging

**Gases:**

```
λ = A · prec^B
```

where A and B are species-specific coefficients (`weta_gas`, `wetb_gas`).

**Aerosols (rain, T ≥ 273 K):**

Based on Laakso et al. (2003):

```
log₁₀(λ) = poly₀ + poly₁·(log d)⁻⁴ + … + poly₅·√prec
```

with a polynomial fit in particle diameter d and precipitation rate.

**Aerosols (snow, T < 273 K):**

Same polynomial form with different coefficients (Kyrö et al., 2009).

### In-Cloud Scavenging

**Aerosols:**

```
S_i = frac_act / cl
λ   = incloud_ratio · S_i · prec / 3.6×10⁶
```

- `frac_act = liquid_frac · ccn + ice_frac · ice_act`
- Cloud water content from input or fallback: `cl = 10⁶ · 2×10⁻⁷ · prec^0.36`

**Gases:**

```
cle = (1 − cl) / (H · (R_air/3500) · T) + cl
S_i = 1 / cle
```

where H is the Henry's law coefficient.

### Mass Update

```
P_wet    = gr_fraction · (1 − exp(−λ · |dt|))
mass_new = mass_old · (1 − P_wet)
```

`gr_fraction` is the precipitating area fraction at the particle's location.

## Source Files

| File | Role |
|------|------|
| `src/shaders/dry_deposition.wgsl` | GPU dry deposition kernel (both paths) |
| `src/shaders/wet_deposition.wgsl` | GPU wet deposition kernel (both paths) |
| `src/physics/deposition.rs` | CPU dry deposition (resistance model, reference) |
| `src/physics/wet_scavenging.rs` | CPU wet scavenging (reference) |
| `src/gpu/deposition.rs` | Dry deposition dispatch |
| `src/gpu/wet_deposition.rs` | Wet deposition dispatch |

## References

- Stohl, A. et al. (2005). *Atmos. Chem. Phys.*, 5, 2461–2474 (Sections 2.5–2.6).
- Laakso, L. et al. (2003). Ultrafine particle scavenging coefficients.
  *J. Geophys. Res.*, 108(D1), 4021.
- Kyrö, E.-M. et al. (2009). Snow scavenging of ultrafine particles.
  *Boreal Env. Res.*, 14, 527–538.
