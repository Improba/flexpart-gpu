# Scientific Foundations

This directory documents the physical and mathematical principles implemented
in `flexpart-gpu`. Each page covers one aspect of the Lagrangian particle
dispersion model, with the actual equations used in the code and references
to the underlying literature.

## Contents

| Document | Topic |
|----------|-------|
| [lagrangian-particle-dispersion.md](lagrangian-particle-dispersion.md) | Overview of the Lagrangian particle dispersion method and time integration |
| [simulation-flow.md](simulation-flow.md) | Step-by-step execution flow of a simulation (what runs when, CPU vs GPU) |
| [advection.md](advection.md) | Mean-wind advection (Petterssen predictor–corrector) |
| [turbulent-diffusion.md](turbulent-diffusion.md) | Langevin equation and Hanna (1982) turbulence parameterisation |
| [deposition.md](deposition.md) | Dry deposition (resistance model) and wet scavenging |
| [concentration-gridding.md](concentration-gridding.md) | Mapping particle masses to Eulerian output grids |
| [coordinate-system.md](coordinate-system.md) | Horizontal and vertical coordinate transforms |
| [known-limitations.md](known-limitations.md) | Known limitations, simplifications, and open questions |

## Key References

- Stohl, A. et al. (2005). Technical note: The Lagrangian particle dispersion
  model FLEXPART version 6.2. *Atmos. Chem. Phys.*, 5, 2461–2474.
  doi:10.5194/acp-5-2461-2005
- Thomson, D. J. (1987). Criteria for the selection of stochastic models of
  particle trajectories in turbulent flows. *J. Fluid Mech.*, 180, 529–556.
- Hanna, S. R. (1982). Applications in air pollution modeling. In
  *Atmospheric Turbulence and Air Pollution Modelling*, ed. F. T. M. Nieuwstadt
  and H. van Dop, Reidel, 275–310.
- Rodean, H. C. (1996). *Stochastic Lagrangian Models of Turbulent Diffusion*.
  Meteorological Monographs, vol. 26, no. 48, AMS.
- Petterssen, S. (1940). *Weather Analysis and Forecasting*. McGraw-Hill.
