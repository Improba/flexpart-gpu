//! Physics modules: advection, turbulence, deposition, wet scavenging, convection.
//!
//! Each sub-module provides a CPU reference implementation and
//! corresponding WGSL compute shader for GPU execution.

pub mod advection;
pub mod cbl;
pub mod convection;
pub mod deposition;
pub mod hanna;
pub mod interpolation;
pub mod langevin;
pub mod rng;
pub mod wet_scavenging;

pub use advection::{
    advect_particle_cpu, advect_particle_cpu_euler, advect_particles_cpu,
    VelocityToGridScale, MAX_VERTICAL_LEVELS,
};
pub use cbl::{
    cbl_transition_factor, compute_cbl_bigaussian_pdf, compute_cbl_moments, infer_cbl_branch,
    reconstruct_cbl_moments, sample_cbl_branch_constrained_velocity, sample_cbl_vertical_velocity,
    sample_cbl_vertical_velocity_with_rng, CblBiGaussianPdf, CblBranch, CblGaussianComponent,
    CblPdfInputs, CblPdfMoments,
};
pub use convection::{
    apply_convective_mixing_to_particles_cpu, apply_redistribution_matrix,
    build_convective_redistribution_matrix, build_simplified_convection_chain,
    compute_simplified_emanuel_column, expected_destination_height_m, level_index_from_height,
    ConvectionError, ConvectiveRedistributionMatrix, SimplifiedEmanuelColumn,
    SimplifiedEmanuelInputs,
};
pub use deposition::{
    accumulate_dry_deposition_probability, aerodynamic_resistance_s_m,
    dry_deposition_probability_step, dry_deposition_velocity_from_resistances_m_s,
    dynamic_viscosity_air_kg_m_s, gas_dry_deposition_velocity_m_s,
    gas_quasi_laminar_resistance_s_m, in_dry_deposition_layer, kinematic_viscosity_air_m2_s,
    particle_dry_deposition_velocity_m_s, water_vapor_diffusivity_m2_s, DryDepMeteoInputs,
    GasSpeciesDepositionInput, LandUseResistance, ParticleBinDepositionInput,
    ParticleSpeciesDepositionInput,
};
pub use hanna::{
    compute_hanna_params, compute_hanna_params_from_pbl, obukhov_length_from_inverse, HannaInputs,
};
pub use interpolation::{interpolate_wind_trilinear, WindVector};
pub use langevin::{
    box_muller_normals, sample_langevin_noise, update_particle_turbulence_langevin_cpu,
    update_particle_turbulence_langevin_with_rng_cpu, update_particles_turbulence_langevin_cpu,
    update_particles_turbulence_langevin_with_rng_cpu, LangevinError, LangevinNoise, LangevinStep,
    TurbulentVelocity,
};
pub use rng::{
    philox4x32, philox4x32_uniforms, philox4x32_with_rounds, philox_counter_add, u32_to_uniform01,
    PhiloxCounter, PhiloxKey, PhiloxRng, PHILOX_ROUNDS,
};
pub use wet_scavenging::{
    apply_wet_scavenging_mass_step, below_cloud_scavenging_coefficient_aerosol,
    below_cloud_scavenging_coefficient_gas, in_cloud_scavenging_coefficient_aerosol,
    in_cloud_scavenging_coefficient_gas, wet_precipitation_state, wet_scavenged_mass_loss_kg,
    wet_scavenging_coefficient, wet_scavenging_probability_step, AerosolBelowCloudParams,
    AerosolInCloudParams, BelowCloudInputs, CloudScavengingRegime, GasBelowCloudParams,
    GasInCloudParams, InCloudInputs, WetPrecipitationInputs, WetPrecipitationState,
    WetScavengingStep,
};
