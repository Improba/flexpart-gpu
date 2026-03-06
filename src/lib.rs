#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod config;
pub mod coords;
pub mod gpu;
pub mod io;
pub mod particles;
pub mod pbl;
pub mod physics;
pub mod release;
pub mod simulation;
pub mod validation;
pub mod wind;

/// Physical constants used throughout the model.
/// Values match FLEXPART Fortran `par_mod.f90`.
pub mod constants {
    /// Pi
    pub const PI: f32 = std::f32::consts::PI;
    /// Earth radius [m]
    pub const R_EARTH: f32 = 6.371e6;
    /// Individual gas constant for dry air [J/(kg·K)]
    pub const R_AIR: f32 = 287.05;
    /// Gravitational acceleration [m/s²]
    pub const GA: f32 = 9.81;
    /// Specific heat for dry air at constant pressure [J/(kg·K)]
    pub const CPA: f32 = 1004.6;
    /// Exponent for potential temperature: R_AIR / CPA
    pub const KAPPA: f32 = 0.286;
    /// Pi / 180
    pub const PI180: f32 = PI / 180.0;
    /// Von Kármán constant
    pub const VON_KARMAN: f32 = 0.4;
    /// Universal gas constant [J/(mol·K)]
    pub const RGAS: f32 = 8.31447;
    /// Specific gas constant for water vapor [J/(kg·K)]
    pub const R_WATER: f32 = 461.495;
    /// Reference height for dry deposition [m]
    pub const HREF: f32 = 15.0;
    /// Convective kinetic energy factor
    pub const CONVKE: f32 = 2.0;
    /// Minimum allowed PBL height [m]
    pub const HMIX_MIN: f32 = 100.0;
    /// Maximum allowed PBL height [m]
    pub const HMIX_MAX: f32 = 4500.0;
    /// Tropospheric horizontal diffusivity [m²/s]
    pub const D_TROP: f32 = 50.0;
    /// Stratospheric vertical diffusivity [m²/s]
    pub const D_STRAT: f32 = 0.1;
    /// Mesoscale turbulence scaling factor
    pub const TURB_MESOSCALE: f32 = 0.16;
    /// Water density [kg/m³]
    pub const RHO_WATER: f32 = 1000.0;
    /// Ratio of molar weights of water vapor and dry air
    pub const XMWML: f32 = 18.016 / 28.960;
    /// Minimum particle mass before termination
    pub const MIN_MASS: f32 = 0.0001;
}
