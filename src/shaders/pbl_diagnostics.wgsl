// PBL diagnostics compute shader (S-02 / Tier 2.1).
//
// Ported from pbl_params.rs: compute_pbl_parameters_from_met()
// Reference: calcpar.f90, obukhov.f90 (FLEXPART 10.4)
//
// Each workgroup item processes one grid cell independently, computing:
//   - Friction velocity u* from surface stress or neutral log-law
//   - Obukhov length L from u* and sensible heat flux
//   - Mixing height hmix (clamp met-provided or fallback)
//   - Convective velocity scale w* for unstable conditions
//
// TODO: Bulk Richardson diagnostics from optional profile points (requires
//       additional input buffers; see pbl_params.rs profile_point path)

// ---------------------------------------------------------------------------
// Physical constants (must match src/lib.rs constants module)
// ---------------------------------------------------------------------------

const R_AIR: f32 = 287.05;
const GA: f32 = 9.81;
const CPA: f32 = 1004.6;
const VON_KARMAN: f32 = 0.4;
const USTAR_MIN: f32 = 1.0e-4;
const NEUTRAL_OLI_THRESHOLD: f32 = 1.0e-5;
const F32_MAX_FINITE: f32 = 3.4028235e38;

// ---------------------------------------------------------------------------
// Structures
// ---------------------------------------------------------------------------

struct SurfaceCellInput {
    surface_pressure_pa: f32,
    temperature_2m_k: f32,
    u10_ms: f32,
    v10_ms: f32,
    surface_stress_n_m2: f32,
    sensible_heat_flux_w_m2: f32,
    solar_radiation_w_m2: f32,
    mixing_height_m: f32,
    friction_velocity_ms: f32,
    inv_obukhov_length_per_m: f32,
    _pad0: f32,
    _pad1: f32,
};

struct PblDiagnosticsParams {
    grid_nx: u32,
    grid_ny: u32,
    cell_count: u32,
    _pad0: u32,
    roughness_length_m: f32,
    wind_reference_height_m: f32,
    heat_flux_neutral_threshold_w_m2: f32,
    _pad1: f32,
    hmix_min_m: f32,
    hmix_max_m: f32,
    fallback_mixing_height_m: f32,
    _pad2: f32,
};

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

// Packed surface field input (one SurfaceCellInput per grid cell)
@group(0) @binding(0)
var<storage, read> surface_input: array<SurfaceCellInput>;

// Output PBL fields — layout-compatible with PblBuffers (separate f32 arrays)
@group(0) @binding(1)
var<storage, read_write> out_ustar: array<f32>;

@group(0) @binding(2)
var<storage, read_write> out_wstar: array<f32>;

@group(0) @binding(3)
var<storage, read_write> out_hmix: array<f32>;

@group(0) @binding(4)
var<storage, read_write> out_oli: array<f32>;

@group(0) @binding(5)
var<storage, read_write> out_sshf: array<f32>;

@group(0) @binding(6)
var<storage, read_write> out_ssr: array<f32>;

@group(0) @binding(7)
var<storage, read_write> out_surfstr: array<f32>;

// Dispatch parameters (grid dims + computation options)
@group(0) @binding(8)
var<uniform> params: PblDiagnosticsParams;

// ---------------------------------------------------------------------------
// Utility functions (match pbl_params.rs sanitize_* helpers)
// ---------------------------------------------------------------------------

fn is_finite_scalar(v: f32) -> bool {
    return (v == v) && (abs(v) <= F32_MAX_FINITE);
}

fn sanitize_positive(v: f32, fallback: f32) -> f32 {
    if (is_finite_scalar(v) && v > 0.0) {
        return v;
    }
    return fallback;
}

fn sanitize_non_negative(v: f32, fallback: f32) -> f32 {
    if (is_finite_scalar(v) && v >= 0.0) {
        return v;
    }
    return fallback;
}

fn sanitize_finite(v: f32, fallback: f32) -> f32 {
    if (is_finite_scalar(v)) {
        return v;
    }
    return fallback;
}

fn positive_infinity() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

// ---------------------------------------------------------------------------
// Physics: friction velocity u* [m/s]
// Ported from pbl_params.rs: estimate_friction_velocity_m_s()
//
// Precedence:
// 1. If valid stress and density: u* = sqrt(tau / rho)
// 2. Else neutral log-law from 10 m wind
// ---------------------------------------------------------------------------

fn estimate_friction_velocity(
    stress: f32,
    air_density: f32,
    u10: f32,
    v10: f32,
    z0: f32,
    z_ref: f32,
) -> f32 {
    let rho = sanitize_positive(air_density, 1.225);
    let safe_stress = sanitize_non_negative(stress, 0.0);
    if (safe_stress > 0.0) {
        return max(sqrt(safe_stress / rho), USTAR_MIN);
    }

    let safe_z0 = sanitize_positive(z0, 0.1);
    let safe_z_ref = max(sanitize_positive(z_ref, 10.0), 1.01 * safe_z0);
    let wind_speed = max(sqrt(u10 * u10 + v10 * v10), 0.0);
    if (wind_speed <= 0.0) {
        return USTAR_MIN;
    }

    let ustar = VON_KARMAN * wind_speed / max(log(safe_z_ref / safe_z0), 1.0e-6);
    return max(sanitize_positive(ustar, USTAR_MIN), USTAR_MIN);
}

// ---------------------------------------------------------------------------
// Physics: Obukhov length L [m] from surface flux
// L = -(rho * cp * T * u*^3) / (kappa * g * H)
// Ported from pbl_params.rs: obukhov_length_from_surface_flux_m()
// ---------------------------------------------------------------------------

fn obukhov_length_from_flux(
    ustar: f32,
    temp_k: f32,
    air_density: f32,
    heat_flux: f32,
    threshold: f32,
) -> f32 {
    let h = sanitize_finite(heat_flux, 0.0);
    if (abs(h) < max(threshold, 0.0)) {
        return positive_infinity();
    }

    let u = max(sanitize_positive(ustar, USTAR_MIN), USTAR_MIN);
    let t = sanitize_positive(temp_k, 300.0);
    let d = sanitize_positive(air_density, 1.225);

    let numerator = -(d * CPA * t * u * u * u);
    let denominator = VON_KARMAN * GA * h;
    if (abs(denominator) < 1.0e-9) {
        return positive_infinity();
    }

    let ol = numerator / denominator;
    if (!is_finite_scalar(ol) || abs(ol) > 1.0e9) {
        return positive_infinity();
    }
    return ol;
}

// Ported from pbl_params.rs: inverse_obukhov_length_per_m()
fn inverse_obukhov_from_length(ol: f32) -> f32 {
    if (!is_finite_scalar(ol) || abs(ol) > 1.0e5) {
        return 0.0;
    }
    return 1.0 / ol;
}

fn clamp_mixing_height(v: f32) -> f32 {
    return max(clamp(v, params.hmix_min_m, params.hmix_max_m), params.hmix_min_m);
}

// ---------------------------------------------------------------------------
// Physics: convective velocity scale w* [m/s]
// w* = (g/T * H/(rho*cp) * hmix)^(1/3)  for H > 0
// Ported from pbl_params.rs: compute_convective_velocity_scale_m_s()
// ---------------------------------------------------------------------------

fn convective_velocity_scale(
    heat_flux: f32,
    air_density: f32,
    temp_k: f32,
    hmix: f32,
) -> f32 {
    if (heat_flux <= 0.0) {
        return 0.0;
    }
    let d = sanitize_positive(air_density, 1.225);
    let t = sanitize_positive(temp_k, 300.0);
    let h = sanitize_positive(hmix, params.hmix_min_m);
    let buoyancy_flux = heat_flux / (d * CPA) * GA / t;
    if (buoyancy_flux <= 0.0 || !is_finite_scalar(buoyancy_flux)) {
        return 0.0;
    }
    return max(pow(buoyancy_flux * h, 1.0 / 3.0), 0.0);
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

@compute @workgroup_size(__WORKGROUP_SIZE_X__)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let cell_id = gid.y * (nwg.x * __WORKGROUP_SIZE_X__u) + gid.x;
    if (cell_id >= params.cell_count) {
        return;
    }

    let input = surface_input[cell_id];

    let temp_k = sanitize_positive(input.temperature_2m_k, 300.0);
    let pressure_pa = sanitize_positive(input.surface_pressure_pa, 101325.0);
    let air_density = pressure_pa / (R_AIR * temp_k);

    // Friction velocity: prefer met-provided if valid, else estimate
    let estimated_ustar = estimate_friction_velocity(
        input.surface_stress_n_m2,
        air_density,
        input.u10_ms,
        input.v10_ms,
        params.roughness_length_m,
        params.wind_reference_height_m,
    );
    var ustar: f32;
    if (is_finite_scalar(input.friction_velocity_ms) && input.friction_velocity_ms > 0.0) {
        ustar = max(input.friction_velocity_ms, USTAR_MIN);
    } else {
        ustar = max(estimated_ustar, USTAR_MIN);
    }

    // Obukhov length: prefer met-provided 1/L if valid, else compute from flux
    let computed_ol = obukhov_length_from_flux(
        ustar,
        temp_k,
        air_density,
        input.sensible_heat_flux_w_m2,
        params.heat_flux_neutral_threshold_w_m2,
    );
    var oli: f32;
    if (is_finite_scalar(input.inv_obukhov_length_per_m)
        && abs(input.inv_obukhov_length_per_m) >= NEUTRAL_OLI_THRESHOLD) {
        oli = input.inv_obukhov_length_per_m;
    } else {
        oli = inverse_obukhov_from_length(computed_ol);
    }

    // Mixing height: use met-provided if valid, else fallback
    // TODO: port bulk Richardson mixing-height estimate from profile points
    //       (pbl_params.rs: estimate_mixing_height_from_bulk_richardson_m)
    var hmix: f32;
    if (is_finite_scalar(input.mixing_height_m) && input.mixing_height_m > 0.0) {
        hmix = clamp_mixing_height(input.mixing_height_m);
    } else {
        hmix = clamp_mixing_height(params.fallback_mixing_height_m);
    }

    // Convective velocity scale w*
    let safe_sshf = sanitize_finite(input.sensible_heat_flux_w_m2, 0.0);
    let wstar = convective_velocity_scale(safe_sshf, air_density, temp_k, hmix);

    // Write outputs (layout-compatible with PblBuffers)
    out_ustar[cell_id] = ustar;
    out_wstar[cell_id] = wstar;
    out_hmix[cell_id] = hmix;
    out_oli[cell_id] = oli;
    out_sshf[cell_id] = sanitize_finite(input.sensible_heat_flux_w_m2, 0.0);
    out_ssr[cell_id] = sanitize_non_negative(input.solar_radiation_w_m2, 0.0);
    out_surfstr[cell_id] = sanitize_non_negative(input.surface_stress_n_m2, 0.0);
}
