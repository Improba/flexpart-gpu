// WGSL mega-kernel: complete particle physics step in a single dispatch (S-05 / Tier 3.4).
//
// Performs all per-particle physics in one pass, eliminating all inter-kernel
// dispatch barriers and intermediate buffer round-trips:
//   1. Advection — Petterssen predictor-corrector with dual-wind temporal interpolation
//   2. Hanna PBL turbulence — sigma_u, sigma_v, sigma_w, T_L computation
//   3. Langevin turbulent velocity update — sub-stepping with PBL reflection
//   4. Dry deposition — exponential mass survival factor
//   5. Wet deposition — scavenging mass survival factor
//
// Physics is identical to the separated kernel path (advection_dual_wind.wgsl +
// hanna_params.wgsl + langevin.wgsl + dry_deposition.wgsl + wet_deposition.wgsl).
// This is the default production path; set FLEXPART_GPU_VALIDATION=1 to use
// the separated multi-dispatch path for scientific debugging.
//
// Bindings (single bind group):
//   @binding(0):  particles storage buffer (read_write)
//   @binding(1):  wind_u_t0 (read)
//   @binding(2):  wind_v_t0 (read)
//   @binding(3):  wind_w_t0 (read)
//   @binding(4):  wind_u_t1 (read)
//   @binding(5):  wind_v_t1 (read)
//   @binding(6):  wind_w_t1 (read)
//   @binding(7):  packed_pbl — ustar ++ wstar ++ hmix ++ oli (read)
//   @binding(8):  uniform ParticleStepParams
//   @binding(9):  packed_deposition — vdep ++ scav_coeff ++ precip_frac (read)
//
// Requires maxStorageBuffersPerShaderStage >= 9 (1 rw + 8 read-only).

const FLAG_ACTIVE: u32 = 1u;

// --- Hanna constants (Hanna 1982) ---
const USTAR_MIN: f32 = 1.0e-4;
const TLU_MIN: f32 = 10.0;
const TLV_MIN: f32 = 10.0;
const TLW_MIN: f32 = 30.0;
const SIGMA_FLOOR: f32 = 1.0e-2;
const H_MIN: f32 = 1.0;
const NEUTRAL_OLI_THRESHOLD: f32 = 1.0e-5;

// --- Philox-4x32 RNG constants ---
const PHILOX_M0: u32 = 0xD2511F53u;
const PHILOX_M1: u32 = 0xCD9E8D57u;
const PHILOX_W0: u32 = 0x9E3779B9u;
const PHILOX_W1: u32 = 0xBB67AE85u;
const PHILOX_ROUNDS: u32 = 10u;
const U32_TO_UNIT_SCALE_24BIT: f32 = 1.0 / 16777216.0;
const BOX_MULLER_U_MIN: f32 = 1.1754944e-38;
const BOX_MULLER_U_MAX: f32 = 0.99999994;
const TWO_PI: f32 = 6.2831855;
const F32_MAX_FINITE: f32 = 3.4028235e38;
const MAX_SUBSTEPS: u32 = 4u;

// --- Dry deposition constants (advance.f90 / get_vdep_prob.f90) ---
const HREF_MIN_M: f32 = 0.1;

// ---------------------------------------------------------------------------
// Structures
// ---------------------------------------------------------------------------

struct Particle {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    cell_x: i32,
    cell_y: i32,
    flags: u32,
    mass: array<f32, 4>,
    vel_u: f32,
    vel_v: f32,
    vel_w: f32,
    turb_u: f32,
    turb_v: f32,
    turb_w: f32,
    time: i32,
    timestep: i32,
    time_mem: i32,
    time_split: i32,
    release_point: i32,
    class_id: i32,
    cbt: i32,
    pad0: u32,
};

struct HannaParams {
    ust: f32,
    wst: f32,
    ol: f32,
    h: f32,
    zeta: f32,
    sigu: f32,
    sigv: f32,
    sigw: f32,
    dsigwdz: f32,
    dsigw2dz: f32,
    tlu: f32,
    tlv: f32,
    tlw: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
};

struct ParticleStepParams {
    // Wind grid dimensions
    nx: u32,
    ny: u32,
    nz: u32,
    particle_count: u32,
    // Advection
    dt_seconds: f32,
    x_scale: f32,
    y_scale: f32,
    z_scale: f32,
    alpha: f32,
    // Philox RNG state
    key0: u32,
    key1: u32,
    counter0: u32,
    counter1: u32,
    counter2: u32,
    counter3: u32,
    // Langevin sub-stepping
    n_substeps: u32,
    rho_grad_over_rho: f32,
    min_height_m: f32,
    // PBL grid
    pbl_nx: u32,
    pbl_ny: u32,
    // Dry deposition
    reference_height_m: f32,
    // 3 pads to reach 16-byte alignment for the vec4 array (offset 84 → 96)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    // 16 level heights packed as 4 x vec4<f32> (uniform alignment)
    level_heights: array<vec4<f32>, 4>,
};

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<storage, read> wind_u_t0: array<f32>;

@group(0) @binding(2)
var<storage, read> wind_v_t0: array<f32>;

@group(0) @binding(3)
var<storage, read> wind_w_t0: array<f32>;

@group(0) @binding(4)
var<storage, read> wind_u_t1: array<f32>;

@group(0) @binding(5)
var<storage, read> wind_v_t1: array<f32>;

@group(0) @binding(6)
var<storage, read> wind_w_t1: array<f32>;

// Packed PBL: ustar[pbl_nx*pbl_ny] ++ wstar[..] ++ hmix[..] ++ oli[..]
@group(0) @binding(7)
var<storage, read> packed_pbl: array<f32>;

@group(0) @binding(8)
var<uniform> params: ParticleStepParams;

// Packed deposition: vdep[particle_count] ++ scav_coeff[..] ++ precip_frac[..]
@group(0) @binding(9)
var<storage, read> packed_deposition: array<f32>;

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

fn is_finite_scalar(value: f32) -> bool {
    return (value == value) && (abs(value) <= F32_MAX_FINITE);
}

fn sanitize_positive(value: f32) -> f32 {
    if (is_finite_scalar(value) && value > 0.0) {
        return value;
    }
    return 0.0;
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a * (1.0 - t) + b * t;
}

// ---------------------------------------------------------------------------
// Wind field: dual-time trilinear interpolation (from advection_dual_wind.wgsl)
// ---------------------------------------------------------------------------

fn wind_flat_index(ix: u32, iy: u32, iz: u32) -> u32 {
    return ((ix * params.ny) + iy) * params.nz + iz;
}

fn get_level_height(k: u32) -> f32 {
    return params.level_heights[k / 4u][k % 4u];
}

fn has_level_heights() -> bool {
    return get_level_height(params.nz - 1u) > 0.0;
}

fn meters_to_grid_level(z_m: f32) -> f32 {
    if (!has_level_heights()) {
        return z_m;
    }
    let h0 = get_level_height(0u);
    if (z_m <= h0) {
        if (h0 > 0.0) {
            return z_m / h0;
        }
        return 0.0;
    }
    for (var k = 1u; k < params.nz; k = k + 1u) {
        let hk = get_level_height(k);
        if (z_m <= hk) {
            let hprev = get_level_height(k - 1u);
            let denom = hk - hprev;
            if (denom > 0.0) {
                return f32(k - 1u) + (z_m - hprev) / denom;
            }
            return f32(k - 1u);
        }
    }
    return f32(params.nz - 1u);
}

fn max_domain_height() -> f32 {
    if (!has_level_heights()) {
        return f32(params.nz - 1u);
    }
    return get_level_height(params.nz - 1u);
}

fn clamp_axis(coord: f32, n: u32) -> vec3<f32> {
    let max_index = f32(n - 1u);
    let clamped = clamp(coord, 0.0, max_index);
    let lower_f = floor(clamped);
    let lower = u32(lower_f);
    let upper = min(lower + 1u, n - 1u);
    let frac = clamped - lower_f;
    return vec3<f32>(f32(lower), f32(upper), frac);
}

fn sample_dual_u(x: f32, y: f32, z: f32) -> f32 {
    let x_axis = clamp_axis(x, params.nx);
    let y_axis = clamp_axis(y, params.ny);
    let z_level = meters_to_grid_level(z);
    let z_axis = clamp_axis(z_level, params.nz);

    let x0 = u32(x_axis.x); let x1 = u32(x_axis.y); let tx = x_axis.z;
    let y0 = u32(y_axis.x); let y1 = u32(y_axis.y); let ty = y_axis.z;
    let z0 = u32(z_axis.x); let z1 = u32(z_axis.y); let tz = z_axis.z;

    let i000 = wind_flat_index(x0, y0, z0);
    let i100 = wind_flat_index(x1, y0, z0);
    let i010 = wind_flat_index(x0, y1, z0);
    let i110 = wind_flat_index(x1, y1, z0);
    let i001 = wind_flat_index(x0, y0, z1);
    let i101 = wind_flat_index(x1, y0, z1);
    let i011 = wind_flat_index(x0, y1, z1);
    let i111 = wind_flat_index(x1, y1, z1);

    let val_t0 = lerp(
        lerp(lerp(wind_u_t0[i000], wind_u_t0[i100], tx), lerp(wind_u_t0[i010], wind_u_t0[i110], tx), ty),
        lerp(lerp(wind_u_t0[i001], wind_u_t0[i101], tx), lerp(wind_u_t0[i011], wind_u_t0[i111], tx), ty),
        tz,
    );
    let val_t1 = lerp(
        lerp(lerp(wind_u_t1[i000], wind_u_t1[i100], tx), lerp(wind_u_t1[i010], wind_u_t1[i110], tx), ty),
        lerp(lerp(wind_u_t1[i001], wind_u_t1[i101], tx), lerp(wind_u_t1[i011], wind_u_t1[i111], tx), ty),
        tz,
    );
    return lerp(val_t0, val_t1, params.alpha);
}

fn sample_dual_v(x: f32, y: f32, z: f32) -> f32 {
    let x_axis = clamp_axis(x, params.nx);
    let y_axis = clamp_axis(y, params.ny);
    let z_level = meters_to_grid_level(z);
    let z_axis = clamp_axis(z_level, params.nz);

    let x0 = u32(x_axis.x); let x1 = u32(x_axis.y); let tx = x_axis.z;
    let y0 = u32(y_axis.x); let y1 = u32(y_axis.y); let ty = y_axis.z;
    let z0 = u32(z_axis.x); let z1 = u32(z_axis.y); let tz = z_axis.z;

    let i000 = wind_flat_index(x0, y0, z0);
    let i100 = wind_flat_index(x1, y0, z0);
    let i010 = wind_flat_index(x0, y1, z0);
    let i110 = wind_flat_index(x1, y1, z0);
    let i001 = wind_flat_index(x0, y0, z1);
    let i101 = wind_flat_index(x1, y0, z1);
    let i011 = wind_flat_index(x0, y1, z1);
    let i111 = wind_flat_index(x1, y1, z1);

    let val_t0 = lerp(
        lerp(lerp(wind_v_t0[i000], wind_v_t0[i100], tx), lerp(wind_v_t0[i010], wind_v_t0[i110], tx), ty),
        lerp(lerp(wind_v_t0[i001], wind_v_t0[i101], tx), lerp(wind_v_t0[i011], wind_v_t0[i111], tx), ty),
        tz,
    );
    let val_t1 = lerp(
        lerp(lerp(wind_v_t1[i000], wind_v_t1[i100], tx), lerp(wind_v_t1[i010], wind_v_t1[i110], tx), ty),
        lerp(lerp(wind_v_t1[i001], wind_v_t1[i101], tx), lerp(wind_v_t1[i011], wind_v_t1[i111], tx), ty),
        tz,
    );
    return lerp(val_t0, val_t1, params.alpha);
}

fn sample_dual_w(x: f32, y: f32, z: f32) -> f32 {
    let x_axis = clamp_axis(x, params.nx);
    let y_axis = clamp_axis(y, params.ny);
    let z_level = meters_to_grid_level(z);
    let z_axis = clamp_axis(z_level, params.nz);

    let x0 = u32(x_axis.x); let x1 = u32(x_axis.y); let tx = x_axis.z;
    let y0 = u32(y_axis.x); let y1 = u32(y_axis.y); let ty = y_axis.z;
    let z0 = u32(z_axis.x); let z1 = u32(z_axis.y); let tz = z_axis.z;

    let i000 = wind_flat_index(x0, y0, z0);
    let i100 = wind_flat_index(x1, y0, z0);
    let i010 = wind_flat_index(x0, y1, z0);
    let i110 = wind_flat_index(x1, y1, z0);
    let i001 = wind_flat_index(x0, y0, z1);
    let i101 = wind_flat_index(x1, y0, z1);
    let i011 = wind_flat_index(x0, y1, z1);
    let i111 = wind_flat_index(x1, y1, z1);

    let val_t0 = lerp(
        lerp(lerp(wind_w_t0[i000], wind_w_t0[i100], tx), lerp(wind_w_t0[i010], wind_w_t0[i110], tx), ty),
        lerp(lerp(wind_w_t0[i001], wind_w_t0[i101], tx), lerp(wind_w_t0[i011], wind_w_t0[i111], tx), ty),
        tz,
    );
    let val_t1 = lerp(
        lerp(lerp(wind_w_t1[i000], wind_w_t1[i100], tx), lerp(wind_w_t1[i010], wind_w_t1[i110], tx), ty),
        lerp(lerp(wind_w_t1[i001], wind_w_t1[i101], tx), lerp(wind_w_t1[i011], wind_w_t1[i111], tx), ty),
        tz,
    );
    return lerp(val_t0, val_t1, params.alpha);
}

fn split_horizontal_coord(coord: f32, n: u32) -> vec2<f32> {
    let max_index = f32(n - 1u);
    let clamped = clamp(coord, 0.0, max_index);
    var cell = u32(floor(clamped));
    var frac = clamped - f32(cell);
    if (cell >= n - 1u) {
        cell = n - 1u;
        frac = 0.0;
    }
    return vec2<f32>(f32(cell), frac);
}

fn clamp_vertical_coord(coord: f32) -> f32 {
    return clamp(coord, 0.0, max_domain_height());
}

// ---------------------------------------------------------------------------
// PBL lookup (packed struct-of-arrays buffer)
// ---------------------------------------------------------------------------

fn pbl_flat_index(i: u32, j: u32) -> u32 {
    return i * params.pbl_ny + j;
}

fn obukhov_length_from_inverse(oli: f32) -> f32 {
    if (abs(oli) < NEUTRAL_OLI_THRESHOLD) {
        return bitcast<f32>(0x7f800000u); // +Inf
    }
    return 1.0 / oli;
}

// ---------------------------------------------------------------------------
// Hanna turbulence computation (Hanna 1982, identical to hanna_params.wgsl)
// ---------------------------------------------------------------------------

fn compute_hanna(ust_in: f32, wst_in: f32, ol_in: f32, h_in: f32, z_in: f32) -> HannaParams {
    var ust = max(ust_in, USTAR_MIN);
    var wst = max(wst_in, 0.0);
    var h = max(h_in, H_MIN);
    var z = clamp(z_in, 0.0, h);
    let ol = ol_in;
    let zeta = clamp(z / h, 0.0, 1.0);

    var sigu: f32;
    var sigv: f32;
    var sigw: f32;
    var dsigwdz: f32;
    var tlu: f32;
    var tlv: f32;
    var tlw: f32;

    if (!is_finite_scalar(ol) || ol == 0.0 || h / abs(ol) < 1.0) {
        let corr = z / ust;
        sigu = SIGMA_FLOOR + 2.0 * ust * exp(-3.0e-4 * corr);
        sigw = 1.3 * ust * exp(-2.0e-4 * corr);
        dsigwdz = -2.0e-4 * sigw;
        sigw = sigw + SIGMA_FLOOR;
        sigv = sigw;

        tlu = 0.5 * z / sigw / (1.0 + 1.5e-3 * corr);
        tlv = tlu;
        tlw = tlu;
    } else if (ol < 0.0) {
        sigu = SIGMA_FLOOR + ust * pow(12.0 - 0.5 * h / ol, 1.0 / 3.0);
        sigv = sigu;

        let sigw_sq = 1.2 * wst * wst * (1.0 - 0.9 * zeta) * pow(zeta, 2.0 / 3.0)
            + (1.8 - 1.4 * zeta) * ust * ust;
        sigw = sqrt(sigw_sq) + SIGMA_FLOOR;

        dsigwdz = 0.5 / sigw / h
            * (-1.4 * ust * ust
                + wst * wst * (0.8 * pow(max(zeta, 1.0e-3), -1.0 / 3.0) - 1.8 * pow(zeta, 2.0 / 3.0)));

        tlu = 0.15 * h / sigu;
        tlv = tlu;
        if (z < abs(ol)) {
            tlw = 0.1 * z / (sigw * (0.55 - 0.38 * abs(z / ol)));
        } else if (zeta < 0.1) {
            tlw = 0.59 * z / sigw;
        } else {
            tlw = 0.15 * h / sigw * (1.0 - exp(-5.0 * zeta));
        }
    } else {
        sigu = SIGMA_FLOOR + 2.0 * ust * (1.0 - zeta);
        sigv = SIGMA_FLOOR + 1.3 * ust * (1.0 - zeta);
        sigw = sigv;
        dsigwdz = -1.3 * ust / h;

        tlu = 0.15 * h / sigu * sqrt(zeta);
        tlv = 0.467 * tlu;
        tlw = 0.1 * h / sigw * pow(zeta, 0.8);
    }

    tlu = max(tlu, TLU_MIN);
    tlv = max(tlv, TLV_MIN);
    tlw = max(tlw, TLW_MIN);

    if (dsigwdz == 0.0) {
        dsigwdz = 1.0e-10;
    }

    if (!is_finite_scalar(sigu) || sigu <= 0.0) { sigu = SIGMA_FLOOR; }
    if (!is_finite_scalar(sigv) || sigv <= 0.0) { sigv = SIGMA_FLOOR; }
    if (!is_finite_scalar(sigw) || sigw <= 0.0) { sigw = SIGMA_FLOOR; }
    if (!is_finite_scalar(tlu)) { tlu = TLU_MIN; }
    if (!is_finite_scalar(tlv)) { tlv = TLV_MIN; }
    if (!is_finite_scalar(tlw)) { tlw = TLW_MIN; }
    if (!is_finite_scalar(dsigwdz)) { dsigwdz = 1.0e-10; }

    let dsigw2dz = 2.0 * sigw * dsigwdz;
    return HannaParams(
        ust, wst, ol, h,
        zeta, sigu, sigv, sigw,
        dsigwdz, dsigw2dz, tlu, tlv,
        tlw, 0.0, 0.0, 0.0
    );
}

// Vertical Hanna re-computation at a new height (Fortran hanna_short).
fn hanna_short_w(ust_in: f32, wst_in: f32, ol: f32, h_in: f32, z_in: f32,
                 out_sigw: ptr<function, f32>,
                 out_dsigwdz: ptr<function, f32>,
                 out_tlw: ptr<function, f32>) {
    let ust = max(ust_in, USTAR_MIN);
    let wst = max(wst_in, 0.0);
    let h = max(h_in, H_MIN);
    let z = clamp(z_in, 0.0, h);
    let zeta = clamp(z / h, 0.0, 1.0);

    var sigw: f32;
    var dsigwdz: f32;
    var tlw: f32;

    if (!is_finite_scalar(ol) || ol == 0.0 || h / abs(ol) < 1.0) {
        let corr = z / ust;
        sigw = 1.3 * ust * exp(-2.0e-4 * corr);
        dsigwdz = -2.0e-4 * sigw;
        sigw = sigw + SIGMA_FLOOR;
        tlw = 0.5 * z / sigw / (1.0 + 1.5e-3 * corr);
    } else if (ol < 0.0) {
        let sigw_sq = 1.2 * wst * wst * (1.0 - 0.9 * zeta) * pow(zeta, 2.0 / 3.0)
            + (1.8 - 1.4 * zeta) * ust * ust;
        sigw = sqrt(sigw_sq) + SIGMA_FLOOR;
        dsigwdz = 0.5 / sigw / h
            * (-1.4 * ust * ust
                + wst * wst * (0.8 * pow(max(zeta, 1.0e-3), -1.0 / 3.0)
                    - 1.8 * pow(zeta, 2.0 / 3.0)));
        if (z < abs(ol)) {
            tlw = 0.1 * z / (sigw * (0.55 - 0.38 * abs(z / ol)));
        } else if (zeta < 0.1) {
            tlw = 0.59 * z / sigw;
        } else {
            tlw = 0.15 * h / sigw * (1.0 - exp(-5.0 * zeta));
        }
    } else {
        sigw = SIGMA_FLOOR + 1.3 * ust * (1.0 - zeta);
        dsigwdz = -1.3 * ust / h;
        tlw = 0.1 * h / sigw * pow(zeta, 0.8);
    }

    tlw = max(tlw, TLW_MIN);
    if (dsigwdz == 0.0) { dsigwdz = 1.0e-10; }
    if (!is_finite_scalar(sigw) || sigw <= 0.0) { sigw = SIGMA_FLOOR; }
    if (!is_finite_scalar(tlw)) { tlw = TLW_MIN; }
    if (!is_finite_scalar(dsigwdz)) { dsigwdz = 1.0e-10; }

    *out_sigw = sigw;
    *out_dsigwdz = dsigwdz;
    *out_tlw = tlw;
}

// ---------------------------------------------------------------------------
// Philox-4x32-10 RNG (identical to langevin.wgsl)
// ---------------------------------------------------------------------------

fn raise_key(key: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(key.x + PHILOX_W0, key.y + PHILOX_W1);
}

fn mul_hi_lo_u32(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;

    let p0 = a0 * b0;
    let p1 = a0 * b1;
    let p2 = a1 * b0;
    let p3 = a1 * b1;

    let p0_hi = p0 >> 16u;
    let tmp = p1 + p0_hi;
    let carry1 = select(0u, 1u, tmp < p1);
    let mid = tmp + p2;
    let carry2 = select(0u, 1u, mid < tmp);
    let carry = carry1 + carry2;

    let lo = ((mid & 0xFFFFu) << 16u) | (p0 & 0xFFFFu);
    let hi = p3 + (mid >> 16u) + (carry << 16u);
    return vec2<u32>(hi, lo);
}

fn philox_round(counter: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    let product0 = mul_hi_lo_u32(PHILOX_M0, counter.x);
    let product1 = mul_hi_lo_u32(PHILOX_M1, counter.z);
    return vec4<u32>(
        product1.x ^ counter.y ^ key.x,
        product1.y,
        product0.x ^ counter.w ^ key.y,
        product0.y
    );
}

fn philox4x32_10(counter: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    var c = counter;
    var k = key;
    for (var round = 0u; round < PHILOX_ROUNDS; round = round + 1u) {
        c = philox_round(c, k);
        k = raise_key(k);
    }
    return c;
}

fn counter_add(base: vec4<u32>, offset: u32) -> vec4<u32> {
    let c0 = base.x + offset;
    let carry0 = select(0u, 1u, c0 < base.x);
    let c1 = base.y + carry0;
    let carry1 = select(0u, 1u, (carry0 == 1u) && (c1 < base.y));
    let c2 = base.z + carry1;
    let carry2 = select(0u, 1u, (carry1 == 1u) && (c2 < base.z));
    let c3 = base.w + carry2;
    return vec4<u32>(c0, c1, c2, c3);
}

fn u32_to_uniform01(value: u32) -> f32 {
    return f32(value >> 8u) * U32_TO_UNIT_SCALE_24BIT;
}

fn box_muller_normals(u0: f32, u1: f32) -> vec2<f32> {
    let radius_uniform = clamp(u0, BOX_MULLER_U_MIN, BOX_MULLER_U_MAX);
    let angle_uniform = clamp(u1, 0.0, BOX_MULLER_U_MAX);
    let radius = sqrt(-2.0 * log(radius_uniform));
    let theta = TWO_PI * angle_uniform;
    return vec2<f32>(radius * cos(theta), radius * sin(theta));
}

// ---------------------------------------------------------------------------
// Langevin velocity update (identical to langevin.wgsl)
// ---------------------------------------------------------------------------

fn update_horizontal_component(previous: f32, sigma_in: f32, tl_in: f32, dt: f32, eta: f32) -> f32 {
    let sigma = sanitize_positive(sigma_in);
    let tl = sanitize_positive(tl_in);
    if (tl == 0.0) {
        return 0.0;
    }

    let ratio = dt / tl;
    var next: f32;
    if (ratio < 0.5) {
        next = (1.0 - ratio) * previous + eta * sigma * sqrt(max(2.0 * ratio, 0.0));
    } else {
        let corr = exp(-ratio);
        next = corr * previous + eta * sigma * sqrt(max(1.0 - corr * corr, 0.0));
    }
    if (is_finite_scalar(next)) {
        return next;
    }
    return 0.0;
}

fn update_vertical_component(
    previous: f32,
    hanna: HannaParams,
    dt: f32,
    rho_grad_over_rho: f32,
    eta: f32
) -> f32 {
    let sigma_w = sanitize_positive(hanna.sigw);
    let tlw = sanitize_positive(hanna.tlw);
    if (tlw == 0.0) {
        return 0.0;
    }

    let drift = sigma_w * (hanna.dsigwdz + rho_grad_over_rho * sigma_w);
    let ratio = dt / tlw;
    var next: f32;
    if (ratio < 0.5) {
        next = (1.0 - ratio) * previous + eta * sigma_w * sqrt(max(2.0 * ratio, 0.0)) + dt * drift;
    } else {
        let corr = exp(-ratio);
        next = corr * previous
            + eta * sigma_w * sqrt(max(1.0 - corr * corr, 0.0))
            + tlw * (1.0 - corr) * drift;
    }

    if (is_finite_scalar(next)) {
        return next;
    }
    return 0.0;
}

fn select_eta_w(sub: u32, g0: f32, g1: f32, g2: f32, g3: f32) -> f32 {
    if (sub == 0u) { return g0; }
    if (sub == 1u) { return g1; }
    if (sub == 2u) { return g2; }
    return g3;
}

// ---------------------------------------------------------------------------
// Deposition (from dry_deposition.wgsl + wet_deposition.wgsl)
// ---------------------------------------------------------------------------

fn compute_survival_factor(vdep_m_s: f32, dt_seconds: f32, href_m: f32) -> f32 {
    let href = max(href_m, HREF_MIN_M);
    let exponent = -max(vdep_m_s, 0.0) * abs(dt_seconds) / (2.0 * href);
    return clamp(exp(exponent), 0.0, 1.0);
}

fn wet_probability(lambda_s_inv: f32, dt_seconds: f32, precip_fraction: f32) -> f32 {
    let lambda = max(lambda_s_inv, 0.0);
    let dt = abs(dt_seconds);
    let fraction = clamp(precip_fraction, 0.0, 1.0);
    if (lambda <= 0.0 || dt <= 0.0 || fraction <= 0.0) {
        return 0.0;
    }
    let local_probability = 1.0 - exp(-lambda * dt);
    return clamp(local_probability * fraction, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Main: mega-kernel entry point
// ---------------------------------------------------------------------------

@compute @workgroup_size(__WORKGROUP_SIZE_X__)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let particle_id = gid.y * (nwg.x * __WORKGROUP_SIZE_X__u) + gid.x;
    if (particle_id >= params.particle_count) {
        return;
    }

    var particle = particles[particle_id];
    if ((particle.flags & FLAG_ACTIVE) == 0u) {
        return;
    }

    // ===================================================================
    // Stage 1: Advection — Petterssen predictor-corrector with dual wind
    // ===================================================================

    let adv_x0 = f32(particle.cell_x) + particle.pos_x;
    let adv_y0 = f32(particle.cell_y) + particle.pos_y;
    let adv_z0 = particle.pos_z;

    let u0 = sample_dual_u(adv_x0, adv_y0, adv_z0) + particle.turb_u;
    let v0 = sample_dual_v(adv_x0, adv_y0, adv_z0) + particle.turb_v;
    let w0 = sample_dual_w(adv_x0, adv_y0, adv_z0);

    let predicted_x = clamp(adv_x0 + (u0 * params.x_scale) * params.dt_seconds, 0.0, f32(params.nx - 1u));
    let predicted_y = clamp(adv_y0 + (v0 * params.y_scale) * params.dt_seconds, 0.0, f32(params.ny - 1u));
    let predicted_z = clamp_vertical_coord(adv_z0 + (w0 * params.z_scale) * params.dt_seconds);

    let u1 = sample_dual_u(predicted_x, predicted_y, predicted_z) + particle.turb_u;
    let v1 = sample_dual_v(predicted_x, predicted_y, predicted_z) + particle.turb_v;
    let w1 = sample_dual_w(predicted_x, predicted_y, predicted_z);

    let corrected_u = 0.5 * (u0 + u1);
    let corrected_v = 0.5 * (v0 + v1);
    let corrected_w = 0.5 * (w0 + w1);

    let final_x = clamp(adv_x0 + (corrected_u * params.x_scale) * params.dt_seconds, 0.0, f32(params.nx - 1u));
    let final_y = clamp(adv_y0 + (corrected_v * params.y_scale) * params.dt_seconds, 0.0, f32(params.ny - 1u));
    let final_z = clamp_vertical_coord(adv_z0 + (corrected_w * params.z_scale) * params.dt_seconds);

    let x_split = split_horizontal_coord(final_x, params.nx);
    let y_split = split_horizontal_coord(final_y, params.ny);

    particle.cell_x = i32(x_split.x);
    particle.cell_y = i32(y_split.x);
    particle.pos_x = x_split.y;
    particle.pos_y = y_split.y;
    particle.pos_z = final_z;
    particle.vel_u = corrected_u;
    particle.vel_v = corrected_v;
    particle.vel_w = corrected_w;

    // ===================================================================
    // Stage 2: Hanna PBL turbulence parameters at post-advection position
    // ===================================================================

    let hanna_x = f32(particle.cell_x) + particle.pos_x;
    let hanna_y = f32(particle.cell_y) + particle.pos_y;

    let pi = u32(floor(clamp(hanna_x, 0.0, f32(params.pbl_nx - 1u))));
    let pj = u32(floor(clamp(hanna_y, 0.0, f32(params.pbl_ny - 1u))));
    let pbl_idx = pbl_flat_index(pi, pj);
    let pbl_size = params.pbl_nx * params.pbl_ny;

    let ust = packed_pbl[pbl_idx];
    let wst = packed_pbl[pbl_size + pbl_idx];
    let h_pbl = packed_pbl[2u * pbl_size + pbl_idx];
    let ol = obukhov_length_from_inverse(packed_pbl[3u * pbl_size + pbl_idx]);

    var hanna = compute_hanna(ust, wst, ol, h_pbl, particle.pos_z);

    // ===================================================================
    // Stage 3: Langevin turbulent velocity update (with sub-stepping)
    // ===================================================================

    let key = vec2<u32>(params.key0, params.key1);
    let base_counter = vec4<u32>(
        params.counter0, params.counter1,
        params.counter2, params.counter3
    );

    let n_vert = max(params.n_substeps, 1u);
    let blocks_needed = (2u + n_vert + 3u) / 4u;

    let block0 = philox4x32_10(
        counter_add(base_counter, particle_id * blocks_needed), key);
    var block1 = vec4<u32>(0u, 0u, 0u, 0u);
    if (blocks_needed > 1u) {
        block1 = philox4x32_10(
            counter_add(base_counter, particle_id * blocks_needed + 1u), key);
    }

    let eta_uv = box_muller_normals(
        u32_to_uniform01(block0.x), u32_to_uniform01(block0.y));
    let eta_w01 = box_muller_normals(
        u32_to_uniform01(block0.z), u32_to_uniform01(block0.w));
    let eta_w23 = box_muller_normals(
        u32_to_uniform01(block1.x), u32_to_uniform01(block1.y));

    // Horizontal turbulence: single full-dt update
    particle.turb_u = update_horizontal_component(
        particle.turb_u, hanna.sigu, hanna.tlu,
        params.dt_seconds, eta_uv.x);
    particle.turb_v = update_horizontal_component(
        particle.turb_v, hanna.sigv, hanna.tlv,
        params.dt_seconds, eta_uv.y);

    // Vertical turbulence with sub-stepping and PBL reflection
    if (params.n_substeps == 0u) {
        particle.turb_w = update_vertical_component(
            particle.turb_w, hanna, params.dt_seconds,
            params.rho_grad_over_rho, eta_w01.x);
    } else {
        let dt_sub = params.dt_seconds / f32(params.n_substeps);
        let h = hanna.h;
        let z_min = params.min_height_m;

        for (var sub = 0u; sub < params.n_substeps; sub = sub + 1u) {
            let eta = select_eta_w(sub,
                eta_w01.x, eta_w01.y, eta_w23.x, eta_w23.y);

            particle.turb_w = update_vertical_component(
                particle.turb_w, hanna, dt_sub,
                params.rho_grad_over_rho, eta);

            particle.pos_z = particle.pos_z + particle.turb_w * dt_sub;

            // PBL reflection (Thomson 1987 / advance.f90)
            if (h > z_min) {
                var z_refl = particle.pos_z;
                var cbt = particle.cbt;
                for (var bounce = 0; bounce < 10; bounce = bounce + 1) {
                    if (z_refl < z_min) {
                        z_refl = 2.0 * z_min - z_refl;
                        cbt = -cbt;
                        particle.turb_w = -particle.turb_w;
                    } else if (z_refl > h) {
                        z_refl = 2.0 * h - z_refl;
                        cbt = -cbt;
                        particle.turb_w = -particle.turb_w;
                    } else {
                        break;
                    }
                }
                z_refl = clamp(z_refl, z_min, h);
                particle.pos_z = z_refl;
                particle.cbt = cbt;
            }

            // Recalculate vertical Hanna at new height (Fortran hanna_short)
            if (sub < params.n_substeps - 1u) {
                var new_sigw: f32;
                var new_dsigwdz: f32;
                var new_tlw: f32;
                hanna_short_w(hanna.ust, hanna.wst, hanna.ol, hanna.h, particle.pos_z,
                              &new_sigw, &new_dsigwdz, &new_tlw);
                hanna.sigw = new_sigw;
                hanna.dsigwdz = new_dsigwdz;
                hanna.dsigw2dz = 2.0 * new_sigw * new_dsigwdz;
                hanna.tlw = new_tlw;
                hanna.zeta = clamp(particle.pos_z / hanna.h, 0.0, 1.0);
            }
        }
    }

    // ===================================================================
    // Stage 4: Dry deposition (ported from advance.f90 / get_vdep_prob.f90)
    // ===================================================================

    let href = max(params.reference_height_m, HREF_MIN_M);
    if (particle.pos_z < 2.0 * href) {
        let vdep = packed_deposition[particle_id];
        let dry_survival = compute_survival_factor(vdep, params.dt_seconds, href);
        particle.mass[0] = particle.mass[0] * dry_survival;
        particle.mass[1] = particle.mass[1] * dry_survival;
        particle.mass[2] = particle.mass[2] * dry_survival;
        particle.mass[3] = particle.mass[3] * dry_survival;
    }

    // ===================================================================
    // Stage 5: Wet deposition (ported from wetdepo.f90)
    // ===================================================================

    let scav_coeff = packed_deposition[params.particle_count + particle_id];
    let precip_frac = packed_deposition[2u * params.particle_count + particle_id];
    let wet_prob = wet_probability(scav_coeff, params.dt_seconds, precip_frac);
    if (wet_prob > 0.0) {
        let wet_survival = 1.0 - wet_prob;
        particle.mass[0] = particle.mass[0] * wet_survival;
        particle.mass[1] = particle.mass[1] * wet_survival;
        particle.mass[2] = particle.mass[2] * wet_survival;
        particle.mass[3] = particle.mass[3] * wet_survival;
    }

    // ===================================================================
    // Store final particle state
    // ===================================================================

    particles[particle_id] = particle;
}
