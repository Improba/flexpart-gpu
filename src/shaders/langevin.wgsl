// WGSL Langevin turbulence update kernel (H-04).
//
// Ported from FLEXPART `advance.f90` turbulence branch via the Rust CPU
// reference in `src/physics/langevin.rs` (H-03).
//
// For each particle slot:
// 1) Generate deterministic Philox4x32-10 uniforms from (key, base_counter+id)
// 2) Convert uniforms to Gaussian forcing (Box-Muller)
// 3) Update turbulent velocity memory `turb_u/turb_v/turb_w`
//
// When n_substeps >= 1, vertical turbulence is sub-stepped (Fortran `ifine`):
//   for each sub-step: update turb_w with dt_sub, apply vertical displacement
//   (pos_z += turb_w * dt_sub), and PBL reflection at z=0 and z=hmix.
// When n_substeps == 0, turb_w is updated without displacement (legacy mode).
//
// The mapping uses ceil((2 + max(n_substeps,1)) / 4) Philox blocks per slot.
// For n_substeps <= 2 this is 1 block (same as legacy).

const FLAG_ACTIVE: u32 = 1u;

// Hanna constants for hanna_short_w (vertical param recalculation)
const USTAR_MIN: f32 = 1.0e-4;
const TLW_MIN: f32 = 30.0;
const SIGMA_FLOOR: f32 = 1.0e-2;
const H_MIN: f32 = 1.0;
const NEUTRAL_OLI_THRESHOLD: f32 = 1.0e-5;

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

struct LangevinDispatchParams {
    key0: u32,
    key1: u32,
    counter0: u32,
    counter1: u32,
    counter2: u32,
    counter3: u32,
    particle_count: u32,
    n_substeps: u32,
    dt_seconds: f32,
    rho_grad_over_rho: f32,
    min_height_m: f32,
    _pad0: u32,
};

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<storage, read> hanna_params: array<HannaParams>;

@group(0) @binding(2)
var<uniform> params: LangevinDispatchParams;

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

fn is_finite_scalar(value: f32) -> bool {
    return (value == value) && (abs(value) <= F32_MAX_FINITE);
}

fn sanitize_positive(value: f32) -> f32 {
    if (is_finite_scalar(value) && value > 0.0) {
        return value;
    }
    return 0.0;
}

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

// Recalculate vertical Hanna turbulence parameters at a new height.
// Mirrors Fortran hanna_short(z): only updates sigw, dsigwdz, tlw.
// PBL fields (ust, wst, ol, h) come from the initial Hanna computation.
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
        // Neutral
        let corr = z / ust;
        sigw = 1.3 * ust * exp(-2.0e-4 * corr);
        dsigwdz = -2.0e-4 * sigw;
        sigw = sigw + SIGMA_FLOOR;
        tlw = 0.5 * z / sigw / (1.0 + 1.5e-3 * corr);
    } else if (ol < 0.0) {
        // Unstable
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
        // Stable
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

// Select the vertical Gaussian forcing for a given sub-step index.
fn select_eta_w(sub: u32, g0: f32, g1: f32, g2: f32, g3: f32) -> f32 {
    if (sub == 0u) { return g0; }
    if (sub == 1u) { return g1; }
    if (sub == 2u) { return g2; }
    return g3;
}

@compute @workgroup_size(__WORKGROUP_SIZE_X__)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let particle_id = gid.y * (nwg.x * __WORKGROUP_SIZE_X__u) + gid.x;
    if (particle_id >= params.particle_count) {
        return;
    }

    let key = vec2<u32>(params.key0, params.key1);
    let base_counter = vec4<u32>(
        params.counter0,
        params.counter1,
        params.counter2,
        params.counter3
    );

    // Determine Philox blocks needed per particle:
    // 2 Gaussians (horizontal) + max(n_substeps, 1) Gaussians (vertical)
    // Each Philox block → 4 u32 → 2 Box-Muller pairs → 4 Gaussians.
    let n_vert = max(params.n_substeps, 1u);
    let blocks_needed = (2u + n_vert + 3u) / 4u;

    let block0 = philox4x32_10(
        counter_add(base_counter, particle_id * blocks_needed), key);
    var block1 = vec4<u32>(0u, 0u, 0u, 0u);
    if (blocks_needed > 1u) {
        block1 = philox4x32_10(
            counter_add(base_counter, particle_id * blocks_needed + 1u), key);
    }

    var particle = particles[particle_id];
    if ((particle.flags & FLAG_ACTIVE) == 0u) {
        return;
    }

    // Gaussian forcing from block 0: horizontal pair + first vertical pair
    let eta_uv = box_muller_normals(
        u32_to_uniform01(block0.x), u32_to_uniform01(block0.y));
    let eta_w01 = box_muller_normals(
        u32_to_uniform01(block0.z), u32_to_uniform01(block0.w));
    // Additional vertical Gaussians from block 1 (for n_substeps > 2)
    let eta_w23 = box_muller_normals(
        u32_to_uniform01(block1.x), u32_to_uniform01(block1.y));

    var hanna = hanna_params[particle_id];

    // --- Horizontal turbulence: one full-dt update (matching Fortran) ---
    particle.turb_u = update_horizontal_component(
        particle.turb_u, hanna.sigu, hanna.tlu,
        params.dt_seconds, eta_uv.x);
    particle.turb_v = update_horizontal_component(
        particle.turb_v, hanna.sigv, hanna.tlv,
        params.dt_seconds, eta_uv.y);

    // --- Vertical turbulence ---
    if (params.n_substeps == 0u) {
        // Legacy mode: update turb_w without displacement or PBL reflection.
        particle.turb_w = update_vertical_component(
            particle.turb_w, hanna, params.dt_seconds,
            params.rho_grad_over_rho, eta_w01.x);
    } else {
        // Sub-stepped vertical diffusion (Fortran ifine).
        let dt_sub = params.dt_seconds / f32(params.n_substeps);
        let h = hanna.h;
        let z_min = params.min_height_m;

        for (var sub = 0u; sub < params.n_substeps; sub = sub + 1u) {
            let eta = select_eta_w(sub,
                eta_w01.x, eta_w01.y, eta_w23.x, eta_w23.y);

            // Langevin vertical update with dt_sub
            particle.turb_w = update_vertical_component(
                particle.turb_w, hanna, dt_sub,
                params.rho_grad_over_rho, eta);

            // Apply vertical displacement
            particle.pos_z = particle.pos_z + particle.turb_w * dt_sub;

            // PBL reflection (Thomson 1987 / advance.f90)
            if (h > z_min) {
                var z = particle.pos_z;
                var cbt = particle.cbt;
                for (var bounce = 0; bounce < 10; bounce = bounce + 1) {
                    if (z < z_min) {
                        z = 2.0 * z_min - z;
                        cbt = -cbt;
                        particle.turb_w = -particle.turb_w;
                    } else if (z > h) {
                        z = 2.0 * h - z;
                        cbt = -cbt;
                        particle.turb_w = -particle.turb_w;
                    } else {
                        break;
                    }
                }
                z = clamp(z, z_min, h);
                particle.pos_z = z;
                particle.cbt = cbt;
            }

            // Recalculate vertical Hanna params at new height (Fortran hanna_short)
            // Skipped after the last sub-step, matching: if (i.ne.ifine) call hanna_short(zt)
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

    particles[particle_id] = particle;
}
