// WGSL Hanna turbulence parameter kernel (H-02).
//
// Ported from FLEXPART `hanna.f90` via the Rust CPU reference in
// `src/physics/hanna.rs` (H-01). For each active particle we compute:
//   - sigma_u, sigma_v, sigma_w
//   - d(sigma_w)/dz and d(sigma_w^2)/dz
//   - TL_u, TL_v, TL_w
//
// Inputs:
//   binding(0): particles storage buffer (Particle[particle_count])
//   binding(1): pbl_ustar flattened (nx * ny)
//   binding(2): pbl_wstar flattened (nx * ny)
//   binding(3): pbl_hmix flattened (nx * ny)
//   binding(4): pbl_oli flattened (nx * ny), where oli = 1/L
//   binding(5): output HannaParams storage buffer (HannaParams[particle_count])
//   binding(6): uniform params (pbl_nx, pbl_ny, particle_count, pad)
//
// Flattening contract for PBL fields (ndarray row-major):
//   flat_index = i * ny + j

const FLAG_ACTIVE: u32 = 1u;

const USTAR_MIN: f32 = 1.0e-4;
const TLU_MIN: f32 = 10.0;
const TLV_MIN: f32 = 10.0;
const TLW_MIN: f32 = 30.0;
const SIGMA_FLOOR: f32 = 1.0e-2;
const H_MIN: f32 = 1.0;
const NEUTRAL_OLI_THRESHOLD: f32 = 1.0e-5;
const F32_MAX_FINITE: f32 = 3.4028235e38;

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

struct HannaDispatchParams {
    pbl_nx: u32,
    pbl_ny: u32,
    particle_count: u32,
    _pad0: u32,
};

@group(0) @binding(0)
var<storage, read> particles: array<Particle>;

@group(0) @binding(1)
var<storage, read> pbl_ustar: array<f32>;

@group(0) @binding(2)
var<storage, read> pbl_wstar: array<f32>;

@group(0) @binding(3)
var<storage, read> pbl_hmix: array<f32>;

@group(0) @binding(4)
var<storage, read> pbl_oli: array<f32>;

@group(0) @binding(5)
var<storage, read_write> outputs: array<HannaParams>;

@group(0) @binding(6)
var<uniform> params: HannaDispatchParams;

fn pbl_flat_index(i: u32, j: u32) -> u32 {
    return i * params.pbl_ny + j;
}

fn is_finite_scalar(value: f32) -> bool {
    return (value == value) && (abs(value) <= F32_MAX_FINITE);
}

fn obukhov_length_from_inverse(oli: f32) -> f32 {
    if (abs(oli) < NEUTRAL_OLI_THRESHOLD) {
        // WGSL/Naga rejects `inf` as a compile-time constant on some toolchains.
        // Materialize +inf from IEEE-754 bits at runtime.
        return bitcast<f32>(0x7f800000u);
    }
    return 1.0 / oli;
}

fn zero_hanna_params() -> HannaParams {
    return HannaParams(
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    );
}

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

    // Branch order matches the CPU reference and Fortran:
    // 1) neutral if h/|L| < 1
    // 2) unstable if L < 0
    // 3) stable otherwise
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

    if (!is_finite_scalar(sigu) || sigu <= 0.0) {
        sigu = SIGMA_FLOOR;
    }
    if (!is_finite_scalar(sigv) || sigv <= 0.0) {
        sigv = SIGMA_FLOOR;
    }
    if (!is_finite_scalar(sigw) || sigw <= 0.0) {
        sigw = SIGMA_FLOOR;
    }
    if (!is_finite_scalar(tlu)) {
        tlu = TLU_MIN;
    }
    if (!is_finite_scalar(tlv)) {
        tlv = TLV_MIN;
    }
    if (!is_finite_scalar(tlw)) {
        tlw = TLW_MIN;
    }
    if (!is_finite_scalar(dsigwdz)) {
        dsigwdz = 1.0e-10;
    }

    let dsigw2dz = 2.0 * sigw * dsigwdz;
    return HannaParams(
        ust, wst, ol, h,
        zeta, sigu, sigv, sigw,
        dsigwdz, dsigw2dz, tlu, tlv,
        tlw, 0.0, 0.0, 0.0
    );
}

@compute @workgroup_size(__WORKGROUP_SIZE_X__)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let particle_id = gid.y * (nwg.x * __WORKGROUP_SIZE_X__u) + gid.x;
    if (particle_id >= params.particle_count) {
        return;
    }

    let particle = particles[particle_id];
    if ((particle.flags & FLAG_ACTIVE) == 0u) {
        outputs[particle_id] = zero_hanna_params();
        return;
    }

    let x = f32(particle.cell_x) + particle.pos_x;
    let y = f32(particle.cell_y) + particle.pos_y;
    let z = particle.pos_z;

    let i = u32(floor(clamp(x, 0.0, f32(params.pbl_nx - 1u))));
    let j = u32(floor(clamp(y, 0.0, f32(params.pbl_ny - 1u))));
    let pbl_idx = pbl_flat_index(i, j);

    let ust = pbl_ustar[pbl_idx];
    let wst = pbl_wstar[pbl_idx];
    let h = pbl_hmix[pbl_idx];
    let ol = obukhov_length_from_inverse(pbl_oli[pbl_idx]);

    outputs[particle_id] = compute_hanna(ust, wst, ol, h, z);
}
