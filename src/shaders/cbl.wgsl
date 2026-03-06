// WGSL CBL vertical-velocity sampling kernel (H-06).
//
// Ported from FLEXPART CBL logic (`cbl.f90`) through the Rust CPU reference in
// `src/physics/cbl.rs` (H-05). For each active particle:
// - reconstruct CBL bi-Gaussian PDF from local (z, h, w*, sigma_w, L)
// - sample one vertical turbulent velocity perturbation w'
//
// Inputs:
//   binding(0): particles storage buffer (Particle[particle_count])
//   binding(1): HannaParams storage buffer (HannaParams[particle_count])
//   binding(2): sampling uniforms (CblSamplingInput[particle_count])
//   binding(3): output CBL sampling records (CblSamplingOutput[particle_count])
//   binding(4): dispatch params (particle_count + padding)

const FLAG_ACTIVE: u32 = 1u;

const PI: f32 = 3.1415927;
const TWO_PI: f32 = 6.2831855;
const BOX_MULLER_U_MIN: f32 = 1.1754944e-38;
const BOX_MULLER_U_MAX: f32 = 0.99999994;

const FLUARW_FACTOR: f32 = 0.6666667; // 2/3
const MIN_H_M: f32 = 1.0;
const MIN_SIGMA_W_M_S: f32 = 1.0e-5;
const SKEW_EPS: f32 = 1.0e-6;
const W3_EPS: f32 = 1.0e-6;
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

struct CblSamplingInput {
    branch_uniform: f32,
    gaussian_uniform0: f32,
    gaussian_uniform1: f32,
    pad0: f32,
};

struct CblSamplingOutput {
    sampled_w_m_s: f32,
    updraft_weight: f32,
    updraft_mean_m_s: f32,
    updraft_sigma_m_s: f32,
    downdraft_weight: f32,
    downdraft_mean_m_s: f32,
    downdraft_sigma_m_s: f32,
    second_moment_w2: f32,
    third_moment_w3: f32,
    skewness: f32,
    transition: f32,
    z_over_h: f32,
};

struct CblDispatchParams {
    particle_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct CblPdf {
    updraft_weight: f32,
    updraft_mean_m_s: f32,
    updraft_sigma_m_s: f32,
    downdraft_weight: f32,
    downdraft_mean_m_s: f32,
    downdraft_sigma_m_s: f32,
    second_moment_w2: f32,
    third_moment_w3: f32,
    skewness: f32,
    transition: f32,
    z_over_h: f32,
};

@group(0) @binding(0)
var<storage, read> particles: array<Particle>;

@group(0) @binding(1)
var<storage, read> hanna_params: array<HannaParams>;

@group(0) @binding(2)
var<storage, read> sampling_inputs: array<CblSamplingInput>;

@group(0) @binding(3)
var<storage, read_write> outputs: array<CblSamplingOutput>;

@group(0) @binding(4)
var<uniform> params: CblDispatchParams;

fn is_finite_scalar(value: f32) -> bool {
    return (value == value) && (abs(value) <= F32_MAX_FINITE);
}

fn cbrt_signed(value: f32) -> f32 {
    if (value >= 0.0) {
        return pow(value, 1.0 / 3.0);
    }
    return -pow(-value, 1.0 / 3.0);
}

fn cbl_transition_factor(h_m: f32, obukhov_length_m: f32) -> f32 {
    if (!is_finite_scalar(h_m) || h_m <= 0.0 || !is_finite_scalar(obukhov_length_m) || obukhov_length_m == 0.0) {
        return 1.0;
    }

    let ratio = -h_m / obukhov_length_m;
    var transition: f32;
    if (ratio < 15.0) {
        transition = sin(((ratio + 10.0) / 10.0) * PI) * 0.5 + 0.5;
    } else {
        transition = 1.0;
    }
    return clamp(transition, 0.0, 1.0);
}

fn box_muller_normals(u0: f32, u1: f32) -> vec2<f32> {
    let radius_uniform = clamp(u0, BOX_MULLER_U_MIN, BOX_MULLER_U_MAX);
    let angle_uniform = clamp(u1, 0.0, BOX_MULLER_U_MAX);
    let radius = sqrt(-2.0 * log(radius_uniform));
    let theta = TWO_PI * angle_uniform;
    return vec2<f32>(radius * cos(theta), radius * sin(theta));
}

fn symmetric_pdf(
    sigma_w_m_s: f32,
    second_moment_w2: f32,
    third_moment_w3: f32,
    skewness: f32,
    transition: f32,
    z_over_h: f32
) -> CblPdf {
    return CblPdf(
        0.5,
        0.0,
        sigma_w_m_s,
        0.5,
        0.0,
        sigma_w_m_s,
        second_moment_w2,
        third_moment_w3,
        skewness,
        transition,
        z_over_h
    );
}

fn compute_cbl_pdf(particle_z_m: f32, hanna: HannaParams) -> CblPdf {
    let h_m = max(hanna.h, MIN_H_M);
    let z_m = clamp(particle_z_m, 0.0, h_m);
    let z_over_h = z_m / h_m;
    let wstar_m_s = max(hanna.wst, 0.0);
    let sigma_w_m_s = max(hanna.sigw, MIN_SIGMA_W_M_S);
    let transition = cbl_transition_factor(h_m, hanna.ol);

    let second_moment_w2 = sigma_w_m_s * sigma_w_m_s;
    let shape = 1.2 * z_over_h * pow(1.0 - z_over_h, 1.5) + W3_EPS;
    let third_moment_w3 = shape * pow(wstar_m_s, 3.0) * transition;
    let skewness = third_moment_w3 / pow(second_moment_w2, 1.5);

    if (!is_finite_scalar(skewness) || abs(skewness) <= SKEW_EPS) {
        return symmetric_pdf(
            sigma_w_m_s,
            second_moment_w2,
            third_moment_w3,
            skewness,
            transition,
            z_over_h
        );
    }

    let fluarw = FLUARW_FACTOR * cbrt_signed(skewness);
    let fluarw2 = fluarw * fluarw;
    if (!is_finite_scalar(fluarw) || abs(fluarw) <= SKEW_EPS) {
        return symmetric_pdf(
            sigma_w_m_s,
            second_moment_w2,
            third_moment_w3,
            skewness,
            transition,
            z_over_h
        );
    }

    let skew2 = skewness * skewness;
    let numerator_r = pow(1.0 + fluarw2, 3.0) * skew2;
    let denominator_r = pow(3.0 + fluarw2, 2.0) * fluarw2;
    let rluarw = max(numerator_r / max(denominator_r, SKEW_EPS), 0.0);
    let xluarw = sqrt(rluarw);

    var aluarw = 0.5 * (1.0 - xluarw / sqrt(4.0 + rluarw));
    aluarw = clamp(aluarw, 1.0e-4, 1.0 - 1.0e-4);
    let bluarw = 1.0 - aluarw;

    let sigmawa = sigma_w_m_s * sqrt(bluarw / (aluarw * (1.0 + fluarw2)));
    let sigmawb = sigma_w_m_s * sqrt(aluarw / (bluarw * (1.0 + fluarw2)));
    let wa = fluarw * sigmawa;
    let wb = fluarw * sigmawb;

    return CblPdf(
        aluarw,
        wa,
        max(sigmawa, MIN_SIGMA_W_M_S),
        bluarw,
        -wb,
        max(sigmawb, MIN_SIGMA_W_M_S),
        second_moment_w2,
        third_moment_w3,
        skewness,
        transition,
        z_over_h
    );
}

fn zero_output() -> CblSamplingOutput {
    return CblSamplingOutput(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let particle_id = gid.y * (nwg.x * 64u) + gid.x;
    if (particle_id >= params.particle_count) {
        return;
    }

    let particle = particles[particle_id];
    if ((particle.flags & FLAG_ACTIVE) == 0u) {
        outputs[particle_id] = zero_output();
        return;
    }

    let hanna = hanna_params[particle_id];
    let sample_input = sampling_inputs[particle_id];
    let pdf = compute_cbl_pdf(particle.pos_z, hanna);

    let eta = box_muller_normals(sample_input.gaussian_uniform0, sample_input.gaussian_uniform1).x;
    var sampled_w_m_s: f32;
    if (sample_input.branch_uniform < pdf.updraft_weight) {
        sampled_w_m_s = pdf.updraft_mean_m_s + pdf.updraft_sigma_m_s * eta;
    } else {
        sampled_w_m_s = pdf.downdraft_mean_m_s + pdf.downdraft_sigma_m_s * eta;
    }

    outputs[particle_id] = CblSamplingOutput(
        sampled_w_m_s,
        pdf.updraft_weight,
        pdf.updraft_mean_m_s,
        pdf.updraft_sigma_m_s,
        pdf.downdraft_weight,
        pdf.downdraft_mean_m_s,
        pdf.downdraft_sigma_m_s,
        pdf.second_moment_w2,
        pdf.third_moment_w3,
        pdf.skewness,
        pdf.transition,
        pdf.z_over_h
    );
}
