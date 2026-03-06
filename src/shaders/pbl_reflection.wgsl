// PBL boundary reflection kernel.
//
// Implements reflecting boundaries at z=0 (ground) and z=h (mixing height)
// following FLEXPART Fortran's advance.f90 (Thomson 1987 well-mixed criterion).
//
// Runs after advection, before the next Hanna/Langevin update.
// Uses the previous step's Hanna params for per-particle mixing height h.

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

struct ReflectionParams {
    particle_count: u32,
    min_height_m: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> hanna_params: array<HannaParams>;
@group(0) @binding(2) var<uniform> params: ReflectionParams;

const FLAG_ACTIVE: u32 = 1u;

fn is_active(p: Particle) -> bool {
    return (p.flags & FLAG_ACTIVE) != 0u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.particle_count {
        return;
    }

    var p = particles[idx];
    if !is_active(p) {
        return;
    }

    let h = hanna_params[idx].h;
    let z_min = params.min_height_m;

    // Skip reflection if mixing height is not meaningful
    if h <= z_min {
        particles[idx] = p;
        return;
    }

    var z = p.pos_z;
    var cbt = p.cbt;

    // Apply reflecting boundaries (up to 10 reflections to handle rare
    // multi-bounce scenarios with large displacement).
    for (var bounce = 0; bounce < 10; bounce++) {
        if z < z_min {
            z = 2.0 * z_min - z;
            cbt = -cbt;
            p.turb_w = -p.turb_w;
        } else if z > h {
            z = 2.0 * h - z;
            cbt = -cbt;
            p.turb_w = -p.turb_w;
        } else {
            break;
        }
    }

    // Final safety clamp (should not be needed after reflections,
    // but prevents NaN propagation).
    z = clamp(z, z_min, h);

    p.pos_z = z;
    p.cbt = cbt;
    particles[idx] = p;
}
