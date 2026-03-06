// WGSL concentration gridding kernel (I-02).
//
// Ported from FLEXPART `conccalc.f90` particle-to-grid accumulation pattern.
//
// MVP contract:
// - each active particle contributes one count to its host cell;
// - selected species mass is atomically accumulated after quantization:
//     mass_scaled = round(max(mass_kg, 0) * mass_scale)
// - host converts back to kg by dividing summed `mass_scaled / mass_scale`.
//
// Buffer contract:
// - binding(0): particles (read)
// - binding(1): particle_count_grid as atomic u32 (read_write)
// - binding(2): concentration_mass_scaled as atomic u32 (read_write)
// - binding(3): params uniform
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

struct ConcentrationGriddingParams {
    nx: u32,
    ny: u32,
    nz: u32,
    particle_count: u32,
    species_index: u32,
    _pad0: u32,
    _pad1: u32,
    mass_scale: f32,
    outheights: array<vec4<f32>, 4>,
};

const FLAG_ACTIVE: u32 = 1u;
const MAX_U32_F32: f32 = 4294967295.0;

@group(0) @binding(0)
var<storage, read> particles: array<Particle>;

@group(0) @binding(1)
var<storage, read_write> particle_count_grid: array<atomic<u32>>;

@group(0) @binding(2)
var<storage, read_write> concentration_mass_scaled: array<atomic<u32>>;

@group(0) @binding(3)
var<uniform> params: ConcentrationGriddingParams;

fn flatten_index(ix: u32, iy: u32, iz: u32) -> u32 {
    return ((ix * params.ny) + iy) * params.nz + iz;
}

fn get_outheight(kz: u32) -> f32 {
    let vec_idx = kz / 4u;
    let comp_idx = kz % 4u;
    let v = params.outheights[vec_idx];
    if (comp_idx == 0u) { return v.x; }
    if (comp_idx == 1u) { return v.y; }
    if (comp_idx == 2u) { return v.z; }
    return v.w;
}

fn pos_z_to_level(z: f32) -> u32 {
    // Check the top level: if it is > 0, outheights are in meters.
    // This avoids false negatives when the lowest level is at z=0m.
    if (get_outheight(params.nz - 1u) > 0.0) {
        for (var kz = 0u; kz < params.nz; kz = kz + 1u) {
            if (z <= get_outheight(kz)) {
                return kz;
            }
        }
        return params.nz - 1u;
    }
    return u32(clamp(i32(floor(z)), 0, i32(params.nz) - 1));
}

fn quantize_mass_to_u32(mass_kg: f32, mass_scale: f32) -> u32 {
    if (mass_kg <= 0.0) {
        return 0u;
    }
    let scaled = mass_kg * mass_scale + 0.5;
    if (scaled >= MAX_U32_F32) {
        return 0xffffffffu;
    }
    return u32(scaled);
}

@compute @workgroup_size(__WORKGROUP_SIZE_X__)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let particle_id = gid.y * (nwg.x * __WORKGROUP_SIZE_X__u) + gid.x;
    if (particle_id >= params.particle_count) {
        return;
    }

    let particle = particles[particle_id];
    if ((particle.flags & FLAG_ACTIVE) == 0u) {
        return;
    }

    let ix = u32(clamp(particle.cell_x, 0, i32(params.nx) - 1));
    let iy = u32(clamp(particle.cell_y, 0, i32(params.ny) - 1));
    let iz = pos_z_to_level(particle.pos_z);
    let cell_index = flatten_index(ix, iy, iz);

    atomicAdd(&particle_count_grid[cell_index], 1u);

    let mass_kg = max(particle.mass[params.species_index], 0.0);
    let mass_scaled = quantize_mass_to_u32(mass_kg, params.mass_scale);
    atomicAdd(&concentration_mass_scaled[cell_index], mass_scaled);
}
