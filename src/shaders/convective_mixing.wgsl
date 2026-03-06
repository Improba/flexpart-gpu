// WGSL convective mixing kernel (C-03).
//
// Applies a precomputed column-stochastic redistribution matrix M(i,j) to
// particle vertical position by moving each active particle to the expected
// destination height:
//   z_new = sum_i( M(i, source_level) * z_center(i) ).
//
// Buffer contract:
// - binding(0): particles storage buffer (read_write)
// - binding(1): redistribution matrix, column-major (read)
// - binding(2): level interfaces [m], len = level_count + 1 (read)
// - binding(3): level centers [m], len = level_count (read)
// - binding(4): dispatch params (particle_count, level_count, pad0, pad1)

const FLAG_ACTIVE: u32 = 1u;

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

struct ConvectiveMixingDispatchParams {
    particle_count: u32,
    level_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<storage, read> matrix_column_major: array<f32>;

@group(0) @binding(2)
var<storage, read> level_interfaces_m: array<f32>;

@group(0) @binding(3)
var<storage, read> level_centers_m: array<f32>;

@group(0) @binding(4)
var<uniform> params: ConvectiveMixingDispatchParams;

fn locate_source_level(level_count: u32, z_m: f32) -> u32 {
    let max_level = level_count - 1u;
    var source_level = max_level;
    for (var level = 0u; level < level_count; level = level + 1u) {
        let lower = level_interfaces_m[level];
        let upper = level_interfaces_m[level + 1u];
        if (z_m >= lower && z_m < upper) {
            source_level = level;
            break;
        }
    }
    return source_level;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let particle_id = gid.y * (nwg.x * 64u) + gid.x;
    if (particle_id >= params.particle_count) {
        return;
    }
    if (params.level_count == 0u) {
        return;
    }

    var particle = particles[particle_id];
    if ((particle.flags & FLAG_ACTIVE) == 0u) {
        return;
    }

    let level_count = params.level_count;
    let z_min = level_interfaces_m[0u];
    let z_max = level_interfaces_m[level_count];
    let z_clamped = clamp(particle.pos_z, z_min, z_max);
    let source_level = locate_source_level(level_count, z_clamped);

    var weighted_height = 0.0;
    var weight_sum = 0.0;
    let matrix_base = source_level * level_count;
    for (var destination_level = 0u; destination_level < level_count; destination_level = destination_level + 1u) {
        let weight = max(matrix_column_major[matrix_base + destination_level], 0.0);
        weighted_height = weighted_height + weight * level_centers_m[destination_level];
        weight_sum = weight_sum + weight;
    }

    if (weight_sum > 0.0) {
        particle.pos_z = clamp(weighted_height / weight_sum, z_min, z_max);
    } else {
        particle.pos_z = z_clamped;
    }
    particle.cbt = i32(source_level);
    particles[particle_id] = particle;
}
