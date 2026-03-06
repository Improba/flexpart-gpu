// WGSL particle advection kernel using dual-time sampled 3-D wind textures.
//
// Variant of advection_texture.wgsl for Tier 1.2 optimization: receives two
// wind texture snapshots (t0, t1) and blends them inline using alpha.
// Hardware trilinear filtering is used for spatial interpolation.
//
// Numerical scheme is identical to advection.wgsl (Euler predictor + Petterssen
// corrector); only the wind sampling changes.
//
// Texture contract:
// - `wind_uvw_texture_t0`, `wind_uvw_texture_t1`: 3-D `rgba32float`,
//   rgba = (u, v, w, pad). Texture axes: x=sim.z, y=sim.y, z=sim.x.
// - `wind_sampler`: shared linear sampler for both textures.
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

struct DualWindAdvectionParams {
    nx: u32,
    ny: u32,
    nz: u32,
    particle_count: u32,
    dt_seconds: f32,
    x_scale: f32,
    y_scale: f32,
    z_scale: f32,
    alpha: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    level_heights: array<vec4<f32>, 4>,
};

const FLAG_ACTIVE: u32 = 1u;

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var wind_uvw_texture_t0: texture_3d<f32>;

@group(0) @binding(2)
var wind_uvw_texture_t1: texture_3d<f32>;

@group(0) @binding(3)
var wind_sampler: sampler;

@group(0) @binding(4)
var<uniform> params: DualWindAdvectionParams;

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

fn to_texture_coords(x: f32, y: f32, z: f32) -> vec3<f32> {
    let clamped_x = clamp(x, 0.0, f32(params.nx - 1u));
    let clamped_y = clamp(y, 0.0, f32(params.ny - 1u));
    let z_level = meters_to_grid_level(z);
    let clamped_z = clamp(z_level, 0.0, f32(params.nz - 1u));

    let tx = (clamped_z + 0.5) / f32(params.nz);
    let ty = (clamped_y + 0.5) / f32(params.ny);
    let tz = (clamped_x + 0.5) / f32(params.nx);
    return vec3<f32>(tx, ty, tz);
}

fn sample_wind_dual(x: f32, y: f32, z: f32) -> vec3<f32> {
    let coords = to_texture_coords(x, y, z);
    let w0 = textureSampleLevel(wind_uvw_texture_t0, wind_sampler, coords, 0.0).xyz;
    let w1 = textureSampleLevel(wind_uvw_texture_t1, wind_sampler, coords, 0.0).xyz;
    let a = params.alpha;
    return w0 * (1.0 - a) + w1 * a;
}

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

    let x0 = f32(particle.cell_x) + particle.pos_x;
    let y0 = f32(particle.cell_y) + particle.pos_y;
    let z0 = particle.pos_z;

    let wind0_raw = sample_wind_dual(x0, y0, z0);
    let u0 = wind0_raw.x + particle.turb_u;
    let v0 = wind0_raw.y + particle.turb_v;
    let w0 = wind0_raw.z;
    let predicted_x = clamp(x0 + (u0 * params.x_scale) * params.dt_seconds, 0.0, f32(params.nx - 1u));
    let predicted_y = clamp(y0 + (v0 * params.y_scale) * params.dt_seconds, 0.0, f32(params.ny - 1u));
    let predicted_z = clamp_vertical_coord(z0 + (w0 * params.z_scale) * params.dt_seconds);

    let wind1_raw = sample_wind_dual(predicted_x, predicted_y, predicted_z);
    let u1 = wind1_raw.x + particle.turb_u;
    let v1 = wind1_raw.y + particle.turb_v;
    let w1 = wind1_raw.z;
    let corrected_wind = vec3<f32>(0.5 * (u0 + u1), 0.5 * (v0 + v1), 0.5 * (w0 + w1));

    let x1 = clamp(x0 + (corrected_wind.x * params.x_scale) * params.dt_seconds, 0.0, f32(params.nx - 1u));
    let y1 = clamp(y0 + (corrected_wind.y * params.y_scale) * params.dt_seconds, 0.0, f32(params.ny - 1u));
    let z1 = clamp_vertical_coord(z0 + (corrected_wind.z * params.z_scale) * params.dt_seconds);

    let x_split = split_horizontal_coord(x1, params.nx);
    let y_split = split_horizontal_coord(y1, params.ny);

    particle.cell_x = i32(x_split.x);
    particle.cell_y = i32(y_split.x);
    particle.pos_x = x_split.y;
    particle.pos_y = y_split.y;
    particle.pos_z = z1;
    particle.vel_u = corrected_wind.x;
    particle.vel_v = corrected_wind.y;
    particle.vel_w = corrected_wind.z;

    particles[particle_id] = particle;
}
