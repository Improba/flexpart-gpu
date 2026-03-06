// WGSL particle advection kernel (Euler predictor + Petterssen correction).
//
// Numerical scheme (close equivalent to FLEXPART advance.f90 predictor/corrector):
// 1) sample wind at current position (u0, v0, w0)
// 2) Euler predict position: x_pred = x0 + dt * v0_scaled
// 3) sample wind at predicted position (u1, v1, w1)
// 4) corrected wind = 0.5 * (v0 + v1)
// 5) final position: x1 = x0 + dt * corrected_scaled
//
// Buffer contract:
// - particle buffer: array<Particle>, read_write
//   Layout mirrors src/particles/mod.rs (repr(C), 96 bytes).
// - wind buffers (u, v, w): flattened ndarray row-major (nx, ny, nz):
//   flat = ((ix * ny) + iy) * nz + iz
// - params uniform:
//   (nx, ny, nz, particle_count, dt_seconds, x_scale, y_scale, z_scale)
// - active particles are identified by bit 0 in Particle.flags.
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

struct AdvectionParams {
    nx: u32,
    ny: u32,
    nz: u32,
    particle_count: u32,
    dt_seconds: f32,
    x_scale: f32,
    y_scale: f32,
    z_scale: f32,
    // 16 level heights packed as 4 x vec4<f32> for uniform alignment.
    level_heights: array<vec4<f32>, 4>,
};

const FLAG_ACTIVE: u32 = 1u;

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<storage, read> wind_u_ms: array<f32>;

@group(0) @binding(2)
var<storage, read> wind_v_ms: array<f32>;

@group(0) @binding(3)
var<storage, read> wind_w_ms: array<f32>;

@group(0) @binding(4)
var<uniform> params: AdvectionParams;

fn flat_index(ix: u32, iy: u32, iz: u32) -> u32 {
    return ((ix * params.ny) + iy) * params.nz + iz;
}

fn get_level_height(k: u32) -> f32 {
    return params.level_heights[k / 4u][k % 4u];
}

fn has_level_heights() -> bool {
    // Check the top level rather than ground level, since z=0m is a valid height.
    return get_level_height(params.nz - 1u) > 0.0;
}

// Convert physical height (meters) to fractional grid level for wind sampling.
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

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a * (1.0 - t) + b * t;
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

fn sample_u(x: f32, y: f32, z: f32) -> f32 {
    let x_axis = clamp_axis(x, params.nx);
    let y_axis = clamp_axis(y, params.ny);
    let z_level = meters_to_grid_level(z);
    let z_axis = clamp_axis(z_level, params.nz);

    let x0 = u32(x_axis.x);
    let x1 = u32(x_axis.y);
    let tx = x_axis.z;

    let y0 = u32(y_axis.x);
    let y1 = u32(y_axis.y);
    let ty = y_axis.z;

    let z0 = u32(z_axis.x);
    let z1 = u32(z_axis.y);
    let tz = z_axis.z;

    let i000 = flat_index(x0, y0, z0);
    let i100 = flat_index(x1, y0, z0);
    let i010 = flat_index(x0, y1, z0);
    let i110 = flat_index(x1, y1, z0);
    let i001 = flat_index(x0, y0, z1);
    let i101 = flat_index(x1, y0, z1);
    let i011 = flat_index(x0, y1, z1);
    let i111 = flat_index(x1, y1, z1);

    let c00 = lerp(wind_u_ms[i000], wind_u_ms[i100], tx);
    let c10 = lerp(wind_u_ms[i010], wind_u_ms[i110], tx);
    let c01 = lerp(wind_u_ms[i001], wind_u_ms[i101], tx);
    let c11 = lerp(wind_u_ms[i011], wind_u_ms[i111], tx);
    let c0 = lerp(c00, c10, ty);
    let c1 = lerp(c01, c11, ty);
    return lerp(c0, c1, tz);
}

fn sample_v(x: f32, y: f32, z: f32) -> f32 {
    let x_axis = clamp_axis(x, params.nx);
    let y_axis = clamp_axis(y, params.ny);
    let z_level = meters_to_grid_level(z);
    let z_axis = clamp_axis(z_level, params.nz);

    let x0 = u32(x_axis.x);
    let x1 = u32(x_axis.y);
    let tx = x_axis.z;

    let y0 = u32(y_axis.x);
    let y1 = u32(y_axis.y);
    let ty = y_axis.z;

    let z0 = u32(z_axis.x);
    let z1 = u32(z_axis.y);
    let tz = z_axis.z;

    let i000 = flat_index(x0, y0, z0);
    let i100 = flat_index(x1, y0, z0);
    let i010 = flat_index(x0, y1, z0);
    let i110 = flat_index(x1, y1, z0);
    let i001 = flat_index(x0, y0, z1);
    let i101 = flat_index(x1, y0, z1);
    let i011 = flat_index(x0, y1, z1);
    let i111 = flat_index(x1, y1, z1);

    let c00 = lerp(wind_v_ms[i000], wind_v_ms[i100], tx);
    let c10 = lerp(wind_v_ms[i010], wind_v_ms[i110], tx);
    let c01 = lerp(wind_v_ms[i001], wind_v_ms[i101], tx);
    let c11 = lerp(wind_v_ms[i011], wind_v_ms[i111], tx);
    let c0 = lerp(c00, c10, ty);
    let c1 = lerp(c01, c11, ty);
    return lerp(c0, c1, tz);
}

fn sample_w(x: f32, y: f32, z: f32) -> f32 {
    let x_axis = clamp_axis(x, params.nx);
    let y_axis = clamp_axis(y, params.ny);
    let z_level = meters_to_grid_level(z);
    let z_axis = clamp_axis(z_level, params.nz);

    let x0 = u32(x_axis.x);
    let x1 = u32(x_axis.y);
    let tx = x_axis.z;

    let y0 = u32(y_axis.x);
    let y1 = u32(y_axis.y);
    let ty = y_axis.z;

    let z0 = u32(z_axis.x);
    let z1 = u32(z_axis.y);
    let tz = z_axis.z;

    let i000 = flat_index(x0, y0, z0);
    let i100 = flat_index(x1, y0, z0);
    let i010 = flat_index(x0, y1, z0);
    let i110 = flat_index(x1, y1, z0);
    let i001 = flat_index(x0, y0, z1);
    let i101 = flat_index(x1, y0, z1);
    let i011 = flat_index(x0, y1, z1);
    let i111 = flat_index(x1, y1, z1);

    let c00 = lerp(wind_w_ms[i000], wind_w_ms[i100], tx);
    let c10 = lerp(wind_w_ms[i010], wind_w_ms[i110], tx);
    let c01 = lerp(wind_w_ms[i001], wind_w_ms[i101], tx);
    let c11 = lerp(wind_w_ms[i011], wind_w_ms[i111], tx);
    let c0 = lerp(c00, c10, ty);
    let c1 = lerp(c01, c11, ty);
    return lerp(c0, c1, tz);
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

    let u0 = sample_u(x0, y0, z0) + particle.turb_u;
    let v0 = sample_v(x0, y0, z0) + particle.turb_v;
    // Vertical mean wind only: turb_w displacement is handled by
    // Langevin sub-stepping (vertical diffusion + PBL reflection).
    let w0 = sample_w(x0, y0, z0);

    let predicted_x = clamp(x0 + (u0 * params.x_scale) * params.dt_seconds, 0.0, f32(params.nx - 1u));
    let predicted_y = clamp(y0 + (v0 * params.y_scale) * params.dt_seconds, 0.0, f32(params.ny - 1u));
    let predicted_z = clamp_vertical_coord(z0 + (w0 * params.z_scale) * params.dt_seconds);

    let u1 = sample_u(predicted_x, predicted_y, predicted_z) + particle.turb_u;
    let v1 = sample_v(predicted_x, predicted_y, predicted_z) + particle.turb_v;
    let w1 = sample_w(predicted_x, predicted_y, predicted_z);

    let corrected_u = 0.5 * (u0 + u1);
    let corrected_v = 0.5 * (v0 + v1);
    let corrected_w = 0.5 * (w0 + w1);

    let x1 = clamp(x0 + (corrected_u * params.x_scale) * params.dt_seconds, 0.0, f32(params.nx - 1u));
    let y1 = clamp(y0 + (corrected_v * params.y_scale) * params.dt_seconds, 0.0, f32(params.ny - 1u));
    let z1 = clamp_vertical_coord(z0 + (corrected_w * params.z_scale) * params.dt_seconds);

    let x_split = split_horizontal_coord(x1, params.nx);
    let y_split = split_horizontal_coord(y1, params.ny);

    particle.cell_x = i32(x_split.x);
    particle.cell_y = i32(y_split.x);
    particle.pos_x = x_split.y;
    particle.pos_y = y_split.y;
    particle.pos_z = z1;
    particle.vel_u = corrected_u;
    particle.vel_v = corrected_v;
    particle.vel_w = corrected_w;

    particles[particle_id] = particle;
}
