// WGSL trilinear interpolation kernel for wind vectors (u, v, w).
//
// Buffer contract (all flattened as row-major ndarray order for shape nx,ny,nz):
//   flat_index = ((ix * ny) + iy) * nz + iz
//
// Bindings:
//   0: u-field storage buffer, f32[cell_count], read-only
//   1: v-field storage buffer, f32[cell_count], read-only
//   2: w-field storage buffer, f32[cell_count], read-only
//   3: query positions, vec4<f32>[query_count], read-only
//      - xyz = fractional grid coordinates
//      - w   = padding (ignored)
//   4: output vectors, vec4<f32>[query_count], read-write
//      - xyz = interpolated (u, v, w)
//      - w   = padding (set to 0)
//   5: uniform params (nx, ny, nz, query_count), u32x4
struct InterpolationParams {
    nx: u32,
    ny: u32,
    nz: u32,
    query_count: u32,
};

@group(0) @binding(0)
var<storage, read> wind_u_ms: array<f32>;

@group(0) @binding(1)
var<storage, read> wind_v_ms: array<f32>;

@group(0) @binding(2)
var<storage, read> wind_w_ms: array<f32>;

@group(0) @binding(3)
var<storage, read> queries_xyz: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read_write> outputs_uvw: array<vec4<f32>>;

@group(0) @binding(5)
var<uniform> params: InterpolationParams;

fn flat_index(ix: u32, iy: u32, iz: u32) -> u32 {
    return ((ix * params.ny) + iy) * params.nz + iz;
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let qid = gid.y * (nwg.x * 64u) + gid.x;
    if (qid >= params.query_count) {
        return;
    }

    let q = queries_xyz[qid];
    let x_axis = clamp_axis(q.x, params.nx);
    let y_axis = clamp_axis(q.y, params.ny);
    let z_axis = clamp_axis(q.z, params.nz);

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

    let u00 = lerp(wind_u_ms[i000], wind_u_ms[i100], tx);
    let u10 = lerp(wind_u_ms[i010], wind_u_ms[i110], tx);
    let u01 = lerp(wind_u_ms[i001], wind_u_ms[i101], tx);
    let u11 = lerp(wind_u_ms[i011], wind_u_ms[i111], tx);
    let u0 = lerp(u00, u10, ty);
    let u1 = lerp(u01, u11, ty);
    let u = lerp(u0, u1, tz);

    let v00 = lerp(wind_v_ms[i000], wind_v_ms[i100], tx);
    let v10 = lerp(wind_v_ms[i010], wind_v_ms[i110], tx);
    let v01 = lerp(wind_v_ms[i001], wind_v_ms[i101], tx);
    let v11 = lerp(wind_v_ms[i011], wind_v_ms[i111], tx);
    let v0 = lerp(v00, v10, ty);
    let v1 = lerp(v01, v11, ty);
    let v = lerp(v0, v1, tz);

    let w00 = lerp(wind_w_ms[i000], wind_w_ms[i100], tx);
    let w10 = lerp(wind_w_ms[i010], wind_w_ms[i110], tx);
    let w01 = lerp(wind_w_ms[i001], wind_w_ms[i101], tx);
    let w11 = lerp(wind_w_ms[i011], wind_w_ms[i111], tx);
    let w0 = lerp(w00, w10, ty);
    let w1 = lerp(w01, w11, ty);
    let w = lerp(w0, w1, tz);

    outputs_uvw[qid] = vec4<f32>(u, v, w, 0.0);
}
