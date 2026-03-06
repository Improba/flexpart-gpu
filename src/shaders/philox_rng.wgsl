// WGSL Philox4x32-10 random generator.
//
// Buffer contract:
//   - binding 0: output `vec4<f32>[sample_count]`, one block/thread
//   - binding 1: uniform params (`key`, `base_counter`, `sample_count`)
//
// The implementation mirrors `src/physics/rng.rs` so GPU/CPU parity is exact.

const PHILOX_M0: u32 = 0xD2511F53u;
const PHILOX_M1: u32 = 0xCD9E8D57u;
const PHILOX_W0: u32 = 0x9E3779B9u;
const PHILOX_W1: u32 = 0xBB67AE85u;
const PHILOX_ROUNDS: u32 = 10u;
const U32_TO_UNIT_SCALE_24BIT: f32 = 1.0 / 16777216.0;

struct PhiloxParams {
    key0: u32,
    key1: u32,
    counter0: u32,
    counter1: u32,
    counter2: u32,
    counter3: u32,
    sample_count: u32,
    pad0: u32,
};

@group(0) @binding(0)
var<storage, read_write> output_uniforms: array<vec4<f32>>;

@group(0) @binding(1)
var<uniform> params: PhiloxParams;

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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let sample_index = gid.y * (nwg.x * 64u) + gid.x;
    if (sample_index >= params.sample_count) {
        return;
    }

    let key = vec2<u32>(params.key0, params.key1);
    let base_counter = vec4<u32>(
        params.counter0,
        params.counter1,
        params.counter2,
        params.counter3
    );
    let counter = counter_add(base_counter, sample_index);
    let block = philox4x32_10(counter, key);

    output_uniforms[sample_index] = vec4<f32>(
        u32_to_uniform01(block.x),
        u32_to_uniform01(block.y),
        u32_to_uniform01(block.z),
        u32_to_uniform01(block.w)
    );
}
