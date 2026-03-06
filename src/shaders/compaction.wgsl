// Active particle stream compaction via parallel prefix-sum (Tier 3.1).
//
// Four-pass GPU algorithm that identifies active particles, builds a
// contiguous index mapping, and physically reorders the particle buffer
// so subsequent dispatches process only active slots.
//
// Pass 1 (local_prefix_sum): Extract bit-0 active flag from each particle,
//   compute workgroup-local exclusive prefix sum via Hillis-Steele scan,
//   store per-workgroup totals.
// Pass 2 (scan_workgroup_sums): Exclusive prefix sum over the per-workgroup
//   totals to produce global offsets. Single-workgroup dispatch with
//   sequential chunking for scalability to millions of particles.
// Pass 3 (scatter_compact): Each active particle writes its original index
//   to compacted_indices[global_offset + local_prefix].
// Pass 4 (gather_reorder): Copies active particles to contiguous leading
//   positions in a staging buffer; tail slots are explicitly deactivated.
//   The caller then copies staging → particles via buffer copy.

const WG_SIZE: u32 = __WORKGROUP_SIZE_X__u;
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

struct CompactionParams {
    particle_count: u32,
    num_workgroups_pass1: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> local_prefixes: array<u32>;
@group(0) @binding(2) var<storage, read_write> workgroup_sums: array<u32>;
@group(0) @binding(3) var<storage, read_write> compacted_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> active_count_buf: array<u32>;
@group(0) @binding(5) var<uniform> params: CompactionParams;
@group(0) @binding(6) var<storage, read_write> staging_particles: array<Particle>;

var<workgroup> shared_data: array<u32, __WORKGROUP_SIZE_X__>;

// ─── Pass 1: Flag extraction + workgroup-local exclusive prefix sum ──────

@compute @workgroup_size(__WORKGROUP_SIZE_X__)
fn local_prefix_sum(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let workgroup_id = wid.y * nwg.x + wid.x;
    let global_id = workgroup_id * WG_SIZE + lid.x;
    let local_id = lid.x;

    var flag = 0u;
    if (global_id < params.particle_count) {
        flag = particles[global_id].flags & FLAG_ACTIVE;
    }

    // Hillis-Steele inclusive prefix sum in shared memory.
    // After the loop, shared_data[i] = sum(flag[0..=i]) within the workgroup.
    shared_data[local_id] = flag;
    workgroupBarrier();

    for (var offset = 1u; offset < WG_SIZE; offset = offset * 2u) {
        var val = 0u;
        if (local_id >= offset) {
            val = shared_data[local_id - offset];
        }
        workgroupBarrier();
        shared_data[local_id] = shared_data[local_id] + val;
        workgroupBarrier();
    }

    let inclusive = shared_data[local_id];
    let exclusive = inclusive - flag;

    if (global_id < params.particle_count) {
        local_prefixes[global_id] = exclusive;
    }

    // Guard against overflow when dispatch_1d uses 2D tiling
    if (local_id == WG_SIZE - 1u && workgroup_id < params.num_workgroups_pass1) {
        workgroup_sums[workgroup_id] = inclusive;
    }
}

// ─── Pass 2: Exclusive prefix sum over per-workgroup totals ──────────────
//
// Single-workgroup dispatch. Each thread sequentially sums a chunk of
// workgroup_sums, then a shared-memory scan propagates inter-chunk offsets.
// Handles up to WG_SIZE^2 workgroups (16M+ particles at WG_SIZE=256).

@compute @workgroup_size(__WORKGROUP_SIZE_X__)
fn scan_workgroup_sums(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let local_id = lid.x;
    let n = params.num_workgroups_pass1;
    let chunk = (n + WG_SIZE - 1u) / WG_SIZE;
    let my_start = local_id * chunk;
    let my_end = min(my_start + chunk, n);

    // Sequential accumulation of this thread's chunk
    var thread_total = 0u;
    for (var i = my_start; i < my_end; i = i + 1u) {
        thread_total = thread_total + workgroup_sums[i];
    }

    // Hillis-Steele inclusive scan of per-thread totals
    shared_data[local_id] = thread_total;
    workgroupBarrier();

    for (var offset = 1u; offset < WG_SIZE; offset = offset * 2u) {
        var val = 0u;
        if (local_id >= offset) {
            val = shared_data[local_id - offset];
        }
        workgroupBarrier();
        shared_data[local_id] = shared_data[local_id] + val;
        workgroupBarrier();
    }

    // Convert inclusive scan to exclusive offset for this chunk
    let base = shared_data[local_id] - thread_total;

    // Write exclusive prefix sums back, replacing the original totals
    var running = base;
    for (var i = my_start; i < my_end; i = i + 1u) {
        let val = workgroup_sums[i];
        workgroup_sums[i] = running;
        running = running + val;
    }

    // Last thread writes total active count (= inclusive scan of last element)
    if (local_id == WG_SIZE - 1u) {
        active_count_buf[0] = shared_data[WG_SIZE - 1u];
    }
}

// ─── Pass 3: Scatter active particle indices to compacted positions ──────

@compute @workgroup_size(__WORKGROUP_SIZE_X__)
fn scatter_compact(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let workgroup_id = wid.y * nwg.x + wid.x;
    let global_id = workgroup_id * WG_SIZE + lid.x;

    if (global_id >= params.particle_count) {
        return;
    }

    if ((particles[global_id].flags & FLAG_ACTIVE) == 0u) {
        return;
    }

    let local_prefix = local_prefixes[global_id];
    let workgroup_offset = workgroup_sums[workgroup_id];
    let output_index = workgroup_offset + local_prefix;

    compacted_indices[output_index] = global_id;
}

// ─── Pass 4: Gather active particles into contiguous leading positions ───
//
// Active particles are copied from their scattered original positions into
// contiguous slots 0..active_count-1 in a staging buffer. Tail slots
// (>= active_count) receive a deactivated copy to prevent false positives
// on host-side active-count recounting after GPU→CPU readback.

@compute @workgroup_size(__WORKGROUP_SIZE_X__)
fn gather_reorder(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let workgroup_id = wid.y * nwg.x + wid.x;
    let global_id = workgroup_id * WG_SIZE + lid.x;

    if (global_id >= params.particle_count) {
        return;
    }

    let active_count = active_count_buf[0];

    if (global_id < active_count) {
        let src_idx = compacted_indices[global_id];
        staging_particles[global_id] = particles[src_idx];
    } else {
        var p = particles[global_id];
        p.flags = p.flags & ~FLAG_ACTIVE;
        staging_particles[global_id] = p;
    }
}
