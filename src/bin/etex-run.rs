use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;
use std::time::Instant;

use flexpart_gpu::config::ReleaseConfig;
use flexpart_gpu::coords::GridDomain;
use flexpart_gpu::gpu::{
    ConcentrationGridShape, ConcentrationGriddingParams, GpuError, MAX_OUTPUT_LEVELS,
};
use flexpart_gpu::io::TimeBoundsBehavior;
use flexpart_gpu::physics::VelocityToGridScale;
use flexpart_gpu::simulation::{
    ForwardStepForcing, ForwardTimeLoopConfig, ForwardTimeLoopDriver, MetTimeBracket,
    ParticleForcingField, TimeLoopError,
};
use flexpart_gpu::wind::{SurfaceFields, WindField3D};
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

const R_EARTH: f64 = 6_371_000.0;

#[derive(Deserialize)]
struct Manifest {
    nx: usize,
    ny: usize,
    nz: usize,
    dx_deg: f64,
    dy_deg: f64,
    xlon0_deg: f64,
    ylat0_deg: f64,
    heights_m: Vec<f32>,
    timesteps: Vec<TimestepEntry>,
    release: ReleaseEntry,
    simulation: SimEntry,
    output: OutputEntry,
}

#[derive(Deserialize)]
struct TimestepEntry {
    epoch_seconds: i64,
    file: String,
}

#[derive(Deserialize)]
struct ReleaseEntry {
    start: String,
    end: String,
    lon: f64,
    lat: f64,
    z_min: f64,
    z_max: f64,
    mass_kg: f64,
    particle_count: u64,
}

#[derive(Deserialize)]
struct SimEntry {
    start: String,
    end: String,
    dt_seconds: i64,
}

#[derive(Deserialize)]
struct OutputEntry {
    nx: usize,
    ny: usize,
    nz: usize,
    heights_m: Vec<f32>,
    interval_seconds: i64,
}

#[derive(Serialize)]
struct EtexOutput {
    grid: GridMeta,
    timesteps: Vec<TimestepOutput>,
    total_steps: usize,
    final_active_particles: usize,
}

#[derive(Serialize)]
struct GridMeta {
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    xlon0: f64,
    ylat0: f64,
    heights_m: Vec<f32>,
}

#[derive(Serialize)]
struct TimestepOutput {
    datetime: String,
    epoch_seconds: i64,
    particle_count_per_cell: Vec<u32>,
    concentration_mass_kg: Vec<f32>,
    active_particles: usize,
}

struct MetSnapshot {
    wind: WindField3D,
    surface: SurfaceFields,
    epoch_seconds: i64,
}

struct MetSnapshotPrefetch {
    target_idx: usize,
    handle: JoinHandle<Result<MetSnapshot, String>>,
}

fn load_met_snapshot(
    met_dir: &Path,
    entry: &TimestepEntry,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Result<MetSnapshot, String> {
    let path = met_dir.join(&entry.file);
    let data = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;

    let n3d = nx * ny * nz;
    let n2d = nx * ny;
    let f32_size = std::mem::size_of::<f32>();
    let expected = (8 * n3d + 15 * n2d) * f32_size;
    if data.len() != expected {
        return Err(format!(
            "file {} has {} bytes, expected {}",
            entry.file,
            data.len(),
            expected
        ));
    }

    let floats: Vec<f32> = data
        .chunks_exact(f32_size)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    let mut offset = 0;

    let read_3d = |off: &mut usize| -> Array3<f32> {
        let slice = &floats[*off..*off + n3d];
        *off += n3d;
        Array3::from_shape_vec((nx, ny, nz), slice.to_vec()).expect("validated shape")
    };
    let read_2d = |off: &mut usize| -> Array2<f32> {
        let slice = &floats[*off..*off + n2d];
        *off += n2d;
        Array2::from_shape_vec((nx, ny), slice.to_vec()).expect("validated shape")
    };

    let u = read_3d(&mut offset);
    let v = read_3d(&mut offset);
    let w = read_3d(&mut offset);
    let t = read_3d(&mut offset);
    let q = read_3d(&mut offset);
    let p = read_3d(&mut offset);
    let rho = read_3d(&mut offset);
    let drhodz = read_3d(&mut offset);

    let sp = read_2d(&mut offset);
    let u10 = read_2d(&mut offset);
    let v10 = read_2d(&mut offset);
    let t2m = read_2d(&mut offset);
    let td2m = read_2d(&mut offset);
    let lsp = read_2d(&mut offset);
    let cp = read_2d(&mut offset);
    let sshf = read_2d(&mut offset);
    let ssr = read_2d(&mut offset);
    let surfstr = read_2d(&mut offset);
    let ustar = read_2d(&mut offset);
    let wstar = read_2d(&mut offset);
    let hmix = read_2d(&mut offset);
    let tropo = read_2d(&mut offset);
    let oli = read_2d(&mut offset);

    Ok(MetSnapshot {
        wind: WindField3D {
            u_ms: u,
            v_ms: v,
            w_ms: w,
            temperature_k: t,
            specific_humidity: q,
            pressure_pa: p,
            air_density_kg_m3: rho,
            density_gradient_kg_m2: drhodz,
        },
        surface: SurfaceFields {
            surface_pressure_pa: sp,
            u10_ms: u10,
            v10_ms: v10,
            temperature_2m_k: t2m,
            dewpoint_2m_k: td2m,
            precip_large_scale_mm_h: lsp,
            precip_convective_mm_h: cp,
            sensible_heat_flux_w_m2: sshf,
            solar_radiation_w_m2: ssr,
            surface_stress_n_m2: surfstr,
            friction_velocity_ms: ustar,
            convective_velocity_scale_ms: wstar,
            mixing_height_m: hmix,
            tropopause_height_m: tropo,
            inv_obukhov_length_per_m: oli,
        },
        epoch_seconds: entry.epoch_seconds,
    })
}

fn start_met_prefetch(
    met_dir: &Path,
    entry: &TimestepEntry,
    nx: usize,
    ny: usize,
    nz: usize,
    target_idx: usize,
) -> MetSnapshotPrefetch {
    let met_dir = met_dir.to_path_buf();
    let file = entry.file.clone();
    let epoch_seconds = entry.epoch_seconds;
    let handle = std::thread::spawn(move || {
        let entry = TimestepEntry { epoch_seconds, file };
        load_met_snapshot(&met_dir, &entry, nx, ny, nz)
    });
    MetSnapshotPrefetch { target_idx, handle }
}

fn consume_prefetch(
    prefetch: MetSnapshotPrefetch,
    expected_idx: usize,
) -> Result<MetSnapshot, String> {
    if prefetch.target_idx != expected_idx {
        return Err(format!(
            "prefetch target mismatch: got {}, expected {}",
            prefetch.target_idx, expected_idx
        ));
    }
    prefetch
        .handle
        .join()
        .map_err(|_| format!("prefetch worker panicked for index {expected_idx}"))?
}

fn main() {
    env_logger::init();

    let manifest_path: PathBuf = std::env::var("ETEX_MANIFEST")
        .unwrap_or_else(|_| "target/etex/gpu_meteo/manifest.json".to_string())
        .into();
    let output_path: PathBuf = std::env::var("OUTPUT_PATH")
        .unwrap_or_else(|_| "target/etex/gpu_output.json".to_string())
        .into();

    eprintln!("=== ETEX-1 GPU Simulation ===");
    eprintln!("manifest: {}", manifest_path.display());

    let manifest_text = fs::read_to_string(&manifest_path).expect("read manifest");
    let manifest: Manifest = serde_json::from_str(&manifest_text).expect("parse manifest");
    let prefetch_enabled = std::env::var("ETEX_STREAM_PREFETCH")
        .map(|v| v != "0")
        .unwrap_or(true);

    let met_dir = manifest_path.parent().unwrap();
    let nx = manifest.nx;
    let ny = manifest.ny;
    let nz = manifest.nz;

    let particle_count: u64 = std::env::var("PARTICLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(manifest.release.particle_count);

    eprintln!(
        "grid: {}×{}×{}, dx={:.4}°, dy={:.4}°",
        nx, ny, nz, manifest.dx_deg, manifest.dy_deg
    );
    eprintln!(
        "domain: [{:.1}, {:.1}] × [{:.1}, {:.1}]",
        manifest.xlon0_deg,
        manifest.xlon0_deg + (nx as f64 - 1.0) * manifest.dx_deg,
        manifest.ylat0_deg,
        manifest.ylat0_deg + (ny as f64 - 1.0) * manifest.dy_deg
    );
    eprintln!(
        "release: ({}, {}), z=[{},{}]m, {} particles, {} kg",
        manifest.release.lon,
        manifest.release.lat,
        manifest.release.z_min,
        manifest.release.z_max,
        particle_count,
        manifest.release.mass_kg
    );
    eprintln!(
        "simulation: {} → {}, dt={}s",
        manifest.simulation.start, manifest.simulation.end, manifest.simulation.dt_seconds
    );
    eprintln!("met prefetch: {}", if prefetch_enabled { "enabled" } else { "disabled" });

    let center_lat = manifest.release.lat;
    let lat_rad = center_lat * std::f64::consts::PI / 180.0;
    let dx_m = R_EARTH * lat_rad.cos() * (manifest.dx_deg * std::f64::consts::PI / 180.0);
    let dy_m = R_EARTH * (manifest.dy_deg * std::f64::consts::PI / 180.0);
    let velocity_scale = VelocityToGridScale {
        x_grid_per_meter: (1.0 / dx_m) as f32,
        y_grid_per_meter: (1.0 / dy_m) as f32,
        z_grid_per_meter: 1.0,
        level_heights_m: {
            let mut h = [0.0_f32; 16];
            for (i, &v) in manifest.heights_m.iter().take(16).enumerate() {
                h[i] = v;
            }
            h
        },
    };

    let release_grid = GridDomain {
        xlon0: manifest.xlon0_deg,
        ylat0: manifest.ylat0_deg,
        dx: manifest.dx_deg,
        dy: manifest.dy_deg,
        nx,
        ny,
    };

    let releases = vec![ReleaseConfig {
        name: "ETEX-1".to_string(),
        start_time: manifest.release.start.clone(),
        end_time: manifest.release.end.clone(),
        lon: manifest.release.lon,
        lat: manifest.release.lat,
        z_min: manifest.release.z_min,
        z_max: manifest.release.z_max,
        mass_kg: manifest.release.mass_kg,
        particle_count,
        raw: BTreeMap::new(),
    }];

    let config = ForwardTimeLoopConfig {
        start_timestamp: manifest.simulation.start.clone(),
        end_timestamp: manifest.simulation.end.clone(),
        timestep_seconds: manifest.simulation.dt_seconds,
        time_bounds_behavior: TimeBoundsBehavior::Clamp,
        velocity_to_grid_scale: velocity_scale,
        sync_particle_store_each_step: false,
        collect_deposition_probabilities_each_step: false,
        ..ForwardTimeLoopConfig::default()
    };

    eprintln!("initializing GPU driver...");
    let mut driver = match pollster::block_on(ForwardTimeLoopDriver::new(
        config,
        &releases,
        release_grid,
        particle_count as usize,
    )) {
        Ok(d) => d,
        Err(TimeLoopError::Gpu(GpuError::NoAdapter)) => {
            eprintln!("ERROR: no GPU adapter found");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("GPU driver ready");

    if manifest.timesteps.len() < 2 {
        eprintln!(
            "ERROR: manifest needs at least 2 met snapshots, got {}",
            manifest.timesteps.len()
        );
        std::process::exit(1);
    }

    eprintln!(
        "loading first met bracket (streaming mode, {} snapshots total)...",
        manifest.timesteps.len()
    );
    let init_load_start = Instant::now();
    let mut current_snapshot = load_met_snapshot(met_dir, &manifest.timesteps[0], nx, ny, nz)
        .unwrap_or_else(|e| panic!("load timestep 0 failed: {e}"));
    let mut next_snapshot = load_met_snapshot(met_dir, &manifest.timesteps[1], nx, ny, nz)
        .unwrap_or_else(|e| panic!("load timestep 1 failed: {e}"));
    let init_load_ms = init_load_start.elapsed().as_secs_f64() * 1_000.0;
    let mut prefetch = if prefetch_enabled && manifest.timesteps.len() > 2 {
        Some(start_met_prefetch(
            met_dir,
            &manifest.timesteps[2],
            nx,
            ny,
            nz,
            2,
        ))
    } else {
        None
    };
    let mut prefetch_hits = 0_usize;
    let mut prefetch_misses = 0_usize;
    let mut met_wait_prefetch_ms = 0.0_f64;
    let mut met_wait_sync_ms = 0.0_f64;
    eprintln!("met streaming initialized");

    let forcing = ForwardStepForcing {
        dry_deposition_velocity_m_s: ParticleForcingField::Uniform(0.0),
        wet_scavenging_coefficient_s_inv: ParticleForcingField::Uniform(0.0),
        wet_precipitating_fraction: ParticleForcingField::Uniform(0.0),
        rho_grad_over_rho: 0.0,
    };

    let out_heights = {
        let mut h = [0.0_f32; MAX_OUTPUT_LEVELS];
        for (i, &v) in manifest.output.heights_m.iter().take(MAX_OUTPUT_LEVELS).enumerate() {
            h[i] = v;
        }
        h
    };

    let mut timestep_outputs = Vec::new();
    let mut total_steps = 0_usize;
    let output_interval = manifest.output.interval_seconds;

    // Compute simulation start epoch from the driver's internal timestamp
    let sim_start_epoch = driver.current_time_seconds();
    let mut next_output_time = sim_start_epoch + output_interval;

    eprintln!("running simulation...");

    let mut bracket_idx = 0_usize;
    while driver.has_remaining_steps() && bracket_idx + 1 < manifest.timesteps.len() {
        let t0 = &current_snapshot;
        let t1 = &next_snapshot;

        let met = MetTimeBracket {
            wind_t0: &t0.wind,
            wind_t1: &t1.wind,
            surface_t0: &t0.surface,
            surface_t1: &t1.surface,
            time_t0_seconds: t0.epoch_seconds,
            time_t1_seconds: t1.epoch_seconds,
        };

        while driver.has_remaining_steps() && driver.current_time_seconds() < t1.epoch_seconds {
            let report = pollster::block_on(driver.run_timestep(&met, &forcing))
                .expect("timestep failed");
            total_steps += 1;

            if driver.current_time_seconds() >= next_output_time {
                let t_h = (driver.current_time_seconds() - sim_start_epoch) as f64 / 3600.0;
                eprintln!(
                    "  t+{:.1}h (bracket {}/{}): {} active particles",
                    t_h,
                    bracket_idx,
                    manifest.timesteps.len() - 1,
                    report.active_particle_count
                );

                let conc = pollster::block_on(driver.accumulate_concentration_grid(
                    ConcentrationGridShape {
                        nx: manifest.output.nx,
                        ny: manifest.output.ny,
                        nz: manifest.output.nz,
                    },
                    ConcentrationGriddingParams {
                        species_index: 0,
                        mass_scale: 1e5,
                        outheights: out_heights,
                    },
                ))
                .expect("concentration gridding failed");

                let epoch = driver.current_time_seconds();
                let dt_str = epoch_to_flexpart_timestamp(epoch);

                timestep_outputs.push(TimestepOutput {
                    datetime: dt_str,
                    epoch_seconds: epoch,
                    particle_count_per_cell: conc.particle_count_per_cell,
                    concentration_mass_kg: conc.concentration_mass_kg,
                    active_particles: report.active_particle_count,
                });

                next_output_time += output_interval;
            }
        }
        bracket_idx += 1;

        if bracket_idx + 1 >= manifest.timesteps.len() {
            break;
        }

        current_snapshot = next_snapshot;
        let needed_idx = bracket_idx + 1;

        next_snapshot = if let Some(handle) = prefetch.take() {
            prefetch_hits += 1;
            let t_wait = Instant::now();
            let result = consume_prefetch(handle, needed_idx)
                .unwrap_or_else(|e| panic!("consume prefetch for index {needed_idx} failed: {e}"))
                ;
            met_wait_prefetch_ms += t_wait.elapsed().as_secs_f64() * 1_000.0;
            result
        } else {
            prefetch_misses += 1;
            let t_wait = Instant::now();
            let result = load_met_snapshot(met_dir, &manifest.timesteps[needed_idx], nx, ny, nz)
                .unwrap_or_else(|e| panic!("load timestep {needed_idx} failed: {e}"))
                ;
            met_wait_sync_ms += t_wait.elapsed().as_secs_f64() * 1_000.0;
            result
        };

        let upcoming_idx = bracket_idx + 2;
        if prefetch_enabled && upcoming_idx < manifest.timesteps.len() {
            prefetch = Some(start_met_prefetch(
                met_dir,
                &manifest.timesteps[upcoming_idx],
                nx,
                ny,
                nz,
                upcoming_idx,
            ));
        }
    }

    let active = timestep_outputs
        .last()
        .map(|t| t.active_particles)
        .unwrap_or(0);
    eprintln!(
        "simulation complete: {} steps, {} outputs, {} active particles",
        total_steps,
        timestep_outputs.len(),
        active
    );
    eprintln!(
        "met prefetch stats: hits={}, misses={}",
        prefetch_hits, prefetch_misses
    );
    eprintln!(
        "met load timing: init={:.2}ms prefetch_wait={:.2}ms sync_wait={:.2}ms total_wait={:.2}ms",
        init_load_ms,
        met_wait_prefetch_ms,
        met_wait_sync_ms,
        met_wait_prefetch_ms + met_wait_sync_ms
    );

    let output = EtexOutput {
        grid: GridMeta {
            nx: manifest.output.nx,
            ny: manifest.output.ny,
            nz: manifest.output.nz,
            dx: manifest.dx_deg,
            dy: manifest.dy_deg,
            xlon0: manifest.xlon0_deg,
            ylat0: manifest.ylat0_deg,
            heights_m: manifest.output.heights_m.clone(),
        },
        timesteps: timestep_outputs,
        total_steps,
        final_active_particles: active,
    };

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let json = serde_json::to_string(&output).expect("serialize output");
    fs::write(&output_path, &json).expect("write output");
    eprintln!("wrote {}", output_path.display());
}

fn epoch_to_flexpart_timestamp(epoch: i64) -> String {
    let days_since_epoch = epoch / 86400;
    let time_of_day = epoch % 86400;
    let h = time_of_day / 3600;
    let m = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;

    // Convert days since 1970-01-01 to date
    let mut y = 1970_i64;
    let mut remaining = days_since_epoch;
    loop {
        let days_in_year = if is_leap(y) { 366 } else { 365 };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        y += 1;
    }
    let month_days: [i64; 12] = if is_leap(y) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut mo = 1_i64;
    for &md in &month_days {
        if remaining < md {
            break;
        }
        remaining -= md;
        mo += 1;
    }
    let d = remaining + 1;
    format!("{y:04}{mo:02}{d:02}{h:02}{m:02}{s:02}")
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}
