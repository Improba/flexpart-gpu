use std::collections::BTreeMap;
use std::time::Instant;

use flexpart_gpu::config::ReleaseConfig;
use flexpart_gpu::coords::GridDomain;
use flexpart_gpu::gpu::GpuError;
use flexpart_gpu::io::TimeBoundsBehavior;
use flexpart_gpu::physics::VelocityToGridScale;
use flexpart_gpu::simulation::{
    ForwardStepForcing, ForwardTimeLoopConfig, ForwardTimeLoopDriver, MetTimeBracket,
    ParticleForcingField, TimeLoopError,
};
use flexpart_gpu::wind::{SurfaceFields, WindField3D};

const GRID_NX: usize = 64;
const GRID_NY: usize = 64;
const GRID_NZ: usize = 24;

fn main() {
    env_logger::init();

    let particle_count: usize = std::env::var("PARTICLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000_000);
    let warmup_steps: usize = std::env::var("WARMUP_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let measure_steps: usize = std::env::var("MEASURE_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);

    eprintln!("=== bench-timeloop ===");
    eprintln!("particles:     {particle_count}");
    eprintln!("warmup_steps:  {warmup_steps}");
    eprintln!("measure_steps: {measure_steps}");

    let mut raw = BTreeMap::new();
    raw.insert("lon_min".to_string(), "10.0".to_string());
    raw.insert("lon_max".to_string(), "30.0".to_string());
    raw.insert("lat_min".to_string(), "5.0".to_string());
    raw.insert("lat_max".to_string(), "25.0".to_string());

    let releases = vec![ReleaseConfig {
        name: "bench".to_string(),
        start_time: "20240101000000".to_string(),
        end_time: "20240101000000".to_string(),
        lon: 10.0,
        lat: 5.0,
        z_min: 1.0,
        z_max: (GRID_NZ.saturating_sub(2)) as f64,
        mass_kg: 1.0,
        particle_count: particle_count as u64,
        raw,
    }];
    let release_grid = GridDomain {
        xlon0: 0.0,
        ylat0: 0.0,
        dx: 1.0,
        dy: 1.0,
        nx: GRID_NX,
        ny: GRID_NY,
    };
    let config = ForwardTimeLoopConfig {
        start_timestamp: "20240101000000".to_string(),
        end_timestamp: "20300101000000".to_string(),
        timestep_seconds: std::env::var("TIMESTEP_SECONDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1),
        time_bounds_behavior: TimeBoundsBehavior::Clamp,
        velocity_to_grid_scale: VelocityToGridScale::IDENTITY,
        sync_particle_store_each_step: false,
        collect_deposition_probabilities_each_step: false,
        ..ForwardTimeLoopConfig::default()
    };

    let init_start = Instant::now();
    let mut driver = match pollster::block_on(ForwardTimeLoopDriver::new(
        config,
        &releases,
        release_grid,
        particle_count,
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
    let init_elapsed = init_start.elapsed();
    eprintln!("init:          {:.3} ms", init_elapsed.as_secs_f64() * 1e3);

    let wind_t0 = deterministic_wind_field(0.0);
    let wind_t1 = deterministic_wind_field(0.1);
    let surface_t0 = deterministic_surface_fields(0.0);
    let surface_t1 = deterministic_surface_fields(0.2);
    let met = MetTimeBracket {
        wind_t0: &wind_t0,
        wind_t1: &wind_t1,
        surface_t0: &surface_t0,
        surface_t1: &surface_t1,
        time_t0_seconds: driver.current_time_seconds(),
        time_t1_seconds: driver.current_time_seconds() + 600,
    };

    let forcing = ForwardStepForcing {
        dry_deposition_velocity_m_s: ParticleForcingField::PerParticle(
            deterministic_scalar_field(particle_count, 0.001, 0.00001, 23),
        ),
        wet_scavenging_coefficient_s_inv: ParticleForcingField::PerParticle(
            deterministic_scalar_field(particle_count, 0.0005, 0.00001, 29),
        ),
        wet_precipitating_fraction: ParticleForcingField::PerParticle(
            (0..particle_count)
                .map(|idx| ((idx % 100) as f32) / 100.0)
                .collect(),
        ),
        rho_grad_over_rho: 2.5e-4,
    };

    // Warmup
    for i in 0..warmup_steps {
        let t = Instant::now();
        pollster::block_on(driver.run_timestep(&met, &forcing))
            .expect("warmup step failed");
        eprintln!("  warmup step {i}: {:.3} ms", t.elapsed().as_secs_f64() * 1e3);
    }
    eprintln!("warmup:        {warmup_steps} steps done");

    // Measure
    let mut times_us = Vec::with_capacity(measure_steps);
    for _ in 0..measure_steps {
        let t = Instant::now();
        pollster::block_on(driver.run_timestep(&met, &forcing))
            .expect("measure step failed");
        times_us.push(t.elapsed().as_micros() as f64);
    }

    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = times_us.len() as f64;
    let mean = times_us.iter().sum::<f64>() / n;
    let median = times_us[times_us.len() / 2];
    let min = times_us[0];
    let max = times_us[times_us.len() - 1];
    let variance = times_us.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();

    eprintln!("--- results ({measure_steps} steps, {particle_count} particles) ---");
    eprintln!("mean:   {:.1} µs  ({:.3} ms)", mean, mean / 1e3);
    eprintln!("median: {:.1} µs  ({:.3} ms)", median, median / 1e3);
    eprintln!("stddev: {:.1} µs", stddev);
    eprintln!("min:    {:.1} µs", min);
    eprintln!("max:    {:.1} µs", max);

    // Machine-readable JSON to stdout
    println!(
        r#"{{"particles":{},"steps":{},"mean_us":{:.1},"median_us":{:.1},"stddev_us":{:.1},"min_us":{:.1},"max_us":{:.1}}}"#,
        particle_count, measure_steps, mean, median, stddev, min, max
    );

    std::process::exit(0);
}

fn deterministic_wind_field(phase: f32) -> WindField3D {
    let mut wind = WindField3D::zeros(GRID_NX, GRID_NY, GRID_NZ);
    for i in 0..GRID_NX {
        for j in 0..GRID_NY {
            for k in 0..GRID_NZ {
                let (x, y, z) = (i as f32, j as f32, k as f32);
                wind.u_ms[[i, j, k]] = 0.2 + 0.01 * x + 0.001 * y + 0.0005 * z + phase;
                wind.v_ms[[i, j, k]] = -0.1 + 0.008 * y - 0.0007 * x + 0.0003 * z + phase * 0.5;
                wind.w_ms[[i, j, k]] = 0.05 + 0.0009 * z - 0.0002 * y + phase * 0.25;
                wind.temperature_k[[i, j, k]] = 285.0 + 0.02 * z + phase;
                wind.specific_humidity[[i, j, k]] = 0.004 + 0.00001 * x + phase * 0.0001;
                wind.pressure_pa[[i, j, k]] = 100_000.0 - 11.0 * z + phase * 10.0;
                wind.air_density_kg_m3[[i, j, k]] = 1.2 - 0.0008 * z + phase * 0.001;
                wind.density_gradient_kg_m2[[i, j, k]] = -0.0008;
            }
        }
    }
    wind
}

fn deterministic_surface_fields(phase: f32) -> SurfaceFields {
    let mut surface = SurfaceFields::zeros(GRID_NX, GRID_NY);
    for i in 0..GRID_NX {
        for j in 0..GRID_NY {
            let (x, y) = (i as f32, j as f32);
            surface.surface_pressure_pa[[i, j]] = 101_000.0 + 0.5 * x - 0.4 * y;
            surface.u10_ms[[i, j]] = 2.0 + 0.02 * x + phase;
            surface.v10_ms[[i, j]] = 1.0 - 0.015 * y + phase * 0.5;
            surface.temperature_2m_k[[i, j]] = 289.0 + 0.01 * y + phase;
            surface.dewpoint_2m_k[[i, j]] = 284.0 + 0.01 * y;
            surface.precip_large_scale_mm_h[[i, j]] = 0.0;
            surface.precip_convective_mm_h[[i, j]] = 0.0;
            surface.sensible_heat_flux_w_m2[[i, j]] = 50.0 + 0.1 * x;
            surface.solar_radiation_w_m2[[i, j]] = 250.0 + 0.2 * y;
            surface.surface_stress_n_m2[[i, j]] = 0.25 + 0.001 * x;
            surface.friction_velocity_ms[[i, j]] = 0.35;
            surface.convective_velocity_scale_ms[[i, j]] = 0.8;
            surface.mixing_height_m[[i, j]] = 900.0 + 0.5 * y;
            surface.tropopause_height_m[[i, j]] = 10_000.0;
            surface.inv_obukhov_length_per_m[[i, j]] = -0.01 + phase * 0.001;
        }
    }
    surface
}

fn deterministic_scalar_field(count: usize, base: f32, step: f32, modulo: usize) -> Vec<f32> {
    (0..count)
        .map(|idx| base + step * (idx % modulo) as f32)
        .collect()
}
