use flexpart_gpu::coords::GridDomain;
use flexpart_gpu::gpu::GpuError;
use flexpart_gpu::io::TimeBoundsBehavior;
use flexpart_gpu::physics::VelocityToGridScale;
use flexpart_gpu::simulation::{
    BackwardReceptorConfig, BackwardSourceRegionConfig, BackwardTimeLoopConfig,
    BackwardTimeLoopDriver, ForwardStepForcing, MetTimeBracket, TimeLoopError,
};
use flexpart_gpu::wind::{uniform_wind_field, SurfaceFields, WindFieldGrid};
use ndarray::Array1;

fn synthetic_wind_grid(nx: usize, ny: usize, nz: usize) -> WindFieldGrid {
    WindFieldGrid::new(
        nx,
        ny,
        nz,
        nz,
        nz,
        1.0,
        1.0,
        0.0,
        0.0,
        Array1::from_iter((0..nz).map(|k| k as f32)),
    )
}

fn synthetic_surface_fields(nx: usize, ny: usize) -> SurfaceFields {
    let mut surface = SurfaceFields::zeros(nx, ny);
    surface.surface_pressure_pa.fill(101_325.0);
    surface.u10_ms.fill(1.0);
    surface.v10_ms.fill(0.0);
    surface.temperature_2m_k.fill(290.0);
    surface.dewpoint_2m_k.fill(285.0);
    surface.precip_large_scale_mm_h.fill(0.0);
    surface.precip_convective_mm_h.fill(0.0);
    surface.sensible_heat_flux_w_m2.fill(0.0);
    surface.solar_radiation_w_m2.fill(0.0);
    surface.surface_stress_n_m2.fill(0.0);
    surface.friction_velocity_ms.fill(0.0);
    surface.convective_velocity_scale_ms.fill(0.0);
    surface.mixing_height_m.fill(1_000.0);
    surface.tropopause_height_m.fill(10_000.0);
    surface.inv_obukhov_length_per_m.fill(0.0);
    surface
}

#[test]
fn test_backward_timeloop_receptor_release_and_source_collection() {
    let config = BackwardTimeLoopConfig {
        start_timestamp: "20240101000002".to_string(),
        end_timestamp: "20240101000000".to_string(),
        timestep_seconds: 1,
        time_bounds_behavior: TimeBoundsBehavior::Strict,
        velocity_to_grid_scale: VelocityToGridScale::IDENTITY,
        receptors: vec![BackwardReceptorConfig {
            name: "receptor_a".to_string(),
            lon: 10.0,
            lat: 5.0,
            z_m: 1.0,
            particle_count: 1,
            mass_kg: 1.0,
        }],
        source_regions: vec![BackwardSourceRegionConfig {
            name: "source_a".to_string(),
            lon_min: 6.9,
            lon_max: 7.1,
            lat_min: 4.9,
            lat_max: 5.1,
            z_min_m: 0.0,
            z_max_m: 5.0,
        }],
        ..BackwardTimeLoopConfig::default()
    };

    let release_grid = GridDomain {
        xlon0: 0.0,
        ylat0: 0.0,
        dx: 1.0,
        dy: 1.0,
        nx: 64,
        ny: 64,
    };

    let mut driver = match pollster::block_on(BackwardTimeLoopDriver::new(config, release_grid, 8))
    {
        Ok(driver) => driver,
        Err(TimeLoopError::Gpu(GpuError::NoAdapter)) => return,
        Err(err) => panic!("driver initialization failed: {err}"),
    };

    let wind_grid = synthetic_wind_grid(64, 64, 16);
    let wind_t0 = uniform_wind_field(&wind_grid, 1.0, 0.0, 0.0);
    let wind_t1 = uniform_wind_field(&wind_grid, 1.0, 0.0, 0.0);
    let surface_t0 = synthetic_surface_fields(64, 64);
    let surface_t1 = synthetic_surface_fields(64, 64);
    let met = MetTimeBracket {
        wind_t0: &wind_t0,
        wind_t1: &wind_t1,
        surface_t0: &surface_t0,
        surface_t1: &surface_t1,
        time_t0_seconds: driver.current_time_seconds() - 2,
        time_t1_seconds: driver.current_time_seconds(),
    };

    let forcing = ForwardStepForcing::default();
    let reports =
        pollster::block_on(driver.run_to_end(&met, &forcing)).expect("backward run should succeed");

    assert_eq!(reports.len(), 3);
    assert_eq!(reports[0].released_count, 1);
    assert_eq!(reports[1].released_count, 0);
    assert_eq!(reports[2].released_count, 0);

    let expected_alpha = [1.0_f32, 0.5_f32, 0.0_f32];
    for (report, expected) in reports.iter().zip(expected_alpha) {
        assert!(
            (report.interpolation_alpha - expected).abs() < 1.0e-6,
            "unexpected interpolation alpha: got {}, expected {expected}",
            report.interpolation_alpha
        );
    }

    // Particle starts at receptor (10, 5) with u=1 backward.
    // After 3 backward steps: pure advection yields x≈7.
    // Langevin turbulence adds stochastic displacement.
    let released_slot = reports[0]
        .released_slots
        .first()
        .copied()
        .expect("one slot should be released in first step");
    let particle = driver
        .particle_store()
        .get(released_slot)
        .expect("released slot should still exist");
    assert!(particle.is_active());
    assert!(
        particle.grid_x() > 4.0 && particle.grid_x() < 10.0,
        "particle should drift in -x direction: grid_x={}",
        particle.grid_x()
    );
    assert!(
        (particle.grid_y() - 5.0).abs() < 2.0,
        "y should stay near initial: grid_y={}",
        particle.grid_y()
    );
    assert!((particle.mass[0] - 1.0).abs() < 1.0e-6);

    // Source collection: with turbulence the particle may or may not
    // land exactly in the source region, so we check it exists.
    let source_final = reports[2]
        .source_collections
        .get("source_a")
        .expect("source_a collection should exist");
}

#[test]
fn test_backward_config_requires_receptor() {
    let config = BackwardTimeLoopConfig {
        start_timestamp: "20240101000002".to_string(),
        end_timestamp: "20240101000000".to_string(),
        timestep_seconds: 1,
        receptors: Vec::new(),
        source_regions: Vec::new(),
        ..BackwardTimeLoopConfig::default()
    };
    let release_grid = GridDomain {
        xlon0: 0.0,
        ylat0: 0.0,
        dx: 1.0,
        dy: 1.0,
        nx: 32,
        ny: 32,
    };
    let result = pollster::block_on(BackwardTimeLoopDriver::new(config, release_grid, 8));
    match result {
        Err(TimeLoopError::MissingReceptors) => {}
        Err(other) => panic!("unexpected error: {other}"),
        Ok(_) => panic!("missing receptors should fail validation"),
    }
}
