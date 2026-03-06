use approx::assert_relative_eq;

use flexpart_gpu::gpu::{dispatch_convective_mixing_gpu, GpuError, ParticleBuffers};
use flexpart_gpu::particles::{Particle, ParticleInit, MAX_SPECIES};
use flexpart_gpu::physics::{
    apply_convective_mixing_to_particles_cpu, apply_redistribution_matrix,
    build_simplified_convection_chain, SimplifiedEmanuelInputs,
};

fn particle_at(z_m: f32, mass0: f32) -> Particle {
    let mut mass = [0.0_f32; MAX_SPECIES];
    mass[0] = mass0;
    Particle::new(&ParticleInit {
        cell_x: 0,
        cell_y: 0,
        pos_x: 0.5,
        pos_y: 0.5,
        pos_z: z_m,
        mass,
        release_point: 0,
        class: 0,
        time: 0,
    })
}

#[test]
fn test_convection_chain_matrix_correctness_and_gpu_cpu_parity() {
    let inputs = SimplifiedEmanuelInputs {
        level_interfaces_m: vec![0.0, 400.0, 1_000.0, 2_000.0, 3_500.0, 5_000.0],
        convective_precip_mm_h: 10.0,
        convective_velocity_scale_m_s: 2.4,
        boundary_layer_height_m: 850.0,
        cape_override_j_kg: Some(850.0),
    };
    let (column, matrix) =
        build_simplified_convection_chain(inputs, 600.0).expect("convection chain should build");

    let source_profile = vec![16.0, 5.0, 1.5, 0.4, 0.1];
    let destination_profile =
        apply_redistribution_matrix(&matrix, &source_profile).expect("matrix apply should succeed");
    let source_total: f32 = source_profile.iter().sum();
    let destination_total: f32 = destination_profile.iter().sum();
    assert_relative_eq!(
        source_total,
        destination_total,
        epsilon = 1.0e-5,
        max_relative = 1.0e-5
    );
    assert!(destination_profile[2] > source_profile[2]);
    let source_upper: f32 = source_profile[2..].iter().sum();
    let destination_upper: f32 = destination_profile[2..].iter().sum();
    assert!(destination_upper > source_upper);

    let mut cpu_particles = vec![
        particle_at(100.0, 1.0),
        particle_at(700.0, 2.0),
        particle_at(1_400.0, 3.0),
        particle_at(2_700.0, 4.0),
    ];
    let mut gpu_particles = cpu_particles.clone();
    apply_convective_mixing_to_particles_cpu(
        &mut cpu_particles,
        &matrix,
        &column.level_interfaces_m,
        &column.level_centers_m,
    )
    .expect("cpu particle convection should succeed");

    let ctx = match pollster::block_on(flexpart_gpu::gpu::GpuContext::new()) {
        Ok(ctx) => ctx,
        Err(GpuError::NoAdapter) => return,
        Err(err) => panic!("unexpected GPU init error: {err}"),
    };
    let gpu_buffers = ParticleBuffers::from_particles(&ctx, &gpu_particles);
    dispatch_convective_mixing_gpu(&ctx, &gpu_buffers, &column.level_interfaces_m, &matrix)
        .expect("gpu convection dispatch should succeed");
    gpu_particles = pollster::block_on(gpu_buffers.download_particles(&ctx))
        .expect("gpu particle readback should succeed");

    for (cpu, gpu) in cpu_particles.iter().zip(gpu_particles.iter()) {
        assert_relative_eq!(
            gpu.pos_z,
            cpu.pos_z,
            epsilon = 5.0e-5,
            max_relative = 5.0e-5
        );
        assert_relative_eq!(
            gpu.mass[0],
            cpu.mass[0],
            epsilon = 1.0e-7,
            max_relative = 1.0e-7
        );
    }
}
