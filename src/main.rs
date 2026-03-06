use anyhow::Result;
use log::info;

fn main() -> Result<()> {
    env_logger::init();
    info!("FLEXPART-GPU v{}", env!("CARGO_PKG_VERSION"));

    let gpu_context = pollster::block_on(flexpart_gpu::gpu::GpuContext::new())?;
    info!("GPU device: {}", gpu_context.device_name());

    Ok(())
}
