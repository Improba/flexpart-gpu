use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use flexpart_gpu::validation::{
    write_outcome_artifacts, EtexValidationError, EtexValidationHarness,
};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/etex/scaffold/validation_fixture.json")
}

fn temp_output_dir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    path.push(format!("flexpart_gpu_{name}_{nonce}"));
    path
}

#[test]
fn etex_fixture_only_metrics_are_deterministic() {
    let harness = EtexValidationHarness::load_from_json(&fixture_path()).expect("fixture loads");
    let outcome = harness
        .run_fixture_only()
        .expect("fixture-only run succeeds");

    let conc = &outcome.report.metrics.concentration_mass_kg;
    assert_eq!(conc.sample_count, 4);
    assert!((conc.bias - 0.125).abs() < 1.0e-12);
    assert!((conc.mae - 0.375).abs() < 1.0e-12);
    assert!((conc.rmse - 0.433_012_701_892_219_3).abs() < 1.0e-12);

    let dry = &outcome.report.metrics.dry_deposition_kg_m2;
    assert_eq!(dry.sample_count, 4);
    assert!((dry.bias - 0.0025).abs() < 1.0e-7);
    assert!((dry.mae - 0.0075).abs() < 1.0e-7);
    assert!((dry.rmse - 0.008_660_254_037_844_387).abs() < 1.0e-7);

    let wet = &outcome.report.metrics.wet_deposition_kg_m2;
    assert_eq!(wet.sample_count, 4);
    assert!((wet.bias - (-0.0025)).abs() < 1.0e-7);
    assert!((wet.mae - 0.0125).abs() < 1.0e-7);
    assert!((wet.rmse - 0.013_228_756_555_322_954).abs() < 1.0e-7);

    let out_dir = temp_output_dir("etex_fixture_only");
    let artifacts = write_outcome_artifacts(&out_dir, &outcome).expect("artifacts written");
    assert!(artifacts.report_path.exists());
    assert!(artifacts.candidate_path.exists());
    assert!(artifacts.reference_path.exists());

    std::fs::remove_dir_all(&out_dir).expect("cleanup temporary directory");
}

#[test]
fn etex_pipeline_mode_runs_or_reports_missing_gpu_adapter() {
    let harness = EtexValidationHarness::load_from_json(&fixture_path()).expect("fixture loads");
    match harness.run_pipeline_synthetic() {
        Ok(outcome) => {
            assert!(
                !outcome.candidate.concentration_mass_kg.values.is_empty(),
                "pipeline mode should produce concentration output"
            );
            assert!(
                !outcome.report.pipeline_trace.is_empty(),
                "pipeline run should include timestep trace"
            );
        }
        Err(EtexValidationError::GpuUnavailable) => {
            eprintln!("No GPU adapter found - pipeline ETEX validation test skipped");
        }
        Err(err) => panic!("unexpected ETEX pipeline error: {err}"),
    }
}
