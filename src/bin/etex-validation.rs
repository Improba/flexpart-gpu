use std::path::PathBuf;

use anyhow::{anyhow, Result};
use flexpart_gpu::validation::{
    write_outcome_artifacts, EtexRunMode, EtexValidationError, EtexValidationHarness,
};

const USAGE: &str = "\
ETEX-style validation harness (I-04).

Usage:
  cargo run --bin etex-validation -- [options]
  cargo run --bin etex-validation -- --help

Options:
  --fixture <path>      Path to validation fixture JSON.
                        Default: fixtures/etex/scaffold/validation_fixture.json
  --mode <mode>         Run mode: pipeline | fixture-only
                        Default: pipeline
  --output-dir <path>   Artifact output directory.
                        Default: target/validation/etex
  --allow-no-gpu        In pipeline mode: return success when no GPU adapter is available.
  --print-json          Print `validation_report.json` content to stdout.
  -h, --help            Show this help
";

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliOptions {
    fixture: PathBuf,
    mode: EtexRunMode,
    output_dir: PathBuf,
    allow_no_gpu: bool,
    print_json: bool,
}

impl Default for CliOptions {
    fn default() -> Self {
        Self {
            fixture: PathBuf::from("fixtures/etex/scaffold/validation_fixture.json"),
            mode: EtexRunMode::PipelineSynthetic,
            output_dir: PathBuf::from("target/validation/etex"),
            allow_no_gpu: false,
            print_json: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CliCommand {
    Help,
    Run(CliOptions),
}

fn parse_mode(raw: &str) -> Result<EtexRunMode> {
    match raw {
        "pipeline" => Ok(EtexRunMode::PipelineSynthetic),
        "fixture-only" => Ok(EtexRunMode::FixtureOnly),
        other => Err(anyhow!(
            "invalid mode `{other}` (expected `pipeline` or `fixture-only`)"
        )),
    }
}

fn parse_cli_args<I>(args: I) -> Result<CliCommand>
where
    I: IntoIterator<Item = String>,
{
    let mut options = CliOptions::default();
    let mut iter = args.into_iter();

    while let Some(argument) = iter.next() {
        match argument.as_str() {
            "-h" | "--help" => return Ok(CliCommand::Help),
            "--allow-no-gpu" => options.allow_no_gpu = true,
            "--print-json" => options.print_json = true,
            "--fixture" => {
                let value = iter
                    .next()
                    .ok_or_else(|| anyhow!("missing value after --fixture"))?;
                options.fixture = PathBuf::from(value);
            }
            "--output-dir" => {
                let value = iter
                    .next()
                    .ok_or_else(|| anyhow!("missing value after --output-dir"))?;
                options.output_dir = PathBuf::from(value);
            }
            "--mode" => {
                let value = iter
                    .next()
                    .ok_or_else(|| anyhow!("missing value after --mode"))?;
                options.mode = parse_mode(&value)?;
            }
            _ if argument.starts_with("--fixture=") => {
                let (_, value) = argument
                    .split_once('=')
                    .ok_or_else(|| anyhow!("invalid --fixture argument"))?;
                options.fixture = PathBuf::from(value);
            }
            _ if argument.starts_with("--output-dir=") => {
                let (_, value) = argument
                    .split_once('=')
                    .ok_or_else(|| anyhow!("invalid --output-dir argument"))?;
                options.output_dir = PathBuf::from(value);
            }
            _ if argument.starts_with("--mode=") => {
                let (_, value) = argument
                    .split_once('=')
                    .ok_or_else(|| anyhow!("invalid --mode argument"))?;
                options.mode = parse_mode(value)?;
            }
            _ => return Err(anyhow!("unknown argument: {argument}")),
        }
    }

    Ok(CliCommand::Run(options))
}

fn run() -> Result<()> {
    env_logger::init();
    match parse_cli_args(std::env::args().skip(1).collect::<Vec<_>>())? {
        CliCommand::Help => {
            println!("{USAGE}");
            return Ok(());
        }
        CliCommand::Run(options) => {
            let harness = EtexValidationHarness::load_from_json(&options.fixture)?;
            let outcome_result = match options.mode {
                EtexRunMode::PipelineSynthetic => harness.run_pipeline_synthetic(),
                EtexRunMode::FixtureOnly => harness.run_fixture_only(),
            };

            let outcome = match outcome_result {
                Ok(outcome) => outcome,
                Err(EtexValidationError::GpuUnavailable) if options.allow_no_gpu => {
                    println!("ETEX validation skipped: no GPU adapter available");
                    return Ok(());
                }
                Err(err) => return Err(err.into()),
            };

            let artifacts = write_outcome_artifacts(&options.output_dir, &outcome)?;
            println!("ETEX validation completed");
            println!("scenario: {}", outcome.report.scenario_id);
            println!("mode: {:?}", outcome.report.run_mode);
            println!("report: {}", artifacts.report_path.display());
            println!("candidate: {}", artifacts.candidate_path.display());
            println!("reference: {}", artifacts.reference_path.display());

            if options.print_json {
                let report_json = std::fs::read_to_string(&artifacts.report_path)?;
                println!("{report_json}");
            }
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ETEX validation failed");
        eprintln!("{error:#}");
        eprintln!("{USAGE}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_defaults() {
        let parsed = parse_cli_args(Vec::<String>::new()).expect("defaults parse");
        assert_eq!(parsed, CliCommand::Run(CliOptions::default()));
    }

    #[test]
    fn parse_fixture_only_mode() {
        let parsed = parse_cli_args(vec![
            "--mode".to_string(),
            "fixture-only".to_string(),
            "--print-json".to_string(),
        ])
        .expect("arguments parse");
        let mut expected = CliOptions::default();
        expected.mode = EtexRunMode::FixtureOnly;
        expected.print_json = true;
        assert_eq!(parsed, CliCommand::Run(expected));
    }
}
