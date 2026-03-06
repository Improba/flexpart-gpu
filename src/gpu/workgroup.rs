//! Workgroup-size runtime configuration and auto-tuning utilities (O-02).
//!
//! This module provides:
//! - runtime workgroup-size resolution for key kernels,
//! - deterministic benchmark-driven selection logic,
//! - optional cache persistence keyed by GPU adapter/backend.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::GpuContext;

const DEFAULT_WORKGROUP_SIZE_X: u32 = 64;
const SHADER_WORKGROUP_TOKEN: &str = "__WORKGROUP_SIZE_X__";
const CACHE_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkgroupKernel {
    Advection,
    HannaParams,
    Langevin,
    DryDeposition,
    WetDeposition,
    ConcentrationGridding,
    PblReflection,
}

impl WorkgroupKernel {
    pub const KEY_KERNELS: [Self; 6] = [
        Self::Advection,
        Self::HannaParams,
        Self::Langevin,
        Self::DryDeposition,
        Self::WetDeposition,
        Self::ConcentrationGridding,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Advection => "advection",
            Self::HannaParams => "hanna_params",
            Self::Langevin => "langevin",
            Self::DryDeposition => "dry_deposition",
            Self::WetDeposition => "wet_deposition",
            Self::ConcentrationGridding => "concentration_gridding",
            Self::PblReflection => "pbl_reflection",
        }
    }

    #[must_use]
    pub const fn env_suffix(self) -> &'static str {
        match self {
            Self::Advection => "ADVECTION",
            Self::HannaParams => "HANNA_PARAMS",
            Self::Langevin => "LANGEVIN",
            Self::DryDeposition => "DRY_DEPOSITION",
            Self::WetDeposition => "WET_DEPOSITION",
            Self::ConcentrationGridding => "CONCENTRATION_GRIDDING",
            Self::PblReflection => "PBL_REFLECTION",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkgroupSizeConfig {
    pub sizes: BTreeMap<WorkgroupKernel, u32>,
}

impl Default for WorkgroupSizeConfig {
    fn default() -> Self {
        let mut sizes = BTreeMap::new();
        for kernel in WorkgroupKernel::KEY_KERNELS {
            sizes.insert(kernel, DEFAULT_WORKGROUP_SIZE_X);
        }
        Self { sizes }
    }
}

impl WorkgroupSizeConfig {
    #[must_use]
    pub fn for_kernel(&self, kernel: WorkgroupKernel) -> u32 {
        self.sizes
            .get(&kernel)
            .copied()
            .unwrap_or(DEFAULT_WORKGROUP_SIZE_X)
            .max(1)
    }

    pub fn set_for_kernel(&mut self, kernel: WorkgroupKernel, size_x: u32) {
        self.sizes.insert(kernel, size_x.max(1));
    }

    fn clamp_to_limits(&mut self, limits: &wgpu::Limits) {
        let max_x = limits.max_compute_workgroup_size_x.max(1);
        let max_invocations = limits.max_compute_invocations_per_workgroup.max(1);
        let upper = max_x.min(max_invocations);
        for value in self.sizes.values_mut() {
            *value = (*value).clamp(1, upper);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkgroupAutoTuneOptions {
    pub candidate_sizes: Vec<u32>,
    pub warmup_runs: usize,
    pub measurement_runs: usize,
    pub deterministic_mode: bool,
}

impl Default for WorkgroupAutoTuneOptions {
    fn default() -> Self {
        Self {
            candidate_sizes: vec![32, 64, 128, 256],
            warmup_runs: 2,
            measurement_runs: 7,
            deterministic_mode: true,
        }
    }
}

impl WorkgroupAutoTuneOptions {
    #[must_use]
    pub fn from_env() -> Self {
        let mut options = Self::default();
        if let Some(candidates) = parse_env_u32_csv("FLEXPART_WG_AUTOTUNE_CANDIDATES") {
            options.candidate_sizes = candidates;
        }
        if let Some(value) = parse_env_usize("FLEXPART_WG_AUTOTUNE_WARMUP_RUNS") {
            options.warmup_runs = value;
        }
        if let Some(value) = parse_env_usize("FLEXPART_WG_AUTOTUNE_MEASUREMENT_RUNS") {
            options.measurement_runs = value.max(1);
        }
        if let Some(value) = parse_env_bool("FLEXPART_WG_AUTOTUNE_DETERMINISTIC") {
            options.deterministic_mode = value;
        }
        options
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelAutoTuneResult {
    pub preferred_workgroup_size: u32,
    pub score_ns_by_candidate: BTreeMap<u32, u128>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkgroupAutoTuneReport {
    pub version: u32,
    pub created_unix_seconds: u64,
    pub backend: String,
    pub adapter_name: String,
    pub deterministic_mode: bool,
    pub candidate_sizes: Vec<u32>,
    pub kernel_results: BTreeMap<WorkgroupKernel, KernelAutoTuneResult>,
}

impl WorkgroupAutoTuneReport {
    #[must_use]
    pub fn to_config(&self) -> WorkgroupSizeConfig {
        let mut config = WorkgroupSizeConfig::default();
        for (kernel, result) in &self.kernel_results {
            config.set_for_kernel(*kernel, result.preferred_workgroup_size);
        }
        config
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct WorkgroupAutoTuneCacheFile {
    version: u32,
    reports: BTreeMap<String, WorkgroupAutoTuneReport>,
}

impl Default for WorkgroupAutoTuneCacheFile {
    fn default() -> Self {
        Self {
            version: CACHE_FORMAT_VERSION,
            reports: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Error)]
pub enum WorkgroupAutoTuneError {
    #[error("no valid workgroup-size candidates after device-limit filtering")]
    NoCandidateSizes,
    #[error(
        "no timing samples produced for kernel `{kernel}` and candidate workgroup size {candidate}"
    )]
    EmptySamples {
        kernel: &'static str,
        candidate: u32,
    },
    #[error(
        "benchmark callback failed for kernel `{kernel}` and candidate {candidate}: {message}"
    )]
    BenchmarkFailed {
        kernel: &'static str,
        candidate: u32,
        message: String,
    },
    #[error("failed to create auto-tune cache directory `{path}`: {source}")]
    CreateCacheDirectory {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to read auto-tune cache `{path}`: {source}")]
    ReadCache {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse auto-tune cache `{path}`: {source}")]
    ParseCache {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to serialize auto-tune cache: {source}")]
    SerializeCache {
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to write auto-tune cache `{path}`: {source}")]
    WriteCache {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

fn parse_env_bool(name: &str) -> Option<bool> {
    let value = env::var(name).ok()?;
    let lowered = value.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn parse_env_usize(name: &str) -> Option<usize> {
    env::var(name).ok()?.trim().parse::<usize>().ok()
}

fn parse_env_u32(name: &str) -> Option<u32> {
    env::var(name).ok()?.trim().parse::<u32>().ok()
}

fn parse_env_u32_csv(name: &str) -> Option<Vec<u32>> {
    let raw = env::var(name).ok()?;
    let parsed: Vec<u32> = raw
        .split(',')
        .filter_map(|value| value.trim().parse::<u32>().ok())
        .collect();
    if parsed.is_empty() {
        None
    } else {
        Some(parsed)
    }
}

fn runtime_config_overrides() -> WorkgroupSizeConfig {
    let mut config = WorkgroupSizeConfig::default();
    if let Some(default_size) = parse_env_u32("FLEXPART_WG_SIZE_DEFAULT") {
        for kernel in WorkgroupKernel::KEY_KERNELS {
            config.set_for_kernel(kernel, default_size);
        }
    }
    for kernel in WorkgroupKernel::KEY_KERNELS {
        let env_name = format!("FLEXPART_WG_SIZE_{}", kernel.env_suffix());
        if let Some(size_x) = parse_env_u32(&env_name) {
            config.set_for_kernel(kernel, size_x);
        }
    }
    config
}

fn adapter_cache_key(ctx: &GpuContext) -> String {
    format!("{:?}::{}", ctx.backend(), ctx.device_name())
}

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn cache_enabled() -> bool {
    !parse_env_bool("FLEXPART_WG_AUTOTUNE_DISABLE_CACHE").unwrap_or(false)
}

fn candidate_sizes_for_device(
    candidate_sizes: &[u32],
    limits: &wgpu::Limits,
) -> Result<Vec<u32>, WorkgroupAutoTuneError> {
    let mut unique = BTreeSet::new();
    let max_x = limits.max_compute_workgroup_size_x.max(1);
    let max_invocations = limits.max_compute_invocations_per_workgroup.max(1);
    let upper = max_x.min(max_invocations);
    for size in candidate_sizes {
        if *size > 0 && *size <= upper {
            unique.insert(*size);
        }
    }
    let filtered: Vec<u32> = unique.into_iter().collect();
    if filtered.is_empty() {
        return Err(WorkgroupAutoTuneError::NoCandidateSizes);
    }
    Ok(filtered)
}

fn score_samples(samples: &[u128], deterministic_mode: bool) -> u128 {
    if deterministic_mode {
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        sorted[sorted.len() / 2]
    } else {
        let sum = samples.iter().copied().sum::<u128>();
        sum / u128::try_from(samples.len()).unwrap_or(1)
    }
}

pub fn select_preferred_workgroup_size(
    score_ns_by_candidate: &BTreeMap<u32, u128>,
    deterministic_mode: bool,
) -> Result<u32, WorkgroupAutoTuneError> {
    let mut best: Option<(u32, u128)> = None;
    for (candidate, score) in score_ns_by_candidate {
        best = match best {
            None => Some((*candidate, *score)),
            Some((best_candidate, best_score)) => {
                if *score < best_score {
                    Some((*candidate, *score))
                } else if *score == best_score && deterministic_mode && *candidate < best_candidate
                {
                    Some((*candidate, *score))
                } else {
                    Some((best_candidate, best_score))
                }
            }
        };
    }
    best.map(|(candidate, _)| candidate)
        .ok_or(WorkgroupAutoTuneError::NoCandidateSizes)
}

pub fn auto_tune_key_kernels<F>(
    ctx: &GpuContext,
    options: &WorkgroupAutoTuneOptions,
    mut benchmark: F,
) -> Result<WorkgroupAutoTuneReport, WorkgroupAutoTuneError>
where
    F: FnMut(WorkgroupKernel, u32) -> Result<Duration, String>,
{
    let candidates = candidate_sizes_for_device(&options.candidate_sizes, &ctx.device.limits())?;
    let mut kernel_results = BTreeMap::new();

    for kernel in WorkgroupKernel::KEY_KERNELS {
        let mut scores = BTreeMap::new();
        for candidate in &candidates {
            for _ in 0..options.warmup_runs {
                benchmark(kernel, *candidate).map_err(|message| {
                    WorkgroupAutoTuneError::BenchmarkFailed {
                        kernel: kernel.as_str(),
                        candidate: *candidate,
                        message,
                    }
                })?;
            }

            let mut samples = Vec::with_capacity(options.measurement_runs);
            for _ in 0..options.measurement_runs.max(1) {
                let elapsed = benchmark(kernel, *candidate).map_err(|message| {
                    WorkgroupAutoTuneError::BenchmarkFailed {
                        kernel: kernel.as_str(),
                        candidate: *candidate,
                        message,
                    }
                })?;
                samples.push(elapsed.as_nanos());
            }
            if samples.is_empty() {
                return Err(WorkgroupAutoTuneError::EmptySamples {
                    kernel: kernel.as_str(),
                    candidate: *candidate,
                });
            }
            scores.insert(
                *candidate,
                score_samples(&samples, options.deterministic_mode),
            );
        }

        let preferred = select_preferred_workgroup_size(&scores, options.deterministic_mode)?;
        kernel_results.insert(
            kernel,
            KernelAutoTuneResult {
                preferred_workgroup_size: preferred,
                score_ns_by_candidate: scores,
            },
        );
    }

    Ok(WorkgroupAutoTuneReport {
        version: CACHE_FORMAT_VERSION,
        created_unix_seconds: now_unix_seconds(),
        backend: format!("{:?}", ctx.backend()),
        adapter_name: ctx.device_name().to_string(),
        deterministic_mode: options.deterministic_mode,
        candidate_sizes: candidates,
        kernel_results,
    })
}

pub fn default_autotune_cache_path() -> Option<PathBuf> {
    if let Ok(path) = env::var("FLEXPART_WG_AUTOTUNE_CACHE") {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed));
        }
    }
    if let Ok(cache_home) = env::var("XDG_CACHE_HOME") {
        let trimmed = cache_home.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed).join("flexpart-gpu/workgroup-autotune.json"));
        }
    }
    env::var("HOME")
        .ok()
        .map(|home| PathBuf::from(home).join(".cache/flexpart-gpu/workgroup-autotune.json"))
}

pub fn save_autotune_report(
    path: &Path,
    report: &WorkgroupAutoTuneReport,
) -> Result<(), WorkgroupAutoTuneError> {
    let mut cache = if path.exists() {
        let raw = fs::read_to_string(path).map_err(|source| WorkgroupAutoTuneError::ReadCache {
            path: path.to_path_buf(),
            source,
        })?;
        serde_json::from_str::<WorkgroupAutoTuneCacheFile>(&raw).map_err(|source| {
            WorkgroupAutoTuneError::ParseCache {
                path: path.to_path_buf(),
                source,
            }
        })?
    } else {
        WorkgroupAutoTuneCacheFile::default()
    };

    cache.version = CACHE_FORMAT_VERSION;
    cache.reports.insert(
        format!("{}::{}", report.backend, report.adapter_name),
        report.clone(),
    );

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| {
            WorkgroupAutoTuneError::CreateCacheDirectory {
                path: parent.to_path_buf(),
                source,
            }
        })?;
    }

    let payload = serde_json::to_string_pretty(&cache)
        .map_err(|source| WorkgroupAutoTuneError::SerializeCache { source })?;
    fs::write(path, payload).map_err(|source| WorkgroupAutoTuneError::WriteCache {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(())
}

pub fn save_autotune_report_default(
    report: &WorkgroupAutoTuneReport,
) -> Result<Option<PathBuf>, WorkgroupAutoTuneError> {
    let Some(path) = default_autotune_cache_path() else {
        return Ok(None);
    };
    save_autotune_report(&path, report)?;
    Ok(Some(path))
}

fn load_cached_autotune_report(
    path: &Path,
    ctx: &GpuContext,
) -> Result<Option<WorkgroupAutoTuneReport>, WorkgroupAutoTuneError> {
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(path).map_err(|source| WorkgroupAutoTuneError::ReadCache {
        path: path.to_path_buf(),
        source,
    })?;
    let cache = serde_json::from_str::<WorkgroupAutoTuneCacheFile>(&raw).map_err(|source| {
        WorkgroupAutoTuneError::ParseCache {
            path: path.to_path_buf(),
            source,
        }
    })?;
    let key = adapter_cache_key(ctx);
    Ok(cache.reports.get(&key).cloned())
}

fn resolved_runtime_workgroup_config(ctx: &GpuContext) -> WorkgroupSizeConfig {
    let mut config = runtime_config_overrides();
    if cache_enabled() {
        if let Some(path) = default_autotune_cache_path() {
            if let Ok(Some(report)) = load_cached_autotune_report(&path, ctx) {
                config = report.to_config();
                let overrides = runtime_config_overrides();
                for kernel in WorkgroupKernel::KEY_KERNELS {
                    if let Some(value) = overrides.sizes.get(&kernel) {
                        config.set_for_kernel(kernel, *value);
                    }
                }
            }
        }
    }
    config.clamp_to_limits(&ctx.device.limits());
    config
}

static RUNTIME_WORKGROUP_CONFIG: OnceLock<WorkgroupSizeConfig> = OnceLock::new();
static OVERRIDE_WORKGROUP_CONFIG: OnceLock<RwLock<BTreeMap<WorkgroupKernel, u32>>> =
    OnceLock::new();

fn override_map() -> &'static RwLock<BTreeMap<WorkgroupKernel, u32>> {
    OVERRIDE_WORKGROUP_CONFIG.get_or_init(|| RwLock::new(BTreeMap::new()))
}

#[must_use]
pub fn runtime_workgroup_size(ctx: &GpuContext, kernel: WorkgroupKernel) -> u32 {
    if let Ok(guard) = override_map().read() {
        if let Some(size_x) = guard.get(&kernel) {
            return (*size_x).max(1);
        }
    }
    let config = RUNTIME_WORKGROUP_CONFIG.get_or_init(|| resolved_runtime_workgroup_config(ctx));
    config.for_kernel(kernel)
}

pub struct ScopedWorkgroupOverride {
    kernel: WorkgroupKernel,
}

impl ScopedWorkgroupOverride {
    #[must_use]
    pub fn new(kernel: WorkgroupKernel, workgroup_size_x: u32) -> Self {
        if let Ok(mut guard) = override_map().write() {
            guard.insert(kernel, workgroup_size_x.max(1));
        }
        Self { kernel }
    }
}

impl Drop for ScopedWorkgroupOverride {
    fn drop(&mut self) {
        if let Ok(mut guard) = override_map().write() {
            guard.remove(&self.kernel);
        }
    }
}

#[must_use]
pub fn render_shader_with_workgroup_size(shader_template: &str, workgroup_size_x: u32) -> String {
    shader_template.replace(SHADER_WORKGROUP_TOKEN, &workgroup_size_x.max(1).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_selection_breaks_ties_with_smaller_candidate() {
        let mut scores = BTreeMap::new();
        scores.insert(64, 100_u128);
        scores.insert(128, 100_u128);
        scores.insert(256, 120_u128);
        let preferred =
            select_preferred_workgroup_size(&scores, true).expect("selection should succeed");
        assert_eq!(preferred, 64);
    }

    #[test]
    fn non_deterministic_selection_keeps_first_tie_candidate() {
        let mut scores = BTreeMap::new();
        scores.insert(128, 80_u128);
        scores.insert(256, 80_u128);
        let preferred =
            select_preferred_workgroup_size(&scores, false).expect("selection should succeed");
        assert_eq!(preferred, 128);
    }

    #[test]
    fn median_scoring_is_stable_for_outliers() {
        let samples = vec![50_u128, 49, 10_000, 51, 48];
        assert_eq!(score_samples(&samples, true), 50);
        assert_eq!(score_samples(&samples, false), 2039);
    }

    #[test]
    fn candidate_filter_respects_limits() {
        let limits = wgpu::Limits {
            max_compute_workgroup_size_x: 256,
            max_compute_invocations_per_workgroup: 128,
            ..wgpu::Limits::default()
        };
        let filtered = candidate_sizes_for_device(&[0, 32, 64, 256, 512], &limits)
            .expect("filtering should retain candidates");
        assert_eq!(filtered, vec![32, 64]);
    }
}
