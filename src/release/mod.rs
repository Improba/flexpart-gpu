//! Particle release manager (source term injection).
//!
//! MVP port of FLEXPART `releaseparticles.f90` intent:
//! - activate releases only within configured time windows
//! - distribute particle counts over the active window
//! - sample deterministic positions in source lon/lat/z bounds
//! - inject particles into [`crate::particles::ParticleStore`]
//! - optionally upload only changed slots to GPU storage buffers

use std::collections::BTreeMap;

use thiserror::Error;

use crate::config::ReleaseConfig;
use crate::coords::{geo_to_grid, GeoCoord, GridDomain};
use crate::gpu::{GpuBufferError, GpuContext, ParticleBuffers};
use crate::particles::{Particle, ParticleError, ParticleInit, ParticleStore, MAX_SPECIES};

/// Result of one release step at a given simulation timestamp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseStepReport {
    /// Number of particles released during this step.
    pub released_count: usize,
    /// Particle slot indices touched in [`ParticleStore`].
    pub released_slots: Vec<usize>,
}

/// Errors produced by the particle release manager.
#[derive(Debug, Error)]
pub enum ReleaseError {
    #[error("invalid timestamp `{value}`: expected 14 digits YYYYMMDDHHMMSS")]
    InvalidTimestamp { value: String },
    #[error("invalid bounds for release `{release}`: {field} min {min} > max {max}")]
    InvalidBounds {
        release: String,
        field: &'static str,
        min: f64,
        max: f64,
    },
    #[error("release `{release}` sampled outside grid domain at lon={lon}, lat={lat}")]
    OutOfGridDomain { release: String, lon: f64, lat: f64 },
    #[error("timestamp `{value}` cannot be represented as i32 simulation seconds")]
    TimeOutOfRange { value: String },
    #[error("particle release failed: {0}")]
    Particle(#[from] ParticleError),
    #[error("gpu upload failed: {0}")]
    Gpu(#[from] GpuBufferError),
}

/// Manages source-term activation and particle injection over time.
///
/// Each [`ReleaseConfig`] is tracked independently with a cumulative emitted
/// count. Calls to [`inject_for_time`](Self::inject_for_time) are deterministic:
/// the same sequence of timestamps always yields the same particle positions.
pub struct ReleaseManager {
    grid: GridDomain,
    states: Vec<ReleaseState>,
}

impl ReleaseManager {
    /// Build a release manager from parsed release configs.
    ///
    /// `grid` is used to convert lon/lat source coordinates into particle
    /// relative grid coordinates (`cell_x/cell_y + frac`).
    pub fn new(releases: &[ReleaseConfig], grid: GridDomain) -> Result<Self, ReleaseError> {
        let mut states = Vec::with_capacity(releases.len());
        for (release_idx, release) in releases.iter().enumerate() {
            let bounds = ReleaseBounds::from_config(release)?;
            let start_seconds = parse_timestamp_seconds(&release.start_time)?;
            let end_seconds = parse_timestamp_seconds(&release.end_time)?;
            states.push(ReleaseState {
                config: release.clone(),
                bounds,
                start_seconds,
                end_seconds,
                released_count: 0,
                release_point: i32::try_from(release_idx).map_err(|_| {
                    ReleaseError::TimeOutOfRange {
                        value: format!("release index {release_idx}"),
                    }
                })?,
            });
        }
        Ok(Self { grid, states })
    }

    /// Emit particles at `timestamp` and inject them into `store`.
    ///
    /// The timestamp must be normalized as `YYYYMMDDHHMMSS`.
    pub fn inject_for_time(
        &mut self,
        timestamp: &str,
        store: &mut ParticleStore,
    ) -> Result<ReleaseStepReport, ReleaseError> {
        let now_seconds = parse_timestamp_seconds(timestamp)?;
        let particle_time =
            i32::try_from(now_seconds).map_err(|_| ReleaseError::TimeOutOfRange {
                value: timestamp.to_string(),
            })?;

        let mut released_slots = Vec::new();

        for state in &mut self.states {
            let target_count = state.target_emitted_count(now_seconds);
            if target_count <= state.released_count {
                continue;
            }

            let to_emit = target_count - state.released_count;
            let mass_per_particle =
                (state.config.mass_kg / state.config.particle_count as f64) as f32;

            for offset in 0..to_emit {
                let sample_index = state.released_count + offset;
                let (lon, lat, z_m) = state
                    .bounds
                    .sample_stratified(sample_index, state.config.particle_count);
                let grid_coord = geo_to_grid(GeoCoord { lat, lon }, &self.grid);
                if !self.grid.contains(&grid_coord) {
                    return Err(ReleaseError::OutOfGridDomain {
                        release: state.config.name.clone(),
                        lon,
                        lat,
                    });
                }
                let rel = grid_coord.to_relative();

                let mut mass = [0.0_f32; MAX_SPECIES];
                mass[0] = mass_per_particle;

                let particle = Particle::new(&ParticleInit {
                    cell_x: rel.cell_x,
                    cell_y: rel.cell_y,
                    pos_x: rel.frac_x,
                    pos_y: rel.frac_y,
                    pos_z: z_m as f32,
                    mass,
                    release_point: state.release_point,
                    class: 0,
                    time: particle_time,
                });
                let slot = store.add(particle)?;
                released_slots.push(slot);
            }

            state.released_count = target_count;
        }

        Ok(ReleaseStepReport {
            released_count: released_slots.len(),
            released_slots,
        })
    }

    /// Emit particles and upload only newly touched slots to the GPU particle buffer.
    pub fn inject_and_upload_for_time(
        &mut self,
        timestamp: &str,
        store: &mut ParticleStore,
        buffers: &ParticleBuffers,
        ctx: &GpuContext,
    ) -> Result<ReleaseStepReport, ReleaseError> {
        let report = self.inject_for_time(timestamp, store)?;
        if !report.released_slots.is_empty() {
            buffers.upload_particle_slots(ctx, store, &report.released_slots)?;
        }
        Ok(report)
    }
}

struct ReleaseState {
    config: ReleaseConfig,
    bounds: ReleaseBounds,
    start_seconds: i64,
    end_seconds: i64,
    released_count: u64,
    release_point: i32,
}

impl ReleaseState {
    fn target_emitted_count(&self, now_seconds: i64) -> u64 {
        if now_seconds < self.start_seconds || now_seconds > self.end_seconds {
            return self.released_count;
        }
        if self.start_seconds == self.end_seconds {
            return self.config.particle_count;
        }
        if now_seconds == self.end_seconds {
            return self.config.particle_count;
        }

        let elapsed = (now_seconds - self.start_seconds) as u64;
        let duration = (self.end_seconds - self.start_seconds) as u64;
        ((u128::from(self.config.particle_count) * u128::from(elapsed)) / u128::from(duration))
            as u64
    }
}

struct ReleaseBounds {
    lon_min: f64,
    lon_max: f64,
    lat_min: f64,
    lat_max: f64,
    z_min: f64,
    z_max: f64,
}

impl ReleaseBounds {
    fn from_config(config: &ReleaseConfig) -> Result<Self, ReleaseError> {
        let lon_min = parse_optional_raw_f64(&config.raw, &["lon_min", "xpoint1", "xlon1"])
            .unwrap_or(config.lon);
        let lon_max = parse_optional_raw_f64(&config.raw, &["lon_max", "xpoint2", "xlon2"])
            .unwrap_or(config.lon);
        let lat_min = parse_optional_raw_f64(&config.raw, &["lat_min", "ypoint1", "ylat1"])
            .unwrap_or(config.lat);
        let lat_max = parse_optional_raw_f64(&config.raw, &["lat_max", "ypoint2", "ylat2"])
            .unwrap_or(config.lat);
        let z_min = config.z_min;
        let z_max = config.z_max;

        if lon_min > lon_max {
            return Err(ReleaseError::InvalidBounds {
                release: config.name.clone(),
                field: "longitude",
                min: lon_min,
                max: lon_max,
            });
        }
        if lat_min > lat_max {
            return Err(ReleaseError::InvalidBounds {
                release: config.name.clone(),
                field: "latitude",
                min: lat_min,
                max: lat_max,
            });
        }
        if z_min > z_max {
            return Err(ReleaseError::InvalidBounds {
                release: config.name.clone(),
                field: "height",
                min: z_min,
                max: z_max,
            });
        }

        Ok(Self {
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            z_min,
            z_max,
        })
    }

    fn sample_stratified(&self, sample_index: u64, total_samples: u64) -> (f64, f64, f64) {
        let nx = (total_samples as f64).cbrt().ceil().max(1.0) as u64;
        let ny = nx;
        let nxy = nx.saturating_mul(ny).max(1);
        let nz = total_samples.div_ceil(nxy).max(1);

        let ix = sample_index % nx;
        let iy = (sample_index / nx) % ny;
        let iz = (sample_index / nxy) % nz;

        let fx = (ix as f64 + 0.5) / nx as f64;
        let fy = (iy as f64 + 0.5) / ny as f64;
        let fz = (iz as f64 + 0.5) / nz as f64;

        let lon = self.lon_min + fx * (self.lon_max - self.lon_min);
        let lat = self.lat_min + fy * (self.lat_max - self.lat_min);
        let z_m = self.z_min + fz * (self.z_max - self.z_min);
        (lon, lat, z_m)
    }
}

fn parse_optional_raw_f64(raw: &BTreeMap<String, String>, keys: &[&str]) -> Option<f64> {
    keys.iter()
        .find_map(|key| raw.get(*key))
        .and_then(|value| value.parse::<f64>().ok())
}

fn parse_timestamp_seconds(value: &str) -> Result<i64, ReleaseError> {
    if value.len() != 14 || !value.chars().all(|c| c.is_ascii_digit()) {
        return Err(ReleaseError::InvalidTimestamp {
            value: value.to_string(),
        });
    }

    let year: i32 = value[0..4]
        .parse()
        .map_err(|_| ReleaseError::InvalidTimestamp {
            value: value.to_string(),
        })?;
    let month: u32 = value[4..6]
        .parse()
        .map_err(|_| ReleaseError::InvalidTimestamp {
            value: value.to_string(),
        })?;
    let day: u32 = value[6..8]
        .parse()
        .map_err(|_| ReleaseError::InvalidTimestamp {
            value: value.to_string(),
        })?;
    let hour: u32 = value[8..10]
        .parse()
        .map_err(|_| ReleaseError::InvalidTimestamp {
            value: value.to_string(),
        })?;
    let minute: u32 = value[10..12]
        .parse()
        .map_err(|_| ReleaseError::InvalidTimestamp {
            value: value.to_string(),
        })?;
    let second: u32 = value[12..14]
        .parse()
        .map_err(|_| ReleaseError::InvalidTimestamp {
            value: value.to_string(),
        })?;

    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return Err(ReleaseError::InvalidTimestamp {
            value: value.to_string(),
        });
    }

    let days = days_from_civil(year, month, day);
    Ok(days * 86_400 + i64::from(hour) * 3_600 + i64::from(minute) * 60 + i64::from(second))
}

/// Gregorian civil date to days since Unix epoch (1970-01-01).
fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let mut y = i64::from(year);
    let m = i64::from(month);
    let d = i64::from(day);

    if m <= 2 {
        y -= 1;
    }
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = m + if m > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn sample_release() -> ReleaseConfig {
        let mut raw = BTreeMap::new();
        raw.insert("lon_min".to_string(), "10.0".to_string());
        raw.insert("lon_max".to_string(), "12.0".to_string());
        raw.insert("lat_min".to_string(), "45.0".to_string());
        raw.insert("lat_max".to_string(), "46.0".to_string());

        ReleaseConfig {
            name: "r1".to_string(),
            start_time: "20240101010000".to_string(),
            end_time: "20240101020000".to_string(),
            lon: 11.0,
            lat: 45.5,
            z_min: 100.0,
            z_max: 200.0,
            mass_kg: 1.0,
            particle_count: 12,
            raw,
        }
    }

    fn sample_grid() -> GridDomain {
        GridDomain {
            xlon0: 0.0,
            ylat0: 0.0,
            dx: 1.0,
            dy: 1.0,
            nx: 40,
            ny: 80,
        }
    }

    #[test]
    fn no_release_outside_time_window() {
        let release = sample_release();
        let mut manager = ReleaseManager::new(&[release], sample_grid()).expect("manager");
        let mut store = ParticleStore::with_capacity(32);

        let before = manager
            .inject_for_time("20240101005959", &mut store)
            .expect("before window");
        assert_eq!(before.released_count, 0);

        let after = manager
            .inject_for_time("20240101020001", &mut store)
            .expect("after window");
        assert_eq!(after.released_count, 0);
        assert_eq!(store.active_count(), 0);
    }

    #[test]
    fn correct_count_released_inside_window() {
        let release = sample_release();
        let mut manager = ReleaseManager::new(&[release], sample_grid()).expect("manager");
        let mut store = ParticleStore::with_capacity(32);

        let quarter = manager
            .inject_for_time("20240101011500", &mut store)
            .expect("quarter point");
        assert_eq!(quarter.released_count, 3);

        let three_quarter = manager
            .inject_for_time("20240101014500", &mut store)
            .expect("three-quarter point");
        assert_eq!(three_quarter.released_count, 6);

        let end = manager
            .inject_for_time("20240101020000", &mut store)
            .expect("end point");
        assert_eq!(end.released_count, 3);
        assert_eq!(store.active_count(), 12);
    }

    #[test]
    fn released_positions_stay_within_bounds() {
        let release = sample_release();
        let mut manager = ReleaseManager::new(&[release], sample_grid()).expect("manager");
        let mut store = ParticleStore::with_capacity(32);

        let report = manager
            .inject_for_time("20240101020000", &mut store)
            .expect("emit full release");
        assert_eq!(report.released_count, 12);

        for slot in report.released_slots {
            let particle = store.get(slot).expect("valid slot");
            let lon = f64::from(particle.cell_x) + f64::from(particle.pos_x);
            let lat = f64::from(particle.cell_y) + f64::from(particle.pos_y);
            assert!((10.0..=12.0).contains(&lon), "lon out of bounds: {lon}");
            assert!((45.0..=46.0).contains(&lat), "lat out of bounds: {lat}");
            assert!(
                (100.0..=200.0).contains(&f64::from(particle.pos_z)),
                "z out of bounds: {}",
                particle.pos_z
            );
        }
    }
}
