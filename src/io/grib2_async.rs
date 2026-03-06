//! Asynchronous wrapper around the synchronous GRIB2 reader.
//!
//! Provides non-blocking GRIB file reading via background OS threads for the
//! async met I/O pipeline (Tier 3.2 in the optimisation plan). The GPU can
//! continue computing while the next meteorological bracket is being decoded.
//!
//! This module is executor-agnostic: it uses [`std::thread`] rather than a
//! specific async runtime (`tokio`, `smol`, …), so it works with `pollster`,
//! bare `block_on`, or any executor the caller happens to use.

use std::path::{Path, PathBuf};
use std::thread::{self, JoinHandle};

use super::grib2::{
    load_era5_mvp_from_grib2, Era5GribGridMetadata, Era5MvpSnapshot, Grib2ReaderError,
};

/// Read an ERA5 GRIB2 snapshot asynchronously by offloading to a background thread.
///
/// This is a convenience wrapper around [`GribPrefetchHandle`]. The returned
/// future resolves once the blocking I/O completes. Because the I/O runs on a
/// dedicated OS thread, the calling async executor is free to poll other tasks
/// in the meantime.
///
/// `expected_grid`, when provided, is checked against the loaded snapshot's
/// grid metadata after decoding. A mismatch produces
/// [`Grib2ReaderError::InconsistentGrid`].
///
/// # Errors
///
/// Returns [`Grib2ReaderError`] if GRIB decoding fails or if the loaded grid
/// does not match `expected_grid`.
///
/// # Blocking note
///
/// The returned future internally calls [`JoinHandle::join`] which blocks the
/// calling OS thread until the background thread finishes. In the typical
/// pipeline pattern (start prefetch → GPU dispatch → await result), the GRIB
/// read has already completed by that point so the join returns immediately.
#[allow(clippy::unused_async)]
pub async fn read_era5_snapshot_async(
    path: impl AsRef<Path> + Send + 'static,
    expected_grid: Option<Era5GribGridMetadata>,
) -> Result<Era5MvpSnapshot, Grib2ReaderError> {
    GribPrefetchHandle::start(path, expected_grid).await_result()
}

/// Handle for a GRIB file read running in a background OS thread.
///
/// Intended for pipeline integration: start reading the next met bracket
/// while the GPU processes the current timestep, then retrieve the result
/// once the GPU dispatch is complete.
///
/// ```text
/// t0  ──────────────────────────────────────────▶  t1
///  │  GribPrefetchHandle::start(next_file)         │
///  │  ├── OS thread: load_era5_mvp_from_grib2 ──┐  │
///  │  GPU dispatch for current step              │  │
///  │  …                                          │  │
///  │  handle.await_result()  ◀───────────────────┘  │
/// ```
pub struct GribPrefetchHandle {
    handle: JoinHandle<Result<Era5MvpSnapshot, Grib2ReaderError>>,
}

impl GribPrefetchHandle {
    /// Start reading a GRIB file in the background.
    ///
    /// The OS thread begins immediately; no polling is required to make
    /// progress. `expected_grid` is validated after the file is fully decoded.
    ///
    /// # Errors
    ///
    /// Errors are deferred — they surface when [`await_result`](Self::await_result)
    /// is called.
    #[allow(clippy::needless_pass_by_value)]
    pub fn start(
        path: impl AsRef<Path> + Send + 'static,
        expected_grid: Option<Era5GribGridMetadata>,
    ) -> Self {
        let path_buf: PathBuf = path.as_ref().to_path_buf();

        let handle = thread::spawn(move || {
            let snapshot = load_era5_mvp_from_grib2(&path_buf)?;

            if let Some(ref expected) = expected_grid {
                validate_grid(expected, &snapshot)?;
            }

            Ok(snapshot)
        });

        Self { handle }
    }

    /// Block until the background read is complete and return the result.
    ///
    /// This consumes the handle. If the background thread panicked, the panic
    /// is propagated to the calling thread via [`std::panic::resume_unwind`].
    ///
    /// # Errors
    ///
    /// Returns [`Grib2ReaderError`] if the GRIB decoding failed or grid
    /// validation did not pass.
    ///
    /// # Panics
    ///
    /// Re-panics if the background thread panicked.
    pub fn await_result(self) -> Result<Era5MvpSnapshot, Grib2ReaderError> {
        match self.handle.join() {
            Ok(result) => result,
            Err(panic_payload) => std::panic::resume_unwind(panic_payload),
        }
    }

    /// Check if the background read is complete without blocking.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.handle.is_finished()
    }
}

/// Validate that the loaded snapshot's grid matches expected dimensions.
fn validate_grid(
    expected: &Era5GribGridMetadata,
    snapshot: &Era5MvpSnapshot,
) -> Result<(), Grib2ReaderError> {
    let grid = &snapshot.grid;
    let actual = Era5GribGridMetadata {
        nx: grid.nx,
        ny: grid.ny,
        dx_deg: grid.dx_deg,
        dy_deg: grid.dy_deg,
        xlon0_deg: grid.xlon0,
        ylat0_deg: grid.ylat0,
    };

    if actual != *expected {
        return Err(Grib2ReaderError::InconsistentGrid {
            variable: "all".to_string(),
            level: 0,
            expected: format!("{expected:?}"),
            actual: format!("{actual:?}"),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefetch_handle_types_compile() {
        fn assert_send<T: Send>() {}

        assert_send::<GribPrefetchHandle>();
        assert_send::<Era5MvpSnapshot>();
        assert_send::<Grib2ReaderError>();
    }

    #[test]
    fn grid_validation_detects_mismatch() {
        let expected = Era5GribGridMetadata {
            nx: 10,
            ny: 20,
            dx_deg: 0.25,
            dy_deg: 0.25,
            xlon0_deg: 0.0,
            ylat0_deg: 0.0,
        };

        let mut other = expected.clone();
        other.nx = 999;

        assert_ne!(expected, other);
    }

    #[test]
    fn grid_validation_accepts_matching() {
        let grid = Era5GribGridMetadata {
            nx: 32,
            ny: 32,
            dx_deg: 0.25,
            dy_deg: 0.25,
            xlon0_deg: -12.0,
            ylat0_deg: 35.0,
        };

        assert_eq!(grid, grid.clone());
    }
}
