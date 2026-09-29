use std::path::Path;

use libafl::executors::ExitKind;
use libloading::Library;

use super::{
    exit_kind_from_run_status, invoke_profile_timing_snapshot, invoke_profile_timing_start,
    optional_profile_timing_snapshot, optional_profile_timing_start, required_cov_map,
    required_simt_memcov_bits, required_symbol, KernelTimingStats, LibAflRunStatus,
    LibaflProfileTimingSnapshotFn, LibaflProfileTimingStartFn, EDGES_MAP_SIZE,
    SIMT_MEMCOV_STORAGE_SIZE,
};

type LibaflTargetFn = unsafe extern "C" fn(*const u8, usize);
type LibaflGetLastRunStatusFn = unsafe extern "C" fn(*mut LibAflRunStatus);
type LibaflLifecycleFn = unsafe extern "C" fn();

#[derive(Clone, Copy)]
pub struct SyncProfileTimingStart {
    callback: Option<LibaflProfileTimingStartFn>,
    snapshot: Option<LibaflProfileTimingSnapshotFn>,
}

impl SyncProfileTimingStart {
    pub fn start(self) -> anyhow::Result<()> {
        invoke_profile_timing_start(self.callback)
    }

    pub fn snapshot(self) -> Option<KernelTimingStats> {
        invoke_profile_timing_snapshot(self.snapshot)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SyncRunResult {
    pub status: LibAflRunStatus,
    pub exit_kind: ExitKind,
}

pub struct SyncCudaBackend {
    target_fn: LibaflTargetFn,
    get_last_run_status_fn: LibaflGetLastRunStatusFn,
    wait_fn: Option<LibaflLifecycleFn>,
    stop_fn: Option<LibaflLifecycleFn>,
    profile_timing_start_fn: Option<LibaflProfileTimingStartFn>,
    profile_timing_snapshot_fn: Option<LibaflProfileTimingSnapshotFn>,
    cov_map: *mut [u8; EDGES_MAP_SIZE],
    simt_memcov_bits: *mut [u8; SIMT_MEMCOV_STORAGE_SIZE],
    stopped: bool,
    _library: Library,
}

impl SyncCudaBackend {
    pub fn new(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let library = unsafe { Library::new(path.as_ref()) }?;
        let target_fn = unsafe { required_symbol(&library, b"libafl_target") }?;
        let get_last_run_status_fn =
            unsafe { required_symbol(&library, b"libafl_get_last_run_status") }?;
        let wait_fn = unsafe { library.get::<LibaflLifecycleFn>(b"libafl_wait") }
            .ok()
            .map(|symbol| *symbol);
        let stop_fn = unsafe { library.get::<LibaflLifecycleFn>(b"libafl_stop") }
            .ok()
            .map(|symbol| *symbol);
        let profile_timing_start_fn = unsafe { optional_profile_timing_start(&library) };
        let profile_timing_snapshot_fn = unsafe { optional_profile_timing_snapshot(&library) };
        let cov_map = unsafe { required_cov_map(&library) }?;
        let simt_memcov_bits = unsafe { required_simt_memcov_bits(&library) }?;

        Ok(Self {
            target_fn,
            get_last_run_status_fn,
            wait_fn,
            stop_fn,
            profile_timing_start_fn,
            profile_timing_snapshot_fn,
            cov_map,
            simt_memcov_bits,
            stopped: false,
            _library: library,
        })
    }

    pub fn run(&mut self, input: &[u8]) -> SyncRunResult {
        unsafe {
            (self.target_fn)(input.as_ptr(), input.len());
            if let Some(wait_fn) = self.wait_fn {
                wait_fn();
            }
        }
        let mut status = LibAflRunStatus::default();
        unsafe {
            (self.get_last_run_status_fn)(&mut status);
        }
        SyncRunResult {
            status,
            exit_kind: exit_kind_from_run_status(status),
        }
    }

    pub fn cov_map_ptr(&self) -> *mut u8 {
        self.cov_map.cast::<u8>()
    }

    pub fn simt_memcov_bits(&mut self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.simt_memcov_bits.cast::<u8>(), SIMT_MEMCOV_STORAGE_SIZE)
        }
    }

    pub fn profile_timing_start(&self) -> anyhow::Result<()> {
        invoke_profile_timing_start(self.profile_timing_start_fn)
    }

    pub fn profile_timing_start_handle(&self) -> SyncProfileTimingStart {
        SyncProfileTimingStart {
            callback: self.profile_timing_start_fn,
            snapshot: self.profile_timing_snapshot_fn,
        }
    }

    pub fn stop(&mut self) {
        if self.stopped {
            return;
        }
        self.stopped = true;
        unsafe {
            if let Some(wait) = self.wait_fn {
                wait();
            }
            if let Some(stop) = self.stop_fn {
                stop();
            }
        }
    }
}

impl Drop for SyncCudaBackend {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        path::PathBuf,
        process::Command,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn c_compiler() -> Option<&'static str> {
        ["cc", "clang", "gcc"]
            .into_iter()
            .find(|compiler| Command::new(compiler).arg("--version").output().is_ok())
    }

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before UNIX_EPOCH")
            .as_nanos();
        std::env::temp_dir().join(format!("{name}-{}-{nanos}", std::process::id()))
    }

    #[test]
    fn run_invokes_optional_wait_before_reading_feedback() {
        let Some(compiler) = c_compiler() else {
            eprintln!("skipping sync backend wait test: no C compiler found");
            return;
        };
        let temp_dir = unique_temp_dir("rapid-sync-backend-test");
        fs::create_dir_all(&temp_dir).expect("create temp test directory");
        let source_path = temp_dir.join("backend.c");
        let library_path = temp_dir.join("libbackend.so");
        fs::write(
            &source_path,
            r#"
#include <stddef.h>
#include <stdint.h>

typedef struct {
  uint16_t code;
  uint16_t stage;
  uint32_t detail;
} LibAflRunStatus;

uint8_t libafl_cov_map[65536];
uint8_t libafl_simt_memcov_bits[8192];

static uint32_t wait_count = 0;

void libafl_target(const uint8_t *input, size_t size) {
  if (size == 3 && input[0] == 1) {
    libafl_cov_map[0] = 1;
  }
}

void libafl_wait(void) {
  wait_count++;
  libafl_simt_memcov_bits[0] = 0x20;
}

void libafl_get_last_run_status(LibAflRunStatus *out_status) {
  out_status->code = 0;
  out_status->stage = 0;
  out_status->detail = wait_count;
}
"#,
        )
        .expect("write test backend source");
        let build = Command::new(compiler)
            .args([
                "-shared",
                "-fPIC",
                source_path.to_str().expect("source path is not UTF-8"),
                "-o",
                library_path.to_str().expect("library path is not UTF-8"),
            ])
            .status()
            .expect("compile test backend");
        assert!(build.success(), "test backend compilation failed");

        let mut backend = SyncCudaBackend::new(&library_path).expect("load test backend");
        let result = backend.run(&[1, 2, 3]);

        assert_eq!(result.status.detail, 1);
        assert_eq!(backend.simt_memcov_bits()[0], 0x20);
        fs::remove_dir_all(&temp_dir).expect("remove temp test directory");
    }
}
