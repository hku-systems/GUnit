mod r#async;
mod ordered;
mod sync;

use std::time::Duration;

use anyhow::Context;
use libafl::executors::ExitKind;
use libloading::Library;

pub use ordered::{OrderedCudaBackend, OrderedQueueCounts, OrderedSubmitError};
pub use r#async::{AsyncCudaBackend, AsyncSubmitError};
pub use sync::{SyncCudaBackend, SyncRunResult};

pub const EDGES_MAP_SIZE: usize = 65_536;
pub const SIMT_MEMCOV_STORAGE_SIZE: usize = 8_192;

pub const LIBAFL_RUN_STATUS_OK: u16 = 0;
pub const LIBAFL_RUN_STATUS_CUDA_ERROR: u16 = 1;
pub const LIBAFL_RUN_STATUS_TIMEOUT: u16 = 2;
pub const LIBAFL_RUN_STATUS_INVALID_INPUT: u16 = 3;
pub const LIBAFL_RUN_STATUS_INTERNAL_ERROR: u16 = 4;

pub const LIBAFL_BACKEND_STAGE_UNKNOWN: u16 = 0;
pub const LIBAFL_BACKEND_STAGE_INIT: u16 = 1;
pub const LIBAFL_BACKEND_STAGE_SUBMIT: u16 = 2;
pub const LIBAFL_BACKEND_STAGE_LAUNCH: u16 = 3;
pub const LIBAFL_BACKEND_STAGE_EXECUTE: u16 = 4;
pub const LIBAFL_BACKEND_STAGE_COLLECT: u16 = 5;

type LibaflCovMap = *mut [u8; EDGES_MAP_SIZE];
type LibaflSimtMemCovBits = *mut [u8; SIMT_MEMCOV_STORAGE_SIZE];
pub(crate) type LibaflProfileTimingStartFn = unsafe extern "C" fn() -> u32;
pub(crate) type LibaflProfileTimingSnapshotFn = unsafe extern "C" fn(*mut KernelTimingStats) -> u32;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KernelTimingStats {
    pub idle_cycles: u64,
    pub init_cov_cycles: u64,
    pub decode_cycles: u64,
    pub feedback_prepare_cycles: u64,
    pub kernel_cycles: u64,
    pub merge_cov_cycles: u64,
    pub signal_cycles: u64,
    pub sync_overhead_cycles: u64,
    pub iterations: u64,
}

impl KernelTimingStats {
    pub fn segments(self) -> [(&'static str, u64); 8] {
        [
            ("idle", self.idle_cycles),
            ("feedback_init", self.init_cov_cycles),
            ("input_decode", self.decode_cycles),
            ("feedback_prepare", self.feedback_prepare_cycles),
            ("target_execution", self.kernel_cycles),
            ("feedback_merge", self.merge_cov_cycles),
            ("signal", self.signal_cycles),
            ("bookkeeping", self.sync_overhead_cycles),
        ]
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LibAflRunStatus {
    pub code: u16,
    pub stage: u16,
    pub detail: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TaskResult {
    pub task_id: u64,
    pub input_ptr: usize,
    pub edge_ptr: *const u8,
    pub simt_memcov_ptr: *const u8,
    pub edge_size: u32,
    pub simt_memcov_size: u32,
    pub status: LibAflRunStatus,
    pub exec_time_ns: u64,
}

impl Default for TaskResult {
    fn default() -> Self {
        Self {
            task_id: 0,
            input_ptr: 0,
            edge_ptr: std::ptr::null(),
            simt_memcov_ptr: std::ptr::null(),
            edge_size: 0,
            simt_memcov_size: 0,
            status: LibAflRunStatus::default(),
            exec_time_ns: 0,
        }
    }
}

impl TaskResult {
    /// Borrow the task-owned edge map before this task is released.
    ///
    /// # Safety
    /// The C++ task result must remain acquired for the returned slice's full
    /// lifetime.
    pub unsafe fn edge_slice(&self) -> anyhow::Result<&[u8]> {
        if self.edge_size as usize != EDGES_MAP_SIZE {
            anyhow::bail!(
                "invalid edge feedback size for task_id={}: expected {}, got {}",
                self.task_id,
                EDGES_MAP_SIZE,
                self.edge_size
            );
        }
        if self.edge_ptr.is_null() {
            anyhow::bail!("null edge feedback pointer for task_id={}", self.task_id);
        }
        Ok(unsafe { std::slice::from_raw_parts(self.edge_ptr, EDGES_MAP_SIZE) })
    }

    /// Borrow the task-owned memory/index bitset before this task is released.
    ///
    /// # Safety
    /// The C++ task result must remain acquired for the returned slice's full
    /// lifetime.
    pub unsafe fn simt_memcov_slice(&self) -> anyhow::Result<&[u8]> {
        if self.simt_memcov_size as usize != SIMT_MEMCOV_STORAGE_SIZE {
            anyhow::bail!(
                "invalid memory/index feedback size for task_id={}: expected {}, got {}",
                self.task_id,
                SIMT_MEMCOV_STORAGE_SIZE,
                self.simt_memcov_size
            );
        }
        if self.simt_memcov_ptr.is_null() {
            anyhow::bail!(
                "null memory/index feedback pointer for task_id={}",
                self.task_id
            );
        }
        Ok(unsafe { std::slice::from_raw_parts(self.simt_memcov_ptr, SIMT_MEMCOV_STORAGE_SIZE) })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LibAflQueueCounts {
    pub pending: usize,
    pub completed: usize,
}

unsafe fn required_symbol<T>(library: &Library, name: &'static [u8]) -> anyhow::Result<T>
where
    T: Copy,
{
    let display_name = String::from_utf8_lossy(name);
    let symbol = unsafe { library.get::<T>(name) }
        .with_context(|| format!("missing required CUDA backend symbol {display_name}"))?;
    Ok(*symbol)
}

unsafe fn required_cov_map(library: &Library) -> anyhow::Result<LibaflCovMap> {
    unsafe { required_symbol(library, b"libafl_cov_map") }
}

unsafe fn required_simt_memcov_bits(library: &Library) -> anyhow::Result<LibaflSimtMemCovBits> {
    unsafe { required_symbol(library, b"libafl_simt_memcov_bits") }
}

pub(crate) unsafe fn optional_profile_timing_start(
    library: &Library,
) -> Option<LibaflProfileTimingStartFn> {
    unsafe {
        library
            .get::<LibaflProfileTimingStartFn>(b"libafl_profile_timing_start")
            .ok()
            .map(|symbol| *symbol)
    }
}

pub(crate) unsafe fn optional_profile_timing_snapshot(
    library: &Library,
) -> Option<LibaflProfileTimingSnapshotFn> {
    unsafe {
        library
            .get::<LibaflProfileTimingSnapshotFn>(b"libafl_profile_timing_snapshot")
            .ok()
            .map(|symbol| *symbol)
    }
}

pub(crate) fn invoke_profile_timing_start(
    hook: Option<LibaflProfileTimingStartFn>,
) -> anyhow::Result<()> {
    if let Some(hook) = hook {
        let status = unsafe { hook() };
        if status == 0 {
            anyhow::bail!("CUDA backend failed to reset persistent timing counters");
        }
    }
    Ok(())
}

pub(crate) fn invoke_profile_timing_snapshot(
    hook: Option<LibaflProfileTimingSnapshotFn>,
) -> Option<KernelTimingStats> {
    let hook = hook?;
    let mut stats = KernelTimingStats::default();
    (unsafe { hook(&mut stats) } != 0).then_some(stats)
}

pub(crate) fn duration_to_timeout_ms(timeout: Duration) -> u64 {
    let millis = timeout.as_millis();
    if millis == 0 {
        1
    } else if millis > u64::MAX as u128 {
        u64::MAX
    } else {
        millis as u64
    }
}

pub fn exit_kind_from_run_status(status: LibAflRunStatus) -> ExitKind {
    match status.code {
        LIBAFL_RUN_STATUS_OK => ExitKind::Ok,
        LIBAFL_RUN_STATUS_TIMEOUT => ExitKind::Timeout,
        _ => ExitKind::Crash,
    }
}

pub fn run_status_requires_fresh_process(status: LibAflRunStatus) -> bool {
    matches!(
        status.code,
        LIBAFL_RUN_STATUS_CUDA_ERROR | LIBAFL_RUN_STATUS_TIMEOUT | LIBAFL_RUN_STATUS_INTERNAL_ERROR
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_status_layout_matches_shared_ffi_header() {
        assert_eq!(std::mem::size_of::<LibAflRunStatus>(), 8);
        assert_eq!(std::mem::offset_of!(LibAflRunStatus, code), 0);
        assert_eq!(std::mem::offset_of!(LibAflRunStatus, stage), 2);
        assert_eq!(std::mem::offset_of!(LibAflRunStatus, detail), 4);
    }

    #[test]
    fn duration_to_timeout_ms_rounds_sub_ms_up_and_saturates() {
        assert_eq!(duration_to_timeout_ms(Duration::from_nanos(1)), 1);
        assert_eq!(duration_to_timeout_ms(Duration::from_millis(25)), 25);
        assert_eq!(duration_to_timeout_ms(Duration::MAX), u64::MAX);
    }

    #[test]
    fn run_status_maps_clean_execution_to_ok() {
        assert_eq!(
            exit_kind_from_run_status(LibAflRunStatus::default()),
            ExitKind::Ok
        );
    }

    #[test]
    fn run_status_maps_timeout_separately_from_crashes() {
        assert_eq!(
            exit_kind_from_run_status(LibAflRunStatus {
                code: LIBAFL_RUN_STATUS_TIMEOUT,
                ..LibAflRunStatus::default()
            }),
            ExitKind::Timeout
        );
    }

    #[test]
    fn run_status_maps_other_non_ok_statuses_to_crash() {
        for code in [
            LIBAFL_RUN_STATUS_CUDA_ERROR,
            LIBAFL_RUN_STATUS_INVALID_INPUT,
            LIBAFL_RUN_STATUS_INTERNAL_ERROR,
        ] {
            assert_eq!(
                exit_kind_from_run_status(LibAflRunStatus {
                    code,
                    ..LibAflRunStatus::default()
                }),
                ExitKind::Crash
            );
        }
    }

    #[test]
    fn run_status_marks_context_poisoning_statuses_for_fresh_process_restart() {
        for code in [
            LIBAFL_RUN_STATUS_CUDA_ERROR,
            LIBAFL_RUN_STATUS_TIMEOUT,
            LIBAFL_RUN_STATUS_INTERNAL_ERROR,
        ] {
            assert!(run_status_requires_fresh_process(LibAflRunStatus {
                code,
                ..LibAflRunStatus::default()
            }));
        }

        for code in [LIBAFL_RUN_STATUS_OK, LIBAFL_RUN_STATUS_INVALID_INPUT] {
            assert!(!run_status_requires_fresh_process(LibAflRunStatus {
                code,
                ..LibAflRunStatus::default()
            }));
        }
    }

    #[test]
    fn task_result_layout_matches_rapid2_ffi() {
        assert_eq!(std::mem::size_of::<TaskResult>(), 56);
        assert_eq!(std::mem::offset_of!(TaskResult, edge_ptr), 16);
        assert_eq!(std::mem::offset_of!(TaskResult, simt_memcov_ptr), 24);
        assert_eq!(std::mem::offset_of!(TaskResult, status), 40);
        assert_eq!(std::mem::offset_of!(TaskResult, exec_time_ns), 48);
    }

    #[test]
    fn task_result_rejects_null_feedback_pointers() {
        let result = TaskResult {
            edge_size: EDGES_MAP_SIZE as u32,
            simt_memcov_size: SIMT_MEMCOV_STORAGE_SIZE as u32,
            ..TaskResult::default()
        };

        assert!(unsafe { result.edge_slice() }
            .unwrap_err()
            .to_string()
            .contains("null edge feedback pointer"));
        assert!(unsafe { result.simt_memcov_slice() }
            .unwrap_err()
            .to_string()
            .contains("null memory/index feedback pointer"));
    }

    #[test]
    fn task_result_rejects_mismatched_feedback_sizes() {
        let byte = 0_u8;
        let result = TaskResult {
            edge_ptr: &byte,
            simt_memcov_ptr: &byte,
            edge_size: (EDGES_MAP_SIZE - 1) as u32,
            simt_memcov_size: (SIMT_MEMCOV_STORAGE_SIZE - 1) as u32,
            ..TaskResult::default()
        };

        assert!(unsafe { result.edge_slice() }
            .unwrap_err()
            .to_string()
            .contains("invalid edge feedback size"));
        assert!(unsafe { result.simt_memcov_slice() }
            .unwrap_err()
            .to_string()
            .contains("invalid memory/index feedback size"));
    }

    #[test]
    fn task_result_exposes_valid_feedback_slices() {
        let edges = [7_u8; EDGES_MAP_SIZE];
        let simt_memcov = [11_u8; SIMT_MEMCOV_STORAGE_SIZE];
        let result = TaskResult {
            edge_ptr: edges.as_ptr(),
            simt_memcov_ptr: simt_memcov.as_ptr(),
            edge_size: EDGES_MAP_SIZE as u32,
            simt_memcov_size: SIMT_MEMCOV_STORAGE_SIZE as u32,
            ..TaskResult::default()
        };

        assert_eq!(unsafe { result.edge_slice() }.unwrap()[0], 7);
        assert_eq!(unsafe { result.simt_memcov_slice() }.unwrap()[0], 11);
    }
}
