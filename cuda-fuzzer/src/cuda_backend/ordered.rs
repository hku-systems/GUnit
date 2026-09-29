use std::{cell::Cell, fmt, num::NonZeroU64, path::Path, time::Duration};

use libloading::Library;

use super::{
    duration_to_timeout_ms, invoke_profile_timing_snapshot, invoke_profile_timing_start,
    optional_profile_timing_snapshot, optional_profile_timing_start, required_cov_map,
    required_symbol, KernelTimingStats, LibaflProfileTimingSnapshotFn, LibaflProfileTimingStartFn,
    TaskResult, EDGES_MAP_SIZE,
};

type LibaflSubmitWithIdFn = unsafe extern "C" fn(*const u8, usize) -> u64;
type LibaflPollResultsFn = unsafe extern "C" fn(*mut TaskResult, usize) -> usize;
type LibaflReleaseTasksFn = unsafe extern "C" fn(*const u64, usize) -> usize;
type LibaflGetOrderedQueueCountsFn = unsafe extern "C" fn(*mut OrderedQueueCounts);
type LibaflSetTargetTimeoutMsFn = unsafe extern "C" fn(u64);
type LibaflLifecycleFn = unsafe extern "C" fn();

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrderedQueueCounts {
    pub pending: usize,
    pub completed: usize,
    pub outstanding: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OrderedSubmitError;

impl fmt::Display for OrderedSubmitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ordered CUDA backend rejected submission with task_id=0")
    }
}

impl std::error::Error for OrderedSubmitError {}

#[derive(Debug)]
pub struct OrderedCudaBackend {
    submit_fn: LibaflSubmitWithIdFn,
    poll_results_fn: LibaflPollResultsFn,
    release_tasks_fn: LibaflReleaseTasksFn,
    get_queue_counts_fn: LibaflGetOrderedQueueCountsFn,
    set_target_timeout_ms_fn: LibaflSetTargetTimeoutMsFn,
    wait_fn: LibaflLifecycleFn,
    stop_fn: LibaflLifecycleFn,
    stopped: Cell<bool>,
    profile_timing_start_fn: Option<LibaflProfileTimingStartFn>,
    profile_timing_snapshot_fn: Option<LibaflProfileTimingSnapshotFn>,
    cov_map: *mut [u8; EDGES_MAP_SIZE],
    _library: Library,
}

impl OrderedCudaBackend {
    pub fn new(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let library = unsafe { Library::new(path.as_ref()) }?;
        let submit_fn = unsafe { required_symbol(&library, b"libafl_submit_with_id") }?;
        let poll_results_fn = unsafe { required_symbol(&library, b"libafl_poll_results") }?;
        let release_tasks_fn = unsafe { required_symbol(&library, b"libafl_release_tasks") }?;
        let get_queue_counts_fn =
            unsafe { required_symbol(&library, b"libafl_get_ordered_queue_counts") }?;
        let set_target_timeout_ms_fn =
            unsafe { required_symbol(&library, b"libafl_set_target_timeout_ms") }?;
        let wait_fn = unsafe { required_symbol(&library, b"libafl_wait") }?;
        let stop_fn = unsafe { required_symbol(&library, b"libafl_stop") }?;
        let profile_timing_start_fn = unsafe { optional_profile_timing_start(&library) };
        let profile_timing_snapshot_fn = unsafe { optional_profile_timing_snapshot(&library) };
        let cov_map = unsafe { required_cov_map(&library) }?;

        Ok(Self {
            submit_fn,
            poll_results_fn,
            release_tasks_fn,
            get_queue_counts_fn,
            set_target_timeout_ms_fn,
            wait_fn,
            stop_fn,
            stopped: Cell::new(false),
            profile_timing_start_fn,
            profile_timing_snapshot_fn,
            cov_map,
            _library: library,
        })
    }

    pub fn submit(&self, input: &[u8]) -> Result<NonZeroU64, OrderedSubmitError> {
        NonZeroU64::new(unsafe { (self.submit_fn)(input.as_ptr(), input.len()) })
            .ok_or(OrderedSubmitError)
    }

    pub fn poll_completed_tasks(&self, max_count: usize) -> Vec<TaskResult> {
        if max_count == 0 {
            return Vec::new();
        }
        let mut results = vec![TaskResult::default(); max_count];
        let count = unsafe { (self.poll_results_fn)(results.as_mut_ptr(), max_count) };
        results.truncate(count.min(max_count));
        results
    }

    pub fn release_completed_tasks(&self, task_ids: &[u64]) -> usize {
        if task_ids.is_empty() {
            return 0;
        }
        unsafe { (self.release_tasks_fn)(task_ids.as_ptr(), task_ids.len()) }
    }

    pub fn get_queue_counts(&self) -> OrderedQueueCounts {
        let mut counts = OrderedQueueCounts::default();
        unsafe {
            (self.get_queue_counts_fn)(&mut counts);
        }
        counts
    }

    pub fn set_target_timeout(&self, timeout: Duration) {
        unsafe {
            (self.set_target_timeout_ms_fn)(duration_to_timeout_ms(timeout));
        }
    }

    pub fn wait(&self) {
        unsafe {
            (self.wait_fn)();
        }
    }

    pub fn stop(&self) {
        if !self.stopped.replace(true) {
            unsafe {
                (self.wait_fn)();
                (self.stop_fn)();
            }
        }
    }

    pub fn cov_map_ptr(&self) -> *mut u8 {
        self.cov_map.cast::<u8>()
    }

    pub fn profile_timing_start(&self) -> anyhow::Result<()> {
        invoke_profile_timing_start(self.profile_timing_start_fn)
    }

    pub fn profile_timing_snapshot(&self) -> Option<KernelTimingStats> {
        invoke_profile_timing_snapshot(self.profile_timing_snapshot_fn)
    }
}

impl Drop for OrderedCudaBackend {
    fn drop(&mut self) {
        self.stop();
    }
}
