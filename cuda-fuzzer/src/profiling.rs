use libafl::{
    corpus::Testcase,
    executors::ExitKind,
    feedbacks::{Feedback, StateInitializer},
    Error,
};
use libafl_bolts::Named;
use libloading::{Library, Symbol};
use serde::Serialize;
use std::{
    borrow::Cow,
    cell::RefCell,
    fs::File,
    io::{BufWriter, Write},
    marker::PhantomData,
    path::PathBuf,
    rc::Rc,
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};

use crate::cuda_backend::KernelTimingStats;

const SCHEMA_VERSION: u32 = 2;
const CUDA_SUCCESS: i32 = 0;
const SEGMENT_COUNT: usize = 9;
type CudaProfilerFn = unsafe extern "C" fn() -> i32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum Segment {
    Submit,
    Poll,
    Coverage,
    Evaluate,
    Release,
    SchedulerStage,
    OtherIdle,
    FeedbackPredicate,
    FeedbackMetadata,
}

impl Segment {
    const ALL: [Self; SEGMENT_COUNT] = [
        Self::Submit,
        Self::Poll,
        Self::Coverage,
        Self::Evaluate,
        Self::Release,
        Self::SchedulerStage,
        Self::OtherIdle,
        Self::FeedbackPredicate,
        Self::FeedbackMetadata,
    ];

    const fn domain(self) -> &'static str {
        match self {
            Self::FeedbackPredicate | Self::FeedbackMetadata => "cpu_feedback",
            _ => "main_loop",
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Submit => "submit",
            Self::Poll => "poll",
            Self::Coverage => "coverage",
            Self::Evaluate => "evaluate",
            Self::Release => "release",
            Self::SchedulerStage => "scheduler_stage",
            Self::OtherIdle => "other_idle",
            Self::FeedbackPredicate => "predicate",
            Self::FeedbackMetadata => "metadata",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SegmentStat {
    count: u64,
    total: u64,
    max: u64,
}

impl SegmentStat {
    fn record(&mut self, elapsed: Duration) {
        let elapsed = duration_ns(elapsed);
        self.count = self.count.saturating_add(1);
        self.total = self.total.saturating_add(elapsed);
        self.max = self.max.max(elapsed);
    }
}

#[derive(Debug)]
struct Frame {
    segment: Segment,
    elapsed: Duration,
}

#[derive(Debug, Default)]
struct ThreadState {
    generation: u64,
    frames: Vec<Frame>,
    last_boundary: Option<Instant>,
}

impl ThreadState {
    fn reset(&mut self, generation: u64, root: Option<Segment>) {
        self.generation = generation;
        self.frames.clear();
        if let Some(segment) = root {
            self.frames.push(Frame {
                segment,
                elapsed: Duration::ZERO,
            });
        }
        self.last_boundary = Some(Instant::now());
    }

    fn record_current_slice(&mut self) {
        let now = Instant::now();
        if let (Some(last), Some(frame)) = (self.last_boundary, self.frames.last_mut()) {
            frame.elapsed = frame.elapsed.saturating_add(now.duration_since(last));
        }
        self.last_boundary = Some(now);
    }
}

#[derive(Debug)]
struct Collector {
    generation: u64,
    output: PathBuf,
    stats: [SegmentStat; SEGMENT_COUNT],
    device: Option<KernelTimingStats>,
}

impl Collector {
    fn new(generation: u64, output: PathBuf) -> Self {
        Self {
            generation,
            output,
            stats: [SegmentStat::default(); SEGMENT_COUNT],
            device: None,
        }
    }
}

thread_local! {
    static THREAD_STATE: RefCell<ThreadState> = RefCell::new(ThreadState::default());
}

static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);
static ACTIVE_GENERATION: AtomicU64 = AtomicU64::new(0);
static COLLECTOR: Mutex<Option<Collector>> = Mutex::new(None);

#[derive(Debug)]
pub(crate) struct Scope {
    generation: u64,
    segment: Segment,
    active: bool,
    _not_send: PhantomData<Rc<()>>,
}

pub(crate) fn scope(segment: Segment) -> Scope {
    let generation = ACTIVE_GENERATION.load(Ordering::Acquire);
    let active = generation != 0;
    if active {
        THREAD_STATE.with(|state| {
            let mut state = state.borrow_mut();
            if state.generation != generation {
                state.reset(generation, None);
            }
            state.record_current_slice();
            state.frames.push(Frame {
                segment,
                elapsed: Duration::ZERO,
            });
        });
    }
    Scope {
        generation,
        segment,
        active,
        _not_send: PhantomData,
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let frame = THREAD_STATE.with(|state| {
            let mut state = state.borrow_mut();
            state.record_current_slice();
            state.frames.pop()
        });
        let Some(frame) = frame else {
            return;
        };
        debug_assert_eq!(frame.segment, self.segment);
        let mut collector = COLLECTOR
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(collector) = collector
            .as_mut()
            .filter(|collector| collector.generation == self.generation)
        {
            collector.stats[frame.segment as usize].record(frame.elapsed);
        }
        THREAD_STATE.with(|state| {
            state.borrow_mut().last_boundary = Some(Instant::now());
        });
    }
}

#[derive(Debug)]
pub struct ProfileSession {
    runtime: Option<Library>,
    generation: u64,
    finished: bool,
    _not_send: PhantomData<Rc<()>>,
}

impl ProfileSession {
    pub fn enabled_from_env() -> Result<bool, String> {
        match std::env::var_os("RAPID_PROFILE") {
            None => Ok(false),
            Some(value) if value == "0" => Ok(false),
            Some(value) if value == "1" => Ok(true),
            Some(value) => Err(format!(
                "RAPID_PROFILE must be 0 or 1; got {:?}",
                value.to_string_lossy()
            )),
        }
    }

    pub fn start_from_env() -> Result<Option<Self>, String> {
        if !Self::enabled_from_env()? {
            return Ok(None);
        }
        let output = std::env::var_os("RAPID_PROFILE_OUTPUT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("rapid-profile.jsonl"));
        let runtime = load_cuda_runtime()?;
        let status = unsafe {
            let start: Symbol<CudaProfilerFn> = runtime
                .get(b"cudaProfilerStart")
                .map_err(|error| format!("failed to resolve cudaProfilerStart: {error}"))?;
            start()
        };
        if status != CUDA_SUCCESS {
            return Err(format!(
                "cudaProfilerStart failed with CUDA status {status}"
            ));
        }

        let generation = NEXT_GENERATION.fetch_add(1, Ordering::Relaxed);
        let mut collector = COLLECTOR
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if collector.is_some() {
            return Err("RAPID profiling session already active".to_string());
        }
        *collector = Some(Collector::new(generation, output));
        THREAD_STATE.with(|state| {
            state
                .borrow_mut()
                .reset(generation, Some(Segment::OtherIdle));
        });
        ACTIVE_GENERATION.store(generation, Ordering::Release);
        Ok(Some(Self {
            runtime: Some(runtime),
            generation,
            finished: false,
            _not_send: PhantomData,
        }))
    }

    pub fn stop_after_measuring<F>(
        mut self,
        started: Instant,
        prepare_device: F,
    ) -> Result<u64, String>
    where
        F: FnOnce() -> Option<KernelTimingStats>,
    {
        finish_after_measurement(
            &mut self,
            || duration_ns(started.elapsed()),
            |session| session.freeze(),
            prepare_device,
            |session| session.stop_profiler(),
            |session, device| session.emit(device),
        )
    }

    fn freeze(&mut self) -> Result<(), String> {
        if self.finished {
            return Ok(());
        }
        ACTIVE_GENERATION
            .compare_exchange(self.generation, 0, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| "RAPID profiling session generation changed unexpectedly".to_string())?;
        finish_owner_thread(self.generation);
        Ok(())
    }

    fn stop_profiler(&mut self) -> Result<(), String> {
        let Some(runtime) = self.runtime.take() else {
            return Ok(());
        };
        let status = unsafe {
            let stop: Symbol<CudaProfilerFn> = runtime
                .get(b"cudaProfilerStop")
                .map_err(|error| format!("failed to resolve cudaProfilerStop: {error}"))?;
            stop()
        };
        if status != CUDA_SUCCESS {
            return Err(format!("cudaProfilerStop failed with CUDA status {status}"));
        }
        Ok(())
    }

    fn emit(&mut self, device: Option<KernelTimingStats>) -> Result<(), String> {
        let mut collector = COLLECTOR
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut collector = collector
            .take()
            .ok_or_else(|| "RAPID profiling collector is missing".to_string())?;
        if collector.generation != self.generation {
            return Err("RAPID profiling collector generation mismatch".to_string());
        }
        collector.device = device;
        self.finished = true;
        write_records(&collector)
    }
}

fn finish_after_measurement<S, T, E>(
    state: &mut S,
    measure: impl FnOnce() -> u64,
    freeze: impl FnOnce(&mut S) -> Result<(), E>,
    prepare: impl FnOnce() -> T,
    stop_profiler: impl FnOnce(&mut S) -> Result<(), E>,
    emit: impl FnOnce(&mut S, T) -> Result<(), E>,
) -> Result<u64, E> {
    let elapsed = measure();
    freeze(state)?;
    let prepared = prepare();
    let stop_result = stop_profiler(state);
    let emit_result = emit(state, prepared);
    stop_result.and(emit_result)?;
    Ok(elapsed)
}

impl Drop for ProfileSession {
    fn drop(&mut self) {
        if !self.finished {
            if let Err(error) = finish_after_measurement(
                self,
                || 0,
                |session| session.freeze(),
                || None,
                |session| session.stop_profiler(),
                |session, device| session.emit(device),
            ) {
                eprintln!("failed to finish RAPID profiling: {error}");
            }
        }
    }
}

fn finish_owner_thread(generation: u64) {
    let frames = THREAD_STATE.with(|state| {
        let mut state = state.borrow_mut();
        if state.generation != generation {
            return Vec::new();
        }
        state.record_current_slice();
        std::mem::take(&mut state.frames)
    });
    let mut collector = COLLECTOR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(collector) = collector
        .as_mut()
        .filter(|collector| collector.generation == generation)
    else {
        return;
    };
    for frame in frames {
        collector.stats[frame.segment as usize].record(frame.elapsed);
    }
}

#[derive(Serialize)]
struct SegmentRecord<'a> {
    schema_version: u32,
    record_type: &'static str,
    domain: &'a str,
    segment: &'a str,
    unit: &'static str,
    count: u64,
    total: u64,
    max: Option<u64>,
}

fn write_records(collector: &Collector) -> Result<(), String> {
    let file = File::create(&collector.output).map_err(|error| {
        format!(
            "failed to create profiling output {}: {error}",
            collector.output.display()
        )
    })?;
    let mut writer = BufWriter::new(file);
    for (segment, stat) in Segment::ALL.into_iter().zip(collector.stats) {
        write_record(
            &mut writer,
            &SegmentRecord {
                schema_version: SCHEMA_VERSION,
                record_type: "segment",
                domain: segment.domain(),
                segment: segment.name(),
                unit: "ns",
                count: stat.count,
                total: stat.total,
                max: Some(stat.max),
            },
        )?;
    }
    if let Some(device) = collector.device {
        for (name, total) in device.segments() {
            write_record(
                &mut writer,
                &SegmentRecord {
                    schema_version: SCHEMA_VERSION,
                    record_type: "segment",
                    domain: "device_kernel",
                    segment: name,
                    unit: "cycles",
                    count: device.iterations,
                    total,
                    max: None,
                },
            )?;
        }
    }
    writer
        .flush()
        .map_err(|error| format!("failed to flush profiling output: {error}"))
}

fn write_record(writer: &mut impl Write, record: &SegmentRecord<'_>) -> Result<(), String> {
    serde_json::to_writer(&mut *writer, record)
        .map_err(|error| format!("failed to serialize profiling record: {error}"))?;
    writer
        .write_all(b"\n")
        .map_err(|error| format!("failed to write profiling record: {error}"))
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

fn load_cuda_runtime() -> Result<Library, String> {
    let candidates = match std::env::var("RAPID_PROFILE_CUDART") {
        Ok(path) => vec![path],
        Err(_) => vec![
            "libcudart.so".to_string(),
            "libcudart.so.12".to_string(),
            "libcudart.so.11.0".to_string(),
            "libcudart.dylib".to_string(),
        ],
    };
    let mut failures = Vec::new();
    for candidate in &candidates {
        match unsafe { Library::new(candidate) } {
            Ok(runtime) => return Ok(runtime),
            Err(error) => failures.push(format!("{candidate}: {error}")),
        }
    }
    Err(format!(
        "failed to load CUDA runtime from [{}]",
        failures.join("; ")
    ))
}

#[derive(Debug)]
pub struct ProfiledFeedback<F> {
    inner: F,
}

impl<F> ProfiledFeedback<F> {
    pub fn new(inner: F) -> Self {
        Self { inner }
    }
}

impl<F: Named> Named for ProfiledFeedback<F> {
    fn name(&self) -> &Cow<'static, str> {
        self.inner.name()
    }
}

impl<F, S> StateInitializer<S> for ProfiledFeedback<F>
where
    F: StateInitializer<S>,
{
    fn init_state(&mut self, state: &mut S) -> Result<(), Error> {
        self.inner.init_state(state)
    }
}

impl<EM, F, I, OT, S> Feedback<EM, I, OT, S> for ProfiledFeedback<F>
where
    F: Feedback<EM, I, OT, S>,
{
    fn is_interesting(
        &mut self,
        state: &mut S,
        manager: &mut EM,
        input: &I,
        observers: &OT,
        exit_kind: &ExitKind,
    ) -> Result<bool, Error> {
        let _scope = scope(Segment::FeedbackPredicate);
        self.inner
            .is_interesting(state, manager, input, observers, exit_kind)
    }

    fn append_metadata(
        &mut self,
        state: &mut S,
        manager: &mut EM,
        observers: &OT,
        testcase: &mut Testcase<I>,
    ) -> Result<(), Error> {
        let _scope = scope(Segment::FeedbackMetadata);
        self.inner
            .append_metadata(state, manager, observers, testcase)
    }

    #[cfg(feature = "track_hit_feedbacks")]
    fn last_result(&self) -> Result<bool, Error> {
        self.inner.last_result()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::{finish_after_measurement, ProfileSession};

    #[test]
    fn persistent_backend_stops_before_profiler_and_emit() {
        let events = RefCell::new(Vec::new());
        let mut state = ();

        let elapsed = finish_after_measurement(
            &mut state,
            || {
                events.borrow_mut().push("measure");
                42
            },
            |_| Ok::<(), String>(()),
            || {
                events.borrow_mut().push("snapshot");
                events.borrow_mut().push("backend stop");
                Some(7)
            },
            |_| {
                events.borrow_mut().push("profiler stop");
                Ok::<(), String>(())
            },
            |_, snapshot| {
                assert_eq!(snapshot, Some(7));
                events.borrow_mut().push("emit");
                Ok::<(), String>(())
            },
        )
        .unwrap();

        assert_eq!(elapsed, 42);
        assert_eq!(
            &*events.borrow(),
            &[
                "measure",
                "snapshot",
                "backend stop",
                "profiler stop",
                "emit",
            ]
        );
    }

    #[test]
    fn profiling_is_runtime_opt_in() {
        let original = std::env::var_os("RAPID_PROFILE");
        std::env::remove_var("RAPID_PROFILE");
        assert!(!ProfileSession::enabled_from_env().unwrap());
        std::env::set_var("RAPID_PROFILE", "1");
        assert!(ProfileSession::enabled_from_env().unwrap());
        std::env::set_var("RAPID_PROFILE", "yes");
        assert!(ProfileSession::enabled_from_env().is_err());
        match original {
            Some(value) => std::env::set_var("RAPID_PROFILE", value),
            None => std::env::remove_var("RAPID_PROFILE"),
        }
    }
}
