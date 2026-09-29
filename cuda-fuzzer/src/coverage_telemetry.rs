use std::{
    fs::File,
    io::Write,
    path::Path,
    sync::{mpsc, Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use anyhow::Context;
use serde::Serialize;

use crate::cuda_backend::{EDGES_MAP_SIZE, SIMT_MEMCOV_STORAGE_SIZE};
use crate::retirement::merge_coverage_map;

pub const MEMORY_FEATURE_BITS: usize = 61_440;
pub const THREAD_ACTIVITY_FEATURE_BITS: usize = 4_096;
pub const THREAD_ACTIVITY_BYTE_BASE: usize = MEMORY_FEATURE_BITS / 8;

const _: [(); SIMT_MEMCOV_STORAGE_SIZE] =
    [(); THREAD_ACTIVITY_BYTE_BASE + THREAD_ACTIVITY_FEATURE_BITS / 8];

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CoverageSnapshot {
    pub executions_submitted: u64,
    pub executions_completed: u64,
    pub cfg_sites: u64,
    pub memory_features: u64,
    pub thread_activity_features: u64,
    pub feedback_features_total: u64,
    pub memory_map_hex: String,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CoverageCountSnapshot {
    pub executions_submitted: u64,
    pub executions_completed: u64,
    pub cfg_sites: u64,
    pub memory_features: u64,
    pub thread_activity_features: u64,
    pub feedback_features_total: u64,
}

impl CoverageCountSnapshot {
    fn has_coverage_change_from(self, previous: Self) -> bool {
        self.cfg_sites != previous.cfg_sites
            || self.memory_features != previous.memory_features
            || self.thread_activity_features != previous.thread_activity_features
    }
}

#[derive(Clone, Debug)]
struct CoverageEventSink {
    started_at: Instant,
    sender: mpsc::Sender<CoverageLogCommand>,
}

#[derive(Debug)]
struct CoverageTelemetryState {
    cfg_map: Vec<u8>,
    memory_map: Vec<u8>,
    executions_submitted: u64,
    executions_completed: u64,
    cfg_sites: u64,
    memory_features: u64,
    thread_activity_features: u64,
    event_sink: Option<CoverageEventSink>,
}

impl Default for CoverageTelemetryState {
    fn default() -> Self {
        Self {
            cfg_map: vec![0; EDGES_MAP_SIZE],
            memory_map: vec![0; SIMT_MEMCOV_STORAGE_SIZE],
            executions_submitted: 0,
            executions_completed: 0,
            cfg_sites: 0,
            memory_features: 0,
            thread_activity_features: 0,
            event_sink: None,
        }
    }
}

/// Cumulative, observation-only feedback statistics for an RQ2 campaign.
///
/// This type owns maps separate from LibAFL's observer and corpus state. Each
/// completed task is ORed into the cumulative maps, so duplicate observations
/// do not increase any feature count.
#[derive(Clone, Debug, Default)]
pub struct CoverageTelemetry {
    state: Arc<Mutex<CoverageTelemetryState>>,
}

impl CoverageTelemetry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_submission(&self) -> anyhow::Result<()> {
        let mut state = self.lock_state()?;
        state.executions_submitted = state
            .executions_submitted
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("coverage telemetry submission counter overflow"))?;
        Ok(())
    }

    pub fn record_completion(&self, edge: &[u8], memory: &[u8]) -> anyhow::Result<()> {
        if edge.len() != EDGES_MAP_SIZE {
            anyhow::bail!(
                "invalid edge map length: expected {EDGES_MAP_SIZE} bytes, got {}",
                edge.len()
            );
        }
        if memory.len() != SIMT_MEMCOV_STORAGE_SIZE {
            anyhow::bail!(
                "invalid memory map length: expected {SIMT_MEMCOV_STORAGE_SIZE} bytes, got {}",
                memory.len()
            );
        }

        let mut state = self.lock_state()?;
        let next_completed = state
            .executions_completed
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("coverage telemetry completion counter overflow"))?;
        let cfg_added = merge_coverage_map(edge, &mut state.cfg_map).map_err(anyhow::Error::msg)?;
        let (memory_features, thread_activity) = memory.split_at(THREAD_ACTIVITY_BYTE_BASE);
        let (cumulative_memory, cumulative_thread_activity) =
            state.memory_map.split_at_mut(THREAD_ACTIVITY_BYTE_BASE);
        let memory_added =
            merge_coverage_map(memory_features, cumulative_memory).map_err(anyhow::Error::msg)?;
        let thread_activity_added = merge_coverage_map(thread_activity, cumulative_thread_activity)
            .map_err(anyhow::Error::msg)?;
        state.executions_completed = next_completed;
        state.cfg_sites = state
            .cfg_sites
            .checked_add(cfg_added)
            .ok_or_else(|| anyhow::anyhow!("coverage telemetry CFG counter overflow"))?;
        state.memory_features = state
            .memory_features
            .checked_add(memory_added)
            .ok_or_else(|| anyhow::anyhow!("coverage telemetry memory counter overflow"))?;
        state.thread_activity_features = state
            .thread_activity_features
            .checked_add(thread_activity_added)
            .ok_or_else(|| {
                anyhow::anyhow!("coverage telemetry thread-activity counter overflow")
            })?;
        if cfg_added != 0 || memory_added != 0 || thread_activity_added != 0 {
            let snapshot = count_snapshot_from_state(&state)?;
            if let Some(sink) = &state.event_sink {
                sink.sender
                    .send(CoverageLogCommand::Change {
                        timestamp_s: sink.started_at.elapsed().as_secs_f64(),
                        snapshot,
                    })
                    .map_err(|_| anyhow::anyhow!("coverage telemetry writer disconnected"))?;
            }
        }
        Ok(())
    }

    pub fn count_snapshot(&self) -> anyhow::Result<CoverageCountSnapshot> {
        let state = self.lock_state()?;
        count_snapshot_from_state(&state)
    }

    pub fn snapshot(&self) -> anyhow::Result<CoverageSnapshot> {
        let state = self.lock_state()?;
        let feedback_features_total = feature_total(
            state.cfg_sites,
            state.memory_features,
            state.thread_activity_features,
        )?;
        Ok(CoverageSnapshot {
            executions_submitted: state.executions_submitted,
            executions_completed: state.executions_completed,
            cfg_sites: state.cfg_sites,
            memory_features: state.memory_features,
            thread_activity_features: state.thread_activity_features,
            feedback_features_total,
            memory_map_hex: encode_lower_hex(&state.memory_map),
        })
    }

    fn lock_state(&self) -> anyhow::Result<std::sync::MutexGuard<'_, CoverageTelemetryState>> {
        self.state
            .lock()
            .map_err(|_| anyhow::anyhow!("coverage telemetry state lock is poisoned"))
    }

    fn close_event_sink(&self, final_sample: bool) -> anyhow::Result<()> {
        let mut state = self.lock_state()?;
        let Some(sink) = state.event_sink.take() else {
            return Ok(());
        };
        let command = if final_sample {
            CoverageLogCommand::Finish {
                timestamp_s: sink.started_at.elapsed().as_secs_f64(),
                snapshot: count_snapshot_from_state(&state)?,
            }
        } else {
            CoverageLogCommand::Cancel
        };
        sink.sender
            .send(command)
            .map_err(|_| anyhow::anyhow!("coverage telemetry writer disconnected"))
    }
}

fn count_snapshot_from_state(
    state: &CoverageTelemetryState,
) -> anyhow::Result<CoverageCountSnapshot> {
    Ok(CoverageCountSnapshot {
        executions_submitted: state.executions_submitted,
        executions_completed: state.executions_completed,
        cfg_sites: state.cfg_sites,
        memory_features: state.memory_features,
        thread_activity_features: state.thread_activity_features,
        feedback_features_total: feature_total(
            state.cfg_sites,
            state.memory_features,
            state.thread_activity_features,
        )?,
    })
}

fn feature_total(cfg: u64, memory: u64, thread_activity: u64) -> anyhow::Result<u64> {
    cfg.checked_add(memory)
        .and_then(|subtotal| subtotal.checked_add(thread_activity))
        .ok_or_else(|| anyhow::anyhow!("coverage telemetry total feature counter overflow"))
}

fn encode_lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

#[derive(Serialize)]
struct CoverageLogRecord {
    schema_version: u32,
    sequence: u64,
    timestamp_s: f64,
    final_sample: bool,
    executions_submitted: u64,
    executions_completed: u64,
    cfg_sites: u64,
    memory_features: u64,
    thread_activity_features: u64,
    feedback_features_total: u64,
    cfg_sites_delta: u64,
    memory_features_delta: u64,
    thread_activity_features_delta: u64,
    feedback_features_total_delta: u64,
}

impl CoverageLogRecord {
    fn from_snapshot(
        sequence: u64,
        timestamp_s: f64,
        final_sample: bool,
        previous: CoverageCountSnapshot,
        snapshot: CoverageCountSnapshot,
    ) -> anyhow::Result<Self> {
        let cfg_sites_delta = counter_delta(snapshot.cfg_sites, previous.cfg_sites, "CFG")?;
        let memory_features_delta =
            counter_delta(snapshot.memory_features, previous.memory_features, "memory")?;
        let thread_activity_features_delta = counter_delta(
            snapshot.thread_activity_features,
            previous.thread_activity_features,
            "thread-activity",
        )?;
        let feedback_features_total_delta = feature_total(
            cfg_sites_delta,
            memory_features_delta,
            thread_activity_features_delta,
        )?;
        Ok(Self {
            schema_version: 1,
            sequence,
            timestamp_s,
            final_sample,
            executions_submitted: snapshot.executions_submitted,
            executions_completed: snapshot.executions_completed,
            cfg_sites: snapshot.cfg_sites,
            memory_features: snapshot.memory_features,
            thread_activity_features: snapshot.thread_activity_features,
            feedback_features_total: snapshot.feedback_features_total,
            cfg_sites_delta,
            memory_features_delta,
            thread_activity_features_delta,
            feedback_features_total_delta,
        })
    }
}

fn counter_delta(current: u64, previous: u64, label: &str) -> anyhow::Result<u64> {
    current.checked_sub(previous).ok_or_else(|| {
        anyhow::anyhow!("coverage telemetry {label} counter regressed from {previous} to {current}")
    })
}

/// Owns the background JSONL writer for a [`CoverageTelemetry`] recorder.
pub struct CoverageLogSession {
    started_at: Instant,
    telemetry: CoverageTelemetry,
    event_sink_active: bool,
    writer: Option<thread::JoinHandle<anyhow::Result<()>>>,
    finish_error: Option<String>,
}

#[derive(Clone, Copy, Debug)]
enum CoverageLogCommand {
    Change {
        timestamp_s: f64,
        snapshot: CoverageCountSnapshot,
    },
    Finish {
        timestamp_s: f64,
        snapshot: CoverageCountSnapshot,
    },
    Cancel,
}

impl CoverageLogSession {
    pub fn start(path: impl AsRef<Path>, telemetry: CoverageTelemetry) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let file = File::create(path).with_context(|| {
            format!("failed to create coverage telemetry log {}", path.display())
        })?;
        let started_at = Instant::now();
        let baseline = telemetry.count_snapshot()?;
        if baseline != CoverageCountSnapshot::default() {
            anyhow::bail!("coverage telemetry session must start before any execution");
        }
        let mut file = file;
        let mut sequence = 0;
        write_sample(
            &mut file,
            &mut sequence,
            0.0,
            false,
            CoverageCountSnapshot::default(),
            baseline,
        )?;
        let (command_sender, command_receiver) = mpsc::channel();
        {
            let mut state = telemetry.lock_state()?;
            if state.event_sink.is_some() {
                anyhow::bail!("coverage telemetry already has an active log session");
            }
            if count_snapshot_from_state(&state)? != CoverageCountSnapshot::default() {
                anyhow::bail!("coverage telemetry session must start before any execution");
            }
            state.event_sink = Some(CoverageEventSink {
                started_at,
                sender: command_sender,
            });
        }
        let writer = match thread::Builder::new()
            .name("coverage-telemetry".to_string())
            .spawn(move || run_log_writer(file, command_receiver, sequence, baseline))
        {
            Ok(writer) => writer,
            Err(error) => {
                let _ = telemetry.close_event_sink(false);
                return Err(error).context("failed to start coverage telemetry writer thread");
            }
        };

        Ok(Self {
            started_at,
            telemetry,
            event_sink_active: true,
            writer: Some(writer),
            finish_error: None,
        })
    }

    /// Request one final sample and wait for the writer to persist it.
    ///
    /// It is safe to call this repeatedly. A previous writer failure is
    /// returned again as a clear error rather than panicking.
    pub fn finish(&mut self) -> anyhow::Result<()> {
        self.close(true)
    }

    /// Stop and join the writer without marking the campaign successful.
    ///
    /// This is used for failed or abandoned campaigns. Periodic samples that
    /// were already flushed remain valid, but no `final_sample=true` record is
    /// emitted.
    pub fn cancel(&mut self) -> anyhow::Result<()> {
        self.close(false)
    }

    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    fn close(&mut self, final_sample: bool) -> anyhow::Result<()> {
        let close_result = if self.event_sink_active {
            self.event_sink_active = false;
            self.telemetry.close_event_sink(final_sample)
        } else {
            Ok(())
        };

        let Some(writer) = self.writer.take() else {
            return match &self.finish_error {
                Some(message) => Err(anyhow::anyhow!(
                    "coverage telemetry writer previously failed: {message}"
                )),
                None => close_result,
            };
        };
        let result = writer.join();
        match result {
            Ok(Ok(())) => close_result,
            Ok(Err(error)) => {
                let message = error.to_string();
                self.finish_error = Some(message.clone());
                Err(anyhow::anyhow!(
                    "coverage telemetry writer failed: {message}"
                ))
            }
            Err(_) => {
                let message = "coverage telemetry writer thread panicked".to_string();
                self.finish_error = Some(message.clone());
                Err(anyhow::anyhow!(message))
            }
        }
    }
}

impl Drop for CoverageLogSession {
    fn drop(&mut self) {
        let _ = self.cancel();
    }
}

fn run_log_writer(
    mut file: File,
    command_receiver: mpsc::Receiver<CoverageLogCommand>,
    mut sequence: u64,
    mut last_emitted: CoverageCountSnapshot,
) -> anyhow::Result<()> {
    while let Ok(command) = command_receiver.recv() {
        match command {
            CoverageLogCommand::Change {
                timestamp_s,
                snapshot,
            } => {
                if !snapshot.has_coverage_change_from(last_emitted) {
                    anyhow::bail!("coverage telemetry change event has no coverage delta");
                }
                write_sample(
                    &mut file,
                    &mut sequence,
                    timestamp_s,
                    false,
                    last_emitted,
                    snapshot,
                )?;
                last_emitted = snapshot;
            }
            CoverageLogCommand::Finish {
                timestamp_s,
                snapshot,
            } => {
                write_sample(
                    &mut file,
                    &mut sequence,
                    timestamp_s,
                    true,
                    last_emitted,
                    snapshot,
                )?;
                return Ok(());
            }
            CoverageLogCommand::Cancel => return Ok(()),
        }
    }
    Ok(())
}

fn write_sample(
    file: &mut File,
    sequence: &mut u64,
    timestamp_s: f64,
    final_sample: bool,
    previous: CoverageCountSnapshot,
    snapshot: CoverageCountSnapshot,
) -> anyhow::Result<()> {
    let record =
        CoverageLogRecord::from_snapshot(*sequence, timestamp_s, final_sample, previous, snapshot)?;
    serde_json::to_writer(&mut *file, &record).context("failed to serialize coverage telemetry")?;
    file.write_all(b"\n")
        .context("failed to terminate coverage telemetry JSONL record")?;
    file.flush()
        .context("failed to flush coverage telemetry JSONL record")?;
    *sequence = sequence
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("coverage telemetry sequence overflow"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::atomic::{AtomicU64, Ordering},
        thread,
        time::{Duration, Instant},
    };

    use super::{CoverageLogSession, CoverageTelemetry, THREAD_ACTIVITY_BYTE_BASE};
    use crate::cuda_backend::{EDGES_MAP_SIZE, SIMT_MEMCOV_STORAGE_SIZE};

    static NEXT_TEST_FILE: AtomicU64 = AtomicU64::new(0);

    fn test_log_path() -> PathBuf {
        let nonce = NEXT_TEST_FILE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "cuda-fuzzer-coverage-telemetry-{}-{nonce}.jsonl",
            std::process::id()
        ))
    }

    fn wait_for_log_records(path: &Path, expected: usize) {
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let records = fs::read_to_string(path).unwrap().lines().count();
            if records >= expected {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {expected} coverage records"
            );
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn record_completion_rejects_invalid_map_lengths_without_counting_completion() {
        let telemetry = CoverageTelemetry::new();
        let edges = vec![0; EDGES_MAP_SIZE - 1];
        let memory = vec![0; SIMT_MEMCOV_STORAGE_SIZE];

        let error = telemetry.record_completion(&edges, &memory).unwrap_err();
        assert!(error.to_string().contains("invalid edge map length"));
        assert_eq!(telemetry.snapshot().unwrap().executions_completed, 0);

        let edges = vec![0; EDGES_MAP_SIZE];
        let memory = vec![0; SIMT_MEMCOV_STORAGE_SIZE - 1];
        let error = telemetry.record_completion(&edges, &memory).unwrap_err();
        assert!(error.to_string().contains("invalid memory map length"));
        assert_eq!(telemetry.snapshot().unwrap().executions_completed, 0);
    }

    #[test]
    fn record_completion_unions_cfg_bytes_and_suppresses_duplicate_bits() {
        let telemetry = CoverageTelemetry::new();
        let mut first_edges = vec![0; EDGES_MAP_SIZE];
        let mut second_edges = vec![0; EDGES_MAP_SIZE];
        let memory = vec![0; SIMT_MEMCOV_STORAGE_SIZE];
        first_edges[17] = 0b0000_0011;
        second_edges[17] = 0b0000_0101;
        second_edges[33] = 0b1000_0000;

        telemetry.record_completion(&first_edges, &memory).unwrap();
        telemetry.record_completion(&second_edges, &memory).unwrap();
        telemetry.record_completion(&second_edges, &memory).unwrap();

        let snapshot = telemetry.snapshot().unwrap();
        assert_eq!(snapshot.executions_completed, 3);
        assert_eq!(snapshot.cfg_sites, 4);
        assert_eq!(snapshot.memory_features, 0);
        assert_eq!(snapshot.thread_activity_features, 0);
        assert_eq!(snapshot.feedback_features_total, 4);
    }

    #[test]
    fn record_completion_partitions_memory_and_thread_activity_bits() {
        let telemetry = CoverageTelemetry::new();
        let edges = vec![0; EDGES_MAP_SIZE];
        let mut memory = vec![0; SIMT_MEMCOV_STORAGE_SIZE];
        memory[THREAD_ACTIVITY_BYTE_BASE - 1] = 0b1000_0000;
        memory[THREAD_ACTIVITY_BYTE_BASE] = 0b0000_0001;

        telemetry.record_completion(&edges, &memory).unwrap();

        let snapshot = telemetry.snapshot().unwrap();
        assert_eq!(snapshot.memory_features, 1);
        assert_eq!(snapshot.thread_activity_features, 1);
        assert_eq!(snapshot.feedback_features_total, 2);
    }

    #[test]
    fn count_snapshot_tracks_only_new_union_features() {
        let telemetry = CoverageTelemetry::new();
        let mut first_edges = vec![0; EDGES_MAP_SIZE];
        let mut second_edges = vec![0; EDGES_MAP_SIZE];
        let mut first_memory = vec![0; SIMT_MEMCOV_STORAGE_SIZE];
        let mut second_memory = vec![0; SIMT_MEMCOV_STORAGE_SIZE];
        first_edges[7] = 0b0000_0011;
        second_edges[7] = 0b0000_0101;
        first_memory[0] = 0b0000_0001;
        second_memory[0] = 0b0000_0011;
        first_memory[THREAD_ACTIVITY_BYTE_BASE] = 0b0000_0001;
        second_memory[THREAD_ACTIVITY_BYTE_BASE] = 0b0000_0011;

        let baseline = telemetry.count_snapshot().unwrap();
        telemetry
            .record_completion(&first_edges, &first_memory)
            .unwrap();
        let first = telemetry.count_snapshot().unwrap();
        telemetry
            .record_completion(&second_edges, &second_memory)
            .unwrap();
        let second = telemetry.count_snapshot().unwrap();
        telemetry
            .record_completion(&second_edges, &second_memory)
            .unwrap();
        let duplicate = telemetry.count_snapshot().unwrap();

        assert_eq!(baseline.cfg_sites, 0);
        assert_eq!(baseline.memory_features, 0);
        assert_eq!(baseline.thread_activity_features, 0);
        assert_eq!(first.cfg_sites, 2);
        assert_eq!(first.memory_features, 1);
        assert_eq!(first.thread_activity_features, 1);
        assert_eq!(second.cfg_sites, 3);
        assert_eq!(second.memory_features, 2);
        assert_eq!(second.thread_activity_features, 2);
        assert_eq!(duplicate.cfg_sites, second.cfg_sites);
        assert_eq!(duplicate.memory_features, second.memory_features);
        assert_eq!(
            duplicate.thread_activity_features,
            second.thread_activity_features
        );
        assert_eq!(duplicate.executions_completed, 3);
    }

    #[test]
    fn counters_are_monotonic_and_completed_work_is_independent_of_submissions() {
        let telemetry = CoverageTelemetry::new();
        let edges = vec![0; EDGES_MAP_SIZE];
        let memory = vec![0; SIMT_MEMCOV_STORAGE_SIZE];

        telemetry.record_submission().unwrap();
        telemetry.record_submission().unwrap();
        let before_completion = telemetry.snapshot().unwrap();
        telemetry.record_completion(&edges, &memory).unwrap();
        let after_completion = telemetry.snapshot().unwrap();

        assert_eq!(before_completion.executions_submitted, 2);
        assert_eq!(before_completion.executions_completed, 0);
        assert_eq!(after_completion.executions_submitted, 2);
        assert_eq!(after_completion.executions_completed, 1);
    }

    #[test]
    fn log_session_writes_monotonic_baseline_and_one_final_sample() {
        let path = test_log_path();
        let telemetry = CoverageTelemetry::new();
        let edges = vec![0; EDGES_MAP_SIZE];
        let memory = vec![0; SIMT_MEMCOV_STORAGE_SIZE];
        let mut session = CoverageLogSession::start(&path, telemetry.clone()).unwrap();

        telemetry.record_submission().unwrap();
        telemetry.record_completion(&edges, &memory).unwrap();
        session.finish().unwrap();
        session.finish().unwrap();

        let records: Vec<serde_json::Value> = fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        fs::remove_file(path).unwrap();

        assert_eq!(records.len(), 2, "expected baseline plus final sample");
        assert_eq!(
            records
                .iter()
                .filter(|record| record["final_sample"] == true)
                .count(),
            1
        );
        assert_eq!(records.last().unwrap()["final_sample"], true);
        for (sequence, record) in records.iter().enumerate() {
            assert_eq!(record["schema_version"], 1);
            assert_eq!(record["sequence"], sequence as u64);
            assert!(record.get("memory_map_hex").is_none());
            if sequence > 0 {
                assert!(
                    record["timestamp_s"].as_f64().unwrap()
                        >= records[sequence - 1]["timestamp_s"].as_f64().unwrap()
                );
                assert!(
                    record["executions_submitted"].as_u64().unwrap()
                        >= records[sequence - 1]["executions_submitted"]
                            .as_u64()
                            .unwrap()
                );
                assert!(
                    record["executions_completed"].as_u64().unwrap()
                        >= records[sequence - 1]["executions_completed"]
                            .as_u64()
                            .unwrap()
                );
            }
        }
    }

    #[test]
    fn log_session_writes_only_baseline_coverage_changes_and_final_state() {
        let path = test_log_path();
        let telemetry = CoverageTelemetry::new();
        let mut edges = vec![0; EDGES_MAP_SIZE];
        let mut memory = vec![0; SIMT_MEMCOV_STORAGE_SIZE];
        edges[3] = 0b0000_0001;
        memory[5] = 0b0000_0011;
        memory[THREAD_ACTIVITY_BYTE_BASE] = 0b0000_0100;
        let mut session = CoverageLogSession::start(&path, telemetry.clone()).unwrap();

        telemetry.record_submission().unwrap();
        telemetry.record_completion(&edges, &memory).unwrap();
        wait_for_log_records(&path, 2);
        session.finish().unwrap();

        let records: Vec<serde_json::Value> = fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        fs::remove_file(path).unwrap();

        assert_eq!(records.len(), 3, "expected baseline, change, and final");
        for (sequence, record) in records.iter().enumerate() {
            assert_eq!(record["schema_version"], 1);
            assert_eq!(record["sequence"], sequence as u64);
            assert!(record.get("memory_map_hex").is_none());
        }
        assert_eq!(records[0]["cfg_sites"], 0);
        assert_eq!(records[0]["cfg_sites_delta"], 0);
        assert_eq!(records[1]["cfg_sites"], 1);
        assert_eq!(records[1]["cfg_sites_delta"], 1);
        assert_eq!(records[1]["memory_features"], 2);
        assert_eq!(records[1]["memory_features_delta"], 2);
        assert_eq!(records[1]["thread_activity_features"], 1);
        assert_eq!(records[1]["thread_activity_features_delta"], 1);
        assert_eq!(records[1]["feedback_features_total_delta"], 4);
        assert_eq!(records[2]["final_sample"], true);
        assert_eq!(records[2]["cfg_sites_delta"], 0);
        assert_eq!(records[2]["memory_features_delta"], 0);
        assert_eq!(records[2]["thread_activity_features_delta"], 0);
        assert_eq!(records[2]["feedback_features_total_delta"], 0);
    }

    #[test]
    fn completion_events_preserve_exact_execution_boundaries_without_polling() {
        let path = test_log_path();
        let telemetry = CoverageTelemetry::new();
        let mut first_edges = vec![0; EDGES_MAP_SIZE];
        let mut second_edges = vec![0; EDGES_MAP_SIZE];
        let mut first_memory = vec![0; SIMT_MEMCOV_STORAGE_SIZE];
        let mut second_memory = vec![0; SIMT_MEMCOV_STORAGE_SIZE];
        first_edges[3] = 0b0000_0001;
        second_edges[3] = 0b0000_0011;
        first_memory[5] = 0b0000_0001;
        second_memory[5] = 0b0000_0011;
        let mut session = CoverageLogSession::start(&path, telemetry.clone()).unwrap();

        telemetry.record_submission().unwrap();
        telemetry
            .record_completion(&first_edges, &first_memory)
            .unwrap();
        telemetry.record_submission().unwrap();
        telemetry
            .record_completion(&second_edges, &second_memory)
            .unwrap();
        telemetry.record_submission().unwrap();
        telemetry
            .record_completion(&second_edges, &second_memory)
            .unwrap();
        session.finish().unwrap();

        let records: Vec<serde_json::Value> = fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        fs::remove_file(path).unwrap();

        assert_eq!(records.len(), 4, "baseline, two changes, and final");
        assert!(records.iter().all(|record| record["schema_version"] == 1));
        assert_eq!(records[1]["executions_completed"], 1);
        assert_eq!(records[1]["cfg_sites_delta"], 1);
        assert_eq!(records[1]["memory_features_delta"], 1);
        assert_eq!(records[2]["executions_completed"], 2);
        assert_eq!(records[2]["cfg_sites_delta"], 1);
        assert_eq!(records[2]["memory_features_delta"], 1);
        assert_eq!(records[3]["executions_completed"], 3);
        assert_eq!(records[3]["cfg_sites_delta"], 0);
        assert_eq!(records[3]["memory_features_delta"], 0);
        assert_eq!(records[3]["final_sample"], true);
    }

    #[test]
    fn full_snapshot_serializes_the_cumulative_memory_map_as_exact_lowercase_hex() {
        let telemetry = CoverageTelemetry::new();
        let edges = vec![0; EDGES_MAP_SIZE];
        let mut first_memory = vec![0; SIMT_MEMCOV_STORAGE_SIZE];
        let mut second_memory = vec![0; SIMT_MEMCOV_STORAGE_SIZE];
        first_memory[0] = 0x81;
        first_memory[THREAD_ACTIVITY_BYTE_BASE] = 0x02;
        second_memory[0] = 0x04;
        second_memory[SIMT_MEMCOV_STORAGE_SIZE - 1] = 0xa0;
        telemetry.record_completion(&edges, &first_memory).unwrap();
        telemetry.record_completion(&edges, &second_memory).unwrap();
        let snapshot = telemetry.snapshot().unwrap();
        let memory_map_hex = snapshot.memory_map_hex;

        assert_eq!(memory_map_hex.len(), SIMT_MEMCOV_STORAGE_SIZE * 2);
        assert!(memory_map_hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
        assert_eq!(&memory_map_hex[..2], "85");
        assert_eq!(
            &memory_map_hex[THREAD_ACTIVITY_BYTE_BASE * 2..THREAD_ACTIVITY_BYTE_BASE * 2 + 2],
            "02"
        );
        assert_eq!(&memory_map_hex[memory_map_hex.len() - 2..], "a0");
        assert_eq!(snapshot.memory_features, 3);
        assert_eq!(snapshot.thread_activity_features, 3);
    }

    #[test]
    fn final_timestamp_uses_the_same_clock_as_the_campaign_deadline() {
        let path = test_log_path();
        let telemetry = CoverageTelemetry::new();
        let mut session = CoverageLogSession::start(&path, telemetry).unwrap();
        let duration = Duration::from_millis(20);

        while session.elapsed() < duration {
            thread::yield_now();
        }
        session.finish().unwrap();

        let contents = fs::read_to_string(&path).unwrap();
        let record: serde_json::Value =
            serde_json::from_str(contents.lines().last().unwrap()).unwrap();
        fs::remove_file(path).unwrap();
        assert!(record["timestamp_s"].as_f64().unwrap() >= duration.as_secs_f64());
    }

    #[test]
    fn cancelled_session_joins_writer_without_a_final_sample() {
        let path = test_log_path();
        let telemetry = CoverageTelemetry::new();
        let mut session = CoverageLogSession::start(&path, telemetry).unwrap();

        session.cancel().unwrap();

        let records: Vec<serde_json::Value> = fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert!(!records.is_empty(), "expected at least one periodic sample");
        assert!(records.iter().all(|record| record["final_sample"] == false));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn dropped_session_cancels_without_writing_a_final_sample() {
        let path = test_log_path();
        let telemetry = CoverageTelemetry::new();
        let session = CoverageLogSession::start(&path, telemetry).unwrap();

        drop(session);

        let contents = fs::read_to_string(&path).unwrap();
        assert!(contents
            .lines()
            .all(
                |line| serde_json::from_str::<serde_json::Value>(line).unwrap()["final_sample"]
                    == false
            ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn log_session_reports_file_creation_failures_without_panicking() {
        let telemetry = CoverageTelemetry::new();
        let result = CoverageLogSession::start(std::env::temp_dir(), telemetry);
        let error = match result {
            Ok(_) => panic!("a directory cannot be opened as a telemetry log"),
            Err(error) => error,
        };

        assert!(error
            .to_string()
            .contains("failed to create coverage telemetry log"));
    }

    #[test]
    fn log_session_rejects_any_nonzero_execution_baseline() {
        let edges = vec![0; EDGES_MAP_SIZE];
        let memory = vec![0; SIMT_MEMCOV_STORAGE_SIZE];
        for case in 0..3 {
            let path = test_log_path();
            let telemetry = CoverageTelemetry::new();
            if case != 1 {
                telemetry.record_submission().unwrap();
            }
            if case != 0 {
                telemetry.record_completion(&edges, &memory).unwrap();
            }

            let error = match CoverageLogSession::start(&path, telemetry) {
                Ok(_) => panic!("a nonzero execution baseline must be rejected"),
                Err(error) => error,
            };
            assert!(error
                .to_string()
                .contains("must start before any execution"));
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn poisoned_state_and_session_start_return_errors_without_panicking() {
        let path = test_log_path();
        let telemetry = CoverageTelemetry::new();
        let poison_target = telemetry.clone();
        let poisoner = thread::spawn(move || {
            let _guard = poison_target.state.lock().unwrap();
            panic!("poison coverage telemetry lock");
        });
        assert!(poisoner.join().is_err());

        assert!(telemetry
            .record_submission()
            .unwrap_err()
            .to_string()
            .contains("coverage telemetry state lock is poisoned"));

        let error = match CoverageLogSession::start(&path, telemetry) {
            Ok(_) => panic!("a poisoned telemetry state cannot start a log session"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("coverage telemetry state lock is poisoned"));
        fs::remove_file(path).unwrap();
    }
}
