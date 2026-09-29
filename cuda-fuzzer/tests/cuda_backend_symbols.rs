use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use cuda_fuzzer::cuda_backend::{
    AsyncCudaBackend, OrderedCudaBackend, SyncCudaBackend, EDGES_MAP_SIZE, SIMT_MEMCOV_STORAGE_SIZE,
};

static NEXT_FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

struct SharedLibraryFixture {
    root: PathBuf,
    library: PathBuf,
}

impl SharedLibraryFixture {
    fn compile(name: &str, source: &str) -> Self {
        let fixture_id = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "rapid-cuda-backend-symbols-{}-{fixture_id}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("failed to create fixture directory");
        let source_path = root.join(format!("{name}.c"));
        let library = root.join(format!("lib{name}.so"));
        fs::write(&source_path, source).expect("failed to write fixture source");

        let output = Command::new("cc")
            .args(["-shared", "-fPIC"])
            .arg(&source_path)
            .arg("-o")
            .arg(&library)
            .output()
            .expect("failed to invoke cc");
        assert!(
            output.status.success(),
            "failed to compile fixture: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        Self { root, library }
    }

    fn path(&self) -> &Path {
        &self.library
    }
}

impl Drop for SharedLibraryFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn sync_backend_reports_missing_run_status_symbol() {
    let fixture = SharedLibraryFixture::compile(
        "sync_missing_status",
        r#"
            #include <stddef.h>
            #include <stdint.h>
            uint8_t libafl_cov_map[65536];
            uint8_t libafl_simt_memcov_bits[8192];
            void libafl_target(const uint8_t *input, size_t size) {
                (void)input;
                (void)size;
            }
        "#,
    );

    let error = match SyncCudaBackend::new(fixture.path()) {
        Ok(_) => panic!("sync backend unexpectedly accepted an incomplete ABI"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("libafl_get_last_run_status"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn sync_backend_explicit_stop_is_not_repeated_on_drop() {
    let marker_root = std::env::temp_dir().join(format!(
        "rapid-sync-backend-drop-{}-{}",
        std::process::id(),
        NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&marker_root).unwrap();
    let marker = marker_root.join("calls.txt");
    let source = format!(
        r#"
            #include <stddef.h>
            #include <stdint.h>
            #include <stdio.h>
            #include <string.h>
            uint8_t libafl_cov_map[65536];
            uint8_t libafl_simt_memcov_bits[8192];
            void libafl_target(const uint8_t *input, size_t size) {{
                (void)input;
                (void)size;
            }}
            void libafl_get_last_run_status(void *status) {{ memset(status, 0, 16); }}
            static void record(const char *name) {{
                FILE *stream = fopen("{}", "a");
                fputs(name, stream);
                fclose(stream);
            }}
            void libafl_wait(void) {{ record("wait\n"); }}
            void libafl_stop(void) {{ record("stop\n"); }}
        "#,
        marker.display()
    );
    let fixture = SharedLibraryFixture::compile("sync_optional_shutdown", &source);

    let mut backend = SyncCudaBackend::new(fixture.path()).unwrap();
    backend.stop();
    drop(backend);

    assert_eq!(fs::read_to_string(&marker).unwrap(), "wait\nstop\n");
    fs::remove_dir_all(marker_root).unwrap();
}

#[test]
fn async_backend_reports_missing_release_symbol() {
    let fixture = SharedLibraryFixture::compile(
        "async_missing_release",
        r#"
            #include <stddef.h>
            #include <stdint.h>
            uint8_t libafl_cov_map[65536];
            uint64_t libafl_submit_with_id(const uint8_t *input, size_t size) {
                (void)input;
                (void)size;
                return 1;
            }
            size_t libafl_poll_results(void *results, size_t max_count) {
                (void)results;
                (void)max_count;
                return 0;
            }
            void libafl_get_queue_counts(void *counts) { (void)counts; }
            void libafl_set_target_timeout_ms(uint64_t timeout_ms) {
                (void)timeout_ms;
            }
            void libafl_wait(void) {}
            void libafl_stop(void) {}
        "#,
    );

    let error = match AsyncCudaBackend::new(fixture.path()) {
        Ok(_) => panic!("async backend unexpectedly accepted an incomplete ABI"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("libafl_release_tasks"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn async_backend_requires_blocking_completion_notification() {
    let fixture = SharedLibraryFixture::compile(
        "async_missing_completion_wait",
        r#"
            #include <stddef.h>
            #include <stdint.h>
            uint8_t libafl_cov_map[65536];
            uint64_t libafl_submit_with_id(const uint8_t *input, size_t size) {
                (void)input;
                (void)size;
                return 1;
            }
            size_t libafl_poll_results(void *results, size_t max_count) {
                (void)results;
                (void)max_count;
                return 0;
            }
            size_t libafl_release_tasks(const uint64_t *ids, size_t count) {
                (void)ids;
                return count;
            }
            void libafl_get_queue_counts(void *counts) { (void)counts; }
            void libafl_set_target_timeout_ms(uint64_t timeout_ms) {
                (void)timeout_ms;
            }
            void libafl_wait(void) {}
            void libafl_stop(void) {}
        "#,
    );

    let error = match AsyncCudaBackend::new(fixture.path()) {
        Ok(_) => panic!("async backend unexpectedly accepted an ABI without completion wait"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("libafl_wait_for_completion"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn ordered_backend_reports_missing_release_symbol() {
    let fixture = SharedLibraryFixture::compile(
        "ordered_missing_release",
        r#"
            #include <stddef.h>
            #include <stdint.h>
            uint8_t libafl_cov_map[65536];
            uint64_t libafl_submit_with_id(const uint8_t *input, size_t size) {
                (void)input;
                (void)size;
                return 1;
            }
            size_t libafl_poll_results(void *results, size_t max_count) {
                (void)results;
                (void)max_count;
                return 0;
            }
            void libafl_get_ordered_queue_counts(void *counts) { (void)counts; }
            void libafl_wait(void) {}
            void libafl_stop(void) {}
        "#,
    );

    let error = match OrderedCudaBackend::new(fixture.path()) {
        Ok(_) => panic!("ordered backend unexpectedly accepted an incomplete ABI"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("libafl_release_tasks"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn ordered_backend_reports_missing_timeout_symbol() {
    let fixture = SharedLibraryFixture::compile(
        "ordered_missing_timeout",
        r#"
            #include <stddef.h>
            #include <stdint.h>
            uint8_t libafl_cov_map[65536];
            uint64_t libafl_submit_with_id(const uint8_t *input, size_t size) {
                (void)input;
                (void)size;
                return 1;
            }
            size_t libafl_poll_results(void *results, size_t max_count) {
                (void)results;
                (void)max_count;
                return 0;
            }
            size_t libafl_release_tasks(const uint64_t *ids, size_t count) {
                (void)ids;
                return count;
            }
            void libafl_get_ordered_queue_counts(void *counts) { (void)counts; }
            void libafl_wait(void) {}
            void libafl_stop(void) {}
        "#,
    );

    let error = match OrderedCudaBackend::new(fixture.path()) {
        Ok(_) => panic!("ordered backend unexpectedly accepted an incomplete timeout ABI"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("libafl_set_target_timeout_ms"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn ordered_backend_exposes_task_owned_results_counts_and_lifecycle() {
    let marker_root = std::env::temp_dir().join(format!(
        "rapid-ordered-backend-{}-{}",
        std::process::id(),
        NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&marker_root).unwrap();
    let marker = marker_root.join("calls.txt");
    let source = format!(
        r#"
            #include <stddef.h>
            #include <stdint.h>
            #include <stdio.h>

            typedef struct {{ uint16_t code, stage; uint32_t detail; }} RunStatus;
            typedef struct {{
                uint64_t task_id;
                uintptr_t input_ptr;
                const uint8_t *edge_ptr;
                const uint8_t *simt_memcov_ptr;
                uint32_t edge_size;
                uint32_t simt_memcov_size;
                RunStatus status;
                uint64_t exec_time_ns;
            }} TaskResult;
            typedef struct {{ size_t pending, completed, outstanding; }} Counts;

            uint8_t libafl_cov_map[65536];
            static uint8_t edges[65536];
            static uint8_t simt_memcov[8192];
            static int emitted;
            static uint64_t configured_timeout_ms;

            uint64_t libafl_submit_with_id(const uint8_t *input, size_t size) {{
                (void)input;
                (void)size;
                edges[17] = 0x71;
                simt_memcov[23] = 0x81;
                return 7;
            }}
            size_t libafl_poll_results(TaskResult *results, size_t max_count) {{
                if (emitted || !results || max_count == 0) return 0;
                results[0].task_id = 7;
                results[0].input_ptr = 0x1234;
                results[0].edge_ptr = edges;
                results[0].simt_memcov_ptr = simt_memcov;
                results[0].edge_size = 65536;
                results[0].simt_memcov_size = 8192;
                results[0].status = (RunStatus){{0, 0, 0}};
                results[0].exec_time_ns = 99;
                emitted = 1;
                return 1;
            }}
            size_t libafl_release_tasks(const uint64_t *ids, size_t count) {{
                return ids && count == 1 && ids[0] == 7 ? 1 : 0;
            }}
            void libafl_set_target_timeout_ms(uint64_t timeout_ms) {{
                configured_timeout_ms = timeout_ms;
            }}
            uint64_t configured_timeout(void) {{ return configured_timeout_ms; }}
            void libafl_get_ordered_queue_counts(Counts *counts) {{
                counts->pending = 1;
                counts->completed = 2;
                counts->outstanding = 3;
            }}
            static void record(const char *name) {{
                FILE *stream = fopen("{}", "a");
                fputs(name, stream);
                fclose(stream);
            }}
            void libafl_wait(void) {{ record("wait\n"); }}
            void libafl_stop(void) {{ record("stop\n"); }}
        "#,
        marker.display()
    );
    let fixture = SharedLibraryFixture::compile("ordered_complete", &source);

    let backend = OrderedCudaBackend::new(fixture.path()).unwrap();
    backend.set_target_timeout(Duration::from_millis(123));
    let probe_library = unsafe { libloading::Library::new(fixture.path()).unwrap() };
    let timeout_probe = unsafe {
        probe_library
            .get::<unsafe extern "C" fn() -> u64>(b"configured_timeout")
            .unwrap()
    };
    assert_eq!(unsafe { timeout_probe() }, 123);
    let task_id = backend.submit(&[1, 2, 3]).unwrap();
    assert_eq!(task_id.get(), 7);
    let results = backend.poll_completed_tasks(4);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].task_id, 7);
    assert_eq!(results[0].exec_time_ns, 99);
    assert_eq!(unsafe { results[0].edge_slice().unwrap() }[17], 0x71);
    assert_eq!(unsafe { results[0].simt_memcov_slice().unwrap() }[23], 0x81);
    assert_eq!(results[0].edge_size as usize, EDGES_MAP_SIZE);
    assert_eq!(
        results[0].simt_memcov_size as usize,
        SIMT_MEMCOV_STORAGE_SIZE
    );
    assert_eq!(backend.release_completed_tasks(&[7]), 1);
    assert_eq!(
        backend.get_queue_counts(),
        cuda_fuzzer::cuda_backend::OrderedQueueCounts {
            pending: 1,
            completed: 2,
            outstanding: 3,
        }
    );
    assert!(!backend.cov_map_ptr().is_null());
    drop(backend);

    assert_eq!(fs::read_to_string(&marker).unwrap(), "wait\nstop\n");
    fs::remove_dir_all(marker_root).unwrap();
}
