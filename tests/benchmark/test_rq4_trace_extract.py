import sqlite3
import tempfile
import unittest
from pathlib import Path


def create_trace(path: Path, *, omit: str | None = None) -> None:
    connection = sqlite3.connect(path)
    statements = {
        "StringIds": "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)",
        "ThreadNames": "CREATE TABLE ThreadNames (globalTid INTEGER, nameId INTEGER, priority INTEGER)",
        "CUPTI_ACTIVITY_KIND_RUNTIME": "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER, nameId INTEGER)",
        "CUPTI_ACTIVITY_KIND_KERNEL": "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, correlationId INTEGER, shortName INTEGER, demangledName INTEGER)",
        "CUPTI_ACTIVITY_KIND_MEMCPY": "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, correlationId INTEGER, bytes INTEGER, copyKind INTEGER)",
    }
    for name, statement in statements.items():
        if name != omit:
            connection.execute(statement)
    if omit is None:
        connection.executemany(
            "INSERT INTO StringIds(id, value) VALUES (?, ?)",
            [
                (1, "RAPID2-Coll-0"),
                (2, "RAPID2-Disp-0"),
                (3, "fuzzer_async"),
                (10, "cuMemcpyDtoHAsync_v2"),
                (11, "cuMemcpyHtoDAsync_v2"),
                (12, "cuLaunchKernel"),
                (20, "target_kernel"),
            ],
        )
        connection.executemany(
            "INSERT INTO ThreadNames(globalTid, nameId, priority) VALUES (?, ?, ?)",
            [(101, 1, 0), (102, 2, 0), (103, 3, 0)],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME(start, end, globalTid, correlationId, nameId) VALUES (?, ?, ?, ?, ?)",
            [
                (100, 180, 101, 1, 10),
                (150, 250, 102, 2, 11),
                (200, 220, 103, 3, 12),
            ],
        )
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start, end, correlationId, shortName, demangledName) VALUES (?, ?, ?, ?, ?)",
            (230, 330, 3, 20, 20),
        )
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY(start, end, correlationId, bytes, copyKind) VALUES (?, ?, ?, ?, ?)",
            (110, 160, 1, 4096, 2),
        )
    connection.commit()
    connection.close()


class RQ4TraceExtractTests(unittest.TestCase):
    def test_extracts_thread_scoped_cuda_api_durations(self) -> None:
        from benchmark.rq4.trace_extract import extract_trace

        with tempfile.TemporaryDirectory() as temp_dir:
            trace = Path(temp_dir) / "trace.sqlite"
            create_trace(trace)

            evidence = extract_trace(trace)

        self.assertEqual(evidence.capture_start_ns, 100)
        self.assertEqual(evidence.capture_end_ns, 330)
        self.assertEqual(evidence.thread_names[101], "RAPID2-Coll-0")
        self.assertEqual(
            evidence.cuda_api_time_ns["RAPID2-Coll-0"]["memcpy"], 80
        )
        self.assertEqual(
            evidence.cuda_api_time_ns["RAPID2-Disp-0"]["memcpy"], 100
        )
        self.assertEqual(evidence.cuda_api_time_ns["fuzzer_async"]["launch"], 20)

    def test_retains_kernel_names_and_correlated_launcher_thread(self) -> None:
        from benchmark.rq4.trace_extract import extract_trace

        with tempfile.TemporaryDirectory() as temp_dir:
            trace = Path(temp_dir) / "trace.sqlite"
            create_trace(trace)

            evidence = extract_trace(trace)

        self.assertEqual(len(evidence.kernels), 1)
        kernel = evidence.kernels[0]
        self.assertEqual(kernel.name, "target_kernel")
        self.assertEqual(kernel.duration_ns, 100)
        self.assertEqual(kernel.launcher_thread, "fuzzer_async")
        self.assertEqual(evidence.gpu_memcpy_time_ns, 50)

    def test_missing_required_table_is_rejected(self) -> None:
        from benchmark.rq4.trace_extract import TraceSchemaError, extract_trace

        with tempfile.TemporaryDirectory() as temp_dir:
            trace = Path(temp_dir) / "trace.sqlite"
            create_trace(trace, omit="ThreadNames")

            with self.assertRaisesRegex(TraceSchemaError, "ThreadNames"):
                extract_trace(trace)

    def test_persistent_trace_may_omit_pre_capture_kernel_launch_table(self) -> None:
        from benchmark.rq4.trace_extract import extract_trace

        with tempfile.TemporaryDirectory() as temp_dir:
            trace = Path(temp_dir) / "trace.sqlite"
            create_trace(trace)
            connection = sqlite3.connect(trace)
            connection.execute("DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL")
            connection.commit()
            connection.close()

            evidence = extract_trace(trace)

        self.assertEqual(evidence.kernels, ())
        self.assertGreater(evidence.capture_end_ns, evidence.capture_start_ns)

    def test_compact_export_removes_transient_sqlite_after_json_is_durable(self) -> None:
        from benchmark.rq4.trace_extract import write_compact_evidence

        with tempfile.TemporaryDirectory() as temp_dir:
            trace = Path(temp_dir) / "trace.sqlite"
            compact = Path(temp_dir) / "trace.evidence.json"
            create_trace(trace)

            write_compact_evidence(trace, compact, keep_sqlite=False)

            self.assertFalse(trace.exists())
            self.assertTrue(compact.exists())
            text = compact.read_text(encoding="utf-8")
            self.assertIn('"RAPID2-Coll-0"', text)
            self.assertIn('"target_kernel"', text)


if __name__ == "__main__":
    unittest.main()
