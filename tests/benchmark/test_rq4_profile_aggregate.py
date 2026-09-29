import csv
import json
import tempfile
import unittest
from pathlib import Path


def evidence() -> dict:
    return {
        "capture_start_ns": 0,
        "capture_end_ns": 1000,
        "thread_names": {
            "1": "fuzzer",
            "2": "RAPID-Sync",
            "3": "RAPID2-Coll-0",
            "4": "RAPID2-Disp-0",
        },
        "cuda_api_time_ns": {
            "fuzzer": {
                "allocation": 10,
                "free": 10,
                "memcpy": 20,
                "launch": 10,
                "synchronization": 300,
            },
            "RAPID-Sync": {"memcpy": 100},
            "RAPID2-Coll-0": {"memcpy": 120},
            "RAPID2-Disp-0": {"memcpy": 80},
        },
        "cuda_api_calls": {},
        "kernels": [
            {
                "start_ns": 100,
                "end_ns": 300,
                "correlation_id": 7,
                "name": "target_kernel",
                "launcher_thread": "fuzzer",
            }
        ],
        "gpu_memcpy_time_ns": 20,
    }


def profiling_records(
    cpu: dict, device: dict | None, *, wall_time_ns: int = 0
) -> list[dict]:
    records = [
        {
            "schema_version": 2,
            "record_type": "segment",
            "domain": "main_loop",
            "segment": segment,
            "unit": "ns",
            "count": 1,
            "total": wall_time_ns if segment == "other_idle" else 0,
            "max": wall_time_ns if segment == "other_idle" else 0,
        }
        for segment in (
            "submit",
            "poll",
            "coverage",
            "evaluate",
            "release",
            "scheduler_stage",
            "other_idle",
        )
    ]
    records.extend(
        (
            {
                "schema_version": 2,
                "record_type": "segment",
                "domain": "cpu_feedback",
                "segment": "predicate",
                "unit": "ns",
                "count": cpu["calls"],
                "total": cpu["predicate_ns"],
                "max": cpu["predicate_ns"],
            },
            {
                "schema_version": 2,
                "record_type": "segment",
                "domain": "cpu_feedback",
                "segment": "metadata",
                "unit": "ns",
                "count": cpu["calls"],
                "total": cpu["metadata_ns"],
                "max": cpu["metadata_ns"],
            },
        )
    )
    if device is not None:
        records.extend(
            {
                "schema_version": 2,
                "record_type": "segment",
                "domain": "device_kernel",
                "segment": segment,
                "unit": "cycles",
                "count": device["iterations"],
                "total": total,
                "max": None,
            }
            for segment, total in device.items()
            if segment != "iterations"
        )
    return records


def profile(configuration: str) -> dict:
    trace = evidence()
    if configuration in ("cufuzz", "libafl", "libafl-plus"):
        trace["cuda_api_time_ns"] = {
            "fuzzer": trace["cuda_api_time_ns"]["fuzzer"]
        }
    elif configuration == "gunit-sync":
        trace["cuda_api_time_ns"] = {
            "fuzzer": trace["cuda_api_time_ns"]["fuzzer"],
            "RAPID-Sync": trace["cuda_api_time_ns"]["RAPID-Sync"],
        }
    elif configuration == "gunit":
        trace["cuda_api_time_ns"] = {
            "fuzzer": trace["cuda_api_time_ns"]["fuzzer"],
            "RAPID2-Coll-0": trace["cuda_api_time_ns"]["RAPID2-Coll-0"],
            "RAPID2-Disp-0": trace["cuda_api_time_ns"]["RAPID2-Disp-0"],
        }
    row = {
        "mode": "fixed",
        "workload_id": "k",
        "configuration": configuration,
        "repetition": 1,
        "window_size": 2 if configuration in ("gunit-sync", "gunit") else None,
        "profile_seconds": 1,
        "benchmark": {"elapsed_ns": 1000},
        "trace_evidence_data": trace,
        "cuda_api": {},
        "gpu_kernel_time_ns": 200,
        "gpu_mem_time_ns": 20,
        "device_timing_cycles": None,
        "cpu_feedback_timing_ns": {
            "calls": 10,
            "predicate_ns": 20,
            "metadata_ns": 10,
            "total_ns": 30,
        },
    }
    if configuration in ("libafl", "libafl-plus", "gunit-sync", "gunit"):
        row["device_timing_cycles"] = {
            "iterations": 10,
            "idle": 0,
            "feedback_init": 20,
            "input_decode": 12,
            "feedback_prepare": 18,
            "target_execution": 300,
            "feedback_merge": 10,
            "signal": 5,
            "bookkeeping": 15,
        }
    if configuration in ("gunit-sync", "gunit"):
        row["trace_evidence_data"]["kernels"] = []
        row["device_timing_cycles"]["idle"] = 50
    row["profiling_records_data"] = profiling_records(
        row["cpu_feedback_timing_ns"], row["device_timing_cycles"]
    )
    return row


def mutating_profiles(
    *,
    repetitions: tuple[int, ...] = (1,),
    mutate_seconds: int = 30,
    wall_time_ns: int = 29_900_000_000,
) -> list[dict]:
    from benchmark.rq4.profile import API_CATEGORIES

    rows = []
    for repetition in repetitions:
        for configuration in (
            "cufuzz",
            "libafl",
            "libafl-plus",
            "gunit-sync",
            "gunit",
        ):
            row = profile(configuration)
            row.update(
                {
                    "mode": "mutating",
                    "repetition": repetition,
                    "fixed_seed": 1,
                    "mutate_seconds": mutate_seconds,
                    "mutating": {
                        "requested_seconds": mutate_seconds,
                        "corpus_size": 2,
                        "solutions": 0,
                        "executions": 1,
                        "mutation_calls": 1,
                        "coverage_nonzero_bytes": (
                            0 if configuration in ("cufuzz", "libafl") else 1
                        ),
                        "simt_memcov_nonzero_bits": 0,
                        "pending": 0,
                        "completed": 0,
                        "outstanding": 0,
                        "in_flight": 0,
                        "queued_submissions": 0,
                    },
                    "cuda_api": {
                        category: {"time_ns": 1, "calls": 1}
                        for category in API_CATEGORIES
                    },
                }
            )
            row.pop("benchmark")
            row["profiling_records_data"] = profiling_records(
                row["cpu_feedback_timing_ns"],
                row["device_timing_cycles"],
                wall_time_ns=wall_time_ns,
            )
            row["trace_evidence_data"]["capture_end_ns"] = wall_time_ns
            rows.append(row)
    return rows


class RQ4ProfileAggregateTests(unittest.TestCase):
    def test_serial_lane_uses_measured_components_and_residual_idle(self) -> None:
        from benchmark.rq4.profile_aggregate import overhead_lanes

        rows = overhead_lanes(profile("cufuzz"))
        values = {row["category"]: row["value"] for row in rows}

        self.assertEqual({row["lane"] for row in rows}, {"CuFuzz"})
        self.assertEqual(values["kernel_exec"], 200)
        self.assertEqual(values["feedback"], 30)
        self.assertEqual(values["memcpy"], 20)
        self.assertEqual(values["allocation"], 10)
        self.assertEqual(values["free"], 10)
        self.assertEqual(values["launch"], 10)
        self.assertEqual(values["idle"], 720)
        self.assertAlmostEqual(sum(row["share"] for row in rows), 1.0)

    def test_persistent_profiles_expand_to_device_and_host_lanes(self) -> None:
        from benchmark.rq4.profile_aggregate import overhead_lanes

        sync_rows = overhead_lanes(profile("gunit-sync"))
        async_rows = overhead_lanes(profile("gunit"))

        self.assertEqual(
            {row["lane"] for row in sync_rows},
            {"GUnit-s-GPU", "GUnit-s-CPU"},
        )
        self.assertEqual(
            {row["lane"] for row in async_rows},
            {"GUnit-GPU", "GUnit-Coll", "GUnit-Disp"},
        )
        gpu = {
            row["category"]: row["value"]
            for row in async_rows
            if row["lane"] == "GUnit-GPU"
        }
        self.assertEqual(gpu["kernel_exec"], 300)
        self.assertEqual(gpu["feedback"], 48)
        self.assertEqual(gpu["gpu_overhead"], 32)
        self.assertEqual(gpu["idle"], 50)
        coll = {
            row["category"]: row["value"]
            for row in async_rows
            if row["lane"] == "GUnit-Coll"
        }
        self.assertEqual(coll["memcpy"], 120)
        self.assertEqual(coll["feedback"], 0)
        self.assertEqual(coll["idle"], 880)
        sync_cpu = {
            row["category"]: row["value"]
            for row in sync_rows
            if row["lane"] == "GUnit-s-CPU"
        }
        self.assertEqual(sync_cpu["memcpy"], 100)
        self.assertEqual(sync_cpu["feedback"], 0)
        self.assertEqual(sync_cpu["idle"], 900)

    def test_concurrent_feedback_is_not_added_to_host_thread_lane(self) -> None:
        from benchmark.rq4.profile_aggregate import overhead_lanes

        row = profile("gunit-sync")
        denominator = 29_981_816_653
        concurrent_feedback = 35_176_145_536
        row["benchmark"]["elapsed_ns"] = denominator
        row["profiling_records_data"] = profiling_records(
            {
                "calls": 1,
                "predicate_ns": concurrent_feedback,
                "metadata_ns": 0,
                "total_ns": concurrent_feedback,
            },
            row["device_timing_cycles"],
        )

        rows = [
            item for item in overhead_lanes(row) if item["lane"] == "GUnit-s-CPU"
        ]
        values = {item["category"]: item["value"] for item in rows}
        self.assertEqual(values["memcpy"], 100)
        self.assertEqual(values["feedback"], 0)
        self.assertEqual(values["idle"], denominator - 100)

    def test_negative_residual_is_rejected_instead_of_clamped(self) -> None:
        from benchmark.rq4.profile_aggregate import overhead_lanes

        row = profile("cufuzz")
        row["benchmark"]["elapsed_ns"] = 100

        with self.assertRaisesRegex(RuntimeError, "negative residual"):
            overhead_lanes(row)

    def test_complete_primary_matrix_contains_exactly_eight_lanes(self) -> None:
        from benchmark.rq4.profile_aggregate import overhead_lanes

        rows = [
            lane
            for configuration in (
                "cufuzz",
                "libafl",
                "libafl-plus",
                "gunit-sync",
                "gunit",
            )
            for lane in overhead_lanes(profile(configuration))
        ]

        lanes = {row["lane"] for row in rows}
        self.assertEqual(
            lanes,
            {
                "CuFuzz",
                "LibAFL",
                "LibAFL+",
                "GUnit-s-GPU",
                "GUnit-s-CPU",
                "GUnit-GPU",
                "GUnit-Coll",
                "GUnit-Disp",
            },
        )
        for lane in lanes:
            lane_rows = [row for row in rows if row["lane"] == lane]
            self.assertAlmostEqual(sum(row["share"] for row in lane_rows), 1.0)

    def test_pilot_gate_checks_five_profiles_and_eight_lanes(self) -> None:
        from benchmark.rq4.profile_aggregate import validate_pilot

        profiles = [
            profile(configuration)
            for configuration in (
                "cufuzz",
                "libafl",
                "libafl-plus",
                "gunit-sync",
                "gunit",
            )
        ]

        summary = validate_pilot(profiles)

        self.assertEqual(summary["profiles"], 5)
        self.assertEqual(summary["lanes"], 8)
        self.assertNotEqual(summary["coll_thread_id"], summary["disp_thread_id"])

    def test_persistent_breakdown_excludes_iteration_counter_and_sums_to_one(self) -> None:
        from benchmark.rq4.profile_aggregate import persistent_breakdown

        profile = {
            "mode": "fixed",
            "workload_id": "k",
            "configuration": "gunit",
            "device_timing_cycles": {
                "iterations": 10,
                "idle": 10,
                "feedback_init": 20,
                "input_decode": 5,
                "feedback_prepare": 10,
                "target_execution": 30,
                "feedback_merge": 15,
                "signal": 5,
                "bookkeeping": 5,
            },
        }
        profile["profiling_records_data"] = profiling_records(
            {"calls": 0, "predicate_ns": 0, "metadata_ns": 0, "total_ns": 0},
            profile["device_timing_cycles"],
        )
        rows = persistent_breakdown(profile)
        self.assertEqual(sum(row["cycles"] for row in rows), 100)
        self.assertAlmostEqual(sum(row["share"] for row in rows), 1.0)
        self.assertEqual(
            {row["bucket"] for row in rows},
            {
                "idle",
                "feedback_init",
                "input_decode",
                "feedback_prepare",
                "target_execution",
                "feedback_merge",
                "signal",
                "bookkeeping",
            },
        )

    def test_validate_profiles_requires_timing_for_both_persistent_levels(self) -> None:
        from benchmark.rq4.profile_aggregate import validate_profiles

        profiles = [profile(name) for name in (
            "cufuzz", "libafl", "libafl-plus", "gunit-sync", "gunit"
        )]
        validate_profiles(profiles, ["k"])
        profiles[-1]["profiling_records_data"] = [
            record
            for record in profiles[-1]["profiling_records_data"]
            if record["domain"] != "device_kernel"
        ]
        with self.assertRaises(RuntimeError):
            validate_profiles(profiles, ["k"])

    def test_validate_profiles_rejects_profiler_teardown_in_elapsed_time(self) -> None:
        from benchmark.rq4.profile_aggregate import validate_profiles

        profiles = [profile(name) for name in (
            "cufuzz", "libafl", "libafl-plus", "gunit-sync", "gunit"
        )]
        profiles[0]["benchmark"]["elapsed_ns"] = 1900

        with self.assertRaisesRegex(RuntimeError, "profile window mismatch"):
            validate_profiles(profiles, ["k"])

    def test_validate_profiles_rejects_mismatched_window_size(self) -> None:
        from benchmark.rq4.profile_aggregate import validate_profiles

        profiles = [profile(name) for name in (
            "cufuzz", "libafl", "libafl-plus", "gunit-sync", "gunit"
        )]
        profiles[-1]["window_size"] = 4

        with self.assertRaisesRegex(RuntimeError, "window_size mismatch"):
            validate_profiles(profiles, ["k"])

    def test_validate_profiles_requires_one_supported_mode(self) -> None:
        from benchmark.rq4.profile_aggregate import validate_profiles

        profiles = [profile(name) for name in (
            "cufuzz", "libafl", "libafl-plus", "gunit-sync", "gunit"
        )]
        profiles[0].pop("mode")
        with self.assertRaisesRegex(RuntimeError, "profile mode"):
            validate_profiles(profiles, ["k"])

        profiles[0]["mode"] = "fixed"
        profiles[-1]["mode"] = "mutating"
        with self.assertRaisesRegex(RuntimeError, "profile modes"):
            validate_profiles(profiles, ["k"])

    def test_mutating_profiles_require_the_canonical_fixed_seed(self) -> None:
        from benchmark.rq4.profile_aggregate import validate_profiles

        profiles = mutating_profiles(mutate_seconds=1, wall_time_ns=1_000_000_000)
        profiles[-1]["fixed_seed"] = 2

        with self.assertRaisesRegex(RuntimeError, "RAPID_FIXED_SEED=1"):
            validate_profiles(profiles, ["k"])

    def test_mutating_profile_without_crash_or_mutation_is_rejected(self) -> None:
        from benchmark.rq4.profile_aggregate import _validate_mutating_result

        row = {
            "workload_id": "k",
            "configuration": "cufuzz",
            "fixed_seed": 1,
            "mutate_seconds": 30,
            "mutating": {
                "requested_seconds": 30,
                "corpus_size": 1,
                "solutions": 0,
                "executions": 11,
                "mutation_calls": 0,
                "coverage_nonzero_bytes": 0,
                "simt_memcov_nonzero_bits": 0,
                "pending": 0,
                "completed": 0,
                "outstanding": 0,
                "in_flight": 0,
                "queued_submissions": 0,
            },
        }

        with self.assertRaisesRegex(RuntimeError, "performed no mutations"):
            _validate_mutating_result(row, require_feedback_activity=False)

    def test_capture_incomplete_cell_is_marked_and_excluded(self) -> None:
        from benchmark.rq4.profile_aggregate import aggregate
        from benchmark.rq4.profile_plot import load_overhead_breakdown

        profiles = mutating_profiles(
            repetitions=(1, 2), mutate_seconds=1, wall_time_ns=1_000_000_000
        )

        incomplete = next(
            row
            for row in profiles
            if row["configuration"] == "libafl-plus" and row["repetition"] == 2
        )
        incomplete.update(
            {
                "capture_incomplete": True,
                "crash_loop": True,
                "mutating": None,
            }
        )
        for key in (
            "trace_evidence_data",
            "profiling_records_data",
            "cuda_api",
            "gpu_kernel_time_ns",
            "gpu_mem_time_ns",
        ):
            incomplete.pop(key)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output = root / "aggregate"
            input_dir.mkdir()
            (input_dir / "profiles.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in profiles),
                encoding="utf-8",
            )
            suite = root / "suite.json"
            suite.write_text(json.dumps({"workloads": ["k"]}), encoding="utf-8")

            aggregate(input_dir, output, suite)

            breakdown_path = output / "rq4_overhead_breakdown.csv"
            with breakdown_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            lane_statistics = load_overhead_breakdown(breakdown_path)

        incomplete_rows = [
            row
            for row in rows
            if row["lane"] == "LibAFL+" and row["repetition"] == "2"
        ]
        self.assertEqual(len(incomplete_rows), 1)
        self.assertEqual(incomplete_rows[0]["status"], "capture_incomplete")
        self.assertEqual(incomplete_rows[0]["category"], "")
        self.assertAlmostEqual(
            lane_statistics["k"]["LibAFL+"]["kernel_exec"], 1.58e-7
        )

    def test_mutating_profiles_aggregate_with_feedback_activity_gate(self) -> None:
        from benchmark.rq4.profile_aggregate import aggregate, validate_profiles

        profiles = mutating_profiles()
        for row in profiles:
            row["mutating"]["corpus_size"] = 1

        feedback_profile = next(
            row for row in profiles if row["configuration"] == "libafl-plus"
        )
        with self.assertLogs("benchmark.rq4.profile_aggregate", level="WARNING"):
            validate_profiles(profiles, ["k"])
        feedback_profile["mutating"].pop("coverage_nonzero_bytes")
        with self.assertRaisesRegex(
            RuntimeError,
            "incomplete mutating profile.*k libafl-plus.*coverage_nonzero_bytes",
        ):
            validate_profiles(profiles, ["k"])
        feedback_profile["mutating"]["coverage_nonzero_bytes"] = 0
        with self.assertRaisesRegex(RuntimeError, "feedback produced no coverage"):
            validate_profiles(profiles, ["k"])
        feedback_profile["mutating"]["coverage_nonzero_bytes"] = 1
        for row in profiles:
            if row["configuration"] not in ("cufuzz", "libafl"):
                row["mutating"]["corpus_size"] = 2
        crash_profile = next(
            row for row in profiles if row["configuration"] == "cufuzz"
        )
        crash_profile["mutating"]["solutions"] = 5
        crash_profile["mutating"]["mutation_calls"] = 0
        crash_profile["trace_evidence_data"]["capture_end_ns"] = 1_154

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output = root / "aggregate"
            input_dir.mkdir()
            (input_dir / "profiles.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in profiles),
                encoding="utf-8",
            )
            suite = root / "suite.json"
            suite.write_text(json.dumps({"workloads": ["k"]}), encoding="utf-8")

            aggregate(input_dir, output, suite)

            breakdown_path = output / "rq4_overhead_breakdown.csv"
            breakdown = breakdown_path.read_text(encoding="utf-8")
            with breakdown_path.open(encoding="utf-8", newline="") as stream:
                crash_rows = [
                    row
                    for row in csv.DictReader(stream)
                    if row["lane"] == "CuFuzz"
                ]
        self.assertIn("GUnit-s-CPU", breakdown)
        self.assertNotIn("GUnit-s-FB", breakdown)
        self.assertNotIn("GUnit-FB", breakdown)
        self.assertEqual({row["solutions"] for row in crash_rows}, {"5"})
        self.assertEqual({row["has_crashes"] for row in crash_rows}, {"True"})
        self.assertEqual(
            {row["capture_incomplete"] for row in crash_rows}, {"False"}
        )
        self.assertEqual({row["status"] for row in crash_rows}, {"crash"})


if __name__ == "__main__":
    unittest.main()
