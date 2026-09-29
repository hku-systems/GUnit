from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from benchmark.rq1.schema import sha256_file
from benchmark.rq2 import campaign
from benchmark.rq2.campaign import (
    BackendArtifact,
    CampaignInput,
    CONFIGURATIONS,
    append_unique_samples,
    campaign_command,
    load_attempted_trials,
    load_campaign_input,
    run_campaign,
    select_campaign_inputs,
    validate_cfg_presence_contracts,
)
from benchmark.rq2.verify_results import (
    CFG_PRESENCE_METRIC_VERSION,
    LEGACY_MEMORY_METRIC_VERSION,
    SIMT_MEMORY_METRIC_VERSION,
    SampleContext,
    parse_cfg_metric_contract,
    parse_memory_metric_contract,
    validate_and_enrich_samples,
    verify_result_directory,
)


def simt_feedback_metadata(
    *,
    instrumented_cfg_sites: int = 17,
    cfg_metric_version: str | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "schema_version": 2,
        "feedback_abi_version": 1,
        "memory_metric_version": "rapid-simt-memcov-v1",
        "memory_map_bits": 61_440,
        "memory_sector_bytes": 32,
        "thread_activity_map_bits": 4_096,
        "memory_pattern_encodings": {
            "single": 0,
            "full_broadcast": 1,
            "full_contiguous": 2,
            "full_other": 3,
            "partial_broadcast": 4,
            "partial_contiguous": 5,
            "partial_other": 6,
        },
        "memory_hash_contract_version": "rapid-simt-memcov-hash-v1",
        "instrumented_cfg_sites": instrumented_cfg_sites,
    }
    if cfg_metric_version is not None:
        metadata["cfg_metric_version"] = cfg_metric_version
    return metadata


def legacy_feedback_metadata() -> dict[str, object]:
    return {
        "schema_version": 1,
        "feedback_abi_version": 1,
        "edge_map_size": 65_536,
        "simt_memcov_buckets": 65_536,
        "simt_memcov_data_buckets": 61_440,
        "entry_symbol": "__rapid_entry__legacy",
        "kernel_id": "legacy__fixture",
        "instrumented_cfg_sites": 1,
        "instrumented_memory_sites": 1,
        "unknown_memory_sites": 0,
        "cfg_sites": [{}],
        "memory_sites": [{}],
    }


class Rq2CampaignMatrixTests(unittest.TestCase):
    def test_campaign_facade_exports_internal_module_contracts(self) -> None:
        from benchmark.rq2 import campaign_model, campaign_storage

        self.assertIs(campaign.CampaignInput, campaign_model.CampaignInput)
        self.assertIs(
            campaign.append_unique_samples,
            campaign_storage.append_unique_samples,
        )

    def test_matrix_has_exact_paper_rows_in_protocol_order(self) -> None:
        self.assertEqual(
            tuple(
                (
                    item.configuration_id,
                    item.paper_label,
                    item.backend,
                    item.window,
                    item.vconfig,
                    item.feedback_enabled,
                    item.asynchronous,
                )
                for item in CONFIGURATIONS
            ),
            (
                ("cufuzz-off", "CuFuzz-style", "cufuzz", 1, "off", False, False),
                ("origin-off", "LibAFL+", "origin", 1, "off", True, False),
                ("origin-on", "LibAFL+", "origin", 1, "on", True, False),
                ("rapid-w1-off", "Sys-s (w1)", "rapid", 1, "off", True, False),
                ("rapid-w1-on", "Sys-s (w1)", "rapid", 1, "on", True, False),
                ("rapid-w4-off", "Sys-s (w4)", "rapid", 4, "off", True, False),
                ("rapid-w4-on", "Sys-s (w4)", "rapid", 4, "on", True, False),
                ("rapid2-off", "Sys", "rapid2", 32, "off", True, True),
                ("rapid2-on", "Sys", "rapid2", 32, "on", True, True),
            ),
        )

    def test_commands_select_frontend_and_only_pass_required_windows(self) -> None:
        by_id = {item.configuration_id: item for item in CONFIGURATIONS}

        def command(configuration_id: str) -> list[str]:
            return campaign_command(
                configuration=by_id[configuration_id],
                fuzzer=Path("bin/fuzzer"),
                fuzzer_async=Path("bin/fuzzer_async"),
                library=Path("backend.so"),
                manifest=Path("manifest.json"),
                coverage_seconds=60,
                coverage_log=Path("raw.jsonl"),
            )

        for configuration_id in ("cufuzz-off", "origin-off", "origin-on"):
            with self.subTest(configuration=configuration_id):
                actual = command(configuration_id)
                self.assertEqual(actual[0], "bin/fuzzer")
                self.assertNotIn("--window-size", actual)
                self.assertNotIn("--retire-worker", actual)
                self.assertNotIn("--supply-threads", actual)

        expected_windows = {
            "rapid-w1-off": "1",
            "rapid-w1-on": "1",
            "rapid-w4-off": "4",
            "rapid-w4-on": "4",
            "rapid2-off": "32",
            "rapid2-on": "32",
        }
        for configuration_id, window in expected_windows.items():
            with self.subTest(configuration=configuration_id):
                actual = command(configuration_id)
                self.assertEqual(
                    actual[0],
                    "bin/fuzzer_async" if configuration_id.startswith("rapid2") else "bin/fuzzer",
                )
                self.assertEqual(actual[-2:], ["--window-size", window])
                self.assertNotIn("--retire-worker", actual)
                self.assertNotIn("--supply-threads", actual)

        self.assertEqual(
            command("rapid2-on")[1:-2],
            [
                "backend.so",
                "--manifest",
                "manifest.json",
                "--coverage-seconds",
                "60",
                "--coverage-log",
                "raw.jsonl",
                "--vconfig",
                "on",
            ],
        )

    def test_cli_has_no_sampling_interval_state(self) -> None:
        argv = [
            "campaign.py",
            "--build-root",
            "build",
            "--output",
            "output",
            "--repetitions",
            "1",
        ]
        with patch("sys.argv", argv), patch.object(campaign, "run_campaign") as run:
            campaign.main()

        (args,) = run.call_args.args
        self.assertFalse(hasattr(args, "sample_interval_seconds"))
        self.assertFalse(hasattr(args, "sample_interval_ms"))
        self.assertEqual(args.coverage_seconds, 10)
        self.assertEqual(args.seed_mode, "per-rep")
        self.assertEqual(args.fixed_seed, 1)

    def test_workload_selection_preserves_catalog_order(self) -> None:
        inputs = tuple(
            CampaignInput(
                workload_id=workload_id,
                build_report_path=Path(f"{workload_id}/build_report.json"),
                build_report_sha256=f"sha256:{workload_id}",
                manifest_path=Path(f"{workload_id}/manifest.json"),
                manifest_sha256=f"sha256:manifest-{workload_id}",
                constraints_sha256=f"sha256:constraints-{workload_id}",
                vconfig_effective=True,
                vconfig_disabled_reason=None,
                backends={},
            )
            for workload_id in ("cutlass_gemm", "shoc_scan", "shoc_reduction")
        )

        selected = select_campaign_inputs(
            inputs,
            ("shoc_reduction", "cutlass_gemm"),
        )

        self.assertEqual(
            tuple(item.workload_id for item in selected),
            ("cutlass_gemm", "shoc_reduction"),
        )

    def test_workload_selection_rejects_invalid_requests(self) -> None:
        inputs = (
            CampaignInput(
                workload_id="cutlass_gemm",
                build_report_path=Path("cutlass_gemm/build_report.json"),
                build_report_sha256="sha256:cutlass",
                manifest_path=Path("cutlass_gemm/manifest.json"),
                manifest_sha256="sha256:manifest",
                constraints_sha256="sha256:constraints",
                vconfig_effective=True,
                vconfig_disabled_reason=None,
                backends={},
            ),
        )
        invalid_cases = (
            (("",), "must not be empty"),
            (("unknown",), "unknown RQ2 workload"),
            (("cutlass_gemm", "cutlass_gemm"), "duplicate RQ2 workload"),
        )

        for requested, message in invalid_cases:
            with self.subTest(requested=requested):
                with self.assertRaisesRegex(ValueError, message):
                    select_campaign_inputs(inputs, requested)


class Rq2TelemetryValidationTests(unittest.TestCase):
    def _context(self, *, feedback_enabled: bool = True) -> SampleContext:
        return SampleContext(
            trial_id="000-workload-origin-off",
            workload_id="workload",
            configuration_id="origin-off" if feedback_enabled else "cufuzz-off",
            paper_label="LibAFL+" if feedback_enabled else "CuFuzz-style",
            backend="origin" if feedback_enabled else "cufuzz",
            window=1,
            repetition=0,
            seed=12345,
            vconfig_requested="off",
            vconfig_effective="off",
            feedback_enabled=feedback_enabled,
            build_report_sha256="sha256:" + "1" * 64,
            manifest_sha256="sha256:" + "2" * 64,
            backend_sha256="sha256:" + "3" * 64,
            instrumentation_metadata_sha256=("sha256:" + "4" * 64)
            if feedback_enabled
            else None,
            instrumented_cfg_sites=17 if feedback_enabled else None,
            memory_metric_version=(SIMT_MEMORY_METRIC_VERSION if feedback_enabled else None),
            memory_map_bits=61_440 if feedback_enabled else None,
            memory_sector_bytes=32 if feedback_enabled else None,
            thread_activity_map_bits=4_096 if feedback_enabled else None,
            memory_hash_contract_version=(
                "rapid-simt-memcov-hash-v1" if feedback_enabled else None
            ),
            raw_telemetry_path="raw/000.jsonl",
            raw_telemetry_sha256="sha256:" + "5" * 64,
        )

    def _valid_records(self) -> list[dict[str, object]]:
        cumulative = (
            (0, 0, 0, 0, 0),
            (2, 5, 7, 11, 23),
            (4, 6, 9, 12, 27),
        )
        deltas = ((0, 0, 0, 0), (5, 7, 11, 23), (1, 2, 1, 4))
        return [
            {
                "schema_version": 1,
                "sequence": sequence,
                "timestamp_s": (0.0, 0.125, 1.25)[sequence],
                "final_sample": sequence == 2,
                "executions_submitted": (0, 3, 4)[sequence],
                "executions_completed": values[0],
                "cfg_sites": values[1],
                "memory_features": values[2],
                "thread_activity_features": values[3],
                "feedback_features_total": values[4],
                "cfg_sites_delta": deltas[sequence][0],
                "memory_features_delta": deltas[sequence][1],
                "thread_activity_features_delta": deltas[sequence][2],
                "feedback_features_total_delta": deltas[sequence][3],
            }
            for sequence, values in enumerate(cumulative)
        ]

    def _write(self, root: Path, records: list[dict[str, object]]) -> Path:
        path = root / "raw.jsonl"
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return path

    def test_feedback_samples_are_validated_enriched_and_aligned_by_completion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq2-telemetry-") as temp_dir:
            samples = validate_and_enrich_samples(
                self._write(Path(temp_dir), self._valid_records()),
                self._context(),
            )

        self.assertEqual(
            [sample["executions_pending"] for sample in samples], [0, 1, 0]
        )
        self.assertEqual(
            [sample["executions_completed"] for sample in samples], [0, 2, 4]
        )
        self.assertEqual(samples[-1]["cfg_sites"], 6)
        self.assertEqual(samples[-1]["instrumented_cfg_sites"], 17)
        self.assertEqual(samples[-1]["memory_metric_version"], SIMT_MEMORY_METRIC_VERSION)
        self.assertEqual(samples[-1]["memory_map_bits"], 61_440)
        self.assertEqual(samples[-1]["memory_sector_bytes"], 32)
        self.assertEqual(samples[-1]["thread_activity_map_bits"], 4_096)
        self.assertEqual(
            samples[-1]["memory_hash_contract_version"],
            "rapid-simt-memcov-hash-v1",
        )
        self.assertEqual(
            samples[-1]["cfg_metric_version"], CFG_PRESENCE_METRIC_VERSION
        )
        self.assertNotIn("cfg_counter_saturation", samples[-1])
        self.assertEqual(
            samples[-1]["instrumentation_metadata_sha256"],
            "sha256:" + "4" * 64,
        )
        self.assertIsNone(samples[-1]["external_features"])
        self.assertTrue(samples[-1]["final_sample"])

    def test_completion_events_are_enriched_as_cumulative_schema_one_points(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq2-delta-telemetry-") as temp_dir:
            samples = validate_and_enrich_samples(
                self._write(Path(temp_dir), self._valid_records()),
                self._context(),
            )

        self.assertEqual([sample["schema_version"] for sample in samples], [1, 1, 1])
        self.assertTrue(all("raw_schema_version" not in sample for sample in samples))
        self.assertEqual([sample["cfg_sites"] for sample in samples], [0, 5, 6])
        self.assertEqual([sample["memory_features"] for sample in samples], [0, 7, 9])
        self.assertEqual([sample["executions_pending"] for sample in samples], [0, 1, 0])
        self.assertNotIn("memory_map_hex", samples[-1])
        self.assertNotIn("cfg_sites_delta", samples[-1])

    def test_obsolete_raw_schema_is_rejected(self) -> None:
        records = self._valid_records()
        records[0]["schema_version"] = 2
        with tempfile.TemporaryDirectory(prefix="rq2-obsolete-telemetry-") as temp_dir:
            with self.assertRaisesRegex(ValueError, "schema_version must be 1"):
                validate_and_enrich_samples(
                    self._write(Path(temp_dir), records),
                    self._context(),
                )

    def test_cufuzz_zero_internal_maps_are_exposed_as_null(self) -> None:
        records = [self._valid_records()[0], self._valid_records()[-1]]
        records[-1]["sequence"] = 1
        for record in records:
            record["cfg_sites"] = 0
            record["memory_features"] = 0
            record["thread_activity_features"] = 0
            record["feedback_features_total"] = 0
            record["cfg_sites_delta"] = 0
            record["memory_features_delta"] = 0
            record["thread_activity_features_delta"] = 0
            record["feedback_features_total_delta"] = 0

        with tempfile.TemporaryDirectory(prefix="rq2-cufuzz-telemetry-") as temp_dir:
            samples = validate_and_enrich_samples(
                self._write(Path(temp_dir), records),
                self._context(feedback_enabled=False),
            )

        for field in (
            "cfg_sites",
            "instrumented_cfg_sites",
            "memory_features",
            "memory_metric_version",
            "memory_map_bits",
            "memory_sector_bytes",
            "thread_activity_features",
            "memory_hash_contract_version",
            "instrumentation_metadata_sha256",
            "external_features",
        ):
            with self.subTest(field=field):
                self.assertIsNone(samples[-1][field])

    def test_invalid_raw_telemetry_is_rejected(self) -> None:
        invalid_cases: dict[str, list[dict[str, object]]] = {}
        base = self._valid_records()

        def changed(index: int, **updates: object) -> list[dict[str, object]]:
            records = [dict(record) for record in base]
            records[index].update(updates)
            return records

        invalid_cases["completed executions exceed"] = changed(
            1, executions_submitted=1
        )
        invalid_cases["cfg_sites exceeds"] = changed(
            2,
            cfg_sites=18,
            cfg_sites_delta=13,
            feedback_features_total=39,
            feedback_features_total_delta=16,
        )
        invalid_cases["cumulative features"] = changed(1, feedback_features_total=28)

        for expected_message, records in invalid_cases.items():
            with self.subTest(expected_message=expected_message):
                with tempfile.TemporaryDirectory(prefix="rq2-invalid-") as temp_dir:
                    path = self._write(Path(temp_dir), records)
                    with self.assertRaisesRegex(ValueError, expected_message):
                        validate_and_enrich_samples(path, self._context())


class Rq2CampaignPersistenceTests(unittest.TestCase):
    def _sha256(self, path: Path) -> str:
        return f"sha256:{sha256_file(path)}"

    def _campaign_input(
        self,
        root: Path,
        *,
        vconfig_effective: bool = True,
        disabled_reason: str | None = None,
    ) -> CampaignInput:
        workload_root = root / "workload"
        workload_root.mkdir(parents=True)
        manifest = workload_root / "manifest.json"
        manifest.write_text('{"kernels": []}\n', encoding="utf-8")
        build_report = workload_root / "build_report.json"
        build_report.write_text('{"schema_version": 1}\n', encoding="utf-8")
        constraints = workload_root / "constraints.json"
        constraints.write_text('{"overrides": []}\n', encoding="utf-8")
        backends: dict[str, BackendArtifact] = {}
        for backend_name in ("cufuzz", "origin", "rapid", "rapid2"):
            backend_root = workload_root / backend_name
            backend_root.mkdir()
            library = backend_root / "target.so"
            library.write_bytes(f"{backend_name}-library".encode())
            metadata = None
            metadata_sha256 = None
            instrumented_cfg_sites = None
            if backend_name != "cufuzz":
                metadata = backend_root / "feedback_metadata.json"
                metadata.write_text(
                    json.dumps(simt_feedback_metadata()) + "\n",
                    encoding="utf-8",
                )
                metadata_sha256 = self._sha256(metadata)
                instrumented_cfg_sites = 17
            backends[backend_name] = BackendArtifact(
                library_path=library,
                library_sha256=self._sha256(library),
                instrumentation_metadata_path=metadata,
                instrumentation_metadata_sha256=metadata_sha256,
                instrumented_cfg_sites=instrumented_cfg_sites,
                memory_metric_version=(
                    SIMT_MEMORY_METRIC_VERSION if backend_name != "cufuzz" else None
                ),
                memory_map_bits=61_440 if backend_name != "cufuzz" else None,
                memory_sector_bytes=32 if backend_name != "cufuzz" else None,
                thread_activity_map_bits=4_096 if backend_name != "cufuzz" else None,
                memory_hash_contract_version=(
                    "rapid-simt-memcov-hash-v1"
                    if backend_name != "cufuzz"
                    else None
                ),
                cfg_metric_version=(
                    CFG_PRESENCE_METRIC_VERSION
                    if backend_name != "cufuzz"
                    else None
                ),
            )
        return CampaignInput(
            workload_id="workload",
            build_report_path=build_report,
            build_report_sha256=self._sha256(build_report),
            manifest_path=manifest,
            manifest_sha256=self._sha256(manifest),
            constraints_sha256=self._sha256(constraints),
            vconfig_effective=vconfig_effective,
            vconfig_disabled_reason=disabled_reason,
            backends=backends,
        )

    def _write_build_report_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        workload_root = root / "built-workload"
        constraints = workload_root / "constraints.json"
        workload_root.mkdir(parents=True)
        constraints.write_text('{"overrides": []}\n', encoding="utf-8")
        phase2 = workload_root / "resolved" / "kernel" / "phase2"
        phase2.mkdir(parents=True)
        manifest = phase2.parent / "manifest.json"
        manifest.write_text('{"kernels": []}\n', encoding="utf-8")
        build_spec = phase2 / "build_spec.json"
        build_spec.write_text('{"not": "feedback metadata"}\n', encoding="utf-8")
        backend_rows: dict[str, object] = {}
        metadata_payload = {
            **simt_feedback_metadata(),
            "entry_symbol": "kernel_entry",
        }
        for backend_name in ("cufuzz", "origin", "rapid", "rapid2"):
            backend_root = workload_root / "backends" / backend_name
            backend_root.mkdir(parents=True)
            library = backend_root / "target.so"
            library.write_bytes(f"{backend_name}-library".encode())
            backend_build: dict[str, object] = {"shared_lib": str(library)}
            if backend_name != "cufuzz":
                metadata = backend_root / "feedback_metadata.json"
                metadata.write_text(
                    json.dumps(metadata_payload, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                backend_build["feedback_metadata"] = str(metadata)
            backend_report_path = backend_root / "backend_build.json"
            backend_report_path.write_text(
                json.dumps(backend_build) + "\n",
                encoding="utf-8",
            )
            backend_rows[backend_name] = {
                "artifacts": {
                    "backend_build.json": {
                        "path": "backend_build.json",
                        "sha256": sha256_file(backend_report_path),
                    },
                    "shared_library": {
                        "path": "target.so",
                        "sha256": sha256_file(library),
                    },
                }
            }
        report_path = workload_root / "build_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "workload_id": "workload",
                    "shared_phase2": {
                        "phase2_dir": str(phase2),
                        "artifacts": {
                            "manifest.json": {"sha256": sha256_file(manifest)}
                        },
                    },
                    "backends": backend_rows,
                    "rq2": {
                        "vconfig_requested": True,
                        "vconfig_effective": True,
                        "vconfig_disabled_reason": None,
                        "constraints_path": str(constraints),
                        "constraints_sha256": self._sha256(constraints),
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return report_path, build_spec, constraints

    def test_campaign_loader_rejects_retired_cfg_metric_version(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq2-cfg-mix-") as temp_dir:
            root = Path(temp_dir)
            presence = self._campaign_input(root / "presence")
            count_backends = dict(presence.backends)
            for backend_name in ("origin", "rapid", "rapid2"):
                count_backends[backend_name] = replace(
                    count_backends[backend_name],
                    cfg_metric_version="rapid-cfg-warpcount-v1",
                )
            count = replace(
                presence,
                workload_id="count-workload",
                backends=count_backends,
            )

            with self.assertRaisesRegex(
                RuntimeError, "invalid CFG metric contract"
            ):
                validate_cfg_presence_contracts((presence, count))

    def test_build_loader_hashes_real_feedback_metadata_and_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq2-build-loader-") as temp_dir:
            report_path, build_spec, constraints = self._write_build_report_fixture(
                Path(temp_dir)
            )

            loaded = load_campaign_input(report_path)

            expected_metadata = (
                report_path.parent
                / "backends"
                / "origin"
                / "feedback_metadata.json"
            )
            expected_hash = self._sha256(expected_metadata)
            self.assertEqual(
                loaded.backends["origin"].instrumentation_metadata_sha256,
                expected_hash,
            )
            self.assertNotEqual(expected_hash, self._sha256(build_spec))
            self.assertEqual(loaded.constraints_sha256, self._sha256(constraints))
            self.assertEqual(loaded.backends["rapid"].instrumented_cfg_sites, 17)
            self.assertEqual(
                loaded.backends["rapid"].memory_metric_version,
                SIMT_MEMORY_METRIC_VERSION,
            )
            self.assertEqual(loaded.backends["rapid"].memory_sector_bytes, 32)
            self.assertIsNone(
                loaded.backends["cufuzz"].instrumentation_metadata_sha256
            )

            origin_build = report_path.parent / "backends" / "origin" / "backend_build.json"
            original_origin_build = origin_build.read_text(encoding="utf-8")
            origin_build.write_text(original_origin_build + " ", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "backend build report hash drift"):
                load_campaign_input(report_path)
            origin_build.write_text(original_origin_build, encoding="utf-8")

            rapid_metadata = report_path.parent / "backends" / "rapid" / "feedback_metadata.json"
            rapid_metadata.write_text(
                json.dumps(
                    {
                        **simt_feedback_metadata(instrumented_cfg_sites=18),
                        "entry_symbol": "kernel_entry",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "feedback metadata differ"):
                load_campaign_input(report_path)

            constraints.write_text('{"overrides": [{"id": "drift"}]}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "constraints hash differs"):
                load_campaign_input(report_path)

    def test_metric_contract_rejects_new_campaign_mismatch_and_classifies_legacy(self) -> None:
        contract = parse_memory_metric_contract(
            simt_feedback_metadata(),
            allow_legacy=False,
        )
        self.assertEqual(contract.memory_metric_version, SIMT_MEMORY_METRIC_VERSION)
        self.assertEqual(contract.memory_sector_bytes, 32)

        mismatched = simt_feedback_metadata()
        mismatched["memory_map_bits"] = 60_000
        with self.assertRaisesRegex(ValueError, "memory_map_bits"):
            parse_memory_metric_contract(mismatched, allow_legacy=False)

        legacy = parse_memory_metric_contract(legacy_feedback_metadata(), allow_legacy=True)
        self.assertEqual(legacy.memory_metric_version, LEGACY_MEMORY_METRIC_VERSION)
        self.assertIsNone(legacy.memory_sector_bytes)
        with self.assertRaisesRegex(ValueError, "explicit memory metric"):
            parse_memory_metric_contract(
                legacy_feedback_metadata(),
                allow_legacy=False,
            )

    def test_cfg_metric_contract_accepts_only_presence(self) -> None:
        presence = parse_cfg_metric_contract(simt_feedback_metadata())
        self.assertEqual(
            presence.cfg_metric_version, CFG_PRESENCE_METRIC_VERSION
        )

        for version in (
            "rapid-cfg-warppresence-v1",
            "rapid-cfg-warpcount-v1",
            "rapid-cfg-threadcount-v1",
        ):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "unsupported CFG metric"):
                    parse_cfg_metric_contract(
                        simt_feedback_metadata(cfg_metric_version=version)
                    )

    def test_legacy_metric_requires_complete_self_consistent_metadata(self) -> None:
        cases = {
            "feedback_abi_version": None,
            "edge_map_size": 8,
            "simt_memcov_buckets": 8,
            "simt_memcov_data_buckets": 1,
            "instrumented_cfg_sites": -1,
            "instrumented_memory_sites": "one",
            "unknown_memory_sites": -1,
            "entry_symbol": "",
            "kernel_id": None,
        }
        for field, invalid_value in cases.items():
            with self.subTest(field=field):
                metadata = legacy_feedback_metadata()
                if invalid_value is None and field == "feedback_abi_version":
                    metadata.pop(field)
                else:
                    metadata[field] = invalid_value
                with self.assertRaisesRegex(ValueError, field):
                    parse_memory_metric_contract(metadata, allow_legacy=True)

        missing_partitions = {"schema_version": 1}
        with self.assertRaisesRegex(ValueError, "feedback_abi_version"):
            parse_memory_metric_contract(missing_partitions, allow_legacy=True)

        count_mismatch = legacy_feedback_metadata()
        count_mismatch["cfg_sites"] = []
        with self.assertRaisesRegex(ValueError, "cfg_sites"):
            parse_memory_metric_contract(count_mismatch, allow_legacy=True)

    def _args(self, root: Path) -> Namespace:
        fuzzer = root / "fuzzer"
        fuzzer_async = root / "fuzzer_async"
        fuzzer.write_bytes(b"fuzzer")
        fuzzer_async.write_bytes(b"fuzzer-async")
        return Namespace(
            build_root=root / "build",
            output=root / "output",
            repetitions=1,
            coverage_seconds=2,
            gpu_device="2",
            fixed_seed=424242,
            seed_mode="per-rep",
            timeout=90,
            fuzzer=fuzzer,
            fuzzer_async=fuzzer_async,
            workload=None,
        )

    def _write_valid_raw(self, command: list[str], *, feedback: bool) -> None:
        raw_path = Path(command[command.index("--coverage-log") + 1])
        features = (3, 5, 7) if feedback else (0, 0, 0)
        raw_path.write_text(
            "".join(
                json.dumps(record) + "\n"
                for record in (
                    {
                        "schema_version": 1,
                        "sequence": 0,
                        "timestamp_s": 0.0,
                        "final_sample": False,
                        "executions_submitted": 0,
                        "executions_completed": 0,
                        "cfg_sites": 0,
                        "memory_features": 0,
                        "thread_activity_features": 0,
                        "feedback_features_total": 0,
                        "cfg_sites_delta": 0,
                        "memory_features_delta": 0,
                        "thread_activity_features_delta": 0,
                        "feedback_features_total_delta": 0,
                    },
                    {
                        "schema_version": 1,
                        "sequence": 1,
                        "timestamp_s": 2.0,
                        "final_sample": True,
                        "executions_submitted": 4,
                        "executions_completed": 4,
                        "cfg_sites": features[0],
                        "memory_features": features[1],
                        "thread_activity_features": features[2],
                        "feedback_features_total": sum(features),
                        "cfg_sites_delta": features[0],
                        "memory_features_delta": features[1],
                        "thread_activity_features_delta": features[2],
                        "feedback_features_total_delta": sum(features),
                    },
                )
            ),
            encoding="utf-8",
        )

    def _jsonl(self, path: Path) -> list[dict[str, object]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_unsupported_vconfig_skips_only_on_rows_and_runs_every_off_row(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq2-campaign-skip-") as temp_dir:
            root = Path(temp_dir)
            args = self._args(root)
            campaign_input = self._campaign_input(
                root,
                vconfig_effective=False,
                disabled_reason="vconfig_barrier_unsupported",
            )
            selected = (
                CONFIGURATIONS[0],
                CONFIGURATIONS[1],
                CONFIGURATIONS[2],
            )

            def completed(
                command: list[str], **kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")
                self.assertEqual(kwargs["env"]["RAPID_FIXED_SEED"], "424242")
                self.assertEqual(
                    Path(command[0]).parent,
                    (args.output / "binaries").resolve(),
                )
                configuration = next(
                    item
                    for item in selected
                    if item.vconfig
                    == command[command.index("--vconfig") + 1]
                    and item.backend
                    == ("cufuzz" if "cufuzz" in command[1] else "origin")
                )
                self._write_valid_raw(command, feedback=configuration.feedback_enabled)
                return subprocess.CompletedProcess(command, 0, "stdout\n", "stderr\n")

            with patch.object(campaign, "CONFIGURATIONS", selected), patch(
                "benchmark.rq2.campaign._capture", return_value={"probe": "ok"}
            ), patch("benchmark.rq2.campaign.subprocess.run", side_effect=completed):
                run_campaign(args, campaign_inputs=(campaign_input,))

            trials = self._jsonl(args.output / "trials.jsonl")
            self.assertEqual(
                [(row["configuration_id"], row["status"]) for row in trials],
                [
                    ("cufuzz-off", "completed"),
                    ("origin-off", "completed"),
                    ("origin-on", "skipped"),
                ],
            )
            self.assertEqual(
                trials[-1]["vconfig_disabled_reason"],
                "vconfig_barrier_unsupported",
            )
            self.assertEqual(trials[-1]["vconfig_effective"], "unsupported")
            self.assertTrue(
                all(
                    row["constraints_sha256"] == campaign_input.constraints_sha256
                    for row in trials
                )
            )
            self.assertEqual(len(self._jsonl(args.output / "samples.jsonl")), 4)
            for relative in (
                "environment.json",
                "configurations.jsonl",
                "samples.jsonl",
                "trials.jsonl",
                "failures.jsonl",
            ):
                self.assertTrue((args.output / relative).is_file(), relative)
            self.assertTrue((args.output / "logs").is_dir())
            self.assertTrue((args.output / "raw").is_dir())
            environment = json.loads(
                (args.output / "environment.json").read_text(encoding="utf-8")
            )
            for field in (
                "captured_at",
                "hostname",
                "platform",
                "python",
                "nvidia_smi",
                "nvcc",
                "clang",
                "rustc",
                "rapid_head",
                "rapid_status",
                "coverage_seconds",
                "coverage_recording_mode",
                "repetitions",
                "timeout_seconds",
            ):
                self.assertIn(field, environment)
            self.assertEqual(
                verify_result_directory(args.output),
                {
                    "schema_version": 1,
                    "status": "ok",
                    "completed": 2,
                    "skipped": 1,
                    "failed": 0,
                    "samples": 4,
                },
            )
            self.assertEqual(
                environment["coverage_recording_mode"], "completion-change-v1"
            )
            for executable_name in ("fuzzer", "fuzzer_async"):
                snapshot = (args.output / "binaries" / executable_name).resolve()
                self.assertTrue(snapshot.is_file())
                self.assertEqual(environment[executable_name]["path"], str(snapshot))
                self.assertEqual(
                    environment[executable_name]["sha256"],
                    "sha256:" + sha256_file(snapshot),
                )

            environment_path = args.output / "environment.json"
            environment["coverage_recording_mode"] = "polling-v1"
            environment_path.write_text(
                json.dumps(environment, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "coverage recording mode"):
                verify_result_directory(args.output)
            environment["coverage_recording_mode"] = "completion-change-v1"
            environment_path.write_text(
                json.dumps(environment, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            args.fuzzer.write_bytes(b"rebuilt-fuzzer")
            self.assertEqual(
                verify_result_directory(args.output)["status"],
                "ok",
            )

            snapshot_fuzzer = args.output / "binaries" / "fuzzer"
            snapshot_fuzzer.write_bytes(b"tampered-fuzzer")
            with self.assertRaisesRegex(RuntimeError, "fuzzer hash mismatch"):
                verify_result_directory(args.output)
            snapshot_fuzzer.write_bytes(b"fuzzer")

            environment.pop("nvcc")
            environment_path.write_text(
                json.dumps(environment, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "missing environment snapshot"):
                verify_result_directory(args.output)
            environment["nvcc"] = {"probe": "ok"}
            environment["coverage_seconds"] = 3
            environment_path.write_text(
                json.dumps(environment, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "ended before"):
                verify_result_directory(args.output)
            environment["coverage_seconds"] = 2
            environment_path.write_text(
                json.dumps(environment, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            args.coverage_seconds = 3
            with patch.object(campaign, "CONFIGURATIONS", selected):
                with self.assertRaisesRegex(RuntimeError, "environment.json differs"):
                    run_campaign(args, campaign_inputs=(campaign_input,))

    def test_environment_records_completion_change_mode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq2-campaign-mode-") as temp_dir:
            root = Path(temp_dir)
            args = self._args(root)
            campaign_input = self._campaign_input(root)

            def completed(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                self._write_valid_raw(command, feedback=False)
                return subprocess.CompletedProcess(command, 0, "stdout\n", "")

            with patch.object(campaign, "CONFIGURATIONS", (CONFIGURATIONS[0],)), patch(
                "benchmark.rq2.campaign._capture", return_value={"probe": "ok"}
            ), patch("benchmark.rq2.campaign.subprocess.run", side_effect=completed):
                run_campaign(args, campaign_inputs=(campaign_input,))

            environment = json.loads(
                (args.output / "environment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                environment["coverage_recording_mode"], "completion-change-v1"
            )
            self.assertNotIn("sample_interval_seconds", environment)
            self.assertNotIn("sample_interval_ms", environment)

    def test_result_verifier_rejects_retired_cfg_metric_version(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq2-result-cfg-mix-") as temp_dir:
            root = Path(temp_dir)
            args = self._args(root)
            campaign_input = self._campaign_input(root)
            selected = (CONFIGURATIONS[1], CONFIGURATIONS[3])

            def completed(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                self._write_valid_raw(command, feedback=True)
                return subprocess.CompletedProcess(command, 0, "stdout\n", "")

            with patch.object(campaign, "CONFIGURATIONS", selected), patch(
                "benchmark.rq2.campaign._capture", return_value={"probe": "ok"}
            ), patch("benchmark.rq2.campaign.subprocess.run", side_effect=completed):
                run_campaign(args, campaign_inputs=(campaign_input,))

            rapid_metadata = campaign_input.backends[
                "rapid"
            ].instrumentation_metadata_path
            assert rapid_metadata is not None
            rapid_metadata.write_text(
                json.dumps(
                    simt_feedback_metadata(
                        cfg_metric_version="rapid-cfg-warpcount-v1"
                    ),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            rapid_hash = "sha256:" + sha256_file(rapid_metadata)

            trials = self._jsonl(args.output / "trials.jsonl")
            rapid_trial_ids = {
                row["trial_id"] for row in trials if row["backend"] == "rapid"
            }
            for row in trials:
                if row["trial_id"] in rapid_trial_ids:
                    row["instrumentation_metadata_sha256"] = rapid_hash
                    row["cfg_metric_version"] = "rapid-cfg-warpcount-v1"
            (args.output / "trials.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in trials),
                encoding="utf-8",
            )
            samples = self._jsonl(args.output / "samples.jsonl")
            for row in samples:
                if row["trial_id"] in rapid_trial_ids:
                    row["instrumentation_metadata_sha256"] = rapid_hash
                    row["cfg_metric_version"] = "rapid-cfg-warpcount-v1"
            (args.output / "samples.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in samples),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                RuntimeError, "unsupported CFG metric version"
            ):
                verify_result_directory(args.output)

    def test_per_rep_seed_mode_is_distinct_deterministic_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq2-campaign-per-rep-seed-") as temp_dir:
            root = Path(temp_dir)
            args = self._args(root)
            args.repetitions = 3
            campaign_input = self._campaign_input(root)
            observed_seeds: list[str] = []

            def completed(
                command: list[str], **kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                observed_seeds.append(str(kwargs["env"]["RAPID_FIXED_SEED"]))
                self._write_valid_raw(command, feedback=False)
                return subprocess.CompletedProcess(command, 0, "stdout\n", "")

            with patch.object(campaign, "CONFIGURATIONS", (CONFIGURATIONS[0],)), patch(
                "benchmark.rq2.campaign._capture", return_value={"probe": "ok"}
            ), patch("benchmark.rq2.campaign.subprocess.run", side_effect=completed):
                run_campaign(args, campaign_inputs=(campaign_input,))

            self.assertEqual(observed_seeds, ["424242", "424243", "424244"])
            trials = self._jsonl(args.output / "trials.jsonl")
            self.assertEqual([row["seed"] for row in trials], [424242, 424243, 424244])
            samples = self._jsonl(args.output / "samples.jsonl")
            self.assertEqual(
                [row["seed"] for row in samples],
                [424242, 424242, 424243, 424243, 424244, 424244],
            )
            environment = json.loads(
                (args.output / "environment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(environment["fixed_seed"], 424242)
            self.assertEqual(environment["seed_mode"], "per-rep")

    def test_fixed_seed_mode_preserves_one_seed_for_all_repetitions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq2-campaign-fixed-seed-") as temp_dir:
            root = Path(temp_dir)
            args = self._args(root)
            args.repetitions = 3
            args.seed_mode = "fixed"
            campaign_input = self._campaign_input(root)
            observed_seeds: list[str] = []

            def completed(
                command: list[str], **kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                observed_seeds.append(str(kwargs["env"]["RAPID_FIXED_SEED"]))
                self._write_valid_raw(command, feedback=False)
                return subprocess.CompletedProcess(command, 0, "stdout\n", "")

            with patch.object(campaign, "CONFIGURATIONS", (CONFIGURATIONS[0],)), patch(
                "benchmark.rq2.campaign._capture", return_value={"probe": "ok"}
            ), patch("benchmark.rq2.campaign.subprocess.run", side_effect=completed):
                run_campaign(args, campaign_inputs=(campaign_input,))

            self.assertEqual(observed_seeds, ["424242", "424242", "424242"])
            trials = self._jsonl(args.output / "trials.jsonl")
            self.assertEqual([row["seed"] for row in trials], [424242, 424242, 424242])

            environment_path = args.output / "environment.json"
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
            environment.pop("seed_mode")
            environment_path.write_text(
                json.dumps(environment, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with patch.object(campaign, "CONFIGURATIONS", (CONFIGURATIONS[0],)), patch(
                "benchmark.rq2.campaign.subprocess.run"
            ) as resumed_run:
                run_campaign(args, campaign_inputs=(campaign_input,))
            resumed_run.assert_not_called()

    def test_workload_filter_limits_trials_and_environment(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq2-campaign-workloads-") as temp_dir:
            root = Path(temp_dir)
            args = self._args(root)
            args.workload = ["selected"]
            campaign_input = self._campaign_input(root)
            inputs = (
                replace(campaign_input, workload_id="unselected"),
                replace(campaign_input, workload_id="selected"),
            )
            commands: list[list[str]] = []

            def completed(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                self._write_valid_raw(command, feedback=False)
                return subprocess.CompletedProcess(command, 0, "stdout\n", "")

            with patch.object(campaign, "CONFIGURATIONS", (CONFIGURATIONS[0],)), patch(
                "benchmark.rq2.campaign._capture", return_value={"probe": "ok"}
            ), patch("benchmark.rq2.campaign.subprocess.run", side_effect=completed):
                run_campaign(args, campaign_inputs=inputs)

            environment = json.loads(
                (args.output / "environment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(commands), 1)
            self.assertEqual(
                [item["workload_id"] for item in environment["workloads"]],
                ["selected"],
            )
            self.assertEqual(
                [item["workload_id"] for item in self._jsonl(args.output / "trials.jsonl")],
                ["selected"],
            )

    def test_nonzero_and_timeout_trials_are_persisted_with_logs(self) -> None:
        for label in ("nonzero", "timeout"):
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                prefix=f"rq2-campaign-{label}-"
            ) as temp_dir:
                root = Path(temp_dir)
                args = self._args(root)
                campaign_input = self._campaign_input(root)

                def fail(command: list[str], **_kwargs: object):
                    if label == "timeout":
                        raise subprocess.TimeoutExpired(
                            command,
                            timeout=args.timeout,
                            output="partial stdout\n",
                            stderr="partial stderr\n",
                        )
                    self._write_valid_raw(command, feedback=False)
                    return subprocess.CompletedProcess(
                        command, 7, "failed stdout\n", "failed stderr\n"
                    )

                with patch.object(campaign, "CONFIGURATIONS", (CONFIGURATIONS[0],)), patch(
                    "benchmark.rq2.campaign._capture", return_value={"probe": "ok"}
                ), patch("benchmark.rq2.campaign.subprocess.run", side_effect=fail):
                    run_campaign(args, campaign_inputs=(campaign_input,))

                failures = self._jsonl(args.output / "failures.jsonl")
                self.assertEqual(len(failures), 1)
                self.assertEqual(failures[0]["failure"], label)
                self.assertIn("stdout_sha256", failures[0])
                self.assertIn("stderr_sha256", failures[0])
                self.assertIn("raw_telemetry_sha256", failures[0])
                self.assertEqual((args.output / failures[0]["stdout_log"]).read_text(),
                                 "partial stdout\n" if label == "timeout" else "failed stdout\n")
                self.assertEqual(
                    load_attempted_trials(args.output),
                    {("workload", "cufuzz-off", 0)},
                )
                self.assertEqual(
                    verify_result_directory(args.output),
                    {
                        "schema_version": 1,
                        "status": "ok",
                        "completed": 0,
                        "skipped": 0,
                        "failed": 1,
                        "samples": 0,
                    },
                )

    def test_interrupted_row_runs_again_without_duplicating_completed_samples(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq2-campaign-resume-") as temp_dir:
            root = Path(temp_dir)
            args = self._args(root)
            campaign_input = self._campaign_input(root)
            selected = (CONFIGURATIONS[0], CONFIGURATIONS[1])
            first_invocation_commands: list[list[str]] = []

            def interrupt_second(command: list[str], **_kwargs: object):
                first_invocation_commands.append(command)
                if len(first_invocation_commands) == 2:
                    raise KeyboardInterrupt
                self._write_valid_raw(command, feedback=False)
                return subprocess.CompletedProcess(command, 0, "first\n", "")

            with patch.object(campaign, "CONFIGURATIONS", selected), patch(
                "benchmark.rq2.campaign._capture", return_value={"probe": "ok"}
            ), patch(
                "benchmark.rq2.campaign.subprocess.run", side_effect=interrupt_second
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_campaign(args, campaign_inputs=(campaign_input,))

            self.assertEqual(
                [row["configuration_id"] for row in self._jsonl(args.output / "trials.jsonl")],
                ["cufuzz-off"],
            )
            self.assertEqual(len(self._jsonl(args.output / "samples.jsonl")), 2)
            resumed_commands: list[list[str]] = []

            def finish(command: list[str], **_kwargs: object):
                resumed_commands.append(command)
                self._write_valid_raw(command, feedback=True)
                return subprocess.CompletedProcess(command, 0, "second\n", "")

            with patch.object(campaign, "CONFIGURATIONS", selected), patch(
                "benchmark.rq2.campaign._capture", return_value={"probe": "ok"}
            ), patch("benchmark.rq2.campaign.subprocess.run", side_effect=finish):
                run_campaign(args, campaign_inputs=(campaign_input,))

            self.assertEqual(len(resumed_commands), 1)
            self.assertIn("origin", resumed_commands[0][1])
            self.assertEqual(
                [row["configuration_id"] for row in self._jsonl(args.output / "trials.jsonl")],
                ["cufuzz-off", "origin-off"],
            )
            sample_keys = [
                (row["trial_id"], row["sequence"])
                for row in self._jsonl(args.output / "samples.jsonl")
            ]
            self.assertEqual(len(sample_keys), len(set(sample_keys)))

    def test_resume_replaces_samples_without_a_terminal_row(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq2-campaign-orphan-") as temp_dir:
            root = Path(temp_dir)
            args = self._args(root)
            campaign_input = self._campaign_input(root)
            selected = (CONFIGURATIONS[0],)

            def first_run(command: list[str], **_kwargs: object):
                self._write_valid_raw(command, feedback=False)
                return subprocess.CompletedProcess(command, 0, "first\n", "")

            with patch.object(campaign, "CONFIGURATIONS", selected), patch(
                "benchmark.rq2.campaign._capture", return_value={"probe": "ok"}
            ), patch("benchmark.rq2.campaign.subprocess.run", side_effect=first_run):
                run_campaign(args, campaign_inputs=(campaign_input,))

            # Model a crash after samples were fsynced but before the terminal row.
            (args.output / "trials.jsonl").write_text("", encoding="utf-8")

            def resumed_run(command: list[str], **_kwargs: object):
                self._write_valid_raw(command, feedback=False)
                raw_path = Path(command[command.index("--coverage-log") + 1])
                records = self._jsonl(raw_path)
                records[-1]["timestamp_s"] = 2.5
                raw_path.write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "resumed\n", "")

            with patch.object(campaign, "CONFIGURATIONS", selected), patch(
                "benchmark.rq2.campaign._capture", return_value={"probe": "ok"}
            ), patch("benchmark.rq2.campaign.subprocess.run", side_effect=resumed_run):
                run_campaign(args, campaign_inputs=(campaign_input,))

            trials = self._jsonl(args.output / "trials.jsonl")
            samples = self._jsonl(args.output / "samples.jsonl")
            self.assertEqual(len(trials), 1)
            self.assertEqual(trials[0]["status"], "completed")
            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[-1]["timestamp_s"], 2.5)

    def test_sample_resume_rejects_same_key_with_different_content(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq2-sample-collision-") as temp_dir:
            samples_path = Path(temp_dir) / "samples.jsonl"
            original = {"trial_id": "trial", "sequence": 0, "value": 1}
            append_unique_samples(samples_path, [original])
            append_unique_samples(samples_path, [dict(original)])
            self.assertEqual(len(self._jsonl(samples_path)), 1)

            with self.assertRaisesRegex(RuntimeError, "sample collision"):
                append_unique_samples(
                    samples_path,
                    [{"trial_id": "trial", "sequence": 0, "value": 2}],
                )


if __name__ == "__main__":
    unittest.main()
