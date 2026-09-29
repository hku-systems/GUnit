import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from benchmark.rq1.schema import (
    EXPECTED_WORKLOAD_IDS,
    load_catalog,
    load_provenance,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RQ1_ROOT = REPO_ROOT / "benchmark" / "rq1"

EXPECTED_LAUNCH = {
    "shoc_reduction": ((1, 1, 1), (256, 1, 1), 1024),
    "shoc_radix_sort": ((1, 1, 1), (128, 1, 1), 0),
    "shoc_scan": ((1, 1, 1), (256, 1, 1), 2048),
    "cutlass_gemm": ((1, 1, 1), (16, 1, 1), 0),
    "flashattention_device1xn": ((1, 1, 1), (128, 1, 1), 0),
    "pytorch_batchnorm": ((1, 1, 1), (32, 1, 1), 0),
}

EXPECTED_UPSTREAM_SHA256 = {
    "shoc_reduction": "9f612f11218d41d521babc23370cde766267f1e518e03fc5603370e983685fc2",
    "shoc_radix_sort": "8ee17b95409225e4a0972914d127f94a91a6209218f5d06236bb6b6a2a1bbaaa",
    "shoc_scan": "3951f4881126e03633ae2b4df53a424c8a03a8f0a74e936712364daedb73d955",
    "cutlass_gemm": "2abd83bd93f3b53b5acfba545adc1fab753540adf9afd67bddcba4255d47cc3b",
    "flashattention_device1xn": "a1a21f9d4d5b146048619ea437f4c00123cb97dbdaaa186f695c0690f1b29245",
    "pytorch_batchnorm": "0bc9f9984ef342142e8cace15c2ce9b3756d9ee09e8c8429b879bd2c3049681e",
}

EXPECTED_ENTRY = {
    workload_id: f"rq1_{workload_id}" for workload_id in EXPECTED_WORKLOAD_IDS
}


class Rq1SourceTests(unittest.TestCase):
    def test_each_workload_has_a_compilable_standalone_kernel(self) -> None:
        compiler = shutil.which("clang++")
        if compiler is None:
            self.skipTest("clang++ is not installed")
        with tempfile.TemporaryDirectory(prefix="rq1-source-compile-") as temp_dir:
            output_root = Path(temp_dir)
            for workload_id in EXPECTED_WORKLOAD_IDS:
                source = RQ1_ROOT / "workloads" / workload_id / "kernel.cu"
                with self.subTest(workload_id=workload_id):
                    self.assertTrue(source.is_file(), source)
                    result = subprocess.run(
                        [
                            compiler,
                            "-x",
                            "cuda",
                            "-c",
                            str(source),
                            "-o",
                            str(output_root / f"{workload_id}.o"),
                            f"--cuda-path={os.environ.get('CUDA_PATH', '/usr/local/cuda')}",
                            f"--cuda-gpu-arch={os.environ.get('CUDA_ARCH', 'sm_86')}",
                        ],
                        cwd=REPO_ROOT,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_provenance_records_fixed_sources_and_launch_contracts(self) -> None:
        catalog = load_catalog(RQ1_ROOT)
        for workload in catalog.workloads:
            with self.subTest(workload_id=workload.workload_id):
                provenance = load_provenance(RQ1_ROOT, workload)
                workload_root = (
                    RQ1_ROOT / workload.provenance_path
                ).resolve().parent
                self.assertEqual(
                    provenance.upstream.sha256,
                    EXPECTED_UPSTREAM_SHA256[workload.workload_id],
                )
                self.assertEqual(
                    sha256_file(workload_root / provenance.upstream.source_file),
                    EXPECTED_UPSTREAM_SHA256[workload.workload_id],
                )
                self.assertEqual(
                    (
                        provenance.execution.grid,
                        provenance.execution.block,
                        provenance.execution.dynamic_shared_bytes,
                    ),
                    EXPECTED_LAUNCH[workload.workload_id],
                )
                self.assertEqual(
                    provenance.execution.entry,
                    EXPECTED_ENTRY[workload.workload_id],
                )

    def test_each_extraction_patch_reconstructs_kernel_offline(self) -> None:
        catalog = load_catalog(RQ1_ROOT)
        with tempfile.TemporaryDirectory(prefix="rq1-patch-check-") as temp_dir:
            temp_root = Path(temp_dir)
            for workload in catalog.workloads:
                with self.subTest(workload_id=workload.workload_id):
                    provenance = load_provenance(RQ1_ROOT, workload)
                    source_root = (
                        RQ1_ROOT / workload.provenance_path
                    ).resolve().parent
                    check_root = temp_root / workload.workload_id
                    check_root.mkdir()
                    shutil.copyfile(
                        source_root / provenance.upstream.source_file,
                        check_root / "upstream.source",
                    )
                    result = subprocess.run(
                        [
                            "patch",
                            "--batch",
                            "--forward",
                            "upstream.source",
                            "-o",
                            "kernel.cu",
                            "-i",
                            str(source_root / provenance.extraction.patch_file),
                        ],
                        cwd=check_root,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(
                        (check_root / "kernel.cu").read_bytes(),
                        (source_root / provenance.extraction.kernel_file).read_bytes(),
                    )

    def test_constraints_select_one_symbol_and_exact_contract(self) -> None:
        for workload_id in EXPECTED_WORKLOAD_IDS:
            path = RQ1_ROOT / "workloads" / workload_id / "constraints.json"
            with self.subTest(workload_id=workload_id):
                registry = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(registry["schema_version"], 1)
                self.assertEqual(registry["project"], f"rq1_{workload_id}")
                self.assertEqual(len(registry["overrides"]), 1)
                override = registry["overrides"][0]
                self.assertEqual(
                    override["symbol_name"], EXPECTED_ENTRY[workload_id]
                )
                launch = override["launch_policy"]
                grid, block, shared = EXPECTED_LAUNCH[workload_id]
                self.assertEqual(tuple(launch["grid"]), grid)
                self.assertEqual(launch["block_candidates"], [block[0]])
                self.assertEqual(launch["physical_block_max"], block[0])
                self.assertEqual(launch["target_dynamic_shared_bytes"], shared)


if __name__ == "__main__":
    unittest.main()
