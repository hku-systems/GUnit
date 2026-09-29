import unittest
from pathlib import Path

from benchmark.workloads.schema import (
    EXPECTED_WORKLOAD_IDS,
    load_catalog,
    load_suite,
    verify_catalog,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKLOAD_ROOT = REPO_ROOT / "benchmark" / "workloads"


class SharedWorkloadCatalogTests(unittest.TestCase):
    def test_catalog_has_one_canonical_twelve_kernel_corpus(self) -> None:
        catalog = load_catalog(WORKLOAD_ROOT)

        self.assertEqual(
            tuple(workload.workload_id for workload in catalog.workloads),
            EXPECTED_WORKLOAD_IDS,
        )
        self.assertEqual(verify_catalog(WORKLOAD_ROOT), [])

    def test_rq_suites_reference_the_shared_corpus_without_duplicates(self) -> None:
        catalog = load_catalog(WORKLOAD_ROOT)
        rq1 = load_suite(REPO_ROOT / "benchmark" / "rq1" / "suite.json", catalog)
        rq2 = load_suite(REPO_ROOT / "benchmark" / "rq2" / "suite.json", catalog)

        self.assertEqual(rq1.workload_ids, EXPECTED_WORKLOAD_IDS)
        self.assertEqual(
            rq2.workload_ids,
            (
                "apex_maybe_cast",
                "llama_upscale_f32_bilinear",
                "gpurir_generate_rir",
                "apex_index_mul_2d_vgeo",
                "cuda_samples_inverse_cnd",
                "synth_complex",
            ),
        )

    def test_rq2_adapters_live_in_the_shared_workload_directories(self) -> None:
        by_id = load_catalog(WORKLOAD_ROOT).by_id()

        for workload_id in ("apex_maybe_cast", "apex_index_mul_2d_vgeo"):
            workload = by_id[workload_id]
            self.assertIsNotNone(workload.adapter)
            self.assertTrue(workload.path(workload.adapter or "").is_file())
            self.assertTrue(
                workload.path(workload.adapter or "").is_relative_to(WORKLOAD_ROOT)
            )


if __name__ == "__main__":
    unittest.main()
