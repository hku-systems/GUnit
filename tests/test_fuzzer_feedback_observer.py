import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class FuzzerFeedbackObserverTest(unittest.TestCase):
    def test_async_paths_do_not_bucket_boolean_cfg_presence_maps(self) -> None:
        source = (
            REPO_ROOT / "cuda-fuzzer" / "src" / "bin" / "fuzzer_async.rs"
        ).read_text(encoding="utf-8")

        self.assertNotIn("HitcountsMapObserver", source)
        self.assertEqual(source.count("StdMapObserver::from_mut_ptr("), 3)


if __name__ == "__main__":
    unittest.main()
