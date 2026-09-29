import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = REPO_ROOT / "tests" / "kernel_pipeline_smoke" / "fixtures" / "project"
DRIVERS_DIR = Path(__file__).resolve().parent
if str(DRIVERS_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVERS_DIR))

from shared_e2e import get_capture_e2e_context  # noqa: E402


class MiniProjectTest(unittest.TestCase):
    """Validate mini-project sources retained under fixtures/project/."""

    def test_project_files_exist(self) -> None:
        """All expected source files are present."""
        self.assertTrue(PROJECT_DIR.is_dir(), "project/ directory missing")
        self.assertTrue((PROJECT_DIR / "main.cu").is_file())
        self.assertTrue((PROJECT_DIR / "kernels.cu").is_file())
        self.assertTrue((PROJECT_DIR / "utils.cuh").is_file())

    def test_project_makefile_exists(self) -> None:
        """project/Makefile must exist for the two-level make setup."""
        self.assertTrue(
            (PROJECT_DIR / "Makefile").is_file(),
            "project/Makefile missing",
        )

    def test_project_makefile_no_rdc(self) -> None:
        """project/Makefile must NOT enable relocatable device code."""
        text = (PROJECT_DIR / "Makefile").read_text(encoding="utf-8")
        self.assertNotIn("-rdc=true", text)
        self.assertNotIn(" -dc ", text)
        self.assertNotIn("--relocatable-device-code", text)

    def test_project_has_multiple_cu_and_cuh(self) -> None:
        cu_files = list(PROJECT_DIR.glob("*.cu"))
        cuh_files = list(PROJECT_DIR.glob("*.cuh"))
        self.assertGreaterEqual(len(cu_files), 2)
        self.assertGreaterEqual(len(cuh_files), 1)

    def test_mini_project_present_in_shared_all_build(self) -> None:
        """Shared all-build should produce project/mini_project."""
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        self.assertTrue(ctx.mini_project_bin.is_file(), "mini_project missing in shared all-build")


if __name__ == "__main__":
    unittest.main()
