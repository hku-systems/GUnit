import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class CanaryTest(unittest.TestCase):
    def test_clean_canary_passes_and_corruption_fails(self) -> None:
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("C++ compiler is unavailable")

        source = textwrap.dedent(
            r"""
            #include "sanitizer/canary.h"

            #include <cassert>
            #include <cstdint>
            #include <vector>

            void fake_kernel(uint8_t *payload, size_t payload_size,
                             bool write_oob) {
              if (write_oob) {
                payload[payload_size + 3] = 0xff;
                payload[payload_size + 4] = 0x00;
              }
            }

            int main() {
              std::vector<uint8_t> storage(32 + rapid::canary::kSize, 0);
              uint8_t *const payload = storage.data();
              uint8_t *const guard = payload + 32;
              rapid::canary::initialize(guard);

              rapid::canary::Mismatch mismatch{};
              fake_kernel(payload, 32, false);
              assert(rapid::canary::validate(guard, &mismatch));

              fake_kernel(payload, 32, true);
              assert(!rapid::canary::validate(guard, &mismatch));
              assert(mismatch.offset == 3);
              assert(mismatch.expected == 0x49);
              assert(mismatch.actual == 0xff);
              return 0;
            }
            """
        )

        with tempfile.TemporaryDirectory(prefix="rapid-canary-") as raw_tmp:
            tmp = Path(raw_tmp)
            source_path = tmp / "test.cpp"
            binary_path = tmp / "test"
            source_path.write_text(source, encoding="utf-8")
            build = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    str(source_path),
                    "-I",
                    str(REPO_ROOT / "cuda-kernel" / "utils"),
                    "-o",
                    str(binary_path),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            subprocess.run([str(binary_path)], check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
