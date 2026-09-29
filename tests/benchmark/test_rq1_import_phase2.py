import json
import tempfile
import unittest
from pathlib import Path

from benchmark.rq1.artifacts import envelope_payload_size
from benchmark.rq1.import_phase2 import imported_kernel_destination, load_import_map


class Rq1ImportPhase2Tests(unittest.TestCase):
    def test_import_preserves_the_kernel_id_directory_name(self) -> None:
        destination = imported_kernel_destination(
            Path("/run/workload"), Path("/rq2/kernels/kernel__12345678")
        )

        self.assertEqual(
            destination,
            Path("/run/workload/resolved/kernels/kernel__12345678"),
        )

    def test_payload_size_requires_a_consistent_task_envelope(self) -> None:
        payload = b"abcd"
        seed = bytearray(32) + bytearray(payload)
        seed[24:32] = len(payload).to_bytes(8, "little")

        self.assertEqual(envelope_payload_size(bytes(seed)), len(payload))
        with self.assertRaisesRegex(RuntimeError, "inconsistent"):
            envelope_payload_size(bytes(seed) + b"drift")

    def test_import_map_matches_the_selected_subset_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            kernel_dir = root / "kernel"
            (kernel_dir / "phase2").mkdir(parents=True)
            (kernel_dir / "manifest.json").write_text("{}", encoding="utf-8")
            map_path = root / "map.json"
            map_path.write_text(
                json.dumps({"selected": str(kernel_dir)}), encoding="utf-8"
            )

            result = load_import_map(map_path, ("selected",))

            self.assertEqual(result, {"selected": kernel_dir.resolve()})
            self.assertEqual(load_import_map(map_path, ()), {})
            with self.assertRaisesRegex(RuntimeError, "missing"):
                load_import_map(map_path, ("absent",))


if __name__ == "__main__":
    unittest.main()
