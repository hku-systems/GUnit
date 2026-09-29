import unittest


class RQ4SchemaTests(unittest.TestCase):
    def test_default_window_size_is_two(self) -> None:
        from benchmark.rq4.schema import DEFAULT_WINDOW_SIZE

        self.assertEqual(DEFAULT_WINDOW_SIZE, 2)

    def test_cumulative_ablation_has_exactly_five_ordered_levels(self) -> None:
        from benchmark.rq4.schema import configurations

        configs = configurations(window_size=4)
        self.assertEqual(
            [config.name for config in configs],
            ["cufuzz", "libafl", "libafl-plus", "gunit-sync", "gunit"],
        )
        self.assertEqual(
            [config.artifact_backend for config in configs],
            ["cufuzz", "origin-no-feedback", "origin", "rapid", "rapid2"],
        )
        self.assertEqual(
            [config.window_size for config in configs],
            [None, None, None, 4, 4],
        )
        self.assertEqual(
            [config.feedback_enabled for config in configs],
            [False, False, True, True, True],
        )

    def test_window_size_must_be_supported(self) -> None:
        from benchmark.rq4.schema import configurations

        for value in (0, 33):
            with self.subTest(value=value), self.assertRaises(ValueError):
                configurations(window_size=value)


if __name__ == "__main__":
    unittest.main()
