"""Unit tests for demangled filter helpers.

Loads test cases from fixtures/demangled_filter_cases.json and verifies
that the extraction functions produce expected outputs for each case.
"""

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "kernel-smoke"
sys.path.insert(0, str(SCRIPT_ROOT))

from utils.demangle import (  # type: ignore[import-not-found]
    HAS_DEMANGLER,
    demangle_symbol,
    qualified_name_from_demangled,
    strip_last_paren_group,
    strip_template_args,
)

FIXTURES_DIR = REPO_ROOT / "tests" / "kernel_pipeline_smoke" / "fixtures"
CASES_PATH = FIXTURES_DIR / "demangled_filter_cases.json"


def _load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


class TestQualifiedNameFromDemangled(unittest.TestCase):
    """Test qualified_name_from_demangled against fixture cases."""

    def test_fixture_cases(self) -> None:
        cases = _load_cases()
        self.assertGreater(len(cases), 0, "No test cases loaded")
        for case in cases:
            with self.subTest(name=case["name"]):
                result = qualified_name_from_demangled(case["demangled"])
                expected = case["expected_qualified"]
                if expected is None:
                    # None means we don't assert on qualified name
                    continue
                self.assertEqual(
                    result,
                    expected,
                    f"Case '{case['name']}': "
                    f"demangled={case['demangled']!r} -> "
                    f"got {result!r}, expected {expected!r}",
                )


class TestDemangleSymbol(unittest.TestCase):
    """Smoke tests for artifact-mode symbol demangling."""

    def test_demangles_with_available_backend(self) -> None:
        if not HAS_DEMANGLER:
            self.skipTest("no cxxfilt package or c++filt binary available")

        self.assertEqual(demangle_symbol("_Z16templated_kernelIiEvPT_S0_"), "void templated_kernel<int>(int*, int)")


class TestStripTemplateArgs(unittest.TestCase):
    """Smoke tests for _strip_template_args."""

    def test_simple(self) -> None:
        strip = strip_template_args
        self.assertEqual(strip("Foo<int>"), "Foo")
        self.assertEqual(strip("A<B<C>>"), "A")
        self.assertEqual(strip("A<B>::C<D>"), "A::C")
        self.assertEqual(strip("no_templates"), "no_templates")


class TestStripLastParenGroup(unittest.TestCase):
    """Smoke tests for _strip_last_paren_group."""

    def test_simple(self) -> None:
        strip = strip_last_paren_group
        self.assertEqual(strip("foo(int, double)"), "foo")
        self.assertEqual(strip("ns::bar(x) const"), "ns::bar")
        self.assertEqual(strip("plain"), "plain")
        self.assertEqual(strip("a(b(c))"), "a")

    def test_nested(self) -> None:
        strip = strip_last_paren_group
        # Only the outermost last group is stripped
        self.assertEqual(strip("f(g(h(x)))"), "f")


if __name__ == "__main__":
    unittest.main()
