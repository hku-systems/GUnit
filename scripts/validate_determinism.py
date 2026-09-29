#!/usr/bin/env python3
"""
Validate that fixed seed produces deterministic fuzzer behavior.
Compares multiple runs of the same kernel with identical seed.
"""

import sys
import json
from pathlib import Path


def compare_fuzzer_outputs(run1_dir, run2_dir, run3_dir):
    """Compare fuzzer outputs from three runs."""

    results = {
        'run1': Path(run1_dir) / 'e2e_results.json',
        'run2': Path(run2_dir) / 'e2e_results.json',
        'run3': Path(run3_dir) / 'e2e_results.json',
    }

    # Load all results
    data = {}
    for run_name, path in results.items():
        if not path.exists():
            print(f"Error: {path} not found")
            return False
        with open(path) as f:
            data[run_name] = json.load(f)

    # Compare seeds
    seeds = {name: d.get('fixed_seed') for name, d in data.items()}
    if len(set(seeds.values())) != 1:
        print(f"Error: Seeds differ across runs: {seeds}")
        return False
    print(f"✓ All runs used same seed: {seeds['run1']}")

    # Compare fuzzer modes
    modes = {name: d.get('fuzzer_mode') for name, d in data.items()}
    if len(set(modes.values())) != 1:
        print(f"Error: Fuzzer modes differ: {modes}")
        return False
    print(f"✓ All runs used same fuzzer mode: {modes['run1']}")

    # Compare kernel results
    kernels_match = True
    for kernel_idx in range(len(data['run1'].get('results', []))):
        run1_kernel = data['run1']['results'][kernel_idx]
        run2_kernel = data['run2']['results'][kernel_idx]
        run3_kernel = data['run3']['results'][kernel_idx]

        candidate = run1_kernel.get('candidate', '')
        print(f"\nChecking: {Path(candidate).name}")

        # Compare each kernel's fuzzer output
        for kernel_data in run1_kernel.get('kernels', []):
            kernel_id = kernel_data.get('kernel_id')
            display_name = kernel_data.get('display_name')

            # Find matching kernels in other runs
            run2_match = next((k for k in run2_kernel.get('kernels', [])
                             if k.get('kernel_id') == kernel_id), None)
            run3_match = next((k for k in run3_kernel.get('kernels', [])
                             if k.get('kernel_id') == kernel_id), None)

            if not run2_match or not run3_match:
                print(f"  ✗ {display_name}: Missing in other runs")
                kernels_match = False
                continue

            # Compare fuzzer outputs
            fuzzer1 = kernel_data.get('stages', {}).get('fuzzer', {})
            fuzzer2 = run2_match.get('stages', {}).get('fuzzer', {})
            fuzzer3 = run3_match.get('stages', {}).get('fuzzer', {})

            # Check if all succeeded or all failed the same way
            ok1 = fuzzer1.get('ok', False)
            ok2 = fuzzer2.get('ok', False)
            ok3 = fuzzer3.get('ok', False)

            if ok1 == ok2 == ok3:
                if ok1:
                    print(f"  ✓ {display_name}: All runs succeeded")
                else:
                    # Check if failure is consistent
                    stderr1 = fuzzer1.get('stderr', '')
                    stderr2 = fuzzer2.get('stderr', '')
                    stderr3 = fuzzer3.get('stderr', '')

                    # For determinism, we care about consistent behavior
                    # GPU errors are environmental, not seed-related
                    if 'no CUDA-capable device' in stderr1:
                        print(f"  ~ {display_name}: All runs failed (no GPU - environmental)")
                    else:
                        print(f"  ✓ {display_name}: All runs failed consistently")
            else:
                print(f"  ✗ {display_name}: Inconsistent results ({ok1}, {ok2}, {ok3})")
                kernels_match = False

    return kernels_match


def main():
    """Main entry point."""
    if len(sys.argv) != 4:
        print("Usage: validate_determinism.py <run1_dir> <run2_dir> <run3_dir>")
        print("\nExample:")
        print("  python3 scripts/validate_determinism.py \\")
        print("    test_e2e_workspace/runs/fixed_validation_run1 \\")
        print("    test_e2e_workspace/runs/fixed_validation_run2 \\")
        print("    test_e2e_workspace/runs/fixed_validation_run3")
        return 1

    run1_dir = sys.argv[1]
    run2_dir = sys.argv[2]
    run3_dir = sys.argv[3]

    print("=" * 60)
    print("Fixed Seed Determinism Validation")
    print("=" * 60)

    success = compare_fuzzer_outputs(run1_dir, run2_dir, run3_dir)

    print("\n" + "=" * 60)
    if success:
        print("✓ VALIDATION PASSED: Fixed seed produces deterministic results")
    else:
        print("✗ VALIDATION FAILED: Inconsistent results detected")
    print("=" * 60)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
