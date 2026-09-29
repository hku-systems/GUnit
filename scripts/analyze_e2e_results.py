#!/usr/bin/env python3
"""Analyze current e2e result JSON files and write a compact SUMMARY.md."""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = Path("test_e2e_workspace/runs")
DEFAULT_RESULT_FILES = (
    DEFAULT_WORKSPACE / "fixture_run_gpu" / "e2e_results.json",
    DEFAULT_WORKSPACE / "third_party_full" / "e2e_results.json",
)


def load_results(file_path):
    """Load and parse e2e_results.json file."""
    try:
        with open(file_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found, skipping", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse {file_path}: {e}", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        print(f"Error: Expected JSON object in {file_path}", file=sys.stderr)
        return None
    return data


def _counter_dict(counter):
    return dict(sorted(counter.items()))


def _display_path(path):
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return str(path)


def _phase1_failure_reason(phase1):
    return (
        phase1.get('failure_reason')
        or phase1.get('reason')
        or phase1.get('status')
        or 'unknown'
    )


def _rewrite_failure_reason(kernel, stages):
    kernel_id = kernel.get('kernel_id', '')
    phase2 = stages.get('phase2', {})
    for record in phase2.get('records', []):
        if record.get('kernel_id') != kernel_id:
            continue
        context = record.get('failure_context')
        if isinstance(context, dict) and context.get('reason_code'):
            return context['reason_code']
        return (
            record.get('failure_reason')
            or record.get('failure_detail')
            or record.get('phase2_status')
            or 'unknown'
        )
    return kernel.get('failure_reason') or kernel.get('phase2_status') or 'unknown'


def _build_failure_reason(build):
    stderr = build.get('stderr', '').lower()
    if 'undefined reference' in stderr or 'cannot find' in stderr or 'ld:' in stderr:
        return 'link_error'
    if 'error:' in stderr:
        return 'compilation_error'
    return build.get('status') or 'build_failed'


def _fuzzer_failure_reason(fuzzer):
    stderr = fuzzer.get('stderr', '').lower()
    if 'no cuda-capable device' in stderr or 'no cuda device' in stderr:
        return 'no_gpu'
    if 'timeout' in stderr:
        return 'timeout'
    if 'assertion' in stderr:
        return 'assertion_failure'
    if 'crash' in stderr or 'segfault' in stderr:
        return 'crash'
    return fuzzer.get('status') or 'fuzzer_failed'


def _write_e2e_summary(report, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# E2E Results Summary\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Overall\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total candidates | {report['overall']['candidates_total']} |\n")
        f.write(f"| Total kernels | {report['overall']['kernels_total']} |\n")
        f.write(f"| Successful kernels | {report['overall']['kernels_success']} |\n")
        f.write(f"| Failed kernels | {report['overall']['kernels_failed']} |\n\n")

        for title, key in (
            ("Phase 1 Failures", "phase1_failures"),
            ("Rewrite Failures", "rewrite_failures"),
            ("Build Failures", "build_failures"),
            ("Fuzzer Failures", "fuzzer_failures"),
        ):
            f.write(f"## {title}\n\n")
            failures = report[key]
            if failures:
                f.write("| Reason | Count |\n")
                f.write("|--------|-------|\n")
                for reason, count in sorted(failures.items(), key=lambda item: (-item[1], item[0])):
                    f.write(f"| {reason} | {count} |\n")
            else:
                f.write("*None*\n")
            f.write("\n")

        f.write("## Successful Kernels\n\n")
        if report['successful_kernels']:
            for kernel in report['successful_kernels']:
                f.write(
                    f"- `{_display_path(kernel['candidate'])}` :: "
                    f"`{kernel['kernel']}` (`{kernel['kernel_id']}`)\n"
                )
        else:
            f.write("*None*\n")
        f.write("\n")

        f.write("## Recommendations\n\n")
        if report['fuzzer_failures'].get('no_gpu', 0):
            f.write("- Re-run fuzzer stages on a GPU-enabled system for no-GPU failures.\n")
        if report['rewrite_failures']:
            f.write("- Review rewrite failure reasons and add stable limitations where support is intentionally missing.\n")
        if report['build_failures']:
            f.write("- Check generated backend link inputs and CUDA library visibility for build failures.\n")
        if not (report['rewrite_failures'] or report['build_failures'] or report['fuzzer_failures']):
            f.write("- No action required for the analyzed results.\n")


def analyze_results(result_files, output_path):
    """Aggregate e2e result files and write a compact markdown summary."""
    report = {
        'overall': {
            'files_analyzed': 0,
            'candidates_total': 0,
            'kernels_total': 0,
            'kernels_success': 0,
            'kernels_failed': 0,
        },
        'phase1_failures': defaultdict(int),
        'rewrite_failures': defaultdict(int),
        'build_failures': defaultdict(int),
        'fuzzer_failures': defaultdict(int),
        'successful_kernels': [],
        'failed_kernels': [],
    }

    for file_path in result_files:
        data = load_results(file_path)
        if not data:
            continue
        report['overall']['files_analyzed'] += 1
        report['overall']['candidates_total'] += data.get('candidate_count', 0)
        for result in data.get('results', []):
            candidate = result.get('candidate', '')
            stages = result.get('stages', {})
            phase1 = stages.get('phase1', {})
            if phase1 and not phase1.get('ok'):
                report['phase1_failures'][_phase1_failure_reason(phase1)] += 1

            for kernel in result.get('kernels', []):
                kernel_id = kernel.get('kernel_id', '')
                display_name = kernel.get('display_name', '')
                kernel_stages = kernel.get('stages', {})
                report['overall']['kernels_total'] += 1

                if kernel.get('phase2_status') == 'failed':
                    reason = _rewrite_failure_reason(kernel, stages)
                    report['rewrite_failures'][reason] += 1
                    report['failed_kernels'].append({
                        'candidate': candidate,
                        'kernel': display_name,
                        'kernel_id': kernel_id,
                        'phase': 'rewrite',
                        'reason': reason,
                    })
                    continue

                build = kernel_stages.get('build_rapid2', {})
                if build and not build.get('ok'):
                    reason = _build_failure_reason(build)
                    report['build_failures'][reason] += 1
                    report['failed_kernels'].append({
                        'candidate': candidate,
                        'kernel': display_name,
                        'kernel_id': kernel_id,
                        'phase': 'build',
                        'reason': reason,
                    })
                    continue

                fuzzer = kernel_stages.get('fuzzer', {})
                if fuzzer and not fuzzer.get('ok'):
                    reason = _fuzzer_failure_reason(fuzzer)
                    report['fuzzer_failures'][reason] += 1
                    report['failed_kernels'].append({
                        'candidate': candidate,
                        'kernel': display_name,
                        'kernel_id': kernel_id,
                        'phase': 'fuzzer',
                        'reason': reason,
                    })
                    continue

                if fuzzer.get('ok'):
                    report['overall']['kernels_success'] += 1
                    report['successful_kernels'].append({
                        'candidate': candidate,
                        'kernel': display_name,
                        'kernel_id': kernel_id,
                    })

    report['overall']['kernels_failed'] = (
        report['overall']['kernels_total'] - report['overall']['kernels_success']
    )
    for key in ('phase1_failures', 'rewrite_failures', 'build_failures', 'fuzzer_failures'):
        report[key] = _counter_dict(report[key])

    _write_e2e_summary(report, Path(output_path))
    return report


def main(argv=None):
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_files",
        nargs="*",
        type=Path,
        help="e2e_results.json files to analyze; defaults to the standard fixture and third_party runs",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_WORKSPACE / "SUMMARY.md",
        help="Markdown summary output path",
    )
    args = parser.parse_args([] if argv is None else argv)

    result_files = tuple(args.result_files) if args.result_files else DEFAULT_RESULT_FILES
    report = analyze_results(result_files, args.output)

    if report["overall"]["files_analyzed"] == 0:
        print("Error: No result files found or all failed to load", file=sys.stderr)
        return 1

    print(f"Report generated: {args.output}")
    print(f"Total files analyzed: {report['overall']['files_analyzed']}")

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
