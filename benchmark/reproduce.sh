#!/usr/bin/env bash
# Unified RQ1/RQ2/RQ4 reproduction driver.
#
# Modes:
#   verify  - source/provenance gates + the GPU-free RQ4 unit-test gate
#   smoke   - minimal end-to-end campaigns proving the pipeline runs
#             (RQ1: all standalone workloads, 1 rep x 1s; RQ2: whole catalog,
#             1 rep x 1s; RQ4: unit-test gate + one-workload 1s profile when a
#             timing build is given)
#   full    - paper-protocol campaigns against the given build roots
#             (RQ1: both roots, 2 rep x 30s + paper CSV export; RQ2: full
#             9-config matrix, 5 seeds x 60s + result validation; RQ4: runs
#             benchmark/rq4/run_profile.sh unless REPRO_SKIP_RQ4=1)
#
# All campaigns are resumable: re-running a mode skips recorded cells.
# Build roots are NOT rebuilt here; build them per benchmark/rq1/README.md
# and benchmark/rq2/README.md, then point the env vars at them.
#
# Common overrides (env):
#   REPRO_GPU              GPU index for campaigns (default: 0)
#   REPRO_RQ1_STANDALONE   RQ1 standalone build root
#   REPRO_RQ1_IMPORTED     RQ1 imported build root
#   REPRO_RQ2_BUILD        RQ2 build root
#   REPRO_RQ4_TIMING_BUILD RQ4 timing build root (smoke profile only)
#   REPRO_RQ4_FUZZER_DIR   cargo target dir holding profiling fuzzer binaries
#   REPRO_RQ2_SMOKE_WORKLOADS  space-separated workload filter for RQ2 smoke
#   REPRO_RQ4_SMOKE_WORKLOAD   workload for the RQ4 smoke profile
#   REPRO_SKIP_RQ4         full mode: skip the RQ4 profile (default: run it)
#   REPRO_OUT_SUFFIX       result-directory suffix (default: latest; keep the
#                          same suffix to resume, pick a new one to restart)
#   REPRO_TMPDIR           scratch dir for compiler/build temporaries
#                          (default: .rapid/reproduce-tmp, kept off /tmp)
#
# Failure semantics: RQ1/RQ2 campaign runners record failed cells in
# failures.jsonl but still exit zero. This driver treats any recorded failure
# as an error. Failed cells are not retried in place; to rerun them, remove
# the result directory or choose a fresh REPRO_OUT_SUFFIX.
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"

# Keep build/compiler temporaries off the (often small) root filesystem.
repro_tmp=${REPRO_TMPDIR:-$repo/.rapid/reproduce-tmp}
mkdir -p "$repro_tmp"
export TMPDIR="$repro_tmp"

python=${REPRO_PYTHON:-$repo/.venv/bin/python}
fuzzer=${REPRO_FUZZER:-$repo/cuda-fuzzer/target/release/fuzzer}
fuzzer_async=${REPRO_FUZZER_ASYNC:-$repo/cuda-fuzzer/target/release/fuzzer_async}
gpu=${REPRO_GPU:-0}
suffix=${REPRO_OUT_SUFFIX:-latest}

rq1_standalone=${REPRO_RQ1_STANDALONE:-benchmark/rq1/build/standalone-sm86-20260807}
rq1_imported=${REPRO_RQ1_IMPORTED:-benchmark/rq1/build/imported-sm86-20260807}
rq2_build=${REPRO_RQ2_BUILD:-benchmark/rq2/build/rq2-formal-sm86-20260802}
rq4_timing=${REPRO_RQ4_TIMING_BUILD:-}
rq4_fuzzer_dir=${REPRO_RQ4_FUZZER_DIR:-}

mode=${1:-smoke}
case "$mode" in
  verify|smoke|full) ;;
  *) echo "usage: $0 [verify|smoke|full]" >&2; exit 2 ;;
esac

require_executable() {
  if [[ ! -x $1 ]]; then
    echo "missing prerequisite: $1" >&2
    echo "$2" >&2
    exit 1
  fi
}

preflight() {
  require_executable "$python" "create the repo venv first (see README.md)"
  if [[ $mode != verify ]]; then
    require_executable "$fuzzer" "build fuzzers: cargo build --manifest-path cuda-fuzzer/Cargo.toml --release --bins"
    require_executable "$fuzzer_async" "build fuzzers: cargo build --manifest-path cuda-fuzzer/Cargo.toml --release --bins"
  fi
}
preflight

require_dir() {
  if [[ ! -d $1 ]]; then
    echo "missing build root: $1" >&2
    echo "build it first per the benchmark READMEs, or override the REPRO_* env var" >&2
    exit 1
  fi
}

# Campaign runners record failed cells but still exit zero; fail loudly here.
check_campaign_failures() {
  local failures=$1/failures.jsonl
  if [[ -s $failures ]]; then
    echo "campaign recorded $(wc -l < "$failures") failed cell(s): $failures" >&2
    echo "failed cells are not retried in place; use a fresh REPRO_OUT_SUFFIX to rerun" >&2
    exit 1
  fi
}

run_verify() {
  "$python" benchmark/rq1/verify.py
  "$python" benchmark/rq2/verify.py
  rq4_gate
}

rq1_campaign() { # build-root output warmup seconds reps
  require_dir "$1"
  "$python" -m benchmark.rq1.campaign \
    --build-root "$1" --output "$2" \
    --repetitions "$5" --warmup-runs "$3" --benchmark-seconds "$4" \
    --window-size 4 --gpu-device "$gpu" --timeout 900 \
    --fuzzer "$fuzzer" --fuzzer-async "$fuzzer_async"
  check_campaign_failures "$2"
}

rq2_campaign() { # output seconds reps extra-workload-args...
  local output=$1 seconds=$2 reps=$3
  shift 3
  require_dir "$rq2_build"
  "$python" -m benchmark.rq2.campaign \
    --build-root "$rq2_build" --output "$output" \
    --repetitions "$reps" --coverage-seconds "$seconds" \
    --seed-mode per-rep --gpu-device "$gpu" --timeout 900 \
    --fuzzer "$fuzzer" --fuzzer-async "$fuzzer_async" "$@"
  check_campaign_failures "$output"
}

rq4_gate() {
  "$python" -m unittest \
    tests.cuda_kernel.test_kernel_timing_build \
    tests.benchmark.test_rq4_build \
    tests.benchmark.test_rq4_campaign \
    tests.benchmark.test_rq4_profile \
    tests.benchmark.test_rq4_schema \
    tests.benchmark.test_rq4_trace_extract \
    tests.benchmark.test_rq4_profile_aggregate \
    tests.benchmark.test_rq4_profile_plot
}

rq4_smoke_profile() {
  if [[ -z $rq4_timing || -z $rq4_fuzzer_dir ]]; then
    echo "RQ4 smoke profile skipped: set REPRO_RQ4_TIMING_BUILD and REPRO_RQ4_FUZZER_DIR to enable" >&2
    return 0
  fi
  require_dir "$rq4_timing"
  require_executable "$rq4_fuzzer_dir/release/fuzzer" "build profiling fuzzers per benchmark/rq4/run_profile.sh"
  require_executable "$rq4_fuzzer_dir/release/fuzzer_async" "build profiling fuzzers per benchmark/rq4/run_profile.sh"
  "$python" -m benchmark.rq4.profile \
    --build-root "$rq1_standalone" --build-root "$rq1_imported" \
    --timing-build-root "$rq4_timing" \
    --workload "${REPRO_RQ4_SMOKE_WORKLOAD:-shoc_scan}" \
    --output "benchmark/rq4/results/smoke-repro-$suffix" \
    --profile-seconds 1 --warmup-runs 10 --window-size 2 \
    --gpu-device "$gpu" --timeout 240 \
    --fuzzer "$rq4_fuzzer_dir/release/fuzzer" \
    --fuzzer-async "$rq4_fuzzer_dir/release/fuzzer_async"
}

case "$mode" in
  verify)
    run_verify
    ;;
  smoke)
    run_verify
    rq1_campaign "$rq1_standalone" \
      "benchmark/rq1/results/smoke-repro-standalone-1r-1s-$suffix" 1 1 1
    # RQ2 smoke covers the whole catalog by default (measured ~1-2 min);
    # REPRO_RQ2_SMOKE_WORKLOADS="shoc_scan ..." restricts it.
    rq2_smoke_args=()
    for w in ${REPRO_RQ2_SMOKE_WORKLOADS:-}; do
      rq2_smoke_args+=(--workload "$w")
    done
    rq2_campaign "benchmark/rq2/results/smoke-repro-$suffix" 1 1 \
      "${rq2_smoke_args[@]}"
    rq4_smoke_profile
    ;;
  full)
    run_verify
    rq1_campaign "$rq1_standalone" \
      "benchmark/rq1/results/repro-standalone-2r-30s-$suffix" 100 30 2
    rq1_campaign "$rq1_imported" \
      "benchmark/rq1/results/repro-imported-2r-30s-$suffix" 100 30 2
    "$python" -m benchmark.rq1.export_paper \
      --input "benchmark/rq1/results/repro-standalone-2r-30s-$suffix" \
      --input "benchmark/rq1/results/repro-imported-2r-30s-$suffix" \
      --repetitions 2 \
      --output "benchmark/rq1/results/repro-rq1-paper-throughput-$suffix.csv"
    rq2_campaign "benchmark/rq2/results/repro-formal-9cfg-5seed-60s-$suffix" 60 5
    "$python" -m benchmark.rq2.verify_results \
      "benchmark/rq2/results/repro-formal-9cfg-5seed-60s-$suffix"
    if [[ ${REPRO_SKIP_RQ4:-0} == 1 ]]; then
      echo "RQ4 full profile skipped (REPRO_SKIP_RQ4=1); run manually with" >&2
      echo "  RQ4_BUILD_ROOTS='$rq1_standalone $rq1_imported' bash benchmark/rq4/run_profile.sh" >&2
    else
      require_dir "$rq1_imported"
      RQ4_BUILD_ROOTS="$rq1_standalone $rq1_imported" RQ4_GPU="$gpu" \
        RQ4_PYTHON="$python" RQ4_PYCACHE_PREFIX="$repro_tmp/rq4-pycache" \
        bash benchmark/rq4/run_profile.sh
    fi
    ;;
esac

echo "reproduce.sh [$mode] done"
