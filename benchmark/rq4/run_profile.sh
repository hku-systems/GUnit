#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=${RQ4_REPO:-"$(cd -- "$script_dir/../.." && pwd)"}
python=${RQ4_PYTHON:-"$repo/.venv/bin/python"}
gpu_device=${RQ4_GPU:-0}
cuda_arch=${RQ4_CUDA_ARCH:-sm_86}
cargo_bin=${RQ4_CARGO:-cargo}
mode=${RQ4_MODE:-mutating}
mutate_seconds=${RQ4_MUTATE_SECONDS:-30}
case "$mode" in
  fixed|mutating) ;;
  *)
    echo "RQ4_MODE must be fixed or mutating, got: $mode" >&2
    exit 2
    ;;
esac
if [[ "$mode" == mutating && ! "$mutate_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "RQ4_MUTATE_SECONDS must be a positive integer, got: $mutate_seconds" >&2
  exit 2
fi
build_roots_value=${RQ4_BUILD_ROOTS:-}
if [[ -z ${build_roots_value//[[:space:]]/} ]]; then
  echo "RQ4_BUILD_ROOTS must list the RQ1 build roots, separated by spaces" >&2
  echo "example: RQ4_BUILD_ROOTS='benchmark/rq1/build/standalone benchmark/rq1/build/imported' $0" >&2
  exit 2
fi
read -r -a build_roots <<< "$build_roots_value"
build_root_args=()
for build_root in "${build_roots[@]}"; do
  build_root_args+=(--build-root "$build_root")
done

revision=$(git -C "$repo" rev-parse --short HEAD)
if [[ "$mode" == mutating ]]; then
  export RAPID_FIXED_SEED=1
  pilot=benchmark/rq4/results/diagnostic-profile-pilot-apex-mutating-${mutate_seconds}s-seed1-r1-w2-feedback-v3-$revision
  full=benchmark/rq4/results/profile-mutating-${mutate_seconds}s-seed1-r1-w2-feedback-v3-$revision
  pilot_measure_args=(--mutate --mutate-seconds "$mutate_seconds")
  full_measure_args=(--mutate --mutate-seconds "$mutate_seconds")
else
  pilot=benchmark/rq4/results/diagnostic-profile-pilot-apex-5s-r1-w2-feedback-v2-$revision
  full=benchmark/rq4/results/profile-10s-r1-w2-feedback-v2-$revision
  pilot_measure_args=(--profile-seconds 5)
  full_measure_args=(--profile-seconds 10)
fi
timing_build=benchmark/rq4/build/timing-w2-feedback-v2-$revision
profile_target=benchmark/rq4/build/rq4-profile-target-$revision

cd "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX=${RQ4_PYCACHE_PREFIX:-/tmp/rq4-feedback-v2-pycache}

"$python" -m unittest \
  tests.cuda_kernel.test_kernel_timing_build \
  tests.benchmark.test_rq4_build \
  tests.benchmark.test_rq4_campaign \
  tests.benchmark.test_rq4_profile \
  tests.benchmark.test_rq4_schema \
  tests.benchmark.test_rq4_trace_extract \
  tests.benchmark.test_rq4_profile_aggregate \
  tests.benchmark.test_rq4_profile_plot

"$cargo_bin" build \
  --manifest-path cuda-fuzzer/Cargo.toml \
  --release \
  --bins \
  --features profiling \
  --target-dir "$profile_target"

"$python" -m benchmark.rq4.build \
  "${build_root_args[@]}" \
  --output "$timing_build" \
  --cuda-path /usr/local/cuda \
  --cuda-arch "$cuda_arch"

profile_common=(
  "${build_root_args[@]}"
  --timing-build-root "$timing_build"
  --window-size 2
  --gpu-device "$gpu_device"
  --timeout 180
  --nsys /usr/local/bin/nsys
  --fuzzer "$repo/$profile_target/release/fuzzer"
  --fuzzer-async "$repo/$profile_target/release/fuzzer_async"
)
if [[ "$mode" == fixed ]]; then
  profile_common+=(--warmup-runs 100)
fi

"$python" -m benchmark.rq4.profile \
  "${profile_common[@]}" \
  --workload apex_maybe_cast \
  --output "$pilot" \
  "${pilot_measure_args[@]}" \
  --repetitions 1

"$python" - "$pilot/profiles.jsonl" <<'PY'
import json
import sys
from pathlib import Path

from benchmark.rq4.profile_aggregate import validate_pilot

path = Path(sys.argv[1])
profiles = [json.loads(line) for line in path.read_text().splitlines() if line]
print(json.dumps(validate_pilot(profiles), sort_keys=True))
PY

"$python" -m benchmark.rq4.profile \
  "${profile_common[@]}" \
  --output "$full" \
  "${full_measure_args[@]}" \
  --repetitions 1

"$python" -m benchmark.rq4.profile_aggregate \
  --input "$full" \
  --output "$full/aggregate"

"$python" -m benchmark.rq4.profile_diagnostic \
  --input "$full" \
  --output "$full/aggregate/rq4_phase_diagnostic.csv"

"$python" -m benchmark.rq4.profile_plot \
  --input "$full/aggregate/rq4_overhead_breakdown.csv" \
  --output "$full/overhead_breakdown.pdf"
