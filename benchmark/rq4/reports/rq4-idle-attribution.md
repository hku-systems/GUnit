# RQ4 GUnit-GPU idle attribution

## Conclusion

The low `kernel_exec` share of the eight selected GUnit workloads is not a
supply-pacing regression, CPU-feedback bottleneck, or launch overhead.  Seven
are light-target cases whose per-task host-to-GPU dispatch/collection path is
longer than useful device work.  Their `idle` is therefore expected under the
current per-task transport contract.  `cutlass_gemm` has the same host-side
limit plus a separate, improvable device `feedback_init` hotspot; removing that
hotspot alone would mostly turn active feedback time into more device wait
unless transport is also shortened.

No selected workload is `host-mutation-bound`, `host-feedback-bound`, or
`launch-overhead-bound` in this capture.  The three controls validate the
classification: their target work exceeds or approaches the transport floor,
so target execution becomes the largest device segment.

## Measurement and denominator

Input is
`benchmark/rq4/results/profile-mutating-30s-seed1-r1-w2-default22-20260811`.
Only the `configuration=gunit` records are used below.

`device_kernel` values are accumulated `clock64()` cycles.  Because the
persistent kernel is already resident when capture starts, Nsight reports no
closed per-task kernel intervals (`gpu_kernel_time_ns=0`); cycles cannot be
converted using that field.  The table therefore uses a data-only,
wall-calibrated conversion:

```text
segment_us_per_iter =
    (segment_cycles / sum(all_device_cycles))
    * (sum(main_loop_ns + cpu_feedback_ns) / device_iterations) / 1000
```

This preserves the exact cycle shares used by `overhead_breakdown.pdf` while
expressing the continuously resident loop's per-completion service interval in
microseconds.  In every row, `coverage`, `evaluate`, and `scheduler_stage`
counts equal `device_iterations`, so their displayed `ns/call` is also their
amortized `ns/iter`.  `other_idle` is the single root profiling frame: its raw
`ns/call` is the whole unscoped interval, and the bracketed value is the useful
amortized `ns/iter`.  External submit/poll and materializer release run on
worker threads outside the owner-thread scopes, so their main-loop counts are
zero; that means "not instrumented in this domain", not zero cost.  CPU
feedback is shown as raw `ns/call` with amortized `ns/iter` in brackets.

Device feedback is `feedback_init + feedback_prepare + feedback_merge`.
Device dispatch is `input_decode + signal + bookkeeping`.  CUDA
`memcpy+sync` is aggregate CUDA API time divided by completed device
iterations; its worker-thread components may overlap, so it is evidence for
the limiting chain rather than an additive wall-time decomposition.

## Per-kernel decomposition

| kernel | device idle us/iter | target us/iter | device feedback us/iter | device dispatch us/iter | host sub/poll/rel calls | host cov/eval/sched ns/call | host other_idle ns/call [ns/iter] | CPU feedback ns/call [ns/iter] | CUDA memcpy+sync us/iter | attribution |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| apex_maybe_cast | 45.40 (74.9%) | 10.22 (16.9%) | 2.71 (4.5%) | 2.26 (3.7%) | 0/0/0 | 263/751/12614 | 22938022805 [46775] | 270055 [188] | 34.36 | gpu-dispatch-bound; essential-light |
| shoc_reduction | 45.64 (81.5%) | 5.85 (10.4%) | 2.37 (4.2%) | 2.11 (3.8%) | 0/0/0 | 240/265/11455 | 23341501391 [44002] | n/a [0] | 32.95 | gpu-dispatch-bound; essential-light |
| shoc_radix_sort | 43.47 (67.4%) | 15.75 (24.4%) | 2.46 (3.8%) | 2.77 (4.3%) | 0/0/0 | 241/269/12715 | 23630539288 [51233] | n/a [0] | 39.84 | gpu-dispatch-bound; essential-light |
| shoc_scan | 45.63 (85.7%) | 3.37 (6.3%) | 2.37 (4.4%) | 1.90 (3.6%) | 0/0/0 | 221/259/10957 | 23310368630 [41837] | 86393 [0] | 37.21 | gpu-dispatch-bound; essential-light |
| cutlass_gemm | 35.18 (55.9%) | 5.12 (8.1%) | 20.79 (33.0%) | 1.86 (2.9%) | 0/0/0 | 227/270/9784 | 24873873338 [52671] | 124128 [1] | 34.66 | gpu-dispatch-bound + device feedback-init hotspot |
| flashattention_device1xn | 44.08 (85.0%) | 2.61 (5.0%) | 3.52 (6.8%) | 1.65 (3.2%) | 0/0/0 | 227/263/10499 | 23394920459 [40870] | n/a [0] | 32.30 | gpu-dispatch-bound; essential-light |
| apex_index_mul_2d_vgeo | 41.72 (67.7%) | 14.72 (23.9%) | 2.65 (4.3%) | 2.53 (4.1%) | 0/0/0 | 225/392/12801 | 23224781687 [48147] | 232606 [57] | 36.09 | gpu-dispatch-bound; essential-light |
| cuda_samples_inverse_cnd | 39.77 (77.8%) | 6.02 (11.8%) | 3.57 (7.0%) | 1.75 (3.4%) | 0/0/0 | 236/300/10042 | 23523786402 [40508] | 282778 [22] | 30.15 | gpu-dispatch-bound; essential-light |
| **controls** | | | | | | | | | | |
| synth_complex | 17.46 (13.1%) | 111.49 (83.8%) | 2.36 (1.8%) | 1.66 (1.3%) | 0/0/0 | 243/447/12093 | 27003989004 [120133] | 216030 [59] | 39.66 | true-device-bound control |
| pytorch_batchnorm | 15.62 (18.8%) | 54.31 (65.2%) | 10.89 (13.1%) | 2.45 (2.9%) | 0/0/0 | 233/293/11623 | 25450812321 [71119] | n/a [0] | 35.15 | true-device-bound control |
| llama_upscale_f32_bilinear | 30.96 (41.2%) | 38.05 (50.7%) | 3.72 (5.0%) | 2.33 (3.1%) | 0/0/0 | 251/678/13040 | 24172453519 [60950] | 269440 [135] | 38.53 | balanced/device-bound control |

## Waiting chain and per-kernel attribution

For the eight selected kernels, device idle is 35.18--45.64 us/iter while
target execution is only 2.61--15.75 us/iter.  Host `scheduler_stage`, which
encloses supply-stage receive/fallback mutation and owner-side submission, is
9.78--12.80 us/call.  It contains no long producer-starvation wait.
Coverage plus exclusive evaluate plus amortized CPU feedback is only
0.48--1.20 us/iter, ruling out host feedback as the service-rate limiter.

The owner-thread root `other_idle` is instead 40.51--52.67 us/iter.  In this
external-retirement mode that frame contains the blocking wait for prepared
retirement batches plus unscoped orchestration.  Independently, the CUDA API
records show 30.15--39.84 us/iter in memcpy plus synchronization, approximately
four memcpy calls and one synchronization call per device iteration.  There
are zero captured launch calls: RAPID2 launches the persistent kernel before
the profiling window and performs no per-input kernel launch.  Taken together,
the observed wait chain is:

```text
supply input (not starved)
  -> dispatch copies / ready publication
  -> persistent target + device feedback
  -> result/feedback copies + stream synchronization
  -> materializer/ordered retirement handoff
  -> owner coverage/evaluate (small)
  -> next admitted task
```

Thus `apex_maybe_cast`, `shoc_reduction`, `shoc_radix_sort`, `shoc_scan`,
`flashattention_device1xn`, `apex_index_mul_2d_vgeo`, and
`cuda_samples_inverse_cnd` are `gpu-dispatch-bound` in the sense of the
host-driven per-task CUDA transport/collection path, not device-side dispatch
instructions.  Their idle is **essential-light** under the current per-task
contract: it is not evidence that the repaired supply/retirement pacing has
regressed.  More producer or retirement workers do not remove this fixed
service floor.

`cutlass_gemm` has the same transport floor, but its device feedback is a real
secondary hotspot.  The plotted 33.0% is entirely device-side: 31.5%
(19.85 us/iter) is `feedback_init`, 1.5% (0.93 us/iter) is
`feedback_prepare`, and 0.018% (0.011 us/iter) is `feedback_merge`.  CPU
feedback is only 1 ns/iter amortized, so it is not the source of the bar.
Source tracing places `feedback_init` around coverage-map binding and
`rapid_feedback_clear_task_maps()`.  That routine clears fixed-size task maps
cooperatively; the generated CUTLASS backend uses a much narrower physical
block than the other profiled workloads, so each thread performs much more
clear work.  This explains the anomaly without attributing it to the target
GEMM itself.

## Synchronous-versus-asynchronous timing semantics

The raw `device_kernel` records for `libafl` and `libafl-plus` have
`idle_cycles=0` by construction.  Their origin wrapper is launched once per
input and begins `clock64()` timing inside that launch.  Time before launch,
host mutation/evaluation, copies, synchronization, and the gap until the next
launch cannot appear in the wrapper's `idle` bucket because no persistent
kernel is resident to observe that gap.  Normalizing only those device buckets
therefore produces 100% non-idle device work by definition; it is not a GPU
occupancy measurement.

`GUnit-GPU` is different: one persistent kernel remains resident, and its
`idle` bucket times the spin waiting for the next ready generation.  It sees
the whole host-induced gap that the serial origin wrapper cannot see.  The two
raw device-cycle decompositions are not directly comparable.

The final PDF's `LibAFL`/`LibAFL+` rows use yet another denominator:
`serial_critical_path` over wall time, with CUDA API time, CPU feedback, scaled
kernel time, and a residual `Idle`.  They are therefore not literally 100%
busy in the final figure, and CUDA API durations that block the host still do
not imply that the GPU is executing.  `GUnit-s-GPU`, unlike the origin lanes,
is itself a persistent-kernel cycle lane and does contain a measured wait
bucket; it is semantically closer to `GUnit-GPU` than `LibAFL`/`LibAFL+` are.

## Improvement boundary

- **Essential-light / no pacing fix warranted:** the seven kernels listed
  above.  Raising their device utilization requires changing the per-task
  transport contract (for example, batching copies/completions, eliminating a
  per-task synchronization point, or reducing result bytes), not retuning the
  already-fed supply/retirement queues.
- **Concrete device-side improvement:** `cutlass_gemm` feedback initialization.
  Candidate directions are generation/epoch-based clearing, sparse or lazy
  reset, or a block-width-independent clear strategy.  This can reduce device
  overhead, but transport must also be shortened before it necessarily raises
  throughput; otherwise the removed feedback time becomes additional wait.
- **Not supported by this capture:** host-mutation tuning, CPU-feedback tuning,
  or launch-overhead tuning as explanations for the eight idle-heavy rows.

## Reproduction

The table is computed from an untracked raw profile directory
(`benchmark/rq4/results/` is ignored experiment output). Regenerate one with
the mutating profile (`RQ4_MODE=mutating bash benchmark/rq4/run_profile.sh`,
which produces `benchmark/rq4/results/profile-mutating-30s-seed1-r1-w2-*`),
then run from the repository root:

```bash
python3 benchmark/rq4/reports/rq4_idle_attribution.py \
  benchmark/rq4/results/<profile-mutating-dir>
```

The original capture analyzed here was
`benchmark/rq4/results/profile-mutating-30s-seed1-r1-w2-default22-20260811`.

The script reads only `profiles.jsonl` and each referenced
`*.profiling.jsonl`, validates the relevant counts, prints the table above,
and prints the ranges and CUTLASS feedback split used in the conclusions.
