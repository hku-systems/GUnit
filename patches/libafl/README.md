# LibAFL patches

`LibAFL/` is pinned to upstream AFLplusplus/LibAFL commit `824f5535`, the
PR #3372 merge. Apply these three patches in order after initializing the
submodule and before building `cuda-fuzzer`, whose LibAFL crates are local path
dependencies:

```bash
git submodule update --init LibAFL
(cd LibAFL && git am ../patches/libafl/*.patch)
```

The patches preserve the three commits previously carried by the LibAFL fork:

1. `0001-introspection-print-cycles-and-add-async-perf-featur.patch` adds async
   polling and pending-task lookup performance features and prints cycle counts.
   The RQ4 profiling path uses this introspection data.
2. `0002-llmp-write-exit-info-via-write_unaligned.patch` writes LLMP client exit
   information into the message buffer with an unaligned write, avoiding a
   read of uninitialized buffer contents.
3. `0003-Fix-Rust-compiler-compatibility.patch` adjusts LibAFL code for the
   Rust toolchain used to build the fuzzer.

After applying all three, the LibAFL source tree matches the former fork pin
`4c7ef79a`. A fresh submodule update restores the upstream checkout, so apply
the patches again before a later fuzzer build if that happens.
