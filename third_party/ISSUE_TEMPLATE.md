# Third-party upstream issue report template

Use this template when reporting a confirmed target-side bug (see `BUG_CANDIDATES.md`) to an upstream third-party repository. Write the issue in English. Keep only the sections that apply; delete the rest. Replace every `<...>` placeholder and verify all links and line numbers against the upstream branch/commit you actually inspected.

---

**Title:** `<kernel_or_function>`: <short defect summary, e.g. "divergent __syncthreads() inside conditional (UB / potential deadlock)">

**Body:**

Found while fuzzing <project> kernels with a CUDA kernel fuzzer (RAPID). The code at [`<path>:<lines>`](<permalink-pinned-to-commit>) (current `<branch>`, commit `<short-sha>`):

```cuda
<verbatim upstream code snippet>
```

**Problem 1 — <defect name> (<classification>).** <What the code does, why it is wrong, and the exact condition that triggers it. State the rule it violates (e.g. CUDA programming guide barrier semantics) and the concrete consequence (deadlock, data race, wrong result).>

**Problem 2 — <defect name> (<classification>).** <Optional; repeat the structure above for each distinct defect in the same report.>

**Suggested fix.** <Minimal corrected code or precise description of the change.>

```cuda
<optional corrected code snippet>
```

---

Reported-by: GUnit (persistent CUDA kernel testing framework) <rapid@spcsky.com>
Generated-by: Kimi Code CLI v<version> (Moonshot AI), with manual verification against upstream `<branch>` @ `<short-sha>`

---

## Checklist before submitting

- The defect is in the vendored upstream code, not introduced by RAPID rewrite/launch (classify first per `BUG_CANDIDATES.md`).
- Links are pinned to a commit permalink, and the referenced line numbers match that commit.
- The code snippet is verbatim upstream, not the rewritten/Phase2 output.
- The trigger condition is stated precisely enough for upstream to reproduce without our infrastructure.
- After submitting, record the issue URL in the corresponding `BUG_CANDIDATES.md` entry.
