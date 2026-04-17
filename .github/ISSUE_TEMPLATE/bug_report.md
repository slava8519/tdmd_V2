---
name: Bug report
about: A defect in TDMD behavior, build, or docs
title: "[BUG] "
labels: bug
assignees: ''
---

## Summary

<!-- One-sentence description. -->

## Environment

- TDMD commit: `<git rev-parse HEAD>`
- BuildFlavor: Fp64ReferenceBuild / Fp64ProductionBuild / MixedFast / MixedFastAggressive / Fp32Experimental
- ExecProfile: Reference / MixedFast / ...
- OS / kernel:
- GPU / compute cap:
- CUDA version:
- Compiler (GCC / Clang / nvcc):

## Reproduction

```
<exact commands>
```

Input files (attach or link):

## Expected vs. actual

**Expected:**

**Actual:**

## Logs / output

```
<stderr, ctest output, stack trace, etc.>
```

## Affects

- [ ] Determinism / bitwise reproducibility
- [ ] Numerical accuracy (differential vs LAMMPS)
- [ ] Performance regression (> 5%)
- [ ] Build / CI
- [ ] Documentation only

## Severity

- [ ] P0 — anchor-test broken / deterministic oracle compromised
- [ ] P1 — production path impacted
- [ ] P2 — edge case, workaround exists
- [ ] P3 — cosmetic / doc

## Additional context

<!-- Related issues, hypothesis of root cause, first-seen commit. -->
