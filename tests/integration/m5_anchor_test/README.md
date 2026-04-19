# M5 anchor-test — local slow-tier dissertation match

<!-- markdownlint-disable MD013 -->

**Status:** local-only (slow tier). CI is Option-A (public runner, no
GPU, no multi-rank >30s); this test is a mandatory pre-push gate per
D-M5-13 but is **not** wired into CI pipelines.

## Purpose

Reproduce Andreev §3.5 (2007 dissertation) Al FCC 10⁶-atom strong-scaling
experiment and check that TDMD's measured performance is within 10 % of
the dissertation reference (hardware-normalised). This is the primary
M5 science gate — its failure is project-level crisis
(CLAUDE.md "anchor-test (M5)").

## How to run

```bash
# 1. Build TDMD with MPI support.
cmake -S . -B build -DTDMD_ENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target tdmd -j

# 2. Drive the harness (invokes mpirun under the hood; picks up mpirun on $PATH).
python3 -m verify.harness.anchor_test_runner \
    --tdmd-bin build/tdmd \
    --lammps-bin "$LAMMPS_BIN" \
    --output build/t3_anchor_report.json

# Exit code: 0=GREEN, 1=YELLOW (abs-perf warn), 2=RED (efficiency fail),
#            3=infra/runtime error.
```

`$LAMMPS_BIN` is only consulted when `setup.data` is missing — the harness
then invokes `verify/data/t3_al_fcc_large_anchor/regen_setup.sh` once and
caches the regenerated file (≈30s / ≈95 MiB) for subsequent runs.

## What's checked

The pass/fail pseudocode lives in
`verify/benchmarks/t3_al_fcc_large_anchor/acceptance_criteria.md`. In
short:

- **fail_if** any measured efficiency is outside 10 % of the
  dissertation reference (`checks.yaml::dissertation_comparison.efficiency_relative`).
- **warn_if** any measured absolute performance (hardware-normalised)
  is outside 25 % of the reference
  (`absolute_performance_relative`). Warnings yield exit 1
  (YELLOW) — do not block merges but open a follow-up.

## Why not in CI

- Runs on ≥ 4 ranks (`{4, 8, 16}`). CI runners do not provide MPI with
  oversubscription, so multi-rank tests must live locally or on the
  future self-hosted runner (post-M6).
- Full run budget is 60 minutes, far exceeding the < 5-minute fast tier
  envelope (master spec §14.2, verify/SPEC §8).

## Pre-push ritual

1. Build (see above).
2. Run the harness.
3. Attach `build/t3_anchor_report.json` to the PR (or paste the
   `format_console_summary()` output in the PR body).
4. If status ≠ GREEN: classify the failure per
   `acceptance_criteria.md` §"Failure modes" and open a follow-up
   issue with the correct classification identifier
   (`REF_DATA_STALE`, `HARDWARE_NORMALIZATION_OFF`, …).
