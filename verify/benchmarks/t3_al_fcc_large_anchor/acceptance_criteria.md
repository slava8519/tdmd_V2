# T3 anchor-test — Acceptance criteria

<!-- markdownlint-disable MD013 -->

The T3 anchor-test gates the entire M5 milestone. This document pins down
the exact pass/fail pseudocode the `anchor_test_runner` harness (T5.11)
implements, the failure-mode catalogue the M5 retrospective expects, and
the escalation path when one of the edges turns up red.

## Pass / fail pseudocode

```
for each N_procs in checks.yaml::ranks_to_probe:
    measured = run_tdmd(config.yaml, mpirun -np N_procs)
    reference = dissertation_reference_data.csv[N_procs]
    normalized_reference = reference.performance_mdps * hardware_normalization.ratio

    fail_if | measured.efficiency_pct - reference.efficiency_pct | / reference.efficiency_pct
        > checks.yaml::dissertation_comparison.efficiency_relative

    warn_if | measured.performance_mdps - normalized_reference | / normalized_reference
        > checks.yaml::dissertation_comparison.absolute_performance_relative

run_tdmd_serial = run_tdmd(config.yaml, single-rank)
run_lammps_serial = run_lammps(lammps_script.in, single-rank)

fail_if max |force_tdmd[i] - force_lammps[i]| / |force_lammps[i]|
    > checks.yaml::lammps_parity.step_0_force_relative

fail_if |etotal_tdmd[1000] - etotal_lammps[1000]| / |etotal_lammps[1000]|
    > checks.yaml::lammps_parity.step_1000_etotal_relative

fail_if |etotal[1000] - etotal[0]| / |etotal[0]|
    > checks.yaml::nve_drift.max_relative_drift

fail_if wall_clock_minutes > checks.yaml::runtime.wall_clock_budget_minutes
```

A full run reports three status levels:

| Level   | Meaning                                                                 |
|---------|-------------------------------------------------------------------------|
| `GREEN` | All `fail_if` clauses clean, no `warn_if` hits.                         |
| `YELLOW`| All `fail_if` clean, ≥ 1 `warn_if` hit. Merges allowed; open follow-up. |
| `RED`   | ≥ 1 `fail_if` hit. M5 merge blocked. Triage per §"Failure modes".       |

## Hardware-equivalence clause

The 10 % efficiency tolerance is expressed as a fraction of the
*dissertation-reported* value, not the *current-hardware-measured* value —
this matters because efficiency compounds differently than raw throughput.
A 2025-era workstation that runs 50× faster per core than 2007 Harpertown
should still show ~the same *efficiency* vs. N_procs, because parallel
efficiency is a ratio (strong scaling) and the ratio is hardware-era
invariant to first order. Empirical check: efficiency should stay within
10 %; absolute `steps/second` is multiplied by `hardware_normalization.py`
before the comparison.

"Equivalent hardware class" for this test is Intel Xeon Harpertown
(9 GFLOPS single-core FP64 peak). Any current x86/ARM server-class chip
comfortably exceeds this, so the normalisation ratio is always ≥ 1.0 on
dev + CI workstations in 2025. If the ratio falls below 1.0 on a given
machine (i.e. the current machine is *slower* than 2007 Harpertown), the
harness emits a hard failure with "hardware mismatch" rather than running
— the comparison is only defined on equal-or-faster hardware.

## Failure modes + escalation path

When T3 fires RED, the M5 retrospective expects one of these classifications
in the incident report:

1. **`REF_DATA_STALE`** — the reference CSV is still the T5.10 placeholder
   (see `dissertation_reference_data.csv` header). Closed by T6.0
   (2026-04-19) — real points extracted from `docs/_sources/fig_29.png`
   and `fig_30.png` and live in the CSV now. If the placeholder detector
   trips on a future regression, re-extract from the same scans or verify
   the shipped CSV header was not reverted.

2. **`HARDWARE_NORMALIZATION_OFF`** — the absolute-performance gate trips
   but efficiency is clean. Fix: replace the Python proxy in
   `hardware_normalization.py` with a native TDMD micro-benchmark
   (planned T5.11 refinement). Usually yields YELLOW, not RED.

3. **`COMM_BACKEND_OVERHEAD`** — efficiency tanks linearly with N_procs.
   Fix: profile `RingBackend::progress()` + `MpiHostStagingBackend` with
   VTune / `mpiP`; inspect packet-size distribution. M5 scope explicitly
   did not ship GPU-aware MPI; revisit at M6/M7.

4. **`DETERMINISM_BREAK`** — bit-exact K=1 P=1 regression fails
   (`test_k1_regression_smoke` green but anchor run diverges). Fix: scope
   for a missing reduction-order guarantee — likely a Kahan ring traversal
   that isn't canonical under some rank count. Master spec §13.5 matrix
   is canonical truth.

5. **`LJ_MORSE_PROXY_MISMATCH`** — LAMMPS parity fails at step 0. Fix: the
   Morse proxy's force magnitude on Al FCC is drifting more than the
   10⁻¹⁰ relative budget allows — expected only once native LJ lands
   post-M5. Flag, do not block.

6. **`NVE_DRIFT_REGRESSION`** — integrator / neighbor-list bug. Fix: bisect
   against the M4 smoke golden; if M4 smoke green and T3 red, the bug is
   in the comm → integrator ordering. Scheduler determinism engineer
   owns.

7. **`RUNTIME_BUDGET_BLOWOUT`** — the 60-minute budget trips but
   everything else is green. Fix: profile; if blowout is during
   `regen_setup.sh`, ship a pre-generated `setup.data.xz` (current T5.10
   decision deferred this).

## Ship criteria for T5.10

This document is fixture-only — T5.10 passes when:

- [x] All seven fixture files (README, config.yaml, lammps_script.in,
      dissertation_reference_data.csv, hardware_normalization.py,
      checks.yaml, acceptance_criteria.md) exist and are well-formed.
- [x] `hardware_normalization.py` runs to completion and emits a numeric
      scalar on stdout (no uncaught exceptions).
- [x] `regen_setup.sh` exists and runs to completion (dry-run: `--help`
      reports usage). Actual LAMMPS regeneration is deferred to T5.11
      harness run-time (no LFS blob in this repo, following T1 precedent).
- [x] `verify/SPEC.md` §4.4 cross-references this directory.

The dissertation-reproduction proofs themselves only become testable at
T5.11; T5.10 lands the scaffolding.
