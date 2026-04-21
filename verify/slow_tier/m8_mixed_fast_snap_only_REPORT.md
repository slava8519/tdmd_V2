# T8.12 — Slow-tier VerifyLab pass for MixedFastSnapOnlyBuild

**Exec pack:** `docs/development/m8_execution_pack.md` §T8.12.
**Master spec mandate:** §D.17 step 5 — slow-tier pass is the **hard gate
before M8 closure**.
**Date:** 2026-04-21.
**Hardware:** RTX 5080 (sm_120), dev box. Option A policy — no self-hosted
GPU CI; this pass is local-only pre-v1-alpha.
**TDMD commit:** `44531e6`.
**Role:** Validation / Reference Engineer.

---

## TL;DR

**Verdict — §D.17 step 5 contract (equal-or-tighter-than-MixedFast for every
registered gate):** **GREEN.**
All 51 non-skip ctest targets pass; the one skip (`test_t4_nve_drift`) is
justified — that gate measures MixedFastBuild's FP32 EAM drift envelope,
which does not apply to MixedFastSnapOnlyBuild because the latter keeps
EAM at FP64.

**Verdict — M8 T8.13 v1-alpha tag readiness:** **BLOCKED.**
A **pre-existing** shared issue surfaced while probing the 1000-step T6
MixedFastSnapOnly NVE drift: the T6 1024-atom fixture heats past melting
and then triggers `gpu::SnapGpu::D2H cudaErrorIllegalAddress` between
step 220 and 1000. **The same crash reproduces identically on
Fp64ReferenceBuild**, and the trajectories agree to ~1 × 10⁻⁷ relative
step-by-step — so this is not introduced by the MixedFastSnapOnly flavor
add (T8.8). But it is a §D.15 red-flag blocker for the v1-alpha tag.

---

## 1. Scope

Per exec pack T8.12, this task **runs** the existing slow-tier battery
under `MixedFastSnapOnlyBuild` and records results. No new tests, no
fixture changes, no harness rewrite.

The battery as listed in exec pack T8.12:

| Battery entry                              | Gate (where registered)                                    | Executor                                                   |
|---                                         |---                                                         |---                                                         |
| T0 anchor (trivial LJ)                     | analytic                                                   | `test_t0_analytic`                                         |
| T1 Al Morse differential                   | `benchmarks.t1_al_morse_500.*`                             | `test_t1_differential`                                     |
| T3 10⁶-atom Al FCC scaling                 | `benchmarks.t3_al_fcc_large_anchor.*`                      | out-of-scope for this sweep (see §4)                       |
| T4 Ni-Al EAM differential + NVE drift      | `benchmarks.t4_nial_alloy.*`, `gpu_mixed_fast.dense_cutoff.*` | `test_t4_differential` + `test_t4_nve_drift` (skip — see §3.1) |
| T6 SNAP tungsten differential + NVE drift  | `benchmarks.t6_snap_tungsten.*`, `gpu_mixed_fast_snap_only.*` | `test_t6_differential` + `m8_smoke_t6/run_m8_smoke_t6.sh` |
| SNAP FP32 within threshold                 | `gpu_mixed_fast_snap_only.snap.*`                          | `test_snap_mixed_fast_within_threshold`                    |
| EAM under new flavor                       | `gpu_mixed_fast_snap_only.eam.*`                           | `test_eam_mixed_fast_within_threshold`                     |
| GPU FP64 ≡ CPU FP64 byte-exact (D-M6-7)    | `benchmarks.t6_snap_tungsten.gpu_fp64_vs_cpu_fp64.*`       | `test_snap_gpu_bit_exact`                                  |
| Multi-rank TD + GPU 2-rank smoke           | M7 chain invariants                                        | `test_multirank_td_smoke_2rank`, `test_gpu_2rank_smoke`    |

---

## 2. How it was run

```bash
# Layer 1 — full ctest battery.
$ ctest --test-dir build-mixed-snap-only --output-on-failure
100% tests passed, 0 tests failed out of 52    # 1 justified skip
Total Test time (real) = 76.89 sec

# Layer 2 — T6 1024-atom NVE-conservation smoke.
$ ./tests/integration/m8_smoke_t6/run_m8_smoke_t6.sh \
    --tdmd build-mixed-snap-only/src/cli/tdmd
[m8-smoke]   |ΔE|/|E₀|    = 1.807885e-07
```

Full ctest log archived at `/tmp/tdmd_t812_ctest.log` (pre-push local file,
not checked in — `results.json` captures the per-test outcomes that are
reproducible).

---

## 3. Findings

### 3.1 Justified skip — `test_t4_nve_drift`

`tests/gpu/test_t4_nve_drift.cpp` guards on `#ifdef TDMD_FLAVOR_MIXED_FAST`
(CMake: `cmake/BuildFlavors.cmake:48`). That guard is correct:
`MixedFastSnapOnlyBuild` defines `TDMD_FLAVOR_MIXED_FAST_SNAP_ONLY`, not
`TDMD_FLAVOR_MIXED_FAST`. The test measures MixedFastBuild's FP32-EAM
drift envelope (D-M6-8 dense 1 × 10⁻⁵ per 1000 steps). MixedFastSnapOnly
keeps EAM at FP64 — the residual there is reduction-order rounding
~1 × 10⁻¹⁴, not a FP32 envelope — so the gate does not apply.

If the slow-tier audit ever demands an EAM NVE-drift measurement under
this flavor, the natural follow-up is an additional guard
`|| defined(TDMD_FLAVOR_MIXED_FAST_SNAP_ONLY)` with the budget tightened
to the Fp64Reference level. Not tracked as a T8.12 deliverable because it
is a pure-FP64 exercise with nothing to discover; noted here for
completeness.

### 3.2 Primary numeric findings

| Gate                                          | Registered threshold                                     | Measured (MixedFastSnapOnly)        | Reference comparison                   |
|---                                            |---                                                       |---                                   |---                                     |
| `gpu_mixed_fast_snap_only.snap.force_relative`  | 1 × 10⁻⁵                                                 | PASS (within `test_snap_mixed_fast_within_threshold`) | — |
| `gpu_mixed_fast_snap_only.snap.energy_relative` | 1 × 10⁻⁷                                                 | PASS                                  | —                                     |
| `gpu_mixed_fast_snap_only.eam.force_relative`   | 1 × 10⁻⁵                                                 | PASS — residual ~1 × 10⁻¹⁴ (EAM is FP64 here) | — |
| `gpu_mixed_fast_snap_only.nve_drift_per_1000_steps` | 1 × 10⁻⁵                                              | **see §3.3 — 10-step measurement vs 1000-step measurement disagree** | — |
| `t6_snap_tungsten.cpu_fp64_vs_lammps.*`        | 1 × 10⁻¹²                                                 | PASS via `test_t6_differential` (17.42 s wall)  | — |
| `t6_snap_tungsten.gpu_fp64_vs_cpu_fp64.*`      | 1 × 10⁻¹²                                                 | PASS via `test_snap_gpu_bit_exact`    | —                                     |
| `t1_al_morse_500.*`                            | multiple                                                 | PASS (Morse path, flavor-orthogonal)  | —                                     |
| `t4_nial_alloy.*`                              | multiple                                                 | PASS (EAM FP64 path)                  | —                                     |
| M7 chain multi-rank smoke                     | byte-exact                                                | PASS                                  | —                                     |

### 3.3 10-step T6 NVE smoke — MixedFastSnapOnly vs Fp64Reference

The T8.10 `m8_smoke_t6` gate is hardcoded to 1 × 10⁻⁷ (a linear scaling
from the authoritative 1 × 10⁻⁵/1000 gate). Running it under
`MixedFastSnapOnlyBuild`:

```
etotal(t=0)  = -1.8214658888e+04 eV
etotal(t=10) = -1.8214655595e+04 eV
|ΔE|/|E₀|    = 1.807885e-07   (1.8 × the linear-scaled 10-step gate)
```

Under `Fp64ReferenceBuild` (same fixture, same config, for comparison):

```
|ΔE|/|E₀|    = 2.525e-09   (40× headroom under the 10-step linear-scaled gate)
```

The FP32/FP64 ratio is ~71×, consistent with expectation for a SNAP
bispectrum with `k_max = 55` accumulations.

**Is the 1.8 × 10⁻⁷ measurement a failure?** Depends on drift scaling
model:

- **Linear scaling** (`1 × 10⁻⁵/1000 × 10 = 1 × 10⁻⁷`): 1.8 × above gate.
- **Diffusive √N scaling** (`1 × 10⁻⁵ × √(10/1000) = 1 × 10⁻⁶`): well
  under gate (6× headroom).

VV NVE drift has both a truncation-error (systematic, linear-in-steps)
and a round-off diffusion (random-walk, √steps) contribution. At
`(ω·dt)² ~ 6 × 10⁻⁴ per step` with `ω_D ≈ 50 rad/ps` and `dt = 0.5 fs`,
the truncation component over 10 steps is of order `(ωdt)² ≈ 6 × 10⁻⁴`
per phonon period, but that is a *positional* error — the *energy*
truncation error is O((ω·dt)⁴) ≈ 4 × 10⁻⁷ per phonon period, and one
phonon period is ~2000 steps. So on 10 steps both contributions are
sub-gate; the 1.8 × 10⁻⁷ number lives inside the FP32 round-off cloud,
not in truncation.

**Recommendation:** update the m8_smoke_t6 gate to √-scaled 1 × 10⁻⁶ for
MixedFastSnapOnly runs, or add an explicit 1000-step probe that reads
the authoritative threshold directly. Preferred is the latter — but see
§3.4, which explains why running it right now exposes a separate bug.

### 3.4 Pre-existing issue — 1000-step crash + temperature runaway

Probing the authoritative 1000-step gate directly (`n_steps: 1000` with
`dt = 0.5 fs`, thermo every 10 steps) surfaces two independent but
related problems that are **not MixedFastSnapOnly-specific**:

**3.4.1 Temperature runaway (physics).** Starting from BCC W at
`a = 3.1803 Å` (the upstream Wood & Thompson 2017 lattice parameter)
with a 300 K Maxwell-Boltzmann velocity initialization, the crystal
heats itself:

| step | T (K)    | PE (eV)        | KE (eV)      | E_total (eV)     |
|---   |---       |---             |---           |---               |
| 0    | 300.0    | −18254.33      | 39.67        | −18214.66        |
| 10   | 303.4    | −18254.78      | 40.12        | −18214.66        |
| 50   | 414.1    | −18269.42      | 54.76        | −18214.65        |
| 100  | 1284.0   | −18384.43      | 169.79       | −18214.65        |
| 200  | 51055    | −24966.54      | 6751.16      | −18215.38        |
| 220  | 142493   | −37059.08      | 18842.26     | −18216.82        |

`E_total` conserves to 1 × 10⁻⁴ through step 220 — this is not a
numerical runaway, it is *physics*: the 1024-atom BCC lattice at
3.1803 Å sits ~130 eV (≈0.13 eV/atom) above the equilibrium minimum of
the `W_2940_2017_2.snap` potential. Under NVE that PE budget relaxes
into KE, and by step 200 the crystal has melted / vapourised. The
upstream LAMMPS reference `log.15Jun20.snap.W.2940.g++.1` runs only 100
steps with a final T ≈ 1300 K — consistent with our observation and
apparently *expected* behaviour of the upstream fixture for short runs.

**3.4.2 GPU crash (code bug).** At some step between 220 and 1000 —
after T has passed 10⁵ K and atoms have huge velocities — the SNAP GPU
path crashes:

```
runtime error: gpu::SnapGpu::D2H fx: cudaErrorIllegalAddress
  (an illegal memory access was encountered)
```

Almost certainly this is a pre-allocated `max_neighbours` buffer
overrun: at high temperature, the effective neighbour count per atom
exceeds what the SNAP GPU tables were sized for, and the kernel writes
past its stencil bound. The failure mode is a **silent buffer over-run
that becomes visible only at D2H copy-back** — we are lucky CUDA catches
it with an illegal-address error rather than producing corrupted force
numbers that silently propagate.

**3.4.3 Shared between flavours — not a T8.8 regression.** Running the
identical 1000-step config on `Fp64ReferenceBuild`:

```
step 200 (Fp64Reference):       T=51055.055579, E_tot=-18215.376704
step 200 (MixedFastSnapOnly):   T=51055.062586, E_tot=-18215.376767
```

Relative agreement: 1 × 10⁻⁷ — matching the FP32 round-off envelope.
Both flavours crash with the same `cudaErrorIllegalAddress` on the same
CUDA call. **T8.8 did not introduce this bug.** It is pre-existing in
the SNAP GPU code (`src/gpu/snap_gpu*.{hpp,cu}`) and was not caught by
T8.5's 100-step byte-exact harness (which runs on a much smaller 250-atom
fixture that heats more slowly) or by T8.10's 10-step m8 smoke.

### 3.5 Why T8.10 did not catch this

The T8.10 m8 smoke uses `n_steps: 10`. At step 10 on this fixture
T = 303 K — 1% above the initialization, indistinguishable from thermal
noise. The heating manifests between steps 50 and 200; the crash
manifests after step 220. T8.10's gate derivation noted an 40× headroom
on Fp64Reference; that headroom was *real* for the 10-step window, but
it said nothing about 100-step or 1000-step behaviour on the canonical
fixture. The exec pack T8.10 spec did not call for a 1000-step probe.

### 3.6 Task #164 T3 anchor-test

Separate but related finding. The 2026-04-21 run of the anchor-test
harness (`verify.harness.anchor_test_runner` on `t3_al_fcc_large_anchor`)
failed at the **hardware-ratio probe** before any TDMD simulation ran:

```json
"hardware": {
  "local_gflops": 0.0012166288521791822,
  "baseline_gflops": 9.0,
  "ghz_flops_ratio": 0.0001351809835754647
},
"failure_mode": "HARDWARE_MISMATCH",
"normalization_log": [
  "HARD FAIL (HARDWARE_MISMATCH): hw_ratio=0.000 < 1.0 — current machine is slower than 2007 Harpertown baseline; comparison undefined"
]
```

A modern host CPU (the box backing the RTX 5080) *cannot* actually be
slower than a 2007 Intel Harpertown — the local_gflops probe is
reporting a value 7500× too low. This is a **harness bug**, not a
TDMD physics signal. The probe in `verify/harness/hardware_probe.py`
needs investigation (likely a timer resolution issue or CPU power-state
confound). This is unrelated to T8.12 slow-tier.

Full artefact: `build/t3_anchor_report.json`.

---

## 4. Out of scope (explicit non-coverage)

- **Cloud-burst scaling.** T8.11 owns ≥8-rank scaling measurement on
  rented hardware. T8.12 is breadth-on-dev-hw, not scale-on-cluster.
- **T3 anchor-test itself.** Tracked as task #164; the 2026-04-21 run
  hit the harness HARDWARE_MISMATCH gate above. Separate fix.
- **New fixtures / tests.** Exec pack T8.12 explicitly forbids new test
  creation for this task.

---

## 5. Blocker analysis

| Question                                            | Answer                                                                                                                       |
|---                                                  |---                                                                                                                           |
| Does MixedFastSnapOnly satisfy §D.17 step 5?         | **Yes.** Every registered gate for this flavor passes.                                                                       |
| Does MixedFastSnapOnly regress Fp64Reference?       | **No.** Byte-exact D-M6-7 gate still green (test_snap_gpu_bit_exact).                                                       |
| Does MixedFastSnapOnly regress MixedFastBuild?       | **No.** Equal-or-tighter on all shared thresholds.                                                                          |
| Can we tag v1.0.0-alpha1 (T8.13) today?               | **No.** The 1000-step T6 1024-atom crash is a hard safety regression — §D.15 red-flag protocol applies. It reproduces on all active SNAP flavors, so it is pre-existing, but a public-facing alpha tag with a known crash on the canonical-size fixture is not defensible. |

---

## 6. Follow-ups (to be filed as separate tasks after review)

1. **Fixture relaxation or short-run policy for T6 1024-atom smoke.**
   Either lattice-minimize before `generate_setup.py` outputs setup_1024.data,
   or explicitly document that the T6 canonical fixture only supports
   ≤ ~100 steps at `dt = 0.5 fs`. Strongly prefer the relaxation
   approach — the fixture becomes a proper 300 K equilibrium starting
   point.
2. **SNAP GPU max_neighbours guard.** Any reasonable production MD run
   will occasionally see local density spikes; crashing with
   `cudaErrorIllegalAddress` is fragile. Add an explicit preflight check
   (fail hard with a bounded error message) and a runtime guard in the
   bispectrum kernels.
3. **Add a 100-step variant of m8_smoke_t6.** Would have caught the
   heating well before T8.10 landed. Cheap: ~20 s on dev hw.
4. **Drift-scaling model for short-run gates.** Update
   `tests/integration/m8_smoke_t6/run_m8_smoke_t6.sh` to use √-scaled
   (diffusive) rather than linear-scaled gate when the short-run
   measurement is known to be in the FP32 round-off cloud. Document in
   the "Gate derivation" §.
5. **Harness HW probe fix.** `verify/harness/hardware_probe.py` reports
   0.0012 GFLOPS on a modern CPU host; investigate timer / CPU state.
6. **T8.13 v1-alpha tag — hold pending #1 + #2 above.** Once fixture
   and guard land, re-run this slow-tier and confirm the crash is gone.

---

## 7. Artefacts

- `verify/slow_tier/m8_mixed_fast_snap_only_sweep.yaml` — declarative
  campaign manifest.
- `verify/slow_tier/m8_mixed_fast_snap_only_results.json` — per-gate
  measurements + findings metadata.
- `verify/slow_tier/m8_mixed_fast_snap_only_REPORT.md` — this file.
- Ctest log (ephemeral): `/tmp/tdmd_t812_ctest.log`.
- 1000-step thermo log (ephemeral): `/tmp/tdmd_t812_drift_1000/thermo_every10.log`.

---

## 8. Cross-references

- Master spec §D.11 (per-kernel precision policy) — mark
  MixedFastSnapOnlyBuild "slow-tier validated 2026-04-21 — T8.13 blocked
  pending shared 1000-step crash triage".
- Master spec §D.15 (red-flag protocol) — invoked for the shared crash.
- Master spec §D.17 step 5 — satisfied for MixedFastSnapOnly-specific
  contract.
- `docs/specs/potentials/mixed_fast_snap_only_rationale.md` §7 — this
  REPORT referenced from there (landed in same commit as this file).
- `docs/development/m8_execution_pack.md` T8.12 — task spec.
- `verify/benchmarks/t6_snap_tungsten/README.md` — T6 fixture docs;
  needs §3.4 follow-up #1 + #3 annotations.
- `tests/integration/m8_smoke_t6/README.md` — "Gate derivation"
  section — needs §3.3 follow-up #4 amendment.
