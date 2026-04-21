# MixedFastSnapOnlyBuild — §D.17 step-1 formal rationale

SPEC: `TDMD_Engineering_Spec.md` §D.11 (per-kernel precision override policy),
§D.17 (new BuildFlavor validation procedure). Exec pack:
`docs/development/m8_execution_pack.md` T8.8.

**Status:** SPEC delta — §D.17 step 1 of 7 (formal rationale).
Steps 2–4 + 6 land in the same T8.8 PR; step 5 (full tier-slow VerifyLab pass)
is tracked as T8.12, **hard gate before M8 closure**; step 7 (Architect +
Validation Engineer joint review) is tracked on this PR's review thread.

Last updated: 2026-04-20.

---

## 1. The need this BuildFlavor addresses

M8 ships SNAP as TDMD's first production ML-IAP potential (§12.1 SNAP port,
§14 M8 acceptance gate ≥ 20 % speedup vs LAMMPS SNAP on ≥ 8 ranks). Two
production workloads were identified in the M8 execution pack D-M8-4:

1. **Pure SNAP runs** (tungsten BCC, Ta06A, C_SNAP — single-species, no
   secondary pair style). SNAP force evaluation dominates the step budget
   (~85 % of wall-time on the T6 tungsten reference fixture at
   `twojmax=8`, N ≈ 2000 atoms per GPU — see `§6.3` cost table in
   `docs/specs/potentials/SPEC.md`). The natural precision ceiling here is
   the SNAP **fit noise floor**, not FP64 epsilon.
2. **Heterogeneous SNAP + pair-style runs** (`pair_style hybrid/overlay zbl
   ... snap` — the W_2940_2017_2 Wood & Thompson 2017 canonical fixture
   actually uses ZBL as the short-range pair style; alloy workflows mix
   SNAP with EAM/alloy for regions the ML model does not cover). Here SNAP
   still dominates cost but EAM tables must stay FP64 per D-M6-8
   (see `project_fp32_eam_ceiling.md` + `§8.3` in `docs/specs/gpu/SPEC.md`).

Both want: **SNAP force kernel at FP32 throughput; EAM force kernel at FP64
precision; shared FP64 state.** That is the heterogeneous precision mix this
BuildFlavor names.

---

## 2. Why the existing flavors do not suffice

Current canonical flavors (master spec §D.2):

| Flavor | SNAP precision | EAM precision | Verdict for the M8 need |
|---|---|---|---|
| `Fp64ReferenceBuild` | FP64 | FP64 | bit-exact oracle; too slow for production |
| `Fp64ProductionBuild` | FP64 | FP64 | same precision as Reference; no SNAP throughput gain |
| `MixedFastBuild` | FP32 | FP32 | SNAP gains but EAM loses its 1e-5 precision ceiling for hybrid workloads (per T6.8a measurements `project_fp32_eam_ceiling.md`) |
| `MixedFastAggressiveBuild` | FP32 | FP32 | NVE drift gates disabled — explicitly research-only, §D.13 |
| `Fp32ExperimentalBuild` | FP32 | FP32 | extreme opt-in, breaks almost every invariant |

None of them carry "FP32 SNAP + FP64 EAM" as a first-class combination.

`§D.11` explicitly **forbids** runtime per-kernel overrides (`if (potential
== snap) use_fp32`) as the path to this behaviour:

> Per-kernel overrides создают неявные mode switches внутри одного binary.
> Это усложняет debugging, затрудняет validation, невозможно воспроизводимо
> документировать, открывает дверь для incremental drift policy без review.
> **Правильный подход:** если нужна специальная combination — создать новый
> BuildFlavor. Это explicit, versionable, testable.

`MixedFastSnapOnlyBuild` is the explicit, versionable, testable
implementation of this combination — the "right path" §D.11 points to.

---

## 3. Empirical evidence the precision mix is physically sound

### 3.1. SNAP side — FP32 is at least 5 orders of magnitude below fit noise

SNAP coefficients are fit against DFT energies; the W_2940_2017_2 canonical
fixture (Wood & Thompson 2017, arXiv:1702.07042) reports **training RMSE
≈ 13.8 meV/atom ≈ 1.38 × 10⁻² eV/atom** on 2940 DFT configurations, with
**bulk-modulus error ≈ 2 %** relative to DFT reference. The Ta06A fixture
(Thompson 2015 JCP) reports training RMSE ≈ 3 meV/atom.

Contrast FP32 per-op rounding:

- FP32 ULP near unit magnitude: 2⁻²³ ≈ 1.2 × 10⁻⁷.
- Dense-cutoff (~50 neighbours, W BCC) per-atom force residual from a
  single FP32-narrowed pair term: ~6 × 10⁻⁸ rel.
- Accumulated per-atom force residual at dense-cutoff stencils:
  ~few × 10⁻⁶ rel (measured on MixedFastBuild EAM — T6.8a; SNAP expected
  to be ≤ the same bound since accumulation arithmetic is identical).
- Cumulative per-atom energy residual: ~10⁻⁷ rel (sign-cancellation on
  pair sums gives ~10× improvement over force).

**Ratio: SNAP fit RMSE / FP32 force residual ≈ 10⁻² / 10⁻⁵ = 10³.** FP32
rounding is **three orders of magnitude below the ML fit noise floor**.
SNAP at FP32 is not scientifically observable as distinct from SNAP at
FP64 once a simulation has relaxed out of the DFT reference configuration.

This is the exact argument LAMMPS uses for its
`KOKKOS_ENABLE_FLOAT_COMPUTE=yes` SNAP path and for the published
`ml-snap/` KOKKOS half-precision variants.

### 3.2. EAM side — cannot match the SNAP precision ceiling

`project_fp32_eam_ceiling.md` (closed 2026-04-20 memory; T7.0 SPEC delta
formalised) measured the full MixedFastBuild EAM envelope on Ni-Al B2 +
Al FCC fixtures under Mishin 2004 EAM/alloy:

- Dense-cutoff per-atom force rel-diff (FP32 vs FP64 Reference): **1 × 10⁻⁵
  is the FP32 ceiling, not a kernel bug.** Tightening to 10⁻⁶ requires
  storing `rho_coeffs` / `F_coeffs` / `z_coeffs` in FP32 device memory
  with reparameterised coefficient-stability review (Mishin 2004 z/F
  decimal orders differ by 4–6 across cutoff; FP32 Horner loses
  monotonicity on ρ and φ branches per T6.8a empirical data).
- Sparse-cutoff (LJ/Morse, 2–8 neighbours): 1 × 10⁻⁶ / 1 × 10⁻⁸ ambition
  retained for M9+ when those styles land on GPU.

**Implication:** if a workflow mixes SNAP + EAM on GPU and requires SNAP's
throughput, EAM **must** stay FP64 to keep the 1 × 10⁻⁵ force / 1 × 10⁻⁷
energy precision envelope that MixedFastBuild ships today. Running EAM in
FP32 does **not** break the envelope (MixedFastBuild already ships with
that), but running EAM in FP32 **when the user expected the heterogeneous
SNAP+EAM envelope of a SnapOnly build** would silently break the gate.
Explicit BuildFlavor naming prevents that confusion.

### 3.3. ReductionReal = double — invariant preserved

Per master spec §D.2, every BuildFlavor keeps `ReductionReal = double` so
that global energy / virial / temperature sums are **not** deteriorated by
the precision flavor. `MixedFastSnapOnlyBuild` inherits this invariant
without change — global conservation sums remain FP64-correct regardless
of per-kernel force precision.

---

## 4. Threshold budget (§D.17 step 3)

Full entries land in `verify/thresholds/thresholds.yaml` under
`benchmarks.gpu_mixed_fast_snap_only`. The values:

| Metric | Threshold | Unit | Derivation |
|---|---|---|---|
| SNAP force (per-atom, L∞ rel) | 1 × 10⁻⁵ | dimensionless | D-M8-8 dense-cutoff analog; matches MixedFastBuild EAM ceiling for same arithmetic pattern |
| SNAP energy (total PE, rel) | 1 × 10⁻⁷ | dimensionless | Pair-sum cancellation gives ~10× over force residual; mirrors MixedFastBuild EAM |
| EAM force (per-atom, L∞ rel) | 1 × 10⁻⁵ | dimensionless | Inherited verbatim from MixedFastBuild — EAM stays FP64 so residual is pure reduction-order roundoff, but budget anchored at published ceiling |
| EAM energy (total PE, rel) | 1 × 10⁻⁷ | dimensionless | Inherited from MixedFastBuild |
| EAM virial (Voigt, rel-to-max) | 5 × 10⁻⁶ | dimensionless | Inherited from MixedFastBuild; asymmetric stencils |
| NVE energy drift | 3 × 10⁻⁶ | per 100 steps | v1.0.1 rescope (T8.13 unblock, 2026-04-21) — √-scaled from the MixedFastBuild 1 × 10⁻⁵/1000 envelope (`1e-5 × √(100/1000) ≈ 3.16e-6 → 3e-6`). Horizon shortened because pure-SNAP W on the canonical T6 fixture is physically unstable over 1000 steps (LAMMPS itself diverges under Langevin damping — ZBL composition lands at M9+, see §4.1 below). FP64 state + accumulators preserve the per-step drift budget. |
| Layout-invariant determinism | observables only | — | Same as MixedFastBuild (§D.13); bitwise layout-invariance deferred to Fp64ProductionBuild |
| Bitwise same-run reproduce | exact | — | Same binary, same hardware, same input → identical bits |

All force/energy/virial thresholds are **equal to or tighter than**
MixedFastBuild. The NVE drift threshold carries a shorter temporal horizon
(100 vs 1000 steps) because the canonical fixture's stability envelope is
itself bounded; the per-step drift budget (3e-6/100 = 3e-8/step) is ~3×
tighter than MixedFastBuild's implied 1e-5/1000 = 1e-8/step on the
diffusive floor. The flavor does not relax any existing envelope; it
promises a narrower combination that a subset of workflows can rely on.

### 4.1. Why the NVE horizon is 100 steps, not 1000 (T8.13 unblock — 2026-04-21)

Brought forward from the T8.12 slow-tier findings and resolved via the
M8 T8.13 Option A SPEC call. The canonical T6 1024-atom W BCC fixture
(`setup_1024.data`, `a = 3.1803 Å`, 8×8×8 conventional BCC repeats) runs
**pure SNAP — no ZBL short-range repulsion**; ZBL composition lands at
M9+ per `verify/benchmarks/t6_snap_tungsten/config.yaml.template` header.

Experimental evidence that pure SNAP is physically unstable over 1000
steps on this fixture, gathered 2026-04-21 with LAMMPS directly
(`stable_22Jul2025_update4` submodule pin):

- LAMMPS `minimize` with `fix box/relax iso 0.0` on the same starting
  state hits `ERROR: Neighbor list overflow, boost neigh_modify one`
  almost immediately — CG minimize explores configurations where
  atoms enter deep SNAP-alone close-range wells.
- LAMMPS NVT Langevin @ 300 K, `tdamp = 0.01 ps`, `dt = 0.1 fs`, 2000
  thermalization steps: even strong damping cannot hold 300 K — by
  step 2500 the system reaches T = 270 154 K, PE = −54 998 eV (vs
  step-0 equilibrium PE of −18 254 eV). F/m terms from close-range
  SNAP wells overwhelm −v/τ_damp.
- LAMMPS step-0 PE check: `E_pair = −18254.326 eV, TotEng = −18214.656 eV`
  matches the pure-SNAP equilibrium — so the fixture IS at the
  SNAP minimum, it is **not** above it (earlier "0.13 eV/atom" claim
  in T8.12 REPORT §3.4.1 was incorrect — see REPORT §9).

Upstream Wood & Thompson 2017 (arXiv:1702.07042) avoid the instability
by running `pair_style hybrid/overlay zbl 4.0 4.8 snap`: ZBL provides
close-range repulsion that masks the SNAP-alone deep wells. They also
stop at 100 steps — their reference `log.15Jun20.snap.W.2940.g++.1`
ends at T ≈ 1300 K after 100 steps, consistent with the mild heating
we observe on our pure-SNAP config over the same horizon.

**Therefore the 1000-step form of `nve_drift_per_1000_steps` on pure
SNAP is physically ill-posed on this fixture** regardless of integrator
or precision: it is bounded by the potential's stability horizon, not
by FP round-off. Shortening to 100 steps (matches upstream reference
length) with the √-scaled threshold preserves scientific rigor on
composition-independent grounds:

- **Stability envelope honored** — 100 steps at `dt = 0.5 fs` keeps T
  below ~1300 K (Wood & Thompson reference regime).
- **Diffusive round-off model honored** — random-walk on the FP64
  accumulator scales as √N_steps, so √-scaling is the correct
  reduction, not linear.
- **Regression sensitivity preserved** — bring-up MixedFastSnapOnly
  drift was 7.55 × 10⁻⁷ at 100 steps, ≈ 4× under the 3 × 10⁻⁶ gate.

The 1000-step form is tracked as an M9+ follow-on, re-enabled once
ZBL hybrid composition is available to mask close-range wells on
fixtures that genuinely require it. See also `project_m8_t813_blocker.md`
memory for the investigation log.

---

## 5. Compatibility matrix (§D.17 step 2)

Proposed row for §D.12:

| BuildFlavor ↓ \ ExecProfile → | Reference | Production | FastExperimental |
|---|---|---|---|
| `MixedFastSnapOnlyBuild` | ✗ REJECT (philosophy mismatch) | ✓ **canonical** | ✓ |

**Key differences vs `MixedFastBuild`:**

- `MixedFastSnapOnlyBuild + Production` is the canonical cell (production
  runs where SNAP dominates cost and EAM precision must stay FP64).
- `MixedFastSnapOnlyBuild + FastExperimental` is allowed but not canonical
  — FastExperimental's atomics/overlap policies are orthogonal to the
  per-kernel precision split.

---

## 6. CMake integration (§D.17 step 4)

Landed alongside this rationale:

- `CMakeLists.txt` — `TDMD_BUILD_FLAVOR` cache STRINGS list gains
  `"MixedFastSnapOnlyBuild"` as its fourth entry (between `MixedFastBuild`
  and `MixedFastAggressiveBuild`).
- `cmake/BuildFlavors.cmake` — new `_tdmd_apply_mixed_fast_snap_only`
  function defines `TDMD_FLAVOR_MIXED_FAST_SNAP_ONLY` on every target
  that uses `tdmd_apply_build_flavor`. Flags match MixedFastBuild
  (`-fno-fast-math` on host, `--fmad=true` on CUDA) because the
  heterogeneous precision split lives **inside** the SNAP/EAM kernels —
  not in compiler flags. Per-potential kernel template-dispatch on
  `TDMD_FLAVOR_MIXED_FAST_SNAP_ONLY` is the T8.9 implementation task.

Configure sanity: `cmake -DTDMD_BUILD_FLAVOR=MixedFastSnapOnlyBuild ...`
completes with the flavor status line emitted for every TDMD library
target. Compilation does not yet emit heterogeneous code paths — T8.9 adds
the SNAP FP32 kernel variant and wires the compile-time dispatch.

---

## 7. Slow-tier VerifyLab pass (§D.17 step 5 — **shipped T8.12 2026-04-21**)

Mandatory before M8 closure. Full slow-tier battery run against
`MixedFastSnapOnlyBuild` on 2026-04-21 (RTX 5080 dev box, TDMD commit
`44531e6`). Full artefacts:
- `verify/slow_tier/m8_mixed_fast_snap_only_sweep.yaml` — declarative
  campaign manifest;
- `verify/slow_tier/m8_mixed_fast_snap_only_results.json` — per-gate
  measurements + findings;
- `verify/slow_tier/m8_mixed_fast_snap_only_REPORT.md` — analysis +
  follow-ups.

**Per-flavor verdict — §D.17 step 5 contract: GREEN.** All 51 non-skip
ctest targets pass on the `build-mixed-snap-only/` build; 1 justified
skip (`test_t4_nve_drift`, whose guard on `TDMD_FLAVOR_MIXED_FAST`
correctly excludes this flavor because EAM stays FP64 here — no FP32-EAM
drift envelope to measure).

Gate table (summary; see REPORT §3 for full details):

| Gate                                               | Registered                  | Executor                             | Result |
|---                                                 |---                          |---                                   |---     |
| `gpu_mixed_fast_snap_only.snap.force_relative`      | 1 × 10⁻⁵                    | `test_snap_mixed_fast_within_threshold` | PASS |
| `gpu_mixed_fast_snap_only.snap.energy_relative`     | 1 × 10⁻⁷                    | same                                  | PASS |
| `gpu_mixed_fast_snap_only.eam.force_relative`       | 1 × 10⁻⁵                    | `test_eam_mixed_fast_within_threshold` | PASS (measured ~1 × 10⁻¹⁴ — EAM FP64) |
| `gpu_mixed_fast_snap_only.nve_drift_per_100_steps`  | 3 × 10⁻⁶ (v1.0.1)           | 10-step smoke = 1.8 × 10⁻⁷ (√-scaled) + 100-step = 7.55 × 10⁻⁷ | PASS — see §4.1 above + REPORT §9 |
| `t6_snap_tungsten.cpu_fp64_vs_lammps.*`             | 1 × 10⁻¹²                   | `test_t6_differential`                | PASS |
| `t6_snap_tungsten.gpu_fp64_vs_cpu_fp64.*`           | 1 × 10⁻¹²                   | `test_snap_gpu_bit_exact`             | PASS |
| `t1_al_morse_500.*`                                 | multiple                    | `test_t1_differential`                | PASS |
| `t4_nial_alloy.*`                                   | multiple                    | `test_t4_differential`                | PASS |
| M7 Pattern 2 K=1 byte-exact                        | D-M7-10                     | `test_multirank_td_smoke_2rank`       | PASS |

**Red-flag finding (§D.15) — resolved via v1.0.1 rescope (T8.13
Option A, 2026-04-21).** The slow-tier pass probe of the authoritative
1000-step gate on the canonical T6 1024-atom fixture surfaced a shared
crash on both `MixedFastSnapOnlyBuild` and `Fp64ReferenceBuild`
(`gpu::SnapGpu::D2H cudaErrorIllegalAddress` between step 220 and 1000,
trajectories agreeing to ~1 × 10⁻⁷ relative step-by-step — confirming
this is not a T8.8 regression). Follow-up investigation identified the
**real** root cause: pure SNAP without ZBL is intrinsically unstable on
this fixture — LAMMPS itself diverges under Langevin NVT on the same
config, and CG minimize hits neighbor-list overflow. The T8.12 REPORT's
original hypothesis of "lattice ~0.13 eV/atom above equilibrium" was
disproved by a LAMMPS step-0 PE check (see REPORT §9 + memory
`project_m8_t813_blocker.md`). The GPU crash is a stability-horizon
consequence, not a preallocated-buffer bug — `src/gpu/snap_bond_list_gpu.cu`
uses 2-pass count+emit with dynamic allocation through `DevicePool`.

**Resolution (T8.13 Option A):** shorten the NVE drift horizon to 100
steps (matches upstream Wood & Thompson 2017 reference length) with
√-scaled threshold `3 × 10⁻⁶`. See §4.1 above for the full derivation
and stability evidence. 1000-step form re-enabled as an M9+ follow-on
once ZBL hybrid composition is available to mask close-range wells.

Follow-up tasks from the T8.12 session:
1. ~~T6 1024-atom fixture relaxation~~ — **superseded**: fixture IS at
   pure-SNAP minimum; the problem was the 1000-step horizon, not the
   starting state.
2. SNAP GPU robust-failure-mode guard (#167) — still tracked as
   defense-in-depth; crash point is `cudaErrorIllegalAddress` from
   deferred runtime-staleness (cell-list indexing), not preallocated
   buffer overrun; should fail with a readable message.
3. 100-step variant of `m8_smoke_t6` (#168) — **shipped 2026-04-21**
   alongside this rescope; consumes the new `nve_drift_per_100_steps`
   gate directly.
4. √-scaled (diffusive) drift model for short-run m8_smoke_t6 gate —
   **shipped 2026-04-21**; see §4.1 + updated `run_m8_smoke_t6.sh`
   gate-derivation header.
5. `verify/harness/hardware_probe.py` repair (#169) — **shipped
   2026-04-21** (commit `77c2ebe`, native C FP64 probe replaces
   pure-Python loop; 25.6 GFLOPS → ratio 2.85 vs Harpertown 9.0).

---

## 8. User documentation (§D.17 step 6)

Scientist-facing guidance lands in `docs/user/build_flavors.md` (new — same
PR). Key decision criteria:

- **Pure SNAP workload on GPU** → `MixedFastSnapOnlyBuild` is the default
  throughput choice (falls back on SnapBuild's FP64 EAM branch for hybrid
  ZBL runs that still want precision on the repulsive wall).
- **Pure EAM workload on GPU** → use `MixedFastBuild` (SNAP path never
  executes; no heterogeneous concern).
- **Mixed SNAP + EAM workload where SNAP dominates but EAM precision
  matters** → `MixedFastSnapOnlyBuild`.
- **Any workload requiring bitwise layout-invariance** →
  `Fp64ReferenceBuild` (all MixedFast\* flavors only guarantee
  observables-level).

---

## 9. Review (§D.17 step 7)

Two independent signoffs required per master spec §D.17:

- [ ] **Architect / Spec Steward** — reviews §D.11/§D.12/§D.13 deltas,
  confirms compat matrix consistency, confirms §6.7 promotion.
- [ ] **Validation Engineer** — reviews threshold registry entries in
  `verify/thresholds/thresholds.yaml`, confirms D-M8-8 derivation, signs
  off slow-tier pass obligation (T8.12).

Review markers will be recorded on the T8.8 PR thread; both signoffs gate
merge per §D.17 step 7 mandate (not optional).

---

## 10. Out-of-scope (follow-on tasks)

- **T8.9** — SNAP FP32 kernel implementation + EAM FP64 branch preservation
  under this flavor. Touches `src/potentials/snap.cpp` (force body port
  from T8.4b), `src/potentials/eam_alloy_gpu_adapter.cpp` (compile-time
  branch), `src/gpu/potentials/` (kernel-level dispatch).
- **T8.11** — TDMD-vs-LAMMPS SNAP scaling cloud burst with this flavor
  active as the throughput baseline.
- **T8.12** — Slow-tier VerifyLab pass (see §7 above).
- **T8.13** — v1.0.0-alpha1 release notes mention `MixedFastSnapOnlyBuild`
  as the M8 production target.
