# TDMD changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/). Versioning is
[semantic](https://semver.org/) with a twist:

- `v1.0.0-alpha` — M8 complete: user-facing, not feature-complete.
- `v1.0.0-beta`  — M11 complete: NVT/NPT under TD research window shipped.
- `v1.0.0`       — M13 complete: production v1.

## [v1.0.0-alpha1] — 2026-04-22

**First public alpha.** Closes milestones **M1 through M8** per the master
spec §14 acceptance gates. User surface (`tdmd run`, `tdmd validate`,
`tdmd explain --perf`) is stable; core physics (Morse, EAM alloy, EAM
Finnis-Sinclair, SNAP) is byte-exact against LAMMPS on the full canonical
benchmark set (T0–T7); the deterministic TD scheduler reproduces Andreev's
1997 / 2007 dissertation result on T3 Al-FCC 10⁶ atoms within master-spec
§13.3 tolerance (anchor-test green). Not yet feature-complete: NVT / NPT
under TD are a research window (M11+); MEAM / PACE / MLIAP arrive at M10 /
M12.

### Acceptance gates met (master spec §14)

Each row below is the **artifact-gate criterion** from the master spec,
marked with the test that validates it:

| Milestone | Scope                                            | Gate criterion (master spec §14)                                                                  | Validator                                                                |
|-----------|--------------------------------------------------|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| **M1**    | CPU reference MD without TD (Morse NVE)          | T1 Al-FCC-Morse differential vs LAMMPS green; `tdmd validate` preflight correct                    | `tests/potentials/test_t1_differential.cpp`                              |
| **M2**    | EAM CPU + perf-model skeleton + `lj` units       | T4 Ni-Al EAM/alloy differential green in `metal`; T1 green in both `metal` + `lj`; `explain --perf` ranks SD>TD | `tests/potentials/test_t4_differential.cpp`, `tests/perfmodel/`         |
| **M3**    | Zoning planner + TD-ready neighbor               | `plan().n_min_per_rank` matches dissertation analytic for all three zoning schemes                 | `tests/zoning/test_n_min_analytic.cpp`                                   |
| **M4**    | Deterministic TD scheduler (single-node CPU)     | T1, T2 green in TD mode K=1; bitwise determinism tests pass                                        | `tests/integration/m4_smoke/run_m4_td_smoke.sh`                          |
| **M5**    | Multi-rank TD on CPU (MPI) + K-batching          | **Anchor-test §13.3 green** — first reproduction of Andreev dissertation                           | `tests/integration/m5_anchor_test/run_m5_anchor_test.sh`                 |
| **M6**    | GPU path single-GPU + MixedFast compile target   | T4 differential FP64 green on GPU; T4 mixed-precision green within `1e-5 rel`                      | `tests/potentials/test_t4_differential_mixed.cpp`, `tests/integration/m6_smoke/` |
| **M7**    | Two-level TD×SD hybrid (Pattern 2 introduction)  | T7 mixed-scaling pass; Pattern 2 validated with `\|predict − measure\| < 25 %`; Pattern 1 regression preserved | `tests/integration/m7_smoke/run_m7_smoke.sh`, `verify/benchmarks/t7_mixed_scaling/` |
| **M8**    | SNAP + proof-of-value + `MixedFastSnapOnlyBuild` | **Case B per D-M8-6** — honest-documentation artifact gate (see below); `MixedFastSnapOnlyBuild` slow-tier VerifyLab pass GREEN | `tests/integration/m8_smoke/run_m8_smoke.sh`, `tests/integration/m8_smoke_t6/run_m8_smoke_t6.sh`, `verify/benchmarks/t6_snap_scaling/REPORT.md`, `verify/slow_tier/m8_mixed_fast_snap_only_REPORT.md` |

### M8 artifact gate outcome (Case B per D-M8-6)

The master-spec M8 gate is a **dual-path** artifact gate:

> на T6 TDMD либо обгоняет LAMMPS ≥ 20 % на целевой конфигурации
> (≥ 8 ranks, commodity network), либо проект документирует, почему не
> обгоняет и что делать дальше (честная постановка).

v1.0.0-alpha1 invokes the **second leg**: honest documentation of the gap
on T6 tungsten SNAP 2000-atom BCC, 1× RTX 5080 (sm_120, CUDA 13.1).

- **TDMD `MixedFastBuild` GPU:** 29.2 ms/step
- **LAMMPS `snap/kk` KOKKOS GPU:** 4.30 ms/step
- **LAMMPS SNAP CPU 1-rank × 1 OMP:** 178.2 ms/step (baseline context)

Gap: **6.79× slower** than KOKKOS `snap/kk`; **6.10× faster** than LAMMPS
CPU 1-rank. Closing the 6.79× gap to the ≥ 20 % beat target (≤ 3.44 ms/step,
further 8.5×) is not plausibly recoverable via single-GPU kernel tuning
(ladder flattened after 10 atomic-free byte-exact commits from pre-T8.6c
316 ms/step → 29.2 ms/step, **10.82× cumulative**) and is only partially
addressed by multi-rank TD because SNAP is **compute-bound, not
halo-bound** — the workload where TD's native lever
(`T_comm = T_p / K` × per-zone `dt`) dominates is MEAM (angular-moments
halo pressure), scheduled for M10.

Full analysis: [`verify/benchmarks/t6_snap_scaling/REPORT.md`](verify/benchmarks/t6_snap_scaling/REPORT.md).

### What's new since the last tagged milestone (M7)

- **SNAP potential (CPU + GPU)** — full `SnapPotential` implementation
  with `MixedFastSnapOnlyBuild` (§D.11 Philosophy B-het: per-kernel
  precision split; SNAP pair-math narrows to FP32, EAM stays FP64).
  Landed via the formal §D.17 seven-step procedure with the slow-tier
  VerifyLab pass required before acceptance.
- **T6 canonical tungsten benchmark** — 1024-atom W BCC with `W_2940_2017_2`
  Wood 2017 coefficients, NVE drift gate `\|ΔE\|/\|E₀\| ≤ 1e-6` per 10-step
  smoke (D-M8-8, √-scaled from the 3e-6 per 100-step baseline). Companion
  8192-atom scaling variant exercised at T8.11.
- **GPU SNAP reduce-then-scatter** (gpu/SPEC §6.1) — zero `atomicAdd(double)`
  on the hot path; per-bond exclusive storage + per-atom gather; bond-list
  emission order matches CPU cell-stencil walk order, validated as the
  `test_bond_list_matches_cpu_stencil_order` byte-exact regression.
- **D-M8-13 byte-exact chain** — extends D-M7-10 to SNAP: 2-rank Pattern 2
  K=1 P_space=2 ≡ 1-rank legacy thermo byte-for-byte on the T6 fixture.
  Locked in as a CI-wired smoke (`tests/integration/m8_smoke/`).
- **LAMMPS submodule integration** — SNAP coefficient files ship via the
  LAMMPS git submodule (D-M8-3, pinned at `stable_22Jul2025_update4`). No
  TDMD-side binary shipped.

### Known limitations

- **NVT / NPT under TD:** v1.0.0-alpha1 supports NVE only. NVT / NPT with
  `pipeline_depth_cap = 1` is accepted (degenerates to SD); any `K > 1`
  with `integrator.style != nve` is rejected at preflight per
  integrator/SPEC §7.3.1. Research window opens at M11.
- **Single-GPU SNAP performance:** 6.79× behind KOKKOS `snap/kk` on T6
  (documented above). Multi-GPU multi-rank TD (≥ 2 physical GPUs, T8.11
  cloud-burst) is v1.5 territory.
- **MEAM / PACE / MLIAP:** not shipped. MEAM arrives at M10; PACE / MLIAP
  at M12.
- **BuildFlavor matrix:** v1-alpha ships five flavors (`Fp64ReferenceBuild`,
  `Fp64ProductionBuild`, `MixedFastBuild`, `MixedFastAggressiveBuild`,
  `MixedFastSnapOnlyBuild`). A sixth flavor requires the master spec §D.17
  formal procedure — not a runtime toggle.
- **Public CI GPU coverage:** `Option A` — no self-hosted GPU runner on
  public CI. GPU smokes self-skip on `ubuntu-latest`; full pre-push local
  gate is the developer's responsibility (D-M6-6). Revisited at M6+ cadence.

### Quality gates that must stay green on any M8+ PR

Local pre-push gates (D-M7-17 + D-M8-14 regression preservation):

- `tests/integration/{m1..m8}_smoke/` — the full byte-exact chain from
  single-rank CPU Morse (M1) through 2-rank GPU Pattern 2 SNAP (M8).
- `tests/integration/m5_anchor_test/` — anchor-test §13.3 reproduction of
  Andreev 10⁶ Al FCC dissertation result.
- `tests/potentials/test_t{1,4,6}_differential*.cpp` — differential vs
  LAMMPS on Morse / EAM / SNAP oracles.
- `tests/integration/m8_smoke_t6/run_m8_smoke_t6.sh` — NVE drift gate on
  the T6 canonical 1024-atom W BCC fixture.

CI coverage (all green):

- `build-cpu` (clang-17, gcc-13) + CPU test suite + M1..M5 smokes.
- `build-gpu` + `build-gpu-snap` (compile + link only, Option A).
- M6..M8 smokes wired end-to-end; self-skip on no-GPU hosts but still
  validate template / LFS / submodule / golden-file integrity.
- T1 / T4 differentials — self-skip on absent LAMMPS submodule per
  Option A.

### Full per-task commit log — T8 series (M8)

See [`docs/development/m8_execution_pack.md`](docs/development/m8_execution_pack.md) §4 + §5
for the T8.0..T8.13 task ledger. Headline T-opt optimisation ladder
(10 commits, 316 → 29.2 ms/step MixedFast, all byte-exact):

| # | Commit     | Scope                                                                     | ms/step | Cum. speedup |
|--:|------------|---------------------------------------------------------------------------|--------:|-------------:|
| 0 | —          | Pre-T8.6c baseline (T8.7 correctness gate, all-tid==0 prototype)          |   316.0 |        1.00× |
| 1 | `1c22694`  | T8.6c-v1 — `snap_ui_kernel` block-parallel `add_uarraytot`                |  ≈316   |        1.00× |
| 2 | `ab0bcff`  | T8.6c-v2 — `snap_yi_kernel` Phase A parallel over `idxz_max`              |   227.3 |        1.39× |
| 3 | `88d23fb`  | T8.6c-v3 — `snap_deidrj_kernel` Phase B warp-shuffle `dedr`               |   208.6 |        1.51× |
| 4 | `223a35a`  | T8.6c-v4 — `compute_uarray` + `compute_duarray` intra-layer parallelism   |    92.0 |        3.43× |
| 5 | `4cf0202`  | T8.6c-v5 S1 — `SnapBondListGpu` CSR+SoA bond-list infra                   |    92.0 |        3.43× |
| 6 | `ac781ac`  | T8.6c-v5 S2 — `snap_ui_bond_kernel` (gather replaces per-atom walk)       |    57.1 |        5.53× |
| 7 | `6c5feb1`  | T8.6c-v5 S3 — `snap_deidrj_bond_kernel` (gather replaces per-atom walk)   |    47.9 |        6.60× |
| 8 | `c380e10`  | T-opt-3b — paired-bond reverse index (`dedr_peer[b] ≡ dedr_own[rev(b)]`)   |    34.4 |        9.19× |
| 9 | `63847c2`  | T-opt-2 — `yi_kernel` Phase B parallel-over-jju via CSR buckets           |    29.5 |       10.71× |
|10 | `e12abd5`  | T-opt-4 Item 1 — bond-list single-walk (stage+compact replaces count+emit)|  **29.2** |    **10.82×** |

### Roadmap

- **M9** — NVT / NPT baseline with `pipeline_depth_cap = 1` enforced;
  Nosé-Hoover CPU + GPU; new T8 / T9 canonical benchmarks.
- **M10** — MEAM integration (`MeamPotential` CPU + GPU). First workload
  where TD's native lever dominates (angular-moments halo pressure).
- **M11** — NVT-under-TD research window; Tuckerman 2010 / Eastwood 2010
  lazy-thermostat adaptability study (Вариант C from integrator/SPEC
  §7.3.3).
- **v1.0.0-beta** — M11 close; production NVT / NPT under TD if research
  window succeeds; fallback path documented otherwise.

### Links

- Master spec: [`TDMD_Engineering_Spec.md`](TDMD_Engineering_Spec.md) §14.
- M8 execution pack: [`docs/development/m8_execution_pack.md`](docs/development/m8_execution_pack.md).
- M8 Case B closure report: [`verify/benchmarks/t6_snap_scaling/REPORT.md`](verify/benchmarks/t6_snap_scaling/REPORT.md).
- `MixedFastSnapOnlyBuild` rationale: [`docs/specs/potentials/mixed_fast_snap_only_rationale.md`](docs/specs/potentials/mixed_fast_snap_only_rationale.md).
- Anchor-test (Andreev reproduction): [`tests/integration/m5_anchor_test/README.md`](tests/integration/m5_anchor_test/README.md).
