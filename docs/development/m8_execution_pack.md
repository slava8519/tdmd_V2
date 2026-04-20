# M8 Execution Pack

**Document:** `docs/development/m8_execution_pack.md`
**Status:** draft, awaiting human review
**Parent:** `TDMD_Engineering_Spec.md` §14 (M8), §D.11 (BuildFlavor matrix), §D.17 (new BuildFlavor procedure), `docs/specs/potentials/SPEC.md` (§4 SNAP interface), `docs/specs/verify/SPEC.md` (T6 tungsten benchmark), `docs/development/m7_execution_pack.md` (template), `docs/development/claude_code_playbook.md` §3
**Milestone:** M8 — SNAP proof-of-value + `MixedFastSnapOnlyBuild` + v1 alpha closure — 6 недель target per master spec §14
**Created:** 2026-04-20 (M7 closed same day)
**Author:** Architect / Spec Steward role (Claude Opus 4.7)

---

## 0. Purpose

Этот документ декомпозирует milestone **M8** master spec'а §14 на **14 PR-size задач**
(T8.0..T8.13), из которых **T8.0 — это carry-forward задача T7.8b из M7** (2-rank
overlap gate infrastructure уже landed 2026-04-20), намеренно первая как cleanup
M7 долга перед SNAP work. Документ — **process artifact**, не SPEC delta.

M8 — **first encounter TDMD с representative ML potential** и **v1 alpha closure**.
После M7 на GPU работает: (a) весь reference-path (M3 ≡ M4 ≡ M5 ≡ M6 ≡ M7 Pattern 2
K=1 byte-exact chain); (b) Pattern 2 two-level TD×SD hybrid (inner NCCL + outer
CUDA-aware MPI); (c) MixedFastBuild для EAM/alloy (≤1e-5 rel force dense, D-M6-8);
(d) PerfModel Pattern 2 с 25% tolerance; (e) T7 mixed-scaling benchmark с ≥80%
1-node × 8 GPU efficiency gate; (f) 2-rank K=4 overlap gate infrastructure (T8.0 =
T7.8b carry-forward, 30% runtime measurement cloud-burst-gated). M8 добавляет:

- **`SnapPotential` CPU + GPU** — новый potential family; bispectrum coefficient
  flow, linear regression на per-atom energy/force; Wood+Thompson+Trott 2014 formulation;
- **LAMMPS SNAP oracle via submodule** — canonical per-atom reference; differential
  harness `t_snap_cpu_vs_lammps` + `t_snap_gpu_vs_lammps`; T6 tungsten
  `W_2940_2017_2.snap` coefficient set resolved via submodule path (no binary в
  tdmd repo per OQ-M8-3);
- **GPU SNAP bit-exact gate** — D-M6-7 extended: GPU FP64 SNAP ≤ 1e-12 rel vs CPU
  FP64 SNAP на T6 tungsten (2048-atom BCC W fixture);
- **`MixedFastSnapOnlyBuild`** — новый BuildFlavor (master spec §D.11 addition):
  `StateReal=double`, `ForceReal=float` для SNAP kernels, `ForceReal=double` для
  EAM kernels (heterogeneous precision gate); formal **§D.17 7-step procedure**
  (rationale, matrix, thresholds, CMake, slow-tier pass, docs, Architect+Validation
  review);
- **SNAP MixedFast kernel** — `src/gpu/snap_gpu_mixed.cu` с ≤ 1e-5 rel force /
  ≤ 1e-7 rel PE gate (mirror D-M6-8 dense-cutoff) vs FP64 GPU reference;
- **T6 tungsten SNAP benchmark** — canonical verify target; single-node + multi-node
  scaling runs; bit-exactness + differential gates;
- **TDMD vs LAMMPS SNAP comparison** — weak/strong scaling на ≥ 8 ranks; commodity
  network; cloud-burst-gated per D-M6-6 Option A CI policy;
- **Slow-tier VerifyLab full pass** для `MixedFastSnapOnlyBuild` (§D.17 step 5
  mandatory gate);
- **M8 integration smoke + v1 alpha tag** — финальный regression gate + git tag
  `v1.0.0-alpha1`.

**Conceptual leap от M7 к M8:**

- M7 = "Pattern 2 landed" (two-level TD×SD hybrid; inner NCCL + outer GpuAwareMPI;
  OuterSdCoordinator; halo snapshot archive; SubdomainBoundaryDependency dep kind;
  engine wired для Pattern 2; T7 mixed-scaling gate).
- **M8 = "proof-of-value on ML kernel"** (SnapPotential new family; LAMMPS SNAP
  oracle; heterogeneous-precision BuildFlavor formal procedure; TDMD vs LAMMPS
  demo; v1 alpha tag).
- M9 = "long-range and NVT/NPT Pattern 2" (Ewald/PPPM cost model; NPT в Pattern 2
  K=1; MEAM/PACE potentials; Morse GPU kernel — unblocks T3-gpu full dissertation
  replication).

Критически — **M7 Pattern 2 + M6 EAM GPU path preserved**. Любой M8 PR проходит:
(a) M1..M7 integration smokes; (b) T1/T4 differentials; (c) T3-gpu anchor; (d) M7
Pattern 2 smoke; (e) D-M7-10 byte-exact chain M3 ≡ M4 ≡ M5 ≡ M6 ≡ M7. Zero-regression
mandate (master spec §14 M8).

**M7 carry-forward — намеренно встроен в M8:**

- **T7.8b → T8.0 (LANDED 2026-04-20)** — 2-rank overlap gate test infrastructure
  (`tests/gpu/test_overlap_budget_2rank.cpp` + main_mpi.cpp + gpu/SPEC §3.2c
  hardware prerequisite + dev SKIP semantics). Runtime 30% measurement requires
  ≥ 2 GPU node, ties into T8.11 cloud burst harness.
- **T7.13b (Pattern 2 ±25% calibration)** → deferred **beyond M8** as low-urgency
  (Pattern 1 ±20% calibration already shipped в T7.13; Pattern 2 ±25% placeholder
  documented в perfmodel/SPEC §11.5 + §11.6). Revisit under M10+ performance
  tuning window.

После успешного закрытия всех 14 задач и acceptance gate (§5) — milestone M8
завершён, git tag `v1.0.0-alpha1` pushed; execution pack для M9 создаётся как
новый аналогичный документ.

---

## 1. Decisions log (зафиксировано до старта T8.1 — this pack)

| # | Решение | Значение | Rationale / источник |
|---|---|---|---|
| **D-M8-1** | SNAP implementation strategy | **Port from LAMMPS USER-SNAP** с explicit attribution в LICENSE + source headers. GPL-compatible license chain (LAMMPS: GPLv2; TDMD: Apache-2.0 compatible via GPLv2 re-license clause). Не reimplement from scratch — risk of subtle bispectrum basis function bugs слишком высок. | OQ-M8-1 resolution; master spec §12.1 "port SNAP from LAMMPS"; USER-SNAP is canonical reference implementation; reimplementation adds months of bug-hunting without science value. |
| **D-M8-2** | LAMMPS oracle integration | **Already landed M1 T1.11** — git submodule at `verify/third_party/lammps/` pinned `stable_22Jul2025_update4`; `PKG_ML-SNAP=on` already in `tools/build_lammps.sh`; `cmake/FindLammps.cmake` locates install prefix `verify/third_party/lammps/install_tdmd/`. M8 reuses existing infrastructure; T8.2 reduces to **SNAP subset verification + canonical fixture selection + docs** (not submodule add + CMake option authoring). | Discovery 2026-04-20 while prepping T8.2: LAMMPS oracle shipped в M1 T1.11 per `verify/third_party/lammps_README.md`; `lammps_README.md` Required packages table lists `ML-SNAP` with milestone = `M8`. No new submodule or CMake work required. |
| **D-M8-3** | T6 tungsten coefficient file | **`W_2940_2017_2.snap`** — single-species pure W BCC, Wood & Thompson 2017, 2940 DFT training configs. Path: `verify/third_party/lammps/examples/snap/W_2940_2017_2.snap`. **No binary tracked by tdmd repo** (lives inside submodule). CMake fixture dir: `${CMAKE_SOURCE_DIR}/verify/third_party/lammps/examples/snap/`. Driver example: `in.snap.W.2940`. | OQ-M8-3 resolution. Earlier draft referenced `W_2940_2017_2.snap` which is actually `WBe_Wood_PRB2019.snap` (W-Be binary alloy, not pure W) — corrected 2026-04-20. Pure W single-species simpler для T6 canonical; WBe available если binary-alloy SNAP gate added M9+. |
| **D-M8-4** | `MixedFastSnapOnlyBuild` scope | **Heterogeneous precision**: SNAP kernels use `ForceReal=float`, EAM kernels use `ForceReal=double`. **State always `double`** (matches MixedFastBuild policy). Motivation: SNAP bispectrum coefficients are empirically fit against FP32-noise-tolerant DFT energies (typical fit RMSE ≈ 1e-3 eV/atom >> FP32 ULP); EAM tabulated potentials require FP64 Horner stability (D-M6-8 evidence). | Master spec §D.11 addition; §D.17 7-step procedure enforces formal justification; slow-tier VerifyLab pass is hard gate. |
| **D-M8-5** | Cloud burst acceptance | **≥ 8-rank scaling measurement requires cloud burst**. Dev hardware = single RTX 5080 (1 GPU). Acceptance run uses cloud-burst-gated harness (AWS p4d.24xlarge = 8× A100, or equivalent); measurement result checked into `verify/benchmarks/t6_snap_scaling/results_<date>.json`. | D-M6-6 Option A CI policy: no self-hosted GPU runner в public repo; scaling measurements are pre-release workflow, not CI gate. Memory `project_option_a_ci.md`. |
| **D-M8-6** | Artifact gate interpretation | Master spec §14 M8 artifact gate literal: "либо обгоняет LAMMPS ≥ 20 % на целевой конфигурации, либо документирует, почему не обгоняет и что делать дальше". **Both outcomes close M8** provided the measurement was performed honestly on a representative config. Cherry-picking degenerate configs (e.g. extreme subdomain aspect) to force 20% win is auto-reject (master spec §11.4). | Honest engineering mandate; project-level credibility gate; full comparison methodology в T8.11 scope. |
| **D-M8-7** | SNAP potential interface surface | `SnapPotential : public Potential` — inherits existing contract (potentials/SPEC §3); per-atom energy + per-pair force API already fits (SNAP is short-range + cutoff-based despite density-coefficient flavour). **No new interface** required. | Master spec §8.2 ownership: potentials own force computation, state owns atoms — SNAP fits without contract extension. |
| **D-M8-8** | Differential thresholds for SNAP | **CPU FP64 SNAP vs LAMMPS FP64 SNAP ≤ 1e-12 rel** per-atom force + ≤ 1e-12 rel total PE (D-M1 diff gate); **GPU FP64 vs CPU FP64 ≤ 1e-12 rel** (D-M6-7); **MixedFastSnapOnly vs GPU FP64 ≤ 1e-5 rel force / ≤ 1e-7 rel PE** (dense-cutoff analog of D-M6-8; SNAP cutoff density typically 20-50 neighbors). | Master spec §13.7 differential gates; numerical analogue of T6.8a EAM dense-cutoff budget; sparse-cutoff gate deferred M9+ (mirrors D-M6-8 sparse resolution в memory `project_fp32_eam_ceiling.md`). |
| **D-M8-9** | SNAP fixture scales | **T8.4–T8.7**: 64-atom W BCC minimum (trivial fixture — single-cell verification); 2048-atom W BCC (8×8×8 unit cell — T6 canonical per verify/SPEC). **T8.10–T8.11**: full T6 8×8×8 + 16×16×16 (16384 atoms) scaling configs. No T8.N uses fixture >16k atoms (N>11 cloud burst only). | verify/SPEC T6 description + M6 fixture sizing precedent; 2048 atoms comfortably fits single GPU; 16384 atoms exercises Pattern 2 multi-GPU. |
| **D-M8-10** | Active BuildFlavors (M8 closure) | `Fp64ReferenceBuild` (oracle) + `MixedFastBuild` (M6/M7 carry) + **`MixedFastSnapOnlyBuild` (new в M8)**. Total three flavors active. Fp64Production/MixedFastAggressive/Fp32Experimental остаются М9+. | §D.17 opens flavor slot; §D.11 matrix updated by T8.8. Keeping М8 flavor count bounded preserves CI matrix size. |
| **D-M8-11** | Active ExecProfiles | `Reference` (byte-exact gate) + `Production` (performance tuning). `Fast` остаётся deferred M9+ (no change from M7). | Unchanged from M7 D-M7-7. |
| **D-M8-12** | CI strategy (M8 addition) | **Option A continues**. New `build-gpu-snap` compile-only matrix cell: `{Fp64ReferenceBuild, MixedFastBuild, MixedFastSnapOnlyBuild}` × `{CPU-only, CUDA}`. LAMMPS oracle build remains local-only (too heavy for public CI; D-M8-2). | Memory `project_option_a_ci.md` + D-M6-6 precedent; no self-hosted runner in public repo. |
| **D-M8-13** | Byte-exact chain extension (M8) | **SNAP Pattern 1 (single-subdomain single-rank) Fp64ReferenceBuild reference thermo byte-exact per D-M6-7 at GPU↔CPU gate**. Chain extension: M3(EAM ref) ≡ M4 ≡ M5 ≡ M6 ≡ M7 Pattern 2 K=1 **parallel to** SNAP Pattern 1 GPU↔CPU gate (different potential, separate chain rooted at T8.5 CPU FP64 oracle). | Master spec §13.5 determinism matrix — Reference profile bitwise oracle across all (potential × build × profile) triples. SNAP chain is fresh — M8 initiates. |
| **D-M8-14** | Timeline | **6 недель target, 7 acceptable, flag at 8**. Most expensive: T8.4 SnapPotential CPU FP64 (~5 days — bispectrum coefficient plumbing + basis function math), T8.6 GPU port (~4 days), T8.9 MixedFast SNAP kernel (~4 days), T8.11 cloud burst scaling (~5 days inclusive of cloud setup + measurement + artifact gate write-up). Остальные 2-4 дня. | Master spec §14 M8 = 6 недель. User confirmed autonomous execution 2026-04-20. Budget в comparison: M6 shipped ~10 days; M7 shipped ~12 days; M8 will land in ~42-56 days. |
| **D-M8-15** | v1 alpha tag format | `v1.0.0-alpha1` annotated git tag pushed at T8.13 closure. Release notes: enumerate M1-M8 acceptance gates met; link to canonical benchmark results (T3-gpu, T6 tungsten SNAP, T7 mixed-scaling); explicit artifact gate outcome (beat LAMMPS OR honest documentation). | Semantic versioning: alpha = user-facing but not feature-complete; first public binary release target post-M11 (M8 — v1 alpha; M11 — v1 beta; M13 — v1.0.0). |

---

## 2. Глобальные параметры окружения

| Параметр | Значение | Примечание |
|---|---|---|
| OS | Linux (Ubuntu 24.04 LTS) | Dev-машина пользователя; ubuntu-latest в CI |
| C++ compiler | GCC 13+ / Clang 17+ | C++20; CI уже проверяет оба (M6/M7 matrix) |
| CMake | 3.25+ | Master spec §15.2 |
| CUDA | **13.1** installed (system `/usr/local/cuda`) | Memory `env_cuda_13_path.md`; CI compile-only |
| GPU archs | sm_80, sm_86, sm_89, sm_90, sm_100, sm_120 | D-M6-1 carry-forward |
| MPI | **CUDA-aware OpenMPI ≥4.1** preferred; non-CUDA-aware → fallback | D-M7-3 carry-forward |
| NCCL | **≥2.18** (bundled с CUDA 13.x) | D-M7-4 carry-forward; intra-node only в M8 |
| LAMMPS oracle | **Already shipped M1 T1.11** — submodule `verify/third_party/lammps/` pinned `stable_22Jul2025_update4`; `tools/build_lammps.sh` with `PKG_ML-SNAP=on`; install prefix `verify/third_party/lammps/install_tdmd/` | D-M8-2 (corrected); SKIP on public CI per Option A |
| LAMMPS SNAP fixture | `verify/third_party/lammps/examples/snap/W_2940_2017_2.snap` (pure W BCC, 2017, 2940 DFT configs) | D-M8-3 (corrected); no binary tracked by tdmd |
| Python | 3.10+ | pre-commit + anchor-test + T6 scaling harness + cloud burst orchestration |
| Test framework | Catch2 v3 (FetchContent) + MPI wrapper | GPU+MPI tests local-only per D-M7-11 |
| Active BuildFlavors | `Fp64ReferenceBuild`, `MixedFastBuild`, **`MixedFastSnapOnlyBuild`** (new) | D-M8-10 |
| Active ExecProfiles | `Reference`, `Production` (GPU) | D-M8-11 |
| Run mode | multi-rank MPI × GPU-per-rank × 1:1 subdomain binding (M7 carry) | D-M7-2 carry-forward |
| Pipeline depth K | `{1, 2, 4, 8}` (as M5/M6/M7); default 1 | Unchanged |
| Subdomain topology | Cartesian 1D/2D/3D (M7 shipped 2D; 3D config allowed but cloud-burst-tested only) | D-M7-1 carry-forward |
| Streams per rank | 2 (default) — compute + mem | D-M6-13 carry; T8.0 2-rank infra shipped |
| CI CUDA | compile-only matrix: `{Ref, Mixed, **MixedSnapOnly**} × {HostStaging, GpuAwareMpi}` | D-M8-12 |
| Local pre-push gates | Full GPU suite + T3-gpu + M1..M7 smokes + M8 SNAP smoke (added T8.13) | D-M7-17 carry + T8.13 addition |
| Cloud burst trigger | T8.11 scaling benchmark (≥ 8 rank) — manual; writes to `verify/benchmarks/t6_snap_scaling/` | D-M8-5 |
| Branch policy | `m8/T8.X-<topic>` per PR → `main` | CI required: lint + build-cpu + build-gpu + build-gpu-snap (new) + M1..M7 smokes; M8 smoke добавляется в T8.13 |

---

## 3. Suggested PR order

**Dependency graph:**

```
T8.0 (T7.8b carry-fwd, LANDED) ─┐
                                │
T8.1 (this pack) ───────────────┼──► T8.2 (verify LAMMPS SNAP subset + T6 fixture)
                                │         │
                                │         ▼
                                │     T8.3 (potentials/SPEC SNAP body)
                                │         │
                                │         ▼
                                │     T8.4 (SnapPotential CPU FP64)
                                │         │
                                │         ▼
                                │     T8.5 (CPU diff vs LAMMPS SNAP)
                                │         │
                                │         ▼
                                │     T8.6 (SnapPotential GPU FP64)
                                │         │
                                │         ▼
                                │     T8.7 (GPU bit-exact gate, D-M6-7)
                                │         │
                                │         ▼
                                │     T8.8 (MixedFastSnapOnlyBuild §D.17)
                                │         │
                                │         ▼
                                │     T8.9 (SNAP MixedFast kernel)
                                │         │
                                │         ▼
                                │     T8.10 (T6 tungsten fixture)
                                │         │
                                │         ▼
                                │     T8.11 (TDMD vs LAMMPS scaling
                                │            + T7.8b 30% runtime measurement
                                │            — CLOUD BURST GATE)
                                │         │
                                │         ▼
                                │     T8.12 (slow-tier VerifyLab pass)
                                │         │
                                │         ▼
                                │     T8.13 (M8 smoke + v1 alpha tag)
                                │         │
                                └─────────┘
```

**Линейная последовательность (single agent):**
T8.0 (DONE) → T8.1 → T8.2 → T8.3 → T8.4 → T8.5 → T8.6 → T8.7 → T8.8 → T8.9 → T8.10
→ T8.11 → T8.12 → T8.13.

**Параллельный режим (multi-agent):** M8 has a **much more linear dep chain** than
M7 because SNAP kernels are a single code path (CPU FP64 → GPU FP64 → GPU MixedFast),
unlike M7 где три comm backends (T7.3/T7.4/T7.6) были параллелизуемы. Only a
shallow parallelization window exists between T8.6 (GPU FP64 functional) and T8.7
(bit-exact gate) — T8.8 SPEC delta and T8.9 implementation скелет могут начинаться
одновременно с T8.7 validation. T8.11 (cloud burst) must be serial — измерение
требует uninterrupted wall-clock.

**Estimated effort:** 6 недель target (single agent, per D-M8-14). Самые длинные —
T8.4 SnapPotential CPU FP64 (~5 дней — bispectrum basis + LAMMPS-style linalg),
T8.6 GPU port (~4 дня), T8.9 MixedFast kernel (~4 дня), T8.11 cloud burst ~5 дней
(setup + measurement + analysis + artifact gate write-up). Остальные 2-4 дня.

---

## 4. Tasks

### T8.0 — M7 T7.8b carry-forward — 2-rank overlap gate infrastructure (CLOSED 2026-04-20)

```
# TDMD Task: Close M7 T7.8b debt — 2-rank overlap gate test infrastructure

## Context
- Master spec: §12.6 (CommBackend), §14 M7 (overlap gate carry-forward)
- Module SPEC: `docs/specs/gpu/SPEC.md` §3.2b (T7.8 K=4 2-stream overlap)
- Role: GPU / Performance Engineer
- Milestone: M8 T8.0 (first task of M8, cleanup of M7 T7.8b debt)
- User directive 2026-04-20: schedule first в M8 as cleanup before SNAP work

## Goal (retrospective)
Ship 2-rank K=4 overlap gate test infrastructure: `test_overlap_budget_2rank.cpp`
with per-rank device pinning (cudaSetDevice(rank % device_count)); synthetic halo
(1024 doubles pinned) exchanged via MPI_Sendrecv interleaved with GPU compute;
K=4 sync vs pipelined timing comparison; median-of-9 repeats; D-M6-7 bit-exact
slot 0 vs serial oracle ≤ 1e-12 rel; overlap ratio REQUIRE(≥ 0.30) final gate.
Honest dev SKIP semantics: `cudaGetDeviceCount() < 2` → SKIP exit 4, Catch2
`SKIP_RETURN_CODE 4` makes CTest report SKIPPED rather than FAIL.

## Scope (delivered)
- [x] `tests/gpu/test_overlap_budget_2rank.cpp` — core T8.0 deliverable (~370 LOC
  after pre-commit reformat)
- [x] `tests/gpu/main_mpi.cpp` — MPI-aware Catch2 entry point mirroring
  runtime/comm/scheduler main_mpi.cpp pattern
- [x] `tests/gpu/CMakeLists.txt` — MPI-gated target with `find_program(MPIEXEC_EXECUTABLE)`
  + `--oversubscribe` for mpirun + `SKIP_RETURN_CODE 4`
- [x] `docs/specs/gpu/SPEC.md` v1.0.16 — new §3.2c "T8.0 (T7.8b carry-forward) —
  2-rank overlap gate: hardware prerequisite + dev SKIP semantics" (hardware
  prerequisite ≥ 2 GPU, dev SKIP semantics, why 30% achievable at 2-rank K=4)
- [x] `docs/development/m7_execution_pack.md` — T7.8b status line under T7.8 entry
- [x] `TDMD_Engineering_Spec.md` Приложение C T8.0 addendum
- [x] Commit 46ef2e0 (2026-04-20); CI 24660486923 — 8/8 green

## Out of scope (deferred)
- [excluded] Runtime 30% measurement on ≥ 2 GPU hardware — deferred to T8.11
  cloud burst harness (shares infrastructure with TDMD vs LAMMPS scaling).

## Mandatory invariants (met)
- [x] D-M6-7 byte-exact gate preserved (slot 0 PE + virial vs serial oracle ≤
  1e-12 rel checked inline, independent of overlap ratio gate).
- [x] Test self-skips cleanly on 1-GPU dev hosts (exit code 4); CTest surfaces
  SKIPPED rather than FAIL.
- [x] Per-rank cudaSetDevice + peer setup correct (`devices < 2 → SKIP` guard
  before any device work).
- [x] Regression: M1..M7 smokes + T1/T4 differentials + T3-gpu anchor all green
  on dev machine and public CI.

## Acceptance criteria (met)
- [x] Test compiles, links, self-skips on dev machine with exit 4.
- [x] Both ranks SKIP cleanly under `mpirun --oversubscribe -np 2`.
- [x] Catch2 reports "1 skipped" per rank; CTest aggregates as SKIPPED.
- [x] CI 8/8 green (Lint, Docs lint, Build CPU gcc-13 + clang-17, Build GPU
  compile-only Fp64Reference + MixedFast, Differential T1 + T4 SKIP on public).
- [x] Pre-impl + session reports attached в PR.
- [x] Human review approval.

## Status
LANDED 2026-04-20. Task closed in execution-pack snapshot; no further work.
```

---

### T8.1 — Author M8 execution pack (this document)

```
# TDMD Task: Create M8 execution pack

## Context
- Master spec: §14 M8
- Role: Architect / Spec Steward
- Milestone: M8 (kickoff после T8.0 closure)

## Goal
Написать `docs/development/m8_execution_pack.md` декомпозирующий M8 на 14 PR-size
задач (T8.0 = carry-forward closure, T8.1 = this pack, T8.2..T8.13 = SNAP +
MixedFastSnapOnly + v1 alpha closure). Document-only PR per playbook §9.1.

## Scope
- [included] `docs/development/m8_execution_pack.md` (single new file)
- [included] Decisions log D-M8-1..D-M8-15
- [included] Task templates T8.0 (retrospective closed) + T8.1..T8.13
- [included] M8 acceptance gate checklist (master spec §14 artifact gate)
- [included] Risks R-M8-1..R-M8-N + open questions OQ-M8-*
- [included] Roadmap alignment (consumers M9-M13)

## Out of scope
- [excluded] Any code changes (T8.2+ territory)
- [excluded] SPEC deltas (T8.3 onwards; T8.8 carries major §D.11/§D.17 delta)
- [excluded] LAMMPS SNAP subset verification (T8.2 territory)

## Required files
- `docs/development/m8_execution_pack.md`

## Acceptance criteria
- Document covers §0-§7 complete (Purpose, Decisions, Env, PR order, Tasks, Gate,
  Risks, Roadmap).
- Markdown lint + pre-commit hooks green.
- Human review approval.
```

---

### T8.2 — Verify LAMMPS SNAP oracle subset + canonical W fixture selection

```
# TDMD Task: Verify M1-landed LAMMPS oracle ML-SNAP subset + pick T6 canonical fixture

## Context
- Master spec: §12.1 ("port SNAP from LAMMPS"), §13.7 (differential thresholds)
- Module SPEC: `docs/specs/verify/SPEC.md` §3 (LAMMPS as oracle, T6 tungsten)
- D-M8-2 (corrected): LAMMPS oracle already shipped M1 T1.11 —
  `verify/third_party/lammps/` submodule pinned `stable_22Jul2025_update4`;
  `tools/build_lammps.sh` has `PKG_ML-SNAP=on`; no new infrastructure needed.
- D-M8-3 (corrected): canonical T6 fixture is `W_2940_2017_2.snap` (pure W BCC,
  Wood & Thompson 2017, 2940 DFT configs); available в submodule at
  `verify/third_party/lammps/examples/snap/`.
- Discovery 2026-04-20: prior draft assumed fresh submodule setup; reality —
  M1 already landed full LAMMPS oracle с SNAP readiness marked for M8.
- Role: Validation / Reference Engineer
- Depends: T8.1 (pack authored)

## Goal
Verify that the LAMMPS ML-SNAP subset is operational на dev hardware and pick
the canonical T6 tungsten SNAP fixture. Much smaller scope than originally
drafted — no new submodule, no new CMake option, no new helper cmake module.
The task reduces to: (a) run `tools/build_lammps.sh` end-to-end и confirm
ML-SNAP compiles cleanly + `lmp -h | grep ML-SNAP` reports present; (b) run
LAMMPS SNAP example `in.snap.W.2940` to confirm SNAP functionally executes;
(c) document canonical T6 fixture choice (`W_2940_2017_2.snap` pure W BCC) в
`verify/third_party/lammps_README.md` SNAP section + `docs/specs/verify/SPEC.md`
§3 T6 sub-section.

## Scope
- [included] Run `tools/build_lammps.sh` end-to-end; record build time + install
  size + `lmp -h | grep ML-SNAP` output в verification log (attached в PR desc)
- [included] Run `verify/third_party/lammps/install_tdmd/bin/lmp -in
  verify/third_party/lammps/examples/snap/in.snap.W.2940` и confirm SNAP
  functional (total PE matches upstream LAMMPS `log.15Jun20.snap.W.2940.g++.1`
  reference to LAMMPS float precision — sanity check, not TDMD gate)
- [included] Update `verify/third_party/lammps_README.md` — new **SNAP fixture**
  section: canonical T6 = `W_2940_2017_2.snap`; optional W-Be alloy =
  `WBe_Wood_PRB2019.snap` (deferred M9+ binary alloy gate)
- [included] Update `docs/specs/verify/SPEC.md` §3 — T6 tungsten benchmark
  fixture choice = `W_2940_2017_2.snap`; driver example `in.snap.W.2940`
- [included] `tests/potentials/test_lammps_oracle_snap_available.cpp` — Catch2
  gate: fixture file exists at canonical path; self-skips exit 4 if oracle
  build missing (matches existing EAM oracle test pattern); на public CI
  skips cleanly (submodule init not required)
- [included] `tests/potentials/CMakeLists.txt` edit — register new test
- [included] pre-impl + session reports

## Out of scope
- [excluded] Re-pinning submodule (D-M8-2 corrected locks current pin)
- [excluded] New submodule add (M1 T1.11 already shipped it)
- [excluded] CMake option authoring (FindLammps.cmake already exists)
- [excluded] Helper cmake module (tools/build_lammps.sh is authoritative)
- [excluded] SnapPotential implementation (T8.4)
- [excluded] Differential harness (T8.5)
- [excluded] potentials/SPEC SNAP body (T8.3)

## Mandatory invariants
- Existing builds unchanged (no CMake surface churn).
- ML-SNAP subset proven functional via `in.snap.W.2940` example run.
- `W_2940_2017_2.snap` path resolution tested в new Catch2 gate.
- All M1..M7 regressions green; no regression on existing EAM oracle tests
  (`test_lammps_oracle_available` or equivalent from M1).

## Required files
- `verify/third_party/lammps_README.md` (edit — add SNAP fixture section)
- `docs/specs/verify/SPEC.md` (edit — §3 T6 tungsten fixture choice)
- `tests/potentials/test_lammps_oracle_snap_available.cpp` (new — Catch2 gate)
- `tests/potentials/CMakeLists.txt` (edit — register test)
- `TDMD_Engineering_Spec.md` Приложение C — T8.2 addendum

## Required tests
- `test_lammps_oracle_snap_available::fixture_path_resolves` — checks
  `verify/third_party/lammps/examples/snap/W_2940_2017_2.snap` exists; self-skips
  с clear message if submodule not initialized.
- Local verification: `tools/build_lammps.sh` completes successfully; `lmp -h |
  grep ML-SNAP` reports non-empty; `lmp -in in.snap.W.2940` produces expected
  total PE.

## Acceptance criteria
- ML-SNAP package builds + installs cleanly via `tools/build_lammps.sh`.
- `in.snap.W.2940` example runs successfully; PE matches reference log.
- Canonical T6 fixture choice documented in two places (submodule README +
  verify/SPEC §3).
- New Catch2 gate green on dev machine; skips cleanly on public CI.
- M1..M7 regressions green.
- Pre-impl + session reports attached.
- Human review approval.
```

---

### T8.3 — potentials/SPEC SNAP body + interface contract

```
# TDMD Task: Flesh out potentials/SPEC §4 SnapPotential interface

## Context
- Master spec: §12.1 (SNAP potential family listed), §D.11 (MixedFastSnapOnly prep)
- Module SPEC: `docs/specs/potentials/SPEC.md` §4 (currently placeholder)
- Role: Physics Engineer + Architect (joint review)
- Depends: T8.2 (oracle available for future differentials)

## Goal
Populate `docs/specs/potentials/SPEC.md` §4 SnapPotential (currently ~20-line
placeholder pointing to "M8 authors body"): full interface contract including
(a) coefficient file format description (LAMMPS `.snap` format); (b) bispectrum
coefficient flow (twojmax order, radial basis, angular basis); (c) per-atom energy
formula (linear combination of bispectrum components); (d) per-pair force flow
(autodiff through bispectrum — LAMMPS uses hand-coded gradient path); (e)
neighbor-list requirements (cutoff per-species `rcutij`); (f) threshold
registry entries (D-M8-8 gates). **No code** — SPEC delta PR.

## Scope
- [included] `docs/specs/potentials/SPEC.md` §4 body (~300 lines added) — interface
  contract, coefficient format, basis function math references (Thompson et al.
  2015 citation), per-atom E + per-pair F formulas, NL requirement, error cases
- [included] `docs/specs/potentials/SPEC.md` §4a — MixedFastSnapOnly preparation
  (heterogeneous-precision hook: SNAP uses ForceReal=float, EAM uses ForceReal=double;
  references master spec §D.11 addition coming в T8.8)
- [included] `docs/specs/verify/SPEC.md` — T6 tungsten fixture description
  extended (coefficient file name, source citation Wood & Thompson PRB 2019,
  submodule path)
- [included] `verify/threshold_registry.yaml` — entries reserved for T8.5/T8.7/T8.9
  differential gates (set to "pending" — populated by impl tasks)
- [included] Master spec Приложение C T8.3 addendum + change log entries в
  potentials/SPEC and verify/SPEC

## Out of scope
- [excluded] Implementation code (T8.4-T8.9 territory)
- [excluded] MixedFastSnapOnlyBuild formal delta (T8.8 — §D.11+§D.17 delta is its
  own PR per playbook §9.1)
- [excluded] Test files (T8.4+)

## Mandatory invariants
- potentials/SPEC §3 existing interface unchanged (SnapPotential conforms to
  current Potential API — D-M8-7 ownership invariant).
- No backwards-incompatible contract changes.
- Cross-references to master spec §D.11 marked "upcoming per T8.8"; avoids
  forward declarations that could become stale.

## Required files
- `docs/specs/potentials/SPEC.md` (edit, ~300 lines added)
- `docs/specs/verify/SPEC.md` (edit, ~30 lines)
- `verify/threshold_registry.yaml` (edit, reserve entries)
- `TDMD_Engineering_Spec.md` Приложение C

## Required tests
- None (SPEC-only PR). Tests land в T8.4+.

## Acceptance criteria
- Markdown lint green.
- Human review approval on contract additions (Architect + Physics Engineer joint).
- potentials/SPEC §4 now covers full interface contract без placeholders;
  ready for T8.4 implementation.
- Threshold registry entries allocated с "pending" values + SPEC xref.
```

---

### T8.4 — SnapPotential CPU FP64 implementation

```
# TDMD Task: SnapPotential CPU FP64 — canonical oracle implementation

## Context
- Master spec: §12.1 (SNAP potential family)
- Module SPEC: `docs/specs/potentials/SPEC.md` §4 (populated in T8.3)
- Role: Physics Engineer
- Depends: T8.3 (SPEC finalized)

## Goal
Реализовать `tdmd::potentials::SnapPotential` — CPU FP64 implementation conforming
to potentials/SPEC §4. Port bispectrum coefficient basis function code from LAMMPS
USER-SNAP с **explicit attribution** в source header + LICENSE notice. Per-atom
energy = linear combination of bispectrum components `B_i^k` scaled by learned
coefficients `beta_k`; per-pair force = autodiff through bispectrum basis (LAMMPS
hand-coded gradient path — port preserving algorithmic structure). **Canonical
FP64 oracle** — all downstream GPU variants measured against this.

## Scope
- [included] `src/potentials/include/tdmd/potentials/snap.hpp` — class decl
  inheriting `Potential`; PIMPL firewall (SNAP internals hidden)
- [included] `src/potentials/snap.cpp` — main body: coefficient file parser
  (LAMMPS `.snap` format); `SnapPotential::compute_forces()` per-atom iteration;
  bispectrum basis computation
- [included] `src/potentials/snap_bispectrum.{hpp,cpp}` — basis function math
  ported from LAMMPS `pair_snap.cpp` + `sna.cpp`; extensive attribution header
  "Ported from LAMMPS USER-SNAP — Thompson et al. 2015; J. Comp. Phys. 285:316;
  Wood & Thompson PRB 2019"
- [included] `src/potentials/snap_coeff_parser.{hpp,cpp}` — LAMMPS `.snap`
  coefficient file reader (header + coefficient matrix)
- [included] `tests/potentials/test_snap_cpu.cpp` — Catch2 unit tests:
  single-atom bispectrum component correctness (known fixture values from LAMMPS
  stdout); two-atom force symmetry (Newton's 3rd law); small-cell energy
  conservation under 100-step VV NVE
- [included] CMake wiring: `src/potentials/CMakeLists.txt` — add snap sources;
  no new BuildFlavor yet (T8.8 does that)
- [included] `docs/specs/potentials/SPEC.md` §4 change log entry
- [included] LICENSE — GPL portion note (SNAP code is GPL; tdmd dual-licensed
  Apache-2.0/GPL для SNAP files; non-SNAP remains Apache-2.0)
- [included] pre-impl + session reports

## Out of scope
- [excluded] GPU implementation (T8.6 territory)
- [excluded] MixedFast variant (T8.9)
- [excluded] Differential harness (T8.5)
- [excluded] T6 full-scale fixture (T8.10 — this task uses W 64-atom minimum
  fixture for unit tests)

## Mandatory invariants
- LAMMPS attribution in source headers + LICENSE.txt (GPL chain preserved).
- `SnapPotential` conforms to existing `Potential` base class (no interface
  changes — D-M8-7).
- Ownership: SnapPotential writes forces into `AtomsSoA::f` only — never mutates
  positions or velocities (auto-reject pattern per CLAUDE.md).
- Single-atom bispectrum matches published LAMMPS reference output to FP64
  precision (sanity check — not full differential, that's T8.5).
- Newton's 3rd law holds per-pair (unit test gate).
- Energy conservation в 100-step NVE on 64-atom W ≤ 1e-12 rel (sanity — mirrors
  M3 VV reference harness).

## Required files
- `src/potentials/include/tdmd/potentials/snap.hpp` (new)
- `src/potentials/snap.cpp` (new)
- `src/potentials/snap_bispectrum.{hpp,cpp}` (new, ~800 LOC ported)
- `src/potentials/snap_coeff_parser.{hpp,cpp}` (new, ~200 LOC)
- `tests/potentials/test_snap_cpu.cpp` (new)
- `tests/potentials/fixtures/W_64atom_minimal.snap` — tiny ad-hoc coefficient
  set (4x4x4 cell W, hand-crafted для unit test — NOT the T6 Wood PRB2019 set;
  this is a synthetic minimal fixture to exercise code paths without submodule
  dep)
- `src/potentials/CMakeLists.txt` (edit)
- `docs/specs/potentials/SPEC.md` (edit — change log entry)
- `LICENSE` (edit — GPL note для SNAP files)
- `TDMD_Engineering_Spec.md` Приложение C — T8.4 addendum

## Required tests
- `test_snap_cpu::bispectrum_component_matches_lammps_reference` — fixture values
  from LAMMPS stdout
- `test_snap_cpu::newton_third_law_holds` — two-atom test config
- `test_snap_cpu::nve_drift_64atom_w_100step` — rel total energy drift ≤ 1e-12
  over 100-step VV NVE (Fp64Reference profile)
- `test_snap_cpu::coefficient_parser_roundtrip` — parse `.snap` file, serialize
  back, byte-compare (catch parser bugs)

## Acceptance criteria
- SnapPotential compiles, links, passes all four unit tests.
- Energy conservation gate met on 64-atom W.
- All M1..M7 regressions green (CPU + GPU smokes, T1/T4 differentials,
  T3-gpu anchor, T7 mixed-scaling smoke).
- LAMMPS attribution correct и complete.
- Pre-impl + session reports.
- Human review approval (Physics Engineer + Architect для LICENSE matter).
```

---

### T8.5 — CPU differential harness vs LAMMPS SNAP

```
# TDMD Task: Differential runner — TDMD CPU FP64 SNAP vs LAMMPS FP64 SNAP

## Context
- Master spec: §13.7 (differential threshold gates), §14 M8 (LAMMPS comparison)
- Module SPEC: `docs/specs/verify/SPEC.md` (T6 tungsten benchmark)
- D-M8-8 CPU FP64 SNAP ≤ 1e-12 rel per-atom force + ≤ 1e-12 rel total PE
- Role: Validation / Reference Engineer
- Depends: T8.4 (SnapPotential CPU), T8.2 (SNAP subset verified + fixture)

## Goal
Добавить differential runner config `verify/differentials/t6_snap_cpu_vs_lammps/` —
uses existing `DifferentialRunner` infrastructure (M1/M6 precedent). Fixture: T6
tungsten 8×8×8 (2048-atom) + `W_2940_2017_2.snap` via submodule path. Runs TDMD
SnapPotential CPU FP64 vs LAMMPS `pair_style snap` FP64; compares per-atom force
+ total PE. Gate: both ≤ 1e-12 rel. This is the canonical CPU FP64 oracle lock
— all downstream GPU variants use TDMD CPU FP64 as source-of-truth.

## Scope
- [included] `verify/differentials/t6_snap_cpu_vs_lammps/config.yaml.template` —
  2048-atom W BCC, W_2940_2017_2.snap, runs=1 (diff is per-atom-force snapshot,
  not time evolution)
- [included] `verify/differentials/t6_snap_cpu_vs_lammps/checks.yaml` — gates:
  `force_per_atom_rel_max ≤ 1e-12`, `total_pe_rel ≤ 1e-12`
- [included] `verify/differentials/t6_snap_cpu_vs_lammps/README.md` — how to run,
  what it validates, LAMMPS oracle version
- [included] `tests/potentials/test_snap_cpu_vs_lammps.cpp` — Catch2 wrapper
  invoking DifferentialRunner; self-skips if `TDMD_ENABLE_LAMMPS_ORACLE=OFF`
  (local-only gate per D-M8-2)
- [included] `verify/threshold_registry.yaml` — populate `snap_cpu_vs_lammps_*`
  entries (reserved в T8.3)
- [included] CI matrix — adds `differential-snap-cpu` job SKIPping on public CI;
  local pre-push runs full differential
- [included] pre-impl + session reports

## Out of scope
- [excluded] GPU differential (T8.7 territory — GPU FP64 bit-exact vs TDMD CPU
  FP64, NOT vs LAMMPS — chain through TDMD CPU FP64 oracle)
- [excluded] MixedFast differential (T8.9+)
- [excluded] Multi-rank SNAP diff (T8.10 scale)

## Mandatory invariants
- 1e-12 rel gate — если violation, блокирует M8 progression (indicates bug в
  T8.4 CPU impl; need to debug before GPU port).
- Submodule path resolves at differential run time; clear error if oracle
  submodule not initialized.
- DifferentialRunner output deterministic (same seed + fixture → same numerics).
- TDMD CPU FP64 becomes **canonical SNAP oracle** from this gate forward (parallel
  to EAM CPU FP64 oracle lock в M1 T1.11).

## Required files
- `verify/differentials/t6_snap_cpu_vs_lammps/config.yaml.template` (new)
- `verify/differentials/t6_snap_cpu_vs_lammps/checks.yaml` (new)
- `verify/differentials/t6_snap_cpu_vs_lammps/README.md` (new)
- `tests/potentials/test_snap_cpu_vs_lammps.cpp` (new)
- `tests/potentials/CMakeLists.txt` (edit — register test + fixtures dir define)
- `verify/threshold_registry.yaml` (edit — populate entries)
- `docs/specs/verify/SPEC.md` (edit — T6 differential runner entry)
- `TDMD_Engineering_Spec.md` Приложение C — T8.5 addendum

## Required tests
- `test_snap_cpu_vs_lammps::t6_tungsten_2048atom_matches_lammps` — full diff run
  on T6 fixture; gates 1e-12 rel force + 1e-12 rel PE
- `test_snap_cpu_vs_lammps::skips_cleanly_without_oracle` — self-skip exit 4
  if `TDMD_ENABLE_LAMMPS_ORACLE=OFF`

## Acceptance criteria
- Per-atom force rel ≤ 1e-12 vs LAMMPS oracle on T6 tungsten 2048-atom.
- Total PE rel ≤ 1e-12 vs LAMMPS.
- Self-skip works on public CI (oracle build disabled).
- Local pre-push gate green.
- M1..M7 regressions preserved.
- Pre-impl + session reports.
- Human review approval (Validation Engineer primary).
```

---

### T8.6 — SnapPotential GPU FP64 implementation (split into T8.6a [landed] + T8.6b [landed])

**Status 2026-04-20.** T8.6a landed (scaffolding only — class skeletons + PIMPL firewall + CPU-only build guards + M8-scope flag fence + sentinel-throw `compute()` path + `SimulationEngine` dispatch wiring + `preflight::check_runtime` relaxation + `test_snap_gpu_plumbing`); see master spec Приложение C + `docs/specs/gpu/SPEC.md` v1.0.17 §7.5.

**T8.6b landed 2026-04-20.** Full CUDA kernel body — three-kernel architecture (`snap_ui_kernel` Wigner-U accumulation → `snap_yi_kernel` Z→Y→B→PE inline → `snap_deidrj_kernel` per-neighbour dE/dr own-side + peer-side for Newton-3 reassembly), one-block-per-atom launch, single-lane CG contractions; host-side Kahan reductions per D-M6-15 (no atomicAdd(double*)). Index tables flattened on host (`snap_gpu_tables.cu`), uploaded once per `compute()` lifetime. Tested on BCC W 250-atom rattled lattice (fixture `W_2940_2017_2`): GPU vs CPU PE matches (−4457.9), worst-force rel err = 1.32e-14 (already at D-M8-13 1e-12 precision — T8.7 locks formal gate). All three flavors green: `Fp64ReferenceBuild` + CUDA 49/49, `MixedFastBuild` + CUDA 49/49, `Fp64ReferenceBuild` + CPU-only-strict 41/41. Test-lifetime trap resolved: pool→stream→adapter construction order required so destruction runs adapter→stream→pool and DevicePtr deleters reach a still-alive pool. See `docs/specs/gpu/SPEC.md` §7.5 + master spec Приложение C (T8.6b entry).

```
# TDMD Task: SnapPotential GPU FP64 — CUDA port (T8.6b kernel body — T8.6a scaffolding already shipped)

## Context
- Master spec: §14 M8 (SnapPotential GPU)
- Module SPEC: `docs/specs/potentials/SPEC.md` §4 (CPU ref landed); gpu/SPEC §7.5 (SNAP GPU contract — T8.6a authored)
- D-M8-8 GPU FP64 ≤ 1e-12 rel vs CPU FP64 (D-M6-7 chain extension)
- Role: GPU / Performance Engineer + Physics Engineer (joint)
- Depends: T8.5 (CPU FP64 oracle locked), T8.6a (scaffolding landed)

## Goal
Port SnapPotential CPU FP64 к CUDA. Strategy: follow M6 EAM/alloy GPU pattern —
`src/gpu/snap_gpu.cu` с `enqueue_snap_fp64()` async launch + `drain_snap_fp64()`
sync; same stream architecture (compute + mem); NVTX ranges per required macro
guard (memory `project_m1_complete.md` T6.11 NVTX audit precedent). Kernel
architecture: one thread block per ~32 atoms (shared-memory bispectrum basis
cache); FP64 throughout; no cross-atom reductions beyond per-pair force pairs
(Newton's 3rd law handled via atomic add on neighbor-pair entry).

## Scope
- [included] `src/gpu/include/tdmd/gpu/snap_gpu.hpp` — class decl matching
  `EamAlloyGpu` pattern; inherits `GpuPotential` interface
- [included] `src/gpu/snap_gpu.cu` — CUDA kernel FP64; NVTX-wrapped launches per
  T6.11 audit rules
- [included] `src/gpu/snap_gpu_bispectrum.{cuh,cu}` — device bispectrum basis
  functions; thread-block-local shared memory for intermediate reductions
- [included] `tests/gpu/test_snap_gpu_functional.cpp` — Catch2 functional unit
  test (single-atom correctness on 64-atom W); self-skips no-CUDA per M6 pattern
- [included] CMake wiring `src/gpu/CMakeLists.txt` (edit)
- [included] `docs/specs/gpu/SPEC.md` §7.3 (SNAP kernel section — ~100 lines
  added, mirrors §7.2 EAM structure)
- [included] pre-impl + session reports

## Out of scope
- [excluded] Bit-exact gate (T8.7 — separate PR carrying the full gate)
- [excluded] MixedFast kernel (T8.9)
- [excluded] T6 full-scale fixture diff (T8.10)

## Mandatory invariants
- `__restrict__` on all pointer params per master spec §D.16.
- TDMD_NVTX_RANGE wrapping every kernel launch per gpu/SPEC §9 (T6.11 audit
  discipline).
- Ownership: snap_gpu writes forces into GPU buffer; no state mutation beyond
  that (CLAUDE.md ownership invariant).
- Reference path preserved: `Fp64ReferenceBuild + Reference ExecProfile` MUST
  produce bitwise-identical results to CPU (D-M6-7 chain; enforced in T8.7).
- NVE drift ≤ 1e-12 rel over 100 steps on 64-atom W (sanity; full T6 is T8.10).

## Required files
- `src/gpu/include/tdmd/gpu/snap_gpu.hpp`
- `src/gpu/snap_gpu.cu`
- `src/gpu/snap_gpu_bispectrum.{cuh,cu}`
- `tests/gpu/test_snap_gpu_functional.cpp`
- `src/gpu/CMakeLists.txt` (edit)
- `docs/specs/gpu/SPEC.md` §7.3 (new subsection)
- `TDMD_Engineering_Spec.md` Приложение C — T8.6 addendum

## Required tests
- `test_snap_gpu_functional::compiles_and_runs_no_cuda_skip` — self-skips on
  no-CUDA; otherwise single-atom correctness check
- `test_snap_gpu_functional::nve_drift_64atom_w_100step` — rel drift ≤ 1e-12
  sanity check

## Acceptance criteria
- Compiles без warnings on sm_80..sm_120.
- Functional unit tests pass on dev machine.
- Self-skips cleanly on public CI (no-CUDA).
- NVTX audit test (`test_nvtx_audit` from M6) remains green after snap_gpu.cu
  added to src/gpu/ — every kernel launch wrapped.
- M1..M7 regressions green.
- Pre-impl + session reports.
- Human review approval (GPU Engineer + Physics Engineer).
```

---

### T8.7 — SnapPotential GPU FP64 bit-exact gate vs CPU FP64

```
# TDMD Task: GPU SNAP vs CPU SNAP bit-exact gate (D-M6-7 chain extension)

## Context
- Master spec: §13.7 (differential gates), §13.5 (determinism matrix)
- Module SPEC: `docs/specs/gpu/SPEC.md` §7.3 (SNAP GPU kernel — T8.6 landed)
- D-M6-7 byte-exact gate (GPU FP64 Reference ≡ CPU FP64 Reference)
- D-M8-13 M8 byte-exact chain: SNAP GPU ≡ SNAP CPU (Fp64ReferenceBuild/Reference)
- Role: Validation / Reference Engineer
- Depends: T8.6 (GPU FP64 kernel functional)

## Goal
Ship `tests/gpu/test_snap_gpu_bit_exact.cpp` — full bit-exact gate: TDMD SnapPotential
GPU FP64 Reference == TDMD SnapPotential CPU FP64 Reference на T6 tungsten
2048-atom. Gate: ≤ 1e-12 rel per-atom force + ≤ 1e-12 rel PE + byte-compare
total virial (packed canonical float double → binary bit compare). This locks
D-M6-7 extension для SNAP path; parallel chain to M6 EAM GPU bit-exact gate.

## Scope
- [included] `tests/gpu/test_snap_gpu_bit_exact.cpp` — Catch2 test instantiating
  both CPU and GPU SnapPotential with identical inputs, comparing outputs at
  1e-12 rel + bit-level for virial tensor; 2048-atom W fixture (T6 canonical)
- [included] `tests/gpu/CMakeLists.txt` — register test; TDMD_TEST_FIXTURES_DIR
  + TDMD_VERIFY_POTENTIALS_DIR defines
- [included] `verify/threshold_registry.yaml` — populate `snap_gpu_vs_cpu_bit_exact`
  entries (rel 1e-12, bit-exact-virial: true)
- [included] `docs/specs/gpu/SPEC.md` §7.3 change log entry — "T8.7 bit-exact
  gate shipped 2026-MM-DD"
- [included] pre-impl + session reports

## Out of scope
- [excluded] MixedFast precision gate (T8.9 — different threshold budget)
- [excluded] T6 scaling runs (T8.11 territory)
- [excluded] Multi-rank SNAP GPU test (T8.10 scope)

## Mandatory invariants
- D-M6-7 byte-exact gate chain extended: SNAP GPU ≡ SNAP CPU (Fp64Reference
  Reference). M8 byte-exact chain (D-M8-13) locked.
- Canonical Kahan-ring reduction order preserved (per D-M5-9) — GPU and CPU
  both consume atoms in canonical atom-ID order.
- Fixture uses same `.snap` coefficient path resolved via T8.2 submodule.

## Required files
- `tests/gpu/test_snap_gpu_bit_exact.cpp` (new)
- `tests/gpu/CMakeLists.txt` (edit)
- `verify/threshold_registry.yaml` (edit)
- `docs/specs/gpu/SPEC.md` (edit — change log entry)
- `TDMD_Engineering_Spec.md` Приложение C — T8.7 addendum

## Required tests
- `test_snap_gpu_bit_exact::t6_tungsten_2048_gpu_eq_cpu` — per-atom force 1e-12
  rel + total PE 1e-12 rel + virial byte-exact

## Acceptance criteria
- Bit-exact gate met на T6 fixture (dev machine).
- Self-skips cleanly on no-CUDA CI hosts (exit 4).
- M1..M7 regressions green; D-M6-7 chain extended.
- Pre-impl + session reports.
- Human review approval (Validation Engineer primary).
```

---

### T8.8 — MixedFastSnapOnlyBuild — new BuildFlavor formal §D.17 procedure

```
# TDMD Task: Ship new BuildFlavor `MixedFastSnapOnlyBuild` per §D.17 7-step

## Context
- Master spec: §D.11 (BuildFlavor matrix), §D.17 (7-step new flavor procedure)
- Module SPEC: `docs/specs/potentials/SPEC.md` §4a (MixedFastSnapOnly prep —
  T8.3 landed)
- D-M8-4 (heterogeneous precision: SNAP=FP32 force, EAM=FP64 force, State=FP64)
- D-M8-10 (third active BuildFlavor)
- Role: Architect / Spec Steward + Physics Engineer + Validation Engineer (joint)
- Depends: T8.7 (GPU SNAP FP64 bit-exact gate landed — baseline для MixedFast
  diff)

## Goal
Ship new BuildFlavor `MixedFastSnapOnlyBuild` через **full §D.17 7-step formal
procedure**. This is a **SPEC delta PR** per playbook §9.1 — only spec changes,
no code. Implementation kernel lands in T8.9. The 7 steps:

1. **Formal rationale doc** — `docs/specs/potentials/mixed_fast_snap_only_rationale.md`
   explaining why this flavor earns keep (SNAP fit RMSE ≈ 1e-3 eV/atom >> FP32
   ULP; FP32 SNAP force is scientifically indistinguishable from FP64;
   EAM tabulated potentials still require FP64 Horner stability per D-M6-8);
   empirical data from T6.8 / T7.0 cited.
2. **Master spec §D.11 matrix update** — add MixedFastSnapOnlyBuild row:
   StateReal=double, ForceReal=float (SNAP) / double (EAM), ExecProfile=Production
   default, CI presence: compile-only.
3. **Threshold registry update** — `verify/threshold_registry.yaml` adds:
   `snap_gpu_mixed_fast_vs_reference`: per-atom force ≤ 1e-5 rel, PE ≤ 1e-7 rel,
   dense-cutoff mode (mirrors D-M6-8 dense-cutoff EAM budget). EAM thresholds
   inherited from MixedFastBuild.
4. **CMake option** — `TDMD_BUILD_FLAVOR=MixedFastSnapOnlyBuild` registered;
   `cmake/tdmd_gpu_profiles.cmake` extended; GPU target link libraries split
   by potential (snap_mixed.o vs eam_alloy.o FP64).
5. **Slow-tier VerifyLab pass placeholder** — §D.17 step 5 mandates full
   regression pass; recorded as "pending T8.12" в SPEC but flagged as **hard
   gate before M8 closure**.
6. **Scientist docs update** — `docs/user/build_flavors.md` new section: when
   to use MixedFastSnapOnlyBuild (pure SNAP workloads on GPU; heterogeneous
   SNAP+EAM runs where SNAP dominates cost; NOT for pure EAM — use MixedFastBuild
   instead).
7. **Architect + Validation Engineer review** — two-reviewer signoff mandate
   (master spec §D.17 final step); signoff recorded в PR description.

## Scope
- [included] `TDMD_Engineering_Spec.md` §D.11 — add row to BuildFlavor matrix
- [included] `docs/specs/potentials/mixed_fast_snap_only_rationale.md` — new file,
  full formal rationale (~200 lines, cites empirical data)
- [included] `docs/specs/potentials/SPEC.md` §4a — populated с finalized interface
  (promoted from "upcoming per T8.8" marker)
- [included] `verify/threshold_registry.yaml` — populate MixedFastSnapOnly
  entries
- [included] `CMakeLists.txt` + `cmake/tdmd_build_flavor.cmake` — register new
  flavor option
- [included] `cmake/tdmd_gpu_profiles.cmake` — heterogeneous precision routing
- [included] `docs/user/build_flavors.md` — scientist-facing explanation
- [included] `TDMD_Engineering_Spec.md` Приложение C — T8.8 addendum (SPEC delta
  marker)
- [included] pre-impl + session reports (with rationale doc excerpt)

## Out of scope
- [excluded] Implementation code (T8.9 — separate PR)
- [excluded] VerifyLab pass actual run (T8.12 territory)
- [excluded] T6 scaling measurement (T8.11)

## Mandatory invariants
- §D.17 7 steps all addressed in this PR (steps 5/7 satisfied by T8.12/human
  review; steps 1-4/6 fully materialized в this PR's diff).
- No code changes (SPEC-only per playbook §9.1).
- Existing BuildFlavors unchanged (Fp64ReferenceBuild + MixedFastBuild
  invariants preserved).
- Rationale cites empirical data (not hand-wave "FP32 is cheaper" — needs SNAP
  fit RMSE + D-M6-8 EAM precision evidence as documented justification).
- Threshold budget matches D-M6-8 dense-cutoff analog (1e-5/1e-7/5e-6 style —
  M9+ sparse-cutoff deferred per memory `project_fp32_eam_ceiling.md`).

## Required files
- `TDMD_Engineering_Spec.md` §D.11 (edit) + Приложение C (edit)
- `docs/specs/potentials/mixed_fast_snap_only_rationale.md` (new)
- `docs/specs/potentials/SPEC.md` §4a (edit)
- `verify/threshold_registry.yaml` (edit)
- `CMakeLists.txt` (edit — flavor option)
- `cmake/tdmd_build_flavor.cmake` (edit)
- `cmake/tdmd_gpu_profiles.cmake` (edit)
- `docs/user/build_flavors.md` (edit)

## Required tests
- None (SPEC-only PR). T8.9 lands implementation + tests; T8.12 slow-tier pass.

## Acceptance criteria
- §D.17 7 steps materialized per above (steps 1/2/3/4/6 delivered в this PR;
  steps 5/7 tracked as T8.12/human review).
- Markdown lint green.
- CMake configures cleanly with `TDMD_BUILD_FLAVOR=MixedFastSnapOnlyBuild`
  (compilation deferred — no code exists yet — but configure succeeds; dummy
  target verifies flavor is registered).
- Human review approval (Architect + Validation Engineer joint — §D.17 step 7
  mandate).
```

---

### T8.9 — SnapPotential MixedFast kernel (GPU FP32 force, state FP64)

```
# TDMD Task: SnapPotential MixedFast GPU kernel — throughput path

## Context
- Master spec: §D.11 (MixedFastSnapOnly flavor — T8.8 landed)
- Module SPEC: `docs/specs/gpu/SPEC.md` §7.3 (SNAP GPU FP64 — T8.6/T8.7 landed);
  extended с §7.3a для MixedFast kernel
- D-M8-8 MixedFast vs GPU FP64 ≤ 1e-5 rel force / ≤ 1e-7 rel PE (dense-cutoff)
- Role: GPU / Performance Engineer + Physics Engineer (joint)
- Depends: T8.8 (flavor SPEC delta landed)

## Goal
Implement `src/gpu/snap_gpu_mixed.cu` — SNAP kernel с `ForceReal=float`;
StateReal=double preserved; bispectrum basis computed в FP32 (saves memory
bandwidth + register pressure vs FP64); force accumulator FP64 (Kahan-ring
friendly even when ops are FP32 — matches MixedFastBuild EAM philosophy from
M6). Kernel emits identical force + PE gate check against FP64 GPU oracle
(T8.6/T8.7 output) — ≤ 1e-5 rel force / ≤ 1e-7 rel PE on T6 tungsten 2048-atom
fixture.

## Scope
- [included] `src/gpu/snap_gpu_mixed.cu` — MixedFast SNAP kernel; FP32 bispectrum;
  FP64 accumulator; NVTX wrapping (NVTX audit test must remain green)
- [included] `src/gpu/snap_gpu_mixed_bispectrum.{cuh,cu}` — FP32 basis functions
  (mirrors snap_gpu_bispectrum but narrower types)
- [included] `src/gpu/snap_gpu_adapter.{hpp,cu}` — BuildFlavor dispatch:
  MixedFastSnapOnlyBuild routes SNAP calls к mixed kernel, EAM calls unchanged
  (FP64 path from M6); MixedFastBuild routes SNAP calls к FP64 kernel (fallback
  per D-M8-4 policy); Fp64Reference stays FP64
- [included] `tests/gpu/test_snap_mixed_fast_within_threshold.cpp` — T6 fixture
  diff vs FP64 GPU oracle; gates 1e-5 rel force + 1e-7 rel PE
- [included] CMake — register new TU under flavor MixedFastSnapOnlyBuild
- [included] `docs/specs/gpu/SPEC.md` §7.3a (new subsection — MixedFast SNAP
  kernel semantics)
- [included] pre-impl + session reports

## Out of scope
- [excluded] Multi-rank Pattern 2 SNAP runs (T8.11 territory)
- [excluded] CPU MixedFast SNAP (decision D-M8-4: CPU-side MixedFast не pays
  back — FP64 CPU remains canonical)
- [excluded] Full T6 scaling measurement (T8.11)

## Mandatory invariants
- `__restrict__` on pointer params per §D.16.
- TDMD_NVTX_RANGE wrapping every kernel launch (NVTX audit test T6.11 remains
  green with snap_gpu_mixed.cu added).
- StateReal=double invariant preserved (CLAUDE.md §architectural invariants).
- EAM kernels unchanged under MixedFastSnapOnlyBuild (verified by re-running
  T6.8a `test_eam_mixed_fast_within_threshold` with new flavor).
- Reference path not degraded: Fp64ReferenceBuild + Reference profile still
  produces FP64 byte-exact output (auto-reject pattern preserved).

## Required files
- `src/gpu/snap_gpu_mixed.cu` (new, ~400 LOC)
- `src/gpu/snap_gpu_mixed_bispectrum.{cuh,cu}` (new)
- `src/gpu/snap_gpu_adapter.{hpp,cu}` (new — flavor-dispatched adapter)
- `tests/gpu/test_snap_mixed_fast_within_threshold.cpp` (new)
- `src/gpu/CMakeLists.txt` (edit)
- `tests/gpu/CMakeLists.txt` (edit)
- `docs/specs/gpu/SPEC.md` §7.3a (new subsection)
- `TDMD_Engineering_Spec.md` Приложение C — T8.9 addendum

## Required tests
- `test_snap_mixed_fast_within_threshold::t6_tungsten_2048_mixed_vs_fp64_gpu` —
  per-atom force rel ≤ 1e-5; PE rel ≤ 1e-7
- `test_nvtx_audit` continues green (new SNAP mixed kernels wrapped)
- `test_eam_mixed_fast_within_threshold` green under new flavor (EAM unchanged
  verification)

## Acceptance criteria
- MixedFast SNAP kernel meets D-M8-8 dense-cutoff gate on T6 fixture.
- EAM precision preserved under MixedFastSnapOnlyBuild (verify EAM diff test).
- NVTX audit green.
- M1..M7 regressions + T8.5 CPU oracle + T8.7 GPU FP64 bit-exact all green.
- Pre-impl + session reports.
- Human review approval.
```

---

### T8.10 — T6 tungsten SNAP fixture + integration test

```
# TDMD Task: T6 tungsten SNAP benchmark — canonical verify fixture integration

## Context
- Master spec: §14 M8 (T6 tungsten benchmark); verify/SPEC T6 description
- Module SPEC: `docs/specs/verify/SPEC.md` (T6 definition populated T8.3)
- D-M8-9 T6 fixture scales (2048-atom standard + 16384-atom scaling)
- Role: Validation / Reference Engineer + Physics Engineer
- Depends: T8.9 (MixedFast kernel landed — all three flavors now runnable)

## Goal
Register T6 tungsten SNAP как official verify benchmark target. Fixture config:
8×8×8 BCC W (2048 atoms) + W_2940_2017_2.snap coefficient set (via submodule);
run configs: single-subdomain single-rank (Fp64Reference + MixedFast + MixedFastSnapOnly);
multi-subdomain 2-rank Pattern 2 K=1 (Fp64Reference only — byte-exact chain
extension); 4-rank Pattern 2 K=1 (opportunistic); all gated by D-M8-8 threshold
budget. T6 becomes regression target in CI **from this PR forward** (pre-push
local; compile-only на public CI per Option A).

## Scope
- [included] `verify/benchmarks/t6_snap_tungsten/` directory (scaffold landed
  T8.10a 2026-04-20 — README + checks.yaml + lammps_script.in + threshold
  entries; T8.10 proper adds TDMD config.yaml + run_differential extension):
  - `config.yaml.template` — 2048-atom W BCC, W_2940_2017_2 via submodule path
  - `config_16384.yaml.template` — 16×16×16 scaling variant
  - `checks.yaml` — thresholds (D-M8-8 budget per flavor) [landed T8.10a]
  - `README.md` — T6 description, how to run, what it validates [landed T8.10a]
  - `lammps_script.in` — LAMMPS oracle driver [landed T8.10a]
- [included] `tests/integration/m8_smoke_t6.cpp` — Catch2 integration smoke:
  runs T6 2048-atom Fp64Reference single-subdomain 10-step NVE; energy
  conservation gate 1e-12 rel; self-skips on fixture absence
- [included] `tests/integration/CMakeLists.txt` (edit — register m8 smoke)
- [included] `verify/threshold_registry.yaml` — populate remaining T6 entries
- [included] `docs/specs/verify/SPEC.md` T6 section — mark "shipped T8.10"
- [included] pre-impl + session reports

## Out of scope
- [excluded] Cloud burst scaling runs (T8.11 — separate cloud workflow)
- [excluded] Single-rank performance measurement (T8.11)
- [excluded] TDMD vs LAMMPS comparison (T8.11)

## Mandatory invariants
- Fixture uses canonical W_2940_2017_2.snap via submodule (no binary tracked).
- Self-skips if submodule not initialized (clean dev experience).
- Energy conservation gate ≤ 1e-12 rel on 10-step Fp64Reference run.
- M1..M7 + M8 T8.4..T8.9 all green; T6 fixture doesn't break existing gates.

## Required files
- `verify/benchmarks/t6_snap_tungsten/config.yaml.template` (new)
- `verify/benchmarks/t6_snap_tungsten/config_16384.yaml.template` (new)
- `verify/benchmarks/t6_snap_tungsten/checks.yaml` (new — **landed T8.10a**)
- `verify/benchmarks/t6_snap_tungsten/README.md` (new — **landed T8.10a**)
- `verify/benchmarks/t6_snap_tungsten/lammps_script.in` (new — **landed T8.10a**)
- `tests/integration/m8_smoke_t6.cpp` (new)
- `tests/integration/CMakeLists.txt` (edit)
- `verify/threshold_registry.yaml` (edit)
- `docs/specs/verify/SPEC.md` (edit)
- `TDMD_Engineering_Spec.md` Приложение C — T8.10 addendum

## Required tests
- `m8_smoke_t6::t6_tungsten_2048_fp64_nve_10step` — energy conservation 1e-12 rel

## Acceptance criteria
- T6 fixture resolves via submodule; self-skip works without submodule.
- 10-step NVE energy conservation met на Fp64Reference.
- M1..M7 + T8.4..T8.9 regressions green.
- Pre-impl + session reports.
- Human review approval.
```

---

### T8.11 — TDMD vs LAMMPS SNAP scaling benchmark (CLOUD BURST GATE)

```
# TDMD Task: TDMD vs LAMMPS SNAP scaling on ≥ 8 ranks — master spec §14 artifact gate

## Context
- Master spec: §14 M8 artifact gate (≥ 20% beat OR honest documentation)
- Module SPEC: `docs/specs/verify/SPEC.md` T6 scaling runs
- D-M8-5 (cloud burst); D-M8-6 (artifact gate interpretation)
- Memory `project_m1_complete.md` T7.8b carry-forward: 30% runtime measurement
  bundled into T8.11
- Role: GPU / Performance Engineer + Validation / Reference Engineer + Scientist
  UX Engineer (joint — this is the milestone deliverable)
- Depends: T8.10 (T6 fixture official)

## Goal
Execute cloud burst scaling campaign comparing TDMD SNAP vs LAMMPS SNAP on T6
tungsten. Cloud config: AWS p4d.24xlarge (8× A100 SXM4-40GB, NVLink, 400 Gbps
EFA) OR equivalent. Matrix:
- Weak scaling: 2048/rank held constant; 1 → 2 → 4 → 8 ranks (+16 opportunistic).
- Strong scaling: 16384 atoms; 1 → 2 → 4 → 8 ranks.
- Builds: MixedFastSnapOnlyBuild (TDMD throughput) vs LAMMPS default (FP64 SNAP,
  USER-SNAP package).
- Deliverable: `verify/benchmarks/t6_snap_scaling/results_<date>.json` with
  wall-clock per step, throughput (atom-steps/sec), parallel efficiency, memory
  bandwidth, NVTX breakdown; same-hardware fair comparison.
- **Bundled**: T7.8b runtime 30% overlap measurement — runs 2-rank K=4 pipelined
  vs sync on same cloud hardware (test_overlap_budget_2rank from T8.0 — infrastructure
  already shipped; this is where real ≥ 2 GPU measurement lands).

## Scope
- [included] `verify/benchmarks/t6_snap_scaling/` directory:
  - `harness.py` — cloud orchestration (AWS boto3 or equivalent; provisions p4d,
    installs TDMD + LAMMPS, runs matrix, collects results); local-only tool
  - `config_weak_*.yaml` — weak scaling configs (1/2/4/8/16 rank)
  - `config_strong_*.yaml` — strong scaling configs
  - `run_lammps_oracle.sh` — parallel LAMMPS invocation (via ompi with same rank
    count; same T6 fixture)
  - `results_<date>.json` — result artifact (checked into repo — small, no binary)
  - `REPORT.md` — artifact gate write-up: raw numbers, methodology, honest
    comparison, does TDMD beat ≥ 20%? if not, why not + what to do next
- [included] `verify/benchmarks/t7_8b_overlap_runtime/` — 2-rank K=4 overlap
  measurement using test_overlap_budget_2rank from T8.0; results recorded
- [included] `docs/specs/verify/SPEC.md` T6 scaling section populated с real numbers
- [included] Master spec Приложение C — T8.11 addendum (biggest single addendum
  of M8; records methodology + result + artifact gate outcome)
- [included] pre-impl + session reports

## Out of scope
- [excluded] Self-hosted GPU runner setup (D-M6-6 policy: cloud burst only)
- [excluded] Long-term cloud CI integration (future M10+ topic)
- [excluded] MPI fabric tuning beyond default (engineering budget limit)

## Mandatory invariants
- **Honest engineering** (D-M8-6): measurement records actual wall-clock;
  methodology disclosed (exact LAMMPS build options, same hardware, same fixture,
  same seed where applicable); cherry-picking config is auto-reject.
- Artifact gate outcome **documented explicitly** in REPORT.md:
  - Case A: TDMD beat LAMMPS ≥ 20% → records where, by what margin, under what
    conditions; M8 closure proceeds.
  - Case B: TDMD did NOT beat LAMMPS ≥ 20% → records by what margin TDMD either
    lost or partially won; explicit "what to do next" roadmap для M9+ (MEAM/PACE
    potentials where TD architecture should shine more; long-range support via
    M9 Ewald service; better inter-node overlap via M10 async halo).
  - Both outcomes close M8 per master spec §14 (the dishonest outcome — shipping
    a "20% beat" claim without the measurement actually landing — is not).
- T7.8b 30% overlap ratio measured on real ≥ 2 GPU hardware; result checked in.

## Required files
- `verify/benchmarks/t6_snap_scaling/harness.py` (new)
- `verify/benchmarks/t6_snap_scaling/config_*.yaml` (new, ~10 configs)
- `verify/benchmarks/t6_snap_scaling/run_lammps_oracle.sh` (new)
- `verify/benchmarks/t6_snap_scaling/results_<YYYY-MM-DD>.json` (new; artifact)
- `verify/benchmarks/t6_snap_scaling/REPORT.md` (new — the artifact gate doc)
- `verify/benchmarks/t7_8b_overlap_runtime/results_<YYYY-MM-DD>.json` (new)
- `verify/benchmarks/t7_8b_overlap_runtime/REPORT.md` (new)
- `docs/specs/verify/SPEC.md` (edit)
- `TDMD_Engineering_Spec.md` Приложение C — T8.11 addendum

## Required tests
- None per se (this task **is** the measurement). But:
  - CI check: harness.py passes `python -m py_compile` (syntactic)
  - Local pre-push check: harness.py can be invoked с `--dry-run` без real cloud
    spend

## Acceptance criteria
- Cloud burst run executed; raw JSON + REPORT.md checked in.
- Artifact gate outcome documented explicitly (A or B per above).
- T7.8b overlap ≥ 30% measured on real 2-GPU hardware (if cloud config supports;
  else documented как "not achievable on current cloud config — retry M10+").
- M1..M7 + M8 T8.0..T8.10 regressions green.
- Pre-impl + session reports.
- Human review approval (Architect + Scientist UX Engineer — milestone
  deliverable review).
```

---

### T8.12 — Slow-tier VerifyLab full pass for MixedFastSnapOnlyBuild

```
# TDMD Task: Slow-tier VerifyLab regression pass for new flavor

## Context
- Master spec: §D.17 step 5 (slow-tier pass mandate)
- Module SPEC: `docs/specs/verify/SPEC.md` (VerifyLab infrastructure)
- D-M8-10 (MixedFastSnapOnlyBuild third active flavor)
- Role: Validation / Reference Engineer
- Depends: T8.11 (scaling measurement complete — gives us empirical confidence
  before slow-tier sign-off)

## Goal
Run full slow-tier VerifyLab regression battery для MixedFastSnapOnlyBuild.
VerifyLab slow-tier includes:
- T0 anchor (trivial 2-atom LJ — not SNAP, but tests flavor build + link)
- T1 (Ni-Al EAM differential — EAM must still work unchanged under new flavor)
- T3 (10^6 atom Al FCC scaling — EAM regression check at scale)
- T4 (Ni-Al EAM 1000-step NVE drift — long-run precision)
- T6 (tungsten SNAP — new flavor's primary target)
- Full differential harness T1+T4+T6 against CPU FP64 oracles

## Scope
- [included] `verify/slow_tier/m8_mixed_fast_snap_only_sweep.yaml` — VerifyLab
  campaign config (matrix: BuildFlavor × ExecProfile × fixture)
- [included] `verify/slow_tier/m8_mixed_fast_snap_only_results.json` — result
  artifact checked in (small JSON)
- [included] `verify/slow_tier/m8_mixed_fast_snap_only_REPORT.md` — analysis:
  all gates met, any surprises, precision ceiling notes
- [included] `docs/specs/potentials/mixed_fast_snap_only_rationale.md` — append
  "Slow-tier pass: shipped T8.12 [date]; all gates green" section
- [included] Master spec §D.11 — mark MixedFastSnapOnlyBuild "slow-tier validated"
- [included] Master spec Приложение C — T8.12 addendum
- [included] pre-impl + session reports

## Out of scope
- [excluded] Any new test creation (this task uses existing T0/T1/T3/T4/T6
  infrastructure)
- [excluded] Fixture changes (uses T8.10 T6 official fixture)
- [excluded] Cloud burst (T8.11 already ran scaling; this task runs on dev
  hardware — slow-tier is about coverage breadth, not scale)

## Mandatory invariants
- All D-M8-8 threshold gates met for new flavor.
- EAM MixedFast unchanged: D-M6-8 dense-cutoff gate still passes (regression
  check).
- D-M6-7 byte-exact gate preserved for Reference path (new flavor doesn't
  regress reference).
- §D.17 step 5 satisfied: explicit slow-tier report in commit history.

## Required files
- `verify/slow_tier/m8_mixed_fast_snap_only_sweep.yaml` (new)
- `verify/slow_tier/m8_mixed_fast_snap_only_results.json` (new)
- `verify/slow_tier/m8_mixed_fast_snap_only_REPORT.md` (new)
- `docs/specs/potentials/mixed_fast_snap_only_rationale.md` (edit — append)
- `TDMD_Engineering_Spec.md` §D.11 (edit) + Приложение C

## Required tests
- None; this task **runs** existing tests and records results.

## Acceptance criteria
- All slow-tier gates green for MixedFastSnapOnlyBuild.
- §D.17 step 5 satisfied.
- Report checked in.
- M1..M7 + M8 T8.0..T8.11 regressions green.
- Pre-impl + session reports.
- Human review approval (Validation Engineer + Architect — §D.17 step 7
  reviewer set).
```

---

### T8.13 — M8 integration smoke + v1 alpha tag

```
# TDMD Task: M8 smoke test + git tag v1.0.0-alpha1

## Context
- Master spec: §14 M8 (closure + v1 alpha tag)
- Module SPEC: regression gate (playbook §3 pre-push protocol)
- D-M8-15 (tag format `v1.0.0-alpha1`)
- Role: Architect / Spec Steward + Core Runtime Engineer
- Depends: T8.12 (slow-tier pass — last hard gate)

## Goal
Ship M8 integration smoke test + tag v1 alpha. Integration smoke: multi-rank
T6 SNAP Pattern 2 K=1 on 2-subdomain 2-rank MixedFastSnapOnly, byte-exact to
single-subdomain single-rank Fp64Reference (extends D-M7-10 byte-exact chain
**parallel to** SNAP path — M8 byte-exact chain per D-M8-13). Finalize change
log, release notes, git tag `v1.0.0-alpha1` annotated.

## Scope
- [included] `tests/integration/m8_smoke/run_m8_smoke.sh` — integration smoke
  script (mirrors M7 `run_m7_smoke.sh` pattern; parameterized by `TDMD_CLI_BIN`)
- [included] `tests/integration/m8_smoke/expected_thermo.json` — golden thermo
  output for byte-exact comparison
- [included] `.github/workflows/ci.yml` — add `m8-smoke` job (local equivalence
  check; SKIP on public CI per Option A but compile-only verification)
- [included] `CHANGELOG.md` (new file) — release notes v1.0.0-alpha1; enumerates
  M1..M8 acceptance gates met; links to T6 scaling REPORT.md and T7 mixed-scaling
  REPORT.md; explicit artifact gate outcome (A or B from T8.11)
- [included] Git tag: `git tag -a v1.0.0-alpha1 -m "v1 alpha — M8 closure,
  SNAP proof-of-value, TD×SD hybrid on ML kernel"`; push tag
- [included] Master spec §14 — mark M8 CLOSED with date
- [included] Master spec Приложение C — T8.13 addendum + M8 closure summary
- [included] `docs/development/m8_execution_pack.md` §5 — mark all T8.0..T8.13
  `[x]`; status CLOSED
- [included] pre-impl + session reports

## Out of scope
- [excluded] M9 execution pack (separate future PR)
- [excluded] Distribution packaging (deb/rpm/docker — M10+ scope)
- [excluded] Public binary release (M11 beta / M13 1.0.0 scope)

## Mandatory invariants
- D-M8-13 byte-exact chain: SNAP Pattern 2 K=1 2-rank Fp64Reference thermo ==
  single-rank Fp64Reference thermo (byte-for-byte).
- D-M7-10 byte-exact chain preserved: M3 ≡ M4 ≡ M5 ≡ M6 ≡ M7 Pattern 2 K=1 for
  EAM path (unchanged by M8).
- All M1..M7 + M8 T8.0..T8.12 regressions green.
- Tag annotated (not lightweight); pushed to origin; release notes reference tag.

## Required files
- `tests/integration/m8_smoke/run_m8_smoke.sh` (new)
- `tests/integration/m8_smoke/expected_thermo.json` (new)
- `tests/integration/m8_smoke/CMakeLists.txt` (new) — register smoke test
- `.github/workflows/ci.yml` (edit) — m8-smoke job
- `CHANGELOG.md` (new)
- `TDMD_Engineering_Spec.md` §14 (edit — mark M8 CLOSED) + Приложение C
- `docs/development/m8_execution_pack.md` (edit — §5 closure)

## Required tests
- `m8_smoke::t6_snap_pattern2_2rank_byte_exact_to_single_rank` — byte-exact
  gate on T6 fixture 2-rank Pattern 2 K=1 Fp64Reference thermo
- Full pre-push regression: M1..M7 + M8 smokes + T1/T4/T6 differentials +
  T3-gpu anchor + T6 tungsten

## Acceptance criteria
- M8 smoke green locally.
- Git tag pushed to origin.
- CHANGELOG.md covers M1..M8 gates and references result artifacts.
- CI green on public (compile-only M8 job; smoke is local).
- Pre-impl + session reports.
- Human review approval on release notes + tag annotation.
- **M8 status: CLOSED** when all of the above met.
```

---

## 5. M8 Acceptance Gate

**Master spec §14 M8 — verbatim artifact gate:**

> Artifact gate: на T6 TDMD либо обгоняет LAMMPS ≥ 20 % на целевой
> конфигурации (≥ 8 ranks, commodity network), либо проект документирует,
> почему не обгоняет и что делать дальше.

**Checklist (all must be `[x]` before M8 closure):**

- [x] **T8.0** — T7.8b carry-forward closed: 2-rank overlap gate infrastructure
  shipped 2026-04-20; runtime 30% measurement deferred to T8.11 cloud burst
  (hardware prerequisite: ≥ 2 GPU).
- [x] **T8.1** — M8 execution pack authored (this document). Shipped commit
  5ed72d2 (2026-04-20); D-M8-2/D-M8-3 factual corrections shipped T8.1b
  commit 0c84b68 (LAMMPS oracle already landed M1 T1.11, pin
  `stable_22Jul2025_update4`, canonical fixture `W_2940_2017_2.snap`).
- [x] **T8.2** — LAMMPS SNAP subset verified (ML-SNAP built via
  `tools/build_lammps.sh`; `lmp -h | grep ML-SNAP` reports present;
  `in.snap.W.2940` example runs cleanly in 1.2 s matching upstream
  `log.15Jun20.snap.W.2940.g++.1` byte-exactly to 5-decimal precision).
  Canonical T6 fixture = `W_2940_2017_2.snap` documented in
  `verify/third_party/lammps_README.md` (SNAP fixture section) and
  `docs/specs/verify/SPEC.md` §4.7 (new). Path-resolution Catch2 gate landed:
  `tests/potentials/test_lammps_oracle_snap_fixture` (two test cases,
  `SKIP_RETURN_CODE 77` on uninitialized submodule). Shipped 2026-04-20.
- [x] **T8.3** — potentials/SPEC **§6** (not §4 — §6 is SNAP; §4 is EAM)
  SnapPotential body authored (2026-04-20): interface contract (`SnapParams`,
  `SnapSpecies`, `SnapData`, `SnapPotential final`) finalized; three-pass
  force evaluation algorithm with LAMMPS USER-SNAP attribution chain;
  parameter file format (`.snap` + `.snapcoeff` + `.snapparam`); §6.7
  precision policy + MixedFastSnapOnly placeholder referring to T8.8 §D.17;
  §6.8 GPU kernel strategy; §6.9 validation matrix with D-M8-7 / D-M8-8
  threshold registry anchors; §8.7 cross-ref back to §6.7 + §D.17. Pure
  SPEC delta per playbook §9.1; ~270 lines of new content.
- [x] **T8.4a** — SNAP types + LAMMPS-format parser + SnapPotential skeleton
  (2026-04-20): `SnapParams` (11 hyperparameters), `SnapSpecies`, `SnapData`
  (with derived `k_max`, symmetric `rcut_sq_ab` n×n, FNV-1a checksum) +
  `parse_snap_coeff` / `parse_snap_param` / `parse_snap_files` (path:line:
  diagnostics, `chemflag=1` + `switchinnerflag=1` rejected с M9+ message),
  `snap_k_max(twojmax)` matching LAMMPS `SNA::compute_ncoeff` (verified 1 / 5 /
  14 / 30 / 55 / 91 / 140 for twojmax = 0 / 2 / 4 / 6 / 8 / 10 / 12; k_max = 55
  cross-checks с W_2940_2017_2 fixture header declaration of 56 coefficients).
  `SnapPotential final : Potential` constructor validates data consistency +
  throws `std::invalid_argument` on mismatch; `compute()` throws
  `std::logic_error` deferring force body port to T8.4b. 13-case / 51-assertion
  Catch2 suite (`test_snap_file`) green, auto-skip 77 on uninitialized
  submodule. Full ctest bank (45/45 passed) clean — no regressions.
- [x] **T8.4b** — SnapPotential force body port landed (2026-04-20):
  verbatim port of LAMMPS USER-SNAP `sna.cpp` (1597 lines → `snap/sna_engine.cpp`
  ~1260 lines + `snap/sna_engine.hpp` ~200 lines) + `pair_snap.cpp::compute()`
  outer loop (→ `snap.cpp` ~260 lines). Three-pass bispectrum → energy → force
  evaluator (Pass 1 compute_ui + add_uarraytot U-matrix accumulation; Pass 2
  compute_yi contraction with β; Pass 3 compute_duidrj + compute_deidrj per-
  neighbour). FP summation ordering preserved verbatim из upstream (load-
  bearing для D-M8-7 byte-exact at T8.5 и MixedFastSnapOnly policy swap at
  T8.9). LAMMPS GPLv2 attribution block reproduced in every new file в
  `src/potentials/snap/`. Half-list → full-list bridge built inside
  `SnapPotential::compute()`: CSR scatter + per-atom sort makes TDMD's
  `newton on` NeighborList feed LAMMPS's `REQ_FULL`-style outer loop without
  touching the neighbor/ module. Virial sign matches EAM convention
  (`Σ F_i · (r_j − r_i)`) so thermo pressure agrees с LAMMPS через the
  compensating sign в `runtime/simulation_engine.cpp:574`. Four Catch2
  structural tests (`test_snap_compute.cpp`): canonical W fixture load,
  2-atom dimer smoke + N3L + virial symmetry, 250-atom BCC W N3L ≤ 1e-12,
  central-difference F == −dE/dR consistency ≤ 1e-4 absolute. T8.4a
  skeleton-throws test retired. Full ctest bank 46/46 green, no regressions.
  Byte-exact vs LAMMPS (D-M8-7) deferred to T8.5 (run_differential harness
  и LAMMPS oracle run — TDMD-side force-body port is the blocker; T8.5
  unblocked). Depends: T8.4a [x].
- [x] **T8.5** — CPU SNAP differential vs LAMMPS: D-M8-7 byte-exact gate GREEN
  on T6 tungsten 5×5×5 BCC (250 atoms, nrep=5 — CellGrid stencil constraint
  L_axis ≥ 3·(cutoff+skin) forces nrep≥5 на SNAP cutoff 4.73442 Å + skin 0.3;
  scaling to 2048-atom deferred to T8.10). Harness landed: committed
  `verify/benchmarks/t6_snap_tungsten/{generate_setup.py,setup.data,lammps_script_metal.in,config_metal.yaml}`
  (pure SNAP path, no ZBL — byte-exact gate scoped standalone; ZBL pair lands
  M9+). Driver: `verify/t6/run_differential.py` (clone of T4 с snap-specific
  `extra_absolute_paths` rewrite for coeff_file/param_file and snap_dir
  -var pass-through). Catch2 wrapper `verify/t6/test_t6_differential.cpp`
  hooked into ctest (test #39). SNAP dispatch path plumbed end-to-end через
  `io::PotentialStyle::Snap` (yaml_config, preflight, SimulationEngine,
  explain/validate CLI). Measured на 100-step NVE W 250 atoms: thermo
  PE/KE/EtotaL max_rel = 0 (exact bytes), Temp max_rel 1.13e-6 (k_B definitional,
  T1/T4 precedent budget 2e-6), Press max_rel 3.2e-11 (budget 1e-10); forces
  max_rel = **8.795e-13** под budget 1e-12 — D-M8-7 closed с decade headroom.
  ctest 39/39 green. Depends: T8.4b [x].
- [x] **T8.6a** — SnapPotential GPU FP64 **scaffolding** landed 2026-04-20:
  `src/gpu/include/tdmd/gpu/snap_gpu.hpp` (PIMPL mirror `EamAlloyGpu` shape) +
  `src/gpu/snap_gpu.cu` (single-TU `#if TDMD_BUILD_CUDA` branches: CUDA →
  `std::logic_error("T8.6b kernel body not landed")` wrapped в
  `TDMD_NVTX_RANGE("snap.compute_stub")`; CPU-only → `std::runtime_error`; NO
  separate stub.cpp; zero `<<<...>>>` launches — `test_nvtx_audit` trivially
  passes) + `src/potentials/include/tdmd/potentials/snap_gpu_adapter.hpp` +
  `src/potentials/snap_gpu_adapter.cpp` (domain facade, ctor validates
  n_species>0 / rcutfac>0 / M8-scope flag fence chemflag/quadraticflag/
  switchinnerflag parity с CPU `SnapPotential`, flattens radius_elem /
  weight_elem / β coefficients) + `src/runtime/*simulation_engine.*` (new
  parallel field `gpu_snap_potential_`, `init()` switches via `dynamic_cast`,
  `recompute_forces()` branches accordingly) + `src/io/preflight.cpp::check_runtime`
  relaxed (`runtime.backend=gpu` accepts `potential.style ∈ {eam/alloy, snap}`),
  CMake wiring, `tests/gpu/test_snap_gpu_plumbing.cpp` (4 Catch2 cases:
  constructs cleanly on W_2940 fixture, rejects all three M8-scope flags,
  sentinel error path reachable, `compute_version()` stays at 0 before T8.6b;
  self-skips exit 77 on uninitialized LAMMPS submodule). SPEC: `gpu/SPEC.md`
  v1.0.17 §7.5 new + §7.4 rewritten. Master spec Приложение C T8.6a addendum.
  All three CI flavors зелёные (Fp64Reference+CUDA 48/48, MixedFast+CUDA
  48/48, CPU-only-strict 40/40); T8.5 D-M8-7 CPU byte-exact gate без
  регрессии.
- [ ] **T8.6b** — SnapPotential GPU FP64 **kernel body**: full CUDA port
  of LAMMPS USER-SNAP three-pass (Ui / Yi / deidrj; ~1500 lines; Wigner-U
  expansion + bispectrum basis cache + Clebsch-Gordan contraction);
  `__restrict__` per master spec §D.16; `TDMD_NVTX_RANGE("snap.{ui,yi,deidrj}_kernel")`
  per gpu/SPEC §9 audit rules (T6.11); Kahan host-side reduction per D-M6-15;
  LAMMPS GPLv2 attribution block в каждом новом `.cu`/`.cuh`;
  `test_snap_gpu_functional` on W 64-atom fixture. Depends: T8.6a [x].
- [ ] **T8.7** — GPU FP64 bit-exact gate (D-M6-7 chain extension): SNAP GPU ≡
  SNAP CPU @ Fp64Reference+Reference на T6 fixture; per-atom 1e-12 rel + virial
  byte-exact.
- [x] **T8.8** — MixedFastSnapOnlyBuild new BuildFlavor shipped per §D.17
  7-step (2026-04-20):
  (1) Formal rationale doc —
  `docs/specs/potentials/mixed_fast_snap_only_rationale.md` (256 lines, 10
  sections covering need, flavor comparison, empirical SNAP-fit-RMSE vs
  FP32-ULP evidence, threshold derivation, compat matrix, CMake integration,
  T8.12 slow-tier pass obligation, scientist docs pointer, two-reviewer
  signoff mandate, out-of-scope follow-ons).
  (2) Master spec §D.12 compat matrix updated (+ §D.11 "shipped flavor"
  paragraph replacing "roadmap extension"; + §7.1 BuildFlavor table
  extended with SNAP/EAM split columns + B-het philosophy row; + §D.2
  five→six canonical flavors; + §D.13 thresholds column; + §D.14 build
  system rewritten to single-binary `TDMD_BUILD_FLAVOR` model matching
  actual CMake implementation; + §D.15 `tdmd_mixed_snap_only` binary label).
  (3) Threshold registry entries —
  `verify/thresholds/thresholds.yaml` new `benchmarks.gpu_mixed_fast_snap_only`
  section with SNAP force 1e-5 / energy 1e-7, EAM inherited 1e-5/1e-7/5e-6,
  and NVE drift 1e-5/1000 steps; all equal-or-tighter than `gpu_mixed_fast`.
  (4) CMake option — `TDMD_BUILD_FLAVOR=MixedFastSnapOnlyBuild` registered
  in `CMakeLists.txt` cache STRINGS (fourth entry); `cmake/BuildFlavors.cmake`
  `_tdmd_apply_mixed_fast_snap_only` function defines
  `TDMD_FLAVOR_MIXED_FAST_SNAP_ONLY` compile symbol + emits "T8.9 kernel
  split pending" status message. Configures cleanly verified via
  `/tmp/tdmd_snaponly_probe`.
  (5) Slow-tier VerifyLab pass **pending T8.12** — recorded as hard gate
  before M8 closure in rationale doc §7 + this checklist (T8.12 entry).
  (6) Scientist docs — `docs/user/build_flavors.md` new 6-flavor guide
  with decision tree + per-flavor when-to-use/when-not-to + warnings for
  research-only flavors.
  (7) Architect + Validation Engineer joint review **pending PR thread
  signoffs** — two-reviewer mandate recorded in rationale doc §9. Not
  optional per §D.17 step 7.
  SPEC delta PR per playbook §9.1 — no functional code changes. Kernel
  split emission = T8.9 (requires T8.4b SNAP force body port). Flavor
  configures cleanly but emits no heterogeneous code paths yet — until
  T8.9 + T8.12, continue using `MixedFastBuild` for production SNAP runs.
- [ ] **T8.9** — SNAP MixedFast kernel ≤ 1e-5 rel force / ≤ 1e-7 rel PE vs FP64
  GPU oracle на T6 tungsten. EAM MixedFast unchanged under new flavor.
- [x] **T8.10a** — `verify/benchmarks/t6_snap_tungsten/` scaffold landed
  2026-04-20 (commit b1ebfd9): README.md (~115 lines, fixture metadata +
  D-M8-3 path resolution + three-variant table + acceptance-threshold chain
  D-M8-7/D-M6-7/D-M8-8 + oracle subset gate recipe + T8.10a/T8.10/T8.11 status
  checklist); checks.yaml (thermo column map identical to T1/T4 + six checks
  referencing `benchmarks.t6_snap_tungsten.cpu_fp64_vs_lammps.*`);
  lammps_script.in (adapted upstream `in.snap.W.2940` with
  `-var workdir`/`-var nrep`/`-var nsteps`/`-var snap_dir` knobs, %.10e
  thermo + %.16e id-sorted forces dump); threshold-registry block
  `benchmarks.t6_snap_tungsten` with `cpu_fp64_vs_lammps` (forces 1e-12,
  PE/KE/E_total 1e-12, pressure 1e-10, temp 2e-6 k_B floor = D-M8-7 budget),
  plus `gpu_fp64_vs_cpu_fp64` (forces 1e-12 = D-M6-7 extension). Declarative-
  only — does NOT depend on SnapPotential force body; measurement landing
  remains T8.10 proper (blocks on T8.4b).
- [ ] **T8.10** — T6 tungsten SNAP fixture canonical (TDMD side): config.yaml
  variants + generate_setup.py + run_differential.py SNAP extension + verify/
  SPEC T6 section marked "shipped"; M8 smoke integration test lands (single-
  subdomain 10-step NVE 1e-12 drift). Depends: T8.4b (SnapPotential compute).
- [ ] **T8.11** — TDMD vs LAMMPS SNAP scaling cloud burst executed; REPORT.md
  checked in с honest artifact gate outcome (A: TDMD beat ≥ 20%, OR B:
  documented why not + M9+ roadmap). T7.8b 30% overlap measured on real ≥ 2 GPU.
- [ ] **T8.12** — Slow-tier VerifyLab pass для MixedFastSnapOnlyBuild: T0+T1+T3+T4+T6
  all green; EAM D-M6-8 gate preserved; report checked in. §D.17 step 5
  satisfied.
- [ ] **T8.13** — M8 integration smoke landed; thermo byte-for-byte == single-rank
  Fp64Reference; CHANGELOG.md v1.0.0-alpha1 notes; git tag annotated and pushed.
- [ ] No regressions: M1..M7 smokes + T1/T4 differentials + T3-gpu anchor + M6
  smoke + M7 Pattern 2 smoke + T5/T6/T7 differentials all green.
- [ ] D-M7-10 byte-exact chain preserved: M3 ≡ M4 ≡ M5 ≡ M6 ≡ M7 Pattern 2 K=1
  for EAM path.
- [ ] D-M8-13 byte-exact chain established: SNAP Pattern 1 single-rank
  Fp64Reference+Reference ≡ SNAP Pattern 2 2-rank K=1 Fp64Reference+Reference.
- [ ] D-M8-8 threshold gates all met: CPU vs LAMMPS ≤ 1e-12, GPU FP64 vs CPU
  FP64 ≤ 1e-12, MixedFast vs GPU FP64 ≤ 1e-5/1e-7.
- [ ] CI Pipelines A (lint+build+smokes), B (unit/property), C (differentials),
  D (build-gpu compile-only), new E (build-gpu-snap compile-only for three
  flavors) — all green.
- [ ] Pre-implementation + session reports attached в каждом PR.
- [ ] Human review approval для каждого PR.

**M8 milestone closure criteria** (master spec §14 M8):

- SnapPotential CPU + GPU shipped and byte-exact chain established.
- T6 tungsten SNAP benchmark landed; differential gates green.
- MixedFastSnapOnlyBuild new BuildFlavor shipped per §D.17 formal procedure;
  slow-tier pass documented.
- TDMD vs LAMMPS scaling measurement executed on ≥ 8 ranks cloud burst;
  artifact gate outcome documented (A or B).
- v1.0.0-alpha1 tag pushed.

**M8 status:** IN PROGRESS (2026-04-20). T8.0 LANDED. Next window after M8
closure is M9 (long-range Ewald/PPPM + NVT/NPT Pattern 2 K=1 + MEAM potential
family + Morse GPU kernel — unblocks T3-gpu full dissertation replication).

---

## 6. Risks & Open Questions

**Risks:**

- **R-M8-1 — LAMMPS USER-SNAP port subtle bugs.** Bispectrum coefficient indexing
  is notoriously error-prone (twojmax-dependent multi-dimensional coefficient
  layout). Port preserving algorithmic structure is the mitigation; T8.5
  differential gate (1e-12 rel vs LAMMPS) catches indexing bugs immediately.
  If T8.5 fails to close 1e-12, **do not proceed to T8.6** — indicates CPU
  impl bug; debug via single-atom bispectrum component comparison с LAMMPS
  verbose stdout.
- **R-M8-2 — LAMMPS submodule build heavy.** Stripped-down `liblammps.a` с
  SNAP+MANYBODY+KSPACE packages может be ~200 MB + 2-5 min build time.
  Mitigation: D-M8-2 keeps oracle build local-only (not in public CI); dev
  onboarding README documents one-time cost; `TDMD_ENABLE_LAMMPS_ORACLE=OFF`
  default preserves fast dev iteration.
- **R-M8-3 — MixedFastSnapOnlyBuild precision ceiling на sparse cutoff.**
  D-M6-8 EAM precision memory `project_fp32_eam_ceiling.md` notes FP32 force
  propagation через ~50-neighbor stencil has intrinsic ceiling ~1e-5 rel.
  SNAP может be even tighter cutoff (cost = O(N^3) per atom in bispectrum
  basis size) — FP32 may not meet 1e-5 gate on sparse-cutoff configs.
  Mitigation: D-M8-8 explicitly scopes to dense-cutoff; sparse-cutoff SNAP
  MixedFast deferred M9+ mirroring EAM sparse decision.
- **R-M8-4 — Cloud burst queue time + cost.** AWS p4d.24xlarge spot instances
  могут иметь variable queue time (minutes-to-hours depending on region);
  on-demand fallback ~$33/hour. M8 scaling campaign estimated 5-20 instance-hours;
  budget constraint. Mitigation: T8.11 scope пишет harness.py с `--dry-run`
  flag + `--cheap` flag (p4d.xlarge alternative — 1 GPU only, skips scaling);
  opportunistic multi-GPU measurement when budget allows.
- **R-M8-5 — Artifact gate fails to beat LAMMPS ≥ 20%.** Plausible — LAMMPS
  USER-SNAP on single GPU is well-tuned; TDMD's advantage is multi-rank
  overlap, but at 8-rank the LAMMPS MPI also scales reasonably well. If TDMD
  doesn't beat, M8 still closes per D-M8-6 (honest documentation path); но
  это reshapes M9-M13 priorities (may accelerate MEAM/PACE roadmap to find
  ML kernel where TD architecture's advantage is pronounced). Mitigation:
  T8.11 REPORT.md includes "what to do next" roadmap regardless of outcome.
- **R-M8-6 — Heterogeneous precision regression risk.** MixedFastSnapOnlyBuild
  routing SNAP=FP32 / EAM=FP64 через adapter дает risk of accidentally using
  FP32 EAM kernel instead of FP64 under new flavor. Mitigation: T8.9 adapter
  test explicitly verifies EAM MixedFast D-M6-8 gate still holds under new
  flavor (regression check).
- **R-M8-7 — T6 fixture coefficient file size.** `W_2940_2017_2.snap` is ~few
  KB (small), but future SNAP coefficients (e.g. tungsten-rhenium alloy) могут
  быть larger. Mitigation: no binary tracked in tdmd repo (D-M8-3); submodule
  approach scales.
- **R-M8-8 — SnapPotential cache pollution.** SNAP basis is O(N_species^2)
  coefficient matrix + per-atom bispectrum cache. On large fixtures (16384
  atoms) cache может exceed L1/L2 budget on GPU SM → perf degradation.
  Mitigation: T8.6/T8.9 design cache in shared memory per-block, not register;
  profile с Nsight Compute on dev machine before committing.
- **R-M8-9 — 6-недельный M8 timeline slippage.** M7 shipped ~12 дней, M6 ~10.
  SNAP surface area (CPU port + GPU port + MixedFast + cloud burst) = ~5-6
  weeks realistic if LAMMPS port goes smoothly. If T8.4 CPU port stalls
  (R-M8-1 materialized), budget risk 7-8 weeks. Mitigation: D-M8-14 explicit
  "7 acceptable, flag at 8"; T8.11 cloud burst can run в parallel с T8.12
  slow-tier pass if calendar bandwidth available.
- **R-M8-10 — SNAP GPL license chain friction.** Porting LAMMPS code introduces
  GPL portion in tdmd repo. Current tdmd LICENSE is Apache-2.0. Need dual-license
  strategy (Apache-2.0 for non-SNAP + GPL for SNAP files). Mitigation: T8.4
  scope includes LICENSE.txt edit; precedent — many ML frameworks mix
  Apache-2.0 core + GPL modules (PyTorch linking libraries under different
  licenses).

**Open questions (resolved before or at task time):**

- **OQ-M8-1 — SNAP implementation strategy.** Port from LAMMPS vs reimplement
  from scratch? **Resolved by D-M8-1:** port с attribution; reimplementation
  adds months without science value.
- **OQ-M8-2 — Cloud burst vendor + hardware.** AWS p4d vs GCP a2 vs other?
  **Resolve at T8.11:** AWS p4d.24xlarge preferred (8×A100-40GB + EFA + NVLink);
  GCP a2-ultragpu-8g (A100-80GB) acceptable alternative если regional availability
  wins; Azure NDv4 (A100-40GB) fallback.
- **OQ-M8-3 — SNAP coefficient file handling.** Track binary в tdmd repo vs
  submodule vs runtime download? **Resolved by D-M8-3:** submodule path; no
  binary в tdmd repo; one-time dev setup cost.
- **OQ-M8-4 — GPU SNAP kernel block size.** 32 atoms/block (matches EAM M6)
  vs larger (64/128/256)? **To decide at T8.6:** start 32 (matches M6), profile
  with Nsight on dev GPU, adjust if bandwidth-bound.
- **OQ-M8-5 — MixedFastSnapOnly vs MixedFast fallback semantics.** When user
  sets `TDMD_BUILD_FLAVOR=MixedFastSnapOnlyBuild` but runs EAM-only workload
  — fall back to MixedFastBuild (FP64 force) silently vs warn vs error?
  **To decide at T8.8:** WARN + use FP64 force (SNAP kernel absent in run →
  no FP32 path activated — effectively identical to MixedFastBuild); explicit
  log message "MixedFastSnapOnlyBuild with no SNAP potential → effectively
  MixedFastBuild behavior".
- **OQ-M8-6 — Artifact gate normalization methodology.** Compare TDMD to LAMMPS
  на same hardware (fair), same fixture (fair), same problem size. Но какую
  metric primary? Wall-clock per step? Atom-step/sec (throughput)? Energy per
  simulation (Joules)? **To decide at T8.11:** primary = atom-step/sec
  (throughput — matches MD community convention); secondary = parallel
  efficiency (strong scaling at fixed N); report all three so readers can
  judge.
- **OQ-M8-7 — 16-rank opportunistic measurement.** Scaling test matrix через
  8 ranks D-M8-5 mandatory; 16 ranks = 2× p4d instances (inter-node via EFA).
  **To decide at T8.11:** opportunistic если cloud budget + queue time permit;
  16-rank не hard gate (master spec §14 M8 = ≥ 8 ranks).
- **OQ-M8-8 — T6 fixture total system size for scaling.** 16384 atoms (16×16×16
  W BCC) vs larger (e.g. 131072-atom — 32×32×32). **To decide at T8.10/T8.11:**
  16384 as primary fixture; 131072 opportunistic on p4d 80GB if a2-ultragpu
  cloud alternative chosen.
- **OQ-M8-9 — Slow-tier pass cycle time.** VerifyLab slow-tier на dev hardware
  может занять 2-4 часа (T3 10^6-atom + T4 1000-step + T6 тиmеing). Parallelizable
  в matrix runner? **To decide at T8.12:** serial run acceptable (one evening
  pre-push); matrix parallelization — M10+ VerifyLab infra work.
- **OQ-M8-10 — v1 alpha release communication.** Blog post? GitHub release
  page? Academic preprint? **To decide at T8.13:** GitHub release page with
  CHANGELOG.md + pointer to T8.11 REPORT.md sufficient; preprint deferred
  post-M11 (v1 beta) when scaling story more complete.

---

## 7. Roadmap Alignment

| Deliverable | Consumer milestone | Why it matters |
|---|---|---|
| T8.0 T7.8b carry-forward closure | M11 long-range overlap (needs 30% overlap baseline) | Clears M7 debt; infrastructure available on ≥ 2 GPU |
| LAMMPS SNAP subset verified (T8.2) | M9 MEAM differential; M10 PACE differential; all future LAMMPS cross-checks | Oracle infrastructure (landed M1) exercised on ML-SNAP path |
| SnapPotential CPU (T8.4) | M9 MEAM potential family (similar density-coefficient flavour); M10 PACE (ML kernel template) | Precedent for porting LAMMPS potentials; attribution pattern locked |
| CPU SNAP diff (T8.5) | M9 MEAM diff template; M11 NVT/NPT SNAP research | Canonical CPU FP64 SNAP oracle for all downstream SNAP work |
| SnapPotential GPU (T8.6) | M8 scaling gate; M9 MEAM GPU (shares kernel architecture) | GPU SNAP in production |
| GPU SNAP bit-exact gate (T8.7) | Continuous regression guard M8-M13; M9 MEAM bit-exact | D-M6-7 chain extension |
| MixedFastSnapOnlyBuild (T8.8) | M10 heterogeneous-precision large-scale runs; M9 MEAM MixedFast variant (precedent established) | §D.17 formal procedure precedent для future heterogeneous-precision flavors |
| SNAP MixedFast kernel (T8.9) | M8 scaling gate throughput path; M10 large-scale SNAP production | Throughput path for ML workloads |
| T6 tungsten fixture (T8.10) | Continuous regression M8-M13; M9-M11 SNAP feature additions | Canonical SNAP benchmark |
| TDMD vs LAMMPS scaling (T8.11) | M9-M13 architecture decisions (pattern selection priorities) | Project-level go/no-go signal; reshapes M9+ priority if outcome B |
| Slow-tier pass (T8.12) | M11 v1 beta gate; M13 v1.0.0 gate | §D.17 formal procedure documented pass |
| M8 smoke + v1 alpha (T8.13) | M9-M13 stability floor | Pre-push gate extended to include SNAP path |

**Downstream milestone impact:**

- **M9 (Ewald/PPPM + NVT/NPT Pattern 2 + MEAM + Morse GPU):** T8.5 CPU oracle
  pattern reused for MEAM diff; T8.6 GPU kernel pattern reused for MEAM GPU;
  Morse GPU kernel unblocks T3-gpu full dissertation replication (T7.12 partial
  substitute retired).
- **M10 (PACE + scaling consolidation):** T8.11 scaling REPORT.md outcome
  informs whether PACE timeline accelerates (if outcome B — TDMD didn't beat
  LAMMPS SNAP — MEAM/PACE become primary hope for proof-of-value).
- **M11 (v1 beta — NVT/NPT mature + long-range overlap):** T8.12 slow-tier
  process reused для new flavors; T8.0 2-rank overlap infra matures к N-rank.
- **M12 (MLIAP + observability hardening):** T8.4 port pattern reused for
  MLIAP; LAMMPS oracle (M1-landed, SNAP subset verified T8.2) remains canonical.
- **M13 (v1.0.0 — public release):** T8.13 v1.0.0-alpha1 tag + CHANGELOG.md
  pattern extended to v1.0.0 final; M8 artifact gate REPORT.md included в
  public release communication.

---

*End of M8 execution pack, дата: 2026-04-20.*
