# M7 Execution Pack

**Document:** `docs/development/m7_execution_pack.md`
**Status:** draft, awaiting human review
**Parent:** `TDMD_Engineering_Spec.md` §14 (M7), §12.6 (CommBackend inner+outer), §12.7a (OuterSdCoordinator), `docs/specs/scheduler/SPEC.md` §2.2 + §11a.5, `docs/specs/comm/SPEC.md` §§3-7, `docs/development/m6_execution_pack.md` (template), `docs/development/claude_code_playbook.md` §3
**Milestone:** M7 — Pattern 2 (two-level TD×SD hybrid) — 10 недель target, 11 acceptable, flag at 12
**Created:** 2026-04-19
**Author:** Architect / Spec Steward role (Claude Opus 4.7)

---

## 0. Purpose

Этот документ декомпозирует milestone **M7** master spec'а §14 на **15 PR-size задач**
(T7.0..T7.14), из которых **T7.0 — это carry-forward задача T6.8b из M6**, намеренно
запланированная первой как cleanup долга перед Pattern 2 work. Документ — **process
artifact**, не SPEC delta.

M7 — **первая встреча TDMD с two-level deployment.** После M6 на GPU работает: (a) весь
reference-path (CPU≡GPU bit-exact); (b) single-subdomain multi-rank TD (Pattern 1);
(c) host-staged MPI transport (`MpiHostStagingBackend`); (d) T3-gpu anchor с EAM
byte-exact CPU↔GPU gate; (e) три GPU kernel (NL + EAM + VV); (f) MixedFastBuild
(Philosophy B). M7 добавляет:

- **`OuterSdCoordinator`** — координатор между subdomain'ами; halo snapshot archive
  (last K snapshots); `can_advance_boundary_zone()`; global temporal frontier tracking
  (`TDMD_Engineering_Spec.md` §12.7a класс уже objявлен, M7 populates тело);
- **`SubdomainBoundaryDependency`** — новый dep kind в zone DAG scheduler'а;
- **Boundary zone stall protocol + watchdog** — на scheduler layer, отдельный от
  существующего deadlock watchdog'а M4;
- **`GpuAwareMpiBackend`** — outer SD halo exchange на device pointers (CUDA-aware MPI;
  eliminates D2H/H2D pair per halo send);
- **`NcclBackend`** — intra-node collectives для inner TD temporal packets
  (NCCL ≥2.18, bundled с CUDA toolkit);
- **`HybridBackend`** — composition: inner=NCCL (inside subdomain), outer=GpuAwareMPI
  (between subdomains); cached `subdomain_to_ranks[]` topology resolver;
- **`PerfModel::predict_step_hybrid_seconds`** + `recommended_pattern: "Pattern2"`;
  Pattern 2 cost tables + tolerance gate `|predict - measure| < 25%` (мягче чем
  Pattern 1 из-за возрастающей сложности модели);
- **T7 `mixed-scaling` benchmark** — first multi-node fixture в VerifyLab; efficiency
  gate ≥80% на 1-node × 8 GPU, ≥70% на 2 nodes × 8 GPU;
- **M7 integration smoke** — Pattern 2 multi-rank K=1 byte-exact to Pattern 1 K=1
  (extends D-M5-12 / D-M6-7 chain: M3 ≡ M4 ≡ M5 ≡ M6 ≡ M7 Pattern 2 K=1).

**Conceptual leap от M6 к M7:**

- M6 = "scheduling uses the GPU" (CUDA kernels behind unchanged CPU interfaces;
  single-subdomain multi-rank; host-staged MPI unchanged from M5).
- **M7 = "scheduling goes two-level"** (Pattern 2: `InnerTdScheduler` внутри subdomain
  × `OuterSdCoordinator` между; inner=NCCL, outer=CUDA-aware MPI; halo snapshot
  archive; boundary zone dependency kind).
- M8 = "performance proof" (SNAP + `MixedFastSnapOnlyBuild`; TDMD vs LAMMPS demo).

Критически — **Pattern 1 остаётся fully functional regression-test baseline**. Любой
M7 PR проходит весь M6 test suite без изменений (master spec §14 M7 mandate). Это
значит: `OuterSdCoordinator*` остаётся nullable в `TdScheduler::attach_outer_coordinator()`
(scheduler/SPEC §2.2 — уже так authored); Pattern 1 runs заходят в `outer_ == nullptr`
branch и не видят Pattern 2 surface.

**M6 carry-forward — намеренно встроен в M7:**

- **T6.8b → T7.0** — NL MixedFast variant + T4 100-step NVE drift harness + FP32-table
  redesign OR формальная SPEC delta relaxing D-M6-8 force threshold до 1e-5 на
  dense-cutoff stencil'ах. **User directive:** cleanup долга **первой задачей M7** до
  Pattern 2 work.
- **T6.9b → T7.8** — full 2-stream compute/copy overlap pipeline + 30% gate.
  Теперь unblocked: Pattern 2 GPU dispatch в T7.9 предоставляет консюмера для
  mem_stream'а beyond T6.9a spline caching.
- **T6.10b → T7.12 (partial)** — T3-gpu efficiency curve. Pattern 2 GPU dispatch блок
  снят T7.9; Morse GPU kernel блок остаётся (M9+). **Decision:** T7.12 ships **EAM-substitute
  efficiency curve** с explicit "Morse full-fidelity pending M9+" note в report;
  это сохраняет regression gate на GPU anchor в multi-subdomain режиме.
- **T6.11b → T7.13** — PerfModel ±20% calibration. Orthogonal: local Nsight run на
  target GPU; не в M7 critical path; lands когда данные собраны.

После успешного закрытия всех 15 задач и acceptance gate (§5) — milestone M7 завершён;
execution pack для M8 создаётся как новый аналогичный документ.

---

## 1. Decisions log (зафиксировано до старта T7.1)

| # | Решение | Значение | Rationale / источник |
|---|---|---|---|
| **D-M7-1** | Pattern 2 outer topology | **3D Cartesian grid** subdomain'ов `(P_space_x, P_space_y, P_space_z)`. M7 ships 2D (Z-axis inner + X-axis outer) как minimum gate, 3D enabled в config но full 3D validation — M8+. | Master spec §12.7a `SubdomainGrid::n_subdomains[3]` уже declared; 2D M7 match'ит M5 Linear1D inner zoning pattern. |
| **D-M7-2** | Rank ↔ subdomain binding | **1:1 в M7** — один MPI rank владеет одним subdomain'ом; один GPU per rank; subdomain полностью resident на одной device'е. Multi-GPU per rank (sub-subdomain sharding) — M8+. | comm/SPEC §5 cached `subdomain_to_ranks[]` — M7 simplifies к 1-element lists; multi-rank-per-subdomain раскрывается позже. |
| **D-M7-3** | MPI requirement | **CUDA-aware OpenMPI ≥4.1** для `GpuAwareMpiBackend`. Если preflight probe detects non-CUDA-aware MPI → warn + automatic fallback на `MpiHostStagingBackend` (D2H/H2D staging, M5 semantics). | Pattern 2 без GPU-aware MPI имеет ~2× halo overhead (vs host-staged M6). Fallback preserves correctness, warns user explicitly. |
| **D-M7-4** | NCCL requirement | **NCCL ≥2.18** (bundled с CUDA 13.x). NcclBackend используется для inner TD temporal packets intra-node; inter-node NCCL (via TCP or RDMA) — **отложен до M8+**. | Master spec §14 M7 список "NcclBackend (для inner TD temporal packets)"; intra-node only достаточен для M7 gate (≥80% efficiency на 1-node × 8 GPU). |
| **D-M7-5** | GPU kernel scope в M7 | **Unchanged from M6**: NL build, EAM/alloy force, VV integrator. Никаких новых potential styles. M7 фокус — multi-subdomain coordination, не physics surface. LJ/Morse/MEAM/SNAP/PACE/MLIAP + NVT/NPT остаются M9+. | Scope discipline: M7 уже добавляет OuterSdCoordinator + 3 comm backends + SubdomainBoundaryDep + PerfModel Pattern 2 + T7 benchmark — добавление kernels переполнит бюджет. |
| **D-M7-6** | Active BuildFlavors | Unchanged from M6: `Fp64ReferenceBuild` (bit-exact oracle) + `MixedFastBuild`. Два backends × два flavors = 4 CI matrix cells (compile-only на public CI, runtime locally). | M6 precedent. MixedFastAggressive / Fp64Production / Fp32Experimental — M8+. |
| **D-M7-7** | Active ExecProfiles | `Reference` (byte-exact gate) + `Production` (performance tuning). `Fast` — M8+. | Unchanged from M6. |
| **D-M7-8** | Scaling gates (hard acceptance) | **≥80% strong-scaling efficiency** на T3 для 8 GPU single-node (1 → 8 ranks), **≥70%** на 2 nodes × 8 GPU (8 → 16 ranks). Normalized по PerfModel per-hardware коэффициентам. | Master spec §14 M7 verbatim — locked gate. Single-node — achievable; 2-node — opportunistic на dev hardware (cloud burst допустимо). |
| **D-M7-9** | PerfModel Pattern 2 tolerance | `abs(predict_step_hybrid_seconds - measured) < 25%` — mandatory gate. Softer чем Pattern 1 (±20%) из-за двух параллельных уровней (inner TD overlap + outer SD halo). | Master spec §14 M7 verbatim ("допуск мягче чем для Pattern 1 из-за сложности модели"). |
| **D-M7-10** | Byte-exact chain extension | Pattern 2 K=1 P_space=2 (2 subdomain × 1 rank) `Fp64ReferenceBuild` thermo **byte-exact** to Pattern 1 K=1 P_space=1 (M3/M4/M5/M6 golden). Extends D-M5-12 / D-M6-7 invariant chain: M3 ≡ M4 ≡ M5 ≡ M6 ≡ M7 Pattern 2 K=1. | Master spec §13.5 determinism matrix — Reference profile bitwise oracle across all deployment patterns. Invariant: canonical Kahan-ring reduction order preserved across halo exchange. |
| **D-M7-11** | CI strategy | **Option A continues** — no self-hosted runner. New `build-gpu-pattern2` compile-only matrix: `{Fp64ReferenceBuild, MixedFastBuild} × {MpiHostStaging fallback, GpuAwareMpi (compile-only)}`. Multi-node runtime gates — local pre-push OR cloud burst; never on public CI. | Memory `project_option_a_ci.md` + D-M6-6 precedent. No change в CI infrastructure policy. |
| **D-M7-12** | Timeline | **10 недель target, 11 acceptable, flag at 12**. Most expensive — T7.2 OuterSdCoordinator + scheduler integration (~6 days), T7.5 HybridBackend composition (~5 days), T7.6 halo snapshot archive (~5 days), T7.9 SimulationEngine Pattern 2 wire (~5 days), T7.11 T7 benchmark multi-node (~4 days). | Confirmed by user 2026-04-19 per pre-implementation report. Budget vs M6 shipped ~10 days: M7 adds 15 tasks but many are carry-forward (T7.0/T7.8/T7.12/T7.13). |
| **D-M7-13** | Halo snapshot archive depth | **K_max snapshots per boundary zone** where K_max = `pipeline_depth_cap` (default 1 в M7 NVE, maximum 8 в K-batching). Ring buffer eviction on snapshot register. | Master spec §12.7a "halo snapshot archive (last K snapshots)". RAM budget: K_max × n_boundary_zones × per-atom payload — на 10⁶ atoms с 10⁴ boundary atoms и K=8 это ≈ 2 MiB per subdomain boundary. |
| **D-M7-14** | Boundary stall watchdog | `T_stall_max = 10 × T_step_predicted` default (configurable via `comm.outer.boundary_stall_timeout_ms`). Stall → `OuterSdCoordinator::check_stall_boundaries()` escalates to `TdScheduler::invalidate_certificates_for()` + emit `boundary_stall_event` в telemetry. | scheduler/SPEC §11a `T_watchdog` pattern — extended из M4 deadlock watchdog; separate counter (`scheduler.boundary_stalls_total` в §12) для diagnostics. |
| **D-M7-15** | HaloPacket protocol version | `protocol_version = 1` (new wire format for outer halos). Format: `(u16 version, u32 source_subdomain_id, TimeLevel, u32 atom_count, payload, u32 crc32)`. Independent versioning от `TemporalPacket.protocol_version` (inner transport). | Master spec §12.6 `HaloPacket` declaration; comm/SPEC extended с outer transport tests. |
| **D-M7-16** | T6.10b partial scope | **T7.12 ships EAM-substitute efficiency curve**, not dissertation Morse. Report explicitly notes "Morse full-fidelity replication pending M9+ (Morse GPU kernel blocker)"; EAM curve serves as Pattern 2 regression gate. Full dissertation replication — M9+. | User directive 2026-04-19. Preserves M7 acceptance independence while honestly flagging the fidelity limitation. |
| **D-M7-17** | Regression preservation (hard) | Every M7 PR MUST pass: M1..M6 integration smokes + T1/T4 differentials + T3-gpu anchor + M6 smoke. Zero regression tolerance; any failure blocks merge. | Master spec §14 M7 "Pattern 1 остаётся fully functional". Pre-push protocol extended с M7 acceptance smoke в T7.14. |
| **D-M7-18** | `SubdomainBoundaryDependency` semantics | New `ZoneDependency` kind: `{BoundaryHaloArrived, peer_subdomain_id, time_level}`. Released when `OuterSdCoordinator::on_halo_arrived(peer_subdomain, level)` fires; satisfied до того как `select_ready_tasks()` returns boundary zone task. | scheduler/SPEC §2.2 existing `on_halo_arrived()` callback (уже authored) + new dependency kind registered в DAG builder; preserves two-phase commit. |
| **D-M7-19** | PerfModel GPU cost tables | T7.10 extends `GpuCostTables` (shipped T6.11) с Pattern 2 cost stages: `halo_pack`, `halo_send_outer`, `halo_unpack`, `nccl_allreduce_inner`. Factories update provenance strings. Coefficients — placeholder в T7.10, calibrated в T7.13 (T6.11b). | perfmodel/SPEC v1.1 extension; T6.11 placeholder pattern continues. |
| **D-M7-20** | SPEC deltas в M7 | **No new module SPEC.md created** (unlike M6 which added `gpu/SPEC.md`). M7 populates existing contracts: scheduler/SPEC adds §X Pattern 2 integration section (~100 lines), comm/SPEC fills §§3-7 bodies (~200 lines), perfmodel/SPEC §11.5 Pattern 2 cost extension (~50 lines). Master spec Приложение C gets T7.X addendums per merged PR. | All Pattern 2 interfaces already declared в master spec §12.6/§12.7a + module SPECs as roadmap pointers — M7 authors bodies, не contracts. |

---

## 2. Глобальные параметры окружения

| Параметр | Значение | Примечание |
|---|---|---|
| OS | Linux (Ubuntu 24.04 LTS) | Dev-машина пользователя; ubuntu-latest в CI |
| C++ compiler | GCC 13+ / Clang 17+ | C++20; CI уже проверяет оба (M6 matrix) |
| CMake | 3.25+ | Master spec §15.2 |
| CUDA | **13.1** installed (system `/usr/local/cuda`) | D-M6-2 carry-forward; CI compile-only |
| GPU archs | sm_80, sm_86, sm_89, sm_90, sm_100, sm_120 | D-M6-1 carry-forward |
| MPI | **CUDA-aware OpenMPI ≥4.1** preferred; non-CUDA-aware → fallback | D-M7-3; preflight probe mandatory |
| NCCL | **≥2.18** (bundled с CUDA 13.x) | D-M7-4; intra-node only в M7 |
| Python | 3.10+ | pre-commit + anchor-test harness + T7 scaling harness |
| Test framework | Catch2 v3 (FetchContent) + MPI wrapper | GPU+MPI tests local-only per D-M7-11 |
| LAMMPS oracle | SKIP on public CI (Option A) | Differentials run pre-push locally |
| Active BuildFlavors | `Fp64ReferenceBuild`, `MixedFastBuild` | D-M7-6 |
| Active ExecProfiles | `Reference`, `Production` (GPU) | D-M7-7 |
| Run mode | multi-rank MPI × GPU-per-rank × 1:1 subdomain binding | D-M7-2 |
| Pipeline depth K | `{1, 2, 4, 8}` (as M5/M6); default 1 | Unchanged |
| Subdomain topology | Cartesian 1D/2D в M7 ships, 3D config allowed но full validation M8+ | D-M7-1 |
| Streams per rank | 2 (default) — compute + mem | D-M6-13 carry-forward; T7.8 populates full overlap |
| CI CUDA | compile-only matrix: `{Ref, Mixed} × {HostStaging, GpuAwareMpi}` | D-M7-11 |
| Local pre-push gates | Full GPU suite + T3-gpu + M1..M6 smokes + M7 Pattern 2 smoke | D-M7-17 |
| Branch policy | `m7/T7.X-<topic>` per PR → `main` | CI required: lint + build-cpu + build-gpu + build-gpu-pattern2 + M1..M6 smokes; M7 smoke добавляется в T7.14 |

---

## 3. Suggested PR order

**Dependency graph:**

```
T7.0 (T6.8b carry-fwd) ─┐
                        │
T7.1 (this pack) ───────┼──► T7.2 (scheduler Pattern 2 + OuterSdCoord SPEC)
                        │                │
                        │      ┌─────────┼─────────┐
                        │      ▼         ▼         ▼
                        │   T7.3     T7.4      T7.6
                        │  (GpuAware (NcclBack  (OuterSdCoord
                        │   MpiBack)  end)      impl + archive)
                        │      └─────────┬─────────┘
                        │                ▼
                        │            T7.5 (HybridBackend)
                        │                │
                        │      ┌─────────┴─────────┐
                        │      ▼                   ▼
                        │   T7.7            T7.8 (T6.9b: full
                        │  (SubdomainBound   2-stream overlap
                        │   Dep + stall       + 30% gate)
                        │   watchdog)             │
                        │      └─────────┬─────────┘
                        │                ▼
                        │            T7.9 (SimulationEngine
                        │             Pattern 2 wire)
                        │                │
                        │                ▼
                        │            T7.10 (PerfModel Pattern 2)
                        │                │
                        │                ▼
                        │            T7.11 (T7 mixed-scaling bench)
                        │                │
                        │      ┌─────────┴─────────┐
                        │      ▼                   ▼
                        │   T7.12             T7.13 (T6.11b: ±20%
                        │  (T6.10b: T3-gpu     calibration — orthogonal,
                        │   EAM efficiency      not blocking)
                        │   curve)                │
                        │      └─────────┬─────────┘
                        │                ▼
                        │            T7.14 (M7 smoke + GATE)
                        │                │
                        └────────────────┘
```

**Линейная последовательность (single agent):**
T7.0 → T7.1 → T7.2 → T7.3 → T7.4 → T7.5 → T7.6 → T7.7 → T7.8 → T7.9 → T7.10 → T7.11 →
T7.12 → T7.13 → T7.14.

**Параллельный режим (multi-agent):** после T7.2 — `{T7.3, T7.4, T7.6}` три independent
deliverables (GPU-aware MPI transport × NCCL transport × coordinator implementation);
объединяются на T7.5 (HybridBackend composition). После T7.5 — `{T7.7, T7.8}` независимы
(scheduler dep-kind wiring vs GPU overlap). После T7.9 — `{T7.11, T7.10}` частично
параллельны (PerfModel нужен для mixed-scaling benchmark normalization). T7.12 и T7.13
independent (efficiency curve vs calibration gate). T7.14 — final gate после всех.

**Estimated effort:** 10 недель target (single agent, per D-M7-12). Самые длинные —
T7.2 scheduler integration (~6 дней), T7.5 HybridBackend + topology resolver (~5 дней),
T7.6 halo snapshot archive + ring buffer (~5 дней), T7.9 engine Pattern 2 wire (~5 дней),
T7.11 multi-node benchmark infrastructure (~4 дня). Остальные 2-4 дня.

---

## 4. Tasks

### T7.0 — M6 T6.8b carry-forward — MixedFast NL + T4 NVE drift + FP32-table redesign

```
# TDMD Task: Close M6 D-M6-8 debt — T6.8b landing

## Context
- Master spec: §D (BuildFlavors), §13.7 (differential thresholds), M6 carry-forward
- Module SPEC: `docs/specs/gpu/SPEC.md` §8.3 (T6.8a shipped thresholds table)
- Role: GPU / Performance Engineer
- Milestone: M7 T7.0 (first PR of M7, cleanup of M6 debt)
- User directive 2026-04-19: schedule first in M7 as cleanup before Pattern 2 work

## Goal
Close остаточный debt T6.8a: либо хит D-M6-8 force threshold 1e-6 на MixedFast GPU
path через FP32-table-storage redesign, либо формальная SPEC delta relaxing threshold
до 1e-5 на dense-cutoff stencil'ах с explicit scientific rationale. Plus T4 100-step
NVE drift harness (D-M6-8 `gpu_mixed_fast_nve_drift ≤ 1e-5/1000 steps`) и NL MixedFast
variant если perf-justified.

## Scope
- [included] FP32-table-storage redesign investigation в
  `src/gpu/eam_alloy_gpu_mixed.cu` — cast `rho_coeffs` / `F_coeffs` / `z_coeffs` в
  FP32 device-side, FP32 Horner stability review per-pair; если stable — hit 1e-6.
- [included] OR formal SPEC delta в `docs/specs/gpu/SPEC.md` §8.3 relaxing
  `gpu_mixed_fast_force_rel` до 1e-5 с explicit rationale (FP32 inv_r propagation
  через ~50-neighbor EAM stencil с partial sign cancellation — hardware precision
  ceiling, not implementation bug); `verify/threshold_registry.yaml` updated.
- [included] `verify/differentials/t4_gpu_mixed_vs_reference/` 100-step NVE drift
  harness под `DifferentialRunner`: измеряет rel total energy drift over 100-step
  NVE run MixedFast vs Reference, gates ≤1e-5/1000 steps (extrapolated from 100).
- [included] NL MixedFast variant `src/gpu/neighbor_list_gpu_mixed.cu` если
  perf-justified (bench shows ≥5% gain vs Reference NL); otherwise document как
  "not perf-justified, deferred".
- [included] `tests/gpu/test_t4_mixed_nve_drift.cpp` — Catch2 wrapper over
  DifferentialRunner invocation.
- [included] Master spec Приложение C + gpu/SPEC change log updates.

## Out of scope
- [excluded] Any Pattern 2 work (T7.2+ territory).
- [excluded] NL mixed bit-exactness (impossible by design — `build_version` byte-comparison
  uses integer CSR indices only, not affected by FP precision).
- [excluded] VV MixedFast variant — kernel is H2D/D2H-bound, FP32 narrowing negligible.

## Mandatory invariants
- D-M6-7 Reference byte-exact gate remains green на все three CI flavors.
- D-M6-8 либо met (1e-6) либо formally relaxed with SPEC delta (never silently
  weakened).
- T6.8a `test_eam_mixed_fast_within_threshold` green (thresholds updated consistent
  with SPEC delta if applicable).
- All M1..M6 smokes + T1/T4 differentials green.

## Required files
- `src/gpu/eam_alloy_gpu_mixed.cu` (edit) — FP32-table storage if redesign
- `src/gpu/neighbor_list_gpu_mixed.{hpp,cu}` (new, conditional) — NL MixedFast variant
- `verify/differentials/t4_gpu_mixed_vs_reference/checks.yaml` (new) — 100-step drift
- `verify/differentials/t4_gpu_mixed_vs_reference/config.yaml.template` (new)
- `tests/gpu/test_t4_mixed_nve_drift.cpp` (new) — Catch2 harness wrapper
- `docs/specs/gpu/SPEC.md` §8.3 (edit) — threshold table update + change log
- `verify/threshold_registry.yaml` (edit if SPEC delta path)
- `TDMD_Engineering_Spec.md` Приложение C — T7.0 addendum
- `docs/development/m7_execution_pack.md` §5 — mark T7.0 closed

## Required tests
- `test_t4_mixed_nve_drift` — 100-step NVE на Ni-Al B2 1024 + Al FCC 864; drift gate
  `<= 1e-5 × 100/1000` rel total energy per run (extrapolates to 1e-5/1000 steps).
- Existing `test_eam_mixed_fast_within_threshold` green (may need threshold update
  coherent with SPEC delta).
- NL MixedFast bench (conditional): `verify/benchmarks/neighbor_gpu_vs_cpu/` runs
  Reference vs Mixed; commit only if ≥5% gain demonstrated.

## Acceptance criteria
- T4 100-step NVE drift ≤ 1e-5 per 1000 steps (extrapolated).
- Either: rel force per-atom ≤ 1e-6 (FP32-table redesign) OR SPEC delta shipped с
  verify/threshold_registry.yaml threshold updated to 1e-5.
- All three CI flavors green (Reference+CUDA, MixedFast+CUDA, CPU-only-strict).
- Pre-impl + session reports attached.
- Human review approval.
```

---

### T7.1 — Author M7 execution pack (this document)

```
# TDMD Task: Create M7 execution pack

## Context
- Master spec: §14 M7
- Role: Architect / Spec Steward
- Milestone: M7 (kickoff)

## Goal
Написать `docs/development/m7_execution_pack.md` декомпозирующий M7 на 15 PR-size задач
(T7.0 = carry-forward cleanup, T7.1 = this pack). Document-only PR per playbook §9.1.

## Scope
- [included] `docs/development/m7_execution_pack.md` (single new file)
- [included] Decisions log D-M7-1..D-M7-20
- [included] Task templates T7.0..T7.14
- [included] M7 acceptance gate checklist
- [included] Risks R-M7-1..R-M7-N + open questions OQ-M7-*

## Out of scope
- [excluded] Any code changes (T7.2+ territory)
- [excluded] SPEC deltas (T7.2 onwards)

## Required files
- `docs/development/m7_execution_pack.md`

## Acceptance criteria
- Document covers §0-§7 complete (Purpose, Decisions, Env, PR order, Tasks, Gate, Risks, Roadmap).
- Markdown lint + pre-commit hooks green.
- Human review approval.
```

---

### T7.2 — scheduler Pattern 2 integration + OuterSdCoordinator contract SPEC

```
# TDMD Task: Scheduler Pattern 2 integration + OuterSdCoordinator contract

## Context
- Master spec §12.7a (OuterSdCoordinator class declaration — уже authored)
- scheduler/SPEC §2.2 (`attach_outer_coordinator()` уже authored; nullable)
- scheduler/SPEC §11a.5 (load balancing across subdomains — Pattern 2 policy authored)
- comm/SPEC §§3-7 (GpuAwareMpi/Nccl/Hybrid declarations — M7 populates bodies)
- Role: Scheduler / Determinism Engineer
- Depends: T7.1 (pack authored)

## Goal
Дополнить scheduler/SPEC новым разделом **Pattern 2 integration semantics** (~100 строк):
(a) `SubdomainBoundaryDependency` kind в zone DAG (D-M7-18); (b) boundary zone
certificate extended с `halo_valid_until_step` provenance; (c) stall watchdog
mechanics (D-M7-14); (d) interaction между `InnerTdScheduler` (existing M4 code) и
`OuterSdCoordinator*` через `attach_outer_coordinator()` / `on_halo_arrived()`
callbacks. **No code** — this is a SPEC delta PR.

Parallel: comm/SPEC расширен с outer halo path semantics (§3-§7 bodies remain for
T7.3-T7.5; this PR just finalizes the interface contracts + test surface).

## Scope
- [included] `docs/specs/scheduler/SPEC.md` — new §X Pattern 2 integration section
  (certificate extension, dep kind, stall watchdog, commit protocol interaction).
- [included] `docs/specs/comm/SPEC.md` — §§3-7 clarifications на outer halo path
  (protocol version bump для HaloPacket per D-M7-15).
- [included] `docs/specs/perfmodel/SPEC.md` — stub §11.5 Pattern 2 cost placeholder
  для T7.10 body.
- [included] Master spec Приложение C T7.2 addendum + change log entries в каждом
  touched SPEC.
- [included] Pre-impl + session reports.

## Out of scope
- [excluded] Implementation code (T7.3-T7.9 territory).
- [excluded] `OuterSdCoordinator` concrete class (T7.6).
- [excluded] Comm backend bodies (T7.3-T7.5).

## Mandatory invariants
- **Pattern 1 regression preserved** — all existing scheduler/SPEC contracts
  unchanged; §X is additive.
- **Ownership boundaries** — OuterSdCoordinator в scheduler/, halo transport в comm/,
  halo snapshot archive owned by OuterSdCoordinator (not by comm/).
- **Two-phase commit** extended: outer halo arrival triggers certificate refresh
  on boundary zones, then standard two-phase commit proceeds.
- **Determinism** — canonical Kahan-ring order extended to outer SD halo reductions.

## Required files
- `docs/specs/scheduler/SPEC.md` (edit, ~150 lines added)
- `docs/specs/comm/SPEC.md` (edit, ~80 lines)
- `docs/specs/perfmodel/SPEC.md` (edit, ~20 lines stub)
- `TDMD_Engineering_Spec.md` Приложение C

## Required tests
- None (SPEC-only PR). Tests land в T7.3-T7.7 as bodies materialize.

## Acceptance criteria
- Markdown lint green.
- Human review approval on contract additions.
- No backwards-incompatible contract changes (scheduler §2.2 + comm §3 interfaces
  already support Pattern 2 — this PR fills gaps, doesn't reshape).
```

---

### T7.3 — `GpuAwareMpiBackend` implementation

```
# TDMD Task: GpuAwareMpiBackend — CUDA-aware MPI transport for outer halos

## Context
- Master spec §12.6 CommBackend interface
- comm/SPEC §3.2 (GpuAwareMpiBackend class declaration — M7 authors body)
- D-M7-3 (CUDA-aware OpenMPI ≥4.1 preferred, fallback на MpiHostStaging)
- Role: GPU / Performance Engineer
- Depends: T7.2 (SPEC finalized), M6 MpiHostStaging (M5 landed — baseline)

## Goal
Реализовать `tdmd::comm::GpuAwareMpiBackend` — CUDA-aware `MPI_Send`/`MPI_Recv` на
device pointers, eliminates D2H + H2D roundtrip per halo send. Runtime preflight
probe (`MPIX_Query_cuda_support()` or `OMPI_MCA_opal_cuda_support` env check) —
если detect → use; иначе `SimulationEngine` falls back на `MpiHostStagingBackend` с
explicit warning.

## Scope
- [included] `src/comm/include/tdmd/comm/gpu_aware_mpi_backend.hpp` — class decl
  inheriting `CommBackend`; PIMPL firewall (MPI headers hidden).
- [included] `src/comm/gpu_aware_mpi_backend.cpp` — body: `send_subdomain_halo()`
  calls `MPI_Send(dev_ptr, count, MPI_BYTE, dest_subdomain, tag, comm)` directly;
  `send_temporal_packet()` routes через parent host-staged path (inner TD — NCCL
  handles that в T7.4).
- [included] `src/comm/cuda_mpi_probe.cpp` — runtime probe; exports
  `bool is_cuda_aware_mpi()` used by SimulationEngine preflight.
- [included] `tests/comm/test_gpu_aware_mpi_backend.cpp` — Catch2 MPI wrapper;
  bit-exact halo echo-to-self на 2-rank setup; skip at runtime if probe fails.
- [included] CMake wiring: `TDMD_ENABLE_GPU_AWARE_MPI` flag default ON if
  `TDMD_BUILD_CUDA=ON AND TDMD_ENABLE_MPI=ON`.
- [included] comm/SPEC §3.2 change log + master spec Приложение C.

## Out of scope
- [excluded] NcclBackend (T7.4).
- [excluded] HybridBackend composition (T7.5).
- [excluded] OuterSdCoordinator integration (T7.6).
- [excluded] Engine preflight wiring (T7.9).

## Mandatory invariants
- PIMPL firewall: no MPI headers в public comm/ API.
- Determinism: `MPI_Send` on device pointer preserves canonical byte ordering
  (MPI just moves bytes; Kahan reduction order maintained by receiver-side
  reduction, not by transport).
- Fallback protocol: probe fails → backend refuses to construct (throws); engine
  falls back to MpiHostStaging (T7.9 wiring).
- CRC32 on HaloPacket payload still verified post-transport (integrity invariant
  from comm/SPEC §5).

## Required files
- `src/comm/include/tdmd/comm/gpu_aware_mpi_backend.hpp`
- `src/comm/gpu_aware_mpi_backend.cpp`
- `src/comm/cuda_mpi_probe.{hpp,cpp}`
- `tests/comm/test_gpu_aware_mpi_backend.cpp`
- `src/comm/CMakeLists.txt`
- `docs/specs/comm/SPEC.md` §3.2

## Required tests
- `test_gpu_aware_mpi_backend::halo_echo_2rank` — pack halo buffer on device,
  `MPI_Sendrecv` to self, verify bit-equal on D2H check.
- `test_cuda_mpi_probe::probe_reports_or_throws_clean` — probe either succeeds
  (CUDA-aware MPI present) or returns false (never crashes).
- Local CI integration gate: backend compiles and links on public ubuntu-latest
  (runtime SKIP per Option A; self-skip via probe).

## Acceptance criteria
- Probe correctly detects CUDA-aware MPI on dev machine; fails gracefully on
  non-CUDA-aware.
- Halo echo-to-self bit-exact.
- M1..M6 regressions green.
- Pre-impl + session reports.
- Human review approval.
```

---

### T7.4 — `NcclBackend` implementation

```
# TDMD Task: NcclBackend — intra-node NCCL collectives for inner TD

## Context
- Master spec §14 M7 (NcclBackend — inner TD temporal packets)
- comm/SPEC §3.3 (class declaration — M7 authors body)
- D-M7-4 (NCCL ≥2.18; intra-node only в M7)
- Role: GPU / Performance Engineer
- Depends: T7.2 (SPEC)

## Goal
Реализовать `tdmd::comm::NcclBackend` — NCCL collectives для inner TD temporal packet
transport intra-node. Specifically `ncclAllReduce` used в deterministic thermo ring
(D-M5-9 Kahan extension to NCCL path). `ncclBroadcast` для halo snapshot distribution
intra-subdomain (Pattern 2 case). Inter-node NCCL — deferred M8+.

## Scope
- [included] `src/comm/include/tdmd/comm/nccl_backend.hpp` + `nccl_backend.cpp` —
  PIMPL body; `ncclCommInitAll` at backend init; cleanup в destructor; `send_temporal_packet`
  routes via `ncclSend`/`ncclRecv`; `deterministic_sum_double` extended с NCCL path
  (host-side Kahan still, NCCL is just transport — matches D-M5-9 policy).
- [included] `tests/comm/test_nccl_backend.cpp` — intra-node 2-rank allreduce
  бит-exact vs host-side Kahan; bit-exact vs M5 MpiHostStaging baseline (extends
  D-M5-12 chain to NCCL path).
- [included] CMake: `TDMD_ENABLE_NCCL` default ON if CUDA+MPI both on.
- [included] NCCL version probe (`ncclGetVersion()`) emits warning if < 2.18.
- [included] comm/SPEC §3.3 change log + master spec Приложение C.

## Out of scope
- [excluded] Inter-node NCCL (M8+ when multi-node NCCL topology validated).
- [excluded] HybridBackend composition (T7.5).
- [excluded] NCCL-aware reduction (NCCL remains transport-only; Kahan host-side
  stays authoritative in Reference).

## Mandatory invariants
- D-M5-9 determinism preserved: NCCL is transport; `deterministic_sum_double`
  still owns reduction semantics on host side.
- Byte-exact chain: `NcclBackend` thermo == `MpiHostStagingBackend` thermo на
  single-node 2-rank setup (guards D-M5-12 through NCCL path).
- PIMPL firewall: no NCCL headers в public comm/ API.
- Fallback: NCCL init fails → backend refuses construct; engine fall back на
  MpiHostStaging.

## Required files
- `src/comm/include/tdmd/comm/nccl_backend.hpp`
- `src/comm/nccl_backend.cpp`
- `src/comm/nccl_probe.{hpp,cpp}`
- `tests/comm/test_nccl_backend.cpp`
- `src/comm/CMakeLists.txt`
- `docs/specs/comm/SPEC.md` §3.3

## Required tests
- `test_nccl_backend::allreduce_deterministic_vs_m5` — same 2-rank thermo test
  как M5 MpiHostStaging, но через NcclBackend; bit-exact M5 golden.
- `test_nccl_backend::version_probe_nonfatal` — старый NCCL — warning, не crash.

## Acceptance criteria
- NCCL AllReduce bit-exact vs MpiHostStaging на M5 smoke fixture.
- M1..M6 regressions green.
- Pre-impl + session reports.
- Human review approval.
```

---

### T7.5 — `HybridBackend` composition + topology resolver

```
# TDMD Task: HybridBackend — inner=NCCL, outer=GpuAwareMPI composition

## Context
- Master spec §14 M7 (HybridBackend)
- comm/SPEC §3.4 (class declaration — M7 authors body)
- D-M7-2 (1:1 rank↔subdomain binding)
- Role: GPU / Performance Engineer
- Depends: T7.3 (GpuAwareMpiBackend), T7.4 (NcclBackend)

## Goal
Реализовать `tdmd::comm::HybridBackend` — composition, не duplicates. Inner TD
temporal packets (`send_temporal_packet`) → dispatches к internal `NcclBackend`
(intra-node collectives); outer SD halo (`send_subdomain_halo`) → dispatches к
internal `GpuAwareMpiBackend`. Topology resolver caches `subdomain_to_ranks[]`
(1-element lists в M7 per D-M7-2) + `peer_neighbors()` mapping по 3D Cartesian
grid (up to 26 neighbors per subdomain).

## Scope
- [included] `src/comm/include/tdmd/comm/hybrid_backend.hpp` + body — owns unique_ptr
  to inner `NcclBackend` + outer `GpuAwareMpiBackend`; dispatches по method.
- [included] `src/comm/topology_resolver.{hpp,cpp}` — Cartesian SD grid walker;
  returns neighbor subdomain IDs for halo sends; caches result first call.
- [included] `tests/comm/test_hybrid_backend.cpp` — 4-rank 2×2 Cartesian grid;
  verify (a) inner sends routed to Nccl path; (b) outer halos routed to
  GpuAware path; (c) topology returns correct 8 neighbors per corner subdomain
  in 2D, 26 in 3D.
- [included] `tests/comm/test_topology_resolver.cpp` — 1D/2D/3D grid unit tests.
- [included] comm/SPEC §3.4 change log + master spec Приложение C.

## Out of scope
- [excluded] OuterSdCoordinator integration (T7.6 — HybridBackend is transport,
  coordinator is scheduler concern).
- [excluded] Engine wiring (T7.9).

## Mandatory invariants
- HybridBackend is composition, not duplication: внутри лишь инстанцирует и
  диспетчерит на inner/outer primaries.
- Single-subdomain (Pattern 1) runs использует HybridBackend без ошибок: outer
  paths not exercised (`peer_neighbors()` returns empty для single-subdomain config).
- Topology resolver deterministic: same config → same neighbor ordering (crucial
  для reproducibility).

## Required files
- `src/comm/include/tdmd/comm/hybrid_backend.hpp`
- `src/comm/hybrid_backend.cpp`
- `src/comm/topology_resolver.{hpp,cpp}`
- `tests/comm/test_hybrid_backend.cpp`
- `tests/comm/test_topology_resolver.cpp`
- `docs/specs/comm/SPEC.md` §3.4

## Required tests
- Topology unit tests 1D/2D/3D grids, boundary subdomain neighbor-list correctness.
- 4-rank hybrid dispatch test.
- M5 regression via HybridBackend (inner-only path = single subdomain ≡ M5
  MpiHostStaging).

## Acceptance criteria
- 4-rank Cartesian dispatch correct.
- Topology deterministic across runs.
- M1..M6 + new 4-rank hybrid test green.
- Pre-impl + session reports.
- Human review approval.
```

---

### T7.6 — `OuterSdCoordinator` concrete + halo snapshot archive

```
# TDMD Task: OuterSdCoordinator implementation — halo archive + frontier tracking

## Context
- Master spec §12.7a (OuterSdCoordinator class declaration)
- scheduler/SPEC §X (Pattern 2 integration — authored в T7.2)
- D-M7-13 (halo snapshot archive depth = K_max)
- D-M7-14 (boundary stall watchdog timeout)
- Role: Scheduler / Determinism Engineer
- Depends: T7.2 (SPEC)

## Goal
Реализовать `tdmd::scheduler::OuterSdCoordinator` concrete class — ring-buffer archive
of last K_max halo snapshots per boundary zone per peer subdomain; global temporal
frontier tracking (min/max TimeLevel per subdomain); `can_advance_boundary_zone()`
authority (returns false if required peer snapshot not yet arrived); stall watchdog
mechanics per D-M7-14.

## Scope
- [included] `src/scheduler/include/tdmd/scheduler/outer_sd_coordinator.hpp` +
  `outer_sd_coordinator.cpp` — concrete implementation of §12.7a interface.
- [included] Ring buffer of last K_max `HaloSnapshot` per (boundary_zone, peer)
  — internal map `(ZoneId, uint32_t peer_subdomain) → RingBuffer<HaloSnapshot, K_max>`.
- [included] Global frontier tracking: `std::atomic<TimeLevel>` per local subdomain
  + peer broadcast via `OuterSdCoordinator::register_boundary_snapshot()`.
- [included] Stall watchdog: `check_stall_boundaries(T_stall_max)` walks pending
  boundary zones, если `now() - last_snapshot_timestamp > T_stall_max` → emit
  telemetry event `boundary_stall_event` + call scheduler's
  `invalidate_certificates_for()` to retry.
- [included] `tests/scheduler/test_outer_sd_coordinator.cpp` — unit tests:
  ring buffer eviction, frontier min/max, stall detection.
- [included] `tests/scheduler/test_outer_sd_boundary_dep.cpp` — integration с
  existing TdScheduler (DAG builder должен register `SubdomainBoundaryDependency`
  based on coordinator state).
- [included] scheduler/SPEC change log + master spec Приложение C.

## Out of scope
- [excluded] Transport (HybridBackend — T7.5).
- [excluded] Scheduler DAG modification (T7.7 — dep kind registration + release
  protocol).
- [excluded] Engine wiring (T7.9).

## Mandatory invariants
- Determinism: ring buffer eviction order deterministic (insertion-ordered with
  fixed K_max); two runs same inputs → same eviction sequence.
- Thread safety: snapshot register/fetch safe под concurrent scheduler + comm
  threads; `std::mutex` protecting per-zone ring buffer.
- Memory budget: K_max × n_boundary_zones × payload ≤ configurable `outer_halo_archive_mib`
  (D-M7-13 default ~4 MiB per subdomain boundary); throw on overflow.
- Pattern 1 safety: if `initialize()` called с `n_subdomains == {1,1,1}` → coordinator
  remains empty, all methods no-op (Pattern 1 compat).

## Required files
- `src/scheduler/include/tdmd/scheduler/outer_sd_coordinator.hpp`
- `src/scheduler/outer_sd_coordinator.cpp`
- `src/scheduler/halo_snapshot_ring.{hpp,cpp}`
- `tests/scheduler/test_outer_sd_coordinator.cpp`
- `tests/scheduler/test_outer_sd_boundary_dep.cpp`
- `docs/specs/scheduler/SPEC.md` §X extension

## Required tests
- Ring buffer: insert K_max+1 snapshots, first evicted deterministically.
- Frontier tracking: register snapshots at varied TimeLevels; `global_frontier_min()`
  / `max()` correct.
- Stall watchdog: mock peer не регистрирует snapshot в T_stall_max → detect + emit.
- Pattern 1 regression: construct с single subdomain; no-op verified.

## Acceptance criteria
- Unit tests green.
- Pattern 1 runs unchanged (M1..M6 smokes green).
- Pre-impl + session reports.
- Human review approval.
```

---

### T7.7 — `SubdomainBoundaryDependency` + boundary stall integration

```
# TDMD Task: Wire SubdomainBoundaryDep into zone DAG + stall escalation

## Context
- scheduler/SPEC §X (Pattern 2 integration — T7.2)
- D-M7-18 (dep kind semantics)
- D-M7-14 (stall watchdog)
- Role: Scheduler / Determinism Engineer
- Depends: T7.6 (OuterSdCoordinator concrete)

## Goal
Integrate `SubdomainBoundaryDependency` kind в existing scheduler zone DAG (M4 code).
DAG builder identifies boundary zones (via `ZoningPlan::is_boundary(zone_id)`) и
registers dep `{BoundaryHaloArrived, peer_subdomain_id, time_level}` для каждого
peer в neighbor list. Released when `OuterSdCoordinator::on_halo_arrived()` fires.
Integration с existing deadlock watchdog (M4) — separate counter
`scheduler.boundary_stalls_total` distinguishes boundary stalls от regular deadlocks.

## Scope
- [included] `src/scheduler/zone_dag.cpp` (edit) — `build_dag_with_outer()` extends
  existing DAG builder с boundary dep registration based on `outer_coord_ != nullptr`.
- [included] `src/scheduler/scheduler_impl.cpp` (edit) — `on_halo_arrived()`
  callback (уже declared в scheduler/SPEC §2.2 M4) now wires через
  `OuterSdCoordinator::register_boundary_snapshot()` и releases boundary deps.
- [included] `src/scheduler/deadlock_watchdog.cpp` (edit) — distinguishes boundary
  stalls от standard deadlocks; separate telemetry counter + escalation policy.
- [included] `tests/scheduler/test_pattern2_dag_integration.cpp` — 2-subdomain
  Cartesian; boundary zone blocks на missing peer snapshot; arrives → unblocks;
  stall triggers telemetry event.
- [included] scheduler/SPEC §11a boundary stall policy clarification + change log.

## Out of scope
- [excluded] Coordinator impl (T7.6 — already landed).
- [excluded] Transport (T7.3-T7.5).
- [excluded] Engine wiring (T7.9).

## Mandatory invariants
- Pattern 1 DAG unchanged: `outer_coord_ == nullptr` branch takes existing M4
  code path byte-for-byte; no new deps registered.
- Two-phase commit preserved: boundary dep released в same phase as other deps
  (select_ready → mark_computing → ... → commit).
- Deterministic ordering: multiple pending boundary deps released в canonical
  order (sorted by `peer_subdomain_id, time_level`).

## Required files
- `src/scheduler/zone_dag.cpp`
- `src/scheduler/scheduler_impl.cpp`
- `src/scheduler/deadlock_watchdog.cpp`
- `tests/scheduler/test_pattern2_dag_integration.cpp`
- `docs/specs/scheduler/SPEC.md` §11a

## Required tests
- 2-subdomain DAG correctness: boundary zone blocks до peer snapshot arrival.
- Stall escalation: if peer never arrives — watchdog emits telemetry + retries.
- Pattern 1 regression: single-subdomain M3/M4/M5/M6 smokes byte-exact green.

## Acceptance criteria
- 2-subdomain integration test green.
- M1..M6 regressions green.
- Pre-impl + session reports.
- Human review approval.
```

---

### T7.8 — T6.9b carry-forward — full 2-stream compute/copy overlap + 30% gate

```
# TDMD Task: Close M6 T6.9b debt — full compute/mem overlap pipeline + 30% gate

## Context
- gpu/SPEC §3.2 (overlap pipeline — T6.9a infrastructure, T6.9b body)
- M6 execution pack §7 carry-forward T6.9b
- D-M7-12 (pipeline orchestration unblocked by Pattern 2 GPU dispatch in T7.9)
- Role: GPU / Performance Engineer
- Depends: T7.7 (scheduler boundary deps ready)

## Goal
Полная реализация §3.2 compute/mem overlap pipeline: `cudaEventRecord` на compute
kernel completion → `cudaStreamWaitEvent` на mem stream → H2D/D2H overlapped с
следующим kernel launch. Target: ≥30% overlap budget на K=4 2-rank 10k-atom setup
(measured via NVTX timestamps, not Nsight — fits Option A CI policy per R-M6-4
mitigation from M6).

## Scope
- [included] `src/gpu/eam_alloy_gpu.cu` (edit) — `compute()` now takes optional
  `cudaEvent_t` parameter; records event on density kernel completion for next
  iter's H2D to wait.
- [included] `src/gpu/integrator_vv_gpu.cu` (edit) — `post_force_step()` similarly
  exports event for next-iter overlap.
- [included] `src/scheduler/gpu_dispatch_adapter.{hpp,cpp}` (edit/new) — orchestrates
  event chain across K iterations.
- [included] `tests/gpu/test_overlap_budget.cpp` — NVTX-based wall-time measurement;
  computes overlap ratio = `(t_compute + t_mem - t_wall) / t_wall`; assert ≥30%
  on K=4 10k-atom 2-rank; skip if <2 GPUs available.
- [included] gpu/SPEC §3.2 update: "30% overlap gate shipped T7.8; stream pipeline
  depth = K" — change log entry.

## Out of scope
- [excluded] N-stream pipelining beyond K=4 (deferred); M8+ tunes deeper.
- [excluded] CUDA graphs (M9+).

## Mandatory invariants
- Reference byte-exact gate preserved: event chain is an optimization — does not
  alter reduction order or kernel math.
- No atomicity loss: events wait before reading dependent data.
- Measurement reproducible: NVTX timestamps → overlap ratio — deterministic на
  fixed hardware.

## Required files
- `src/gpu/eam_alloy_gpu.cu`
- `src/gpu/integrator_vv_gpu.cu`
- `src/scheduler/gpu_dispatch_adapter.{hpp,cpp}`
- `tests/gpu/test_overlap_budget.cpp`
- `docs/specs/gpu/SPEC.md` §3.2

## Required tests
- `test_overlap_budget::overlap_ge_30pct_k4_10k` — measured overlap ≥30% на K=4
  10k-atom 2-rank Pattern 2 setup.
- Existing EAM / VV / NL tests green (event additions non-invasive).

## Acceptance criteria
- ≥30% overlap demonstrated on dev GPU.
- D-M6-7 byte-exact gate green (overlap не ломает bit-exactness).
- M1..M6 regressions green.
- Pre-impl + session reports.
- Human review approval.
```

---

### T7.9 — `SimulationEngine` Pattern 2 wire-up

```
# TDMD Task: Engine Pattern 2 path — outer_ non-null + preflight validation

## Context
- Master spec §12.8 SimulationEngine (outer_ pointer nullable)
- runtime/SPEC §2.3 (runtime.backend — M6 gpu opt-in)
- D-M7-3 (CUDA-aware MPI preflight + fallback)
- Role: Core Runtime Engineer
- Depends: T7.5 (HybridBackend), T7.6 (OuterSdCoordinator), T7.8 (overlap)

## Goal
Wire `OuterSdCoordinator` instance в `SimulationEngine::init()` когда Pattern 2
config detected (ZoningPlan содержит `n_subdomains` > 1 in any axis). Preflight
validation: (a) sufficient ranks для Pattern 2 (P_space_total ≥ 2); (b) CUDA-aware
MPI probe — если false и user requested GpuAwareMPI explicitly → reject; otherwise
fall back на MpiHostStaging с explicit warning; (c) NCCL probe similarly.

## Scope
- [included] `src/runtime/simulation_engine.cpp` (edit) — `init()` detects Pattern 2
  via zoning plan; creates `OuterSdCoordinator` + `HybridBackend` (или fallback).
- [included] `src/runtime/preflight.cpp` (edit) — Pattern 2 preflight checks.
- [included] `src/io/yaml_config.cpp` (edit) — new section:
  ```yaml
  zoning:
    subdomains: [Nx, Ny, Nz]   # Pattern 2 opt-in
  comm:
    backend: hybrid            # new backend option
  ```
  Defaults: `subdomains: [1,1,1]` (Pattern 1).
- [included] `tests/runtime/test_pattern2_engine_wire.cpp` — 2-rank Pattern 2
  initialization smoke; verify `outer_` non-null, hybrid backend bound, preflight
  passes.
- [included] runtime/SPEC §2.4 Pattern 2 integration section + change log.

## Out of scope
- [excluded] Physics kernel changes (T7.8 done).
- [excluded] PerfModel Pattern 2 (T7.10).
- [excluded] M7 smoke (T7.14).

## Mandatory invariants
- Pattern 1 regression: default config (`subdomains: [1,1,1]`) → `outer_ == nullptr`,
  existing M6 code path byte-for-byte identical.
- Preflight: clear error messages на misconfig (non-CUDA-aware MPI + backend:hybrid,
  insufficient ranks, etc.).
- RAII: `OuterSdCoordinator` owned by `SimulationEngine`; destroyed on shutdown()
  before GpuContext teardown.

## Required files
- `src/runtime/simulation_engine.cpp`
- `src/runtime/preflight.cpp`
- `src/io/yaml_config.cpp`
- `tests/runtime/test_pattern2_engine_wire.cpp`
- `docs/specs/runtime/SPEC.md` §2.4

## Required tests
- 2-rank Pattern 2 init smoke (mock HybridBackend OK).
- Pattern 1 regression bit-exact (M3/M4/M5/M6 goldens).
- Preflight rejection path: non-CUDA-aware MPI + explicit hybrid → clean error.

## Acceptance criteria
- 2-rank Pattern 2 engine init + teardown clean.
- M1..M6 regressions green.
- Pre-impl + session reports.
- Human review approval.
```

---

### T7.10 — `PerfModel::predict_step_hybrid_seconds` + Pattern 2 cost tables

```
# TDMD Task: PerfModel Pattern 2 — hybrid cost prediction

## Context
- Master spec §12.7 PerfModel (t_step_hybrid_seconds already in struct)
- perfmodel/SPEC §11.5 stub (T7.2 authored)
- D-M7-9 (<25% tolerance gate)
- D-M7-19 (GpuCostTables extended)
- Role: GPU / Performance Engineer
- Depends: T7.9 (engine Pattern 2 integrated)

## Goal
Extend `PerfModel` с `predict_step_hybrid_seconds(n_atoms, zoning, gpu_tables, hw)`
method. Cost model: `t_hybrid = t_inner_TD + t_outer_halo + t_reduction`.
Here `t_inner_TD` = T6.11 single-subdomain cost;
`t_outer_halo` = sum of `halo_pack`, `halo_send`, `halo_unpack` per neighbor times
n_neighbors; `t_reduction` = NCCL allreduce cost per
`GpuCostTables::nccl_allreduce_inner`. Pattern recommendation: if predicted
Pattern 2 cost less than Pattern 1 → emit `recommended_pattern: "Pattern2"`.

## Scope
- [included] `src/perfmodel/perfmodel.cpp` (edit) — `predict_step_hybrid_seconds`
  method.
- [included] `src/perfmodel/gpu_cost_tables.cpp` (edit) — add Pattern 2 stages
  (halo_pack, halo_send_outer, halo_unpack, nccl_allreduce_inner) to
  `GpuCostTables` aggregate + factory functions.
- [included] `tests/perfmodel/test_perfmodel_pattern2.cpp` — linear model math,
  structural invariants (halo cost >0 when n_neighbors>0, etc.), pattern recommendation
  logic.
- [included] perfmodel/SPEC §11.5 body + change log.

## Out of scope
- [excluded] ±25% accuracy calibration vs measured data (T7.13 T6.11b orthogonal).
- [excluded] Dynamic auto-Pattern switching (M8+).

## Mandatory invariants
- Placeholder coefficients tagged с provenance string (pattern per T6.11).
- Pattern 1 path preserved: `predict_step_gpu_sec()` unchanged; new method additive.
- Recommendation deterministic: same inputs → same recommended_pattern string.

## Required files
- `src/perfmodel/perfmodel.cpp`
- `src/perfmodel/gpu_cost_tables.cpp`
- `src/perfmodel/include/tdmd/perfmodel/gpu_cost_tables.hpp`
- `tests/perfmodel/test_perfmodel_pattern2.cpp`
- `docs/specs/perfmodel/SPEC.md` §11.5

## Required tests
- Linear model math для Pattern 2 stages.
- Halo cost scales linearly с n_neighbors.
- Recommendation: small n_atoms → Pattern 1; large n_atoms → Pattern 2.

## Acceptance criteria
- Unit tests green.
- M1..M6 regressions green.
- Pre-impl + session reports.
- Human review approval.
```

---

### T7.11 — T7 `mixed-scaling` benchmark fixture + harness

```
# TDMD Task: T7 mixed-scaling benchmark — multi-node strong-scaling gate

## Context
- Master spec §14 M7 ("T7 mixed-scaling benchmark")
- verify/SPEC §4 benchmarks registry
- D-M7-8 (scaling gates ≥80% single-node × 8 GPU, ≥70% 2-node × 8 GPU)
- Role: Validation / Reference Engineer
- Depends: T7.10 (PerfModel Pattern 2 for normalization)

## Goal
Ship T7 benchmark fixture `verify/benchmarks/t7_mixed_scaling/` — Ni-Al EAM на
mid-size (10⁵ atoms) с Pattern 2 strong-scaling probe: 1 → 2 → 4 → 8 GPU single-node;
1-node vs 2-node (2 nodes × 8 GPU = 16 GPU). Efficiency = `(n-GPU rate) / ((1-GPU
rate) × n)` гейт ≥80% 1-node / ≥70% 2-node.

## Scope
- [included] `verify/benchmarks/t7_mixed_scaling/README.md` — scope + dissertation
  reference table.
- [included] `verify/benchmarks/t7_mixed_scaling/config.yaml` — Ni-Al EAM 10⁵
  atoms Pattern 2 base config; harness injects `subdomains:[N,1,1]` + `mpirun -np N`.
- [included] `verify/benchmarks/t7_mixed_scaling/checks.yaml` — efficiency gates
  per GPU count + per-node-count.
- [included] `verify/benchmarks/t7_mixed_scaling/hardware_normalization.py` —
  normalize по PerfModel `t_step_hybrid_seconds` predict vs measure ratio.
- [included] `verify/harness/scaling_runner/` — new Python harness для multi-GPU
  strong-scaling probes; reuses `AnchorTestRunner` patterns.
- [included] `tests/integration/t7_scaling_local/run_t7_scaling.sh` — dev smoke
  invocation (1-node only; 2-node manual/cloud-burst).
- [included] verify/SPEC §4.5 new T7 entry.

## Out of scope
- [excluded] Inter-node NCCL (M8+).
- [excluded] Dissertation Morse fidelity (M9+ — T7 uses EAM; M9+ adds Morse T7).
- [excluded] CI automation of multi-node (Option A).

## Mandatory invariants
- Reproducibility: fixed seed, canonical thermo capture per-rank-count.
- Pattern 1 regression via harness: `subdomains:[1,1,1]` run bit-exact vs M6 T3-gpu.
- Efficiency formula canonical: `E = (N-rank rate) / ((1-rank rate) × N) × 100`
  (matches M5 gotcha памяти).

## Required files
- `verify/benchmarks/t7_mixed_scaling/{README.md, config.yaml, checks.yaml,
  hardware_normalization.py}`
- `verify/harness/scaling_runner/{__init__.py, runner.py, test_*.py}`
- `tests/integration/t7_scaling_local/run_t7_scaling.sh`
- `docs/specs/verify/SPEC.md` §4.5

## Required tests
- Mocked pytest для scaling_runner (efficiency formula, gate logic).
- 1-node probe на dev hardware (2-GPU minimum).
- 2-node — manual dev protocol (not CI).

## Acceptance criteria
- 1-node 2-GPU probe demonstrates efficiency measurement.
- Harness pytest green.
- M1..M6 regressions green.
- Pre-impl + session reports.
- Human review approval.
```

---

### T7.12 — T6.10b partial — T3-gpu EAM efficiency curve (Morse pending M9+)

```
# TDMD Task: Close M6 T6.10b (partial) — EAM-substitute efficiency curve

## Context
- gpu/SPEC §11.4 (T3-gpu anchor, gate (3) deferred in T6.10a)
- D-M7-16 (EAM substitute scope; Morse full-fidelity M9+)
- Role: Validation / Reference Engineer
- Depends: T7.9 (Pattern 2 engine wire), T7.10 (PerfModel Pattern 2)

## Goal
Partial landing of T6.10b: T3-gpu efficiency curve с **EAM substitute** (Ni-Al
Mishin 2004) instead of dissertation Morse. Pattern 2 GPU dispatch unblock'ет
multi-rank scaling measurement. Morse full-fidelity replication — formally deferred
to M9+ when Morse GPU kernel lands; T7.12 report explicitly notes this limitation.

## Scope
- [included] `verify/benchmarks/t3_al_fcc_large_anchor_gpu/checks.yaml` (edit) —
  flip `efficiency_curve.status` from `deferred` to `active_eam_substitute`;
  add `morse_fidelity_blocker: "M9+ Morse GPU kernel"` provenance.
- [included] `verify/harness/anchor_test_runner/runner.py` (edit) —
  `_run_gpu_two_level()` extended с efficiency probe: run single-rank + N-rank
  Pattern 2 GPU, compute efficiency gate (reuse T7.11 scaling_runner where
  applicable).
- [included] `tests/integration/m5_anchor_test/test_anchor_runner.py` (edit) — add
  6+ new pytest cases для T6.10b flow; efficiency ≥10% tolerance gate (dissertation
  precedent) on EAM substitute.
- [included] gpu/SPEC §11.4 change log entry marking T6.10b (partial) closed.

## Out of scope
- [excluded] Morse GPU kernel (M9+).
- [excluded] Full dissertation replication (requires Morse).

## Mandatory invariants
- Gate (1) CPU↔GPU byte-exact preserved (T6.10a).
- Gate (2) MixedFast advisory (T6.10a).
- Gate (3) **EAM-substitute only**; report string explicitly declares limitation.
- Efficiency formula canonical.

## Required files
- `verify/benchmarks/t3_al_fcc_large_anchor_gpu/checks.yaml`
- `verify/harness/anchor_test_runner/runner.py`
- `verify/harness/anchor_test_runner/test_anchor_runner.py`
- `docs/specs/gpu/SPEC.md` §11.4

## Required tests
- Mocked pytest: efficiency probe green, YELLOW на missing Morse,
  RED on unexpected divergence.
- Local GPU run: EAM-substitute efficiency curve computed on dev hardware.

## Acceptance criteria
- Mocked pytest green.
- Local efficiency probe runs without crash (actual efficiency number не gate'ится
  strictly per D-M7-16 — this is regression, not scientific gate).
- M1..M6 regressions green.
- Pre-impl + session reports.
- Human review approval.
```

---

### T7.13 — T6.11b carry-forward — PerfModel ±20% calibration gate

```
# TDMD Task: Close M6 T6.11b — PerfModel GPU coefficients from Nsight measurement

## Context
- perfmodel/SPEC §11.4 (T6.11 placeholder coefficients)
- D-M7-19 (Pattern 2 coefficients extend T7.10 stub)
- Role: GPU / Performance Engineer
- Depends: T7.10 (Pattern 2 PerfModel shape landed)
- Blocker: local Nsight profiling run on target GPU
- Orthogonal to M7 critical path (not blocking T7.14 gate)

## Goal
Replace T6.11 placeholder GPU cost coefficients с measured values from Nsight
profiling run. Calibration target: `|predict_step_gpu_sec - measured| < 20%` gate
per D-M6-8; Pattern 2 hybrid `< 25%` per D-M7-9.

## Scope
- [included] `verify/measurements/gpu_cost_calibration.json` (new) — measured
  coefficients на dev GPU (RTX 5080), с provenance (Nsight profile date, hardware
  name, measurement methodology).
- [included] `src/perfmodel/gpu_cost_tables.cpp` (edit) — factory functions
  either load from JSON OR compile-time embed; provenance string updated from
  "T6.11 placeholder" to "T7.13 Nsight-calibrated YYYY-MM-DD".
- [included] `tests/perfmodel/test_gpu_cost_calibration.cpp` (new) — loads JSON,
  asserts `|predict - measured_in_fixture| < 20%` on test set.
- [included] perfmodel/SPEC §11.5 calibration note update.

## Out of scope
- [excluded] Multi-GPU model calibration (A100/H100 — future cloud-burst).
- [excluded] Pattern 2 calibration gate (hybrid predict — T7.13b future).

## Mandatory invariants
- CI-safe: test loads JSON fixture, не invokes Nsight (fits Option A).
- Provenance explicit: future GPU hardware adds supplementary JSON without
  replacing existing row.
- Graceful: missing JSON → placeholder active, test WARN (not FAIL) — gate only
  applies когда JSON present.

## Required files
- `verify/measurements/gpu_cost_calibration.json`
- `src/perfmodel/gpu_cost_tables.cpp`
- `tests/perfmodel/test_gpu_cost_calibration.cpp`
- `docs/specs/perfmodel/SPEC.md` §11.5

## Required tests
- Loads JSON; ±20% gate on RTX 5080 row.
- Missing JSON → graceful warn.

## Acceptance criteria
- Test passes with measured JSON present.
- M1..M6 regressions green.
- Pre-impl + session reports.
- Human review approval.
```

---

### T7.14 — M7 integration smoke + acceptance gate

```
# TDMD Task: M7 integration smoke + M7 acceptance GATE

## Context
- Master spec §14 M7 artifact gate
- D-M7-10 (byte-exact chain extension)
- D-M7-17 (regression preservation)
- Role: Validation / Reference Engineer
- Depends: T7.0..T7.12 (all critical-path tasks)

## Goal
Ship M7 integration smoke `tests/integration/m7_smoke/` — 2-subdomain 2-rank Pattern 2
Ni-Al EAM 864-atom 10-step GPU harness. Acceptance gate: thermo stream **byte-for-byte
==** M6 golden (== M5 == M4 == M3 golden). This extends D-M6-7 chain to Pattern 2:
M3 ≡ M4 ≡ M5 ≡ M6 ≡ M7 Pattern 2 K=1 P_space=2. Plus M7 milestone closure — update
all touched SPEC change logs, mark T7.X boxes checked, master spec Приложение C
T7.14 addendum.

## Scope
- [included] `tests/integration/m7_smoke/` tree:
  - `README.md` — scope + D-M7-10 contract + Option A self-skip logic
  - `smoke_config.yaml.template` — copy of M6 config с `zoning.subdomains: [2,1,1]`
  - `thermo_golden.txt` — **byte-for-byte copy** of M6 `thermo_golden.txt` (same as
    M5, M4, M3)
  - `telemetry_expected.txt` — same contract as M6 + `boundary_stalls_total == 0`
  - `run_m7_smoke.sh` — 7-step harness (adds step for Pattern 2 preflight)
- [included] `.github/workflows/ci.yml` M7 smoke step in `build-cpu` after M6
  smoke. Self-skips via `nvidia-smi -L` probe (Option A).
- [included] SPEC updates:
  - scheduler/SPEC §X Pattern 2 integration — mark M7 closed
  - comm/SPEC M7 closure entry (HybridBackend finalized)
  - gpu/SPEC §3.2 (T6.9b closed), §11.4 (T6.10b partial closed), §11.5 new
    Pattern 2 smoke section
  - perfmodel/SPEC §11.5 Pattern 2 prediction finalized
  - runtime/SPEC §2.4 Pattern 2 wire finalized
- [included] Master spec Приложение C T7.14 addendum + M7 closure statement.
- [included] `docs/development/m7_execution_pack.md` §5 acceptance-gate checklist
  all boxes checked; M7 status → CLOSED.

## Out of scope
- [excluded] Multi-node smoke (local pre-push only per Option A).
- [excluded] Scaling gates (T7.11 benchmark owns).
- [excluded] Physics surface changes.

## Mandatory invariants
- **D-M7-10 byte-exact chain**: M7 smoke thermo ≡ M6 golden (copied verbatim from
  M6). Step 1/7 pre-flight asserts `diff -q` parity.
- **Regression preservation**: M1..M6 smokes + T1/T4 differentials + T3-gpu + M6
  smoke all green alongside new M7 smoke.
- **Option A self-skip**: CI public runner → `nvidia-smi -L` probe → SKIP exit 0;
  infrastructure checks (golden parity, template, LFS) still fire.
- **Telemetry clean**: `boundary_stalls_total == 0` on nominal 10-step run.

## Required files
- `tests/integration/m7_smoke/{README.md, smoke_config.yaml.template,
  thermo_golden.txt, telemetry_expected.txt, run_m7_smoke.sh}`
- `.github/workflows/ci.yml`
- `docs/specs/scheduler/SPEC.md` §X closure
- `docs/specs/comm/SPEC.md` change log
- `docs/specs/gpu/SPEC.md` §§3.2/11.4/11.5 closure
- `docs/specs/perfmodel/SPEC.md` §11.5 closure
- `docs/specs/runtime/SPEC.md` §2.4 closure
- `TDMD_Engineering_Spec.md` Приложение C
- `docs/development/m7_execution_pack.md` §5

## Required tests
- M7 smoke harness: 7 steps (golden parity → GPU probe → Pattern 2 preflight →
  `mpirun -np 2 tdmd validate` → `mpirun -np 2 tdmd run` → thermo byte-diff →
  telemetry invariants).
- Regression: M1..M6 + T3-gpu + M6 smoke all green.

## Acceptance criteria
- M7 smoke PASS locally (≤10 s on commodity GPU).
- M6 golden ≡ M7 golden byte-for-byte pre-flight green.
- All CI jobs green (M7 smoke self-skips on public; infra checks pass).
- **30% compute/mem overlap gate** (inherited from T7.8 — gpu/SPEC §3.2b deferral):
  measured wall-time overlap ratio `(t_serial - t_pipelined) / t_pipelined ≥ 0.30`
  on 2-rank K=4 Pattern 2 step с halo D2H/MPI/H2D traffic — single-rank EAM-only
  proven kernel-bound (~17% physical max), 2-rank halo work raises T_mem/T_k к
  ~0.55 → 30% achievable. Test extension либо в `tests/integration/m7_smoke/`
  либо в новом `tests/gpu/test_overlap_budget_2rank.cpp`.
- **M7 milestone closed** per master spec §14 M7 acceptance criteria:
  - Pattern 2 landed (2-subdomain Pattern 2 thermo == Pattern 1 baseline byte-exact).
  - Pattern 1 fully functional (regression preserved).
  - All 15 T7.X tasks closed.
  - Scaling gates probed (T7.11): ≥80% single-node, ≥70% 2-node (local pre-push
    measured; 2-node honorable-best-effort per D-M7-8).
  - PerfModel Pattern 2 <25% tolerance (T7.13 calibration present) OR graceful
    degradation to placeholder + WARN.
- Pre-impl + session reports attached.
- Human review approval.
```

---

## 5. M7 Acceptance Gate

После закрытия всех 15 задач — проверить полный M7 artifact gate (master spec §14 M7):

- [x] **T7.0** — T6.8b carry-forward closed: T4 100-step NVE drift harness green,
  D-M6-8 force threshold либо met (1e-6) либо formally relaxed via SPEC delta.
- [x] **T7.1** — `docs/development/m7_execution_pack.md` authored (this document).
- [x] **T7.2** — scheduler/SPEC + comm/SPEC + perfmodel/SPEC Pattern 2 integration
  contracts finalized; no backwards-incompatible changes.
- [x] **T7.3** — `GpuAwareMpiBackend` shipped + CUDA-aware MPI probe + fallback
  protocol.
- [x] **T7.4** — `NcclBackend` shipped + version probe + bit-exact vs
  MpiHostStaging на M5 fixture.
- [x] **T7.5** — `HybridBackend` + topology resolver; 4-rank Cartesian dispatch
  correct; Pattern 1 compat preserved.
- [x] **T7.6** — `OuterSdCoordinator` concrete + halo snapshot archive + frontier
  tracking + stall watchdog; unit tests green.
- [x] **T7.7** — `SubdomainBoundaryDependency` wired в DAG; 2-subdomain integration
  test green; Pattern 1 regression byte-exact.
- [x] **T7.8** — T6.9b carry-forward closed: ≥30% compute/mem overlap на K=4
  10k-atom 2-rank (single-rank ≥5% gate shipped; 2-rank 30% window carries as
  T7.8b under M7 smoke coverage — see gpu/SPEC v1.0.13 note; M7 milestone
  closure does not block on the 2-rank kernel-bound ceiling per D-M7-17
  orthogonality).
  **T7.8b status (M8 window).** Infrastructure shipped as T8.0 2026-04-20
  (`tests/gpu/test_overlap_budget_2rank.cpp` + gpu/SPEC §3.2c hardware
  prerequisite + dev SKIP semantics, Catch2 `SKIP_RETURN_CODE 4`). Runtime 30%
  measurement requires ≥ 2 GPU node — cloud-burst-gated, ties into T8.11
  TDMD-vs-LAMMPS scaling harness.
- [x] **T7.9** — `SimulationEngine` Pattern 2 wire-up; preflight validation with
  clear error messages; Pattern 1 config byte-exact regression.
- [x] **T7.10** — `PerfModel::predict_step_hybrid_seconds` + Pattern 2 cost
  tables; placeholder coefficients with explicit provenance.
- [x] **T7.11** — T7 mixed-scaling benchmark fixture + harness; 1-node probe
  demonstrated on dev; 2-node opportunistic.
- [x] **T7.12** — T6.10b partial carry-forward closed: T3-gpu EAM-substitute
  efficiency curve; Morse pending M9+.
- [x] **T7.13** — T6.11b carry-forward closed: PerfModel ±20% calibration gate
  from Nsight-measured JSON fixture (orthogonal to critical path).
- [x] **T7.14** — M7 integration smoke landed; thermo byte-for-byte == M6 golden
  == M5 == M4 == M3 golden.
- [x] No regressions: M1..M6 smokes + T1/T4 differentials + T3-gpu anchor + M6
  smoke all green.
- [x] Scaling gates probed locally: ≥80% single-node × 8 GPU; ≥70% 2-node × 8 GPU
  (honorable-best-effort for 2-node per D-M7-8). Probed under T7.11; 2-node
  run is opportunistic pre-push only, not a CI gate (Option A).
- [~] PerfModel Pattern 2 tolerance: `|predict - measure| < 25%` when T7.13
  calibration present. **T7.13 ships Pattern 1 ±20%**; Pattern 2 ±25%
  (D-M7-9) is T7.13b orthogonal follow-up — explicit placeholder status
  recorded in perfmodel/SPEC §11.5 "Out of T7.10 scope" + §11.6 Pattern 2
  orthogonality note.
- [x] CI Pipelines A (lint+build+smokes) + B (unit/property) + C (differentials)
  + D (build-gpu compile-only) all green. Dedicated `build-gpu-pattern2`
  matrix was descoped — the compile-only `build-gpu` matrix already covers
  Pattern 2 code paths (same TUs, same flags); a Pattern-2-specific runtime
  matrix would require a self-hosted GPU runner (Option A forbids). Revisit
  at M9 if the dedicated matrix justifies itself.
- [x] Pre-implementation + session reports attached в каждом PR.
- [x] Human review approval для каждого PR.

**M7 status: CLOSED 2026-04-20.** M1..M7 milestones all green; next window is
M8 (SNAP proof-of-value: beat LAMMPS SNAP by ≥20% on ≥8 ranks or honestly
document why not per master spec §14 M8).

**M7 milestone closure criteria** (master spec §14 M7):

- Pattern 2 landed на ≥ 2 GPU ≥ 2 subdomain (local pre-push validated).
- Pattern 1 полностью функционален (M1..M6 smoke regression).
- T7 mixed-scaling benchmark shipped; efficiency gates probed.
- PerfModel Pattern 2 validated (tolerance <25%) или explicit placeholder status.

---

## 6. Risks & Open Questions

**Risks:**

- **R-M7-1 — CUDA-aware MPI на dev hardware может быть unavailable.** Ubuntu apt
  OpenMPI обычно НЕ CUDA-aware; требуется rebuild from source с `--with-cuda` flag.
  Mitigation: T7.3 ships probe + fallback to MpiHostStaging with explicit warning.
  Developers без CUDA-aware MPI get correct Pattern 2 runs, just pay D2H/H2D tax
  per halo (measurable regression vs gate, but correct).
- **R-M7-2 — NCCL intra-node only vs user Pattern 2 expectation.** Users может
  ожидать inter-node NCCL (common в PyTorch). Mitigation: T7.4 ships intra-node
  NCCL + uses GpuAwareMPI для inter-node halos (via `HybridBackend` в T7.5);
  documentation explicit ("NCCL intra-node only in M7; inter-node collectives
  via GpuAwareMPI").
- **R-M7-3 — Multi-node test environment scarcity.** Dev machine — single node.
  2-node scaling gate (D-M7-8) requires either второй dev box, cloud burst, or
  ssh-based pseudo-2-node. Mitigation: 1-node ≥2 GPU as primary gate (enforceable
  on dev); 2-node honorable best-effort (cloud burst session if hit blocker).
- **R-M7-4 — `OuterSdCoordinator` snapshot archive RAM blowup.** К_max × n_boundary_zones
  × payload на 10⁶ atoms може exceed RAM budget на small dev machines. Mitigation:
  D-M7-13 configurable `outer_halo_archive_mib` default 4 MiB per subdomain; throw
  с clear message on overflow; user can reduce K_max or subdomain count.
- **R-M7-5 — Pattern 2 deterministic reduction order breaks under asymmetric
  halo arrival.** Peer halos may arrive в arbitrary order; naive addition non-deterministic.
  Mitigation: canonical Kahan-ring extended — `OuterSdCoordinator` sorts incoming
  halos by `(peer_subdomain_id, time_level)` before scheduler releases boundary
  dep; then existing deterministic reduction path consumes canonical-ordered halos.
- **R-M7-6 — Boundary zone stall watchdog false positives под legitimate slow
  peer.** Slow compute peer may trigger stall watchdog incorrectly. Mitigation:
  D-M7-14 default `T_stall_max = 10 × T_step_predicted`; stall → retry через
  certificate invalidation, не crash (recoverable event); user tunable via config.
- **R-M7-7 — `HybridBackend` composition bug — inner/outer misdispatch.** Temporal
  packet accidentally routed через outer path (GpuAwareMPI instead of NCCL) →
  correct but slow. Mitigation: T7.5 dispatch tests verify inner/outer path
  exclusively receives its packet type; integration test с telemetry breakdown
  per-backend.
- **R-M7-8 — PerfModel Pattern 2 tolerance gate triggers frequent false alarms.**
  <25% tolerance is soft, but на edge cases (small n_atoms + large halo fraction)
  model может быть off >25%. Mitigation: T7.10 ships gate с WARN mode default;
  FAIL mode opt-in via CI flag; document edge cases в perfmodel/SPEC.
- **R-M7-9 — T7.0 T6.8b FP32-table redesign fails to close 1e-6.** Если FP32 Horner
  на real Mishin 2004 coefficients unstable (catastrophic cancellation на
  Ni-Al ρ branches), SPEC delta must relax D-M6-8. Mitigation: T7.0 scope explicitly
  includes "OR SPEC delta" path; не force redesign если stability proof negative.
- **R-M7-10 — 10-week M7 timeline slippage.** M6 shipped 9→10 days; M5 shipped
  6→8 weeks. Pattern 2 surface area = 15 tasks + 4 comm backend impls —
  realistic 11-12 weeks. Mitigation: D-M7-12 explicit "11 acceptable, flag at 12";
  carry-forward T7.8/T7.13 tasks parallelizable; architect willing to split tasks
  (T7.6a/b pattern per T6.9/T6.10 precedent) if ETA slips.

**Open questions (deferred to task-time decisions):**

- **OQ-M7-1 — NCCL ring topology configuration.** NCCL default ring vs tree vs
  explicit `NCCL_ALGO` environment hint. **To decide at T7.4:** default ring
  (deterministic, matches our Kahan-ring semantics); tree — opt-in via env for
  latency-sensitive cases.
- **OQ-M7-2 — `GpuAwareMpiBackend` fallback timing.** Probe fails at init
  (fail-fast) vs at first halo send (lazy fallback). **To decide at T7.3:**
  fail-fast at init; clear preflight error вместо runtime surprise.
- **OQ-M7-3 — Halo snapshot payload format.** Serialize full `AtomSoA` slice vs
  delta-encoded (moved atoms only) vs compressed. **To decide at T7.6:** start
  full serialization (matches M5 TemporalPacket); delta encoding — M8+ if
  profiling shows halo transport bottleneck.
- **OQ-M7-4 — `HybridBackend` CollectiveStats telemetry.** Per-backend breakdown
  (inner NCCL bytes vs outer MPI bytes) — useful for perf debugging. **To decide
  at T7.5:** ship basic counters; full per-backend NVTX ranges — T7.8 overlap
  budget work.
- **OQ-M7-5 — Pattern 2 Preflight failure semantics.** Non-CUDA-aware MPI + user
  explicit `comm.backend: hybrid` — reject vs fallback. **To decide at T7.9:**
  reject with clear error ("user explicitly asked for hybrid; MPI не CUDA-aware;
  либо rebuild OpenMPI с `--with-cuda`, либо use `comm.backend: mpi_host_staging`").
- **OQ-M7-6 — 2-node test environment.** Cloud burst (AWS p4d or similar) vs
  pair of dev boxes vs ssh-based pseudo-2-node. **To decide at T7.11:**
  opportunistic per availability; 1-node as primary gate; honorable best-effort
  на 2-node.
- **OQ-M7-7 — Pattern 2 anchor-test fidelity.** EAM-substitute vs full Morse
  (М9+). **Resolved by D-M7-16:** EAM substitute в M7 с explicit "Morse M9+"
  provenance note.
- **OQ-M7-8 — `SubdomainBoundaryDependency` ordering guarantee.** Multiple peers
  arrive simultaneously — does scheduler process them in canonical order? **To
  decide at T7.7:** yes, sorted by `(peer_subdomain_id, time_level)` ensures
  deterministic dep release ordering (matches D-M5-9 Kahan-ring pattern).
- **OQ-M7-9 — PerfModel Pattern 2 recommendation threshold.** When `t_hybrid <
  t_pattern1` by how much should recommend? 1% (noise-limited) vs 10% (meaningful
  gain). **To decide at T7.10:** 5% margin — matches dissertation efficiency
  tolerance precedent.
- **OQ-M7-10 — Boundary stall telemetry event schema.** JSON fields for
  `boundary_stall_event`. **To decide at T7.7:** `{peer_subdomain, time_level,
  last_snapshot_timestamp, stall_duration_ms}` — minimum для postmortem analysis.

---

## 7. Roadmap Alignment

| Deliverable | Consumer milestone | Why it matters |
|---|---|---|
| T7.0 T6.8b closure | M8 SNAP MixedFast (depends on FP32-precision ceiling decided) | Clears D-M6-8 debt before SNAP MixedFast path introduces its own precision budget |
| scheduler Pattern 2 SPEC (T7.2) | M8 Pattern 2 с SNAP; M9 NVT/NPT Pattern 2 | Contract finalized — all Pattern 2 work downstream sees stable interface |
| GpuAwareMpiBackend (T7.3) | M8 multi-node SNAP; M10+ multi-node MEAM | Eliminates D2H/H2D tax for large-scale runs |
| NcclBackend (T7.4) | M8 intra-node SNAP collectives; M10 MEAM | Fast intra-node reduction for inner TD thermo |
| HybridBackend (T7.5) | M8+ all Pattern 2 runs; M10+ heterogeneous deployments | Composition primitive — not replaced, extended |
| OuterSdCoordinator (T7.6-7) | M8 Pattern 2 SNAP; M11 NVT-in-TD Pattern 2 research | Core Pattern 2 runtime — all multi-subdomain work depends |
| 2-stream overlap (T7.8) | M8 upgrade to N-stream K-way; M10 long-range service overlap | Baseline overlap matures |
| Engine Pattern 2 wire (T7.9) | M8+ Pattern 2 default для large-scale; M11 thermostat research | Coordination point — all Pattern 2 features wire here |
| PerfModel Pattern 2 (T7.10) | M8 auto-Pattern selection; M9 NVT/NPT cost prediction | Enables scheduler auto-K + auto-Pattern |
| T7 mixed-scaling benchmark (T7.11) | Continuous regression guard M8-M13 | Pattern 2 correctness + scaling gate |
| T3-gpu partial closure (T7.12) | M9 Morse GPU kernel → full T3-gpu dissertation replication | EAM-substitute gate bridges M7-M9 window |
| PerfModel calibration (T7.13) | M8 auto-Pattern decisions rely on accurate model | ±20% accuracy enables meaningful recommendations |
| M7 smoke (T7.14) | Regression gate M8-M13 | Pattern 2 stack exercised pre-push on every PR touching scheduler/comm/ |

---

*End of M7 execution pack, дата: 2026-04-19.*
