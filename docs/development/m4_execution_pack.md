# M4 Execution Pack

**Document:** `docs/development/m4_execution_pack.md`
**Status:** draft, awaiting human review
**Parent:** `TDMD_Engineering_Spec.md` §14 (M4), `docs/specs/scheduler/SPEC.md` v1.0, `docs/development/claude_code_playbook.md` §3
**Milestone:** M4 — Deterministic TD scheduler (single-node, CPU) (8 нед.)
**Created:** 2026-04-18
**Author:** Architect / Spec Steward role (Claude Opus 4.7)

---

## 0. Purpose

Этот документ декомпозирует milestone **M4** master spec'а §14 на **11 PR-size задач**, каждая сформулирована по каноническому шаблону `claude_code_playbook.md` §3.1. Документ — **process artifact**, не SPEC delta.

M4 — **сердце TDMD**: первый рабочий детерминированный TD-scheduler. После M4 движок умеет: (а) строить `SafetyCertificate` для `(zone, time_level)` пары и доказуемо монотонно invalidat'ить их на изменениях state/neighbor/dt; (б) вести `ZoneState`-машину с checked переходами, гарантирующими все семь инвариантов §13.4 (I1–I7); (в) детерминистически выбирать ready tasks через `CausalWavefrontScheduler::select_ready_tasks()`; (г) коммитить зоны по two-phase протоколу без losing intermediate state; (д) запускать TD-enabled run loop через `SimulationEngine` на K=1 single-rank, байт-экзактно эквивалентный legacy NVE path'у M1–M3.

M4 — **первый milestone, где scheduler реально принимает решения**. До M4 движок был последовательной NVE-петлёй; после M4 — time-decomposition pipeline с явными safety certificates и канонизированным обходом.

**Conceptual leap** от M3 к M4:

- M3 = "decomposition is correct" (how to slice the box, how to walk the zones).
- **M4 = "scheduling is correct"** (which zone runs when, respecting causal dependencies, with a proof of safety).
- M5 = "scheduling is parallel" (multi-rank через MPI, K>1 K-batching, anchor-test).

После успешного закрытия всех 11 задач и acceptance gate (§5) — milestone M4 завершён; execution pack для M5 (multi-rank + anchor-test) создаётся как новый аналогичный документ.

---

## 1. Decisions log (зафиксировано до старта T4.1)

| # | Решение | Значение | Rationale / источник |
|---|---|---|---|
| **D-M4-1** | Pipeline depth в M4 | Только `K=1`. `K ∈ {2, 4, 8}` разблокируется на M5 с K-batching. | Master spec §14 M4: "K=1 baseline"; K>1 требует multi-rank MPI (M5). |
| **D-M4-2** | Deployment pattern | Pattern 1 only. `attach_outer_coordinator(nullptr)` в M4; `SubdomainBoundaryDependency` — пустой тип. | scheduler/SPEC §1.3 + §14: Pattern 2 — M7+. |
| **D-M4-3** | ExecProfile в M4 | Только `Reference`. Production / FastExperimental разблокируются на M6+ (GPU), когда появляется measured perf для tuning. | Master spec §D: Reference — canonical oracle; optimization profiles нужны когда есть что оптимизировать на measured baseline. |
| **D-M4-4** | Canonical zone order source | `ZoningPlan.canonical_order` (собран на M3). Scheduler **не пересчитывает** Hilbert — читает готовый permutation vector. | Ownership boundary §8.2: scheduler consumes zoning, не mutates. |
| **D-M4-5** | CertificateStore backing | In-memory `std::unordered_map<(ZoneId, TimeLevel), SafetyCertificate>`. Archive-on-invalidate в telemetry — M5 scope. | scheduler/SPEC §4.4 + §13 — archive хук есть, но M4 emit'ит в `/dev/null` до M5 telemetry ring buffer'а. |
| **D-M4-6** | Commit protocol в single-rank | Phase A: `mark_completed` (force+integrate done, version++). Phase B (M4 single-rank): no peer ⇒ direct `Completed → Committed`. `PackedForSend/InFlight` transitions реализуются, но в K=1 single-rank пусты (mark_packed/inflight становятся no-op для internal zones). Multi-rank peer flow — M5. | scheduler/SPEC §6.2 bullet 2: "Если peer'а нет (internal zone): state machine → Committed напрямую". В Pattern 1 single-rank все зоны — internal. |
| **D-M4-7** | Deadlock watchdog threshold | `T_watchdog = 30s` (scheduler/SPEC §11.1 default + §8.1). Unit-test триггер сокращает до `100ms` через injection. | scheduler/SPEC §8.1 + §11.1: 30s — default; tests overrideят. |
| **D-M4-8** | Property fuzzer scale | **10⁶ sequences per PR в CI** (scheduler/SPEC §12.2 hard gate). Каждая sequence ~10⁴ random events (§13.4). Разбиение: I1–I5 transition-fuzzer (~500k seqs), I6 frontier-fuzzer (~250k seqs), I7 cert-monotonicity-fuzzer (~250k seqs). | Master spec §13.4 + scheduler/SPEC §12.2 — non-negotiable; playbook auto-reject pattern запрещает <10⁵ для новых invariants. |
| **D-M4-9** | TD K=1 ≡ legacy NVE (thermo-byte-exact) | T4.9 wiring TD-enabled path **обязан** давать байт-экзактно идентичный thermo log на M3 smoke config vs legacy path. Это acceptance gate для K=1: "TD mode не должен ломать физику". | Master spec §14 M4 artifact gate: "T1, T2 green в TD mode; bitwise determinism tests pass". K=1 математически тождественен последовательному Velocity-Verlet; любое отклонение — bug. |
| **D-M4-10** | Telemetry hooks в M4 | Emit только read-only counters + single-rank metrics: `zones_ready_count`, `zones_committed_total`, `certificate_failures_total`, `current_frontier_min/max`, `pipeline_depth`, `task_selection_time_ms`, `commit_latency_ms`. Load-balance metrics (p50/p95/imbalance_ratio) — M5 (§11a.2 Phase 1). | scheduler/SPEC §13 lists 11 metrics; multi-rank metrics (boundary_stalls, rank_utilization) — trivially constant/N/A в single-rank, emit в M5. |
| **D-M4-11** | SimulationEngine migration strategy | Add **opt-in** `td_mode: bool` (default `false` в M4) в YAML config; default path остаётся untouched legacy NVE. При `td_mode: true` — scheduler-driven path. После M5 default переключается на `true`; legacy path удаляется в M6. | Не ломаем M1–M3 regression gates во время M4 development; даём безопасный rollback если что-то просачивается. M3 smoke + M2 smoke продолжают работать без изменений. |
| **D-M4-12** | Scheduler module placement | `src/scheduler/{include/tdmd/scheduler/*.hpp, *.cpp}` + `tdmd_scheduler` CMake target. Headers split: `types.hpp`, `policy.hpp`, `safety_certificate.hpp`, `certificate_store.hpp`, `zone_state_machine.hpp`, `td_scheduler.hpp` (abstract), `causal_wavefront_scheduler.hpp` (concrete). | Единообразие с zoning/, runtime/, potentials/; header-split минимизирует rebuild fan-out. |
| **D-M4-13** | Retry policy в Reference | `max_retries_per_task = 3` (SPEC §7, §11.1). Deterministic — retry_count монотонно растёт, без rand-backoff. M4 single-rank: единственный triggered retry — cert invalidation mid-compute (migrations, neighbor rebuild race). | scheduler/SPEC §7.2 explicit: Reference profile — no rand backoffs, canonical retry ordering. |
| **D-M4-14** | SPEC deltas в M4 | Ни одна из 11 задач не должна требовать SPEC delta. scheduler/SPEC v1.0 уже contracts всё, что M4 строит. Если task upfront предполагает SPEC edit — flag в pre-impl report, выделяем в отдельный spec-delta PR (playbook §9.1). | Playbook §9.1: SPEC edits — doc-only PR, separate from code. |

---

## 2. Глобальные параметры окружения

| Параметр | Значение | Примечание |
|---|---|---|
| OS | Linux (Ubuntu 24.04 LTS) | Dev-машина пользователя; ubuntu-latest в CI |
| C++ compiler | GCC 13+ / Clang 17+ | C++20; CI уже проверяет оба |
| CMake | 3.25+ | Master spec §15.2 |
| CUDA | 13.1 installed, **не используется в M4** | GPU path — M6 |
| MPI | **не используется в M4** | Multi-rank — M5 |
| Python | 3.10+ | pre-commit helpers |
| Test framework | Catch2 v3 (FetchContent) | Унаследовано из M0 |
| LAMMPS oracle | SKIP on public CI (Option A) | M4 не вводит новых differentials; T1/T4 продолжают работать |
| Active BuildFlavor | `Fp64ReferenceBuild` | D-M4-3 |
| Active ExecProfile | `Reference` | D-M4-3 |
| Run mode | single-rank, single-thread, CPU only | Master spec §14 M4 |
| Pipeline depth | `K_max = 1` | D-M4-1 |
| Fuzz RNG | `std::mt19937_64` с fixed seed `0x4D345F53434845` ("M4_SCHE") + per-PR seed variation | D-M4-8: reproducible но не статический |
| Branch policy | `m4/T4.X-<topic>` per PR → `main` | CI required: lint + build-cpu + M1/M2/M3 smokes; M4 smoke добавляется в T4.11 |

---

## 3. Suggested PR order

Dependency graph:

```
                                 ┌─► T4.3 (cert math + I7) ──────┐
                                 │                                │
T4.1 ─► T4.2 ──────────────────► ┼─► T4.4 (state machine + I1-5)─┼─► T4.5 (scheduler core) ─► T4.6 (select_ready + I6) ─► T4.7 (commit protocol) ─► T4.8 (watchdog) ─► T4.9 (engine wiring) ─► T4.10 (determinism) ─► T4.11 (M4 smoke, GATE)
(pack)  (skeleton)               │                                │
                                 └───────── independent ──────────┘
```

**Линейная последовательность (single agent):**
T4.1 → T4.2 → T4.3 → T4.4 → T4.5 → T4.6 → T4.7 → T4.8 → T4.9 → T4.10 → T4.11.

**Параллельный режим (multi-agent):** после T4.2 — `{T4.3, T4.4}` независимы (cert math vs state machine). T4.5 depends on T4.3 + T4.4 (needs both types). T4.6 → T4.7 → T4.8 строго последовательные (каждый использует infrastructure предыдущего). T4.9 depends on T4.8. T4.10 depends on T4.9. T4.11 depends on T4.10.

**Estimated effort:** 8 недель (single agent). Самые длинные — T4.5 (scheduler core, ~1 неделя), T4.6 (select_ready_tasks + frontier math, ~1 неделя), T4.9 (SimulationEngine wiring + byte-exact acceptance, ~1 неделя), T4.11 (M4 smoke + CI wiring, ~0.5 недели).

---

## 4. Tasks

### T4.1 — Author M4 execution pack

```
# TDMD Task: Create M4 execution pack

## Context
- Master spec: §14 M4
- Role: Architect / Spec Steward
- Milestone: M4 (kickoff)

## Goal
Написать `docs/development/m4_execution_pack.md` декомпозирующий M4 на 11 PR-size задач, по шаблону m3_execution_pack.md. Document-only PR per playbook §9.1.

## Scope
- [included] docs/development/m4_execution_pack.md (single new file)
- [included] Decisions log D-M4-1..D-M4-14
- [included] Task templates T4.1..T4.11
- [included] M4 acceptance gate checklist
- [included] Risks R-M4-1..R-M4-N + open questions

## Out of scope
- [excluded] Any code changes (T4.2+ territory)
- [excluded] SPEC deltas (separate PR per playbook §9.1)

## Required files
- docs/development/m4_execution_pack.md

## Required tests
- pre-commit clean
- markdownlint clean

## Acceptance criteria
- [ ] Pack committed + pushed
- [ ] Task list matches §14 M4 deliverables
- [ ] Decisions anchored to scheduler/SPEC
```

---

### T4.2 — `scheduler/` module skeleton + types + abstract interface

```
# TDMD Task: scheduler/ module skeleton

## Context
- Master spec: §12.4
- Module SPEC: docs/specs/scheduler/SPEC.md §2, §11
- Role: Core Runtime Engineer
- Milestone: M4

## Goal
Создать `src/scheduler/` module: CMakeLists, namespaces, public headers с типами `ZoneState`, `ZoneTask`, `SafetyCertificate`, `TimeLevel`, `Version`, `ZoneId`, abstract `TdScheduler`, `SchedulerPolicy` + `PolicyFactory::for_reference()`. Никакой concrete реализации scheduler'а (T4.5).

## Scope
- [included] src/scheduler/CMakeLists.txt — new static lib `tdmd_scheduler`
- [included] src/scheduler/include/tdmd/scheduler/types.hpp — ZoneState, ZoneTask, SafetyCertificate, TimeLevel, Version
- [included] src/scheduler/include/tdmd/scheduler/policy.hpp — SchedulerPolicy + PolicyFactory
- [included] src/scheduler/include/tdmd/scheduler/td_scheduler.hpp — abstract base
- [included] src/scheduler/policy.cpp — PolicyFactory::for_reference() body
- [included] src/CMakeLists.txt — add_subdirectory(scheduler)
- [included] tests/scheduler/CMakeLists.txt + test_scheduler_types.cpp (type presence, layout, PolicyFactory)

## Out of scope
- [excluded] SafetyCertificate math (T4.3)
- [excluded] ZoneState machine enforcement (T4.4)
- [excluded] CausalWavefrontScheduler implementation (T4.5+)

## Mandatory invariants
- `TdScheduler` has virtual destructor + pure-virtual methods matching SPEC §2.2
- `SchedulerPolicy` default-constructible + copyable (value type)
- `PolicyFactory::for_reference()` returns policy with `use_canonical_tie_break=true`, `allow_task_stealing=false`, `deterministic_reduction_cert=true`, `exponential_backoff=false`, `two_phase_commit=true` (SPEC §11.1)
- No dependency on state/, neighbor/, comm/, or any runtime/ header (scheduler is policy/data)

## Required files
- src/scheduler/{CMakeLists.txt, include/tdmd/scheduler/*.hpp, policy.cpp}
- tests/scheduler/{CMakeLists.txt, test_scheduler_types.cpp}
- src/CMakeLists.txt (updated)
- tests/CMakeLists.txt (updated)

## Required tests
- test_scheduler_types: compile + instantiate empty SafetyCertificate, ZoneTask; PolicyFactory::for_reference() field assertions
- ctest clean

## Acceptance criteria
- [ ] All files committed
- [ ] ctest green
- [ ] pre-commit clean
- [ ] No link regressions in src/cli or src/runtime
```

---

### T4.3 — `SafetyCertificate` math + `CertificateStore` + I7 monotonicity fuzzer

```
# TDMD Task: SafetyCertificate — build/validate math + store + monotonicity fuzzer

## Context
- Master spec: §6.4
- Module SPEC: docs/specs/scheduler/SPEC.md §4
- Role: Scheduler / Determinism Engineer
- Depends on: T4.2
- Milestone: M4

## Goal
Реализовать `SafetyCertificate::build(zone, time_level, state_snapshot)` с displacement bound δ(dt) = v·dt + ½·a·dt², safe predicate δ < min(buffer, skin_remaining, frontier_margin), + `CertificateStore` для хранения (zone, time_level) → cert + invalidation API. Property-тест I7 monotonicity ≥10⁵ cases (доля от global 10⁶/PR per D-M4-8).

## Scope
- [included] src/scheduler/include/tdmd/scheduler/safety_certificate.hpp
- [included] src/scheduler/safety_certificate.cpp — δ(dt), safe(), build() factory
- [included] src/scheduler/include/tdmd/scheduler/certificate_store.hpp
- [included] src/scheduler/certificate_store.cpp — unordered_map<CertKey, SafetyCertificate> + invalidate_for(ZoneId) + invalidate_all(reason)
- [included] tests/scheduler/test_safety_certificate.cpp — unit: all boundary cases (v=0, a=0, dt=0, buffer=0, skin=0, frontier=0); I7 monotonicity 100 manual cases
- [included] tests/scheduler/fuzz_cert_monotonicity.cpp — ≥250 000 random (v, a, dt1, dt2, buffer, skin, frontier) sextuples verifying `safe(C[dt2]) ∧ dt1 < dt2 ⟹ safe(C[dt1])`

## Out of scope
- [excluded] ZoneMeta integration (T4.5)
- [excluded] Archive on invalidate (M5 telemetry scope)
- [excluded] Adaptive buffer policy (Production profile, M6+)

## Mandatory invariants
- I7 monotonicity holds symbolically AND over random fuzzer corpus
- `SafetyCertificate::safe` ⇔ `δ < min(buffer_width, skin_remaining, frontier_margin)` with buffer/skin/frontier each ≥ 0; negative inputs → `safe = false` (defensive)
- `CertificateStore::invalidate_for(zone_id)` removes cert for all time_levels matching zone_id
- Reproducible: same fuzzer seed → same corpus → same pass/fail verdict
- Thread-safe reads **not required** in M4 (single-threaded) — will be revisited M5

## Required files
- src/scheduler/include/tdmd/scheduler/{safety_certificate.hpp, certificate_store.hpp}
- src/scheduler/{safety_certificate.cpp, certificate_store.cpp}
- tests/scheduler/{test_safety_certificate.cpp, fuzz_cert_monotonicity.cpp}

## Required tests
- ≥ 20 manual unit cases (boundaries, symmetry)
- ≥ 250 000 fuzzer cases (I7) under CI budget (<30s)
- ctest clean; pre-commit clean

## Acceptance criteria
- [ ] All invariants test-covered
- [ ] I7 fuzzer green on ≥250k cases
- [ ] CertificateStore API surface complete (build/get/invalidate_for/invalidate_all)
- [ ] No UB on denormal/NaN inputs (documented behavior: safe=false)
```

---

### T4.4 — `ZoneState` machine + I1–I5 transition fuzzer

```
# TDMD Task: ZoneState machine — legal transitions + property fuzzer

## Context
- Master spec: §6.2, §13.4 I1-I5
- Module SPEC: docs/specs/scheduler/SPEC.md §3
- Role: Scheduler / Determinism Engineer
- Depends on: T4.2
- Milestone: M4

## Goal
Реализовать `ZoneStateMachine` — класс, обеспечивающий checked transitions между `ZoneState`'ами по диаграмме §3.1. Любая illegal transition → exception (в Reference) / assert (в Production). Property-фуззер ≥500k random event sequences проверяющий I1-I5.

## Scope
- [included] src/scheduler/include/tdmd/scheduler/zone_state_machine.hpp
- [included] src/scheduler/zone_state_machine.cpp — enforce table, illegal transitions throw
- [included] src/scheduler/include/tdmd/scheduler/zone_meta.hpp — ZoneMeta { state, time_level, version, cert_id, in_ready_queue, in_inflight_queue }
- [included] tests/scheduler/test_zone_state_machine.cpp — каждая из ~20 legal transitions; каждая illegal rejected (I1 subsumes)
- [included] tests/scheduler/fuzz_zone_state_invariants.cpp — ≥500k random event sequences, каждая N=10⁴ events, проверяет I1-I5 после каждого события

## Out of scope
- [excluded] Queue data structures (T4.5)
- [excluded] Scheduler-level event dispatch (T4.5)
- [excluded] I6 frontier invariant (T4.6)
- [excluded] I7 cert monotonicity (T4.3)

## Mandatory invariants
- I1: Committed → Ready without time_level++ → reject
- I2: Empty → Computing → reject (no data)
- I3: zone cannot be simultaneously in_ready_queue AND in_inflight_queue → enforced via ZoneMeta bitmask
- I4: two tasks with same (zone_id, time_level, version) → reject (duplicate detection in dispatch)
- I5: Completed → Committed requires explicit `commit_completed()` call (not automatic on mark_completed)
- Reproducible: same fuzzer seed → same corpus → same verdict

## Required files
- src/scheduler/include/tdmd/scheduler/{zone_state_machine.hpp, zone_meta.hpp}
- src/scheduler/zone_state_machine.cpp
- tests/scheduler/{test_zone_state_machine.cpp, fuzz_zone_state_invariants.cpp}

## Required tests
- Every legal transition: unit test
- Every illegal transition: unit test (throws / rejects)
- Fuzzer: ≥500k sequences × 10⁴ events; CI budget <60s
- ctest clean; pre-commit clean

## Acceptance criteria
- [ ] All 20+ legal transitions test-covered
- [ ] All illegal transitions rejected with diagnostic
- [ ] I1-I5 fuzzer green
- [ ] ZoneMeta API ready for consumption by T4.5
```

---

### T4.5 — `CausalWavefrontScheduler` core: DAG, refresh, lifecycle

```
# TDMD Task: CausalWavefrontScheduler — core structure, refresh, lifecycle

## Context
- Master spec: §6.3, §6.6, §12.4
- Module SPEC: docs/specs/scheduler/SPEC.md §2.3, §9, §10
- Role: Scheduler / Determinism Engineer
- Depends on: T4.3, T4.4
- Milestone: M4

## Goal
Создать `CausalWavefrontScheduler` (concrete implementation of `TdScheduler`): хранит per-zone `ZoneMeta`, читает `canonical_order` из `ZoningPlan`, строит spatial dependency DAG (соседи в пределах `r_c + r_skin`), lifecycle `initialize/attach_outer_coordinator/shutdown`, `refresh_certificates()` proactive построение cert'ов для `(z, current_frontier + 1)`. Но **без** `select_ready_tasks()` — это T4.6.

## Scope
- [included] src/scheduler/include/tdmd/scheduler/causal_wavefront_scheduler.hpp
- [included] src/scheduler/causal_wavefront_scheduler.cpp — constructor, initialize, attach_outer_coordinator, shutdown, refresh_certificates, invalidate_certificates_for, invalidate_all_certificates, introspection (finished, frontier_min/max, current_pipeline_depth, min_zones_per_rank, optimal_rank_count)
- [included] src/scheduler/include/tdmd/scheduler/zone_dag.hpp — compute_spatial_dependencies(ZoningPlan, r_c+r_skin) → per-zone neighbor mask
- [included] src/scheduler/zone_dag.cpp — уход в T4.5 per-cell-grid spatial dep calc
- [included] tests/scheduler/test_scheduler_core.cpp — lifecycle, canonical_order echo, spatial DAG sanity (2×2×2 box → expected neighbor masks), refresh_certificates reads zones in canonical_order

## Out of scope
- [excluded] select_ready_tasks (T4.6)
- [excluded] Commit protocol (T4.7)
- [excluded] Deadlock watchdog (T4.8)
- [excluded] SimulationEngine wiring (T4.9)
- [excluded] OuterSdCoordinator (Pattern 2, M7+)

## Mandatory invariants
- `initialize(ZoningPlan)` stores canonical_order by value (plan doesn't need to outlive scheduler)
- `refresh_certificates()` iterates zones in canonical_order; same plan → same iteration order — byte-stable
- Spatial DAG symmetric: `dep_mask[z1] contains z2 ⟺ dep_mask[z2] contains z1` (radius is metric)
- `attach_outer_coordinator(nullptr)` — explicit Pattern 1 contract
- `finished()` returns true ⟺ all zones reached target time_level AND no pending work (to be refined in T4.9)

## Required files
- src/scheduler/include/tdmd/scheduler/{causal_wavefront_scheduler.hpp, zone_dag.hpp}
- src/scheduler/{causal_wavefront_scheduler.cpp, zone_dag.cpp}
- tests/scheduler/test_scheduler_core.cpp

## Required tests
- Lifecycle smoke (initialize → refresh → shutdown, no leaks)
- Spatial DAG: 2×2×1 thin plate → each zone has 1 neighbor (x) or 1 neighbor (y); 2×2×2 cube → each has 3 face-adjacent
- Canonical order echo: scheduler.canonical_order() == plan.canonical_order
- refresh_certificates creates one cert per (zone, frontier+1) — count check

## Acceptance criteria
- [ ] Scheduler lifecycle green
- [ ] Spatial DAG symmetric + count-correct on 3 test geometries
- [ ] No hidden coupling to state/neighbor at interface boundary (all access via injected snapshots)
```

---

### T4.6 — `select_ready_tasks()` + I6 frontier invariant fuzzer

```
# TDMD Task: CausalWavefrontScheduler — task selection + I6 frontier guard

## Context
- Master spec: §6.7 (pseudocode), §13.4 I6
- Module SPEC: docs/specs/scheduler/SPEC.md §5, §9
- Role: Scheduler / Determinism Engineer
- Depends on: T4.5
- Milestone: M4

## Goal
Реализовать `select_ready_tasks()` per pseudocode §6.7. Tie-break в Reference: `(time_level_asc, canonical_zone_order_asc, version_asc)`. Frontier guard `t ≤ global_frontier_min + K_max`. Pattern 2 boundary branch — no-op при nullptr coordinator. Property-fuzzer ≥250k sequences проверяет I6.

## Scope
- [included] src/scheduler/causal_wavefront_scheduler.cpp — select_ready_tasks() body
- [included] src/scheduler/include/tdmd/scheduler/queues.hpp — ready_queue / blocked_queue / inflight_queue typed wrappers; deterministic priority queue compare functor
- [included] tests/scheduler/test_select_ready_tasks.cpp — manual unit cases: 2-zone chain, 4-zone 2×2, all-ready, all-blocked, frontier saturation
- [included] tests/scheduler/fuzz_frontier_invariant.cpp — ≥250k random state snapshots, каждая прогон `select_ready_tasks()`, check I6 post-condition

## Out of scope
- [excluded] mark_computing / mark_completed etc (T4.7)
- [excluded] Task stealing (Fast profile, M8+)
- [excluded] Cost-aware priority (Production profile, M6+)

## Mandatory invariants
- I6: after `select_ready_tasks()` каждая returned task satisfies `task.time_level ≤ global_frontier_min + K_max`
- Deterministic tie-break: два identical state snapshots → identical candidate ordering
- At most one task per zone per iteration (§5.1 pseudocode `break` after first match)
- `max_tasks_per_iteration` respected (Reference: min(streams, available))

## Required files
- src/scheduler/include/tdmd/scheduler/queues.hpp
- src/scheduler/causal_wavefront_scheduler.cpp (updated)
- tests/scheduler/{test_select_ready_tasks.cpp, fuzz_frontier_invariant.cpp}

## Required tests
- 10+ manual select scenarios
- 250k fuzzer sequences I6 post-condition check
- Tie-break determinism: 100 random state orderings → identical output

## Acceptance criteria
- [ ] I6 fuzzer green
- [ ] Tie-break byte-stable
- [ ] Frontier saturation correctly blocks further tasks
```

---

### T4.7 — Two-phase commit protocol + retry semantics

```
# TDMD Task: Commit protocol — mark_* lifecycle, two-phase, retry

## Context
- Master spec: §6.6 (pseudocode), §13.4 I5
- Module SPEC: docs/specs/scheduler/SPEC.md §6, §7
- Role: Scheduler / Determinism Engineer
- Depends on: T4.6
- Milestone: M4

## Goal
Имплементировать: `mark_computing`, `mark_completed` (Phase A: force+integrate done, version++), `mark_packed`, `mark_inflight`, `mark_committed`, `commit_completed()` (Phase B: в M4 single-rank direct Completed → Committed per D-M4-6). Retry: canonical `max_retries_per_task = 3`, deterministic counter.

## Scope
- [included] src/scheduler/causal_wavefront_scheduler.cpp — all mark_* + commit_completed
- [included] src/scheduler/include/tdmd/scheduler/retry_state.hpp — RetryTracker { per_task retry_count, max_retries }
- [included] tests/scheduler/test_commit_protocol.cpp — Phase A only (mark_completed), Phase A+B (commit_completed), I5 test (Completed != Committed unless commit_completed called)
- [included] tests/scheduler/test_retry_canonical.cpp — deterministic retry count monotonicity

## Out of scope
- [excluded] Multi-rank peer pack/send (M5)
- [excluded] HaloPacket in Pattern 2 (M7)
- [excluded] Halo timeout retry (M5)

## Mandatory invariants
- I5: `Completed → Committed` happens ONLY via `commit_completed()` (separate phase, not a side-effect of mark_completed)
- In single-rank M4: `commit_completed()` drains all Completed zones → Committed in one pass
- `version` monotonically increments per mark_completed
- retry_count ≤ max_retries_per_task; exceeding → hard failure (throw)

## Required files
- src/scheduler/causal_wavefront_scheduler.cpp (updated)
- src/scheduler/include/tdmd/scheduler/retry_state.hpp
- tests/scheduler/{test_commit_protocol.cpp, test_retry_canonical.cpp}

## Required tests
- 10-zone run synthetic: all pass through Empty → ResidentPrev → Ready → Computing → Completed → Committed in correct order
- I5 regression: directly check mark_completed doesn't set Committed
- Retry canonical: ≥100 random failure injection sequences, same seed → same retry_count trajectory

## Acceptance criteria
- [ ] Two-phase commit enforced
- [ ] I5 test green
- [ ] Retry deterministic
```

---

### T4.8 — Deadlock watchdog + diagnostic dump

```
# TDMD Task: Watchdog — progress tracking + diagnostic on stall

## Context
- Master spec: §6.6
- Module SPEC: docs/specs/scheduler/SPEC.md §8
- Role: Scheduler / Determinism Engineer
- Depends on: T4.7
- Milestone: M4

## Goal
`check_deadlock(T_watchdog)` — если за `T_watchdog` не было ни одного progress event (ready dispatch, inflight→committed, event processed, frontier++), dump diagnostic (состояния зон по count, queue sizes, events last 100) и throw hard diagnostic failure. Unit test с injection T_watchdog=100ms + intentional cycle → throws within tolerance.

## Scope
- [included] src/scheduler/causal_wavefront_scheduler.cpp — check_deadlock body + progress_timestamps tracking
- [included] src/scheduler/include/tdmd/scheduler/diagnostic_dump.hpp — DiagnosticReport struct + formatter
- [included] src/scheduler/diagnostic_dump.cpp — format zone-state histogram, queue lengths, last-100 events (ring buffer in scheduler)
- [included] tests/scheduler/test_deadlock_watchdog.cpp — intentional cycle (2 zones each waiting on the other's cert never arriving), verify throw within T_watchdog ± 10%; long-slow-no-deadlock scenario (progress every 50ms, T_watchdog=100ms) → no throw

## Out of scope
- [excluded] Pattern 2 boundary stall diagnostics (M7)
- [excluded] HaloTimeout retry escalation (M5)
- [excluded] Nsight NVTX ranges (M6)

## Mandatory invariants
- `check_deadlock` is idempotent (multiple calls without progress don't double-throw)
- Progress definition per SPEC §8.2: ready-dispatch OR inflight-committed OR event processed OR frontier increase
- Diagnostic dump includes: `frontier_min/max`, per-state zone count, queue.size() for each of 5 queues, last 100 events
- Test-injectable `T_watchdog` (default 30s; tests override to 100ms)

## Required files
- src/scheduler/causal_wavefront_scheduler.cpp (updated)
- src/scheduler/include/tdmd/scheduler/diagnostic_dump.hpp
- src/scheduler/diagnostic_dump.cpp
- tests/scheduler/test_deadlock_watchdog.cpp

## Required tests
- Intentional deadlock: triggers within T_watchdog ± 10%
- Slow-but-progressing: doesn't trigger
- Diagnostic dump string contains expected keywords (frontier, zones, queues)

## Acceptance criteria
- [ ] Watchdog fires on deadlock, doesn't fire on slow progress
- [ ] Dump is human-readable + grep-able
```

---

### T4.9 — `SimulationEngine` TD-mode run loop (K=1 single-rank)

```
# TDMD Task: SimulationEngine — TD-enabled run loop, K=1 byte-exact to legacy

## Context
- Master spec: §6.6 (pseudocode), §14 M4 "SimulationEngine orchestrates TD-enabled run"
- Module SPEC: docs/specs/runtime/SPEC.md + scheduler/SPEC.md §10
- Role: Core Runtime Engineer
- Depends on: T4.8
- Milestone: M4

## Goal
Добавить opt-in `td_mode: true` в YAML config. При включении — `SimulationEngine::run()` использует `CausalWavefrontScheduler` через `iteration(step h)` pseudocode §6.6. K=1: последовательный обход зон в canonical_order, per-zone force+integrate, commit. Acceptance: thermo log `td_mode: true` vs `td_mode: false` — **byte-exact** на M3 smoke config (Ni-Al EAM 864 atoms 10 steps) + M1 smoke (Al Morse).

## Scope
- [included] src/runtime/include/tdmd/runtime/simulation_engine.hpp — add `td_mode_` flag + TdScheduler ptr
- [included] src/runtime/simulation_engine.cpp — new `run_td_mode(n_steps, thermo_out)` path вызывающий iteration(h) per-step
- [included] src/io/yaml_config — parse `scheduler.td_mode: bool` (default false)
- [included] tests/integration/m4_smoke_td_nve_byteexact/ — test: same Ni-Al config both modes, diff thermo bit-for-bit
- [included] tests/runtime/test_td_mode_smoke.cpp — 5-step NVE td_mode, byte-exact to legacy

## Out of scope
- [excluded] K>1 pipeline (M5)
- [excluded] Multi-rank dispatch (M5)
- [excluded] Legacy path removal (M6)
- [excluded] Dump frame changes (already exists, no edit)

## Mandatory invariants
- Default `td_mode: false` — all existing smokes (M1/M2/M3) unchanged byte-for-byte
- `td_mode: true` on K=1 path: thermo byte-exact to legacy on fixed config (absent floating-point reduction order changes, which must be verified)
- scheduler's canonical_zone_order = ZoningPlan.canonical_order (echoed, not recomputed)
- SimulationEngine ownership: still owns atoms/box/species (§8.2); scheduler is *policy* injected, not *state*

## Required files
- src/runtime/simulation_engine.{hpp,cpp}
- src/io/yaml_config.{hpp,cpp} (updated)
- tests/integration/m4_smoke_td_nve_byteexact/* (scripted byte-diff)
- tests/runtime/test_td_mode_smoke.cpp

## Required tests
- td_mode=true byte-exact to td_mode=false on M1 smoke (Al Morse 10 steps)
- td_mode=true byte-exact to td_mode=false on M3 smoke (Ni-Al EAM 10 steps)
- td_mode=false regression: M1, M2, M3 smokes unchanged

## Acceptance criteria
- [ ] Both smokes byte-exact across modes
- [ ] No regression in existing smokes
- [ ] TD-mode path passes ctest in-process
```

---

### T4.10 — Bitwise determinism tests (same-seed + queue ordering)

```
# TDMD Task: Determinism — same-seed byte-exact + queue ordering 10⁴ inserts

## Context
- Master spec: §7.3, §13.4
- Module SPEC: docs/specs/scheduler/SPEC.md §12.3
- Role: Validation / Reference Engineer
- Depends on: T4.9
- Milestone: M4

## Goal
Формальные determinism tests: (a) два TD-run с identical seed → byte-exact event log + byte-exact final state; (b) 10⁴ случайных вставок в ready_queue в различном temporal order → identical извлечение sequence.

## Scope
- [included] tests/scheduler/test_determinism_same_seed.cpp — spawn two engines same yaml, run 10 steps each, compare full atom SoA + final scheduler state
- [included] tests/scheduler/test_queue_ordering_determinism.cpp — 10⁴ random ZoneTask inserts at various order → same pop order
- [included] src/scheduler/include/tdmd/scheduler/event_log.hpp — optional compile-time event log ring buffer (for test introspection)

## Out of scope
- [excluded] Cross-compiler determinism (Level 2 — M5 anchor-test)
- [excluded] Cross-arch determinism (Level 3 — post-v1)
- [excluded] Multi-rank determinism (M5)

## Mandatory invariants
- Two runs, same seed, same binary → identical thermo (already covered by T4.9 byte-exact test) + identical event log
- 10⁴ random ready_queue inserts with different arrival orders → identical extraction order (verified via event log)
- Event log buffer doesn't allocate in hot path (pre-sized)

## Required files
- tests/scheduler/{test_determinism_same_seed.cpp, test_queue_ordering_determinism.cpp}
- src/scheduler/include/tdmd/scheduler/event_log.hpp

## Required tests
- Same-seed same-binary: byte-exact
- Queue ordering: 10⁴ inserts × 100 shuffles → identical extraction
- CI budget <30s

## Acceptance criteria
- [ ] Determinism tests green
- [ ] Event log infrastructure available for T4.11 smoke introspection
```

---

### T4.11 — M4 integration smoke + acceptance gate

```
# TDMD Task: M4 integration smoke — closes the milestone

## Context
- Master spec: §14 M4 final artifact gate
- Role: Validation / Reference Engineer
- Depends on: T4.10
- Milestone: M4 (final)

## Goal
End-to-end (< 10s, CI-integrated) exercising the M4 user surface: Ni-Al EAM NVE run с `td_mode: true`, K=1, scheduler.watchdog injected short, verifies: (a) thermo byte-exact to M3 smoke golden; (b) scheduler emits expected telemetry counters (zones_committed, pipeline_depth=1); (c) fuzzer tests pass in 10⁶ total seqs (I1-I7). Mirrors M3 smoke in structure.

## Scope
- [included] tests/integration/m4_smoke/:
  - README.md (smoke philosophy, what "TD mode byte-exact" means, acceptance gate)
  - smoke_config.yaml.template (reuses T4 assets, adds `scheduler.td_mode: true`)
  - run_m4_td_smoke.sh (driver)
  - thermo_golden.txt (reuses M3 golden byte-for-byte — D-M4-9 gate)
  - telemetry_expected.txt (counters expected values)
- [included] .github/workflows/ci.yml — M4 smoke step after M3 smoke
- [included] docs/specs/scheduler/SPEC.md — append to change log: "M4 landed (2026-MM-DD), Reference K=1 single-rank"

## Out of scope
- [excluded] Multi-rank smoke (M5)
- [excluded] GPU smoke (M6)
- [excluded] Performance measurement (no perf gate in M4)

## Mandatory invariants
- Wall-time < 10s CI budget
- Smoke uses existing T4 assets (same as M3)
- Thermo byte-match M3 golden (K=1 ≡ sequential = legacy = M3)
- Telemetry sanity: zones_committed_total == 9·10 (9 zones × 10 steps); pipeline_depth == 1; certificate_failures_total == 0; deadlock_warnings_total == 0
- No regressions in M1 / M2 / M3 smokes + T1 / T4 differentials

## Required files
- tests/integration/m4_smoke/*
- .github/workflows/ci.yml (updated)
- docs/specs/scheduler/SPEC.md (change log only)

## Required tests
- [smoke local] < 10s, exit 0
- [CI] passes on gcc-13 + clang-17
- [regression] M1, M2, M3 smokes + T1, T4 differentials unchanged
- [property] fuzzer (T4.3/T4.4/T4.6) total ≥10⁶ cases green

## Acceptance criteria
- [ ] All artifacts committed
- [ ] CI green (all 4 smokes: M1, M2, M3, M4)
- [ ] M4 acceptance gate (§5) fully closed
- [ ] Milestone M4 ready to declare done
```

---

## 5. M4 Acceptance Gate

После закрытия всех 11 задач — проверить полный M4 artifact gate (master spec §14 M4):

- [ ] **`ZoneState` state machine** (T4.4), все ~20 legal transitions enforced, все illegal rejected
- [ ] **Invariants I1-I5 fuzzer** (T4.4), ≥500k sequences green в CI
- [ ] **SafetyCertificate math** (T4.3), δ(dt) + safe predicate + all boundary cases covered
- [ ] **Invariant I7 (cert monotonicity) fuzzer** (T4.3), ≥250k sequences green
- [ ] **CertificateStore** (T4.3), build/get/invalidate_for/invalidate_all API complete
- [ ] **CausalWavefrontScheduler (Reference mode)** (T4.5-T4.8), full lifecycle: initialize → refresh → select → commit → shutdown
- [ ] **Invariant I6 (frontier guard) fuzzer** (T4.6), ≥250k sequences green
- [ ] **Two-phase commit** (T4.7), I5 test green, `commit_completed()` explicit phase
- [ ] **Deadlock watchdog** (T4.8), intentional cycle triggers within T_watchdog ± 10%; slow-but-progressing doesn't trigger
- [ ] **SimulationEngine TD-mode** (T4.9), K=1 byte-exact to legacy NVE on M1 + M3 smokes — **mandatory gate** (D-M4-9)
- [ ] **Bitwise determinism tests** (T4.10), same-seed byte-exact runs; 10⁴ queue-ordering determinism
- [ ] **Property fuzzer cumulative ≥10⁶ sequences/PR in CI** (scheduler/SPEC §12.2 hard gate)
- [ ] **M4 integration smoke** (T4.11) < 10s, CI-integrated; telemetry counters match expected
- [ ] No regressions: M1 smoke + M2 smoke + M3 smoke + T1 diff + T4 diff all green
- [ ] CI Pipelines A (lint+build+smokes) + B (unit/property) + C (differentials) all green
- [ ] Pre-implementation + session reports attached в каждом PR
- [ ] Human review approval для каждого PR

---

## 6. Risks & Open Questions

**Risks:**

- **R-M4-1 — Scheduler module size / timeline overrun.** scheduler/SPEC.md 789 строк (1.5× других модулей); 8-недельный M4 может реально занять 10+ недель в single-agent режиме. Mitigation: строгая декомпозиция на 11 PR-size задач с раннего T4.2 skeleton'а (чтобы компилироваться + тестироваться умели с первого commit'а); коммит-per-task; rolling post-impl reports после каждой крупной задачи (не только в конце).
- **R-M4-2 — I7 monotonicity corner cases на denormals/NaN.** Cert math `δ(dt) = v·dt + 0.5·a·dt²` с `v=0, a=0, dt=0` или `buffer=NaN` может нарушить monotonicity в arithmetic боковых режимах. Mitigation: T4.3 explicit defensive policy — negative/NaN input → safe=false; fuzzer includes denormal corpus.
- **R-M4-3 — K=1 byte-exact acceptance слишком строгий.** D-M4-9 требует байт-экзактно идентичный thermo между legacy и TD-mode. Если force reduction order меняется через scheduler (e.g., канонический порядок зон vs atom-index order) — FP summation non-associative → мелкий диффер. Mitigation: T4.9 должен сохранить идентичный reduction order — для K=1 это значит iterate zones в canonical_order and within-zone atoms sequentially (как делают M1-M3). Если всё-таки diff — fallback на tight threshold (e.g., PE delta < 1e-14 eV), задокументированный как RESOLVED в OQ.
- **R-M4-4 — Property fuzzer CI budget overrun.** 10⁶ sequences × 10⁴ events = 10¹⁰ events total. Naive generation не уложится в 30s CI budget per PR. Mitigation: fuzzer разбит на отдельные CI jobs (I1-5, I6, I7); per-seq events capped at 10³ в CI, 10⁴ в local pre-push; parallelize across Catch2 tags.
- **R-M4-5 — Watchdog false-positive в CI under load.** `T_watchdog = 30s` default с shared GitHub Actions runner может сработать spuriously при CI load spikes. Mitigation: M4 runs watchdog test с injection T_watchdog=100ms (M4 unit test); production default 30s не triggerится на valid run. Long-progress test использует injection 5s.
- **R-M4-6 — Dep-graph spatial radius edge cases.** Zones на границе box'а в неортогональных ячейках могут иметь asymmetric dep_mask если radius `r_c + r_skin` ≥ zone size. Mitigation: T4.5 restrictя M4 к ортогональным box'ам (где zoning планнер и есть); non-ortho — M9 NPT.
- **R-M4-7 — Session-scope overrun.** M4 — 8 недель; single conversation на auto-pilot не уложится в 11 tasks. Mitigation: execute in documented order; commit-per-task; final post-impl covers landed subset; remaining tasks roll forward с кратким status-sync в начале следующей сессии.

**Open questions (deferred to task-time decisions):**

- **OQ-M4-1 — `ZoneTask::dep_mask` encoding.** SPEC §2.1 указывает `uint64_t dep_mask` — ограничение 64 зон subdomain. Что делать при >64? **Answer at T4.2:** M4 Pattern 1 single-rank typical 8-64 zones; если >64 — escalate в `std::bitset<N>` alias `ZoneDepMask`. Type defined at module scope; fits 64 zones in fast path.
- **OQ-M4-2 — `mode_policy_tag` derivation.** SPEC §2.1 lists `uint64_t mode_policy_tag` — откуда берётся? **Answer at T4.2:** compile-time BuildFlavor + ExecProfile hash, baked into `PolicyFactory::for_reference()` output. M4 single value.
- **OQ-M4-3 — Does `PolicyFactory::for_production()` exist in M4?** SPEC §11.1 declares it; M4 не использует Production. **Answer at T4.2:** declare stubs that throw "not implemented in M4"; full bodies — M6+.
- **OQ-M4-4 — Event log ring buffer size.** T4.10 hints "pre-sized"; concrete N? **Answer at T4.10:** N=1024 events, circular. Diagnostic dump (T4.8) reads last 100.
- **OQ-M4-5 — Byte-exact acceptance threshold fallback.** If R-M4-3 бьёт по D-M4-9 (true byte-exact невозможен), какой численный порог acceptance'а? **Answer at T4.9:** max |Δthermo_field| < 1e-14 eV (field-wise), otherwise SPEC delta + human review. Start от byte-exact и relax только при доказанной необходимости.
- **OQ-M4-6 — Scheduler-level telemetry begin_run()/end_run() hooks.** Нужно ли scheduler'у иметь свои begin_run/end_run или всё идёт через `telemetry::Telemetry*`? **Answer at T4.5:** reuse existing `telemetry::Telemetry*` sink (injected в SimulationEngine); scheduler emit'ит свои метрики через тот же sink. No new entry points.
- **OQ-M4-7 — Pattern 2 boundary dependency storage в M4.** scheduler/SPEC §3.1 упоминает `SubdomainBoundaryDependency`. В M4 этот тип пустой? **Answer at T4.2:** declared as empty struct с `attach_outer_coordinator(OuterSdCoordinator*)` signature nullptr-acceptable. Full Pattern 2 body — M7.
- **OQ-M4-8 — Reference profile allow `PolicyFactory::for_fast_experimental()` return value?** M4 не потребляет Fast. **Answer at T4.2:** same as OQ-M4-3 — stub that throws.

---

## 7. Roadmap Alignment

| Deliverable | Consumer milestone | Why it matters |
|---|---|---|
| `SafetyCertificate` + `CertificateStore` (T4.3) | M5 multi-rank cert refresh across ranks; M7 Pattern 2 halo cert invalidation | Без formal cert math TD-legality недоказуема; M5 anchor-test без cert невозможен |
| `ZoneState` machine + I1-I5 enforcement (T4.4) | M5 Ring backend depends on PackedForSend/InFlight/Committed transitions | I1-I5 — fundamental TD correctness; M5-M7 layer peer dispatch поверх этих transitions |
| `CausalWavefrontScheduler` core (T4.5-T4.8) | M5 K-batching (K∈{2,4,8}); M7 `InnerTdScheduler` rename + `attach_outer_coordinator` | Scheduler — **единственный** module managing time progression; M5-M7 — it layering, не замена |
| TD-mode `SimulationEngine` (T4.9) | M5 multi-rank run loop; M6 GPU dispatch кранами через scheduler tasks | SimulationEngine orchestrates но не owns time; M4 делает this explicit |
| Determinism tests (T4.10) | M5 anchor-test (layout-invariant determinism prerequisite); M8 SNAP proof-of-value (performance claims требуют byte-reproducible baseline) | Level 1 determinism (same-seed-same-binary) foundational для всех scientific reproducibility claims |
| Property fuzzer 10⁶ (T4.3-T4.6) | M5-M13 regression gate for any scheduler edit | Любое изменение scheduler invariants должно быть proven не breaking I1-I7 — fuzzer это canonical gate |
| M4 smoke (T4.11) | Continuous regression guard M5-M13 | Любой PR, трогающий scheduler, breaks this smoke до того как reaches multi-rank |

---

*End of M4 execution pack, дата: 2026-04-18.*
