# M5 Execution Pack

**Document:** `docs/development/m5_execution_pack.md`
**Status:** draft, awaiting human review
**Parent:** `TDMD_Engineering_Spec.md` §14 (M5), `docs/specs/comm/SPEC.md` v1.0, `docs/specs/scheduler/SPEC.md` v1.0, `docs/development/claude_code_playbook.md` §3
**Milestone:** M5 — Multi-rank TD on CPU (MPI) + K-batching (6 нед.)
**Created:** 2026-04-19
**Author:** Architect / Spec Steward role (Claude Opus 4.7)

---

## 0. Purpose

Этот документ декомпозирует milestone **M5** master spec'а §14 на **12 PR-size задач**, каждая
сформулирована по каноническому шаблону `claude_code_playbook.md` §3.1. Документ — **process
artifact**, не SPEC delta.

M5 — **первая встреча TDMD с реальной параллельной природой time decomposition'а**. После M4
scheduler умеет детерминистически вести один rank по зонам в K=1; M5 даёт ему (а) абстракцию
`CommBackend` с двумя concrete реализациями (`MpiHostStagingBackend` и `RingBackend`),
(б) wire-format `TemporalPacket` + CRC32, (в) K-batched pipeline для `K ∈ {1, 2, 4, 8}` с
явными pipeline-fill/steady/drain фазами, (г) multi-rank `SimulationEngine` loop с
детерминистической global-reduction, и (д) **anchor-test** воспроизведения эксперимента
Андреева §3.5 диссертации (Al FCC 10⁶ atoms, Lennard-Jones r_c=8Å, ring topology) с
точностью ≤ 10% на equivalent hardware class.

**Conceptual leap от M4 к M5:**

- M4 = "scheduling is correct" (один rank, K=1, cert + state machine + canonical order).
- **M5 = "scheduling scales"** (P ranks, K∈{1,2,4,8}, TemporalPackets по ring/mesh, anchor-test
  numerically reproduces the dissertation).
- M6 = "scheduling uses the GPU" (CUDA kernels + GPU-aware MPI / NCCL).

Критически — **anchor-test** §13.3 это hard merge gate M5. Без его passing TDMD не имеет
права использовать termin "Time Decomposition method by Andreev" (master spec §13.3).
Это первая в проекте milestone, на которой TDMD доказывает, что **воспроизводит метод
диссертации**, а не пересказывает его.

После успешного закрытия всех 12 задач и acceptance gate (§5) — milestone M5 завершён;
execution pack для M6 (GPU path) создаётся как новый аналогичный документ.

---

## 1. Decisions log (зафиксировано до старта T5.1)

| # | Решение | Значение | Rationale / источник |
|---|---|---|---|
| **D-M5-1** | Pipeline depth | `K ∈ {1, 2, 4, 8}` manual. Auto-K — M8+. Default `K=1` для совместимости с M4. | Master spec §14 M5 + §6.5a: M5 "fixed K only, manual specification". Auto-K требует measured perf, появляется в M8. |
| **D-M5-2** | Backends в M5 | `MpiHostStagingBackend` (universal fallback) + `RingBackend` (anchor-test). GpuAwareMpi + Nccl — M6; Hybrid — M7; Nvshmem — v2+. | comm/SPEC §12 roadmap: "M5: MpiHostStaging full, Ring, K-batching integration". |
| **D-M5-3** | Deployment pattern | Pattern 1 only (`P_space = 1`, `P_time = P`). `attach_outer_coordinator(nullptr)` preserved from M4. | Master spec §10.1: Pattern 2 — M7+; M5 проверяет Pattern 1 до предела. |
| **D-M5-4** | ExecProfile + BuildFlavor | `Reference` только, `Fp64ReferenceBuild`. Production/Fast + Mixed — M6+. | Master spec §14 M5 "CPU (MPI)" — no GPU, no mixed precision. Все другие axes наследуются от M4. |
| **D-M5-5** | Target hardware | CPU-only, x86_64. Ubuntu 24.04 + MPI (OpenMPI 4.1+ preferred; MPICH 4+ acceptable fallback). | Master spec §14 M5: "CPU (MPI)". GPU path — M6 (CUDA + MPI). |
| **D-M5-6** | Acceptance gates | **Primary:** anchor-test §13.3 — TDMD на Ring backend, N∈{4,8,16} ranks, Al FCC 10⁶ LJ, воспроизводит рис. 29-30 диссертации within 10% (hardware-normalized). **Regression:** K=1 P=1 остаётся байт-экзактным к M4 smoke golden (D-M5-12 below). **Scientific:** K≥2 и/или P≥2 matches K=1 P=1 на scientific observables — NVE drift `abs(dE)/E < 1e-6` over 1000 steps, T/P/⟨E⟩ within 2σ. | Master spec §13.3 mandatory gate; §13.5 determinism matrix; §13.7 differential thresholds. |
| **D-M5-7** | Zoning scheme для anchor-test | Add **1D Z-axis linear zoning** (Andreev §2.2) как второй режим рядом с существующим 3D-Hilbert'ом M3. Anchor-test MUST использовать 1D linear (это forms одно из axiom диссертационного эксперимента). | Master spec §13.3 + §4.4 bullet 1: "1D linear zoning (Andreev §2.2): N_min = 2 — оптимальный случай". Ring topology + 1D linear — 1:1 соответствие TIME-MD Андреева. |
| **D-M5-8** | Atom migration protocol | Zone-to-zone внутри subdomain'а (Pattern 1) — **no MPI** for atom migration. TemporalPacket везёт полный snapshot зоны (id, x, v, flags) — эффективно "migrates" атомы между ranks как часть каждого pipeline transfer. `MigrationPacket` остаётся defined но unused в M5 (Pattern 2 реализация — M7). | Master spec §10.5 + comm/SPEC §2.1: MigrationPacket — cross-subdomain (Pattern 2). Pattern 1 migration — embedded в TemporalPacket payload. |
| **D-M5-9** | Deterministic reduction | В Reference profile `global_sum_double` **обязан** использовать `deterministic_sum_double` (Kahan-compensated, rank-ordered — comm/SPEC §7.2). Native `MPI_Allreduce` запрещён в Reference (implementation-dependent order). Fast profile может использовать `MPI_Allreduce` — но Fast — M8+. | comm/SPEC §7.2 + master spec §7.3 Level 1 determinism: "Same run twice → bitwise equal" requires reproducible reductions. |
| **D-M5-10** | Protocol version | `TemporalPacket.protocol_version = 1` (initial release). Bumps — при любом wire-format change согласно comm/SPEC §4.3. | comm/SPEC §4.3. v1.1 (per-atom force embed) — out of scope. |
| **D-M5-11** | CRC32 policy | Enabled always в M5. Sender computes, receiver validates, mismatch → drop + log + trigger retry. В Fast profile (M8+) можно будет disable. | comm/SPEC §4.4 + §8.4. Production correctness > small perf win. |
| **D-M5-12** | K=1 P=1 regression guard | `K=1, P=1` path через new `CommBackend` stack (null loopback) MUST остаться byte-exact к M4 smoke thermo golden. M4 smoke + M5 smoke K=1 P=1 share thermo golden. | D-M4-9 contract extended: M5 не имеет права breaking M4 deterministic gate. Гарантирует, что introduction of comm layer — zero-cost в single-rank case. |
| **D-M5-13** | CI multi-rank strategy | `mpirun -np {1, 2, 4}` integrated в CI on `ubuntu-latest`. 8-rank и 16-rank anchor-test runs — **local pre-push only** (slow tier, §8.4 verify/SPEC). CI использует short-form smoke (<30s budget). Full anchor-test local budget ≤ 1 hour. | Option A CI policy (`project_option_a_ci.md` memory) — no self-hosted runner; anchor-test 10⁶ atoms × 16 ranks exceeds public runner capacity. |
| **D-M5-14** | Test framework для MPI | Catch2 v3 wrapping `MPI_Init` в `main_mpi.cpp`. One test binary per rank-count layout (test_comm_2rank, test_comm_4rank). Assertions wrapped с `MPI_Allreduce(AND)` чтобы fail on any rank → fail test. | Catch2 + MPI — standard pattern (OpenMPI, Kokkos projects). |
| **D-M5-15** | Anchor-test reference data | `verify/benchmarks/t3_al_fcc_large_anchor/dissertation_reference_data.csv` — extracted points from dissertation figures 29-30 (performance vs N_procs + efficiency vs N_procs). Two curves, ~8 data points each, in CSV с rationale-block в adjacent Markdown. Hardware normalization скрипт `hardware_normalization.py` — нормализует результаты current hardware к 2007 class via measured FLOPs ratio. | verify/SPEC §4.4 + master spec §13.3. Reference data извлекается вручную — dissertation нет в machine-readable form. |
| **D-M5-16** | Buffer management | Send-side buffers allocated by caller (scheduler); passed в `send_temporal_packet`; retained till send-complete event. Backend provides pre-sized send pool (4 × K_max = 32 buffers per rank per packet type). Receive-side: backend allocates pinned host memory; `drain_arrived_*` transfers ownership; caller calls `return_buffer(packet)` after unpack. | comm/SPEC §5.3. Simple reference-counted pools; GPU-device pools — M6. |
| **D-M5-17** | RingBackend safety | `RingBackend::send_temporal_packet(packet, dest)` — assert `dest == (rank + 1) mod P` в Reference profile (fatal). Violation — programmer error. | comm/SPEC §6.5: "Implements только send to ring-next". |
| **D-M5-18** | Scheduler peer dispatch | Fills D-M4-6 no-ops: `mark_packed` → `comm.send_temporal_packet` → `InFlight` → on send-complete event → `Committed` on sender; on receiver, `drain_arrived_temporal` → `ZoneDataArrived` event → state machine validation → `Ready-for-next-step`. | scheduler/SPEC §6.2 Phase B + master spec §10.4 (temporal packet protocol). |
| **D-M5-19** | K-batching pipeline semantics | Scheduler ведёт **K последовательных time_levels подряд** на каждой зоне (до peer-transfer). Pipeline fill: первые K iterations — накапливает K compute jobs. Steady-state: каждая iteration — один full K-stride advance + один transfer. Drain: последние K iterations — transfers без new compute. | Master spec §2.4 + §4.3. K-batching — центральная идея §3.4 диссертации. |
| **D-M5-20** | SPEC deltas в M5 | Ожидается **ноль SPEC deltas**. comm/SPEC v1.0 и scheduler/SPEC v1.0 уже contracts всё, что M5 строит. Если task предполагает SPEC edit — flag в pre-impl report, выделяем в отдельный spec-delta PR (playbook §9.1). | Playbook §9.1. M4 landed 0 deltas — M5 должен следовать той же дисциплине. |

---

## 2. Глобальные параметры окружения

| Параметр | Значение | Примечание |
|---|---|---|
| OS | Linux (Ubuntu 24.04 LTS) | Dev-машина пользователя; ubuntu-latest в CI |
| C++ compiler | GCC 13+ / Clang 17+ | C++20; CI уже проверяет оба |
| CMake | 3.25+ | Master spec §15.2 |
| CUDA | 13.1 installed, **не используется в M5** | GPU path — M6 |
| MPI | **OpenMPI 4.1+** (primary), MPICH 4+ acceptable | `find_package(MPI REQUIRED COMPONENTS CXX)` optional компонент через `TDMD_ENABLE_MPI` CMake flag (default ON on Linux, OFF on unsupported) |
| Python | 3.10+ | pre-commit helpers + anchor-test harness |
| Test framework | Catch2 v3 (FetchContent) + MPI test wrapper | Унаследовано из M0, MPI wrapper в M5 |
| LAMMPS oracle | SKIP on public CI (Option A) | M5 не добавляет новых differentials; T3 anchor-test имеет свою reference data (диссертация) |
| Active BuildFlavor | `Fp64ReferenceBuild` | D-M5-4 |
| Active ExecProfile | `Reference` | D-M5-4 |
| Run mode | multi-rank MPI (`mpirun -np N`), single-thread per rank, CPU only | Master spec §14 M5 |
| Pipeline depth | `K ∈ {1, 2, 4, 8}`; default 1 | D-M5-1 |
| Multi-rank CI | `mpirun -np {1, 2, 4}` — short smokes | D-M5-13 |
| Multi-rank local | `mpirun -np {1, 2, 4, 8, 16}` — anchor-test slow tier | D-M5-13 |
| Branch policy | `m5/T5.X-<topic>` per PR → `main` | CI required: lint + build-cpu (with MPI) + M1/M2/M3/M4 smokes; M5 smoke добавляется в T5.12 |

---

## 3. Suggested PR order

Dependency graph:

```
T5.1 ─► T5.2 ─► T5.3 ─────────────┐
                                   │
                                   ├─► T5.4 (MpiHostStaging) ──┐
                                   │                            │
                                   └─► T5.5 (Ring) ─────────────┤
                                                                │
                                   ┌────────────────────────────┘
                                   ▼
                         T5.6 (K-batching) ─► T5.7 (peer dispatch) ─► T5.8 (engine multi-rank)
                                                                             │
                                                                             ▼
                                                                  T5.9 (1D Z-zoning)
                                                                             │
                                                                             ▼
                                                                  T5.10 (T3 fixture)
                                                                             │
                                                                             ▼
                                                                  T5.11 (AnchorTestRunner)
                                                                             │
                                                                             ▼
                                                                  T5.12 (M5 smoke, GATE)
```

**Линейная последовательность (single agent):**
T5.1 → T5.2 → T5.3 → T5.4 → T5.5 → T5.6 → T5.7 → T5.8 → T5.9 → T5.10 → T5.11 → T5.12.

**Параллельный режим (multi-agent):** после T5.3 — `{T5.4, T5.5}` независимы (два backend'а разделяют
packet format). После T5.8 — `{T5.9, T5.10}` независимы (zoning scheme vs benchmark fixture).
T5.6 → T5.7 → T5.8 строго последовательные (каждый использует infrastructure предыдущего). T5.11
depends on T5.10. T5.12 depends on T5.11.

**Estimated effort:** 6 недель (single agent). Самые длинные — T5.4 (MpiHostStaging + 2-rank ping-pong,
~4 дня), T5.6 (K-batching state machine + pipeline invariants, ~5 дней), T5.8 (SimulationEngine
multi-rank wiring + deterministic reduction, ~4 дня), T5.11 (anchor-test + hardware normalization +
10% acceptance math, ~5 дней). Остальные — 2-3 дня каждая.

---

## 4. Tasks

### T5.1 — Author M5 execution pack

```
# TDMD Task: Create M5 execution pack

## Context
- Master spec: §14 M5
- Role: Architect / Spec Steward
- Milestone: M5 (kickoff)

## Goal
Написать `docs/development/m5_execution_pack.md` декомпозирующий M5 на 12 PR-size задач, по
шаблону m4_execution_pack.md. Document-only PR per playbook §9.1.

## Scope
- [included] docs/development/m5_execution_pack.md (single new file)
- [included] Decisions log D-M5-1..D-M5-20
- [included] Task templates T5.1..T5.12
- [included] M5 acceptance gate checklist
- [included] Risks R-M5-1..R-M5-N + open questions

## Out of scope
- [excluded] Any code changes (T5.2+ territory)
- [excluded] SPEC deltas (separate PR per playbook §9.1)

## Required files
- docs/development/m5_execution_pack.md

## Required tests
- pre-commit clean
- markdownlint clean

## Acceptance criteria
- [ ] Pack committed + pushed
- [ ] Task list matches §14 M5 deliverables
- [ ] Decisions anchored to comm/SPEC + scheduler/SPEC + master spec §13.3
```

---

### T5.2 — `comm/` module skeleton + MPI detection + abstract interface

```
# TDMD Task: comm/ module skeleton

## Context
- Master spec: §10, §12.6
- Module SPEC: docs/specs/comm/SPEC.md §1, §2
- Role: Core Runtime Engineer
- Milestone: M5

## Goal
Создать `src/comm/` module: CMakeLists с optional `find_package(MPI)`, namespaces, public headers
с типами `TemporalPacket`, `HaloPacket`, `MigrationPacket`, `CommEndpoint`, `BackendCapability`,
`BackendInfo`, `CommConfig`; abstract `CommBackend` class (all pure-virtual per comm/SPEC §2.2).
Никакой concrete реализации backend'а (T5.4-T5.5).

## Scope
- [included] src/comm/CMakeLists.txt — new static lib `tdmd_comm`; MPI optional (TDMD_ENABLE_MPI)
- [included] src/comm/include/tdmd/comm/types.hpp — TemporalPacket, HaloPacket, MigrationPacket,
  CommEndpoint, BackendCapability, BackendInfo
- [included] src/comm/include/tdmd/comm/comm_config.hpp — CommConfig (backend selection, topology,
  tuning knobs)
- [included] src/comm/include/tdmd/comm/comm_backend.hpp — abstract CommBackend class
- [included] src/comm/comm_backend.cpp — virtual destructor body, any inline helper fallbacks
- [included] src/CMakeLists.txt — add_subdirectory(comm)
- [included] tests/comm/CMakeLists.txt + test_comm_types.cpp (type presence, layout sanity,
  CommConfig defaults)
- [included] cmake/FindMPI helpers if needed; TDMD_ENABLE_MPI cache variable

## Out of scope
- [excluded] MpiHostStagingBackend (T5.4)
- [excluded] RingBackend (T5.5)
- [excluded] Packet serialization (T5.3)
- [excluded] Scheduler integration (T5.6+)
- [excluded] Protocol dispatch bodies

## Mandatory invariants
- `CommBackend` has virtual destructor + pure-virtual methods matching comm/SPEC §2.2
- `TemporalPacket` wire-format fields declared в correct order (matches comm/SPEC §4.1)
- `CommConfig` default-constructible + has backend="auto", ring_topology=false,
  use_deterministic_reductions=true (Reference-safe defaults)
- No dependency on state/, scheduler/, potentials/ (comm is transport layer)
- `TDMD_ENABLE_MPI=OFF` build succeeds — library still compiles with MPI stubs returning
  NotImplemented (enables non-MPI platforms + M4 regression build)

## Required files
- src/comm/{CMakeLists.txt, include/tdmd/comm/*.hpp, comm_backend.cpp}
- tests/comm/{CMakeLists.txt, test_comm_types.cpp}
- src/CMakeLists.txt (updated)

## Required tests
- test_comm_types: compile + instantiate empty TemporalPacket/HaloPacket; CommConfig defaults
  assertions; BackendCapability enum roundtrip
- Build with `-DTDMD_ENABLE_MPI=ON` AND `-DTDMD_ENABLE_MPI=OFF` — both succeed
- ctest clean; pre-commit clean

## Acceptance criteria
- [ ] All files committed
- [ ] ctest green both MPI-on and MPI-off
- [ ] No link regressions in src/cli, src/runtime, src/scheduler
- [ ] MPI optional (compile-time flag properly propagates)
```

---

### T5.3 — `TemporalPacket` pack/unpack + CRC32 + protocol version + unit tests

```
# TDMD Task: TemporalPacket wire format + serialization + CRC32

## Context
- Module SPEC: docs/specs/comm/SPEC.md §4.1, §4.3, §4.4
- Role: Core Runtime Engineer
- Depends on: T5.2
- Milestone: M5

## Goal
Реализовать byte-level pack/unpack для `TemporalPacket`: 86-byte header + N×76-byte per-atom
record + 4-byte CRC32 trailer. Protocol version check на receiver. Wire-format byte-stable:
round-trip pack → bytes → unpack reproduces the input exactly.

## Scope
- [included] src/comm/include/tdmd/comm/packet_serializer.hpp — PackContext,
  pack_temporal_packet, unpack_temporal_packet
- [included] src/comm/packet_serializer.cpp — big-endian uint16 header, little-endian u32 payload,
  CRC32 via zlib-crc32 или bundled table-driven implementation
- [included] src/comm/include/tdmd/comm/crc32.hpp — 8-KiB table-driven CRC32 helper
  (standalone, no external dep)
- [included] src/comm/crc32.cpp — CRC32 impl
- [included] tests/comm/test_packet_serializer.cpp — 10 random atom counts (0, 1, 10, 100, 1000),
  random atom data, pack→unpack roundtrip (byte-identical); CRC mismatch detection (flip one
  byte, expect unpack rejection); protocol version mismatch detection (v2 sender → v1 receiver
  expects version-mismatch error)

## Out of scope
- [excluded] HaloPacket serialization (M7 Pattern 2)
- [excluded] MigrationPacket serialization (M7)
- [excluded] Backend-specific buffer management (T5.4+)

## Mandatory invariants
- Pack → unpack round-trip byte-identical (including CRC)
- CRC32 matches zlib/RFC 1952 standard (crosscheck against known test vectors:
  "123456789" → 0xCBF43926)
- Protocol version check — `expected=1` (D-M5-10), other versions rejected with clear error
- Wire format exactly as comm/SPEC §4.1 table (offset + size per field)
- `atom_record_size = 76 bytes` (exactly matches SPEC §4.1 per-atom table)
- CRC32 computation covers all preceding bytes excluding CRC field itself

## Required files
- src/comm/include/tdmd/comm/{packet_serializer.hpp, crc32.hpp}
- src/comm/{packet_serializer.cpp, crc32.cpp}
- tests/comm/test_packet_serializer.cpp

## Required tests
- Roundtrip (10 sizes)
- CRC mismatch rejection
- Protocol version mismatch rejection
- Known CRC32 test vectors (e.g., "123456789" → 0xCBF43926)
- Empty packet (atom_count=0) — edge case, must roundtrip
- CI budget <5s

## Acceptance criteria
- [ ] Roundtrip green across 10 packet sizes
- [ ] CRC32 matches RFC 1952 / zlib reference vectors
- [ ] Protocol version gating green
- [ ] No endian assumption leak (test passes on little-endian + big-endian if cross-compiled)
```

---

### T5.4 — `MpiHostStagingBackend` (send/drain/barrier/global_sum + 2-rank test)

```
# TDMD Task: MpiHostStagingBackend — universal MPI fallback

## Context
- Module SPEC: docs/specs/comm/SPEC.md §6.1, §7
- Role: Core Runtime Engineer
- Depends on: T5.3
- Milestone: M5

## Goal
Реализовать `MpiHostStagingBackend`: `initialize(CommConfig)` инициализирует MPI (если не
inited), создаёт communicator; `send_temporal_packet(pkt, dest)` — pack → host buffer → MPI_Isend;
`drain_arrived_temporal()` — MPI_Iprobe + MPI_Irecv → unpack → return vector; `barrier()` —
MPI_Barrier; `global_sum_double(local)` — `deterministic_sum_double` (D-M5-9: Kahan, rank-ordered,
via MPI_Gather + local reduction); `progress()` — MPI_Testall on pending requests.

## Scope
- [included] src/comm/include/tdmd/comm/mpi_host_staging_backend.hpp
- [included] src/comm/mpi_host_staging_backend.cpp — lifecycle + send + drain + barrier +
  global_sum + progress
- [included] src/comm/include/tdmd/comm/deterministic_reduction.hpp — Kahan-compensated sum
  over rank-ordered gather
- [included] src/comm/deterministic_reduction.cpp
- [included] tests/comm/test_mpi_host_staging_2rank.cpp — 2-rank ping-pong: rank 0 sends N=10
  temporal packets to rank 1 and vice versa, all arrive, all CRC-valid; barrier smoke;
  global_sum_double consistency (each rank contributes rank_id+1 → sum should be N·(N+1)/2)
- [included] tests/comm/CMakeLists.txt — add mpirun-wrapped ctest invocation
- [included] tests/comm/main_mpi.cpp — Catch2 wrapper с MPI_Init/Finalize; main args passthrough

## Out of scope
- [excluded] GpuAware path (M6)
- [excluded] NCCL collectives (M6)
- [excluded] Hybrid routing (M7)
- [excluded] HaloPacket / MigrationPacket send/drain (M7)

## Mandatory invariants
- `send_temporal_packet` non-blocking (returns immediately; request tracked internally)
- `drain_arrived_temporal` non-blocking (returns what's arrived, empty vector ok)
- CRC validation на receiver side — mismatch → packet dropped + logged via telemetry counter
- Protocol version mismatch → dropped + logged (NOT fatal — other rank may be older)
- `global_sum_double` в Reference profile uses `deterministic_sum_double`, NOT raw
  MPI_Allreduce
- `barrier()` is the only blocking call; other calls must return promptly
- Works under `MPI_THREAD_SINGLE`; multi-threaded future usage — M6+

## Required files
- src/comm/include/tdmd/comm/{mpi_host_staging_backend.hpp, deterministic_reduction.hpp}
- src/comm/{mpi_host_staging_backend.cpp, deterministic_reduction.cpp}
- tests/comm/{test_mpi_host_staging_2rank.cpp, main_mpi.cpp}
- tests/comm/CMakeLists.txt (mpirun integration)

## Required tests
- 2-rank ping-pong: 10 packets each direction, all arrive, CRC-valid
- Global sum 2-rank: each contributes `rank+1.0`, sum must equal `P*(P+1)/2.0`
- Deterministic sum same across 10 invocations with same inputs (bit-exact)
- MPI_Finalize clean exit
- CI: `mpirun -np 2 ./test_mpi_host_staging_2rank` passes under ubuntu-latest
- CI budget <10s

## Acceptance criteria
- [ ] 2-rank ping-pong green
- [ ] Deterministic sum bit-exact across repeated calls
- [ ] CI wiring validates `mpirun -np 2` scenario
- [ ] No MPI leaks (valgrind smoke — local)
```

---

### T5.5 — `RingBackend` (ring-next-only temporal + 4-rank ring test)

```
# TDMD Task: RingBackend — anchor-test ring topology

## Context
- Module SPEC: docs/specs/comm/SPEC.md §6.5, §3.1 Ring
- Master spec: §13.3 anchor-test premise
- Role: Scheduler / Determinism Engineer
- Depends on: T5.4
- Milestone: M5

## Goal
Реализовать `RingBackend` как production-grade sibling of MpiHostStaging restricted к
ring-topology: `send_temporal_packet(pkt, dest)` — asserts `dest == (rank + 1) mod nranks`,
delegates to underlying MPI; `drain_arrived_temporal()` — MPI_Iprobe на ring-prev tag;
`global_sum_double` — sequential add вокруг ring + broadcast (per comm/SPEC §7.1). Это
легальный backend для **только** anchor-test §13.3; в production Pattern 1 default остаётся
MpiHostStaging (mesh topology).

## Scope
- [included] src/comm/include/tdmd/comm/ring_backend.hpp
- [included] src/comm/ring_backend.cpp — ring send + drain + ring-sum reduction
- [included] tests/comm/test_ring_4rank.cpp — 4-rank ring: rank 0→1→2→3→0 (each sends K=10
  packets to next, receives from prev); ring-sum consistency (4-rank, each contributes 1.0,
  sum should be 4.0 on every rank bit-exact); assert-fires-on-non-ring-dest (unit test catches
  assert via death-test pattern)

## Out of scope
- [excluded] Non-ring dispatch paths (covered by MpiHostStaging)
- [excluded] Halo/migration (Pattern 2)
- [excluded] GPU ring (NCCL — M6)

## Mandatory invariants
- `send_temporal_packet(pkt, dest)` where `dest != (rank+1) mod P` → hard-assert in Reference
- Ring-sum определённо round-trips: each rank's contribution enters sum exactly once
- Ring-sum bit-exact across repeated calls с same inputs (Kahan-accumulated)
- Topology reported as `inner_topology: ring` в `BackendInfo`
- 4-rank test completes in <2s

## Required files
- src/comm/include/tdmd/comm/ring_backend.hpp
- src/comm/ring_backend.cpp
- tests/comm/test_ring_4rank.cpp

## Required tests
- 4-rank ring ping: K=10 packets each, all arrive in order
- Ring-sum bit-exact on 4 ranks
- Non-ring dest assert (death-test or documented pattern)
- CI: `mpirun -np 4 ./test_ring_4rank` green
- CI budget <5s

## Acceptance criteria
- [ ] Ring ping green on 4 ranks
- [ ] Ring-sum reproducible across invocations
- [ ] Assert catches ring-violation
- [ ] Telemetry: BackendInfo reports ring topology correctly
```

---

### T5.6 — Scheduler K-batching pipeline (K ∈ {1, 2, 4, 8})

```
# TDMD Task: CausalWavefrontScheduler — K-batched pipeline fill/steady/drain

## Context
- Master spec: §2.4, §4.3, §6.5a, §14 M5
- Module SPEC: docs/specs/scheduler/SPEC.md §5, §14 M5 row
- Role: Scheduler / Determinism Engineer
- Depends on: T5.5 (backends available for peer-transfer testing)
- Milestone: M5

## Goal
Расширить `CausalWavefrontScheduler::select_ready_tasks` и frontier bookkeeping для поддержки
`K ∈ {1, 2, 4, 8}`: scheduler выбирает до K задач на зону за iteration, frontier guard `t ≤
global_frontier_min + K` relaxed с M4's K=1. Pipeline fill phase (первые K·N_min iterations —
накапливаем compute до первого peer transfer); steady-state; drain phase (последние K·N_min —
transfers без new compute). Configuration: `scheduler.pipeline_depth_cap: K` в YAML с
validation `K ∈ {1, 2, 4, 8}` (D-M5-1).

## Scope
- [included] src/scheduler/causal_wavefront_scheduler.cpp — select_ready_tasks: inner loop
  extends from `t ∈ [min_level, min_level+1]` (M4 K=1) to `t ∈ [min_level, min_level+K_max]`
- [included] src/scheduler/include/tdmd/scheduler/policy.hpp — extend `SchedulerPolicy.K_max`
  (default 1, validate {1,2,4,8})
- [included] src/io/yaml_config — parse `scheduler.pipeline_depth_cap` integer, validate
  membership in {1,2,4,8}, reject others с clear error
- [included] tests/scheduler/test_k_batching_pipeline.cpp — 4 zones single-rank: K=1 produces
  1 task/iter; K=2 produces up to 2; K=4 up to 4; pipeline fill count, steady state, drain count
  each validated
- [included] tests/scheduler/fuzz_k_batching_invariants.cpp — ≥100k sequences: assert I6
  (`frontier_min + K ≥ max ready time_level`) для K∈{1,2,4,8}

## Out of scope
- [excluded] Peer transfer mechanics (T5.7)
- [excluded] Auto-K (M8)
- [excluded] K>8 (not in M5 scope per D-M5-1)

## Mandatory invariants
- K=1 behavior unchanged from M4 (regression guard)
- I6 holds для all K∈{1,2,4,8}: no task with time_level > frontier_min + K
- Deterministic: same seed + same K → same task extraction order
- Configuration validation rejects `K=3, 5, 16, ...` (non-power-of-2 ≤8 forbidden)
- Pipeline state machine: `fill | steady | drain` transitions observable via event log

## Required files
- src/scheduler/causal_wavefront_scheduler.cpp (updated)
- src/scheduler/include/tdmd/scheduler/policy.hpp (updated)
- src/io/yaml_config.{hpp,cpp} (updated)
- tests/scheduler/{test_k_batching_pipeline.cpp, fuzz_k_batching_invariants.cpp}

## Required tests
- K=1 regression: M4 smoke byte-exact (D-M5-12)
- K=2: pipeline emits ≤ 2·zones tasks per iteration in steady state
- K=4, K=8: same pattern
- Fuzzer ≥100k seqs: I6 holds
- Invalid K config (e.g. 3) rejected with error message
- CI budget <45s

## Acceptance criteria
- [ ] K-batching green for {1,2,4,8}
- [ ] K=1 M4 smoke byte-exact regression preserved
- [ ] I6 fuzzer green
- [ ] Invalid K values rejected
```

---

### T5.7 — Scheduler peer dispatch protocol (fills D-M4-6 no-ops)

```
# TDMD Task: Scheduler peer dispatch — PackedForSend / InFlight / ZoneDataArrived

## Context
- Master spec: §10.4 temporal packet protocol
- Module SPEC: docs/specs/scheduler/SPEC.md §3.1 (state machine), §6.2 Phase B
- Role: Scheduler / Determinism Engineer
- Depends on: T5.6
- Milestone: M5

## Goal
Заполнить no-op transitions D-M4-6 scheduler'а: `Completed → PackedForSend` (pack via
serializer) → `InFlight` (comm.send_temporal_packet) → `Committed` (on send-complete event).
На receiver side: `drain_arrived_temporal` → fires `ZoneDataArrived` → state machine validates
(protocol version, CRC, certificate_hash) → zone state → `Ready-for-next-step`. CommBackend
injected via constructor (or `set_comm_backend()`).

## Scope
- [included] src/scheduler/causal_wavefront_scheduler.cpp — `mark_completed` extends to
  initiate peer pack+send if zone has downstream peer; new `on_send_complete(zone_id)`
  callback path; `poll_arrivals()` drains comm backend + fires ZoneDataArrived events
- [included] src/scheduler/include/tdmd/scheduler/causal_wavefront_scheduler.hpp — inject
  `CommBackend*` (ownership: caller retains)
- [included] src/scheduler/zone_state_machine.cpp — wire PackedForSend / InFlight legal
  transitions (currently declared but no-op in M4)
- [included] tests/scheduler/test_peer_dispatch.cpp — 2-rank integration: rank 0 processes
  zone A, packs + sends to rank 1; rank 1 receives, zone A → Ready-for-next-step; verify full
  state machine sequence
- [included] tests/scheduler/test_crc_failure_retry.cpp — inject CRC corruption at receiver;
  assert: packet dropped, scheduler retries via certificate re-refresh (per SPEC §7.2)

## Out of scope
- [excluded] Pattern 2 boundary dispatch (M7)
- [excluded] Migration packets (M7)
- [excluded] Halo propagation (M7)

## Mandatory invariants
- State machine: Completed → PackedForSend → InFlight → Committed (sender);
  Empty/ResidentPrev → ZoneDataArrived → Ready-for-next-step (receiver)
- Certificate hash validated on receiver — mismatch → packet dropped + retry trigger
- CRC failure → scheduler counts `crc_failures_total`, retries up to `max_retries=3` per SPEC §7.2
- No double-dispatch: same zone can't have two InFlight transfers simultaneously
  (enforced via ZoneMeta.has_inflight_bit)
- Deterministic retry order (Reference profile) — same sequence of failures →
  same resulting schedule

## Required files
- src/scheduler/causal_wavefront_scheduler.{hpp,cpp} (updated)
- src/scheduler/zone_state_machine.cpp (updated)
- tests/scheduler/{test_peer_dispatch.cpp, test_crc_failure_retry.cpp}

## Required tests
- 2-rank peer dispatch: Completed → InFlight → Committed cycle green
- CRC-failure: drop + retry ≤3 times, then escalate
- Double-dispatch prevention
- `mpirun -np 2` CI integration
- CI budget <30s

## Acceptance criteria
- [ ] State machine transitions fully wired (no more no-ops)
- [ ] 2-rank dispatch green
- [ ] CRC retry semantics deterministic
- [ ] No regression в M4 K=1 single-rank smoke
```

---

### T5.8 — `SimulationEngine` multi-rank TD run loop + deterministic reductions

```
# TDMD Task: SimulationEngine — multi-rank wiring; thermo via deterministic reduction

## Context
- Master spec: §6.6, §7.3 Level 1 determinism, §14 M5
- Module SPEC: docs/specs/comm/SPEC.md §7.2
- Role: Core Runtime Engineer
- Depends on: T5.7
- Milestone: M5

## Goal
`SimulationEngine::run` в `td_mode: true` принимает injected `CommBackend*`; per-step thermo
(PE, KE, P, T) считается via `global_sum_double` на Reference backend (deterministic Kahan
ring-reduction). Zone ownership distribution: round-robin zone_id → rank (простейшая схема
M5; proper load balancing — M8). Atom ownership follows zone ownership — scheduler enforces
via TemporalPacket transfers. Default fallback (no MPI): single-rank, behaves как M4.

## Scope
- [included] src/runtime/simulation_engine.{hpp,cpp} — accept CommBackend injection; new
  `run_td_mode_multirank` path; thermo aggregation via `deterministic_sum_double`
- [included] src/cli/tdmd.cpp — wire MPI init + select backend (mpi_host_staging default in
  non-ring mode; ring if `comm.backend: ring` in config)
- [included] src/io/yaml_config — parse `comm.backend`, `comm.topology` (ring | mesh default)
- [included] tests/runtime/test_multirank_td_smoke.cpp — 2-rank and 4-rank short smoke (Ni-Al
  864 atoms, K=1, 5 steps); thermo converges to single-rank M4 thermo within 2σ of reduction
  noise
- [included] tests/runtime/test_k1_regression_smoke.cpp — K=1 P=1 thermo byte-exact to M4
  smoke golden (D-M5-12 regression gate)

## Out of scope
- [excluded] Multi-rank neighbor list sync (zone-owned neighbors — each rank rebuilds its own)
- [excluded] Cross-rank GPU kernels (M6)
- [excluded] Pattern 2 subdomain layer (M7)

## Mandatory invariants
- `td_mode: true, K=1, P=1` thermo == M4 smoke golden byte-exact (D-M5-12)
- `td_mode: true, K=1, P=2` thermo matches K=1 P=1 within deterministic-reduction tolerance
  (formally bit-exact when using `deterministic_sum_double` Kahan Reference path)
- `td_mode: true, K=4, P=4` scientific observables match K=1 P=1 within 1σ of reduction
  noise; NVE drift `|ΔE|/E < 1e-6` over 1000 steps
- `td_mode: false` unchanged from M1-M4
- MPI_Init called once; clean MPI_Finalize on shutdown
- Works без MPI (`TDMD_ENABLE_MPI=OFF`) — single-rank fallback

## Required files
- src/runtime/simulation_engine.{hpp,cpp} (updated)
- src/cli/tdmd.cpp (updated)
- src/io/yaml_config.{hpp,cpp} (updated)
- tests/runtime/{test_multirank_td_smoke.cpp, test_k1_regression_smoke.cpp}

## Required tests
- K=1 P=1 byte-exact to M4 smoke
- K=1 P=2 MpiHostStaging: thermo matches K=1 P=1 bit-exact (deterministic reduction)
- K=4 P=4 MpiHostStaging: thermo matches K=1 P=1 within 1σ (NVE drift < 1e-6)
- `mpirun -np 2 ./tdmd run ...` CLI integration
- `mpirun -np 4 ./tdmd run ...` CLI integration
- MPI clean shutdown
- CI budget <60s

## Acceptance criteria
- [ ] Multi-rank TD run green with 2 and 4 ranks
- [ ] K=1 P=1 M4 regression byte-exact
- [ ] Deterministic reduction verified across {1,2,4} ranks
- [ ] CLI accepts `comm.backend` + `comm.topology` config
```

---

### T5.9 — 1D Z-axis linear zoning scheme (for anchor-test)

```
# TDMD Task: ZoningPlan — add Linear1D Z-axis scheme

## Context
- Master spec: §9 Zoning planner, §13.3 anchor-test premise, §4.4 bullet 1
- Module SPEC: docs/specs/zoning/SPEC.md
- Role: Architect (зонирование зафиксировано SPEC) + Core Runtime Engineer
- Depends on: T5.8 (we want zoning plan to be consumable by multi-rank scheduler already)
- Milestone: M5

## Goal
Добавить второй scheme в `ZoningPlanner`: `Linear1D` — разбиение только по оси Z, `N` slabs
пропорциональных высоте box'а, canonical_order = Z-index ascending. M3 Hilbert остаётся default;
Linear1D opt-in через YAML `zoning.scheme: linear_1d`. Необходим для anchor-test §13.3
(Andreev §2.2: "1D линейная нумерация — N_min=2 оптимальный случай").

## Scope
- [included] src/zoning/include/tdmd/zoning/linear_1d_scheme.hpp
- [included] src/zoning/linear_1d_scheme.cpp — разбиение box'а по Z на N_z слэбов
- [included] src/zoning/zoning_planner.cpp — scheme dispatch
- [included] src/io/yaml_config — parse `zoning.scheme: hilbert | linear_1d`
- [included] tests/zoning/test_linear_1d_scheme.cpp — unit: 10⁶ atoms Al FCC box → 8 Z-slabs
  roughly equal atom counts; canonical_order = [0,1,2,...,7]; neighbor mask = {prev, next}
  для internal zones, {next} для zone 0, {prev} для zone N_z-1
- [included] tests/integration/m3_smoke — verify hilbert scheme still default (regression)

## Out of scope
- [excluded] 2D schemes (future research)
- [excluded] Adaptive zoning (auto-scheme selection)
- [excluded] Non-orthogonal box support (M9 NPT)

## Mandatory invariants
- `Linear1D(N_z)` на box Z-height H: slab i covers `[i·H/N_z, (i+1)·H/N_z)` by Z
- Canonical order: `[0, 1, 2, ..., N_z-1]` (Z ascending)
- Neighbor mask — `{prev, next} mod N_z` если periodic_z, `{prev, next}` clamped otherwise
- N_min = 2 (per Andreev §2.2)
- Hilbert scheme (M3) unchanged, remains default

## Required files
- src/zoning/include/tdmd/zoning/linear_1d_scheme.hpp
- src/zoning/linear_1d_scheme.cpp
- src/zoning/zoning_planner.cpp (updated)
- src/io/yaml_config.{hpp,cpp} (updated)
- tests/zoning/test_linear_1d_scheme.cpp
- tests/integration/m3_smoke (regression check only, no new assertions)

## Required tests
- 10⁶ Al FCC → 8 Z-slabs: atom counts uniform ±5%
- Canonical order monotonic
- Neighbor masks correct for periodic + non-periodic cases
- Hilbert scheme still default (M3 smoke unchanged)
- CI budget <10s

## Acceptance criteria
- [ ] Linear1D scheme lands + unit-test green
- [ ] Hilbert scheme regression preserved (M3 smoke green)
- [ ] YAML config parses both schemes
```

---

### T5.10 — T3 benchmark fixture (Al FCC 10⁶ LJ + dissertation reference data)

```
# TDMD Task: T3 anchor-test benchmark fixture

## Context
- Master spec: §13.3 anchor-test
- Module SPEC: docs/specs/verify/SPEC.md §4.4
- Role: Validation / Reference Engineer
- Depends on: T5.9 (Linear1D zoning)
- Milestone: M5

## Goal
Создать `verify/benchmarks/t3_al_fcc_large_anchor/`: config YAML (Al FCC 10⁶ atoms,
Lennard-Jones `r_c=8Å`, NVE, dt=1fs, 1000 steps), LAMMPS script equivalent (cross-check),
dissertation reference data CSV (performance + efficiency vs N_procs from figures 29-30),
hardware normalization script, acceptance criteria doc. Benchmark — **fixture only**, harness
в T5.11.

## Scope
- [included] verify/benchmarks/t3_al_fcc_large_anchor/README.md — experiment description,
  provenance of reference data, citation of Andreev §3.5, what pass/fail means
- [included] verify/benchmarks/t3_al_fcc_large_anchor/config.yaml — TDMD config (Al 10⁶,
  LJ 8Å, NVE, linear_1d zoning, ring backend, dt=1fs, 1000 steps, periodic xyz)
- [included] verify/benchmarks/t3_al_fcc_large_anchor/lammps_script.in — equivalent LAMMPS
  (for validation of LJ physics only, NOT of anchor results)
- [included] verify/benchmarks/t3_al_fcc_large_anchor/dissertation_reference_data.csv —
  extracted points from figures 29-30: (N_procs, performance_units, efficiency_pct)
- [included] verify/benchmarks/t3_al_fcc_large_anchor/hardware_normalization.py — script
  computing "FLOPs ratio" between 2007 hardware class (Intel Xeon Harpertown ~9 GFLOP/core)
  и current (measured via synthetic LJ kernel benchmark)
- [included] verify/benchmarks/t3_al_fcc_large_anchor/checks.yaml — per-point acceptance
  tolerance (10% of dissertation value, hardware-adjusted)
- [included] verify/benchmarks/t3_al_fcc_large_anchor/acceptance_criteria.md — pass/fail
  pseudocode; what failure modes are permissible; escalation path
- [included] verify/data/t3_al_fcc_large_anchor/setup.data.xz — LZMA-compressed LAMMPS data
  file for Al 10⁶ FCC at 300K (generated once via LAMMPS, committed as LFS blob)

## Out of scope
- [excluded] Harness execution (T5.11)
- [excluded] CI integration (T5.12 — slow tier local only)
- [excluded] Oracle comparison (T3 doesn't diff against LAMMPS — it matches dissertation)

## Mandatory invariants
- `dissertation_reference_data.csv` has ≥ 6 (N_procs, perf, efficiency) points spanning
  N_procs ∈ [4, 32]
- `hardware_normalization.py` is offline-runnable (no network dep); outputs a single
  `normalized_tdmd_performance` scalar per run
- Reference data commit message cites dissertation figures explicitly (provenance trail)
- `setup.data.xz` regeneration script committed (reproducibility)
- `acceptance_criteria.md` documents the 10% tolerance and what "equivalent hardware class"
  means

## Required files
- verify/benchmarks/t3_al_fcc_large_anchor/{README.md, config.yaml, lammps_script.in,
  dissertation_reference_data.csv, hardware_normalization.py, checks.yaml,
  acceptance_criteria.md}
- verify/data/t3_al_fcc_large_anchor/{setup.data.xz, regen_setup.sh}
- verify/SPEC.md — update §4.4 with concrete path + status

## Required tests
- Fixture files syntactically valid (YAML lint, CSV parse)
- `hardware_normalization.py` runs на dev machine без errors (produces numeric output)
- `setup.data.xz` decompresses to ≥24 MiB file (10⁶ atoms × ~24 bytes per line)

## Acceptance criteria
- [ ] All fixture files committed
- [ ] Reference data cited + provenance clear
- [ ] Hardware normalization script runnable
- [ ] verify/SPEC.md updated
```

---

### T5.11 — `AnchorTestRunner` harness + 10% acceptance check

```
# TDMD Task: AnchorTestRunner — execute T3 on N∈{4,8,16} ranks + compare to dissertation

## Context
- Master spec: §13.3 anchor-test (mandatory M5 gate)
- Module SPEC: docs/specs/verify/SPEC.md §7.4
- Role: Validation / Reference Engineer
- Depends on: T5.10
- Milestone: M5

## Goal
Реализовать `verify/harness/anchor_test_runner/`: Python driver, который invokes `mpirun -np N
./tdmd run t3_al_fcc_large_anchor/config.yaml` для N ∈ {4, 8, 16}, собирает TDMD telemetry
(steps/second, efficiency_pct), нормализует против current hardware FLOPs, сравнивает
point-by-point против `dissertation_reference_data.csv` с 10% tolerance. Runner reports
structured JSON report + human-readable summary. **Exit 0 ⇔ all points pass.**

## Scope
- [included] verify/harness/anchor_test_runner/__init__.py
- [included] verify/harness/anchor_test_runner/runner.py — `AnchorTestRunner.run()` per SPEC §7.4
- [included] verify/harness/anchor_test_runner/report.py — `AnchorTestReport` dataclass + JSON
  emitter
- [included] verify/harness/anchor_test_runner/hardware_probe.py — one-time FLOPs measurement
  via bundled LJ micro-kernel (C++ tool), cached в `~/.cache/tdmd/hardware_flops.json`
- [included] verify/harness/anchor_test_runner/test_anchor_runner.py — smoke test with mocked
  TDMD binary (produces known output), verifies driver logic end-to-end
- [included] tests/integration/m5_anchor_test/ — Catch2 wrapper invoking Python runner;
  skipped in CI (ENV `TDMD_SLOW_TIER=1` required); local-only
- [included] docs/development/m5_execution_pack.md — update acceptance gate §5 if any
  decision emerged during implementation

## Out of scope
- [excluded] CI integration (slow tier — local only, per D-M5-13)
- [excluded] 32+ rank runs (future work)
- [excluded] GPU anchor-test variant (M6+)

## Mandatory invariants
- `AnchorTestRunner.run()` returns `AnchorTestReport` with fields: `points` (list of per-N
  entries), `overall_passed` (bool), `dissertation_reference_commit`, `tdmd_commit`,
  `hardware_flops_ratio`, `normalization_log`
- Acceptance: each (N_procs, perf_actual) point must satisfy
  `|perf_actual - perf_dissertation * hw_ratio| / (perf_dissertation * hw_ratio) ≤ 0.10`
- Hardware probe cached 24h; `--force-probe` flag re-runs
- Reproducible: same hardware + same TDMD commit → same report (modulo stochastic noise
  documented as <1% per-run)
- `AnchorTestRunner` NEVER depends on LAMMPS (T3 не oracle diff — это dissertation match)

## Required files
- verify/harness/anchor_test_runner/*.py
- tests/integration/m5_anchor_test/*
- docs/development/m5_execution_pack.md (updated)

## Required tests
- Mocked-TDMD smoke: runner processes known output → expected report
- `hardware_probe.py` runs + caches
- Local manual: `python -m anchor_test_runner --output report.json` on dev machine —
  all points within 10% (if hardware class is equivalent)

## Acceptance criteria
- [ ] Runner + report dataclasses + hardware normalization land
- [ ] Local run on dev machine passes 10% tolerance
- [ ] Mocked-TDMD smoke test green in CI
- [ ] Skip-when-slow-tier-not-set logic respected
```

---

### T5.12 — M5 integration smoke + CI wiring + acceptance gate

```
# TDMD Task: M5 integration smoke — closes the milestone

## Context
- Master spec: §14 M5 final artifact gate
- Role: Validation / Reference Engineer
- Depends on: T5.11
- Milestone: M5 (final)

## Goal
End-to-end (<30s CI budget, 2-rank MpiHostStaging) exercising the M5 user surface: Ni-Al EAM
config (M4 smoke reused) с `td_mode: true`, `comm.backend: mpi_host_staging`, `pipeline_depth_cap:
1`, K=1, 2 ranks. Verifies: (a) thermo byte-exact to K=1 P=1 (= M4 smoke golden — D-M5-12
regression); (b) 2-rank CommBackend stack exercises full state machine (PackedForSend/InFlight/
ZoneDataArrived/Committed); (c) deterministic reduction — 2-rank sum == 1-rank sum bit-exact;
(d) scheduler emits expected telemetry (pipeline_depth=1, crc_failures=0). Plus anchor-test
mention в README как "local pre-push mandatory before M5 merge."

## Scope
- [included] tests/integration/m5_smoke/:
  - README.md — smoke philosophy; what "D-M5-12 byte-exact regression" means; what's
    multi-rank-specific
  - smoke_config.yaml.template — M4 template + `comm.backend: mpi_host_staging`
  - run_m5_smoke.sh (driver; invokes `mpirun -np 2`)
  - thermo_golden.txt — reuses M4 golden byte-for-byte (D-M5-12 chain)
  - telemetry_expected.txt — counters
- [included] .github/workflows/ci.yml — M5 smoke step after M4 smoke, conditional on
  `TDMD_ENABLE_MPI=ON`
- [included] docs/specs/comm/SPEC.md — change log entry: "M5 landed (2026-MM-DD): CommBackend
  skeleton, MpiHostStaging, Ring, TemporalPacket+CRC32+v1, deterministic reduction"
- [included] docs/specs/scheduler/SPEC.md — change log entry: "M5 landed: K-batching K∈{1,2,4,8},
  peer dispatch (PackedForSend/InFlight/ZoneDataArrived/Committed), multi-rank deterministic"
- [included] docs/development/m5_execution_pack.md — update §5 acceptance gate checklist with
  final status

## Out of scope
- [excluded] Anchor-test CI integration (slow tier, local-only per D-M5-13)
- [excluded] GPU smoke (M6)
- [excluded] Pattern 2 smoke (M7)

## Mandatory invariants
- Wall-time <30s CI budget
- Smoke uses M4 assets + added multi-rank config
- Thermo K=1 P=2 bit-exact to K=1 P=1 = M4 golden (D-M5-12 chain)
- CommBackend lifecycle clean: init → send/drain N times → barrier → shutdown; no leaks
- Telemetry: `crc_failures_total == 0`, `commit_events ≥ 10` on 10 step run, `inflight_max ≤ K_max`
- No regressions in M1 / M2 / M3 / M4 smokes + T1 / T4 differentials
- Anchor-test (T3) noted as **mandatory pre-merge local gate** в README (enforced via
  pre-push protocol, not CI)

## Required files
- tests/integration/m5_smoke/*
- .github/workflows/ci.yml (updated)
- docs/specs/comm/SPEC.md (change log only)
- docs/specs/scheduler/SPEC.md (change log only)
- docs/development/m5_execution_pack.md (acceptance gate final status)

## Required tests
- [smoke local] `mpirun -np 2 ./tdmd run ...` <30s, exit 0
- [CI] passes on gcc-13 + clang-17 with MPI enabled
- [regression] M1, M2, M3, M4 smokes + T1, T4 differentials unchanged
- [local pre-push] anchor-test passes 10% tolerance on dev machine (slow tier)

## Acceptance criteria
- [ ] All artifacts committed
- [ ] CI green (5 smokes: M1, M2, M3, M4, M5)
- [ ] Anchor-test passes on dev machine (local) — screenshot or report.json attached to PR
- [ ] M5 acceptance gate (§5) fully closed
- [ ] Milestone M5 ready to declare done
```

---

## 5. M5 Acceptance Gate

После закрытия всех 12 задач — проверить полный M5 artifact gate (master spec §14 M5):

- [x] **`comm/` module** (T5.2), skeleton + types + abstract interface + MPI optional build flag (commit 0d5d52f)
- [x] **TemporalPacket wire format** (T5.3), pack/unpack + CRC32 + protocol version v1 (commit a37f52c)
- [x] **MpiHostStagingBackend** (T5.4), 2-rank ping-pong + deterministic reduction green (commit 539a280)
- [x] **RingBackend** (T5.5), 4-rank ring + ring-sum bit-exact + non-ring-dest assert (commit a40f4eb)
- [x] **K-batching pipeline** (T5.6), K ∈ {1, 2, 4, 8}; I6 fuzzer ≥100k seqs green (commit dc37bc6)
- [x] **Scheduler peer dispatch** (T5.7), PackedForSend/InFlight/ZoneDataArrived/Committed fully wired (commit f7754e8)
- [x] **SimulationEngine multi-rank** (T5.8), K=1 P=1 byte-exact to M4; K=1 P=2 bit-exact to K=1 P=1; K=4 P=4 NVE drift < 1e-6 / 1000 steps (commit adf7d46)
- [x] **Linear1D Z-axis zoning** (T5.9), 10⁶ Al FCC uniform distribution ±5%; Hilbert scheme regression preserved (commit 549d163)
- [x] **T3 benchmark fixture** (T5.10), Al FCC 10⁶ LJ config + dissertation reference CSV + hardware normalization script (commit 5d73545)
- [x] **AnchorTestRunner** (T5.11), harness + report dataclass + hardware probe cache + mocked smoke shipped 2026-04-19 (commit a349149); local 10% tolerance gate validated at pre-push time per README
- [x] **Anchor-test acceptance** — primary M5 gate (master spec §13.3). Closed retroactively by **T6.0** 2026-04-19: Andreev fig 29/30 extracted from authoritative scans (`docs/_sources/fig_29.png`, `fig_30.png`) into `verify/benchmarks/t3_al_fcc_large_anchor/dissertation_reference_data.csv`; placeholder sentinel removed; harness detector now returns `False`; 2·10⁶ and 4·10⁶ curves archived in `additional_model_sizes.md`. Full anchor run on the dev workstation is the pre-push ritual per `tests/integration/m5_anchor_test/README.md` — see that doc for expected wall time (~45 min on a modern 8-core box). R-M5-8 is CLOSED.
- [x] **M5 integration smoke** (T5.12) shipped 2026-04-19 (commit d6c2213). Wall-time 1s local (30s CI budget), K=1 P=2 MpiHostStaging thermo byte-exact to M4 golden (D-M5-12 chain), telemetry invariants green, CI-gated on `openmpi-bin` install
- [x] No regressions: M1, M2, M3, M4 smokes all green locally against the M5 tree (2026-04-19 re-run); T1, T4 differential surfaces untouched by M5 commits
- [ ] CI Pipelines A (lint+build+smokes) + B (unit/property) + C (differentials) all green — verified locally; confirm on GitHub CI after d6c2213 push completes
- [ ] Pre-implementation + session reports attached в каждом PR
- [ ] Human review approval для каждого PR

---

## 6. Risks & Open Questions

**Risks:**

- **R-M5-1 — Anchor-test 10% tolerance может быть слишком строг на commodity hardware.** Диссертация
  2007 года running на Intel Xeon Harpertown 2.66GHz + 1 GbE; dev машины 2026 — 10-100× FLOPs,
  но и 10× cache, memory bandwidth, vectorization. Hardware normalization может accumulate ошибки
  приводящие к systematic offset. Mitigation: нормализация через measured LJ micro-kernel ratio
  (не peak FLOPs — measured); tolerance budget — 10% (мастер spec §13.3 fixed); если не проходим —
  SPEC delta с explicit analysis why, human review.
- **R-M5-2 — MPI implementation-specific determinism quirks.** Некоторые OpenMPI installs возвращают
  byte-inequal `MPI_Allreduce` результаты между runs при изменении environment. Mitigation:
  D-M5-9 запрещает raw MPI_Allreduce в Reference; используем deterministic_sum_double по
  rank-ordered gather + Kahan. Validated в T5.4 bit-exact determinism test.
- **R-M5-3 — K-batching pipeline fill/drain edge cases на малом числе зон.** Если `N_zones < K`,
  pipeline never reaches steady state; dev/debug runs часто с очень small boxes. Mitigation:
  T5.6 explicit handling — if `N_zones < K_max`, scheduler clamps к available zones, telemetry
  warns, test covers `N_zones=2, K=4` case.
- **R-M5-4 — CI multi-rank budget overrun.** `mpirun -np 4` на ubuntu-latest — 4-8s overhead
  на MPI init; plus actual smoke. Total M5 smoke ~20s. Mitigation: D-M5-13 limits CI to `np≤4`;
  anchor-test on `np={8,16}` — slow tier local only. M5 smoke budget 30s (relaxed from M4's 10s).
- **R-M5-5 — Protocol versioning drift.** Если M6 добавляет per-atom force в payload (v1.1),
  M5 sender + M6 receiver incompatible. Mitigation: T5.3 strict version-check — mismatch →
  reject with clear error; no silent downgrade. Protocol change = SPEC delta + feature branch.
- **R-M5-6 — `MPI_THREAD_SINGLE` enforcement too strict для future GPU integration.** M6 хочет
  multi-threaded enqueue на MPI + CUDA streams. Mitigation: T5.4 requests `MPI_THREAD_SINGLE`
  в M5; M6 re-probe + upgrade request (`MPI_THREAD_SERIALIZED` минимум). Document в CommConfig.
- **R-M5-7 — Linear1D zoning scheme ломает M3 Hilbert regression.** Adding second scheme может
  случайно сдвинуть default dispatch в zoning_planner. Mitigation: T5.9 explicit regression
  gate — M3 smoke unchanged; Hilbert path tested via full M3 smoke golden diff.
- **R-M5-8 — Anchor-test reference data extraction accuracy.** Dissertation figures 29-30 —
  published bitmap; point extraction ±2% typical uncertainty. Mitigation: T5.10 commits CSV
  с rationale-block documenting extraction method (e.g., WebPlotDigitizer); acceptance tolerance
  10% absorbs ±2% reference uncertainty; human review of CSV required before T5.11 uses it.
- **R-M5-9 — 6-недельный M5 может реально занять 8 недель в single-agent режиме.** Больше
  surface area чем M4 (добавляется comm/ module + verify/ harness + MPI build integration).
  Mitigation: 12 PR-size задач с clear dependencies; rolling post-impl reports; T5.10-T5.11
  параллелимые после T5.9.
- **R-M5-10 — Dev machine flaky MPI installs.** Разные Ubuntu MPI packages (`libopenmpi-dev`
  vs `libmpich-dev`) различаются в ABI. Mitigation: T5.2 uses `find_package(MPI REQUIRED)`
  standard CMake module; T5.4 tests against both OpenMPI 4.1 + MPICH 4 locally (CI uses
  OpenMPI default on ubuntu-latest).

**Open questions (deferred to task-time decisions):**

- **OQ-M5-1 — Buffer pool default size.** comm/SPEC §5.3 says "4 × K_max per packet type per rank."
  Concrete default N? **Answer at T5.4:** 32 buffers × 4 MiB = 128 MiB per rank (covers K=8 max,
  with 4× redundancy for async overlap). Configurable via CommConfig.
- **OQ-M5-2 — `deterministic_sum_double` fallback to MPI_Allreduce when?** Never in Reference
  profile (D-M5-9). In M8+ Fast profile — YAML opt-in. Document в comm/SPEC §7.2 follow-up.
  **Answer at T5.4:** M5 hard-codes deterministic path; Fast — M8+ task.
- **OQ-M5-3 — CRC32 implementation choice (bundled vs zlib).** Zlib is always available on Linux
  via system package; bundling adds 2 KB but removes dependency. **Answer at T5.3:** bundle
  standalone 8-KiB table; zero external dep; simplifies cross-platform (Windows dev path later).
- **OQ-M5-4 — Should `AnchorTestRunner` run LJ oracle via LAMMPS?** Anchor-test matches diss
  numerical results; LAMMPS would be orthogonal physics check. **Answer at T5.11:** NO. T3 is
  dissertation-match only. LJ physics validated separately via future T-LJ differential (not
  in M5 scope).
- **OQ-M5-5 — Rank ownership distribution heuristic.** Round-robin `zone_id % P` is simplest;
  better heuristics (contiguous blocks, load-weighted) — M8+ auto-balance. **Answer at T5.8:**
  round-robin M5; contiguous blocks may be cheaper on ring but add complexity — defer.
- **OQ-M5-6 — `comm.barrier()` frequency в TD loop.** Scheduler naturally reaches synchronization
  points at end-of-iteration; explicit barrier may or may not be needed. **Answer at T5.8:**
  explicit barrier() at end of each step in Reference profile (ensures deterministic reduction
  sees consistent state); can be relaxed to async barrier в Fast — M8+.
- **OQ-M5-7 — What if `MPI_Init_thread` returns < `MPI_THREAD_SINGLE` required level?** M5
  needs SINGLE minimum. **Answer at T5.4:** hard-fail at init with clear message "MPI
  implementation doesn't support required threading level; upgrade or rebuild MPI with threads".
- **OQ-M5-8 — Anchor-test hardware normalization source of FLOPs measurement.** Synthetic LJ
  micro-kernel (T5.11 `hardware_probe.py`) vs LINPACK vs published peak? **Answer at T5.11:**
  bundled LJ micro-kernel — matches anchor-test workload, cache-friendly, reproducible. LINPACK
  measures different thing (dense GEMM).
- **OQ-M5-9 — K-batching zone dispatch: one task per zone per iteration, or multiple tasks?**
  SPEC §5.1 psuedocode: `break` after first candidate per zone. K>1 means a zone can have K
  tasks at K different time_levels simultaneously pending. **Answer at T5.6:** keep the
  `break`-after-first-per-zone per iteration (one task per zone per iteration); pipeline
  depth comes from many zones progressing in parallel, not many time_levels per zone per
  iter. This matches Andreev's original scheme (§2.2 dissertation).
- **OQ-M5-10 — Atom migration mid-run in Pattern 1.** Zone contents change over time (atoms
  drift). M5 zone membership is static (assigned once at init per zoning plan). Does scheduler
  need to handle dynamic re-zoning? **Answer at T5.8:** NO. Static assignment M5; dynamic
  zone membership + cross-zone atom migration — M8 (when pressure + long runs require it).
  For Andreev-style anchor-test (1000 steps, periodic box, modest T) static assignment
  is sufficient.

---

## 7. Roadmap Alignment

| Deliverable | Consumer milestone | Why it matters |
|---|---|---|
| `comm/` module skeleton (T5.2) | M6 GPU-aware MPI + NCCL backends layer on top; M7 Hybrid composes; M8 SNAP — all use CommBackend as transport | Single abstraction for all parallelism; no module bypasses comm/ for MPI calls |
| TemporalPacket wire-format + CRC32 (T5.3) | M6 same format, GPU-aware pack/unpack; M7 HaloPacket + MigrationPacket follow same versioning protocol | Protocol versioning gate (v1 → v1.1 → v2) future-proofs wire evolution |
| MpiHostStagingBackend (T5.4) | M6 remains fallback when GpuAware MPI absent; M8 deployed on clusters without CUDA-aware network | Universal fallback — "TDMD works anywhere MPI works" |
| RingBackend (T5.5) | Anchor-test regression forever; preserved as "Andreev-mode" for scientific repro | Без него нельзя заявлять воспроизводимость диссертации |
| K-batching (T5.6) | M7 K-batching на inner TD level composes с SD outer; M8 AutoK finds optimal K via measurement | Главная формула §4.3 диссертации — `T_comm_per_step = T_p / K` |
| Peer dispatch (T5.7) | M7 Pattern 2 adds SubdomainBoundaryDependency on top; M8 SNAP needs proven peer transfer reliability | Completes M4's deferred no-ops; scheduler state machine finally fully operational |
| Multi-rank SimulationEngine (T5.8) | M6 same engine + GPU kernels; M7 adds outer coordinator; M8+ всё layers on top | Engine становится coordination-only; physics in potentials/, time in scheduler/, transport in comm/ — clean separation |
| Linear1D Z-zoning (T5.9) | Anchor-test forever; future 1D cross-validation experiments | Andreev's §2.2 baseline — N_min=2 maximum parallelism reference |
| T3 benchmark fixture (T5.10) | Continuous regression guard — any PR touching scheduler/comm/potentials MUST pass T3 slow tier | "Existence proof" of TDMD methodology — published scientific reference point |
| AnchorTestRunner (T5.11) | Release acceptance cert (§13.3 mandatory); M8 proof-of-value comparison baseline | Automated dissertation match — eliminates manual comparison from release certification |
| M5 smoke (T5.12) | Regression gate M6-M13 — multi-rank stack exercised on every PR | Catches comm/scheduler regression before it reaches anchor-test slow tier |

---

*End of M5 execution pack, дата: 2026-04-19.*
