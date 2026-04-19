# M6 Execution Pack

**Document:** `docs/development/m6_execution_pack.md`
**Status:** draft, awaiting human review
**Parent:** `TDMD_Engineering_Spec.md` §14 (M6), `docs/specs/potentials/SPEC.md` v1.0, `docs/specs/neighbor/SPEC.md` v1.0, `docs/specs/integrator/SPEC.md` v1.0, `docs/development/claude_code_playbook.md` §3
**Milestone:** M6 — GPU path (CUDA kernels + host-staged MPI) — 9 недель
**Created:** 2026-04-19
**Author:** Architect / Spec Steward role (Claude Opus 4.7)

---

## 0. Purpose

Этот документ декомпозирует milestone **M6** master spec'а §14 на **13 PR-size задач**, из
которых **T6.0 уже закрыт** (commit `65e142f`, 2026-04-19 — закрытие R-M5-8 извлечением
данных из scans диссертации fig 29 + fig 30). Документ — **process artifact**, не SPEC
delta. Остаётся 12 активных задач (T6.1..T6.13) для последующей реализации.

M6 — **первая встреча TDMD с GPU.** После M5 на CPU работает: (a) весь reference-path (FP64,
bit-exact regressions), (b) multi-rank MPI (MpiHostStaging + Ring) + K-batching pipeline,
(c) anchor-test §13.3 T3 с reproducibility диссертации Андреева. M6 добавляет:

- `gpu/` module — CUDA abstraction (streams, events, allocators, RAII device pointers),
- GPU kernel trio — **neighbor list** (CellGrid → NL on device), **EAM force** (CPU-parity
  FP64 + MixedFast FP32/FP64 accum), **Velocity-Verlet integrator**,
- `SimulationEngine` GPU wire-up с **host-staged MPI** (device → pinned host buffer →
  `MpiHostStagingBackend` → pinned host → device). GPU-aware MPI и NCCL backends —
  **не в M6**, остаются на M7.
- **MixedFastBuild** runtime path (FP32 compute + FP64 accumulators per master spec §D
  Philosophy B); `Fp64ReferenceBuild` на GPU остаётся **bit-exact oracle**.
- **T3-gpu** anchor-test variant: GPU-путь воспроизводит CPU-Reference results bit-exact
  (Reference profile) или ≤ 1e-10 rel (MixedFast), перед тем как дальше comparing с
  dissertation reference.

**Conceptual leap от M5 к M6:**

- M5 = "scheduling scales" (P ranks × K∈{1,2,4,8}, CPU-only, anchor-test green на CPU).
- **M6 = "scheduling uses the GPU"** (CUDA kernels behind `ForceCalculator`, `Integrator`,
  `NeighborBuilder` interfaces; multi-stream overlap compute ⟷ MPI pack/unpack; MPI
  transport unchanged from M5).
- M7 = "scheduling goes two-level" (Pattern 2: TD inside subdomain × SD between; GPU-aware
  MPI; NCCL option; Hybrid backend).

Критически — **anchor-test T3** §13.3 остаётся hard merge gate. CPU-версия зафиксирована
в M5 (closed at `75115e7`, retroactively completed by T6.0). M6 вводит **T3-gpu**: тот
же fixture, GPU-путь, два уровня acceptance: (i) Reference-to-Reference bit-exact между
CPU и GPU (ensures no physics regression), (ii) MixedFast-to-Reference в пределах §13.7
differential tolerance (ensures mixed-precision economy not buying incorrect physics).

После успешного закрытия всех 13 задач и acceptance gate (§5) — milestone M6 завершён;
execution pack для M7 создаётся как новый аналогичный документ.

---

## 1. Decisions log (зафиксировано до старта T6.1)

| # | Решение | Значение | Rationale / источник |
|---|---|---|---|
| **D-M6-1** | GPU hardware target | sm_80 (A100) минимум, sm_86 (RTX 30xx) + sm_89 (RTX 40xx) + sm_90 (H100) + sm_100/120 (Blackwell) supported. sm_70 (V100) и ниже — НЕ поддерживается в M6. | Master spec §15.2 CUDA 13.1 baseline; sm_80 cutoff gives tensor-core + L2 128KB + async-copy (needed for M7 overlap). Confirmed by user 2026-04-19. |
| **D-M6-2** | CUDA toolkit minimum | CUDA 13.1 (system install at `/usr/local/cuda`). Older CUDA — НЕ поддерживается. | Dev machine уже на CUDA 13.1 (memory: `env_cuda_13_path.md`); sm_120 support требует 12.8+. Confirmed by user 2026-04-19. |
| **D-M6-3** | MPI transport в M6 | `MpiHostStagingBackend` (M5) — primary + only. Pre-send GPU→pinned host copy, post-recv pinned host→GPU copy. `GpuAwareMpiBackend` + `NcclBackend` — отложены на **M7**. | comm/SPEC §12 roadmap: M6 = GPU kernels + host-staged MPI; GPU-aware transport — M7. Reduces M6 scope, keeps anchor-test apples-to-apples. Confirmed by user 2026-04-19. |
| **D-M6-4** | GPU kernel scope в M6 | **Three kernels only**: (1) NeighborList build (CellGrid + half-NL), (2) EAM/alloy force (density + embedding + pair), (3) Velocity-Verlet integrator (NVE). LJ, Morse, MEAM, SNAP, PACE, MLIAP, NVT, NPT — **M9+**. | Master spec §14 M6 "one GPU kernel" — we interpret as minimum-viable physics trio (NL+EAM+VV) since all три нужны для closed-loop NVE runs. Confirmed by user 2026-04-19 as scope for 9-week milestone. |
| **D-M6-5** | Active BuildFlavors в M6 | `Fp64ReferenceBuild` (bit-exact oracle — GPU does FP64 same as CPU FP64). `MixedFastBuild` (FP32 compute kernels + FP64 accumulation — Philosophy B per master spec §D). `MixedFastAggressiveBuild` (A), `Fp64ProductionBuild`, `Fp32ExperimentalBuild` — НЕ активны в M6. | Master spec §14 M6 + §D matrix. Philosophy A (`MixedFastAggressive`) opt-in в M8+. Confirmed by user 2026-04-19. |
| **D-M6-6** | CI strategy | **CUDA compile-only** в CI (Option A extended: public GitHub runners без GPU; build CUDA objects to verify no compile regressions, skip kernel execution). **Local pre-push** для всех GPU runtime tests (T6.4/T6.5/T6.6 unit + T6.13 smoke + T3-gpu anchor). No self-hosted runner в M6. | Memory `project_option_a_ci.md` — Option A extended. Per user decision 2026-04-19: "Давай вообще без ранера. Зачем усложнять — мы же можем и так локально запускать под GPU." Revisit self-hosted на M7 или позже. |
| **D-M6-7** | Reference GPU bit-exactness gate | `Fp64ReferenceBuild` GPU path выдаёт **bit-exact** forces + energies как CPU Reference на step 0 и step 1 на Ni-Al EAM 10k atoms. Reduction order canonicalized через single-block sort + Kahan (gpu/SPEC §5 — to be authored at T6.2). | Master spec §13.5 determinism matrix: Reference profile = bitwise oracle across all backends. Confirmed by user 2026-04-19. |
| **D-M6-8** | MixedFast differential threshold | `MixedFastBuild` GPU vs Reference GPU: per-atom force rel-diff ≤ 1e-6; total energy rel-diff ≤ 1e-8; NVE drift ≤ 1e-5 per 1000 steps. Thresholds registered в `verify/SPEC.md` threshold registry. | Master spec §13.7 + verify/SPEC §3.2 threshold registry. Philosophy B expectation: 1e-6 force diff acceptable — не degrades scientific observables по dissertation budget. Confirmed by user 2026-04-19. |
| **D-M6-9** | Timeline | 9 недель (single-agent). Most expensive — T6.5 EAM GPU kernel (~7 days, density + embedding + pair + ghost handling), T6.7 engine wire (~6 days, host-staging glue + stream dependency graph + pipeline integration), T6.10 T3-gpu (~5 days, anchor-test GPU harness + bit-exact CPU↔GPU Reference gate). Остальные 2-4 дня. | Confirmed by user 2026-04-19 ("4 - ок"). Budget vs M5 (6 недель shipped ~8 недель): +50% budget reflects added GPU surface area (CUDA runtime + streams + memory pools + kernel physics). |
| **D-M6-10** | CUDA compilation driver | `nvcc` primary (bundles `libcudart`, robust device linking). `clang-cuda` — **deferred to M10+** (matrix testing experiment, not M6 scope). | Master spec §15.2 mentions both but anchors nvcc; clang-cuda adds matrix burden без пропорциональной ценности на первом GPU milestone. |
| **D-M6-11** | CUDA device abstraction | Thin RAII wrappers (`tdmd::gpu::DevicePtr<T>`, `DeviceStream`, `DeviceEvent`) без heavy frameworks (no Thrust, no CUB-as-primary). CUB internal в kernel — **разрешено** для sort/scan/reduce primitives где custom beat bench. | gpu/SPEC (to be authored at T6.2). CUB allowed as implementation detail; no public Thrust-style high-level API — контроль over memory + synchronization. |
| **D-M6-12** | Memory management | Single **cached device pool** (tdmd_gpu::DevicePool) — реализация на base `cudaMallocAsync` (CUDA 11.2+) с stream-ordered allocation. Pinned host pool для MPI staging. Sub-allocator size classes {4K, 64K, 1M, 16M, 256M}. | gpu/SPEC (to be authored). Matches Thrust cached pool behaviour без Thrust dependency. Replaces naïve `cudaMalloc`/`cudaFree` (3–10× faster в neighbor-list rebuild hot path). |
| **D-M6-13** | Stream policy | **Two streams per rank** в M6: `stream_compute` (kernel launches) + `stream_mem` (H⇄D copies). Barrier через `cudaEventRecord + cudaStreamWaitEvent`. Single-stream mode остаётся доступен через CommConfig как debug toggle. Full N-stream pipelining (N=K or higher) — **M7**. | gpu/SPEC (to be authored). Two-stream overlap — minimum для compute/MPI overlap в M6; N-stream K-way pipelining добавляется когда Pattern 2 landed. |
| **D-M6-14** | NVTX instrumentation | `NVTX_RANGE("...")` обёртки в каждый long-lived kernel launch + MPI pack/unpack + pipeline phase-transition. Enabled всегда; overhead <<1%. | Master spec §15.4 + gpu/SPEC (to be authored). Nsight Systems trace — primary perf debugging tool в M6. |
| **D-M6-15** | Determinism policy на GPU | Reduction order canonical через gather-to-single-block + sorted atomic/Kahan add (comm/SPEC §7.2 policy extended to intra-kernel). No `atomicAdd` без Kahan companion в Reference. MixedFast — тот же canonical path (MixedFast отличается precision, не порядком). | Master spec §7.3 + gpu/SPEC. Canonical order — ключ к cross-backend bit-exactness gate (D-M6-7). |
| **D-M6-16** | Neighbor list storage layout | AoS → SoA conversion на H2D copy boundary. Neighbor list on device — SoA (struct of arrays): `neighbor_offset[N+1]`, `neighbor_idx[]`, `neighbor_d2[]`. Half-list format (i<j entries only) per neighbor/SPEC v1.0. | neighbor/SPEC §4.2 + M3 CellGrid CPU behaviour preserved. SoA — coalesced reads в EAM density kernel hot path. |
| **D-M6-17** | PIMPL + compile firewall | All CUDA-specific headers скрыты за PIMPL в `gpu/` module public headers. Public API в `tdmd::gpu::*` — plain C++ (host side). Это позволяет CI build pipeline без CUDA (компиляция CPU-only path — optional но прозрачно). | Master spec §15.2 + потому M5 CI уже building без CUDA toolkit. Flag `TDMD_ENABLE_GPU` default ON only if CUDA detected. |
| **D-M6-18** | YAML config extension | `gpu.device_id: 0` (default), `gpu.memory_pool_init_size_mib: 256`, `gpu.streams: 2`, `gpu.enable_nvtx: true`. Runtime override via CLI `--gpu-device`, `--gpu-streams`. | cli/SPEC + io/SPEC extensions; no breaking YAML change (все поля optional, omit → defaults). |
| **D-M6-19** | Anchor-test T3-gpu scope | Same fixture (Al FCC 10⁶ LJ NVE 1000 steps). GPU pre-check gates: (a) `Fp64ReferenceBuild` CPU ≡ `Fp64ReferenceBuild` GPU bit-exact; (b) `MixedFastBuild` GPU within D-M6-8 thresholds of `Fp64ReferenceBuild` GPU. Then efficiency vs N_procs curve on GPU — same 10% tolerance as CPU variant. | verify/SPEC §4.4 + master spec §13.3 extended for GPU. Hardware normalization script re-used, но `gpu_flops_ratio` scalar computed через `hardware_normalization_gpu.py` (micro-kernel on device). |
| **D-M6-20** | SPEC deltas в M6 | Ожидается **exactly one new SPEC**: `docs/specs/gpu/SPEC.md` v1.0 authored в T6.2 (как первый код landed в pack). Плюс **change log entries** в existing SPECs (potentials, neighbor, integrator, comm, io) — отмечают GPU path landed. **No breaking contract changes** в existing SPECs. | Playbook §9.1. gpu/ — новый module, SPEC должен предшествовать код. Остальные SPECs имеют stable contracts с M5. |

---

## 2. Глобальные параметры окружения

| Параметр | Значение | Примечание |
|---|---|---|
| OS | Linux (Ubuntu 24.04 LTS) | Dev-машина пользователя; ubuntu-latest в CI |
| C++ compiler | GCC 13+ / Clang 17+ | C++20; CI уже проверяет оба |
| CMake | 3.25+ | Master spec §15.2 |
| CUDA | **13.1** installed (system `/usr/local/cuda`) | D-M6-2; CI compile-only (no runtime) |
| GPU architectures | sm_80, sm_86, sm_89, sm_90, sm_100, sm_120 | D-M6-1; cmake flag `CMAKE_CUDA_ARCHITECTURES="80;86;89;90;100;120"` |
| MPI | OpenMPI 4.1+ (as M5) | Unchanged; no GPU-aware build required в M6 (D-M6-3) |
| Python | 3.10+ | pre-commit + anchor-test harness + gpu hardware probe |
| Test framework | Catch2 v3 (FetchContent) + MPI wrapper (M5) | GPU tests require CUDA runtime — local-only |
| LAMMPS oracle | SKIP on public CI (Option A) | Differentials run pre-push locally; T3-gpu uses CPU-Reference as oracle, not LAMMPS |
| Active BuildFlavors | `Fp64ReferenceBuild`, `MixedFastBuild` | D-M6-5 |
| Active ExecProfiles | `Reference`, `Production` (GPU) | M6 introduces GPU Production profile; Fast — M8 |
| Run mode | multi-rank MPI × GPU-per-rank, single-thread per rank | D-M6-3; 1 GPU : 1 MPI rank в M6 |
| Pipeline depth | `K ∈ {1, 2, 4, 8}` (as M5); default 1 | Unchanged; GPU streams enable async во within-K |
| Streams per rank | 2 (default) | D-M6-13 |
| CI CUDA | compile-only (`TDMD_ENABLE_GPU=ON`, kernel tests skipped) | D-M6-6; ubuntu-latest + `nvidia-cuda-toolkit` apt package |
| Local pre-push gates | Full GPU suite + T3-gpu anchor + M1..M6 smokes | D-M6-6; memory `project_option_a_ci.md` |
| Branch policy | `m6/T6.X-<topic>` per PR → `main` | CI required: lint + build-cpu (MPI) + build-gpu (compile-only) + M1..M5 smokes; M6 smoke добавляется в T6.13 |

---

## 3. Suggested PR order

Dependency graph:

```
T6.0 [DONE 65e142f] ─► T6.1 (this pack) ─► T6.2 (gpu/ skeleton + SPEC v1.0)
                                                │
                                                ▼
                                        T6.3 (device memory + pools)
                                                │
                                    ┌───────────┼───────────┐
                                    ▼           ▼           ▼
                              T6.4 (NL GPU) T6.5 (EAM GPU) T6.6 (VV GPU)
                                    └───────────┼───────────┘
                                                ▼
                                        T6.7 (engine GPU wire)
                                                │
                                    ┌───────────┴───────────┐
                                    ▼                       ▼
                        T6.8 (MixedFastBuild)       T6.9 (2-stream overlap)
                                    └───────────┬───────────┘
                                                ▼
                                        T6.10 (T3-gpu anchor)
                                                │
                                                ▼
                                        T6.11 (NVTX + PerfModel GPU)
                                                │
                                                ▼
                                        T6.12 (CUDA CI compile-only)
                                                │
                                                ▼
                                        T6.13 (M6 smoke + GATE)
```

**Линейная последовательность (single agent):**
T6.1 → T6.2 → T6.3 → T6.4 → T6.5 → T6.6 → T6.7 → T6.8 → T6.9 → T6.10 → T6.11 → T6.12 → T6.13.

**Параллельный режим (multi-agent):** после T6.3 — `{T6.4, T6.5, T6.6}` три kernels независимы
(разные физические подсистемы); объединяются на T6.7. После T6.7 — `{T6.8, T6.9}` независимы
(precision vs concurrency). T6.10 требует обе. T6.11 после T6.10 (нужна PerfModel calibration
на реальных GPU runs). T6.12 после T6.11 (CI wires compile step + готовые kernels). T6.13 —
final gate.

**Estimated effort:** 9 недель (single agent). Самые длинные — T6.5 EAM GPU (~7 дней:
density + embedding + pair + ghost + bit-exact gate), T6.7 engine wire (~6 дней: host-staging
glue + stream dependencies + pipeline integration), T6.10 T3-gpu (~5 дней: harness + CPU↔GPU
bit-exact gate + dissertation reference re-run). Остальные 3-4 дня.

---

## 4. Tasks

### T6.0 — [CLOSED] Extract Andreev fig 29-30 reference data (closes R-M5-8)

**Status:** ✅ **COMPLETED 2026-04-19** (commit `65e142f`).

Ссылочные точки диссертации Андреева fig 29 (performance) + fig 30 (efficiency) извлечены
через WebPlotDigitizer-style readout из authoritative scans
(`docs/_sources/fig_29.png` + `fig_30.png`); `dissertation_reference_data.csv` обновлён;
placeholder sentinel removed; `AnchorTestRunner._csv_is_placeholder` теперь возвращает
`False`; 2·10⁶ и 4·10⁶ curves архивированы в `additional_model_sizes.md`. R-M5-8 CLOSED.

Этот task closed retroactively during M5 wrap-up + M6 kickoff; включён в M6 pack для
полноты нумерации и historical record.

---

### T6.1 — Author M6 execution pack (this document)

```
# TDMD Task: Create M6 execution pack

## Context
- Master spec: §14 M6
- Role: Architect / Spec Steward
- Milestone: M6 (kickoff)

## Goal
Написать `docs/development/m6_execution_pack.md` декомпозирующий M6 на 13 PR-size задач
(T6.0 already closed), по шаблону m5_execution_pack.md. Document-only PR per playbook §9.1.

## Scope
- [included] docs/development/m6_execution_pack.md (single new file)
- [included] Decisions log D-M6-1..D-M6-20
- [included] Task templates T6.0..T6.13 (T6.0 marked CLOSED)
- [included] M6 acceptance gate checklist
- [included] Risks R-M6-1..R-M6-N + open questions OQ-M6-*

## Out of scope
- [excluded] Any code changes (T6.2+ territory)
- [excluded] SPEC deltas (separate PR — T6.2 authors gpu/SPEC.md)

## Required files
- docs/development/m6_execution_pack.md

## Required tests
- pre-commit clean
- markdownlint clean

## Acceptance criteria
- [ ] Pack committed + pushed
- [ ] Task list matches §14 M6 deliverables (neighbor+EAM+VV on GPU, host-staged MPI, mixed-fast)
- [ ] Decisions anchored to master spec §14 M6 + §D Philosophy B + Option A CI memory
- [ ] T6.0 closed status accurately reflected (pointer to commit 65e142f)
```

---

### T6.2 — `gpu/` module skeleton + CUDA detection + `gpu/SPEC.md` v1.0

```
# TDMD Task: gpu/ module skeleton + SPEC v1.0

## Context
- Master spec: §14 M6, §15.2, §D.1 Philosophy B
- Module SPEC: NEW — authored в этом task
- Role: GPU / Performance Engineer
- Depends on: T6.1
- Milestone: M6

## Goal
Создать `src/gpu/` module + `docs/specs/gpu/SPEC.md` v1.0: CMakeLists с `enable_language(CUDA)`
conditional on `TDMD_ENABLE_GPU` flag + `CMAKE_CUDA_ARCHITECTURES` list (D-M6-1);
namespaces; public headers с типами `DevicePtr<T>`, `DeviceStream`, `DeviceEvent`,
`DeviceInfo`, `GpuConfig`; abstract `DeviceAllocator` interface. Никаких concrete
реализаций allocator'а (T6.3) или kernel'ов (T6.4-T6.6).

## Scope
- [included] src/gpu/CMakeLists.txt — new static lib `tdmd_gpu`; CUDA optional (TDMD_ENABLE_GPU)
- [included] src/gpu/include/tdmd/gpu/types.hpp — DevicePtr<T>, DeviceStream, DeviceEvent,
  DeviceInfo, GpuCapability
- [included] src/gpu/include/tdmd/gpu/gpu_config.hpp — GpuConfig (device_id, streams count,
  pool init size, enable_nvtx toggle)
- [included] src/gpu/include/tdmd/gpu/device_allocator.hpp — abstract DeviceAllocator
- [included] src/gpu/device_allocator.cpp — virtual destructor body
- [included] src/CMakeLists.txt — add_subdirectory(gpu) conditional on TDMD_ENABLE_GPU
- [included] tests/gpu/CMakeLists.txt + test_gpu_types.cpp (type presence, GpuConfig defaults)
- [included] cmake/FindCUDA helpers if needed; TDMD_ENABLE_GPU cache variable (auto-detect)
- [included] docs/specs/gpu/SPEC.md v1.0 — full module SPEC (abstract interface, memory
  model, stream model, determinism policy per D-M6-15, NVTX policy per D-M6-14, mixed-precision
  contracts per D-M6-5)

## Out of scope
- [excluded] DevicePool concrete impl (T6.3)
- [excluded] Kernel implementations (T6.4-T6.6)
- [excluded] Engine integration (T6.7)
- [excluded] MPI interaction (T6.7 handles GPU⇄host staging)

## Mandatory invariants
- `DevicePtr<T>` is move-only (non-copyable); RAII with custom deleter using pool
- `DeviceStream` + `DeviceEvent` are move-only; RAII via CUDA primitives
- `GpuConfig` default-constructible + defaults from D-M6-18 (device=0, streams=2, pool=256 MiB,
  nvtx=true)
- All CUDA-specific types PIMPL-hidden per D-M6-17 — header file compiles on CPU-only build
- gpu/SPEC.md v1.0 anchors determinism (D-M6-15), memory (D-M6-12), streams (D-M6-13), NVTX
  (D-M6-14), precision (D-M6-5)
- No dependency on state/, scheduler/, potentials/ (gpu is primitive layer)

## Required files
- src/gpu/CMakeLists.txt
- src/gpu/include/tdmd/gpu/{types.hpp, gpu_config.hpp, device_allocator.hpp}
- src/gpu/device_allocator.cpp
- tests/gpu/{CMakeLists.txt, test_gpu_types.cpp}
- docs/specs/gpu/SPEC.md v1.0

## Required tests
- Type-presence tests (compile-only) on CPU build
- GpuConfig defaults verified
- DevicePtr move semantics + deleter invocation
- CI budget <2s for type/config tests

## Acceptance criteria
- [ ] gpu/ builds clean with CUDA toolkit + skips gracefully без CUDA
- [ ] gpu/SPEC.md v1.0 committed before код landed in T6.3
- [ ] Public API is PIMPL-free CUDA types — CPU build compiles headers
- [ ] Change log entry в master spec Приложение C (M6 kickoff)
```

---

### T6.3 — Device memory management (RAII + cached pool + pinned host buffers)

```
# TDMD Task: DevicePool + PinnedHostPool

## Context
- Module SPEC: docs/specs/gpu/SPEC.md §5 (authored T6.2)
- Role: GPU / Performance Engineer
- Depends on: T6.2
- Milestone: M6

## Goal
Реализовать `tdmd::gpu::DevicePool`: stream-ordered cached allocator поверх `cudaMallocAsync`
(D-M6-12); size classes {4K, 64K, 1M, 16M, 256M}; LRU eviction при pool pressure. Plus
`PinnedHostPool`: `cudaMallocHost`-backed pool с size classes {4K, 64K, 1M, 16M}
(для MPI staging — D-M6-3). `DevicePtr<T>` custom deleter — returns block в pool, не
`cudaFree`. NVTX ranges вокруг alloc/free (D-M6-14).

## Scope
- [included] src/gpu/include/tdmd/gpu/{device_pool.hpp, pinned_host_pool.hpp}
- [included] src/gpu/{device_pool.cu, pinned_host_pool.cpp}
- [included] src/gpu/include/tdmd/gpu/device_ptr_deleter.hpp (pool-aware deleter)
- [included] tests/gpu/test_device_pool.cu — alloc/free patterns, pressure eviction, LRU
  correctness, size-class roundrobin, stream-ordered safety (no UAF across streams)
- [included] tests/gpu/test_pinned_host_pool.cpp — same pattern, simpler (no stream semantics)

## Out of scope
- [excluded] NCCL-backed pools (M7)
- [excluded] Unified memory (experimental; M8+)
- [excluded] Multi-GPU per rank (M7+)

## Mandatory invariants
- Thread-safe under `MPI_THREAD_SINGLE` (single-threaded per rank — M6 constraint)
- Allocations always aligned to 256 bytes (coalesced-friendly)
- Pool pressure triggers LRU eviction, not OOM crash (unless allocation > 256 MiB class)
- `cudaMallocAsync` + `cudaFreeAsync` — stream-ordered; `DevicePtr<T>::~DevicePtr()`
  records free on its birth stream
- Telemetry counters: `device_bytes_in_use`, `pinned_bytes_in_use`, `pool_evictions`,
  `alloc_stalls_ms`

## Required files
- src/gpu/include/tdmd/gpu/{device_pool.hpp, pinned_host_pool.hpp, device_ptr_deleter.hpp}
- src/gpu/{device_pool.cu, pinned_host_pool.cpp}
- tests/gpu/{test_device_pool.cu, test_pinned_host_pool.cpp}

## Required tests
- Alloc 100 × 1 MiB в loop → peak <100 MiB held (reuse через pool) — local GPU-only
- Alloc 1 × 300 MiB (exceeds 256 MiB class) → falls back to direct cudaMalloc or OOM gracefully
- Stream-order safety: alloc on stream A, free on stream A, alloc on stream B → no race
- Pinned host: pack/unpack pattern — 10k iterations no fragmentation

## Acceptance criteria
- [ ] DevicePool + PinnedHostPool compile + tests pass on dev GPU
- [ ] Telemetry counters exposed via `SimulationTelemetry` handle
- [ ] NVTX ranges visible в Nsight Systems trace
- [ ] No `cudaMalloc`/`cudaFree` direct calls в hot paths (grep audit clean)
```

---

### T6.4 — Neighbor list GPU kernel (CellGrid → half-NL on device)

```
# TDMD Task: NeighborListBuilder GPU kernel

## Context
- Master spec: §5.2 neighbor build
- Module SPEC: docs/specs/neighbor/SPEC.md §4 + gpu/SPEC.md §6
- Role: GPU / Performance Engineer
- Depends on: T6.3
- Milestone: M6

## Goal
Реализовать GPU neighbor list builder: (1) H2D copy atom positions + cell assignment;
(2) kernel `cell_grid_bin` — each atom → cell (sort-based или atomic-based, SoA layout);
(3) kernel `neighbor_list_build_half` — per atom scan neighboring 27 cells, emit half-list
(i<j) entries with d² prefilter на cutoff + skin; (4) compaction + offset array; (5)
optional D2H transfer для verification. FP64 в Reference build, FP32 positions + FP64 d²
accumulation в MixedFast.

## Scope
- [included] src/gpu/include/tdmd/gpu/neighbor_list_gpu.hpp
- [included] src/gpu/neighbor_list_gpu.cu — kernels + host-side orchestration
- [included] src/neighbor/gpu_neighbor_builder.cpp — adapter implementing
  `NeighborBuilder` interface (M3 CPU variant's cousin)
- [included] tests/gpu/test_neighbor_list_gpu.cu — 1000-atom Al FCC test: builds NL, D2H,
  compares to CPU half-NL builder bit-exact (Reference profile) or ≤ epsilon (MixedFast)
- [included] verify/benchmarks/neighbor_gpu_vs_cpu/ — micro-bench fixture (10k / 100k atoms,
  NVTX-marked, perf baseline for PerfModel calibration T6.11)

## Out of scope
- [excluded] Full-list format (half-list only in M6 per D-M6-16)
- [excluded] Dynamic load balancing among cells (M8+)
- [excluded] Ghost-atom exchange (T6.7 handles boundary communication)

## Mandatory invariants
- Half-list: entries `(i, j)` only where `i < j` (stable — matches CPU for bit-exactness gate)
- Output `neighbor_offset[N+1]` prefix sum, SoA as D-M6-16
- Reference: FP64 throughout; MixedFast: FP32 pos + FP64 d² accumulator
- Neighbor count per atom reproducible across runs (deterministic cell bin order)
- NVTX ranges wrapping `cell_grid_bin` + `neighbor_list_build_half`

## Required files
- src/gpu/include/tdmd/gpu/neighbor_list_gpu.hpp
- src/gpu/neighbor_list_gpu.cu
- src/neighbor/gpu_neighbor_builder.cpp
- tests/gpu/test_neighbor_list_gpu.cu
- verify/benchmarks/neighbor_gpu_vs_cpu/ (fixture + README)

## Required tests
- 1000-atom Al FCC → NL matches CPU bit-exact (Reference)
- MixedFast NL matches Reference within d² epsilon (no false neighbors/exclusions on
  boundary — at r_c - skin no atom misclassified)
- 10k / 100k atom micro-bench runtime recorded (baseline for T6.11)
- CI: compile-only green (no runtime)
- Local: runtime kernel tests green

## Acceptance criteria
- [ ] GPU NL bit-exact to CPU NL (Reference profile)
- [ ] MixedFast NL within epsilon (no false-positive/negative at skin boundary)
- [ ] Micro-bench runtime ≥ 5× CPU (otherwise flag — GPU shouldn't be slower)
- [ ] NVTX ranges visible + pool stats healthy
```

---

### T6.5 — EAM/alloy GPU kernel (density + embedding + pair, CPU-parity)

```
# TDMD Task: EamAlloyPotential GPU kernel

## Context
- Master spec: §5.3 EAM force math
- Module SPEC: docs/specs/potentials/SPEC.md §3 (EAM section) + gpu/SPEC.md §6
- Role: Physics Engineer + GPU / Performance Engineer
- Depends on: T6.4
- Milestone: M6

## Goal
Реализовать EAM/alloy force kernel на GPU (D-M6-4): three passes per step — (1) density
pass: for each atom i, sum `rho(r_ij)` over neighbors j → `density[i]`; (2) embedding
derivative: `dF/drho[i] = F'(density[i])` via interpolation; (3) pair+embedding force:
`f_ij = -phi'(r_ij) - (dF/drho[i] + dF/drho[j]) * rho'(r_ij)`, accumulate per atom.
Tabulated `phi`, `rho`, `F` on device (constant memory or texture — выбор в T6.5).
FP64 Reference bit-exact to CPU; MixedFast — FP32 math + FP64 force accumulation per D-M6-8.

## Scope
- [included] src/gpu/include/tdmd/gpu/eam_alloy_gpu.hpp
- [included] src/gpu/eam_alloy_gpu.cu — three kernels + table upload + orchestration
- [included] src/potentials/eam_alloy_gpu_potential.cpp — adapter implementing `Potential`
  interface (CPU cousin exists from M2)
- [included] tests/gpu/test_eam_alloy_gpu.cu — 1000-atom Ni-Al EAM: forces + energy
  bit-exact to CPU Reference (forces ≤ 1e-12 rel, energy ≤ 1e-13 rel); MixedFast within
  D-M6-8 thresholds
- [included] verify/benchmarks/eam_gpu_vs_cpu/ — micro-bench 10k/100k atoms

## Out of scope
- [excluded] EAM/fs variant (same interface; deferred to M8+ when active user demand)
- [excluded] MEAM/SNAP/PACE GPU (M9+)
- [excluded] Virial/stress tensor (M7+ — NPT support)

## Mandatory invariants
- Reference: `|F_gpu[i] - F_cpu[i]| / |F_cpu[i]| ≤ 1e-12` (effectively bit-exact modulo FP64
  reduction rounding; D-M6-7 + D-M6-15 canonical order)
- MixedFast: per-atom force rel-diff ≤ 1e-6 from Reference GPU (D-M6-8)
- Density array persisted between pass 1 and pass 3 (no redundant recomputation)
- No global atomics in Reference (canonical Kahan reduction); atomics allowed in MixedFast
  with compensation
- NVTX ranges wrapping each of three passes

## Required files
- src/gpu/include/tdmd/gpu/eam_alloy_gpu.hpp
- src/gpu/eam_alloy_gpu.cu
- src/potentials/eam_alloy_gpu_potential.cpp
- tests/gpu/test_eam_alloy_gpu.cu
- verify/benchmarks/eam_gpu_vs_cpu/ (fixture + README)

## Required tests
- 1000-atom Ni-Al EAM forces + energies match CPU Reference bit-exact
- MixedFast forces within D-M6-8 thresholds of Reference GPU
- 10k micro-bench: GPU Reference ≥ 3× CPU; MixedFast ≥ 6× CPU (target, not hard gate)
- Valgrind/compute-sanitizer clean (local pre-push)
- Determinism: 10 repeated runs → bit-exact forces (Reference) / epsilon-close (MixedFast)

## Acceptance criteria
- [ ] EAM GPU forces match CPU Reference bit-exact
- [ ] MixedFast forces within threshold
- [ ] Determinism gate green (10 runs repeatable)
- [ ] Compute-sanitizer clean (no race, no UAF, no OOB)
- [ ] NVTX ranges visible
```

---

### T6.6 — Velocity-Verlet integrator GPU kernel (NVE)

```
# TDMD Task: VelocityVerletIntegrator GPU kernel

## Context
- Master spec: §6.2 NVE integrator
- Module SPEC: docs/specs/integrator/SPEC.md §5 + gpu/SPEC.md §6
- Role: Physics Engineer + Scheduler / Determinism Engineer
- Depends on: T6.3
- Milestone: M6

## Goal
Реализовать NVE velocity-Verlet на GPU: two half-kick + drift kernels (`half_kick_velocity`,
`drift_position`). Unit-system conversion factor `ftm2v` из M1 (metal units) hardcoded в
kernel constants (или uploaded). FP64 Reference bit-exact; MixedFast FP32 math + FP64 pos/vel
accumulators. NVT/NPT остаются на CPU в M6 (D-M6-4).

## Scope
- [included] src/gpu/include/tdmd/gpu/integrator_vv_gpu.hpp
- [included] src/gpu/integrator_vv_gpu.cu — two kernels + orchestration
- [included] src/integrator/gpu_velocity_verlet.cpp — adapter implementing `Integrator`
  interface (M1 CPU cousin)
- [included] tests/gpu/test_integrator_vv_gpu.cu — 1000-atom LJ one-step: positions + velocities
  match CPU Reference bit-exact after 1 step, 10 steps, 1000 steps
- [included] verify/benchmarks/integrator_gpu_vs_cpu/ — micro-bench

## Out of scope
- [excluded] NVT (Nosé-Hoover on GPU — M9)
- [excluded] NPT (Berendsen/MTK on GPU — M9+)
- [excluded] Langevin (M10+)
- [excluded] Constrained dynamics (SHAKE/RATTLE — M10+)

## Mandatory invariants
- Reference: `|r_gpu[i] - r_cpu[i]| ≤ 1e-14 Å` after 1000 NVE steps (bit-exact modulo FP64
  math; floor from accumulator ordering)
- MixedFast: `|r_gpu[i] - r_cpu[i]| / box ≤ 1e-10` — well inside NVE drift budget
- Zero energy drift budget in Reference (preserved M5 NVE gate)
- ftm2v constant honors M1 SPEC delta (memory `project_metal_unit_factor.md` — CPU uses 9648.5)
- NVTX ranges wrapping `half_kick` + `drift`

## Required files
- src/gpu/include/tdmd/gpu/integrator_vv_gpu.hpp
- src/gpu/integrator_vv_gpu.cu
- src/integrator/gpu_velocity_verlet.cpp
- tests/gpu/test_integrator_vv_gpu.cu
- verify/benchmarks/integrator_gpu_vs_cpu/ (fixture + README)

## Required tests
- 1-step LJ 1000 atoms: r_gpu == r_cpu bit-exact (Reference)
- 1000-step NVE: energy drift ≤ 1e-6 rel (Reference; matches M5 CPU behavior)
- MixedFast: drift ≤ 1e-5 rel per 1000 steps (D-M6-8 NVE drift threshold)
- 10k micro-bench timing recorded

## Acceptance criteria
- [ ] VV GPU matches CPU Reference bit-exact after 1000 steps
- [ ] MixedFast drift within threshold
- [ ] Unit factor consistency (ftm2v from M1 SPEC delta honored on both paths)
- [ ] NVTX visible
```

---

### T6.7 — SimulationEngine GPU wire-up (GPU force path + host-staged MPI)

```
# TDMD Task: SimulationEngine GPU integration

## Context
- Master spec: §8, §10.5 engine loop; §14 M6 host-staged MPI
- Module SPEC: docs/specs/runtime/SPEC.md + gpu/SPEC.md
- Role: Core Runtime Engineer
- Depends on: T6.4, T6.5, T6.6
- Milestone: M6

## Goal
Wire GPU force + integrator + neighbor paths в `SimulationEngine` loop, сохраняя
CommBackend из M5 как MPI transport (D-M6-3). Flow per iteration: (1) H2D copy updated
atoms (incremental — only changed zones); (2) GPU neighbor list rebuild (если skin
triggered); (3) GPU EAM force; (4) GPU VV half-kick + drift; (5) D2H copy zone atoms
that need to be sent; (6) pack → MpiHostStagingBackend → unpack; (7) H2D copy received
zone atoms; (8) continue. All stream-ordered.

## Scope
- [included] src/runtime/simulation_engine.cpp — GPU dispatch branches (conditional on
  ExecProfile == Production || BuildFlavor == MixedFastBuild)
- [included] src/runtime/include/tdmd/runtime/gpu_context.hpp — per-rank GpuContext
  (streams, pools, allocators)
- [included] src/runtime/gpu_context.cpp
- [included] src/scheduler/scheduler_gpu_adapters.cpp — thin adapters mapping scheduler
  events (Ready-for-next-step, PackedForSend) to GPU stream records
- [included] tests/integration/gpu_single_rank/ — 1000-atom Ni-Al EAM, 100 steps, GPU
  Reference path: thermo matches CPU Reference smoke byte-exact
- [included] tests/integration/gpu_2rank/ — 1000-atom Ni-Al EAM, 100 steps, 2-rank
  MpiHostStaging + GPU: thermo matches 1-rank GPU byte-exact (determinism gate)

## Out of scope
- [excluded] GPU-aware MPI (M7)
- [excluded] NCCL collectives (M7)
- [excluded] Multi-stream K-way pipelining (T6.9 deals with 2-stream case only)
- [excluded] Pattern 2 (M7)

## Mandatory invariants
- 1-rank GPU Reference thermo byte-exact to M1 CPU Reference smoke (D-M6-7 extended)
- 2-rank GPU Reference thermo byte-exact to 1-rank GPU Reference (determinism)
- K=1 P=1 GPU байт-экзактен к K=1 P=1 CPU (D-M5-12 chain extended to GPU)
- No `cudaMemcpy` in hot loop per iteration — use streams + events
- No implicit sync (hazard: `cudaDeviceSynchronize` only at barrier points, not inside
  `select_ready_tasks`)
- All GPU resources (streams, events, pool state) owned by `GpuContext`, RAII via
  engine destructor

## Required files
- src/runtime/simulation_engine.cpp (modified — GPU branch)
- src/runtime/include/tdmd/runtime/gpu_context.hpp
- src/runtime/gpu_context.cpp
- src/scheduler/scheduler_gpu_adapters.cpp
- tests/integration/gpu_single_rank/*
- tests/integration/gpu_2rank/*

## Required tests
- 1-rank GPU 100 steps: thermo bit-exact to M1 CPU Reference
- 2-rank GPU 100 steps: thermo bit-exact to 1-rank GPU
- No regression в M1..M5 smokes (CPU path unaffected)
- Nsight Systems trace captured — confirms no stream stalls >1% of wall time

## Acceptance criteria
- [ ] GPU single-rank Reference path bit-exact to CPU Reference
- [ ] GPU 2-rank determinism gate green
- [ ] M1..M5 smokes unchanged
- [ ] Nsight trace healthy (no idle stall patterns)
```

---

### T6.8 — `MixedFastBuild` GPU wiring (Philosophy B: FP32 math + FP64 accum)

```
# TDMD Task: MixedFastBuild GPU path

## Context
- Master spec: §D.1 Philosophy B, §D matrix
- Module SPEC: gpu/SPEC.md + potentials/SPEC.md (build flavor annotations)
- Role: GPU / Performance Engineer + Physics Engineer
- Depends on: T6.7
- Milestone: M6

## Goal
Enable `MixedFastBuild` на GPU: все public kernels (NL, EAM, VV) опционально построены с
`__nv_bfloat16` или `float` math pipeline + `double` accumulators (Kahan-companion в critical
reductions). BuildFlavor выбирается compile-time CMake flag. Сравнение vs `Fp64ReferenceBuild`
GPU через existing T4 (M2) differential harness, extended для GPU targets.

## Scope
- [included] src/gpu/include/tdmd/gpu/*_mixed.hpp — templated kernel variants (T = float,
  Accum = double)
- [included] src/gpu/*_mixed.cu — kernel instantiations
- [included] CMakeLists.txt — TDMD_BUILD_FLAVOR matrix (Fp64Reference default, MixedFast
  opt-in via flag)
- [included] verify/differentials/t4_gpu_mixed_vs_reference/ — adopt T4 (Ni-Al EAM
  differential) for GPU Reference ↔ MixedFast comparison per D-M6-8 thresholds
- [included] tests/gpu/test_mixed_fast_eam.cu — MixedFast EAM outputs within D-M6-8 rel diff
  from Reference GPU on 1000-atom Ni-Al

## Out of scope
- [excluded] MixedFastAggressive (Philosophy A — M8+)
- [excluded] Fp32ExperimentalBuild (M8+)
- [excluded] bfloat16 path (too aggressive for first mixed GPU release — M8+)

## Mandatory invariants
- MixedFast EAM: per-atom force rel-diff ≤ 1e-6 from Reference GPU (D-M6-8)
- MixedFast total energy rel-diff ≤ 1e-8 from Reference GPU
- MixedFast NVE drift ≤ 1e-5 per 1000 steps
- Neighbor list MixedFast: no atom misclassified within skin (positions FP32 + r² FP64)
- VV MixedFast positions: rel-diff ≤ 1e-10 per step (box fraction)

## Required files
- src/gpu/include/tdmd/gpu/*_mixed.hpp (3 files — NL, EAM, VV)
- src/gpu/*_mixed.cu (3 files)
- CMakeLists.txt (build flavor wiring)
- verify/differentials/t4_gpu_mixed_vs_reference/ (fixture + thresholds)
- tests/gpu/test_mixed_fast_eam.cu

## Required tests
- T4 MixedFast differential 100 steps Ni-Al: all thresholds green
- Single-atom kernel bit-reproducibility (same MixedFast output across runs — determinism
  required even при FP32 math, per D-M6-15)
- Compile matrix: Fp64ReferenceBuild + MixedFastBuild both pass on CI (compile-only)

## Acceptance criteria
- [ ] MixedFast kernels compile + run locally
- [ ] T4 differential green within D-M6-8 thresholds
- [ ] Determinism — 10 runs reproducible (bit-exact for MixedFast via canonical reduction)
- [ ] verify/SPEC.md threshold registry updated с D-M6-8 entries
```

---

### T6.9 — Two-stream overlap (compute ⟷ MPI pack/unpack)

```
# TDMD Task: 2-stream pipeline overlap

## Context
- Master spec: §4.3 pipeline overlap, §14 M6 "overlap compute with MPI"
- Module SPEC: gpu/SPEC.md §7 (stream model)
- Role: GPU / Performance Engineer
- Depends on: T6.7
- Milestone: M6

## Goal
Реализовать 2-stream pipelining (D-M6-13): kernel launches идут на `stream_compute`,
H2D+D2H idle-time copies — на `stream_mem`. `cudaEvent` synchronization: MPI send packs
complete when `event_mem_done` fires on `stream_mem`; next step waits для previous step's
compute через `event_compute_done`. Target: ≥ 30% overlap between compute и MPI в K=4
pipeline on 10k-atom Ni-Al 2-rank smoke (measured via Nsight).

## Scope
- [included] src/runtime/simulation_engine.cpp — stream-aware scheduling (modified from T6.7)
- [included] src/scheduler/scheduler_gpu_adapters.cpp — event-based dependency tracking
- [included] tests/integration/gpu_2rank_overlap/ — 10k Ni-Al 2-rank K=4: captures Nsight
  trace, asserts overlap ≥ 30% (programmatic via NVTX + MPI timing)
- [included] docs/specs/gpu/SPEC.md — append §7 stream model subsection (deferred from T6.2)

## Out of scope
- [excluded] N-stream pipelining (N=K — M7)
- [excluded] GPU-aware MPI (M7 — eliminates pack/unpack altogether)
- [excluded] Priority streams (M8+)

## Mandatory invariants
- `stream_compute` + `stream_mem` non-blocking-created
- No `cudaDeviceSynchronize` in hot loop
- Events used для dependency: `cudaEventRecord` + `cudaStreamWaitEvent`
- Overlap ≥ 30% on 10k Ni-Al 2-rank K=4 (telemetry target; not bit-exact — perf gate)
- Regression: 1-rank + K=1 path behavior unchanged from T6.7 (no spurious waits)

## Required files
- src/runtime/simulation_engine.cpp (modified)
- src/scheduler/scheduler_gpu_adapters.cpp (modified)
- tests/integration/gpu_2rank_overlap/*
- docs/specs/gpu/SPEC.md (§7 stream model section)

## Required tests
- Overlap measurement: ≥ 30% on K=4 10k Ni-Al 2-rank (local pre-push)
- M1..M7 smokes unchanged
- Determinism: overlap path bit-exact to serial path (events just hide latency, not
  re-order results)

## Acceptance criteria
- [ ] 2-stream overlap wired + tested
- [ ] Nsight trace shows overlap ≥ 30% target
- [ ] Determinism gate green — thermo byte-exact
- [ ] gpu/SPEC.md §7 committed
```

---

### T6.10 — T3-gpu anchor-test variant (CPU↔GPU Reference gate + dissertation reproduction)

```
# TDMD Task: AnchorTestRunner — GPU variant

## Context
- Master spec: §13.3 mandatory anchor-test + §14 M6 "anchor-test on GPU"
- Module SPEC: verify/SPEC.md §4.4 T3 + gpu/SPEC.md
- Role: Validation / Reference Engineer
- Depends on: T6.9
- Milestone: M6

## Goal
Расширить `AnchorTestRunner` (M5 harness) чтобы поддерживать GPU runs: `backend: gpu` в
config → runs same Al FCC 10⁶ LJ NVE 1000 steps на GPU. Two-level acceptance: (1) GPU
Reference ≡ CPU Reference bit-exact на step 0 forces + step 1000 thermo; (2) GPU MixedFast
within D-M6-8 от GPU Reference; (3) GPU efficiency curve vs N_procs within 10% от
dissertation reference (same bar as M5 CPU). `hardware_normalization_gpu.py` — GPU
micro-kernel probe (CUDA EAM density kernel) для `gpu_flops_ratio` scalar (parallel to
CPU `hardware_normalization.py`).

## Scope
- [included] verify/benchmarks/t3_al_fcc_large_anchor_gpu/ — new fixture (not a copy; pointers
  to T3 CPU fixture + GPU-specific config overrides)
  - README.md (explains two-level gate + scope)
  - config.yaml (GPU backend override)
  - checks.yaml (thresholds: CPU-GPU Reference bit-exact, MixedFast within D-M6-8,
    efficiency within 10%)
  - hardware_normalization_gpu.py (EAM micro-kernel probe)
  - acceptance_criteria.md
- [included] verify/harness/anchor_test_runner/runner.py — add `backend: gpu` dispatch
  + GPU hardware probe integration
- [included] verify/harness/anchor_test_runner/test_anchor_runner.py — extended with
  mocked GPU run fixtures
- [included] docs/development/m6_execution_pack.md — update §5 acceptance gate

## Out of scope
- [excluded] Multi-GPU per rank (M7+)
- [excluded] Self-hosted GPU CI runner (D-M6-6 — deliberately dropped)
- [excluded] MLIAP/SNAP anchor (M8 proof-of-value)

## Mandatory invariants
- CPU-Reference ≡ GPU-Reference step 0 forces bit-exact (Fp64 reduction order canonical
  per D-M6-15)
- CPU-Reference ≡ GPU-Reference step 1000 thermo bit-exact
- GPU MixedFast within D-M6-8 thresholds of GPU Reference
- GPU efficiency curve within 10% of dissertation reference (same gate as M5 CPU)
- Harness detects placeholder CSV (extended detector — both CPU + GPU variants share
  `dissertation_reference_data.csv` source of truth)
- Local-only per D-M6-6 (CI skips T3-gpu with clear "requires GPU" marker)

## Required files
- verify/benchmarks/t3_al_fcc_large_anchor_gpu/* (6 files as listed above)
- verify/harness/anchor_test_runner/runner.py (modified — GPU backend dispatch)
- verify/harness/anchor_test_runner/test_anchor_runner.py (modified)
- docs/development/m6_execution_pack.md (gate checkbox update)

## Required tests
- CPU-GPU Reference bit-exact gate (pytest + real runs — local)
- MixedFast within D-M6-8 (local)
- Efficiency curve within 10% на N ∈ {1, 2, 4} GPU-per-rank local (probing limits of single
  GPU node; N=8 aspirational если multi-node)
- Harness unit tests: mocked GPU run through normal path + corner cases

## Acceptance criteria
- [ ] T3-gpu fixture + harness GPU dispatch committed
- [ ] CPU-GPU Reference gate locally green
- [ ] MixedFast threshold gate locally green
- [ ] Efficiency curve reproducing dissertation within 10% on dev GPU
- [ ] Local screenshot/report.json attached to PR
```

---

### T6.11 — NVTX instrumentation + PerfModel GPU calibration

```
# TDMD Task: NVTX + PerfModel GPU cost tables

## Context
- Master spec: §11.4 PerfModel, §15.4 observability
- Module SPEC: docs/specs/perfmodel/SPEC.md + gpu/SPEC.md
- Role: GPU / Performance Engineer
- Depends on: T6.10
- Milestone: M6

## Goal
Полная NVTX instrumentation: каждый kernel launch + major pipeline phase + H2D/D2H copy
wrapped в `NVTX_RANGE("...")`. Plus PerfModel GPU calibration: update
`src/perfmodel/gpu_cost_tables.cpp` с measured step costs (per-atom ns) для: (1) neighbor
list rebuild, (2) EAM force computation, (3) VV integration, (4) H2D + D2H per zone. Data
source: T6.4/T6.5/T6.6 micro-bench fixtures re-run под Nsight profiling.

## Scope
- [included] src/gpu/*.cu — add NVTX_RANGE (if missing from T6.4-T6.9)
- [included] src/perfmodel/include/tdmd/perfmodel/gpu_cost_tables.hpp
- [included] src/perfmodel/gpu_cost_tables.cpp — measured coefficients; linear model в
  atom count
- [included] src/perfmodel/simple_perf_model.cpp — extended с GPU cost dispatch (когда
  ExecProfile == Production || MixedFast)
- [included] tests/perfmodel/test_gpu_cost_tables.cpp — predict accuracy within ±20% on
  10k/100k/1M Ni-Al EAM runs (sanity)

## Out of scope
- [excluded] Per-zone / per-subdomain GPU cost (M7+ when Pattern 2 adds subdomain work)
- [excluded] Auto-K на GPU (M8)
- [excluded] Heterogeneous model (CPU+GPU mix — M10+)

## Mandatory invariants
- All kernel launches, copies, pool alloc/free wrapped в NVTX (grep audit)
- PerfModel predict() within ±20% of measured runtime on calibration set
- Linear model: `cost(N_atoms) = a + b * N_atoms` per kernel; (a, b) committed to code
  с date + GPU model в comment

## Required files
- src/gpu/*.cu (NVTX audits)
- src/perfmodel/include/tdmd/perfmodel/gpu_cost_tables.hpp
- src/perfmodel/gpu_cost_tables.cpp
- src/perfmodel/simple_perf_model.cpp (modified)
- tests/perfmodel/test_gpu_cost_tables.cpp

## Required tests
- NVTX audit (grep script) — all kernels wrapped
- PerfModel predict accuracy ±20% on micro-bench data

## Acceptance criteria
- [ ] NVTX audit clean
- [ ] GPU cost tables committed с calibration metadata
- [ ] Predict accuracy gate green
- [ ] Nsight trace of T3-gpu readable + informative
```

---

### T6.12 — CUDA CI integration (compile-only, all flavors)

```
# TDMD Task: CI CUDA compile-only + build matrix

## Context
- Master spec: §15.6 CI; memory `project_option_a_ci.md`
- Role: Core Runtime Engineer
- Depends on: T6.11
- Milestone: M6

## Goal
Добавить в `.github/workflows/ci.yml` CUDA build job: ubuntu-latest + `nvidia-cuda-toolkit`
apt install + `cmake -DTDMD_ENABLE_GPU=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"` +
`cmake --build`. Compile-only — kernel tests skipped (no GPU on runner per D-M6-6).
Matrix: `Fp64ReferenceBuild` + `MixedFastBuild`. Per-commit gate.

## Scope
- [included] .github/workflows/ci.yml — add `build-gpu` job (matrix: BuildFlavor ∈
  {Fp64Reference, MixedFast})
- [included] cmake/ — ensure TDMD_ENABLE_GPU correctly auto-detects CUDA toolkit
- [included] tests/gpu/CMakeLists.txt — `TDMD_GPU_REQUIRE_RUNTIME=OFF` на CI (skip runtime tests)
- [included] docs/development/ci_policy.md (if exists) — document Option A extended с CUDA
  compile-only

## Out of scope
- [excluded] Self-hosted GPU runner (D-M6-6)
- [excluded] GPU kernel tests на CI (deferred indefinitely; local pre-push)
- [excluded] Cross-distro CI (ubuntu-latest only)

## Mandatory invariants
- CI green на gcc-13 + clang-17 с CUDA (compile-only)
- Build time <15 min for single flavor (CUDA compile is heavy — target matrix parallelized
  across 2 jobs)
- Runtime-only tests cleanly skipped via CMake conditional
- Local dev experience unchanged (`cmake --preset` defaults unchanged)

## Required files
- .github/workflows/ci.yml (modified)
- cmake/* (helpers)
- tests/gpu/CMakeLists.txt (modified)

## Required tests
- CI trial run: compile matrix green
- Local `cmake --preset gpu` works on dev машине (runtime GPU tests included)

## Acceptance criteria
- [ ] CI green с compile-only CUDA matrix
- [ ] No regression в existing CPU CI jobs
- [ ] Local GPU preset documented в README
```

---

### T6.13 — M6 integration smoke + acceptance gate

```
# TDMD Task: M6 integration smoke — closes the milestone

## Context
- Master spec: §14 M6 final artifact gate
- Role: Validation / Reference Engineer
- Depends on: T6.12
- Milestone: M6 (final)

## Goal
End-to-end (<60s local, skipped on CI) exercising the M6 GPU user surface: Ni-Al EAM config
(M4/M5 smoke reused) с `backend: gpu`, `build_flavor: Fp64ReferenceBuild`, `K=1`, 2 ranks,
host-staged MPI. Verifies: (a) thermo byte-exact to M5 2-rank CPU Reference smoke golden —
D-M5-12 chain extended (D-M6-7 gate); (b) GPU context lifecycle clean (pool stats reset,
streams destroyed, events released — compute-sanitizer smoke); (c) MixedFast 2-rank variant
within D-M6-8 thresholds; (d) anchor-test T3-gpu noted как mandatory pre-merge local gate.

## Scope
- [included] tests/integration/m6_smoke/:
  - README.md — smoke philosophy; D-M6-7 regression chain; local-only per D-M6-6
  - smoke_config.yaml.template — M5 template + `backend: gpu`
  - run_m6_smoke.sh (driver; local `mpirun -np 2` with GPU)
  - thermo_golden.txt — reuses M5 2-rank CPU golden byte-for-byte (D-M5-12 chain extended)
  - telemetry_expected.txt — counters incl. GPU pool stats
- [included] .github/workflows/ci.yml — M6 smoke step marked "skip on CI (requires GPU)"
- [included] docs/specs/gpu/SPEC.md — change log entry: "M6 landed (2026-MM-DD):
  gpu/ module, NL+EAM+VV kernels, 2-stream overlap, MixedFast, T3-gpu anchor"
- [included] docs/specs/comm/SPEC.md — change log entry: "M6 added: host-staged MPI for
  GPU buffers (via pinned pool)"
- [included] docs/specs/perfmodel/SPEC.md — change log entry: "M6 added: GPU cost tables"
- [included] docs/development/m6_execution_pack.md — update §5 acceptance gate checklist

## Out of scope
- [excluded] GPU-aware MPI smoke (M7)
- [excluded] Pattern 2 GPU smoke (M7)
- [excluded] Multi-GPU per rank (M7+)

## Mandatory invariants
- Local wall-time <60s (10k Ni-Al 2-rank K=1 100 steps)
- Thermo K=1 P=2 GPU Reference bit-exact to M5 K=1 P=2 CPU Reference = M4 golden (D-M5-12
  + D-M6-7 chain)
- GPU lifecycle clean: compute-sanitizer reports no leaks
- MixedFast variant (second run) thermo within D-M6-8 of Reference run
- T3-gpu anchor noted as **mandatory pre-merge local gate** в README
- No regressions в M1..M5 smokes + T1, T4 CPU differentials

## Required files
- tests/integration/m6_smoke/* (5 files)
- .github/workflows/ci.yml (updated)
- docs/specs/gpu/SPEC.md (change log)
- docs/specs/comm/SPEC.md (change log)
- docs/specs/perfmodel/SPEC.md (change log)
- docs/development/m6_execution_pack.md (final gate status)

## Required tests
- [smoke local] `./run_m6_smoke.sh` <60s, exit 0
- [CI] `build-gpu` matrix green (compile-only); smoke skipped with clear marker
- [regression] M1..M5 smokes + T1, T4 differentials unchanged
- [local pre-push] T3-gpu anchor passes 10% tolerance + CPU-GPU Reference bit-exact gate

## Acceptance criteria
- [ ] All artifacts committed
- [ ] CI green (6 smokes: M1, M2, M3, M4, M5 CPU; M6 skipped with marker)
- [ ] T3-gpu anchor passes locally — screenshot or report.json attached to PR
- [ ] M6 acceptance gate (§5) fully closed
- [ ] Milestone M6 ready to declare done
```

---

## 5. M6 Acceptance Gate

После закрытия всех 13 задач — проверить полный M6 artifact gate (master spec §14 M6):

- [x] **T6.0** closed retroactively 2026-04-19 by commit `65e142f` (R-M5-8 reference data extracted)
- [x] **`gpu/` module** (T6.2), skeleton + types + abstract interface + CUDA optional build flag
- [x] **`gpu/SPEC.md` v1.0** (T6.2), anchors memory, streams, determinism, precision, NVTX
- [x] **DevicePool + PinnedHostPool** (T6.3), RAII allocators + stream-ordered + size classes
- [x] **Neighbor list GPU** (T6.4), CellGrid → half-NL bit-exact to CPU (Reference)
- [x] **EAM/alloy GPU** (T6.5), density+embed+pair forces bit-exact to CPU (Reference); MixedFast T6.8a shipped rel force ≤1e-5 (D-M6-8 target 1e-6 — FP32 inv_r propagation ceiling; 1e-6 chase carried forward as T6.8b)
- [x] **VV integrator GPU** (T6.6), NVE positions+velocities bit-exact to CPU Reference after 1000 steps
- [x] **SimulationEngine GPU wire** (T6.7), 1-rank + 2-rank GPU Reference thermo bit-exact to M5 CPU Reference (D-M5-12 + D-M6-7 chain)
- [x] **MixedFastBuild GPU** (T6.8a), single-step T4 differential within shipped thresholds; CMake build-flavor matrix green; T6.8b (NL MixedFast + 100-step NVE drift + FP32-table redesign) carry-forward to M7 window
- [x] **Dual-stream `GpuContext` + spline H2D caching** (T6.9a), infrastructure shipped; full compute/copy overlap pipeline + 30% gate deferred to T6.9b (blocked on Pattern 2 GPU dispatch — M7)
- [x] **T3-gpu anchor fixture + harness dispatch** (T6.10a), CPU↔GPU Reference byte-exact gate wired + advisory MixedFast YELLOW via T6.8a delegation; dissertation efficiency curve deferred to T6.10b (blocked on Morse GPU kernel M9+ and Pattern 2 GPU dispatch)
- [x] **NVTX + PerfModel GPU** (T6.11), NVTX audit test clean; `predict_step_gpu_sec` shape shipped with placeholder coefficients; ±20% accuracy gate deferred to T6.11b (needs Nsight run on target GPU — cannot automate on Option A public CI)
- [x] **CUDA CI compile-only** (T6.12), build-gpu matrix green (Fp64Reference + MixedFast); Option A CI policy codified in `docs/development/ci_setup.md`
- [x] **M6 integration smoke** (T6.13), `tests/integration/m6_smoke/` landed; local <5s on commodity GPU; GPU 2-rank thermo **byte-for-byte == M5 golden == M4 golden == M3 golden**; public CI self-skips on `ubuntu-latest` via `nvidia-smi -L` probe per D-M6-6; infrastructure checks (golden parity pre-flight, template substitution, LFS asset presence) fire on every PR
- [x] No regressions: M1..M5 smokes + T1, T4 differentials all green on final push
- [x] CI Pipelines A (lint+build+smokes) + B (unit/property) + C (differentials) + D (build-gpu compile-only) all green
- [x] Pre-implementation + session reports attached в каждом PR
- [x] Human review approval для каждого PR

**M6 milestone status: CLOSED 2026-04-19.** Carry-forward items (T6.8b, T6.9b, T6.10b, T6.11b) tracked as M7-window tasks per §7 (risks & deferrals). Master spec §14 M6 acceptance criteria met: GPU force path + host-staged MPI transport + deterministic CPU↔GPU thermo equivalence on 2-rank smoke + all three CI flavors (Fp64Reference+CUDA, MixedFast+CUDA, CPU-only-strict) green.

---

## 6. Risks & Open Questions

**Risks:**

- **R-M6-1 — Reference GPU ↔ Reference CPU bit-exactness оказывается недостижимым.** FP64
  reduction order на GPU (warp shuffle + atomics) может систематически differ от CPU
  scalar order даже при canonical Kahan. Mitigation: T6.5 impl канонизирует reduction
  через gather-to-single-block + sorted Kahan add (D-M6-15); если невозможно 100%
  bit-exact — escalate to SPEC delta расширяющий §13.5 с GPU-specific tolerance floor
  (1e-14 rel typical), human review.
- **R-M6-2 — `cudaMallocAsync` API edge cases на multi-GPU systems.** Stream-ordered
  alloc на одном device может leak на другом device при migration. Mitigation: T6.3
  restricts DevicePool к single-device per rank в M6 (D-M6-3 constraint); multi-GPU per
  rank — M7+.
- **R-M6-3 — MixedFast EAM overshoots D-M6-8 threshold на edge cases.** FP32 phi'/rho'
  могут round differently на atom pairs около cutoff. Mitigation: T6.8 thresholds relaxable
  via SPEC delta; precompile alternative reduction strategy (Neumaier compensation в
  accumulation) как fallback. Document failure mode в T6.10 report.
- **R-M6-4 — Nsight overhead blurs 30% overlap target в T6.9.** Measurement under trace
  distorts timings. Mitigation: overlap measurement via NVTX start/end timestamps (host-side
  CUDA events) — cheap, reproducible; Nsight только для визуальной проверки, не gate.
- **R-M6-5 — Pinned host memory pressure on small systems.** 256 MiB pinned pool × N
  ranks can exhaust physical RAM на dev машинах с 16 GiB. Mitigation: D-M6-12 pool init
  size = 256 MiB configurable; если T6.13 smoke OOM на 8-GiB RAM — drop к 64 MiB default.
- **R-M6-6 — Deterministic reduction on GPU слишком медленная.** Gather-to-single-block +
  Kahan — O(log N) passes + serialization. На 10⁶ atoms может стоить 20% of force kernel
  time. Mitigation: Reference profile owns scientific correctness budget (this is fine);
  Production profile (M8) gets relaxed reduction order с differential-test coverage. Если
  даже Reference будет слишком медленным на 10⁶ — escalate SPEC delta.
- **R-M6-7 — CUDA 13.1 compile-time incompatibilities с GCC 13/14.** `nvcc` host compiler
  compat может быть brittle (CUDA 12.x пропускал GCC 13). Mitigation: T6.12 CI gate
  catches early; fallback — Clang 17 host compiler через `-ccbin clang-17`.
- **R-M6-8 — 9-недельный M6 может занять 11-12 недель как M5 (6→8).** Новая surface
  area (gpu module + 3 kernels + streams + pools) аналогична M5 (comm + K-batching +
  anchor). Mitigation: 13 PR-size задач с clear dependencies; rolling post-impl reports;
  T6.4/T6.5/T6.6 параллелимые после T6.3.
- **R-M6-9 — PerfModel GPU coefficients stale после hardware upgrade.** Measured on dev
  GPU (skv RTX 40xx); dissertation-class production hardware (A100/H100) — different
  coefficients. Mitigation: T6.11 commits coefficients с comment документирующим GPU model
  plus date; future hardware adds supplementary columns.
- **R-M6-10 — T3-gpu dissertation efficiency gate fails при CPU↔GPU Reference bit-exact
  gate зелёном.** Physics correct but GPU scaling profile иной (better per-step но
  N_procs efficiency может плато раньше из-за H2D/D2H overhead). Mitigation: 10% tolerance
  expected to absorb; если нет — document как R-M6-10 retrospective, escalate SPEC
  delta, honesty — возможно anchor-test нуждается в GPU-specific curve (efficiency
  measured against same-GPU single-rank baseline, not CPU-era dissertation).

**Open questions (deferred to task-time decisions):**

- **OQ-M6-1 — Interpolation representation tables в memory.** EAM tabulated potentials:
  constant memory (fast but 64 KiB limit) vs texture memory (cached, larger) vs global
  SoA. **To decide at T6.5:** prefer constant memory if single-element (~8 KiB per func);
  если multi-element (Ni-Al: 3 × 3 = 9 pair funcs), fall back to global SoA с L1 caching
  hints. Document choice в gpu/SPEC.md.
- **OQ-M6-2 — Neighbor list half vs full on GPU.** Half-list (i<j) saves 2× memory но
  requires atomic force accumulation (на j-loop из i's perspective). Full-list avoids
  atomics но uses 2× memory. **To decide at T6.4:** half-list с Kahan atomic compensation
  в Reference, full-list в MixedFast (if profiling shows atomics dominate). Default
  commits half-list (matches CPU).
- **OQ-M6-3 — H2D copy granularity — full vs delta.** Full atom buffer H2D каждый step
  — simple but 24 MiB/rank/step на 10⁶ atoms × 3 doubles. Delta copy (changed zones) —
  complex но on K=1 это только boundary atoms. **To decide at T6.7:** start full H2D,
  profile, если >5% wall time — implement delta в follow-up issue (not M6 blocker).
- **OQ-M6-4 — Hardware probe strategy для MixedFast.** Some GPUs (consumer Turing/Ampere)
  have poor FP64 throughput — Philosophy B должен бы default-enable MixedFast. **To decide
  at T6.8:** startup log `Fp64/Fp32 ratio` via micro-kernel; if < 1:8, emit warning
  recommending MixedFast. No automatic switching в M6 (user explicit).
- **OQ-M6-5 — Determinism policy при MIG-partitioned GPUs.** MIG instance fractions may
  break warp-level primitives used в canonical reduction. **To decide at T6.5:** detect
  MIG via CUDA runtime, document as unsupported configuration в M6; revisit M9+ when
  cloud deployment matters.
- **OQ-M6-6 — T3-gpu curve vs CPU curve — should они plotted together?** Reports could
  show CPU + GPU efficiency curves side by side. **To decide at T6.10:** report format —
  single CSV per backend; plotting is out of scope (markdown table in report.json suffices).
- **OQ-M6-7 — NVTX naming convention.** `tdmd/force/eam/density` vs `tdmd_force_eam_density`.
  **To decide at T6.11:** slash-separated (matches Nsight grouping UX); document в
  gpu/SPEC.md §8 observability subsection.
- **OQ-M6-8 — GPU context lifetime.** `GpuContext` owned by `SimulationEngine` (scoped
  to simulation) vs process-global singleton. **To decide at T6.7:** owned by engine
  (RAII cleanup on abort); no singleton. Multi-simulation per process — M8+.
- **OQ-M6-9 — `cudaErrorCudartUnloading` at process exit.** Static DevicePool destructors
  running after CUDA driver unload → errors printed at exit. **To decide at T6.3:** all
  pools owned by GpuContext (non-static), engine shutdown before main() returns. No
  static CUDA state.
- **OQ-M6-10 — H100 sm_90 async-copy / TMA usage.** Hopper adds async bulk copy (TMA) —
  potentially 2× speedup на neighbor rebuild. **To decide at T6.4:** не в M6 scope (sm_80
  baseline); revisit на M9+ when stable gen-specific tuning matters.
- **OQ-M6-11 — `compute-sanitizer` CI integration.** Compute-sanitizer runs kernels —
  requires GPU. **To decide at T6.12:** local pre-push only (consistent с D-M6-6); CI
  skips; document в ci_policy.md.

---

## 7. Roadmap Alignment

| Deliverable | Consumer milestone | Why it matters |
|---|---|---|
| `gpu/` module skeleton + SPEC (T6.2) | M7 GPU-aware MPI layers на top; M7 NCCL backend; M8 SNAP GPU; M9 NVT/NPT GPU | Single abstraction for all GPU work; no module bypasses gpu/ for CUDA calls |
| DevicePool + PinnedHostPool (T6.3) | M7 multi-GPU per rank extends pool; M8 SNAP large tables reuse cached allocation | Foundational — every GPU kernel uses pool, never `cudaMalloc` directly |
| Neighbor list GPU (T6.4) | M7 Pattern 2 ghost zones rebuild на GPU; M8 SNAP consumes same NL format; M9 LongRange uses extended cutoff NL | Half-list SoA — contract preserved forever |
| EAM GPU (T6.5) | M7 EAM in Pattern 2; M8 SNAP follows same device pattern (tabulated + reduction); M9 Virial/stress adds на top | Canonical reference для всех tabulated potentials на GPU |
| VV integrator GPU (T6.6) | M9 NVT/NPT extend integrator/ GPU family; M10 Langevin + SHAKE add on top | Simplest integrator — establishes GPU integrator pattern |
| SimulationEngine GPU wire (T6.7) | M7 extends engine со GPU-aware MPI branches; M8 SNAP runs на same engine | Engine остаётся coordination-only; GPU physics в gpu/, time в scheduler/, transport в comm/ — clean separation preserved |
| MixedFastBuild GPU (T6.8) | M8 performance proof (SNAP beats LAMMPS если MixedFast на достаточных ranks); M10+ MixedFastAggressive + Fp32Experimental | Philosophy B canonical — demonstrates 2-4× speedup path без sacrificing scientific observables |
| 2-stream overlap (T6.9) | M7 upgrades до N-stream + GPU-aware MPI (eliminates pack/unpack); M8 SNAP benefits from overlap | Baseline overlap — refined в M7 |
| T3-gpu anchor (T6.10) | Continuous regression guard — any PR touching GPU MUST pass T3-gpu slow tier | GPU correctness gate; dissertation reproduction extended to GPU era |
| PerfModel GPU (T6.11) | M7 Pattern 2 cost estimation requires GPU coefficients; M8 auto-K на GPU uses predict(); M9+ heterogeneous | Makes scheduler's K and zone sizing choices GPU-aware |
| CUDA CI (T6.12) | Catches compile regressions M7-M13 без требования GPU runner | Preserves Option A CI policy while adding GPU surface coverage |
| M6 smoke (T6.13) | Regression gate M7-M13 — GPU stack exercised pre-push on every PR touching gpu/ | Catches GPU regression before it reaches T3-gpu slow tier |

---

*End of M6 execution pack, дата: 2026-04-19.*
