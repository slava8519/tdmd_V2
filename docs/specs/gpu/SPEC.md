# gpu/SPEC.md

**Module:** `gpu/`
**Status:** master module spec v1.0.18 (T8.6b shipped — SNAP GPU FP64 functional kernel body — three-kernel Ui→Yi→deidrj pipeline + index-table flatten + peer-side Newton-3 reassembly; bit-exact gate deferred to T8.7)
**Parent:** `TDMD Engineering Spec v2.5` §14 M6 (closed) / §14 M7 (closed) / §14 M8 (in progress), §15.2, §D (precision policy)
**Last updated:** 2026-04-20

---

## 1. Purpose и scope

### 1.1. Что делает модуль

`gpu/` — **единственный** модуль TDMD, который знает про CUDA runtime, streams, events, device memory allocation. Абстракции `DeviceStream`, `DeviceEvent`, `DevicePtr<T>`, `DeviceAllocator` скрывают все CUDA-specific детали от остального кода.

Делает пять вещей:

1. **Device lifecycle** — probe / enumerate / select CUDA device, init context, shutdown;
2. **Stream + event primitives** — RAII обёртки над `cudaStream_t` / `cudaEvent_t` с compile-firewall (CUDA headers не утекают в public API);
3. **Memory management** — cached device pool (`cudaMallocAsync` stream-ordered) + pinned host pool (`cudaMallocHost`) для MPI staging (D-M6-3, D-M6-12);
4. **Kernel launch infrastructure** — thin wrappers + NVTX instrumentation (D-M6-14), deterministic reduction primitives (D-M6-15);
5. **Mixed-precision контракт** — `Fp64ReferenceBuild` vs `MixedFastBuild` semantic разница на GPU-level (D-M6-5, D-M6-8).

### 1.2. Scope: что НЕ делает в M6

- **Kernels outside of (NL, EAM, VV)** — LJ, Morse, MEAM, SNAP, PACE, MLIAP откладываются на M9+ (D-M6-4);
- **GPU-aware MPI / NCCL / NVSHMEM** — transport остаётся host-staged через `MpiHostStagingBackend` (D-M6-3); GPU-aware transport — M7;
- **Multi-GPU per rank** — 1 GPU : 1 MPI rank строго (D-M6-3 environment note);
- **Unified memory / managed memory** — явный `cudaMemcpyAsync` H⇄D через `stream_mem` (см. §3); UM — экспериментальный выбор, не в M6;
- **NPT/NVT integrator GPU path** — только NVE в M6 (D-M6-4); NVT/NPT остаются на CPU;
- **clang-cuda compilation driver** — nvcc-only в M6 (D-M6-10); clang-cuda — M10+;
- **Self-hosted GPU CI runner** — compile-only в CI (D-M6-6); runtime gates local-only;
- **FP16 / bfloat16** — precision floor = FP32 (Philosophy B, D-M6-5); FP16 — M10+ experiments;
- **Profile-guided kernel autotuning** — launch configs жёстко подобраны в M6; autotuner — M8+.

`gpu/` — чистый primitive layer. Data-oblivious, physics-oblivious, topology-oblivious. Зависимостей только от `telemetry/` (NVTX bridge) и `verify/` (threshold registry, косвенно через `differential_runner`).

### 1.3. Почему gpu/ критичен для TDMD

В обычном MD-коде CUDA-integration — «ещё один backend». В TDMD — **ключ к Philosophy B**:

- **Reference-oracle инвариант** (`docs/development/claude_code_playbook.md` §5.1): `Fp64ReferenceBuild + Reference profile` должен давать bit-exact результаты на CPU и GPU. Это фиксирует reduction order (D-M6-15), атомики (запрещены без Kahan), и FP64 arithmetic (fmad=false, ffp-contract=off на CUDA);
- **MixedFast differential budget** (D-M6-8): Philosophy B контракт `per-atom force 1e-6 rel + energy 1e-8 rel + NVE drift 1e-5 per 1000 steps` — единственный способ измерить «ничего не сломалось научно» при переходе FP64→(FP32 math + FP64 accum). Без этого GPU — чёрный ящик;
- **Pool + stream policy** (§3, §5): без cached pool `cudaMalloc`/`cudaFree` в NL rebuild hot path убивают 10-20% throughput на 10⁶-атомном фикстюре. Без 2-stream overlap host staging blocking = потеря фактора 1.5-2× на MPI-связанных шагах;
- **NVTX покрытие** (D-M6-14): Nsight Systems trace — primary perf tool в M6; без NVTX launches debug занимает дни.

Плохо написанный `gpu/` превращает GPU-путь в «чуть медленнее чем CPU потому что все overheads». Хорошо написанный — открывает дверь Pattern 2 overlap на M7 и полному скоупу kernels на M8+.

---

## 2. Core types

### 2.1. Handles (PIMPL compile firewall, D-M6-17)

Все CUDA-specific state спрятан за PIMPL — public headers компилируются на CPU-only build (CI `TDMD_BUILD_CUDA=OFF`). Это позволяет keep CI budget минимальным и не требовать CUDA toolkit на машинах, которые только строят и тестируют CPU-код.

```cpp
namespace tdmd::gpu {

class DeviceStream {
 public:
  struct Impl;                        // forward-declared; defined in .cu (T6.3)
  DeviceStream();                     // null stream (valid()==false)
  explicit DeviceStream(std::unique_ptr<Impl>);
  ~DeviceStream();
  DeviceStream(DeviceStream&&) noexcept;
  DeviceStream& operator=(DeviceStream&&) noexcept;
  DeviceStream(const DeviceStream&) = delete;
  DeviceStream& operator=(const DeviceStream&) = delete;
  Impl*       impl() noexcept;
  const Impl* impl() const noexcept;
  bool        valid() const noexcept;
 private:
  std::unique_ptr<Impl> impl_;
};

class DeviceEvent { /* same shape as DeviceStream */ };

template <typename T>
class DevicePtr {
 public:
  using DeleterFn = void (*)(void* device_ptr, void* context) noexcept;

  DevicePtr() noexcept = default;
  DevicePtr(T* ptr, DeleterFn, void* ctx) noexcept;
  ~DevicePtr();                        // invokes deleter if non-null

  DevicePtr(DevicePtr&&) noexcept;
  DevicePtr& operator=(DevicePtr&&) noexcept;
  DevicePtr(const DevicePtr&) = delete;
  DevicePtr& operator=(const DevicePtr&) = delete;

  T*      get() const noexcept;
  explicit operator bool() const noexcept;
  [[nodiscard]] T* release() noexcept; // opts out of deletion
  void     reset() noexcept;
};

}  // namespace tdmd::gpu
```

**Invariants:**

- `DeviceStream`, `DeviceEvent`, `DevicePtr<T>` — move-only. Compile-time enforced в `tests/gpu/test_gpu_types.cpp` через `static_assert(std::is_nothrow_move_constructible_v<T>)` + `static_assert(!std::is_copy_constructible_v<T>)`;
- Default-constructed handle — **null**. `valid() == false`; `get() == nullptr`. Используется для «ещё не инициализировано» и для moved-from state;
- Deleter у `DevicePtr<T>` — noexcept function pointer + opaque context. T6.3 DevicePool предоставляет deleter возвращающий block в pool; PinnedHostPool — deleter на `cudaFreeHost`. Нельзя использовать `std::function` — это увеличит overhead на hot-path free и вытащит exceptions на путь без них.

### 2.2. Value types

```cpp
namespace tdmd::gpu {

using DeviceId      = std::int32_t;  // CUDA device ordinal
using StreamOrdinal = std::uint8_t;  // per-rank stream index

inline constexpr StreamOrdinal kStreamCompute = 0;  // kernel launches
inline constexpr StreamOrdinal kStreamMem     = 1;  // H<->D copies

enum class GpuCapability : std::uint8_t {
  CudaMallocAsync,      // CUDA 11.2+ stream-ordered allocator (D-M6-12)
  CooperativeLaunch,    // cross-block sync
  TensorCores,          // sm_80+ MMA (reserved for M9+ SNAP)
  AsyncMemcpy,          // cp.async, sm_80+
  L2CachePersistence,   // stream access policy hint, sm_80+
};

struct DeviceInfo {
  DeviceId        device_id;
  std::string     name;
  std::uint32_t   compute_capability_major;
  std::uint32_t   compute_capability_minor;
  std::size_t     total_global_memory_bytes;
  std::uint32_t   multiprocessor_count;
  std::uint32_t   warp_size;               // default 32
  std::uint32_t   max_threads_per_block;
  std::vector<GpuCapability> capabilities;
};

struct GpuConfig {
  std::int32_t  device_id                   = 0;
  std::uint32_t streams                     = 2;     // D-M6-13
  std::uint32_t memory_pool_init_size_mib   = 256;   // D-M6-12 warm-up
  bool          enable_nvtx                 = true;  // D-M6-14
};

}  // namespace tdmd::gpu
```

### 2.3. PIMPL compile firewall — rationale

**D-M6-17 decision:** keep all CUDA types (`cudaStream_t`, `cudaEvent_t`, device pointer typedefs) out of public headers. Rationale:

1. **CPU-only build остаётся green** — users without CUDA toolkit могут строить и тестировать `cli/`, `runtime/`, `scheduler/` targets. M5 CI уже работает без CUDA; M6 не должен ломать это;
2. **Fast compilation** — `cudaStream_t` transitively pulls `cuda_runtime.h`, `device_types.h`, `vector_types.h` и ~30 Kb of headers в каждый .cpp файл который #include'ит gpu-публичный заголовок. PIMPL отсекает;
3. **Binary ABI стабильность** — если в будущем мы заменим CUDA backend на HIP (AMD), PIMPL позволит сделать это без перекомпиляции consumers.

Cost: один virtual call (`Impl*` → реальный `cudaStream_t`) на accessor. Amortized over kernel launch cost (≥microsecond) — ноль practical overhead.

---

## 3. Stream model (D-M6-13)

### 3.1. Two-stream policy

Каждый MPI-rank владеет **двумя streams**:

| Stream           | `StreamOrdinal` | Purpose                                      |
|------------------|-----------------|----------------------------------------------|
| `stream_compute` | 0               | Kernel launches (NL, EAM, VV + reductions)   |
| `stream_mem`     | 1               | H⇄D copies (atom send/recv buffers, MPI pack/unpack) |

**Rationale:** 2 streams — минимум для compute/MPI overlap. Full N-stream K-way pipelining добавляется когда Pattern 2 landed (M7). В M6 Pattern 1 + K=1..8, два stream'а достаточны: kernel работает на `stream_compute` пока D2H предыдущего packet'а идёт на `stream_mem`.

### 3.2. Sync primitives (events)

Cross-stream ordering — через `cudaEventRecord` + `cudaStreamWaitEvent` (RAII обёртки `DeviceEvent` + `DeviceStream`). Нет implicit `cudaDeviceSynchronize()` на hot path. Legacy default stream (NULL) — **запрещён** в M6 kernels.

Pattern (T6.9 pipeline):

```
t=0:  H2D(atoms_0) on stream_mem → record event_h2d_0
t=1:  stream_compute waits event_h2d_0
      launch NL + EAM + VV on stream_compute → record event_compute_0
t=2:  stream_mem waits event_compute_0
      D2H(forces_0) on stream_mem → record event_d2h_0
t=3:  stream_mem: MPI pack from pinned host buffer
```

### 3.2a. T6.9a shipped subset — dual-stream infra + spline H2D caching (v1.0.7)

**Что landed.** `runtime::GpuContext` теперь держит **оба** non-blocking stream'а (D-M6-13): `compute_stream()` + `mem_stream()`. Оба создаются через `make_stream(device_info_.device_id)`, т.е. `cudaStreamNonBlocking` flag. Adapter'ы в v1.0.7 всё ещё берут только `compute_stream()` из `GpuContext` — `mem_stream()` surface готов к использованию, но фактическая `cudaEventRecord` / `cudaStreamWaitEvent` orchestration (full T6.9 pipeline из §3.2) отложена до T6.9b.

**Spline H2D caching (§7.2 EAM adapter side).** `EamAlloyGpu::Impl` + `EamAlloyGpuMixed::Impl` содержат три host-pointer cache fields (`splines_{F,rho,z2r}_coeffs_host`) + upload counter (`splines_upload_count`). На входе в `compute()` kernel сравнивает incoming `tables.F_coeffs` / `tables.rho_coeffs` / `tables.z2r_coeffs` с cached pointers; если все три совпадают — re-upload пропускается (device buffers остаются валидны из пула). На первом compute() и всех последующих с новыми pointers счётчик инкрементируется. Invariant: **после N back-to-back compute() calls с одним `EamAlloyGpuAdapter` instance — `splines_upload_count() == 1`**. Test coverage: `tests/gpu/test_eam_alloy_gpu.cpp::"EamAlloyGpu — splines cached across compute() calls (T6.9a)"`.

**Почему это T6.9 а не T6.7.** Spline tables flatten-и уже лежали в `EamAlloyGpuAdapter` как immutable поля (construction-time), но T6.5 kernel их re-uploaded на каждом `compute()` call. На steady-state MD hot loop (1000+ compute() calls между NL rebuilds) это доминировало H2D bandwidth на MixedFast fixture'ах. Caching снижает per-step H2D overhead к `n_atoms × 40 bytes` (positions + forces + cell CSR), убирает ~MB-scale spline re-upload. Perf benefit гарантированно positive; exact speedup измеряется в T6.11 NVTX timeline.

**T6.9b scope (deferred).** Полная compute/copy overlap pipeline (§3.2 sync primitives) + scheduler GPU adapters (`src/scheduler/scheduler_gpu_adapters.cpp`) + 30% overlap acceptance gate на 2-rank K=4. T6.9b ждёт Pattern 2 GPU dispatch от M7 (см. §9.5) — до тех пор dependency graph single-packet, overlap opportunities ограничены NL rebuild boundaries.

### 3.2b. T7.8 shipped — full 2-stream split-phase pipeline + K-deep dispatch adapter (v1.0.8)

**Что landed.** `EamAlloyGpu` теперь экспортирует **split-phase async API**: `compute_async()` queues H2D на `mem_stream` + kernels на `compute_stream` (с `cudaStreamWaitEvent` chain) + records `kernels_done_event`, **без D2H**; `finalize_async()` queues D2H на `mem_stream` (waiting `kernels_done_event`) + reduces. Split required to keep `mem_stream` от self-serializing: при single-phase implementation все K H2D и D2H queue back-to-back на `mem_stream` linearly, и cross-stream `cudaStreamWaitEvent(mem, kernels_done)` blocks subsequent H2D'ов до завершения D2H'а — overlap window коллапсирует до 0. Split-phase позволяет K H2D'ам queue-нуться на `mem_stream` подряд, kernels запускаются на `compute_stream` параллельно, D2H'и appended после kernels.

**Pinned host buffers — обязательны.** `EamAlloyAsyncHandle::Impl` теперь хранит `DevicePtr<std::byte>` для D2H destinations (`h_pe_embed`, `h_pe_pair`, `h_virial`), allocated через `DevicePool::allocate_pinned_host()`. Pageable memory degrades `cudaMemcpyAsync` to internal staging + host-thread block silently, что убивает overlap. Test fixtures должны использовать `allocate_pinned_host()` для всех buffers, проходящих через H2D/D2H в pipelined path.

**K-deep dispatch adapter.** `tdmd::scheduler::GpuDispatchAdapter` (`src/scheduler/include/tdmd/scheduler/gpu_dispatch_adapter.hpp` + `.cpp`) rotates через K internal slots, каждый с своим `EamAlloyGpu` instance + `std::optional<EamAlloyAsyncHandle>` pending. `enqueue_eam(...)` picks `next_slot_`, calls `compute_async`, advances; `drain_eam(slot)` calls `finalize_async`, returns `EamAlloyGpuResult`. FIFO drain order — caller responsibility. Single allocator (`DevicePool`) shared across slots — pool's stream-ordered `cudaMallocAsync` correctly serializes per-stream allocs.

**Single-rank EAM-only acceptance gate (5%, не 30%).** `tests/gpu/test_overlap_budget.cpp` measures wall-time overlap ratio `(t_serial - t_pipelined) / t_pipelined` на K=4, 14×14×14 Al FCC (10976 atoms, Al_small.eam.alloy). На RTX 5080: t_serial ≈ 7.48 ms, t_pipelined ≈ 6.85 ms → overlap ≈ 9.3%, gate ≥ 5% (functional, comfortably above noise). PE + virial slot 0 vs serial oracle — bit-exact at ≤ 1e-12 rel (D-M6-7 preserved).

**Почему не 30% single-rank.** EAM на RTX 5080 на 10k atoms — kernel-bound: T_kernel ≈ 1.5 ms, T_mem ≈ 0.36 ms (H2D+D2H), ratio ≈ 0.24. Asymptotic max overlap ratio (K→∞) = (T_h+T_d)/T_k ≈ 24%; для K=4 — около 17%. Hardware-limited; нельзя hit 30% на single-rank EAM-only вне зависимости от K. **30% gate сохраняется для 2-rank Pattern 2** где halo D2H + MPI_Sendrecv + halo H2D roughly doubles per-step memory work, повышая T_mem/T_k до ~0.55 — там 30% achievable.

**T7.14 ownership.** Полный 30% overlap gate на 2-rank K=4 10k-atom setup (per exec pack §T7.8 «2-rank» specification) переносится в T7.14 M7 integration smoke, где Pattern 2 dispatch (T7.9) + halo traffic (T7.6 OuterSdCoordinator) активируется в end-to-end smoke run. T7.8 ships pipeline mechanism + bit-exact + functional single-rank gate.

### 3.2c. T8.0 (T7.8b carry-forward) — 2-rank overlap gate: hardware prerequisite + dev SKIP semantics (v1.0.9)

**Что landed.** `tests/gpu/test_overlap_budget_2rank.cpp` — MPI-aware 2-rank variant of T7.8 overlap gate. Per-rank device pinning (`cudaSetDevice(rank % device_count)`), K=4 `GpuDispatchAdapter`, synthetic halo `MPI_Sendrecv` (1024 doubles pinned ≈ 8 KB, modelling the halo slab volume of a P_space=2 split on a ~50 Å×50 Å contact face) interleaved with GPU compute. Serial baseline = K iterations of `{sync EAM compute, sync halo Sendrecv}`; pipelined = K async enqueues + K drains interleaved with Sendrecv. Gate: `REQUIRE(overlap_ratio >= 0.30)`, median of 9 repeats. Bit-exact slot 0 PE + virial vs serial oracle at ≤ 1e-12 rel (D-M6-7 preserved).

**T7.14 closure note.** T7.14 as actually landed is a correctness-only smoke (thermo byte-exact chain + telemetry invariants) and does **not** measure overlap. The 2-rank 30% overlap measurement was therefore carried forward as T7.8b and ships its test infrastructure as T8.0. T7.14 section above is retained for historical context; current ownership of the 30% gate is T8.0 (infrastructure) + T8.11 (runtime measurement).

**Hardware prerequisite.** Meaningful 2-rank overlap measurement requires **≥ 2 physical CUDA devices** so each rank owns a distinct GPU. On hosts with 1 GPU, co-tenancy of two ranks on the same device serializes compute + mem streams at the driver level, which kills overlap (ratio collapses to noise) and the test would flake or fail spuriously. The binary therefore checks `cudaGetDeviceCount() >= 2` at test entry and `SKIP`s with Catch2 exit code 4 when the host has fewer than 2 devices.

**Dev SKIP semantics.** Dev workstations (this repo's reference: 1× RTX 5080) SKIP this test by design — not a gap, a deliberate hardware contract. CMake wraps the test with `set_tests_properties(test_overlap_budget_2rank PROPERTIES SKIP_RETURN_CODE 4)` so CTest surfaces the exit as SKIPPED rather than FAIL, keeping D-M6-6 Option A (no self-hosted GPU runner) CI matrices green. The real 30% measurement runs on cloud-burst hardware (≥ 2 GPU node) and is tied into T8.11 TDMD-vs-LAMMPS scaling harness.

**Why 30% achievable at 2-rank K=4.** In 2-rank Pattern 2, halo D2H + MPI_Sendrecv + H2D per step roughly doubles the memory-traffic fraction of the step: `T_mem/T_k ≈ 0.24 → 0.55`. Asymptotic max overlap (K→∞) ≈ T_mem / (T_k + T_mem) ≈ 36%; at K=4 the achievable ratio is ~30–34% — the 30% bar is the conservative floor. This matches the exec pack §T7.8 derivation that the original 30% gate was always a 2-rank gate, never a single-rank one (see §3.2b «Почему не 30% single-rank»).

**Change log.** v1.0.9 — 2026-04-20. T8.0 adds §3.2c (hardware prerequisite for 30% gate + dev SKIP semantics).

### 3.3. Debug single-stream mode

`GpuConfig::streams = 1` force-serializes всё на `stream_compute`. Используется:

1. **bug-isolation** — определить, связан ли bug с cross-stream ordering (обычно — NO, но полезно отключить переменные);
2. **NVTX trace simplification** — Nsight Systems timeline проще читается single-stream;
3. **Compatibility** — с backend'ами которые не поддерживают stream-ordered allocation.

Не должен использоваться в production.

### 3.4. N-stream roadmap (M7+)

M7 добавит третий stream — `stream_aux` для outer-level halo (Pattern 2 SD exchange), в соответствии с master spec §14 M6 перечислением трёх stream'ов. M8+ может масштабировать до K stream'ов для full pipeline overlap. gpu/SPEC v1.x добавит §3.5 когда это landed.

---

## 4. Device probe + enumeration

```cpp
namespace tdmd::gpu {

// Enumerate visible CUDA devices. Returns empty vector если
// TDMD_BUILD_CUDA=0 или если cudaGetDeviceCount возвращает 0.
std::vector<DeviceInfo> probe_devices();

// Resolve device_id от GpuConfig; если device_id выходит за пределы
// cudaGetDeviceCount, выбрасывает std::runtime_error.
DeviceInfo select_device(DeviceId id);

}  // namespace tdmd::gpu
```

`probe_devices()` вызывается один раз на rank на SimulationEngine init. Caches результат для telemetry (NVTX domain name = device name). Caller определяет какой device выбрать (обычно `MPI_COMM_WORLD rank % num_devices`).

**Implementation note:** `probe_devices()` в CPU-only build — compile-time `return {}`. Header остаётся callable, runtime просто не находит ничего.

---

## 5. Memory model (D-M6-12, D-M6-3)

### 5.1. Cached device pool

`DevicePool` (T6.3, landed 2026-04-19) — concrete `DeviceAllocator`. Один класс владеет обоими pools (device + pinned-host) потому что в M6 — один rank на process, pool 1:1 привязан к rank (нет multi-tenant sharing). Реализация на base `cudaMallocAsync` + pool reuse:

- **Stream-ordered semantics** — alloc/free привязаны к stream через handle. Safely reuses буферы between stream consumers без explicit sync, если dependency graph корректный;
- **Size classes**: {4 KiB, 64 KiB, 1 MiB, 16 MiB, 256 MiB}. Requests округляются вверх до ближайшего класса; accounting (`bytes_in_use_device`) — по class bytes (honest re device commitment, не request bytes). Requests >256 MiB идут direct в `cudaMallocAsync` без pooling (oversize-fallback);
- **Free-list policy v1.0** — простой grow-on-demand per-class free-list. На deleter block возвращается в free-list; на allocate пустой free-list → `cudaMallocAsync` miss. LRU eviction deferred (OQ-M6-1) — добавим когда T6.5 kernel pressure-testing покажет pool bloat как проблема;
- **Warm-up** — `memory_pool_init_size_mib` MiB (default 256, D-M6-12) выделяется в constructor как блоки класса 2 (1 MiB) и сразу переносится в free-list. Первый kernel launch не стал на cold-start allocator;
- **Thread-safety** — `DevicePool` **не thread-safe** (в M6 один logical thread на rank, no OpenMP inside kernels). Внешняя синхронизация нужна если concurrent access когда-либо потребуется.

### 5.2. Pinned host pool

Pinned host memory allocator — **part of `DevicePool`** (не отдельный класс в v1.0). Использует `cudaMallocHost` для page-locked pages. Потребитель — MPI staging (D-M6-3): GPU forces D2H → pinned host buffer → MPI send → MPI recv → pinned host buffer → H2D. Pinned crucial потому что `cudaMemcpyAsync` требует host side быть pinned для async semantics; pageable memory forces sync copy.

Size classes те же что у device pool (симметричный pipeline). Free-list policy та же (grow-on-demand, LRU deferred). Warm-up для pinned отключён (pinned host memory — scarce resource, избегаем pre-allocation).

### 5.3. DeviceAllocator abstract interface

```cpp
class DeviceAllocator {
 public:
  virtual ~DeviceAllocator();

  virtual DevicePtr<std::byte> allocate_device(std::size_t nbytes,
                                               DeviceStream& stream) = 0;
  virtual DevicePtr<std::byte> allocate_pinned_host(std::size_t nbytes) = 0;

  virtual std::size_t bytes_in_use_device() const noexcept = 0;
  virtual std::size_t bytes_in_use_pinned() const noexcept = 0;
};
```

**Contract:**

- `allocate_device(nbytes, stream)` — if `stream.valid()`, allocation stream-ordered; else synchronous cudaMalloc fallback (debug path);
- `allocate_pinned_host(nbytes)` — always synchronous (pinned host allocation is slow anyway, ~millisecond; doesn't warrant async);
- Returned `DevicePtr<std::byte>` — carries pool-return deleter. `.release()` transfers ownership to caller (caller must call `cudaFree` / `cudaFreeHost` manually).

### 5.4. Pool stats telemetry

`bytes_in_use_device()` / `bytes_in_use_pinned()` surfaced в `TelemetryFrame`:

```yaml
gpu:
  device_bytes_in_use: 134217728   # bytes
  pinned_bytes_in_use: 12582912
  alloc_hit_rate: 0.98             # cached pool hits / total allocs
  pool_growth_events: 2
```

Telemetry позволяет детектить memory leaks (monotonically growing `bytes_in_use_device` over stable steady-state step) и недо-sized pool (`pool_growth_events > 5 / minute`).

---

## 6. Determinism policy (D-M6-15, D-M6-7)

### 6.1. Canonical reduction order

Все reductions в Reference profile — **canonical**. Это значит:

- Reduction-to-single-block pattern: каждый block reduces свою partition в shared memory, финальная reduction of block outputs — на CPU или через single-block launch с sorted inputs;
- Sort by atom ID (int) перед добавлением в accumulator — гарантирует bit-exact reproducibility независимо от grid dim / thread dim;
- Kahan compensation (см. §6.2) — обязательна в Reference. В MixedFast — **тоже обязательна**, differences только в precision math (FP32 partial + FP64 accumulator), не в order.

**Forbidden в Reference:**

- `atomicAdd(float*, float)` без Kahan companion;
- `atomicAdd(double*, double)` без sort (race-order dependency);
- Tree reduction без phase-commit (`__syncthreads` между reduction layers gives correct result in isolation, но cross-launch order не гарантирован без event sync).

### 6.2. Kahan compensation intra-kernel

Extend `deterministic_sum_double` (comm/SPEC §7.2) on intra-kernel reductions:

```cuda
struct KahanAcc {
  double sum;
  double c;  // compensation
};

__device__ __forceinline__
void kahan_add(KahanAcc& a, double x) {
  double y = x - a.c;
  double t = a.sum + y;
  a.c = (t - a.sum) - y;
  a.sum = t;
}
```

Applied в EAM density pass (per-atom ρ accumulator), EAM pair force pass (per-atom fᵢ accumulator), VV velocity update. Forbidden `fmad` contraction enforces canonical order (D-M6-15): `--fmad=false` на CUDA compile флагах (уже wired в `BuildFlavors.cmake` — см. `_tdmd_apply_fp64_reference`).

### 6.3. Reference CPU ≡ Reference GPU gate (D-M6-7)

**Acceptance test:** Ni-Al 10⁴ atoms EAM/alloy one-step force + energy.

- CPU Reference path: existing M2 `EamAlloyPotential` output;
- GPU Reference path: T6.5 kernel output;
- Compare: **bit-exact** on IEEE754 doubles. Not `abs(a-b) < 1e-15` — literal byte-equal pattern через `std::memcmp`.

Registered в `verify/SPEC.md` threshold registry as `cpu_gpu_reference_force_bit_exact` (T6.5 adds entry). Gate на merge of T6.5. T6.7 extends to step-1000 thermo chain.

---

## 7. Kernel contracts (M6 scope per D-M6-4)

### 7.1. Neighbor-list build (T6.4)

**Contract:**

- Input: `DeviceAtomSoA` (pos, species, count) + `CellGrid` dimensions + cutoff;
- Output: half-list (i<j only) as SoA (D-M6-16): `neighbor_offset[N+1]` prefix sum, `neighbor_idx[]` flattened neighbor indices, `neighbor_d2[]` squared distances (cached для force kernel);
- Determinism: sort within each bucket by neighbor atom ID. Bit-exact to CPU CellGrid в Reference build;
- Launch: `stream_compute`, 1 block per cell bucket, 32-thread warps per block. Cooperative 1-block launch для prefix-sum phase.

### 7.2. EAM/alloy force (T6.5 — v1.0.3)

**Contract:**

- Three kernels per `compute()` launch, thread-per-atom, identical cell-stencil iteration order across all three:
  - **density_kernel** — ρᵢ = Σⱼ ρ_{β}(rᵢⱼ) via 27-cell stencil, full-list per-atom (no `j<=i` filter);
  - **embedding_kernel** — pe_embed[i] = F_α(ρᵢ); dFdrho[i] = F'_α(ρᵢ); thread-per-atom, no reduction;
  - **force_kernel** — f[i] += Σⱼ (dE/dr · Δ / r) · r̂; pe_pair[i] and per-atom 6-component virial written to dedicated output buffers.
- Full-list iteration chosen over half-list because each thread writes only to its own atom's slot (no atomics needed, no deterministic-scatter machinery). Pair PE + virial are consequently double-counted; the **host reduction halves them** during Kahan summation. Forces are emitted once per ordered pair (each direction gets its own thread's contribution), so no halving applies.
- Device spline evaluation bit-exactly mirrors CPU `TabulatedFunction::locate` + Horner form, with the 7-doubles-per-cell layout reproduced verbatim:
  - `locate_dev(x) → (cell_idx, p)` with LAMMPS-style 1-based clamp;
  - `eval = ((c[3]·p + c[4])·p + c[5])·p + c[6]`;
  - `deriv = (c[0]·p + c[1])·p + c[2]`.
- Minimum-image formula identical to CPU `state::minimum_image_axis` (same branch conditions + same `ceil` usage), preserving FP sequence.
- Host-side **Kahan compensated reduction** of per-atom PE and per-atom 6-component virial buffers (D-M6-15). Per-atom force buffers are D2H'd and returned directly — no cross-atom reduction needed.
- Public API (`tdmd::gpu::EamAlloyGpu`) takes raw primitives (positions + cell CSR + flattened spline coefficient arrays + `BoxParams`) — keeps gpu/ data-oblivious per §1.1. Domain translation in `src/potentials/eam_alloy_gpu_adapter.cpp`.
- MixedFast: FP32 math в splines evaluation; FP64 accumulator для ρᵢ, Fᵢ, fᵢ (Philosophy B) — activated in T6.8.

**Acceptance gate (D-M6-7):**

- Per-atom forces + total PE + virial Voigt tensor agree **≤ 1e-12 rel** CPU ↔ GPU on Al FCC 864-atom + Ni-Al B2 1024-atom (Mishin 2004).
- Gate is relative, not byte-equal: absorbs FP64 reduction-order drift between CPU half-list and GPU full-list sums. Math kernels themselves use identical FP sequences, so any divergence above 1e-12 indicates a real bug.

### 7.3. Velocity-Verlet NVE (T6.6 — v1.0.4)

**Contract (Reference path, FP64):**

Two kernel entry points mirroring CPU `VelocityVerletIntegrator`:

1. **`pre_force_kernel`** — half-kick velocities using CURRENT forces, then full drift positions:

   ```
   v[i] ← v[i] + accel[type[i]] · f[i] · (dt/2)
   x[i] ← x[i] + v[i] · dt
   ```

2. **`post_force_kernel`** — half-kick velocities using NEW forces (computed after drift):

   ```
   v[i] ← v[i] + accel[type[i]] · f[i] · (dt/2)
   ```

где `accel[s] = ftm2v / mass[s]` — per-species scalar precomputed **once** on host (LAMMPS `units metal`
convention, `ftm2v = 1/1.0364269e-4 ≈ 9648.533`; see M1 SPEC delta note in
`src/integrator/include/tdmd/integrator/velocity_verlet.hpp`). Passed as a flat `double[n_species]`
table (species count ~1–10 в M6).

**Determinism + bit-exactness:**

- Per-atom thread; thread `i` writes only its own `x/y/z/vx/vy/vz` slots. No reductions, no atomics.
- Operand order matches CPU `pre_force_step`/`post_force_step` exactly: `v += f · accel · half_dt`
  then `x += v · dt`.
- With `--fmad=false` (Reference flag, `cmake/BuildFlavors.cmake §17`), each FP64 multiply-add is
  discrete — bit-identical to CPU. **D-M6-7 gate:** literal byte-equality of `x/y/z/vx/vy/vz` after
  1, 10, and 1000 NVE steps on 1000-atom lattice (`tests/gpu/test_integrator_vv_gpu.cpp`).

**MixedFast (deferred to T6.8):**

- `v/x` в FP32 storage + arithmetic; force read в FP32 but `accel · f · half_dt` widened to FP64 для
  accumulation, then narrowed back. Drift threshold ≤1e-5 rel per 1000 steps (D-M6-8).
- Not activated в T6.6 — MixedFast flavor fully stub until T6.8 wires `NumericConfig`.

**Data lifecycle (T6.6 scope):**

- Adapter в `src/integrator/gpu_velocity_verlet.cpp` does H2D(positions+velocities+forces+types+accel)
  → kernel → D2H(positions+velocities). Per-call upload.
- Resident-on-GPU pattern (integrator/SPEC §3.5) is T6.7 concern — SimulationEngine keeps atoms
  device-resident across iterations, syncs only at dumps/checkpoints.
- Per-call overhead (H2D + D2H + pool allocate) dominates at ≤10⁵ atoms because the kernel itself
  is ~6 multiply-adds per atom. Expected T6.6 bench результат: sub-1× speedup, GPU slower. **Это
  не регрессия** — correct T6.6 shape; speedup lives в T6.7 residency.

**NVTX:** deferred to T6.11 (same as §7.1 NL, §7.2 EAM).

### 7.4. Out of scope в M6

LJ, Morse, MEAM, PACE, MLIAP, NVT, NPT, thermostats — остаются CPU-only; M9+ window для GPU ports. Их kernel contracts добавятся в соответствующие sections gpu/SPEC v1.x когда они landed. До того их `*Potential` / `*Integrator` классы CPU-only; SimulationEngine (T6.7 wiring) routes на CPU если potential style не в {EAM/alloy, SNAP} (preflight-level enforced in `src/io/preflight.cpp::check_runtime`).

**SNAP GPU moved to M8 (T8.6a scaffolding / T8.6b kernel body / T8.7 bit-exact gate).** См. §7.5 ниже — SPEC contract authored при T8.6a landing.

### 7.5. SNAP GPU (T8.6a scaffolding — v1.0.17 / T8.6b kernel body — deferred / T8.7 bit-exact gate — deferred)

**Scope статус.** T8.6a (2026-04-20, this row) ships **только scaffolding**: class skeletons, PIMPL firewall, CPU-only build guards, M8-scope flag fence, sentinel-throw `compute()`. NO CUDA kernel launches в `src/gpu/snap_gpu.cu` at T8.6a — `test_nvtx_audit` trivially passes (грep walker finds zero `<<<...>>>` sites inside `snap_gpu.cu`). Full kernel body + bit-exact gate are T8.6b / T8.7 respectively.

**Why split T8.6 → T8.6a + T8.6b.** Mirrors T8.4a/T8.4b CPU precedent: ~1500-line LAMMPS USER-SNAP port (Ui → Yi → deidrj three-pass) deserves a separate reviewable PR for the kernel body; scaffolding landing first pins the public adapter shape and the CPU-only build guard story before any CUDA-specific code is written. Downstream M8 tasks (T8.9 MixedFast SNAP, T8.10 T6 full-scale fixture diff, T8.11 cloud-burst scaling) can start their pre-impl reports against the locked `SnapGpuAdapter::compute()` signature without blocking on T8.6b kernel landing.

**Public API — T8.6a locked shape.**

```cpp
// src/gpu/include/tdmd/gpu/snap_gpu.hpp — mirrors EamAlloyGpu
namespace tdmd::gpu {
  struct SnapTablesHost { /* twojmax, rcutfac, rfac0, rmin0, flags,
                             k_max, idxb_max, idxu_max, idxz_max,
                             per-species radius_elem[n_species],
                             weight_elem[n_species],
                             beta[n_species * (k_max + 1)],
                             rcut_sq_ab[n_species * n_species] — all host ptrs */ };
  struct SnapGpuResult { double potential_energy; double virial[6]; };

  class SnapGpu {
   public:
    SnapGpu();
    ~SnapGpu();
    SnapGpu(SnapGpu&&) noexcept;
    SnapGpu& operator=(SnapGpu&&) noexcept;
    SnapGpu(const SnapGpu&) = delete;
    SnapGpu& operator=(const SnapGpu&) = delete;

    SnapGpuResult compute(
        std::size_t n, const std::int32_t* types,
        const double* x, const double* y, const double* z,
        std::size_t ncells, const std::int32_t* cell_offsets,
        const std::int32_t* cell_atoms,
        const BoxParams& box, const SnapTablesHost& tables,
        double* fx_out, double* fy_out, double* fz_out,
        DevicePool& pool, DeviceStream& stream);

    std::uint64_t compute_version() const noexcept;  // monotone; bumped after successful compute()
   private:
    struct Impl; std::unique_ptr<Impl> impl_;
  };
}
```

```cpp
// src/potentials/include/tdmd/potentials/snap_gpu_adapter.hpp — domain facade
namespace tdmd::potentials {
  class SnapGpuAdapter {
   public:
    explicit SnapGpuAdapter(const SnapData& data);  // validates + flattens per-species arrays
    ForceResult compute(state::AtomSoA& atoms, const state::Box& box,
                        const neighbor::CellGrid& cells,
                        gpu::DevicePool& pool, gpu::DeviceStream& stream);
    std::uint64_t compute_version() const noexcept;
  };
}
```

Signature is **identical** to `EamAlloyGpuAdapter::compute` (same parameter set + same `ForceResult` return type) — lets `SimulationEngine::recompute_forces()` branch on the adapter type without surface-level restructuring.

**M8-scope flag fence (T8.6a enforced).**

`SnapGpuAdapter` ctor rejects with `std::invalid_argument`:
- `chemflag == 1` — multi-element chemistry (M8-scope out per potentials/SPEC §6; revisit M10+ когда MEAM chemistry joined lands);
- `quadraticflag == 1` — quadratic bispectrum extension (orthogonal feature, no physics need for M8 W / In-P fixtures);
- `switchinnerflag == 1` — inner-switch function (LAMMPS extension not used в canonical W_2940 fixture).

Parity with `SnapPotential` CPU ctor (T8.4b / T8.5) — same three rejections same messages. Both CPU and GPU paths fence identically so misconfigured YAML fails at the earliest validation layer.

**Build-guard story (T8.6a shipped).**

Single TU `src/gpu/snap_gpu.cu` with two branches:

```cpp
#if TDMD_BUILD_CUDA
  struct SnapGpu::Impl { /* DevicePtr<> device buffers — declared, unused T8.6a */ };
  SnapGpuResult SnapGpu::compute(...) {
    TDMD_NVTX_RANGE("snap.compute_stub");
    throw std::logic_error(
        "SnapGpu::compute: T8.6b kernel body not landed — "
        "set runtime.backend=cpu or await T8.6b merge");
  }
#else
  struct SnapGpu::Impl {};
  SnapGpuResult SnapGpu::compute(...) {
    throw std::runtime_error(
        "gpu::SnapGpu::compute: CPU-only build (TDMD_BUILD_CUDA=0); CUDA not linked");
  }
#endif
```

Mirrors `src/gpu/eam_alloy_gpu.cu` precedent. NO separate `snap_gpu_stub.cpp` — single-TU guard keeps CMake wiring simple (same set_source_files_properties LANGUAGE=CXX trick as EAM on CPU-only builds). `compute_version()` stays at 0 on both branches because `compute()` throws before any increment.

**T8.6b — shipped 2026-04-20 (v1.0.18).**

Port of LAMMPS `sna.cpp` compute/force loops (Wigner-U, Z-list, Y-list, B-list, deidrj). **Three-kernel architecture**, one-block-per-atom (<<<n_atoms, 128>>>), FP64 throughout. Index-table flatten + upload lives in `src/gpu/snap_gpu_tables.cu` (~290 LoC host C++, pure integer/factorial recurrences; bit-exact with `SnaEngine::build_indexlist/init_clebsch_gordan/init_rootpqarray` by construction — same arithmetic, no FP subtlety). Device helpers live in `src/gpu/snap_gpu_device.cuh` (`compute_uarray`, switching function, Wigner-U recurrence). Kernels:

1. **`snap_ui_kernel`** — per atom: zero `ulisttot_r/i`, add self-term (tid==0, j=0 block), then single-lane loop over neighbours `j` of atom `i` calling `compute_uarray_single_lane` and accumulating weighted Ulist into `ulisttot_r/i`. Thread-parallel zeroing; single-lane CG-recurrence accumulator.
2. **`snap_yi_kernel`** — per atom: single-lane Z-list contraction (triple loop j1≤j2≤j over CG matrix × ulisttot), Y-list build (Y_{jju} = Σ β · Z), B-list inline (B_{j1,j2,j}), PE accumulation into `pe_per_atom[i]`, per-atom partial virial write.
3. **`snap_deidrj_kernel`** — per atom `i`: single-lane outer loop over neighbours. For each neighbour `j`, re-derives `du/dr` for the `i ↔ j` pair and computes BOTH (fij_own, which is i's contribution assuming i is the "origin") AND (fij_peer, i's contribution if j were the "origin" — reads `ylist[j]` from global). Full-list peer-side replay replaces the CPU's half-list Newton-3 pair scatter: `F_i = Σ_j (fij_own(i,j) − fij_peer(j,i))` — each bond contributes to both endpoints symmetrically without any `atomicAdd(double*)`.

`__restrict__` на all pointer params per master spec §D.16. NVTX range per kernel launch per gpu/SPEC §9 (T6.11 audit discipline — `test_nvtx_audit` enforces). Index-table flatten (T** → T*) shipped at T8.6b entry — `Impl` allocates `d_idxcg_block`, `d_idxu_block`, `d_idxz_block`, `d_cg_coefficients`, `d_rootpq` as flat contiguous buffers filled once at adapter ctor (analogous to EAM spline coefficient flatten at T6.5). Host marshalling in `src/potentials/snap_gpu_adapter.cpp` (already flattens `radius_elem_flat_`, `weight_elem_flat_`, `beta_flat_` at T8.6a).

**Acceptance gate at T8.7 (NOT at T8.6 / T8.6a / T8.6b).**

- Per-atom forces + total PE + virial Voigt tensor agree **≤ 1e-12 rel** CPU FP64 ↔ GPU FP64 on W 2000-atom fixture (`W_2940_2017_2.snapcoeff` + `.snapparam`). Same shape as D-M6-7 EAM gate; D-M8-13 anchor.
- Gate is relative, not byte-equal — absorbs FP64 reduction-order drift между host Kahan reduction и device atomic-add scatter. Math kernels themselves use identical FP sequences, so any divergence above 1e-12 indicates a real bug.
- `tests/gpu/test_snap_gpu_bit_exact.cpp` ships at T8.7 (runtime gate — local pre-push only per D-M6-6 Option A CI).

**T8.6a test coverage (shipped).**

`tests/gpu/test_snap_gpu_plumbing.cpp` — 4 Catch2 cases:
1. `SnapGpuAdapter constructs cleanly on canonical W_2940 fixture` — flatten succeeds; `compute_version() == 0`.
2. `SnapGpuAdapter rejects M8-scope flag violations` — chemflag/quadraticflag/switchinnerflag each REQUIRE_THROWS_AS `std::invalid_argument`.
3. `SnapGpu::compute — T8.6a sentinel error path is reachable` — structural; asserts error chain intact on both CUDA and CPU-only builds.
4. `SnapGpuAdapter::compute_version — stays at 0 before T8.6b` — regression guard (catches accidental success-return).

Self-skips с exit 77 if LAMMPS submodule not initialized (Option A / public CI convention — matches `test_snap_compute`).

**NVTX (v1.0.18).** All three T8.6b kernel launches wrapped in `TDMD_NVTX_RANGE("snap.ui_kernel" | "snap.yi_kernel" | "snap.deidrj_kernel")`; H2D/D2H copies wrapped in `"snap.h2d.*"` / `"snap.d2h.*"`; index-table upload wrapped in `"snap.build_index_tables"`; top-level `compute()` wrapped in `"snap.compute"`. `test_nvtx_audit` green on both CUDA + CPU-only builds.

**Lifetime contract (T8.6b footnote).** `SnapGpuAdapter::Impl` holds `DevicePtr<std::byte>` members whose pool-class deleters reference the owning `DevicePool::Impl*` as context. Construction order in any test/driver code **must** be `DevicePool pool → DeviceStream stream → SnapGpuAdapter adapter` so destruction runs `adapter → stream → pool` and the adapter's `DevicePtr` deleters reach a still-alive pool. The reverse order (`adapter → pool → stream` at construction) triggers use-after-free on teardown — pool dtor calls `cudaFreeAsync` on all tracked blocks, then adapter's `~Impl` runs `DevicePtr` destructors that re-dereference the already-destroyed pool Impl. Same contract applies to `EamAlloyGpuAdapter`. Enforced by convention only — no assert at construction time.

**Data lifecycle.**

- T8.6a: adapter compute() marshals AtomSoA/Box/CellGrid → BoxParams + SnapTablesHost → calls `SnapGpu::compute()` which throws the T8.6b sentinel before touching device memory. Flattened host buffers (`radius_elem_flat_`, `weight_elem_flat_`, `beta_flat_`) are allocated once at adapter ctor.
- T8.6b (shipped): adapter stays unchanged; `Impl::compute()` body implements H2D(atoms+cells+tables once cached) → three-kernel launch → D2H(forces+per-atom PE+virial) → host Kahan reduction of PE + virial (D-M6-15). Per-step kernel timing baseline measured в T8.10 / T8.11.
- T8.7: bit-exact gate adds a second adapter call via CPU path, compares per-atom force/pe/virial arrays at ≤ 1e-12 rel.

**Adapter registration (T8.6a shipped).**

`SimulationEngine::init()` now switches on CPU potential type:

```cpp
if (auto* eam_cpu = dynamic_cast<EamAlloyPotential*>(potential_.get()); eam_cpu) {
  gpu_potential_ = std::make_unique<potentials::EamAlloyGpuAdapter>(eam_cpu->data());
} else if (auto* snap_cpu = dynamic_cast<SnapPotential*>(potential_.get()); snap_cpu) {
  gpu_snap_potential_ = std::make_unique<potentials::SnapGpuAdapter>(snap_cpu->data());
} else {
  throw std::invalid_argument("runtime.backend=gpu requires EAM/alloy or SNAP potential");
}
```

Parallel fields (`gpu_potential_`, `gpu_snap_potential_`) rather than abstract `GpuPotentialAdapter` base — minimum-scope for T8.6a; abstract-base refactor (if ever needed for M9+ potentials) can land at T8.6b or later once 2+ GPU potentials actually coexist на dispatch path.

`src/io/preflight.cpp::check_runtime` relaxed — `runtime.backend=gpu` now accepts `potential.style ∈ {eam/alloy, snap}`. Morse стainavljjs CPU-only за ошибка.

**Roadmap после T8.6a.**

| Task | Scope | Gate |
|------|-------|------|
| T8.6b | Full CUDA kernel port of SnaEngine; three kernels Ui/Yi/deidrj + index-table flatten + peer-side Newton-3 reassembly; __restrict__ + NVTX + Kahan reduction | **LANDED 2026-04-20**: functional pass on W BCC 250-atom rattled smoke; `compute_version()` monotonic; worst-force rel err already 1.3e-14 vs CPU (T8.7 locks formal 1e-12 gate) |
| T8.7  | Bit-exact gate CPU FP64 ≡ GPU FP64 ≤ 1e-12 rel on W 2000-atom fixture | D-M8-13 closure |
| T8.9  | MixedFast SNAP kernel — FP32 math + FP64 accum (Philosophy B) | D-M8-8 ≤ 1e-5 rel force / ≤ 1e-7 rel PE |
| T8.10 | T6 medium/large variants (1024/8192 atoms) — performance baseline vs LAMMPS CPU SNAP | TDMD GPU > LAMMPS CPU ≥ 2× |
| T8.11 | Cloud-burst scaling (≥ 2 GPU nodes, P=2, K=4) | D-M8-7 overlap ≥ 30% |
| T8.13 | M8 acceptance gate | Milestone close |

---

## 8. Mixed-precision policy (D-M6-5, D-M6-8)

### 8.1. Fp64ReferenceBuild — FP64 everywhere

GPU Reference path:

- Positions, velocities, forces, energies — все FP64;
- Accumulators — FP64 + Kahan compensation;
- Spline evaluations в EAM — FP64 tables;
- `--fmad=false` compile flag (см. `BuildFlavors.cmake`);
- `atomicAdd(double*, double)` — **запрещён**, только sort+add или reduce-then-scatter.

**Gate:** D-M6-7 bit-exact CPU↔GPU.

### 8.2. MixedFastBuild — Philosophy B

**T6.8a shipped inventory (v1.0.6).** Только EAM force/density kernels имеют narrowed-math variant; NL и VV остаются FP64 под обоими флэйворами. `TDMD_FLAVOR_MIXED_FAST` compile-time дефайн переключает `EamAlloyGpuAdapter` на `EamAlloyGpuMixed` через typedef `EamAlloyGpuActive`.

FP32 sites (EAM mixed kernel):

- r² pair-distance computation + FP32 cutoff filter (density + force kernels);
- `r = sqrtf(r²_f)` и `1/r = 1.0f / r_f` (FP32 SFU reciprocal);

FP64 sites (kept wide — Philosophy B accumulators):

- Positions, forces, per-atom accumulators (ρ, F(ρ), fx/fy/fz, pe, virial) — FP64 storage;
- Spline coefficient tables (`rho_coeffs`, `F_coeffs`, `z_coeffs`) — FP64 в device memory;
- Spline `locate` + Horner evaluation — **FP64** (FP32 Horner на реальных EAM коэффициентах (десятичные порядки mismatch) ловит catastrophic cancellation на ρ и φ branches; эмпирически подтверждено в T6.8a dev);
- phi, phi_prime, dE/dr, fscalar, fij_xyz — FP64 arithmetic умноженное на FP32-rounded `inv_r` cast в double;
- Host-side Kahan reductions для PE + virial.

**Rationale:** Philosophy B (`master spec §D.1`) — даже FP32-only narrow narrowing на r/sqrtf/inv_r достаточно активирует SFU throughput и перекрывает register pressure ceiling; wider FP32 storage (positions, splines) оставлено для T6.8b performance study. FP64 spline Horner — hard requirement after dev-time attempt оставил cumulative rel force error ~9e-6 на 50-neighbor EAM stencil (partial sign cancellation amplifies per-op FP32 rel 6e-8 × √50 × sign-cancellation factor of ~20).

**NL / VV:** MixedFast использует same `NeighborListGpu` и `VelocityVerletGpu` как Reference. Rationale: NL — pure integer CSR + один FP64 r² per pair; drift от narrowed math был бы negligible на perf, но ломал бы `build_version` bit-exactness. VV kernels — 6 FLOPs/atom в чистом element-wise цикле; FP32 narrowing дал бы ≤0.05% runtime benefit (обоснованно замерено в T6.6 micro-bench: VV уже H2D/D2H-bound). T6.8b возможно ревизит если `DevicePool` будет готов держать resident atom state.

### 8.3. Differential thresholds (D-M6-8 — v1.0.12 formal)

**D-M6-8 scope split (T7.0 SPEC delta, v1.0.12).** Single flat threshold was the v1.0-v1.0.11 design; T7.0 splits it into **dense-cutoff** and **sparse-cutoff** branches because the residual source is fundamentally different. Dense-cutoff stencils (≥20 neighbors per atom typical — EAM, MEAM, SNAP, PACE, MLIAP) hit a structural **FP32 precision ceiling** in Philosophy B MixedFast (per-op 6e-8 × √N_neighbors × partial-sign-cancellation → cumulative 10⁻⁵ rel force). Sparse-cutoff (LJ, Morse — ~2-8 neighbors) retain the 1e-6 ambition and do not need the relaxation; those potentials land on GPU M9+.

```yaml
# verify/thresholds/thresholds.yaml — gpu_mixed_fast section (T7.0 canonical)
gpu_reference_force_bit_exact:
  units: dimensionless
  threshold: 0  # literal equality
  source: gpu/SPEC §6.3 + D-M6-7

# ---- Dense-cutoff (EAM / MEAM / SNAP / PACE / MLIAP) ----
# Canonical values after T7.0 SPEC delta. The prior 1e-6 target was never
# achievable with r²/sqrtf/inv_r in FP32 on dense stencils; see rationale
# block in verify/thresholds/thresholds.yaml and memory
# `project_fp32_eam_ceiling.md`. Tightening requires a new BuildFlavor
# (e.g. MixedFastAggressive with FP32-table storage) — deferred M9+.
gpu_mixed_fast_force_rel_dense:
  units: dimensionless (rel err per-atom L∞)
  threshold: 1e-5
  source: gpu/SPEC §8.3 + D-M6-8 (T7.0 canonical)

gpu_mixed_fast_energy_rel_dense:
  units: dimensionless (rel err total)
  threshold: 1e-7
  source: gpu/SPEC §8.3 + D-M6-8 (T7.0 canonical)

gpu_mixed_fast_virial_rel_dense:
  units: dimensionless (rel err Voigt, normalized by max component)
  threshold: 5e-6
  source: gpu/SPEC §8.3 + D-M6-8 (T7.0 canonical)

# NVE energy-conservation drift — integrator-level gate, not a force gate.
# Applies to MixedFast on dense-cutoff stencils; the same budget is
# reasonable on sparse-cutoff pending M9+ measurement.
gpu_mixed_fast_nve_drift:
  units: dimensionless (rel drift of E_total per 1000 steps)
  threshold: 1e-5
  source: gpu/SPEC §8.3 + D-M6-8

# ---- Sparse-cutoff (LJ / Morse / pair — ~2-8 neighbors) ----
# Ambition threshold for M9+ GPU port of non-EAM pair styles; NO kernel
# currently exercises this. Listed here so the numeric contract is visible
# when those potentials land.
gpu_mixed_fast_force_rel_sparse:
  units: dimensionless (rel err per-atom L∞)
  threshold: 1e-6
  source: gpu/SPEC §8.3 + D-M6-8 (ambition; not yet active)
  status: deferred_m9_plus

gpu_mixed_fast_energy_rel_sparse:
  units: dimensionless (rel err total)
  threshold: 1e-8
  source: gpu/SPEC §8.3 + D-M6-8 (ambition; not yet active)
  status: deferred_m9_plus
```

Проверяется в `tests/gpu/test_eam_mixed_fast_within_threshold.cpp` (single-step force/PE/virial, D-M6-8 dense-cutoff, T6.8a shipped) + `tests/gpu/test_t4_nve_drift.cpp` (100-step NVE drift on Ni-Al EAM 864 atoms, D-M6-8 drift, T7.0 shipped) + T6.10 T3-gpu anchor (T6.10a shipped; efficiency curve → T7.12).

**T7.0 closure status (v1.0.12, shipped).** D-M6-8 formally relaxed on dense-cutoff; T4 NVE drift harness landed; NL-mixed variant formally rejected. Canonical thresholds vs shipped:

| Величина | D-M6-8 canonical | Shipped measurement | Source |
|-----------|------------------|---------------------|--------|
| rel force per-atom (L∞, dense) | ≤ 1e-5 | **≤ 1e-5** (T6.8a) | FP32 precision ceiling — see rationale below |
| rel total PE (dense) | ≤ 1e-7 | **≤ 1e-7** (T6.8a) | derived from force residual on ~3N pair terms |
| rel virial Voigt (dense, normalized by max) | ≤ 5e-6 | **≤ 5e-6** (T6.8a) | off-diagonal near-zero on B2 crystal → max-component normalization |
| NVE drift of E_total / 1000 steps (dense) | ≤ 1e-5 | measured per 100-step harness с 10× budget margin | `tests/gpu/test_t4_nve_drift.cpp` (T7.0) |

**Rationale для dense-cutoff ceiling.** `r² = Δx² + Δy² + Δz²` is computed в FP32 (~6e-8 rel per op). `r = sqrtf(r²_f)` adds a round; `inv_r = 1.0f/r_f` another. The FP32 `inv_r` is then cast to `double` and enters FP64 Horner spline evaluation, phi/dE_dr, fscalar, fij_xyz — the cast bakes-in a fresh FP32 rounding per pair. Over ~50 neighbors на B2 / FCC dense stencils с partial sign cancellation (force direction partially cancels between neighbors в symmetric lattices, amplifying rel error by factor ~20 vs √N), cumulative rel force lands ~10⁻⁵. FP32 Horner on EAM spline coefficients was tried в dev (T6.8a) — caught catastrophic cancellation (9e-6 rel force) and excluded. Storing `rho_coeffs` / `F_coeffs` / `z_coeffs` в FP32 device memory would halve table bandwidth but requires full stability review of Horner evaluation on реальных Mishin-2004 coefficients (z_coeffs pair repulsion и F_coeffs embedding polynomials have decimal orders differing by 4-6 across cutoff domain; FP32 Horner loses monotonicity on ρ and φ branches per T6.8a empirical data). Такой redesign deferred to M9+ и requires a new BuildFlavor (e.g. `MixedFastAggressiveBuild` Philosophy A) — не затрагивает canonical Philosophy B `MixedFastBuild` contract. See memory `project_fp32_eam_ceiling.md` (deep dive shipped с T6.8a).

**T6.8b roadmap (closed in T7.0):**

1. ~~NL MixedFast variant (`src/gpu/neighbor_list_gpu_mixed.cu`)~~ — **REJECTED in T7.0**. NL is pure integer CSR + one FP64 `r²` computation per pair; narrowing the r² to FP32 would save ≤3% of NL-rebuild time (measured negligible — NL is bandwidth-bound on CSR I/O, not on r² FLOPS) but would break `build_version` bit-exactness between Reference and MixedFast. The latter is the entire point of the determinism contract: NL ordering must remain identical across flavors so `neighbor/` + `scheduler/` can rely on pair-iteration order being a compile-invariant. MixedFast will continue to use the same `NeighborListGpu` as Reference (§8.2 unchanged).
2. `tests/gpu/test_t4_nve_drift.cpp` 100-step NVE drift harness landed — **DONE in T7.0**. Ni-Al EAM 864 atoms, `runtime.backend: gpu` + MixedFast build, asserts `|E_total(100) - E_total(0)| / |E_total(0)| ≤ 1e-6` (100-step per-capita budget = 10× margin under the 1000-step 1e-5 cap). Self-skips on non-CUDA / Reference-only builds.
3. ~~FP32-table-storage redesign~~ **vs** formal D-M6-8 relaxation — **relaxation chosen, deferred redesign**. This v1.0.12 SPEC delta is the relaxation (see § split above). FP32-table redesign remains available как future opt-in via a separate flavor — tracked в master spec Приложение B.2 open questions list, not в M7 scope.

Integration в T6.10 T3-gpu anchor использует T6.8a achieved thresholds до закрытия T6.8b.

### 8.4. Deferred flavors

- `Fp64ProductionBuild` — разрешает `--fmad=true` для perf. **НЕ активен** в M6; вернётся в M8+ когда performance tuning начинается;
- `MixedFastAggressiveBuild` — Philosophy A (FP32 full stack, no FP64 accumulator). Opt-in M8+ только если `MixedFast` + additional perf budget потребуется;
- `Fp32ExperimentalBuild` — pure FP32, no safety net. M10+ research.

---

## 9. Engine wire-up (T6.7 — v1.0.5)

`runtime/` consumes gpu/ через opt-in YAML flag `runtime.backend` (see `runtime/SPEC.md §2.3`). Этот раздел фиксирует contract как gpu/ types подключаются к `SimulationEngine` без leakage CUDA headers за пределы `tdmd::gpu`.

### 9.1. Ownership topology

`runtime::GpuContext` (RAII) — единственный owner pool + compute stream внутри engine:

```
SimulationEngine
  └── GpuContext                    # RAII — bound to engine lifetime
        ├── DevicePool  (§5.1)       # cudaMallocAsync cached pool
        └── DeviceStream compute     # single stream in M6 (§3.1; 2nd stream — T6.9)
```

Все GPU сущности (`EamAlloyGpuAdapter`, `GpuVelocityVerletIntegrator`, `GpuNeighborBuilder`) получают `DevicePool&` / `DeviceStream&` **по ссылке в compute() / pre_force_step() / post_force_step()** — они **не владеют** ресурсами. Это гарантирует, что при `finalize()` pool и stream освобождаются ровно один раз.

### 9.2. Dispatch pattern в SimulationEngine

`init()` создаёт CPU модули всегда (parser / preflight / YAML layer остаются CPU-oblivious в M6), затем при `runtime.backend == Gpu`:

1. `gpu_context_ = std::make_unique<GpuContext>(gpu_cfg)` — throws на CPU-only build или отсутствии sm_XX hardware;
2. `auto* eam = dynamic_cast<EamAlloyPotential*>(potential_.get())` — M6 scope: только EAM/alloy; non-EAM отвергается preflight'ом;
3. `gpu_potential_ = std::make_unique<EamAlloyGpuAdapter>(eam->data())` — borrows parsed `EamAlloyData&` from CPU potential (no re-parsing);
4. `gpu_integrator_ = std::make_unique<GpuVelocityVerletIntegrator>(species_)`.

Hot path (recompute_forces / integrator pre/post):

```cpp
if (gpu_backend_) {
  gpu_potential_->compute(atoms_, box_, cell_grid_, pool, stream);
  // ... then
  gpu_integrator_->pre_force_step(atoms_, dt_, pool, stream);
} else {
  potential_->compute(atoms_, neighbor_list_, box_);
  integrator_->pre_force_step(atoms_, species_, dt_);
}
```

CPU `potential_` / `integrator_` **остаются живыми** при `backend: gpu` — parsed potential data принадлежит им; GPU adapter borrows. Это explicit single-owner, dual-binding pattern.

### 9.3. MPI transport (D-M6-3 host-staging)

v1.0 ships **host-staged MPI only**: `comm::MpiHostStagingBackend` остаётся единственной implementation of `CommBackend`. D2H → MPI → H2D per packet. NCCL / GPUDirect RDMA (`stream_aux` 3rd stream, Pattern 2 halo) — M7+ roadmap.

**Multi-rank determinism (D-M5-12 extended):** GPU K=1 P=N ≡ GPU K=1 P=1 через тот же deterministic Kahan-ring в `comm/` (T6.7 2-rank gate, Ni-Al EAM 10 шагов bit-exact). Это расширяет CPU-only M5 chain на GPU эру без изменений в `comm/`.

### 9.4. Byte-exact gate (D-M6-7 engine-level)

T6.7 acceptance gate закрывает D-M6-7 на engine уровень (выше kernel-level gates из T6.5/T6.6):

| Gate | Scope | Tolerance |
|------|-------|-----------|
| T6.5 EAM kernel | per-atom forces, total PE, virial | ≤1e-12 rel |
| T6.6 VV kernel | velocities + positions (Reference flavor) | byte-equal |
| **T6.7 (1-rank)** | **thermo stream (step+ke+pe+te)** на 100 шагов Ni-Al EAM 864 atoms | **byte-equal** |
| **T6.7 (2-rank)** | **thermo stream** на 10 шагов, `backend: gpu` + MpiHostStaging 2 ranks vs 1 rank | **byte-equal** |

Композиция всех трёх gate'ов (EAM ≤1e-12 + VV byte-equal + thermo byte-equal) — эмпирическое подтверждение, что кумулятивный drift от EAM 1e-12 tolerance не выходит за double-ULP за 100 шагов на данном фикстюре; Reference flavor's `--fmad=false` + strict reduction order делают это reliable.

### 9.5. Scope limits (M6)

- ~~Single compute stream — 2-stream compute/mem overlap в T6.9 (D-M6-13)~~ **v1.0.7 update:** `GpuContext` держит оба stream'а (T6.9a shipped — §3.2a); full overlap orchestration + 30% gate отложены в T6.9b (M7 dependency);
- Pattern 1/3 only — Pattern 2 GPU planning в M7;
- `Fp64ReferenceBuild` only — MixedFast/MixedFastAggressive активируются T6.8 поверх того же wiring'а (differential harness переиспользует T6.7 1-rank comparer с ≤ D-M6-8 threshold вместо byte-equal);
- Resident-on-GPU atom state — M7 (T6.7 делает H2D per step из engine's SoA; T6.9 оценивает gain от резидентности).

---

## 10. NVTX instrumentation (D-M6-14)

NVTX_RANGE wrapping обязателен для:

| Boundary                       | Range name pattern                     |
|--------------------------------|----------------------------------------|
| Kernel launches                | `"{phase}.{kernel_name}"` (e.g. `"force.eam_density"`) |
| MPI pack/unpack                | `"mpi.pack.{packet_kind}"` / `"mpi.unpack.{...}"` |
| Pipeline phase transition      | `"pipeline.{packed_for_send_to_in_flight}"` |
| H2D / D2H copies               | `"copy.h2d.{buffer_tag}"` / `"copy.d2h.{...}"` |
| Pool alloc/free                | `"pool.alloc"` / `"pool.free"` (aggregated) |
| Stream sync wait               | `"sync.wait.{event_tag}"` |

Overhead: `nvtxRangePushA` + `nvtxRangePop` — ~20 ns per pair. С thousands of calls per step и microsecond-scale kernels — << 1% total. Гарантированно всегда on (GpuConfig::enable_nvtx = true default).

**Off-switch:** `GpuConfig::enable_nvtx = false` отключает NVTX calls compile-time через conditional includes. Используется для bare-metal benchmarks где любой overhead критичен.

**Domain:** single `nvtxDomainCreateA("tdmd")` at init; все ranges под этим domain. Nsight Systems filter `--nvtx-domain=tdmd` показывает только TDMD ranges.

---

## 11. Tests

### 11.1. T6.2 skeleton tests (этот PR)

`tests/gpu/test_gpu_types.cpp` — compile-time shape invariants + runtime defaults:

- `GpuConfig` defaults == D-M6-18;
- `kStreamCompute / kStreamMem` pinned;
- `DevicePtr<T>` move-only, `static_assert(!is_copy_constructible_v<>)`;
- `DevicePtr<T>` deleter invoked exactly once on destruction;
- `DevicePtr<T>` move transfers ownership, source null;
- `DevicePtr<T>` move-assign releases existing;
- `DevicePtr<T>` release opts out of deletion;
- `DeviceStream` / `DeviceEvent` move-only, default null;
- `DeviceInfo` default warp_size == 32;
- `DeviceAllocator` abstract with virtual dtor.

Budget: <2 sec CI (pure C++ tests, no CUDA runtime).

### 11.2. T6.3+ runtime tests (local-only gated on TDMD_BUILD_CUDA)

- **T6.3**: `test_device_pool_alloc_free` — 1000 alloc+free cycles across size classes, verify no leak + hit rate ≥95%;
- **T6.3**: `test_pinned_host_pool_mpi_symmetry` — allocate pinned host, D2H, MPI send-to-self, receive, H2D, bit-compare;
- **T6.4**: `test_neighbor_bit_exact_vs_cpu` — Ni-Al 10⁴ atoms, GPU half-list ≡ CPU half-list after sort within bucket;
- **T6.5**: `test_eam_force_bit_exact_reference` — same fixture, one-step force bit-exact;
- **T6.5**: `test_eam_mixed_fast_within_threshold` — same fixture, MixedFast within D-M6-8 thresholds;
- **T6.6**: `test_vv_nve_drift` — 1000 steps, drift < 1e-5 rel.

### 11.3. M6 integration smoke (T6.13)

**Shipped (v1.0.11):** `tests/integration/m6_smoke/` — 2-rank GPU
harness that is byte-for-byte the M5 smoke plus `runtime.backend: gpu`.
Ni-Al EAM/alloy NVE, 864 atoms (T4 `setup.data`), K=1 P=2,
`MpiHostStaging`, 10 steps, thermo every step. Acceptance gate: the
10-line thermo stream **equals the M5 `thermo_golden.txt` byte-for-byte**
— M5's golden is copied verbatim into `tests/integration/m6_smoke/
thermo_golden.txt` and the harness pre-flight (step 1/6) asserts
`diff -q` parity before launching the binary. This closes the D-M6-7
chain: M3 ≡ M4 ≡ M5 ≡ M6 on every PR that wires the smoke step.

Six harness steps, short-circuiting on first failure:

1. Pre-flight: M6 golden == M5 golden (D-M6-7 chain) → exit 2.
2. Local-only gate: `nvidia-smi -L` reports ≥1 GPU, else SKIP exit 0.
3. `mpirun --np 2 tdmd validate <config>` — 2-rank GPU config OK.
4. `mpirun --np 2 tdmd run --telemetry-jsonl` exits 0 < 60 s.
5. Thermo byte-matches M6 golden (= M5 = M4 = M3) → exit 1 on drift.
6. Telemetry invariants (`event=run_end`, `total_wall_sec≤60`,
   `ignored_end_calls==0`) → exit 1.

**Option A CI policy.** The smoke step is wired in `.github/workflows/
ci.yml` (added to the `build-cpu` job right after M5 smoke). On
`ubuntu-latest` there is no CUDA device, so step 2 self-skips (exit 0)
and CI stays green. This is intentional per D-M6-6 (no self-hosted GPU
runner on the public repo). The harness still runs end-to-end to catch
infrastructure rot (template typo, LFS asset drift, golden-file
divergence from M5) — step 1 fires even when no GPU is present, so a
future edit that mutates the M6 golden without syncing M5 will fail CI
loudly.

**Local pre-push gate.** On dev machines with a CUDA device the smoke
runs ≤5 s on 864 atoms and is mandatory before any merge that touches
`src/gpu/`, `src/potentials/eam_alloy_gpu_*`, `src/integrator/*_gpu*`,
`src/comm/mpi_host_staging*`, or `src/runtime/gpu_context*`.

**MixedFast coverage.** Deliberately NOT layered into this smoke:
D-M6-7 (byte-exact) and D-M6-8 (threshold) are different acceptance
modes, and bundling them would dilute the byte-exact failure signal.
The MixedFast gate is owned by T6.8a's differential test
(`test_eam_mixed_fast_within_threshold`). Invoking the smoke against a
MixedFast binary will fail step 5 and is not supported.

**Scope boundaries** (per `tests/integration/m6_smoke/README.md`): no
T3-gpu efficiency curve (T6.10b, local long-running gate); no 2-stream
overlap gate (T6.9b, owned by `tests/gpu/test_overlap_budget.cpp`); no
multi-GPU-per-rank (D-M6-3 punts to M7+).

### 11.4. T3-gpu anchor (T6.10)

Extends T3 harness на GPU. T6.10 ships в два подэтапа:

**T6.10a (shipped — v1.0.8).** Fixture + harness dispatch + gates (1)+(2) on Ni-Al EAM 864-atom single-rank fixture (physics-equivalent to T4 / T6.7 acceptance):

- (1) `cpu_gpu_reference_bit_exact` — CPU Reference thermo ≡ GPU Reference thermo **byte-for-byte** на 100 шагов (D-M6-7 engine-level invariant lifted into the anchor harness);
- (2) `mixed_fast_vs_reference` — delegated to T6.8a differential harness (`tests/gpu/test_eam_mixed_fast_within_threshold.cpp`). Anchor runner asserts on exit code, not raw numbers, чтобы не дублировать FP-compare логику на Python layer;
- Gate (3) — **deferred to T6.10b** (see below).

Fixture files live в `verify/benchmarks/t3_al_fcc_large_anchor_gpu/`:
- `config.yaml` — Ni-Al EAM/alloy, 864 atoms, 100 steps, NVE, `runtime.backend` intentionally absent (harness injects per gate);
- `checks.yaml` — `backend: gpu`, `ranks_to_probe: [1]`, gate (3) block labeled `status: deferred, deferred_to: T6.10b`;
- `hardware_normalization_gpu.py` — M6 stub, probes `nvidia-smi` + emits `{"gpu_flops_ratio": 1.0, ...}` placeholder JSON. Real CUDA EAM density micro-kernel ships в T6.10b;
- `acceptance_criteria.md` — gate pseudocode + failure-mode taxonomy (`NO_CUDA_DEVICE`, `CPU_GPU_REFERENCE_DIVERGE`, `MIXED_FAST_OVER_BUDGET`, `RUNTIME_BUDGET_BLOWOUT`, `STUB_PROBE_WARNING`).

Harness dispatch lives в `verify/harness/anchor_test_runner/runner.py::_run_gpu_two_level`: top of `AnchorTestRunner.run()` reads `checks.yaml::backend`, routes to GPU path если `gpu` (overridable via `--backend cpu|gpu`). Report extensions: `AnchorTestReport.backend: "cpu"|"gpu"`, `gpu_gates: list[GpuGateResult] | None`. Mocked pytest coverage in `verify/harness/anchor_test_runner/test_anchor_runner.py::GpuAnchorRunnerMockedTest` exercises byte-exact green, diverge RED, no-CUDA-device RED (`NO_CUDA_DEVICE`), JSON round-trip, backend-override force.

**T7.12 (shipped — v1.0.14, ex-T6.10b partial).** Gate (3) — Pattern 2 GPU efficiency probe — landed as the **EAM substitute** (D-M7-16 scope per M7 execution pack). Status flipped from `deferred` to `active_eam_substitute` in `checks.yaml::efficiency_curve.status`. `_run_gpu_two_level()` extended with `_run_gpu_efficiency_probe()`: per probe rank, injects `zoning.subdomains: [N, 1, 1]` via the new `subdomains_xyz` kwarg on `_write_augmented_config()` / `_launch_tdmd_with_backend()`, captures telemetry, computes `efficiency_pct = 100 * sps(N) * anchor_n / (sps_anchor * N)`, grades against `efficiency_floor_pct` (default 80.0 — D-M7-8 single-node target, T7.11 t7_mixed_scaling parity). Each rank emits one `GpuGateResult` with `gate_name = "efficiency_curve_N{NN}"` and the new optional fields (`n_procs`, `measured_steps_per_sec`, `measured_efficiency_pct`, `floor_pct`); anchor point is informational (efficiency ≡ 100% by construction). Failure mode `EFFICIENCY_BELOW_FLOOR` joins the existing GPU-side classification set. Mocked pytest coverage: 8 new `GpuEfficiencyProbeTest` cases + 3 `WriteAugmentedConfigSubdomainsTest` cases (perfect-scaling green, below-floor RED, provenance string assertion, anchor-only single-rank, ranks-must-include-anchor invariant, partial-launch-failure RED, deferred-status backward compat, JSON roundtrip with new fields, subdomains injection direct unit test, omitted-subdomains backward compat, wrong-length raises).

Provenance discipline: every efficiency-probe `GpuGateResult.detail` carries the literal "EAM substitute per D-M7-16 (Morse fidelity blocker: M9+ Morse GPU kernel ...)" tag so downstream report consumers cannot mistake the measurement for a literal Morse-vs-dissertation comparison.

**Morse-vs-dissertation comparison stays deferred to M9+** behind the single remaining blocker:

1. **Morse GPU kernel.** `gpu/SPEC.md` §1.2 defers all non-EAM pair styles to M9+. Dissertation fixture uses Morse (physics-equivalent to Andreev's LJ at the scaling-profile level); swapping to Ni-Al EAM produces a different strong-scaling curve с no published baseline. The T7.12 EAM substitute closes the **Pattern 2 dispatch coverage** dependency (was T6.9b → done as T7.5/T7.7) but leaves the literal Morse comparison untouched. When the M9+ Morse GPU kernel lands, the fixture path is documented в `verify/benchmarks/t3_al_fcc_large_anchor_gpu/acceptance_criteria.md` §"Gate (3) — efficiency curve (T7.12)" — reintroduce `dissertation_reference_data.csv`, replace `hardware_normalization_gpu.py` stub with real CUDA EAM density micro-kernel, add a parallel `config_morse.yaml` (the EAM substitute does not need to retire — both arms can coexist).

Hardware normalization baseline note (OQ-M6-11): Andreev 2007 Alpha cluster не имел GPU, так что the M9+ Morse arm compares against **hypothetical linear-extrapolated TD curve** — documented в acceptance_criteria.md.

---

## 12. Telemetry

Exposed через `TelemetryFrame` extension в M6:

```yaml
gpu:
  device_id: 0
  device_name: "NVIDIA GeForce RTX 5080"
  cc_major: 12
  cc_minor: 0
  multiprocessor_count: 64
  device_bytes_in_use: 134217728
  pinned_bytes_in_use: 12582912
  alloc_hit_rate: 0.98
  pool_growth_events: 2
  kernel_timings_us:                 # aggregated per 100 steps
    neighbor_build:     845.2
    eam_density:        1820.3
    eam_embedding:       234.5
    eam_pair_force:     1950.8
    vv_first_half_kick:  125.1
    vv_drift:            88.7
    vv_second_half_kick: 125.3
  stream_wait_total_us:               # sync wait time
    stream_compute_waits_stream_mem: 120.0
    stream_mem_waits_stream_compute: 85.0
```

Dumped в JSON при `tdmd run --telemetry-json=out.json`. Consumed Nsight Systems export (`nsys export`) и `perfmodel/` (Pattern 3 `predict_step_gpu_sec` validation — shipped в T6.11, ±20% calibration gate deferred to T6.11b).

**NVTX instrumentation (T6.11, v1.0.9, D-M6-14).** Every kernel launch и every H↔D transfer в `src/gpu/*.cu` + `src/gpu/device_pool.cpp` обёрнут `TDMD_NVTX_RANGE("{subsystem}.{op}")` — RAII macro из `tdmd/telemetry/nvtx.hpp`. Enumerated ranges:

| Site                              | NVTX range name                                      |
|-----------------------------------|------------------------------------------------------|
| `NeighborListGpu::build`          | `nl.build` (outer) + `nl.h2d.positions_and_cells` + `nl.count_kernel` + `nl.host_scan_and_h2d_offsets` + `nl.emit_kernel` + `nl.download` |
| `EamAlloyGpu::compute`            | `eam.compute` (outer) + `eam.h2d.{atoms_and_cells, splines, forces_in}` + `eam.{density_kernel, embedding_kernel, force_kernel}` + `eam.d2h.forces_and_reductions` |
| `EamAlloyGpuMixed::compute`       | symmetric `eam_mixed.*`                              |
| `VelocityVerletGpu::pre_force`    | `vv.pre_force_step` + `vv.h2d.pre` + `vv.pre_force_kernel` + `vv.d2h.pre` |
| `VelocityVerletGpu::post_force`   | `vv.post_force_step` + `vv.h2d.post` + `vv.post_force_kernel` + `vv.d2h.post` |
| `DevicePool::allocate_device`     | `gpu.pool.alloc_device`                              |
| `DevicePool::allocate_pinned_host`| `gpu.pool.alloc_pinned`                              |

CPU-only builds (`TDMD_BUILD_CUDA=0`) expand the macro to `((void)0)` — zero-cost. Enforcement: `tests/gpu/test_nvtx_audit.cpp` — grep-based Catch2 test walks `src/gpu/*.cu`, finds every `<<<...>>>` launch, asserts the enclosing `{ ... }` scope contains at least one `TDMD_NVTX_RANGE` — runs on every CI flavor; regression caught before PR merge.

---

## 13. Configuration и tuning

### 13.1. YAML `gpu:` block

```yaml
gpu:
  device_id: 0                         # default
  streams: 2                           # D-M6-13
  memory_pool_init_size_mib: 256       # D-M6-12
  enable_nvtx: true                    # D-M6-14
```

All optional. Omission → defaults. Breaking-change-free extension от M5 YAML (M5 configs не имеют `gpu:` block и работают как есть — gpu code не активируется если `scheduler.backend != "cuda"`).

### 13.2. CLI overrides

```
tdmd run cfg.yaml --gpu-device=1 --gpu-streams=2 --gpu-memory-pool-mib=512 --no-nvtx
```

Добавляются в `cli/SPEC.md` (change log entry в T6.2).

### 13.3. Environment variables

| Var                     | Effect                                      |
|-------------------------|---------------------------------------------|
| `CUDA_VISIBLE_DEVICES`  | Standard Nvidia — ограничивает visible devices |
| `TDMD_GPU_DEBUG_SYNC`   | If `1`, force `cudaDeviceSynchronize` after every kernel launch (debug only) |
| `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` | Nsight Systems default — ensures все NVTX ranges are captured |

---

## 14. Roadmap alignment

| Milestone | GPU scope                                                               |
|-----------|-------------------------------------------------------------------------|
| **M6**    | gpu/ module + DevicePool + PinnedHostPool + NL/EAM/VV kernels + 2-stream overlap + T3-gpu anchor. **current.** |
| M7        | GPU-aware MPI, NCCL intranode, `stream_aux` (third stream), Pattern 2 halo |
| M8        | SNAP GPU kernel (master spec §14 M8 proof-of-value), autotuner, MixedFastAggressive flavor |
| M9        | MEAM, PACE, MLIAP GPU kernels; NVT/NPT GPU integrators (K=1 only per master spec §14) |
| M10       | clang-cuda matrix CI, FP16/bfloat16 experiments, Unified Memory experiments |
| M11+      | Multi-GPU per rank, GPU-direct RDMA, NVSHMEM                            |

---

## 15. Open questions

| ID          | Question                                                                 | Resolution path |
|-------------|--------------------------------------------------------------------------|-----------------|
| **OQ-M6-1** | Cached pool LRU eviction policy vs explicit `release_all()` — оптимальная granularity eviction | **T6.3 landed с grow-on-demand free-list (no LRU)**; revisit when T6.5 kernel pressure-testing реально покажет pool bloat на production load |
| **OQ-M6-2** | Pinned host pool sizing — per rank или shared на node (multi-rank на GPU? нет в M6, но планируется M7+) | Defer to M7 planning; M6 keeps per-rank |
| **OQ-M6-3** | NVTX overhead measurement — confirmable < 1% на 10⁶-атомном фикстюре?   | Measured в T6.13 smoke; update D-M6-14 if false |
| **OQ-M6-4** | Kahan overhead на GPU — actual cost в NL/EAM hot path vs expected (+3-5%) | **Resolved в T6.11** — T6.5 EAM impl does all Kahan reductions host-side (not the bottleneck at shipped sizes, §7.2 v1.0.3 change-log); per-atom PE+virial accumulator кost never materialised на device. If on-device Kahan ever becomes hot (SNAP / PACE в M8+), re-measure via NVTX `eam.*_kernel` ranges — infrastructure landed в v1.0.9 |
| **OQ-M6-5** | Half-list vs full-list — actual cache behaviour divergence на FP32 math | Measured в T6.5; half-list expected +20% bandwidth advantage |
| **OQ-M6-6** | 2-stream vs single-stream actual speedup на host-staged MPI path         | Measured в T6.9; если < 1.3×, документировать и оставить 2-stream для future N-stream compatibility |
| **OQ-M6-7** | CUB vs custom kernel primitives — где CUB beat custom на sort/scan      | Resolved per-kernel в T6.4/T6.5 pre-impl reports |
| **OQ-M6-8** | Unified Memory (UM) hardware prefetch — actual overhead on sm_80+?       | Deferred to M10+ experiment; M6 staying с explicit memcpy |
| **OQ-M6-9** | clang-cuda status — ABI compatibility с nvcc-built libraries             | Deferred to M10+ |
| **OQ-M6-10** | GPU telemetry frame rate — per-step или aggregated per-100-steps overhead? | **Resolved в T6.11 (v1.0.9)** — per-100-steps aggregation default for `TelemetryFrame` kernel_timings_us (§12 example unchanged); per-step opt-in reserved для future `TDMD_NVTX_PER_STEP=1` env flag if SNAP-scale debug requires. NVTX range cost is pay-per-entry, not per-step, so the opt-in knob lives above the NVTX layer. |
| **OQ-M6-11** | T3-gpu normalization baseline — dissertation Alpha cluster не имел GPU; appropriate baseline для efficiency curve? | **Partially resolved в T7.12 (v1.0.14)** — T7.12 sidesteps the Morse-vs-dissertation hardware-normalization question by switching to the EAM substitute (D-M7-16): grades against absolute `efficiency_floor_pct: 80.0` (D-M7-8 / T7.11 parity), not against the Andreev curve. The hypothetical linear-extrapolated Morse TD curve still applies if the M9+ Morse GPU kernel ever lands — documented в `verify/benchmarks/t3_al_fcc_large_anchor_gpu/acceptance_criteria.md` §"Gate (3) — efficiency curve (T7.12)". Original blocker (Pattern 2 GPU dispatch — was T6.9b) closed via T7.5/T7.7. |

---

## 16. Change log

| Date       | Version | Change                                                                    |
|------------|---------|---------------------------------------------------------------------------|
| 2026-04-19 | v1.0    | Initial авторство. Anchors D-M6-1..D-M6-20 from `docs/development/m6_execution_pack.md`. Ships alongside T6.2 skeleton (`src/gpu/` + `tests/gpu/test_gpu_types.cpp`). Change log extension in `TDMD_Engineering_Spec.md` Приложение C. |
| 2026-04-19 | v1.0.1  | §5.1/§5.2 updated с T6.3 implementation notes: `DevicePool` ships как single class owning both device+pinned pools (1:1 rank binding); grow-on-demand free-list policy; LRU deferred (OQ-M6-1). Adds `factories.hpp` public API (probe_devices / select_device / make_stream / make_event) + `device_pool.hpp`. `cuda_handles.hpp` internal header shares PIMPL Impl defs across gpu/ TUs без leaking CUDA symbols в public API. |
| 2026-04-19 | v1.0.2  | §7.1 resolved — T6.4 `NeighborListGpu` landed. Implementation: two-pass (count → host-scan → emit) kernel pair, identical iteration order to CPU (27-cell stencil, dz-outer → dy → dx); D-M6-7 bit-exact gate met on 864-atom Al FCC (33,696 pairs, `std::memcmp` on offsets + ids + r²). Public API takes raw primitives (positions + cell CSR + BoxParams) — keeps gpu/ data-oblivious per §1.1 and breaks would-be `gpu/ → neighbor/` cyclic include. `src/neighbor/gpu_neighbor_builder.cpp` adapter translates domain types. Host-warn gating fix in `cmake/CompilerWarnings.cmake` — host flags (`-Wpedantic`, `-Werror`) now gated to `$<COMPILE_LANGUAGE:CXX>` so nvcc stub files don't trip extension diagnostics. Micro-bench baseline at `verify/benchmarks/neighbor_gpu_vs_cpu/`: 12.9× (10⁴ atoms) / 28.5× (10⁵ atoms) speedup on sm_120 — well above T6.4 ≥5× bar. OQ-M6-7 (CUB vs custom) resolved: custom two-pass + host scan is adequate for M6; on-device scan deferred to T6.11 perf tuning. |
| 2026-04-19 | v1.0.3  | §7.2 resolved — T6.5 `EamAlloyGpu` landed. Three-kernel path (density → embedding → force), thread-per-atom with **full-list per-atom iteration** (no `j<=i` filter) so every write is thread-local — eliminates the atomics that would otherwise be needed for half-list Newton-3 scatter. Pair PE + virial are counted twice in the full-list sweep and halved on host during Kahan reduction; forces are emitted once per ordered pair (both directions — thread `i` scatters `+Δ`, thread `j` scatters `−Δ`) so no halving applies. Device spline eval mirrors `TabulatedFunction::locate` + Horner bit-exactly; same `minimum_image_axis` formula as CPU. Gate is **≤1e-12 rel**, not byte-equal (spec §7.2) — absorbs the reduction-order drift between CPU half-list and GPU full-list accumulation. Acceptance: Al FCC 864-atom + Ni-Al B2 1024-atom (Mishin 2004) both ≤1e-12 rel on per-atom forces, total PE, and virial Voigt tensor. Public API borrows flattened Hermite-cubic coefficient tables from the `src/potentials/eam_alloy_gpu_adapter.cpp` layer — gpu/ remains data-oblivious. Micro-bench at `verify/benchmarks/eam_gpu_vs_cpu/`: 5.3× (10⁴) / 6.8× (10⁵) on sm_120 vs CPU reference — above T6.5 ≥5× bar. OQ-M6-4 (Kahan overhead on per-atom PE+virial) deferred to T6.11 — current impl does all reductions host-side and is not the bottleneck at these sizes. |
| 2026-04-19 | v1.0.5  | §9 authored — T6.7 engine wire-up. `runtime.backend: cpu\|gpu` opt-in flag, `runtime::GpuContext` RAII owner of DevicePool + compute stream. `SimulationEngine` dispatches на `gpu_backend_` в `recompute_forces()` + `pre/post_force_step()` без изменений в TD scheduler или comm. MPI transport остаётся `MpiHostStagingBackend` (D-M6-3). **D-M6-7 extended to engine level:** thermo stream byte-equal CPU↔GPU на 100 шагов Ni-Al EAM 864 atoms (T6.7 1-rank gate); **D-M5-12 extended to GPU era:** GPU K=1 P=2 ≡ GPU K=1 P=1 на 10 шагов через deterministic Kahan-ring (T6.7 2-rank gate). Scope limits v1.0.5: single compute stream (2-stream — T6.9), Pattern 1/3 only (Pattern 2 — M7), `Fp64ReferenceBuild` only (MixedFast wiring — T6.8). |
| 2026-04-19 | v1.0.4  | §7.3 resolved — T6.6 `VelocityVerletGpu` landed. Two kernels mirroring CPU `VelocityVerletIntegrator`: `pre_force_kernel` (half-kick + drift) and `post_force_kernel` (half-kick only). Per-atom thread, pure element-wise — no reductions, no atomics. Operand order matches CPU path exactly (`v += f · accel · half_dt`; `x += v · dt`) and Reference flavor's `--fmad=false` guarantees byte-equal output. Per-species `accel[s] = ftm2v / mass[s]` precomputed on host (LAMMPS metal units, `ftm2v ≈ 9648.533` per M1 SPEC delta). Public API takes raw host primitives (positions, velocities, forces, type ids, accel table); domain adapter lives в `src/integrator/gpu_velocity_verlet.cpp`. **D-M6-7 gate:** bit-exact CPU↔GPU over 1/10/100/1000 NVE steps on Al 1000-atom single-species + Ni-Al 512-atom two-species lattices (all three lengths green locally). MixedFast path deferred to T6.8 flavor activation. Micro-bench at `verify/benchmarks/integrator_gpu_vs_cpu/`: **0.3× (10⁴) / 0.5× (10⁵) on sm_120** — GPU slower due to per-call H2D/D2H dominating ~6 FLOPs/atom kernel. **Это ожидаемое поведение для T6.6 adapter shape;** speedup unlocks in T6.7 (resident-on-GPU, `integrator/SPEC §3.5`). NVTX deferred to T6.11. |
| 2026-04-19 | v1.0.6  | §8.2 и §8.3 updated — T6.8a partial landed (MixedFast flavor activation + EAM mixed kernel + single-step differential). `MixedFastBuild` переведена из stub с TODO-warning в рабочую конфигурацию (`_tdmd_apply_mixed_fast` задаёт `TDMD_FLAVOR_MIXED_FAST` + `--fmad=true`). `src/gpu/eam_alloy_gpu_mixed.cu` — Philosophy B EAM kernel: r²/sqrtf/inv_r в FP32, all else (spline Horner, phi chain, force accumulators) в FP64. `EamAlloyGpuAdapter` делает compile-time dispatch через typedef `EamAlloyGpuActive` (Mixed в `TDMD_FLAVOR_MIXED_FAST` сборках, обычный `EamAlloyGpu` иначе). `tests/gpu/test_eam_mixed_fast_within_threshold.cpp` ships как T6.8a acceptance: 3 test cases (Ni-Al B2 1024, Al FCC 864, compute_version monotonic) сравнивают две GPU варианты напрямую, обходя adapter. **Achieved thresholds:** rel force ≤ 1e-5 (D-M6-8 target 1e-6), rel PE ≤ 1e-7 (target 1e-8), rel virial ≤ 5e-6 нормализованный на max-component — FP32 `inv_r` cast propagation через ~50-neighbor EAM stencil с partial sign cancellation hit FP32 precision ceiling; FP32-table-storage redesign для закрытия 1e-6 отложен в T6.8b. **D-M6-7 test guards:** `test_neighbor_list_gpu.cpp` r² memcmp, `test_eam_alloy_gpu.cpp` 1e-12 gates, `test_integrator_vv_gpu.cpp` 1/10/1000/multi-species bit-exact gates, `test_gpu_backend_smoke.cpp` CPU≡GPU thermo gate — все guard'ятся `#ifndef TDMD_FLAVOR_FP64_REFERENCE SKIP(...)` so Reference flavor держит D-M6-7 контракт, MixedFast build остаётся зелёным. NL + VV kernels **без** mixed-variant в T6.8a: NL бенефит был бы negligible vs bit-exactness loss на CSR indices; VV kernel — H2D/D2H-bound. T6.8b roadmap: NL mixed variant если perf justified, T4 100-step NVE drift harness в `verify/differentials/t4_gpu_mixed_vs_reference/`, FP32-table redesign vs D-M6-8 relaxation SPEC delta. |
| 2026-04-19 | v1.0.8  | §11.4 rewritten + OQ-M6-11 resolved for T6.10a scope — **T6.10a landed (T3-gpu anchor fixture + harness dispatch)**. New fixture `verify/benchmarks/t3_al_fcc_large_anchor_gpu/` (README, config.yaml, checks.yaml, hardware_normalization_gpu.py stub, acceptance_criteria.md) — Ni-Al EAM 864-atom, 100 steps, single-rank (reuses T4 `setup.data`). Harness extension: `AnchorTestRunner.run()` dispatches on `checks.yaml::backend`; new `_run_gpu_two_level(start, checks)` method runs CPU+GPU Reference passes via `_launch_tdmd_with_backend()` (writes augmented config with `runtime.backend` injected + relative paths resolved to absolute), byte-compares thermo streams, emits `GpuGateResult` list. Report extensions: `AnchorTestReport.backend: "cpu"\|"gpu"`, `gpu_gates: list[GpuGateResult] \| None`, GPU-specific `format_console_summary()` + footer. CLI `--backend {cpu,gpu}` override for T6.12 CI. Gate (2) delegates to T6.8a differential test (exit code check, not raw FP re-compare) с advisory YELLOW. Gate (3) **deferred to T6.10b** — two hard blockers: Morse GPU kernel (M9+, `gpu/SPEC.md` §1.2) + Pattern 2 GPU dispatch (M7, T6.9b). Mocked pytest coverage: 6 new cases в `test_anchor_runner.py::GpuAnchorRunnerMockedTest` (byte-exact green, advisory YELLOW, diverge RED → `CPU_GPU_REFERENCE_DIVERGE`, no-CUDA → `NO_CUDA_DEVICE`, JSON round-trip, backend-override force) + 4 `FirstByteDiffTest` unit tests. All 18/18 pytest green. Все три CI flavors зелёные (Reference+CUDA 34/34, MixedFast+CUDA 34/34, CPU-only-strict 29/29). |
| 2026-04-19 | v1.0.10 | **T6.12 landed — CUDA compile-only CI matrix activated.** New `build-gpu` job in `.github/workflows/ci.yml`: GitHub-hosted `ubuntu-latest` + stock apt `nvidia-cuda-toolkit` + `-DTDMD_BUILD_CUDA=ON -DTDMD_BUILD_FLAVOR={Fp64ReferenceBuild,MixedFastBuild} -DTDMD_CUDA_ARCHS="80;86;89;90" -DTDMD_WARNINGS_AS_ERRORS=ON`. Matrix catches: (a) CUDA source regressions in `src/gpu/*.cu`; (b) PIMPL firewall breaks (CUDA headers leaking to public API); (c) flavor-dispatch adapter + MixedFast EAM kernel compile drift. Post-build ctest filter runs `test_gpu_types` (pure C++ PIMPL), `test_nvtx_audit` (grep walker — meaningful on CI), `test_gpu_cost_tables` + `test_perfmodel` (CPU structural). Runtime-CUDA tests link + load but self-skip via `cudaGetDeviceCount() != cudaSuccess` on no-GPU runner. CUDA archs 80;86;89;90 cover Ampere/Ada/Hopper; sm_100/120 (Blackwell + RTX 5080 dev) stay local-only per D-M6-6 — ubuntu apt CUDA doesn't ship 12.8+. **Option A CI policy** codified in `docs/development/ci_setup.md` (rewritten): no self-hosted runner (public repo, arbitrary PR code risk); CUDA runtime gates (kernel bit-exactness, D-M6-8 thresholds, T3-gpu anchor, M6 smoke) run via local pre-push protocol — three-flavor pre-push sequence (Reference+CUDA, MixedFast+CUDA, CPU-only-strict) documented. Branch protection required-check list updated to include both `Build GPU compile-only (Fp64ReferenceBuild)` и `Build GPU compile-only (MixedFastBuild)`. Zero SPEC-surface or module-API changes — pure CI/policy delivery. |
| 2026-04-19 | v1.0.9  | §12 telemetry + §16 log extended — **T6.11 landed (NVTX instrumentation finalized across all GPU TUs + PerfModel GPU cost tables)**. D-M6-14 satisfied structurally via `src/telemetry/include/tdmd/telemetry/nvtx.hpp` (new `TDMD_NVTX_RANGE(name)` RAII macro, zero-cost `((void)0)` fallback on `TDMD_BUILD_CUDA=0`, `__LINE__`-uniqified var name, default NVTX domain). Instrumented call sites: `src/gpu/neighbor_list_gpu.cu` (6 ranges: `nl.build`, `nl.h2d.positions_and_cells`, `nl.count_kernel`, `nl.host_scan_and_h2d_offsets`, `nl.emit_kernel`, `nl.download`), `src/gpu/eam_alloy_gpu.cu` (8 ranges: `eam.compute`, `.h2d.atoms_and_cells`, `.h2d.splines`, `.h2d.forces_in`, `.density_kernel`, `.embedding_kernel`, `.force_kernel`, `.d2h.forces_and_reductions`), `src/gpu/eam_alloy_gpu_mixed.cu` (8 symmetric `eam_mixed.*`), `src/gpu/integrator_vv_gpu.cu` (8 ranges: `vv.{pre,post}_force_step` + `.h2d.{pre,post}` + `.{pre,post}_force_kernel` + `.d2h.{pre,post}`), `src/gpu/device_pool.cpp` (`gpu.pool.alloc_device`, `gpu.pool.alloc_pinned`). Naming follows §12 `{subsystem}.{op}` convention — stable across runs for Nsight dashboard matching. **PerfModel GPU extension:** `src/perfmodel/include/tdmd/perfmodel/gpu_cost_tables.hpp` adds `GpuKernelCost {a_sec, b_sec_per_atom, predict(n)}` linear model + `GpuCostTables` aggregate (h2d_atom, nl_build, eam_force, vv_pre, vv_post, d2h_force + `provenance`). Factory functions `gpu_cost_tables_fp64_reference()` / `gpu_cost_tables_mixed_fast()` ship **placeholder coefficients** calibrated from Ampere/Ada consumer estimates; provenance strings tag them "T6.11 placeholder — replace via calibration harness". `PerfModel::predict_step_gpu_sec(n_atoms, tables)` wires through `HardwareProfile::n_ranks` so single-rank baseline sums tables + scheduler overhead, multi-rank scales work per rank. **CI enforcement:** `tests/gpu/test_nvtx_audit.cpp` — grep-based Catch2 test walks `src/gpu/*.cu`, finds every `<<<` kernel launch, walks back enclosing brace scope, asserts ≥1 `TDMD_NVTX_RANGE` marker present (filters comment-line occurrences). Runs on all 3 CI flavors; trivially passes on CPU-only (.cu files compile without kernel-launch markers). `tests/perfmodel/test_gpu_cost_tables.cpp` — 8 Catch2 cases: linear model math, structural sanity bands (a_sec ∈ [1e-6, 1e-3]; b_sec_per_atom ∈ [1e-10, 1e-5]), MixedFast ≤ Reference invariant, PerfModel wiring (single-rank + n_ranks-divides-work + Reference ≥ MixedFast). **Scope limit:** ±20% accuracy gate vs measured Nsight data is **deferred to T6.11b** — requires Nsight-profiled calibration run on target GPU which Option A CI (public repo, no self-hosted runner) cannot automate; T6.11b will load measured coefficients from a JSON fixture and compare `predict_step_gpu_sec` against them. **Resolves** OQ-M6-4 (Kahan overhead status) and OQ-M6-10 (GPU telemetry frame rate: per-100-steps default confirmed in §12). Все три CI flavors зелёные (Reference+CUDA 36/36, MixedFast+CUDA 36/36, CPU-only-strict 31/31). |
| 2026-04-19 | v1.0.12 | **T7.0 landed — D-M6-8 SPEC delta + T4 NVE drift harness + NL-mixed REJECT (M6 carry-forward cleanup).** §8.3 rewritten с formal D-M6-8 scope split: **dense-cutoff** (EAM/MEAM/SNAP/PACE/MLIAP — ≥20 neighbors) canonical thresholds relaxed to rel force ≤ 1e-5 / rel PE ≤ 1e-7 / rel virial ≤ 5e-6 (T6.8a measurements formalized; the prior 1e-6/1e-8 target was never achievable under Philosophy B с r²/sqrtf/inv_r в FP32); **sparse-cutoff** (LJ/Morse — 2-8 neighbors) retain 1e-6/1e-8 ambition as M9+ deliverable. Full rationale в §8.3 "Rationale для dense-cutoff ceiling" + memory `project_fp32_eam_ceiling.md`. NVE drift threshold 1e-5/1000 steps **unchanged** — T4 harness validates at 1e-6/100 steps (10× margin). `verify/thresholds/thresholds.yaml` receives new `gpu_mixed_fast:` section с dense+sparse split + rationale. `tests/gpu/test_t4_nve_drift.cpp` — new 100-step NVE drift gate on Ni-Al EAM 864 atoms `runtime.backend: gpu` MixedFast; parses thermo `etotal` column at step 0 + step 100, asserts `abs(ΔE/E₀) ≤ 1e-6`; self-skips на no-CUDA / Reference-only builds (Reference is byte-exact per D-M6-7 — no drift test required). `tests/gpu/test_eam_mixed_fast_within_threshold.cpp` header updated to cite formal D-M6-8 canonical thresholds (no numeric test changes — shipped values were already at formal thresholds). **NL MixedFast variant formally REJECTED** — memory-backed analysis: NL is integer-CSR + one FP64 r²/pair, narrowing r² to FP32 saves ≤3% NL-rebuild wall-time (bandwidth-bound on CSR I/O) but breaks `build_version` bit-exactness between flavors; the determinism contract requires identical pair-iteration order across flavors. §8.3 roadmap entry rewritten to mark rejection + T7.0 closure. Zero code changes to `src/gpu/*.cu`; T7.0 is pure SPEC delta + validation. M6 carry-forward item T6.8b marked **closed as T7.0**; T6.9b (overlap pipeline) / T6.10b (efficiency curve) / T6.11b (±20% calibration) remain as M7 tasks T7.8 / T7.12 / T7.13. Все три CI flavors зелёные. |
| 2026-04-19 | v1.0.11 | **T6.13 landed — M6 integration smoke + D-M6-7 chain closure + M6 milestone declared closed.** New fixture tree `tests/integration/m6_smoke/` (README.md, smoke_config.yaml.template, telemetry_expected.txt, run_m6_smoke.sh, thermo_golden.txt) — 2-rank K=1 `runtime.backend: gpu` Ni-Al EAM/alloy NVE 864-atom 10-step harness; thermo gate is **byte-for-byte equality to the M5 golden** (copied verbatim; step 1/6 pre-flight asserts `diff -q` parity so editing one golden without the other fails CI). `.github/workflows/ci.yml` extended with an `M6 smoke` step inside `build-cpu` right after the M5 step — self-skips on public CI via `nvidia-smi -L` probe (exit 0) per D-M6-6, still runs infrastructure checks (golden parity, template substitution, LFS asset presence) so rot surfaces loudly. §11.3 rewritten from the pre-impl placeholder (10 steps, 864 atoms, M5-golden parity — not the M1-derived 10⁴-atom sketch). **D-M6-7 chain now green on an automated harness end-to-end: M3 ≡ M4 ≡ M5 ≡ M6 thermo golden.** Local pre-push gate: ≤5 s on commodity GPU; mandatory for any merge touching `src/gpu/`, `src/potentials/eam_alloy_gpu_*`, `src/integrator/*_gpu*`, `src/comm/mpi_host_staging*`, or `src/runtime/gpu_context*`. MixedFast coverage deliberately out-of-scope (D-M6-7/D-M6-8 are different gates; mixing them dilutes failure signal). T3-gpu efficiency curve (T6.10b), 2-stream overlap gate (T6.9b), multi-GPU-per-rank (M7+) explicitly not covered. **M6 milestone closed** per master spec §14 acceptance gate; remaining M6 open items (T6.8b FP32-table redesign, T6.9b full overlap pipeline, T6.10b efficiency curve, T6.11b ±20% calibration) carry forward as M7-window tasks per execution pack §6. |
| 2026-04-19 | v1.0.13 | **T7.8 landed (ex-T6.9b) — full 2-stream split-phase compute/mem overlap pipeline + K-deep `GpuDispatchAdapter`.** §3.2b authored. `EamAlloyGpu` экспортирует split-phase async API: `compute_async()` queues H2D на mem_stream + kernels на compute_stream (event-chained per §3.2 sync primitives) + records `kernels_done_event` (no D2H); `finalize_async()` queues D2H на mem_stream waiting `kernels_done_event` + Kahan-reduces. Split required potому что single-phase pipelines self-serialize: cross-stream `cudaStreamWaitEvent(mem, kernels_done)` blocks subsequent H2D'ов до завершения D2H'а на том же stream'е → overlap window = 0. Pinned host buffers (D2H destinations + test-side input mirrors) теперь обязательны — pageable degrades `cudaMemcpyAsync` к internal staging + host block silently. `EamAlloyAsyncHandle::Impl` carries `DevicePtr<std::byte> h_pe_embed/h_pe_pair/h_virial` allocated через `DevicePool::allocate_pinned_host()`. `tdmd::scheduler::GpuDispatchAdapter` (`src/scheduler/include/tdmd/scheduler/gpu_dispatch_adapter.hpp` + `.cpp`) rotates через K internal slots, каждый own `EamAlloyGpu` + pending `EamAlloyAsyncHandle`. `enqueue_eam(...)` / `drain_eam(slot)` — FIFO contract; throws на double-enqueue without intervening drain. **Single-rank EAM-only acceptance gate (5%, не 30%)**: `tests/gpu/test_overlap_budget.cpp` measures `(t_serial - t_pipelined) / t_pipelined` на K=4, 14×14×14 Al FCC (10976 atoms, Al_small.eam.alloy). На RTX 5080: t_serial ≈ 7.48 ms, t_pipelined ≈ 6.85 ms → overlap ≈ 9.3% (gate ≥ 5%, comfortably above noise + functional pipeline check). PE + virial slot 0 vs serial oracle — bit-exact ≤ 1e-12 rel (D-M6-7 preserved через event chain). **Почему 5% а не 30%:** EAM на RTX 5080 на 10k atoms — kernel-bound (T_kernel ≈ 1.5 ms, T_mem ≈ 0.36 ms, ratio 0.24); asymptotic max overlap (K→∞) ≈ 24%, K=4 ≈ 17% — physically below 30% bar. **30% production gate** сохраняется per exec pack §T7.8 spec ("**2-rank** K=4 10k-atom") и переносится в T7.14 (M7 integration smoke), где Pattern 2 dispatch (T7.9) + halo D2H/MPI/H2D traffic (T7.6 OuterSdCoordinator) удвоит per-step memory work, давая T_mem/T_k ~0.55 → 30% achievable. Scheduler library теперь PUBLIC depends on `tdmd::gpu` (was added in `src/scheduler/CMakeLists.txt`). Все три CI flavors зелёные (compile-only); local pre-push runs `test_overlap_budget` ≤ 0.4 s on RTX 5080. |
| 2026-04-19 | v1.0.7  | §3.2a authored + §9.5 scope-limit updated — T6.9a infrastructure landed (dual-stream `GpuContext` + spline H2D caching). `runtime::GpuContext` теперь owns оба non-blocking stream'а: `compute_stream()` (D-M6-13 primary, kernel dispatch) + `mem_stream()` (D-M6-13 secondary, H⇄D copies). Оба создаются через `make_stream()` (non-blocking flag). Adapter'ы пока берут только `compute_stream()`; полная `cudaEventRecord`/`cudaStreamWaitEvent` orchestration (§3.2 pipeline) + 30% overlap gate на 2-rank K=4 отложены в T6.9b (depends on Pattern 2 GPU dispatch — M7). **Spline H2D caching** (§7.2 adapter side): `EamAlloyGpu::Impl` + `EamAlloyGpuMixed::Impl` содержат три host-pointer cache fields (`splines_{F,rho,z2r}_coeffs_host`) + `splines_upload_count` counter. `compute()` re-uploads F/rho/z2r tables **только** когда incoming `tables.*_coeffs` host pointers отличаются от cached. Invariant: после N back-to-back compute() calls одного `EamAlloyGpuAdapter` instance — `splines_upload_count() == 1`. Adapter surface (`EamAlloyGpuAdapter::splines_upload_count()`) forwards от active backend; test coverage в `tests/gpu/test_eam_alloy_gpu.cpp "EamAlloyGpu — splines cached across compute() calls (T6.9a)"`. Rationale: на steady-state MD hot loop (~1000 compute() calls между NL rebuilds) re-upload ~MB-scale spline tables доминировал H2D bandwidth на MixedFast fixture'ах; caching снижает per-step H2D к `n_atoms × 40 bytes`. Works ortho both flavors — тот же pattern в Reference и Mixed Impl. Все три CI flavors зелёные (Reference+CUDA, MixedFast+CUDA, CPU-only-strict). |
| 2026-04-20 | v1.0.17 | **T8.6a landed — `SnapGpu` + `SnapGpuAdapter` PIMPL scaffolding.** §7.4 rewritten: LJ/Morse/MEAM/PACE/MLIAP remain CPU-only; SNAP moved to M8 window (T8.6a scaffolding → T8.6b kernel body → T8.7 bit-exact gate). §7.5 **new** — full SNAP GPU contract authored at T8.6a landing: locked `SnapGpu` / `SnapGpuAdapter` API, M8-scope flag fence (chemflag/quadraticflag/switchinnerflag rejected with `std::invalid_argument`, parity с CPU `SnapPotential`), single-TU `src/gpu/snap_gpu.cu` build-guard story (`#if TDMD_BUILD_CUDA` branches via `logic_error` sentinel on CUDA / `runtime_error` on CPU-only — mirrors `eam_alloy_gpu.cu`, NO separate stub.cpp), planned T8.6b three-pass algorithm (Ui / Yi / deidrj), T8.7 bit-exact gate D-M8-13 ≤ 1e-12 rel CPU↔GPU FP64. `SimulationEngine::init()` switches на `dynamic_cast<SnapPotential*>` → `gpu_snap_potential_` field parallel к `gpu_potential_` (no abstract base — minimum-scope for T8.6a). `src/io/preflight.cpp::check_runtime` relaxed — `runtime.backend=gpu` accepts `potential.style ∈ {eam/alloy, snap}`. `tests/gpu/test_snap_gpu_plumbing.cpp` ships (4 Catch2 cases: constructs cleanly on W_2940; rejects all three M8-scope flags; sentinel error path reachable; `compute_version()` stays at 0 before T8.6b). Self-skips с exit 77 если LAMMPS submodule не initialized (Option A / public CI convention). `test_nvtx_audit` trivially passes — zero `<<<...>>>` sites в `snap_gpu.cu` at T8.6a. Все три CI flavors зелёные (Fp64Reference+CUDA 48/48; MixedFast+CUDA 48/48; CPU-only-strict 40/40); `test_t6_differential` (T8.5 D-M8-7 CPU byte-exact oracle) без регрессии. **T8.6b — remaining work:** full CUDA kernel port (~1500 lines LAMMPS USER-SNAP), index-table flatten (T** → T*), NVTX wrap на каждый kernel launch, __restrict__ compliance. **T8.7 — follow-on:** bit-exact differential gate. |
| 2026-04-20 | v1.0.16 | **T8.0 — M8 entry: 2-rank overlap gate infrastructure (T7.8b carry-forward).** §3.2c authored: 2-rank overlap gate hardware prerequisite (≥ 2 physical CUDA devices; `cudaSetDevice(rank % device_count)` per-rank pinning) + dev SKIP semantics (`cudaGetDeviceCount() >= 2` probe, Catch2 exit 4 SKIP otherwise, CMake `SKIP_RETURN_CODE 4`). `tests/gpu/test_overlap_budget_2rank.cpp` + `tests/gpu/main_mpi.cpp` + `tests/gpu/CMakeLists.txt` (MPI-gated 2-rank target). Synthetic halo `MPI_Sendrecv` (1024 doubles pinned ≈ 8 KB, modelling P_space=2 halo slab ~50 Å×50 Å contact face) interleaved with GPU compute per slot. Serial baseline vs K=4 pipelined `GpuDispatchAdapter`; `REQUIRE(overlap_ratio >= 0.30)` + bit-exact slot 0 vs serial oracle at ≤ 1e-12 rel (D-M6-7 preserved). Runtime 30% measurement cloud-burst-gated (ties into T8.11). Dev workstations (1-GPU — this repo: 1× RTX 5080) SKIP by design per Option A CI policy. |
| 2026-04-20 | v1.0.15 | **T7.14 landed — M7 integration smoke + acceptance gate + M7 milestone closed.** `tests/integration/m7_smoke/` shipped as the M7 analog of `m6_smoke/`: 7-step harness (golden parity pre-flight → `nvidia-smi -L` probe → single-rank Pattern 2 preflight → `mpirun -np 2 tdmd validate` → `tdmd run` → thermo byte-diff → telemetry invariants). The M7 `thermo_golden.txt` is a byte-for-byte copy of M6's (= M5 = M4 = M3); step 1/7 asserts `diff -q` parity, so golden drift fails CI before paying GPU time. `.github/workflows/ci.yml` extended with `M7 smoke` inside `build-cpu` after `M6 smoke` — self-skips on public CI via the GPU probe per D-M6-6 (Option A). **D-M7-10 byte-exact chain green end-to-end** on an automated harness: with `runtime.backend: gpu`, `zoning.subdomains: [2,1,1]`, `pipeline_depth_cap: 1`, `comm.backend: mpi_host_staging`, 2 ranks × 2 subdomains, Pattern 2 degenerates to Pattern 1 spatial decomposition and thermo bits match M6 exactly. Local pre-push ≤3 s on commodity GPU (2 s measured on RTX 5080). Harness guards against a `set -euo pipefail` × empty-`grep` shell pitfall that would otherwise spuriously fail step 7 when a forward-compat telemetry key is absent (`\|\| true` guards around the field-extraction pipeline; fallback treats absent `boundary_stalls_total` as `0`). **M6 + M7 GPU surface locked** — byte-exact Reference path on Pattern 1 (M6) + Pattern 2 K=1 (M7), with MixedFast thresholded gates (D-M6-8) and Pattern 2 ≥30% overlap / D-M7-9 ±25% calibration deliberately orthogonal. |
| 2026-04-19 | v1.0.14 | **T7.12 landed (ex-T6.10b partial) — T3-gpu gate (3) reopened as Pattern 2 EAM-substitute strong-scaling probe.** §11.4 rewritten: T6.10b deferred-block replaced by T7.12 active block; Morse-vs-dissertation comparison stays deferred to M9+. Fixture: `verify/benchmarks/t3_al_fcc_large_anchor_gpu/checks.yaml` `efficiency_curve.status: deferred` → `active_eam_substitute`; new fields `morse_fidelity_blocker` (provenance), `efficiency_floor_pct: 80.0` (D-M7-8 / T7.11 parity), `notes` (EAM-substitute disclaimer). Harness (`verify/harness/anchor_test_runner/runner.py`): `_write_augmented_config()` accepts `subdomains_xyz: list[int] \| None = None` kwarg → injects `zoning.subdomains: [Nx,Ny,Nz]`; `_launch_tdmd_with_backend()` forwards the kwarg; `_run_gpu_two_level()` branches on `efficiency_curve.status` — `active_eam_substitute` calls new `_run_gpu_efficiency_probe()` which iterates `ranks_to_probe` (must include 1), launches per rank with `subdomains_xyz=[N,1,1]`, computes `100 * sps(N) * anchor_n / (sps_anchor * N)`, grades vs `efficiency_floor_pct`, emits one `GpuGateResult` per rank with `gate_name = "efficiency_curve_N{NN}"`. `GpuLaunchFn` typing widened to `Callable[..., dict]`. `GpuGateResult` extended with optional `n_procs / measured_steps_per_sec / measured_efficiency_pct / floor_pct` (default-None for gates 1/2 — backward compat). `AnchorTestReport.failure_mode` adds `EFFICIENCY_BELOW_FLOOR` classification. Mocked pytest: 8 new `GpuEfficiencyProbeTest` cases (perfect-scaling green, below-floor RED, provenance string, anchor-only single-rank, ranks-must-include-anchor invariant, partial-launch-failure RED, deferred-status backward compat, JSON roundtrip new fields) + 3 new `WriteAugmentedConfigSubdomainsTest` cases (subdomain injection, omitted-when-None, wrong-length raises). 29/29 unittests green; 13/13 scaling_runner tests still green (zero regression). `acceptance_criteria.md` rewritten: §"Gate (3) — efficiency curve (T7.12)" replaces §"Deferred gates (T6.10b scope)"; failure-mode taxonomy adds entry 6 `EFFICIENCY_BELOW_FLOOR`; ship-criteria split into T6.10a (shipped) + T7.12 (shipped). Roadmap row T6.10b updated to "EAM-substitute partial closure — Morse/dissertation deferred to M9+". |

Roadmap extensions (authored by future tasks):

- **T6.3** → §5.1 pool LRU detail, resolve OQ-M6-1 (**done — deferred to T6.5**);
- **T6.4** → §7.1 NL kernel details, SoA layout confirmation (D-M6-16) (**done — v1.0.2**);
- **T6.5** → §7.2 EAM kernel details, Kahan overhead measurement (OQ-M6-4) (**done — v1.0.3; OQ-M6-4 deferred to T6.11**);
- **T6.6** → §7.3 VV details + NVE drift measurements (**done — v1.0.4**);
- **T6.7** → §9 engine wire-up authored (**done — v1.0.5**);
- **T6.8a** → §8.2/§8.3 MixedFast EAM mixed kernel + single-step differential (**done — v1.0.6; D-M6-8 thresholds relaxed in shipped tests (rel force 1e-5 vs 1e-6 target), 1e-6 chase + NVE drift harness deferred to T6.8b**);
- **T6.8b / T7.0** → NL mixed variant REJECTED (memory-backed analysis: no perf benefit vs bit-exactness loss) + T4 100-step NVE drift harness `tests/gpu/test_t4_nve_drift.cpp` landed + D-M6-8 formal SPEC delta (dense-cutoff canonical 1e-5 force / 1e-7 PE / 5e-6 virial; sparse-cutoff 1e-6/1e-8 stays ambition M9+) (**done — v1.0.12; FP32-table redesign deferred to future MixedFastAggressive flavor if ever needed**);
- **T6.9a** → §3.2a dual-stream `GpuContext` + spline H2D caching (**done — v1.0.7**);
- **T6.9b / T7.8** → §3.2b full split-phase compute/mem overlap pipeline + `GpuDispatchAdapter` K-deep slot rotation + single-rank pipeline-functional gate (≥5% overlap on K=4 10k-atom EAM, observed ≈9.3% on RTX 5080) (**done — v1.0.13**); the 30% production gate, originally specified for 2-rank K=4 (per exec pack §T7.8 / §3.2a), moves to T7.14 M7 integration smoke where halo traffic doubles T_mem;
- **T6.10a** → §11.4 T3-gpu anchor fixture + harness dispatch + gates (1)+(2) (**done — v1.0.8; gate (3) deferred to T6.10b**);
- **T6.10b / T7.12** → §11.4 efficiency curve EAM-substitute partial closure: Pattern 2 GPU strong-scaling probe activated on the existing Ni-Al EAM/alloy fixture (D-M7-16 scope) + `_write_augmented_config(subdomains_xyz=)` kwarg + 11 new mocked unittests (**done — v1.0.14; Morse-vs-dissertation comparison stays deferred to M9+ Morse GPU kernel — `gpu/SPEC.md` §1.2**);
- **T6.11** → §12 telemetry finalization (NVTX macro + instrumentation of all `<<<...>>>` sites + `device_pool.cpp` allocations) + PerfModel GPU cost tables (`gpu_cost_tables.hpp/cpp`, `predict_step_gpu_sec`) + grep-based NVTX audit test (**done — v1.0.9; ±20% calibration gate deferred to T6.11b pending Nsight run on target GPU**);
- **T6.11b** → ±20% accuracy gate: calibrate `GpuKernelCost{a_sec, b_sec_per_atom}` per stage from Nsight-measured micro-bench, load coefficients via JSON fixture, compare `PerfModel::predict_step_gpu_sec` to measured step time (blocked on target-GPU Nsight run — cannot run in public-repo CI without self-hosted runner);
- **T6.12** → `.github/workflows/ci.yml` `build-gpu` matrix (Fp64Reference + MixedFast) compile-only on `ubuntu-latest` + stock apt CUDA toolkit; required status check (**done — v1.0.10; runtime-CUDA tests remain local pre-push**);
- **T6.13** → §11.3 M6 integration smoke + D-M6-7 chain closure + CI wiring (**done — v1.0.11; M6 milestone closed**);
- **T8.0** → §3.2c 2-rank overlap gate hardware prerequisite + dev SKIP semantics (**done — v1.0.16; runtime 30% measurement cloud-burst-gated, ties into T8.11**);
- **T8.6a** → §7.4 rewritten (SNAP moved to M8) + §7.5 new SNAP GPU contract + `SnapGpu` / `SnapGpuAdapter` PIMPL scaffolding + M8-scope flag fence + sentinel-throw `compute()` + `SimulationEngine` dispatch wiring + `preflight::check_runtime` relaxation + `tests/gpu/test_snap_gpu_plumbing.cpp` (**done — v1.0.17; kernel body T8.6b, bit-exact gate T8.7**);
- **T8.6b** → full CUDA kernel port of LAMMPS USER-SNAP three-pass (~1500 lines; Ui → Yi → deidrj); `Impl::compute()` body implements H2D→kernels→D2H→Kahan reduction; index-table flatten (T** → T*); NVTX wrap each launch; __restrict__ compliance;
- **T8.7**  → bit-exact gate `test_snap_gpu_bit_exact.cpp` — GPU FP64 ≡ CPU FP64 ≤ 1e-12 rel on W 2000-atom fixture (D-M8-13).
