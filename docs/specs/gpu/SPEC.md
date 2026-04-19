# gpu/SPEC.md

**Module:** `gpu/`
**Status:** master module spec v1.0
**Parent:** `TDMD Engineering Spec v2.5` §14 M6, §15.2, §D (precision policy)
**Last updated:** 2026-04-19

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

LJ, Morse, MEAM, SNAP, PACE, MLIAP, NVT, NPT, thermostats — все откладываются на M9+. Их kernel contracts добавятся в соответствующие sections gpu/SPEC v1.x когда они landed. До того их `*Potential` / `*Integrator` классы остаются CPU-only; SimulationEngine (T6.7 wiring) routes на CPU path если potential name не в {EAM/alloy, Ni-Al}.

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

Compute math в FP32, accumulation в FP64:

- Positions, velocities — FP32 storage, FP32 arithmetic;
- Forces — FP64 storage per atom, FP32 partial contributions summed into FP64 accumulator;
- Energies — FP64 storage + FP32 partials;
- EAM splines — FP32 tables;
- Embedding function F(ρ) — evaluate в FP32, accumulator ρᵢ уже FP64.

**Rationale:** Philosophy B (`master spec §D.1`) — FP32 compute предоставляет 2× throughput + 2× bandwidth против FP64 на all current Nvidia GPUs; FP64 accumulator сохраняет catastrophic-cancellation safety.

### 8.3. Differential thresholds (D-M6-8)

```yaml
# verify/threshold_registry.yaml extension (T6.8 adds these)
gpu_reference_force_bit_exact:
  units: dimensionless
  threshold: 0  # literal equality
  source: gpu/SPEC §6.3 + D-M6-7

gpu_mixed_fast_force_rel:
  units: dimensionless (rel err per-atom L∞)
  threshold: 1e-6
  source: gpu/SPEC §8.3 + D-M6-8

gpu_mixed_fast_energy_rel:
  units: dimensionless (rel err total)
  threshold: 1e-8
  source: gpu/SPEC §8.3 + D-M6-8

gpu_mixed_fast_nve_drift:
  units: dimensionless (rel drift per 1000 steps)
  threshold: 1e-5
  source: gpu/SPEC §8.3 + D-M6-8
```

Проверяется в T6.8 differential (DifferentialRunner extension) + T6.10 T3-gpu anchor.

### 8.4. Deferred flavors

- `Fp64ProductionBuild` — разрешает `--fmad=true` для perf. **НЕ активен** в M6; вернётся в M8+ когда performance tuning начинается;
- `MixedFastAggressiveBuild` — Philosophy A (FP32 full stack, no FP64 accumulator). Opt-in M8+ только если `MixedFast` + additional perf budget потребуется;
- `Fp32ExperimentalBuild` — pure FP32, no safety net. M10+ research.

---

## 9. NVTX instrumentation (D-M6-14)

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

## 10. Tests

### 10.1. T6.2 skeleton tests (этот PR)

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

### 10.2. T6.3+ runtime tests (local-only gated on TDMD_BUILD_CUDA)

- **T6.3**: `test_device_pool_alloc_free` — 1000 alloc+free cycles across size classes, verify no leak + hit rate ≥95%;
- **T6.3**: `test_pinned_host_pool_mpi_symmetry` — allocate pinned host, D2H, MPI send-to-self, receive, H2D, bit-compare;
- **T6.4**: `test_neighbor_bit_exact_vs_cpu` — Ni-Al 10⁴ atoms, GPU half-list ≡ CPU half-list after sort within bucket;
- **T6.5**: `test_eam_force_bit_exact_reference` — same fixture, one-step force bit-exact;
- **T6.5**: `test_eam_mixed_fast_within_threshold` — same fixture, MixedFast within D-M6-8 thresholds;
- **T6.6**: `test_vv_nve_drift` — 1000 steps, drift < 1e-5 rel.

### 10.3. M6 integration smoke (T6.13)

Extends M1 smoke — same Ni-Al 10⁴ thermodynamic trajectory run on GPU Reference path. Thermo at step {0, 100, 500, 1000} byte-exact to CPU Reference from M1. MixedFast variant (second run) within D-M6-8 thresholds.

### 10.4. T3-gpu anchor (T6.10)

Extends T3 harness на GPU:

- (a) `Fp64ReferenceBuild` CPU ≡ `Fp64ReferenceBuild` GPU bit-exact forces + thermo at step 0, 1000;
- (b) `MixedFastBuild` GPU within D-M6-8 thresholds of `Fp64ReferenceBuild` GPU;
- (c) GPU efficiency curve vs N_procs within 10% of dissertation (same bar as M5 CPU anchor).

Hardware normalization: extra `hardware_normalization_gpu.py` computes `gpu_flops_ratio` от current machine до dissertation baseline (placeholder: 1 GFLOP/s per rank — Andreev 2007 Alpha cluster не имел GPU, так что GPU anchor compares против **hypothetical linear-extrapolated TD curve**. Detail documented в T6.10).

---

## 11. Telemetry

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

Dumped в JSON при `tdmd run --telemetry-json=out.json`. Consumed Nsight Systems export (`nsys export`) и `perfmodel/` (Pattern 3 predict() validation в T6.11).

---

## 12. Configuration и tuning

### 12.1. YAML `gpu:` block

```yaml
gpu:
  device_id: 0                         # default
  streams: 2                           # D-M6-13
  memory_pool_init_size_mib: 256       # D-M6-12
  enable_nvtx: true                    # D-M6-14
```

All optional. Omission → defaults. Breaking-change-free extension от M5 YAML (M5 configs не имеют `gpu:` block и работают как есть — gpu code не активируется если `scheduler.backend != "cuda"`).

### 12.2. CLI overrides

```
tdmd run cfg.yaml --gpu-device=1 --gpu-streams=2 --gpu-memory-pool-mib=512 --no-nvtx
```

Добавляются в `cli/SPEC.md` (change log entry в T6.2).

### 12.3. Environment variables

| Var                     | Effect                                      |
|-------------------------|---------------------------------------------|
| `CUDA_VISIBLE_DEVICES`  | Standard Nvidia — ограничивает visible devices |
| `TDMD_GPU_DEBUG_SYNC`   | If `1`, force `cudaDeviceSynchronize` after every kernel launch (debug only) |
| `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` | Nsight Systems default — ensures все NVTX ranges are captured |

---

## 13. Roadmap alignment

| Milestone | GPU scope                                                               |
|-----------|-------------------------------------------------------------------------|
| **M6**    | gpu/ module + DevicePool + PinnedHostPool + NL/EAM/VV kernels + 2-stream overlap + T3-gpu anchor. **current.** |
| M7        | GPU-aware MPI, NCCL intranode, `stream_aux` (third stream), Pattern 2 halo |
| M8        | SNAP GPU kernel (master spec §14 M8 proof-of-value), autotuner, MixedFastAggressive flavor |
| M9        | MEAM, PACE, MLIAP GPU kernels; NVT/NPT GPU integrators (K=1 only per master spec §14) |
| M10       | clang-cuda matrix CI, FP16/bfloat16 experiments, Unified Memory experiments |
| M11+      | Multi-GPU per rank, GPU-direct RDMA, NVSHMEM                            |

---

## 14. Open questions

| ID          | Question                                                                 | Resolution path |
|-------------|--------------------------------------------------------------------------|-----------------|
| **OQ-M6-1** | Cached pool LRU eviction policy vs explicit `release_all()` — оптимальная granularity eviction | **T6.3 landed с grow-on-demand free-list (no LRU)**; revisit when T6.5 kernel pressure-testing реально покажет pool bloat на production load |
| **OQ-M6-2** | Pinned host pool sizing — per rank или shared на node (multi-rank на GPU? нет в M6, но планируется M7+) | Defer to M7 planning; M6 keeps per-rank |
| **OQ-M6-3** | NVTX overhead measurement — confirmable < 1% на 10⁶-атомном фикстюре?   | Measured в T6.13 smoke; update D-M6-14 if false |
| **OQ-M6-4** | Kahan overhead на GPU — actual cost в NL/EAM hot path vs expected (+3-5%) | Measured в T6.5 regression; update §6.2 if significantly higher |
| **OQ-M6-5** | Half-list vs full-list — actual cache behaviour divergence на FP32 math | Measured в T6.5; half-list expected +20% bandwidth advantage |
| **OQ-M6-6** | 2-stream vs single-stream actual speedup на host-staged MPI path         | Measured в T6.9; если < 1.3×, документировать и оставить 2-stream для future N-stream compatibility |
| **OQ-M6-7** | CUB vs custom kernel primitives — где CUB beat custom на sort/scan      | Resolved per-kernel в T6.4/T6.5 pre-impl reports |
| **OQ-M6-8** | Unified Memory (UM) hardware prefetch — actual overhead on sm_80+?       | Deferred to M10+ experiment; M6 staying с explicit memcpy |
| **OQ-M6-9** | clang-cuda status — ABI compatibility с nvcc-built libraries             | Deferred to M10+ |
| **OQ-M6-10** | GPU telemetry frame rate — per-step или aggregated per-100-steps overhead? | Resolved в T6.11 — per-100-steps default, per-step opt-in |
| **OQ-M6-11** | T3-gpu normalization baseline — dissertation Alpha cluster не имел GPU; appropriate baseline для efficiency curve? | Documented в T6.10 pre-impl; текущий план — hypothetical linear-extrapolated TD curve |

---

## 15. Change log

| Date       | Version | Change                                                                    |
|------------|---------|---------------------------------------------------------------------------|
| 2026-04-19 | v1.0    | Initial авторство. Anchors D-M6-1..D-M6-20 from `docs/development/m6_execution_pack.md`. Ships alongside T6.2 skeleton (`src/gpu/` + `tests/gpu/test_gpu_types.cpp`). Change log extension in `TDMD_Engineering_Spec.md` Приложение C. |
| 2026-04-19 | v1.0.1  | §5.1/§5.2 updated с T6.3 implementation notes: `DevicePool` ships как single class owning both device+pinned pools (1:1 rank binding); grow-on-demand free-list policy; LRU deferred (OQ-M6-1). Adds `factories.hpp` public API (probe_devices / select_device / make_stream / make_event) + `device_pool.hpp`. `cuda_handles.hpp` internal header shares PIMPL Impl defs across gpu/ TUs без leaking CUDA symbols в public API. |
| 2026-04-19 | v1.0.2  | §7.1 resolved — T6.4 `NeighborListGpu` landed. Implementation: two-pass (count → host-scan → emit) kernel pair, identical iteration order to CPU (27-cell stencil, dz-outer → dy → dx); D-M6-7 bit-exact gate met on 864-atom Al FCC (33,696 pairs, `std::memcmp` on offsets + ids + r²). Public API takes raw primitives (positions + cell CSR + BoxParams) — keeps gpu/ data-oblivious per §1.1 and breaks would-be `gpu/ → neighbor/` cyclic include. `src/neighbor/gpu_neighbor_builder.cpp` adapter translates domain types. Host-warn gating fix in `cmake/CompilerWarnings.cmake` — host flags (`-Wpedantic`, `-Werror`) now gated to `$<COMPILE_LANGUAGE:CXX>` so nvcc stub files don't trip extension diagnostics. Micro-bench baseline at `verify/benchmarks/neighbor_gpu_vs_cpu/`: 12.9× (10⁴ atoms) / 28.5× (10⁵ atoms) speedup on sm_120 — well above T6.4 ≥5× bar. OQ-M6-7 (CUB vs custom) resolved: custom two-pass + host scan is adequate for M6; on-device scan deferred to T6.11 perf tuning. |
| 2026-04-19 | v1.0.3  | §7.2 resolved — T6.5 `EamAlloyGpu` landed. Three-kernel path (density → embedding → force), thread-per-atom with **full-list per-atom iteration** (no `j<=i` filter) so every write is thread-local — eliminates the atomics that would otherwise be needed for half-list Newton-3 scatter. Pair PE + virial are counted twice in the full-list sweep and halved on host during Kahan reduction; forces are emitted once per ordered pair (both directions — thread `i` scatters `+Δ`, thread `j` scatters `−Δ`) so no halving applies. Device spline eval mirrors `TabulatedFunction::locate` + Horner bit-exactly; same `minimum_image_axis` formula as CPU. Gate is **≤1e-12 rel**, not byte-equal (spec §7.2) — absorbs the reduction-order drift between CPU half-list and GPU full-list accumulation. Acceptance: Al FCC 864-atom + Ni-Al B2 1024-atom (Mishin 2004) both ≤1e-12 rel on per-atom forces, total PE, and virial Voigt tensor. Public API borrows flattened Hermite-cubic coefficient tables from the `src/potentials/eam_alloy_gpu_adapter.cpp` layer — gpu/ remains data-oblivious. Micro-bench at `verify/benchmarks/eam_gpu_vs_cpu/`: 5.3× (10⁴) / 6.8× (10⁵) on sm_120 vs CPU reference — above T6.5 ≥5× bar. OQ-M6-4 (Kahan overhead on per-atom PE+virial) deferred to T6.11 — current impl does all reductions host-side and is not the bottleneck at these sizes. |
| 2026-04-19 | v1.0.4  | §7.3 resolved — T6.6 `VelocityVerletGpu` landed. Two kernels mirroring CPU `VelocityVerletIntegrator`: `pre_force_kernel` (half-kick + drift) and `post_force_kernel` (half-kick only). Per-atom thread, pure element-wise — no reductions, no atomics. Operand order matches CPU path exactly (`v += f · accel · half_dt`; `x += v · dt`) and Reference flavor's `--fmad=false` guarantees byte-equal output. Per-species `accel[s] = ftm2v / mass[s]` precomputed on host (LAMMPS metal units, `ftm2v ≈ 9648.533` per M1 SPEC delta). Public API takes raw host primitives (positions, velocities, forces, type ids, accel table); domain adapter lives в `src/integrator/gpu_velocity_verlet.cpp`. **D-M6-7 gate:** bit-exact CPU↔GPU over 1/10/100/1000 NVE steps on Al 1000-atom single-species + Ni-Al 512-atom two-species lattices (all three lengths green locally). MixedFast path deferred to T6.8 flavor activation. Micro-bench at `verify/benchmarks/integrator_gpu_vs_cpu/`: **0.3× (10⁴) / 0.5× (10⁵) on sm_120** — GPU slower due to per-call H2D/D2H dominating ~6 FLOPs/atom kernel. **Это ожидаемое поведение для T6.6 adapter shape;** speedup unlocks in T6.7 (resident-on-GPU, `integrator/SPEC §3.5`). NVTX deferred to T6.11. |

Roadmap extensions (authored by future tasks):

- **T6.3** → §5.1 pool LRU detail, resolve OQ-M6-1 (**done — deferred to T6.5**);
- **T6.4** → §7.1 NL kernel details, SoA layout confirmation (D-M6-16) (**done — v1.0.2**);
- **T6.5** → §7.2 EAM kernel details, Kahan overhead measurement (OQ-M6-4) (**done — v1.0.3; OQ-M6-4 deferred to T6.11**);
- **T6.6** → §7.3 VV details + NVE drift measurements (**done — v1.0.4**);
- **T6.7** → §3.2 pipeline pattern extended;
- **T6.8** → §8.3 threshold registry wired;
- **T6.9** → §3.5 N-stream / K-way pipelining pre-study;
- **T6.10** → §10.4 anchor-test normalization resolution (OQ-M6-11);
- **T6.11** → §11 telemetry finalization;
- **T6.13** → §10.3 smoke chain confirmed.
