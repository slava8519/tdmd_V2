# comm/SPEC.md

**Module:** `comm/`
**Status:** master module spec
**Parent:** `TDMD Engineering Spec v2.1` §10, §12.6, §4a
**Last updated:** 2026-04-16

---

## 1. Purpose и scope

### 1.1. Что делает модуль

`comm/` — **единственный** модуль TDMD, который знает про MPI, NCCL, NVSHMEM, network. Абстракция `CommBackend` скрывает все детали транспорта от остального кода.

Делает три вещи:

1. **Inner-level transfer** — temporal packets между ranks в пределах одного subdomain'а (TD pipeline);
2. **Outer-level transfer** — halo exchange между subdomain'ами (Pattern 2, SD semantic);
3. **Collective operations** — global reductions для energy, temperature, etc.

### 1.2. Scope: что НЕ делает

- **не решает что передавать** — scheduler формирует `TemporalPacket` и говорит `send`;
- **не принимает решения о времени передачи** — scheduler дирижирует;
- **не знает про физику** — packet_content opaque bytes для comm;
- **не управляет GPU memory** — атомы в `AtomSoA`, pack/unpack buffers — responsibility callers;
- **не занимается serialization форматов** — byte-level `pack`/`unpack` provided by state/zoning;
- **не делает disk I/O** (это `io/`).

Comm — чистый транспорт. Data-oblivious, time-oblivious, policy-aware.

### 1.3. Почему comm критичен именно для TDMD

В обычном MD-коде `comm/` — «один из модулей». В TDMD — **центральная точка выигрыша**:

- **Структурная экономия bandwidth** (формула Андреева, §4.3 master spec): TD передаёт `T_p / K` vs SD передаёт `T_h` per step;
- **Топологическое соответствие железу** (§4a.3): intra-node через NVLink, inter-node через InfiniBand — разные backends для разных уровней;
- **Async overlap**: compute текущей зоны ↔ передача предыдущей зоны ↔ приём следующей.

Плохо написанный `comm/` сводит на нет theoretical advantage TD. Хорошо написанный — превращает TDMD в реально быстрый код на commodity networks.

---

## 2. Public interface

### 2.1. Базовые типы

```cpp
namespace tdmd {

struct TemporalPacket {
    uint16_t    protocol_version;
    ZoneId      zone_id;
    TimeLevel   time_level;
    Version     version;
    uint32_t    atom_count;
    Box         box_snapshot;           // для periodic wrap consistency
    std::vector<uint8_t>  payload;      // serialized AtomSoA slice
    uint64_t    certificate_hash;
    uint32_t    crc32;
};

struct HaloPacket {
    uint16_t    protocol_version;
    uint32_t    source_subdomain_id;
    uint32_t    dest_subdomain_id;
    TimeLevel   time_level;
    uint32_t    atom_count;
    std::vector<uint8_t>  payload;
    uint32_t    crc32;
};

struct MigrationPacket {
    uint16_t    protocol_version;
    uint32_t    source_subdomain_id;
    uint32_t    dest_subdomain_id;
    uint32_t    atom_count;
    std::vector<uint8_t>  payload;      // atoms being migrated cross-subdomain
    uint32_t    crc32;
};

enum class CommEndpoint {
    InnerTdPeer,       // rank в моём subdomain'е, для temporal packets
    OuterSdPeer,       // other subdomain, для halo/migration
    GlobalRoot         // для collectives
};

enum class BackendCapability {
    GpuAwarePointers,       // can send device pointers без staging
    RemoteDirectMemory,     // RDMA (IB verbs, GDRDMA)
    CollectiveOptimized,    // NCCL-like fast reductions
    RingTopologyNative      // native primitives для ring (rare; Ring backend)
};

struct BackendInfo {
    std::string                   name;
    std::vector<BackendCapability>  capabilities;
    uint64_t                      protocol_version;
    double                        measured_bw_bytes_per_sec;   // auto-benched on init
    double                        measured_latency_us;
};

} // namespace tdmd
```

### 2.2. Главный интерфейс

```cpp
namespace tdmd {

class CommBackend {
public:
    // Lifecycle:
    virtual void  initialize(const CommConfig&) = 0;
    virtual void  shutdown() = 0;

    // Inner-level (TD temporal packets, внутри subdomain'а):
    virtual void  send_temporal_packet(const TemporalPacket&, int dest_rank) = 0;
    virtual std::vector<TemporalPacket>  drain_arrived_temporal() = 0;

    // Outer-level (SD halo, Pattern 2 only):
    virtual void  send_subdomain_halo(const HaloPacket&, int dest_subdomain) = 0;
    virtual std::vector<HaloPacket>  drain_arrived_halo() = 0;

    // Cross-subdomain migration (Pattern 2):
    virtual void  send_migration_packet(const MigrationPacket&, int dest_subdomain) = 0;
    virtual std::vector<MigrationPacket>  drain_arrived_migrations() = 0;

    // Collectives:
    virtual double  global_sum_double(double local) = 0;
    virtual double  global_max_double(double local) = 0;
    virtual void    barrier() = 0;

    // Progress (для async mode):
    virtual void  progress() = 0;       // drive outstanding async ops

    // Query:
    virtual BackendInfo  info() const = 0;
    virtual int         rank() const = 0;
    virtual int         nranks() const = 0;

    virtual ~CommBackend() = default;
};

} // namespace tdmd
```

### 2.3. Concrete backends

```cpp
class MpiHostStagingBackend  final : public CommBackend {
    // universal fallback: pack → host → MPI → host → unpack
    // Works anywhere MPI is available
};

class GpuAwareMpiBackend      final : public CommBackend {
    // CUDA-aware MPI: device pointers переданы напрямую
    // Requires: MPI built with CUDA support (OpenMPI с ucx, MVAPICH2-GDR, ...)
};

class NcclBackend             final : public CommBackend {
    // NCCL для intra-node collectives; точка силы Pattern 1 и inner level Pattern 2
};

class NvshmemBackend          final : public CommBackend {
    // NVSHMEM для ultra-low-latency (research only; v2+)
};

class RingBackend             final : public CommBackend {
    // Ring topology native: для anchor-test §13.3 master spec
    // Voсpроиzvоdит TIME-MD Андреева 1:1
};

class HybridBackend           final : public CommBackend {
    // Pattern 2 implementation: inner через NCCL, outer через GPU-aware MPI
};
```

---

## 3. Topology and addressing

### 3.1. Logical topologies

TDMD supports три логических топологии (§10.1-10.2 master spec):

**Ring (legacy / anchor-test):** rank `i` → rank `(i+1) mod P`. Одно направление, one neighbor.

**Mesh / Cartesian (default inner):** rank видит себя в 3D-решётке `(p_x, p_y, p_z)`. До 26 соседей (3³ - 1).

**Cartesian SD grid (outer, Pattern 2):** subdomain'ы образуют 3D-решётку. Также до 26 соседей.

### 3.2. Addressing abstraction

**Inner:** `dest_rank` — absolute rank в MPI_COMM_WORLD.
**Outer:** `dest_subdomain` — subdomain ID.

Резолвинг «какой physical rank принимает halo для subdomain X» делается в `HybridBackend` через cached `subdomain_to_ranks[]` mapping.

### 3.3. Configuration

```yaml
comm:
  backend: auto              # auto | mpi_host | gpu_mpi | nccl | nvshmem | hybrid | ring
  inner_topology: mesh       # mesh | ring | auto
  outer_topology: mesh       # mesh (Pattern 2 only)

  # для Pattern 2:
  subdomain_layout: auto     # auto | explicit [P_space_x, P_space_y, P_space_z]

  # optimization flags:
  use_gpu_aware: true
  use_nccl_intranode: true
  nvshmem_research: false
```

`auto` mode: backend probes hardware + capabilities, выбирает best combination. Fallback chain:

```
try HybridBackend (GpuAwareMpi + NCCL) → if intra-node MPI-CUDA works
  fallback: MpiHostStagingBackend (always works)
```

### 3.4. TopologyResolver

```cpp
namespace tdmd::comm {

struct CartesianGrid {
    int nx{1}, ny{1}, nz{1};
    bool periodic_x{false}, periodic_y{false}, periodic_z{false};
    [[nodiscard]] int total() const noexcept;          // = nx * ny * nz
};

class TopologyResolver {
public:
    explicit TopologyResolver(CartesianGrid grid);     // throws on non-positive dims
    [[nodiscard]] int subdomain_id(int ix, int iy, int iz) const;
    void coords(int sd, int& ix, int& iy, int& iz) const;
    [[nodiscard]] std::vector<int> owner_ranks(int sd) const;       // M7: identity (D-M7-2)
    [[nodiscard]] std::vector<int> peer_neighbors(int sd) const;    // Moore neighborhood
    [[nodiscard]] const CartesianGrid& grid() const noexcept;
};

}  // namespace tdmd::comm
```

Pure stateless Cartesian-grid helper. `HybridBackend` (§6.4) and the engine
preflight (T7.9) use it as the single source of truth for `subdomain → rank`
mapping and per-subdomain peer enumeration. Two queries:

- **`owner_ranks(sd)`** — D-M7-2: 1:1 subdomain↔rank binding at M7 → returns
  exactly `{ sd }`. Vector return shape is reserved for M8+ subdomain sharding.
- **`peer_neighbors(sd)`** — Moore neighborhood (3³ - 1 = 26 in 3D, 8 in 2D,
  2 in 1D) within the Cartesian grid. Non-periodic boundaries drop out-of-grid
  offsets. Periodic boundaries wrap, with dedup so tiny grids like 2×2×2
  periodic produce 7 unique peers (not 26 with duplicates).

**Determinism:** iteration order is fixed (z then y then x), so `peer_neighbors`
returns a sorted vector that is bit-identical for the same grid config across
runs and ranks. This matters for D-M5-12 byte-exact reduction chains where
peer-iteration order affects the order of partial sums.

**Pattern 1 (single subdomain):** `peer_neighbors(0)` on a 1×1×1 grid (with or
without periodicity) returns empty. `HybridBackend` therefore never issues an
outer halo send in Pattern 1 — the wrapped behavior is bit-identical to the
inner backend alone.

**Implementation (T7.5, M7):** `src/comm/{include/tdmd/comm/topology_resolver.hpp,
topology_resolver.cpp}`. Validation throws `std::invalid_argument` for
non-positive dimensions and `std::out_of_range` for invalid `sd` / coordinate
queries. `std::set` provides the dedup + sort in `peer_neighbors`. Unit
coverage in `tests/comm/test_topology_resolver.cpp` (always-built, no MPI):
1D / 2D / 3D round-trip, boundary counts (3 / 5 / 7 / 8 / 26), 2×2×2 periodic
dedup, single-subdomain Pattern 1 emptiness, deterministic re-construction.

---

## 4. Protocol specifications

### 4.1. TemporalPacket wire format

```
offset  size   field
0       2      protocol_version (uint16, big-endian)
2       4      zone_id (uint32)
6       8      time_level (uint64)
14      8      version (uint64)
22      4      atom_count (uint32)
26      48     box_snapshot (6 × double)
74      8      certificate_hash (uint64)
82      4      payload_size (uint32)
86      N      payload (per-atom binary data)
86+N    4      crc32 (uint32, over all preceding bytes)
```

Per-atom payload (size = atom_count × atom_record_size):
```
offset in atom record  field
0       8      atom_id (AtomId = uint64)
8       4      species (SpeciesId = uint32)
12      24     position (x, y, z as double)
36      24     velocity (vx, vy, vz as double)
60      12     image flags (3 × int32)
72      4      flags (uint32)
```

`atom_record_size = 76 bytes`.

### 4.2. HaloPacket wire format

Same как TemporalPacket, но без `zone_id` (halo — це границы subdomain'а в целом, не per-zone granularity). Добавлено `source_subdomain_id` и `dest_subdomain_id`.

Atoms — subset локальных atoms subdomain'а, которые находятся в пределах `r_c + r_skin` от границы, отданной к dest subdomain.

**Ownership boundary (T7.2 clarification, master §8.2):** `HaloPacket` — это **wire format**, owned by `comm/`. Receiver-side unpack `HaloPacket → HaloSnapshot` (in-memory archived record, see scheduler/SPEC §4.6) — **out-of-scope for comm**: scheduler's `OuterSdCoordinator::unpack_halo()` owns the conversion, потому что unpack is subdomain-aware (peer zone id mapping, atom filtering by local boundary radius). `comm/` ровно три обязательства относительно halo:

1. **deliver bytes** — `send_subdomain_halo` / `drain_halo_arrived` ровно как для temporal packets, без знания о content;
2. **CRC32 integrity** — §4.4 catches network corruption pre-coordinator;
3. **eager-protocol commit** — §4.5 same pattern as TemporalPacket (no explicit ACK).

Per-payload meaning, archive lifecycle, eviction policy — все в scheduler/SPEC §4.6.

### 4.3. Protocol versioning

`protocol_version` — monotonic uint16. Incompatible change = bump version + reject packets с `other_version < current_version` с clear error.

**v1:** `protocol_version = 1`.
**v1.1:** extend payload — add per-atom force (для some flight scenarios). Bumps to 2.

Receiver всегда валидирует version первым делом:
```
if (packet.protocol_version != PROTOCOL_VERSION):
    error("protocol mismatch: got v%d, expected v%d", ...)
```

### 4.4. CRC32 policy

CRC32 over entire packet (excluding CRC field itself). Computed на sender, validated на receiver.

CRC failure → drop packet + log + trigger retry protocol. **Не silent accept**.

В production это catches rare memory corruption или network bit flips, которых быть не должно, но случаются на commodity hardware.

### 4.5. Acknowledgment protocol

Для TemporalPacket — **eager protocol** (не требуется explicit ACK):
- Sender `send_async` → marks `InFlight`.
- When `MPI_Test` says complete → sender marks `Committed` locally.
- Receiver's `drain_arrived_temporal()` возвращает packet → receiver knows it's committed.

Для HaloPacket в Pattern 2 — eager same way.

**Explicit ACK не нужен в TDMD**, потому что каузальный DAG scheduler'а заменяет ACK semantics: если receiver использовал данные для compute — он вернёт свой `Completed` event, и sender видит это через logical clock.

---

## 5. Async model

### 5.1. Базовый принцип: всё асинхронно

Ни один `send_*` не блокирует. Возвращает сразу, запускает underlying async primitive (MPI_Isend / ncclSend).

`drain_arrived_*()` non-blocking: возвращает what's arrived так далеко.

`progress()` — explicit "tick" для backends, требующих manual progression (некоторые MPI implementations).

### 5.2. Streams (GPU)

В GpuAwareMpiBackend / NcclBackend / HybridBackend pack/unpack operations associated с CUDA streams:

- `stream_compute` — force / integrate kernels (из §9.2 master spec);
- `stream_comm` — pack / unpack + network;
- `stream_aux` — reorders, telemetry.

Comm backend **никогда** не синхронизирует compute stream. Все synchronization — через CUDA events, registered при send/receive.

### 5.3. Buffer management

**Send-side buffers:**
- allocated by caller (scheduler/zoning), passed to `send_*` via `TemporalPacket.payload`;
- caller retains ownership **until send completes**;
- backend signals completion via callback или CUDA event;
- caller is responsible for buffer lifetime.

**Receive-side buffers:**
- allocated by backend (pinned host memory + device buffer pool);
- `drain_arrived_*` moves ownership to caller;
- caller must release via explicit `return_buffer(packet)`.

Buffer pool size — configurable, default `4 × K_max` packets worth (holds pipeline of K-batched sends in flight).

### 5.4. Progress in compute path

Scheduler calls `comm.progress()` once per iteration. MPI backends (OpenMPI) often need manual progression; NCCL / NVSHMEM — less so.

`progress()` is **cheap and non-blocking**. Can be called mid-compute на idle threads.

---

## 6. Backend implementations

### 6.1. MpiHostStagingBackend (universal fallback)

**Always works** (anywhere MPI is available, даже без GPU-aware).

Flow:
```
send_temporal_packet(packet, dest):
    # 1. copy device → pinned host buffer
    cudaMemcpyAsync(host_buf, device_buf, size, D2H, stream_comm)
    cudaEventRecord(send_event, stream_comm)

    # 2. when event done, post MPI_Isend
    on send_event ready:
        MPI_Isend(host_buf, size, dest, tag, comm, &request)

    # record request для progress()
```

**Cost:** extra D2H + H2D copy = 2× PCIe traversal. На PCIe Gen4 x16 это ~25 GB/s, noticeable for big packets (Mbytes range).

Recommendation: **always available**, use for small clusters without IB-CUDA integration. Production — замена на GpuAwareMpi.

### 6.2. GpuAwareMpiBackend

**Requires:** MPI с CUDA support (OpenMPI + UCX, MVAPICH2-GDR, Cray MPI, Spectrum MPI on Summit).

Flow:
```
send_temporal_packet(packet, dest):
    MPI_Isend(device_buf, size, dest, tag, comm, &request)
    # MPI library handles device → network напрямую (если IB verbs с CUDA)
```

**Cost:** best possible latency; zero-copy for large transfers.

Requirement check at init: probe via `MPIX_Query_cuda_support()` или try MPI_Isend with device pointer; if fails — fallback to MpiHostStaging with warning.

**Implementation (T7.3, M7):**

- `tdmd::comm::is_cuda_aware_mpi()` — runtime probe cached via atomic; checks `MPIX_Query_cuda_support()` under `__has_include(<mpi-ext.h>)`, then env overrides `OMPI_MCA_opal_cuda_support` / `MV2_USE_CUDA`. **Never aborts**; returns `false` on any MPI implementation without CUDA symbols. `reset_cuda_mpi_probe_cache_for_testing()` lets unit tests drive the cache.
- `GpuAwareMpiBackend` constructor throws `std::runtime_error` if the probe is negative. Engine preflight (T7.9) catches this and dispatches `MpiHostStagingBackend` as the fallback, satisfying SPEC §6.2 above.
- PIMPL firewall: public header pulls only TDMD types + abstract `CommBackend`; `<mpi.h>` / `<mpi-ext.h>` live in the `.cpp`. Downstream targets including `gpu_aware_mpi_backend.hpp` don't force an MPI transitive dependency.
- MPI tags: `kTagTemporal = 1011`, `kTagHalo = 1012`. Distinct from `MpiHostStagingBackend` (which uses `1001`) so a composed `HybridBackend` (T7.5) that shares an MPI communicator across both transports cannot crosstalk at drain.
- Halo wire format (T7.3): mirror of `TemporalPacket` serializer — see §4.2; CRC32 is validated **before** payload allocation so a corrupted `payload_size` header cannot trigger a multi-GB alloc. Telemetry: `dropped_halo_crc_count()`, `dropped_halo_version_count()`.
- `send_migration_packet` is a no-op stub at T7.3. Migration routing over the outer communicator is wired by `HybridBackend` (T7.5) + engine Pattern 2 (T7.9).
- At T7.3 the halo send path uses a host `std::vector<uint8_t>` buffer (CUDA-aware MPI accepts host and device pointers interchangeably, so this is semantically identical — GPU-direct payload routing will land when the halo snapshot builder writes directly into pinned/device memory in T7.9).

### 6.3. NcclBackend

**Best for:** intra-node, SM-to-SM transfers.

Flow:
```
send_temporal_packet(packet, dest):
    ncclSend(device_buf, size, ncclFloat, dest, nccl_comm, stream_comm)
```

NCCL especially fast для:
- NVLink-connected GPUs;
- intra-node collectives (AllReduce, AllGather);
- strong scaling within node.

**Limitation:** NCCL не работает across nodes без NVSwitch. Для multi-node Pattern 2 — используется только inner level.

**Implementation (T7.4, M7):**

- `tdmd::comm::is_nccl_available()` / `nccl_runtime_version()` — runtime probe cached via atomic; wraps `ncclGetVersion()` when the NCCL SDK is linked, returns `false` / `0` when it isn't. Never aborts. `reset_nccl_probe_cache_for_testing()` drives the cache for unit tests.
- `NcclBackend` constructor throws `std::runtime_error` on probe-negative. Engine preflight (T7.9) catches this and routes the inner level to `GpuAwareMpiBackend` or `MpiHostStagingBackend` per §6.4. `TDMD_ENABLE_NCCL` is a top-level CMake option that auto-disables when NCCL headers / library aren't found; the stub build path is identical (constructor always throws) so downstream code doesn't branch.
- PIMPL firewall: public header pulls only TDMD types + abstract `CommBackend`. `<nccl.h>` + `<cuda_runtime.h>` + `<mpi.h>` live in the `.cpp` TU.
- Init protocol: rank 0 generates `ncclUniqueId` via `ncclGetUniqueId`; broadcast over a sidecar `MPI_Comm_dup`; all ranks call `ncclCommInitRank`. Each rank picks `device_id = rank % visible_gpus` — MPS-compatible on single-GPU dev boxes, one-GPU-per-rank on multi-GPU nodes.
- Dedicated `cudaStream_t` per backend instance. NCCL ops enqueue on it; collectives sync the stream before returning. T7.5 HybridBackend can share an external stream for overlap with compute.
- **Determinism (D-M5-9 + D-M7-4):** NCCL is transport-only. `global_sum_double` does `ncclAllGather` → host pinned scratch → `kahan_sum_ordered` (the same rank-ordered Kahan fold used by `deterministic_sum_double(MPI_Comm)`). The M5 thermo byte-exact chain (D-M5-12) extends through the NCCL path by construction: the only floats that touch NCCL are the per-rank local sums; the global reduction order is identical to the MPI path.
- `global_max_double`: direct `ncclAllReduce(ncclMax)`. Max is associative — no host fold needed.
- `barrier`: `MPI_Barrier` on the sidecar communicator — NCCL has no explicit barrier primitive.
- Temporal / halo / migration: no-op stubs at T7.4. NCCL P2P requires `ncclGroupStart`/`ncclGroupEnd` pairing which is incompatible with the scheduler's `Iprobe`-drain model; `HybridBackend` (T7.5) routes these through the outer `GpuAwareMpi` backend instead. Inner-TD grouped dispatch is a T7.9 item.
- **Version warning** (D-M7-4): `NcclBackend::initialize` prints a `stderr` warning if `nccl_runtime_version() < 2.18`. Non-fatal so dev boxes with distro-pinned 2.12/2.16 can still exercise the path.

### 6.4. HybridBackend (Pattern 2 default)

Combines:
- **inner**: NCCL (if intra-node) или GpuAwareMpi (fallback);
- **outer**: GpuAwareMpi (inter-node) или MpiHostStaging (fallback).

Automatic dispatch: `send_temporal_packet` → NCCL; `send_subdomain_halo` → MPI.

**This is the production default in Pattern 2.**

**Routing rules (T7.2 clarification, T7.5 implementation):**

| Operation | Backend selection | Rationale |
|---|---|---|
| `send_temporal_packet` (intra-subdomain TD) | inner backend (`NcclBackend` if node-local; `GpuAwareMpiBackend` else; `MpiHostStagingBackend` last-resort) | TD packets — high-frequency intra-subdomain; NCCL latency on NVLink wins on intra-node ranks |
| `send_subdomain_halo` (inter-subdomain SD) | outer backend (`GpuAwareMpiBackend` if available; `MpiHostStagingBackend` else) | SD halo — lower-frequency inter-node; MPI is the only universal cross-node transport at v1 (NCCL multi-node — NVSwitch-only, deferred to NvshmemBackend research) |
| `global_sum_double` (collectives §7) | preferentially inner if all ranks intra-node, else outer | Reference profile uses `deterministic_sum_double` regardless of backend (§7.2) |
| `progress()` | calls both backends' `progress()` once per iteration | scheduler invariant: one progress call per `select_ready_tasks()` |

Topology resolution — runtime probe at `init()`: per-rank `cudaDeviceGetP2PStatus` against rank-0's GPU, plus `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` to identify node-local subset. Per-rank decision cached in `BackendInfo` struct, surfaced to `cli/explain` per cli/SPEC.

**Pattern 2 startup contract (cross-link scheduler/SPEC §2.4):** at `SimulationEngine::init()` after `HybridBackend::init()`, runtime constructs `OuterSdCoordinator(grid, K_max)` (master §12.7a / scheduler/SPEC §2.4) and binds the outer backend's `drain_halo_arrived` poll into the coordinator's input pipeline. Inner-backend send/receive остаётся owned by `InnerTdScheduler` без изменений relative to M5.

**Implementation (T7.5, M7):**

- `src/comm/{include/tdmd/comm/hybrid_backend.hpp, hybrid_backend.cpp}` — composition, **not** duplication. Owns `unique_ptr<CommBackend>` for inner + outer plus a `TopologyResolver` (§3.4); each method dispatches per the routing table above. No I/O of its own — every wire-level operation is a forward call.
- Construction policy is policy-free: caller passes already-built inner + outer backends. The fallback chain (try `NcclBackend`, fall back to `GpuAwareMpiBackend`, fall back to `MpiHostStagingBackend`) lives in the engine preflight (T7.9), not in `HybridBackend` itself. Constructor throws `std::invalid_argument` on null backends.
- `initialize` calls inner first, then outer; validates `inner->rank() == outer->rank()` and `inner->nranks() == outer->nranks()` and `inner->nranks() == grid.total()` (D-M7-2 guard). Mismatch → `std::runtime_error` rather than silent corruption later.
- `shutdown` reverses construction: outer first, inner second — collectives remain available during outer teardown.
- `global_sum_double` / `global_max_double` / `barrier` route to **inner only** so the D-M5-12 byte-exact thermo chain stays on a single reduction tree; never use the outer backend's collective even if the outer probe succeeds.
- `progress` ticks **both** backends once per call; deterministic order (inner then outer) keeps any progress-driven side effect reproducible.
- `info()` composes `name = "HybridBackend(inner=...,outer=...)"` and unions both backends' capability lists so `cli/explain` shows the effective transport surface.
- Unit coverage in `tests/comm/test_hybrid_backend.cpp` (always-built, no MPI): a `SpyBackend` mock records each call so the routing matrix, init parity guard, D-M7-2 grid mismatch guard, Pattern 1 single-subdomain empty-peer behavior, capability union, and shutdown order are verified deterministically across all CI flavors. The 4-rank end-to-end exercise lands with `SimulationEngine` Pattern 2 wire (T7.9).

### 6.5. RingBackend (legacy / anchor-test)

Implements только:
```
send_temporal_packet(packet, dest):
    assert dest == (rank + 1) % nranks   # only ring-next allowed
    # Simple MPI_Isend directly в ring neighbor
```

Used **только** для anchor-test §13.3 master spec — воспроизведение TIME-MD Андреева. Namely:
- one axis (Z) разбиения;
- linear zone ordering;
- ring communication;
- no sender-to-arbitrary-rank.

В production не используется.

### 6.6. NvshmemBackend (research / v2+)

One-sided RMA через NVSHMEM. Experimental, для future HPC systems с tight GPU integration (H100 with NVSwitch + NVLink Network).

**Not production in v1**. Reserved для v2+ research mode.

---

## 7. Collectives

### 7.1. Global energy / temperature aggregation

Each rank computes local PE, KE, virial. Global sum через:

```
global_pe = comm.global_sum_double(local_pe)
global_ke = comm.global_sum_double(local_ke)
```

Implementation:
- **NcclBackend**: `ncclAllReduce(sum)`;
- **MpiBackend**: `MPI_Allreduce(MPI_SUM)`;
- **RingBackend**: sequential add around ring + broadcast.

### 7.2. Deterministic reduction

В Reference profile: **fixed reduction tree**, reproducible byte-for-byte.

`MPI_Allreduce` sums в implementation-defined order → **not deterministic across MPI implementations**. В Reference mode, мы используем our own tree reduction:

```
function deterministic_sum_double(local, comm):
    # Binary tree reduction:
    # depth log2(P) levels
    # at each level, pairwise add, fixed ordering
    values = gather_to_all(local, comm)      # N values, ordered by rank ascending
    # Sum в fixed order
    result = 0.0
    for v in values (rank-ordered):
        result = kahan_sum(result, v)
    return result
```

Kahan summation eliminates floating-point associativity noise.

Cost: slightly slower than native Allreduce (2-3x). Acceptable in Reference, off в Production/Fast.

### 7.3. Barrier

`comm.barrier()` — global synchronization. Used в:

- end of each iteration в Reference profile;
- after migration в Pattern 2 (v1: sync barrier);
- startup / shutdown.

In FastExperimental — barriers are expensive, minimized. But Reference-mode всегда has barrier at end of iteration для determinism.

---

## 8. Correctness properties

### 8.1. Message ordering

**Within one (sender, receiver) pair:** messages arrive in send order.

This is **not** guaranteed by MPI by default (different tags могут reorder). To guarantee:
- use single tag for temporal packets between pair;
- rely на MPI ordering spec для point-to-point same tag.

For NCCL: ordering guaranteed within one stream.

### 8.2. No loss

Sent packet must either arrive or `progress()` reports error. No silent drop.

- MPI: failures surface через `MPI_ERROR` return codes;
- NCCL: errors через `ncclCommGetAsyncError`;
- comm backend elevates to scheduler's `on_comm_error` callback.

### 8.3. No duplication

Same packet should not be delivered twice. Guaranteed by reliable transport (MPI, NCCL); not an issue for RingBackend.

### 8.4. No corruption

CRC32 validation catches bit-level corruption (rare but non-zero).

---

## 9. Tests

### 9.1. Unit tests (backend-local)

- **Pack/unpack round-trip:** random atoms → pack → unpack → should match exactly;
- **CRC32:** correct for canonical input; detects single-bit flip;
- **Protocol versioning:** sender v1 → receiver v2 → clear error;
- **Buffer allocation:** stress test with many concurrent sends;
- **Empty packet:** 0 atoms — edge case.

### 9.2. Integration tests (multi-rank)

Test harness: spawn 2, 4, 8 ranks with MPI, exchange packets.

- **2-rank ring:** rank 0 sends to rank 1, rank 1 sends to rank 0; n packets each; all arrive.
- **4-rank mesh:** каждый rank sends to all 4 neighbors; all packets arrive.
- **Latency measurement:** ping-pong minimum between 2 ranks, same node and different nodes; report backend capabilities.
- **Bandwidth measurement:** large packet stream; measure GB/s; update `BackendInfo.measured_bw`.

### 9.3. Determinism tests

- **Same run twice:** identical message flow (sequence of sends, sequence of receives) bitwise.
- **Collective reproducibility:** `deterministic_sum_double` should produce identical result across 10 runs.
- **Different backends same result:** MpiHostStaging vs GpuAwareMpi — идентичные packet contents (not necessarily timing, но content).

### 9.4. Fault injection tests

- **Kill receiver mid-transfer:** sender should see error, not hang.
- **Corrupt packet (flip random bits):** CRC should catch, receiver rejects.
- **Network slowdown simulation:** add artificial latency; system должно completes, just slower.

### 9.5. Anchor-test integration

Part of §13.3 anchor test: RingBackend delivers correct packet sequence matching Andreev's TIME-MD behavior for Al FCC 10⁶ benchmark.

---

## 10. Telemetry

Metrics (per backend):

```
comm.packets_sent_total                           (by type: temporal | halo | migration)
comm.packets_received_total
comm.bytes_sent_total
comm.bytes_received_total
comm.send_latency_ms_avg
comm.send_latency_ms_p99
comm.bytes_per_second_measured
comm.crc_failures_total
comm.protocol_version_mismatches_total
comm.fallback_to_host_staging_count               (if applicable)

comm.inner_traffic_bytes_total                    (Pattern 2)
comm.outer_traffic_bytes_total                    (Pattern 2)
comm.inner_vs_outer_ratio                         (expected высокий: TD scales inner, not outer)
```

NVTX ranges:
- `comm::send_temporal_packet`;
- `comm::drain_arrived_temporal`;
- `comm::global_sum_double`;
- `comm::barrier`.

---

## 11. Configuration и tuning

### 11.1. Auto-detection

On startup:

```
function select_backend(config):
    if config.backend == "auto":
        probes = []
        probes.append(MpiHostStaging)            # always
        if has_cuda_aware_mpi():
            probes.append(GpuAwareMpi)
        if has_nccl():
            probes.append(Nccl)
        if has_nvshmem() and config.research_mode:
            probes.append(Nvshmem)

        if pattern == Pattern2:
            return Hybrid(inner=best(Nccl, GpuAwareMpi), outer=best(GpuAwareMpi, MpiHostStaging))
        else:
            return best_probe(probes)
    else:
        return explicit_backend(config.backend)
```

### 11.2. Backend-specific tuning

Each backend имеет свои knobs:

**MpiHostStaging:**
- `staging_buffer_pool_size` — default 64 buffers × 4 MB;
- `use_pinned_memory` — default true.

**NcclBackend:**
- `nccl_channel_count` — how many parallel channels (default 4);
- `nccl_algorithm` — ring | tree | auto.

**GpuAwareMpi:**
- `mpi_gpudirect_policy` — use GPUDirect RDMA if available.

All exposed через `tdmd.yaml` `comm.backend_tuning` block, with defaults sensible for most users.

### 11.3. Preflight diagnostics

`tdmd validate case.yaml` reports:

```
Comm backend diagnostics:
  Selected backend: HybridBackend
  Inner: NCCL (intra-node via NVLink detected, BW ≥ 200 GB/s)
  Outer: GpuAwareMpi (OpenMPI 4.1 + UCX + GDRCopy)
  Measured latency: 8 us (inner), 25 us (outer)
  Measured bandwidth: 180 GB/s (inner), 22 GB/s (outer)
  Recommended K: 4 (from perfmodel)
  Warnings: none
```

---

## 12. Roadmap alignment

| Milestone | Comm deliverable |
|---|---|
| M1 | skeleton `CommBackend`, nothing functional yet |
| M2 | `MpiHostStagingBackend`, basic collectives, CPU-only |
| M4 | backend used by scheduler в single-node TD (multi-thread faking multi-rank для testing) |
| **M5** | **`MpiHostStagingBackend` full**, **`RingBackend`**, K-batching integration; **anchor-test §13.3 passes** |
| M6 | `GpuAwareMpiBackend`, `NcclBackend` (intra-node) |
| **M7** | **`HybridBackend`**, outer halo protocol, Pattern 2 integration, migration packets |
| M8+ | NvshmemBackend (research); auto-tuning improvements; multi-rail / multi-NIC optimization |

---

## 13. Open questions

1. **MPI vs NCCL для intra-node small messages** — NCCL latency worse than MPI для small packets (<1KB). Benchmark и define threshold.
2. **Cross-node NVSHMEM** — может быть enabled если both ends have GPUDirect Async. Requires cautious feature detection.
3. **Collective ring reduction ordering** — `deterministic_sum_double` использует linear sum. Better determinism via pair-wise Kahan-compensated? Perf tradeoff.
4. **Backend composition** — что если user хочет, например, NCCL для energy reductions но MPI для temporal packets? Needs configurable per-operation routing.
5. **Sensitivity to `MPI_THREAD_MULTIPLE` / `MPI_THREAD_SERIALIZED`** — TDMD spawns compute threads; MPI threading level matters. Document requirements, fallback если `MPI_THREAD_SERIALIZED`.
6. **Graceful degradation** — if `GpuAwareMpi` initially probes ok но fails at runtime, can we hot-swap to `MpiHostStaging`? Complicated because compute streams already reference GPU buffers.

---

---

## 14. Change log

- **2026-04-20** — **T7.14 — M7 acceptance gate landed; M7 milestone closed
  on the MpiHostStaging path.** `tests/integration/m7_smoke/` exercises
  the M7 Pattern 2 K=1 P_space=2 surface end-to-end on top of the M5
  `MpiHostStagingBackend` transport — `comm.backend: mpi_host_staging`
  is the canonical byte-exact path for the D-M7-10 golden chain (M3 ≡
  M4 ≡ M5 ≡ M6 ≡ M7). HybridBackend / NCCL / GpuAwareMPI paths are
  exercised by T7.4 / T7.5 unit gates (byte-exact vs MpiHostStaging on
  M5 fixture) and T7.11 scaling benchmark — deliberately NOT layered
  into the M7 smoke to keep the failure signal tight (one transport,
  one byte-exact gate). The M7 smoke step self-skips on public CI via
  `nvidia-smi -L` (Option A / D-M6-6), still asserts golden parity and
  infrastructure integrity on every PR. D-M5-9 deterministic Kahan
  reduction contract preserved through Pattern 2 (`OuterSdCoordinator`
  peer-halo canonicalisation feeds the same reduction tree).
- **2026-04-19** — **T7.5 — `HybridBackend` composition + `TopologyResolver`
  landed (M7).** New §3.4 (`TopologyResolver`) and §6.4 "Implementation"
  subsection authored. `HybridBackend` is policy-free composition of inner
  (NCCL/GpuAwareMpi/MpiHostStaging fallback chain — choice owned by engine
  preflight T7.9) and outer (GpuAwareMpi/MpiHostStaging) backends, dispatching
  per the §6.4 routing matrix. Collectives go inner-only to preserve the
  D-M5-12 byte-exact thermo chain on a single reduction tree. `TopologyResolver`
  is the single source of truth for D-M7-2 (`owner_ranks(sd) = {sd}` at M7) and
  Cartesian Moore-neighborhood enumeration (up to 26 peers in 3D), with
  deterministic z→y→x iteration so the same grid produces the same peer vector
  on every rank. Pattern 1 (single-subdomain) returns empty peers, so
  HybridBackend wraps inner without ever touching the outer path — bit-identical
  to the inner backend alone. Unit coverage via `SpyBackend` mock — runs in
  every CI flavor (cpu-strict, mixed, default) without requiring MPI / NCCL /
  CUDA. End-to-end 4-rank dispatch lands with the engine wire in T7.9.

- **2026-04-19** — **T7.4 — `NcclBackend` implementation landed (M7).**
  Second concrete delivery of §6.3 alongside T7.3's §6.2. Top-level
  `TDMD_ENABLE_NCCL` option auto-detects NCCL at configure time (falls
  back to a stub-throw body on absence). Probe-gated constructor mirrors
  the T7.3 pattern: engine preflight (T7.9) catches the runtime_error
  and routes the inner TD level through `GpuAwareMpiBackend` or
  `MpiHostStagingBackend` per §6.4. `global_sum_double` uses
  `ncclAllGather` + host-side `kahan_sum_ordered`, preserving the
  D-M5-9 deterministic reduction contract and extending the D-M5-12
  byte-exact thermo chain through the NCCL path. No SPEC-surface
  changes — §6.3 was already authored; T7.4 is its implementation.
  2-rank unit test SKIPs cleanly when NCCL or CUDA is unavailable
  (Option A CI posture). See §6.3 "Implementation" subsection for
  init protocol (uniqueId MPI-broadcast), stream ownership, temporal-
  stub deferral rationale (T7.5 HybridBackend routing), and D-M7-4
  version warning.

- **2026-04-19** — **T7.3 — `GpuAwareMpiBackend` implementation landed
  (M7).** First concrete delivery of §6.2: probe-gated constructor,
  PIMPL firewall, halo + temporal multiplexed on distinct MPI tags
  (1011 / 1012, distinct from `MpiHostStaging`'s 1001), 2-rank halo
  echo test with runtime SKIP on probe-negative nodes (Option A CI
  posture — no self-hosted CUDA-aware-MPI runner). `HaloPacket` wire
  format mirrors `TemporalPacket`: 26 B header + payload + 4 B CRC32,
  CRC validated before payload allocation. No SPEC-surface changes —
  §6.2 was already specified; T7.3 is its implementation. See §6.2
  "Implementation" subsection for integration notes (engine preflight
  fallback contract, PIMPL rationale, migration-packet stub deferral
  to T7.5/T7.9).

- **2026-04-19** — **T7.2 — Pattern 2 SPEC integration sister edit
  (M7 entry).** Pure SPEC delta paired with `scheduler/SPEC §2.4 / §2.5
  / §4.6` (T7.2 main delivery). Two clarifications, no new interface
  surface (`send_subdomain_halo` / `drain_halo_arrived` / `HaloPacket`
  uchanged from §2 + §4):
  - **§4.2 ownership boundary** — `HaloPacket = wire format owned by
    comm/`; receiver-side unpack into `HaloSnapshot` (scheduler/SPEC §4.6
    archive record) is owned by `OuterSdCoordinator::unpack_halo()`, not
    by comm. Comm's three obligations enumerated (deliver bytes, CRC32,
    eager commit). Per-payload meaning + archive lifecycle + eviction —
    explicitly defer-pointed to scheduler/SPEC §4.6, preserving master
    §8.2 ownership boundary.
  - **§6.4 HybridBackend routing rules + Pattern 2 startup contract** —
    4-row dispatch matrix (temporal → inner; halo → outer; collectives →
    inner-preferred; progress → both), topology resolution via
    `cudaDeviceGetP2PStatus` + `MPI_Comm_split_type(SHARED)` cached in
    `BackendInfo`. Cross-link to scheduler/SPEC §2.4: `OuterSdCoordinator`
    constructed at `SimulationEngine::init()` after `HybridBackend::init()`,
    outer-backend `drain_halo_arrived` poll bound into coordinator input.
    Inner-backend send/receive unchanged from M5. Concrete implementation
    of `HybridBackend` itself — T7.5 (depends on T7.3
    `GpuAwareMpiBackend` and T7.4 `NcclBackend`).

- **2026-04-19** — **M5 landed**. T5.2–T5.12 implemented the full M5
  comm surface: skeleton `CommBackend` abstract interface + types +
  `TDMD_ENABLE_MPI` optional build flag (T5.2); `TemporalPacket` wire
  format — pack/unpack, CRC32 integrity, protocol version v1 (T5.3);
  `MpiHostStagingBackend` — 2-rank ping-pong, deterministic reduction
  via `deterministic_sum_double` (D-M5-9), Kahan-compensated ring sum
  (T5.4); `RingBackend` — 4-rank ring, ring-sum bit-exact, non-ring-dest
  assert in Reference (T5.5); K-batching pipeline integration (T5.6);
  scheduler peer dispatch (T5.7); multi-rank `SimulationEngine` with
  deterministic thermo reduction (T5.8, D-M5-12 byte-exact chain); M5
  integration smoke — K=1 P=2 MpiHostStaging thermo byte-exact to M4
  golden, CI-gated on openmpi-bin install (T5.12).
  - Anchor-test (T3) — dissertation reproduction harness
    (`verify/harness/anchor_test_runner/`, T5.11) is the primary M5
    science gate; local slow-tier, not CI-wired per D-M5-13.
  - GPU-aware MPI backend deferred to M6; Pattern 2 / `HybridBackend`
    deferred to M7.

- **2026-04-19** — **M6 closure (T6.13) — no SPEC-surface changes;
  `MpiHostStagingBackend` confirmed as the M6 transport.** Per D-M6-3
  the GPU path stages force-pair halos through pinned host buffers from
  `DevicePool` and reuses the existing M5 backend unchanged (no
  CUDA-aware `MPI_Send(devptr,…)` in v1). The M6 integration smoke
  (`tests/integration/m6_smoke/`, 2-rank K=1 `runtime.backend: gpu`)
  validates end-to-end: GPU forces → D2H into pinned → `MpiHostStaging`
  Kahan-ring reduction → deterministic thermo = M5 golden byte-for-byte.
  This closes D-M5-12 through the GPU era and confirms §3 (interface),
  §5.1 (TemporalPacket), §5.2 (MpiHostStaging state machine), and the
  `deterministic_sum_double` contract hold identical on GPU-era traffic
  patterns. Pattern 2 / `HybridBackend` / GPU-aware MPI (`cudaMemcpy`
  elision via `MPI_Send` on a device pointer) remain deferred to M7+ per
  master spec §14.

---

*Конец comm/SPEC.md v1.0, дата: 2026-04-16.*
