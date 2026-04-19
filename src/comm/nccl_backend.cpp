// SPEC: docs/specs/comm/SPEC.md §6.3 (NcclBackend), §7
// Master spec: §10, §12.6, §14 M7, §D.14
// Exec pack: docs/development/m7_execution_pack.md T7.4

#include "tdmd/comm/nccl_backend.hpp"

#include "tdmd/comm/nccl_probe.hpp"

#if defined(TDMD_ENABLE_NCCL) && TDMD_ENABLE_NCCL

#include "tdmd/comm/deterministic_reduction.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

namespace tdmd::comm {

namespace {

// NCCL ≥ 2.18 is the D-M7-4 floor. We warn-not-throw on older NCCL so dev
// machines with distro-pinned 2.12/2.16 can still exercise the code paths.
// ncclGetVersion encodes as MAJOR*1000 + MINOR*100 + PATCH.
constexpr int kNcclMinVersion = 2 * 1000 + 18 * 100;  // 2.18.0

[[noreturn]] void throw_nccl(ncclResult_t rc, const char* what) {
  throw std::runtime_error(std::string{"NcclBackend: "} + what +
                           " failed: " + ncclGetErrorString(rc));
}

[[noreturn]] void throw_cuda(cudaError_t rc, const char* what) {
  throw std::runtime_error(std::string{"NcclBackend: "} + what +
                           " failed: " + cudaGetErrorString(rc));
}

void check_nccl(ncclResult_t rc, const char* what) {
  if (rc != ncclSuccess) {
    throw_nccl(rc, what);
  }
}

void check_cuda(cudaError_t rc, const char* what) {
  if (rc != cudaSuccess) {
    throw_cuda(rc, what);
  }
}

}  // namespace

struct NcclBackend::Impl {
  CommConfig config{};

  // Dedicated MPI communicator for the uniqueId broadcast and for fallback
  // collectives (e.g. barrier). NCCL itself doesn't provide a barrier.
  MPI_Comm comm{MPI_COMM_NULL};
  bool owns_mpi_init{false};

  ncclComm_t nccl_comm{nullptr};
  int rank{-1};
  int nranks{0};
  int device_id{-1};
  cudaStream_t stream{nullptr};

  // Pinned scratch for the allgather'd doubles: one slot per rank. Allocated
  // once at init; reused per collective call. Pinned-host keeps the D2H copy
  // cheap and avoids a per-call cudaMallocHost.
  double* d_scratch_send{nullptr};       // 1 double on device
  double* d_scratch_recv{nullptr};       // nranks doubles on device
  std::vector<double> h_scratch_recv{};  // nranks doubles host-side

  // Telemetry counters — kept to match GpuAwareMpiBackend's interface so
  // HybridBackend (T7.5) can aggregate across backends uniformly. NCCL
  // itself doesn't surface CRC/version drops (it manages framing) but the
  // counters stay at 0 rather than throwing "not implemented".
  std::uint64_t dropped_crc{0};
  std::uint64_t dropped_version{0};
  std::uint64_t dropped_halo_crc{0};
  std::uint64_t dropped_halo_version{0};
};

NcclBackend::NcclBackend() : impl_(std::make_unique<Impl>()) {
  // Hard refusal at construction if NCCL isn't linked or ncclGetVersion
  // fails. Engine preflight (T7.9) catches this and routes the inner level
  // to GpuAwareMpi / MpiHostStaging. SPEC §6.3 makes the probe mandatory.
  if (!is_nccl_available()) {
    throw std::runtime_error(
        "NcclBackend: NCCL runtime not available "
        "(ncclGetVersion failed or library not linked). Engine should fall "
        "back to GpuAwareMpiBackend or MpiHostStagingBackend.");
  }
}

NcclBackend::~NcclBackend() {
  if (!impl_) {
    return;
  }
  // Best-effort cleanup. We don't throw from the destructor; any leftover
  // NCCL state on an unclean shutdown is a dev error that will surface via
  // an abort inside NCCL itself.
  if (impl_->nccl_comm != nullptr) {
    ncclCommDestroy(impl_->nccl_comm);
    impl_->nccl_comm = nullptr;
  }
  if (impl_->stream != nullptr) {
    cudaStreamDestroy(impl_->stream);
    impl_->stream = nullptr;
  }
  if (impl_->d_scratch_send != nullptr) {
    cudaFree(impl_->d_scratch_send);
    impl_->d_scratch_send = nullptr;
  }
  if (impl_->d_scratch_recv != nullptr) {
    cudaFree(impl_->d_scratch_recv);
    impl_->d_scratch_recv = nullptr;
  }
  if (impl_->comm != MPI_COMM_NULL) {
    MPI_Comm_free(&impl_->comm);
    impl_->comm = MPI_COMM_NULL;
  }
  if (impl_->owns_mpi_init) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
    impl_->owns_mpi_init = false;
  }
}

NcclBackend::NcclBackend(NcclBackend&&) noexcept = default;
NcclBackend& NcclBackend::operator=(NcclBackend&&) noexcept = default;

void NcclBackend::initialize(const CommConfig& config) {
  impl_->config = config;

  // D-M7-4: warn (not throw) on sub-2.18 NCCL so dev boxes with distro pins
  // still exercise the code path. Written to stderr because the backend
  // doesn't own a logger at M7; CLI/telemetry can capture this in T7.9.
  const int v = nccl_runtime_version();
  if (v > 0 && v < kNcclMinVersion) {
    std::fprintf(stderr,
                 "[tdmd::comm::NcclBackend] WARNING: NCCL runtime version %d.%d is below "
                 "the recommended %d.%d; some collectives may be slower or less robust.\n",
                 v / 1000,
                 (v % 1000) / 100,
                 kNcclMinVersion / 1000,
                 (kNcclMinVersion % 1000) / 100);
  }

  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited) {
    int provided = 0;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SINGLE, &provided);
    impl_->owns_mpi_init = true;
  }

  MPI_Comm_dup(MPI_COMM_WORLD, &impl_->comm);
  MPI_Comm_rank(impl_->comm, &impl_->rank);
  MPI_Comm_size(impl_->comm, &impl_->nranks);

  // Pick a CUDA device. Round-robin by rank so 2 ranks on 1 GPU still share
  // (MPS-compatible); on 2+ GPU nodes each rank lands on its own device.
  int visible_devices = 0;
  check_cuda(cudaGetDeviceCount(&visible_devices), "cudaGetDeviceCount");
  if (visible_devices <= 0) {
    throw std::runtime_error(
        "NcclBackend: no CUDA devices visible; NCCL requires at least one GPU.");
  }
  impl_->device_id = impl_->rank % visible_devices;
  check_cuda(cudaSetDevice(impl_->device_id), "cudaSetDevice");
  check_cuda(cudaStreamCreate(&impl_->stream), "cudaStreamCreate");

  // NCCL uniqueId broadcast: rank 0 generates, MPI_Bcast to all.
  ncclUniqueId id{};
  if (impl_->rank == 0) {
    check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
  }
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, impl_->comm);

  check_nccl(ncclCommInitRank(&impl_->nccl_comm, impl_->nranks, id, impl_->rank),
             "ncclCommInitRank");

  // Scratch buffers for allgather-based deterministic sum.
  check_cuda(cudaMalloc(&impl_->d_scratch_send, sizeof(double)), "cudaMalloc send");
  check_cuda(
      cudaMalloc(&impl_->d_scratch_recv, sizeof(double) * static_cast<std::size_t>(impl_->nranks)),
      "cudaMalloc recv");
  impl_->h_scratch_recv.assign(static_cast<std::size_t>(impl_->nranks), 0.0);
}

void NcclBackend::shutdown() {
  if (impl_->nccl_comm != nullptr) {
    ncclCommDestroy(impl_->nccl_comm);
    impl_->nccl_comm = nullptr;
  }
  if (impl_->stream != nullptr) {
    cudaStreamDestroy(impl_->stream);
    impl_->stream = nullptr;
  }
  if (impl_->d_scratch_send != nullptr) {
    cudaFree(impl_->d_scratch_send);
    impl_->d_scratch_send = nullptr;
  }
  if (impl_->d_scratch_recv != nullptr) {
    cudaFree(impl_->d_scratch_recv);
    impl_->d_scratch_recv = nullptr;
  }
  if (impl_->comm != MPI_COMM_NULL) {
    MPI_Comm_free(&impl_->comm);
    impl_->comm = MPI_COMM_NULL;
  }
  if (impl_->owns_mpi_init) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
    impl_->owns_mpi_init = false;
  }
}

// M7 scope per exec pack: temporal point-to-point via NCCL is intentionally
// a no-op stub at T7.4. NCCL P2P requires group-scoped paired send/recv,
// which is incompatible with the scheduler's "Iprobe-drain" model. T7.5
// HybridBackend routes temporal packets through GpuAwareMpi for the same
// communicator, and the NCCL fabric stays for collectives + future inner-TD
// grouped dispatch (T7.9).
void NcclBackend::send_temporal_packet(const TemporalPacket&, int) {}

std::vector<TemporalPacket> NcclBackend::drain_arrived_temporal() {
  return {};
}

// Halo + migration stay on the outer backend per SPEC §6.4 routing matrix.
void NcclBackend::send_subdomain_halo(const HaloPacket&, int) {}
std::vector<HaloPacket> NcclBackend::drain_arrived_halo() {
  return {};
}
void NcclBackend::send_migration_packet(const MigrationPacket&, int) {}
std::vector<MigrationPacket> NcclBackend::drain_arrived_migrations() {
  return {};
}

double NcclBackend::global_sum_double(double local) {
  // D-M5-9: NCCL is transport; reduction folds on the host via Kahan.
  // Allgather pattern mirrors deterministic_sum_double(MPI_Comm) exactly so
  // the result is bit-exact to the M5 baseline on the same rank ordering.
  check_cuda(cudaMemcpyAsync(impl_->d_scratch_send,
                             &local,
                             sizeof(double),
                             cudaMemcpyHostToDevice,
                             impl_->stream),
             "cudaMemcpy H2D");
  check_nccl(ncclAllGather(impl_->d_scratch_send,
                           impl_->d_scratch_recv,
                           1,
                           ncclDouble,
                           impl_->nccl_comm,
                           impl_->stream),
             "ncclAllGather");
  check_cuda(cudaMemcpyAsync(impl_->h_scratch_recv.data(),
                             impl_->d_scratch_recv,
                             sizeof(double) * static_cast<std::size_t>(impl_->nranks),
                             cudaMemcpyDeviceToHost,
                             impl_->stream),
             "cudaMemcpy D2H");
  check_cuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize");

  // Host-side Kahan fold — rank-ordered, matching
  // `deterministic_sum_double(MPI_Comm)` exactly.
  return kahan_sum_ordered(impl_->h_scratch_recv);
}

double NcclBackend::global_max_double(double local) {
  // Max is associative — no determinism concern. Single ncclAllReduce op.
  check_cuda(cudaMemcpyAsync(impl_->d_scratch_send,
                             &local,
                             sizeof(double),
                             cudaMemcpyHostToDevice,
                             impl_->stream),
             "cudaMemcpy H2D");
  check_nccl(ncclAllReduce(impl_->d_scratch_send,
                           impl_->d_scratch_send,
                           1,
                           ncclDouble,
                           ncclMax,
                           impl_->nccl_comm,
                           impl_->stream),
             "ncclAllReduce Max");
  double result = 0.0;
  check_cuda(cudaMemcpyAsync(&result,
                             impl_->d_scratch_send,
                             sizeof(double),
                             cudaMemcpyDeviceToHost,
                             impl_->stream),
             "cudaMemcpy D2H");
  check_cuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize");
  return result;
}

void NcclBackend::barrier() {
  // NCCL has no explicit barrier — MPI_Barrier on the sidecar communicator
  // is the correct and standard pattern.
  MPI_Barrier(impl_->comm);
}

void NcclBackend::progress() {
  // NCCL collectives are synchronous to the stream; progress is a no-op.
  // T7.5 HybridBackend may add stream polling if it owns async ops.
}

BackendInfo NcclBackend::info() const {
  BackendInfo i;
  i.name = "NcclBackend";
  i.protocol_version = kCommProtocolVersion;
  i.capabilities = {BackendCapability::GpuAwarePointers, BackendCapability::CollectiveOptimized};
  return i;
}

int NcclBackend::rank() const {
  return impl_->rank;
}

int NcclBackend::nranks() const {
  return impl_->nranks;
}

std::uint64_t NcclBackend::dropped_crc_count() const noexcept {
  return impl_ ? impl_->dropped_crc : 0;
}
std::uint64_t NcclBackend::dropped_version_count() const noexcept {
  return impl_ ? impl_->dropped_version : 0;
}
std::uint64_t NcclBackend::dropped_halo_crc_count() const noexcept {
  return impl_ ? impl_->dropped_halo_crc : 0;
}
std::uint64_t NcclBackend::dropped_halo_version_count() const noexcept {
  return impl_ ? impl_->dropped_halo_version : 0;
}

}  // namespace tdmd::comm

#else  // TDMD_ENABLE_NCCL

// Stub for builds without NCCL (NCCL SDK not found at configure time, or
// CUDA/MPI disabled). Constructor throws identically to the runtime-probe
// refusal; the engine preflight (T7.9) handles both uniformly.

#include <stdexcept>

namespace tdmd::comm {

struct NcclBackend::Impl {};

NcclBackend::NcclBackend() : impl_(nullptr) {
  throw std::runtime_error(
      "NcclBackend: built without TDMD_ENABLE_NCCL; rebuild with NCCL SDK "
      "installed (-DTDMD_ENABLE_NCCL=ON) to use this backend.");
}
NcclBackend::~NcclBackend() = default;
NcclBackend::NcclBackend(NcclBackend&&) noexcept = default;
NcclBackend& NcclBackend::operator=(NcclBackend&&) noexcept = default;

void NcclBackend::initialize(const CommConfig&) {}
void NcclBackend::shutdown() {}
void NcclBackend::send_temporal_packet(const TemporalPacket&, int) {}
std::vector<TemporalPacket> NcclBackend::drain_arrived_temporal() {
  return {};
}
void NcclBackend::send_subdomain_halo(const HaloPacket&, int) {}
std::vector<HaloPacket> NcclBackend::drain_arrived_halo() {
  return {};
}
void NcclBackend::send_migration_packet(const MigrationPacket&, int) {}
std::vector<MigrationPacket> NcclBackend::drain_arrived_migrations() {
  return {};
}
double NcclBackend::global_sum_double(double local) {
  return local;
}
double NcclBackend::global_max_double(double local) {
  return local;
}
void NcclBackend::barrier() {}
void NcclBackend::progress() {}
BackendInfo NcclBackend::info() const {
  BackendInfo i;
  i.name = "NcclBackend(disabled)";
  return i;
}
int NcclBackend::rank() const {
  return -1;
}
int NcclBackend::nranks() const {
  return 0;
}
std::uint64_t NcclBackend::dropped_crc_count() const noexcept {
  return 0;
}
std::uint64_t NcclBackend::dropped_version_count() const noexcept {
  return 0;
}
std::uint64_t NcclBackend::dropped_halo_crc_count() const noexcept {
  return 0;
}
std::uint64_t NcclBackend::dropped_halo_version_count() const noexcept {
  return 0;
}

}  // namespace tdmd::comm

#endif  // TDMD_ENABLE_NCCL
