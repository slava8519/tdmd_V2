// SPEC: docs/specs/comm/SPEC.md §6.2 (GpuAwareMpiBackend), §7
// Master spec: §10, §12.6, §D.14
// Exec pack: docs/development/m7_execution_pack.md T7.3

#include "tdmd/comm/gpu_aware_mpi_backend.hpp"

#include "tdmd/comm/cuda_mpi_probe.hpp"

#if defined(TDMD_ENABLE_MPI) && TDMD_ENABLE_MPI

#include "tdmd/comm/deterministic_reduction.hpp"
#include "tdmd/comm/halo_packet_serializer.hpp"
#include "tdmd/comm/packet_serializer.hpp"

#include <list>
#include <stdexcept>
#include <utility>

#include <mpi.h>

namespace tdmd::comm {

namespace {

// MPI tags. Distinct from MpiHostStaging so a HybridBackend that owns
// both backends on the same MPI_COMM doesn't confuse drain_arrived_*
// across the two transports. Halo + temporal each get their own tag so
// a single backend instance can multiplex them on the same communicator.
constexpr int kTagTemporal = 1011;
constexpr int kTagHalo = 1012;

}  // namespace

struct GpuAwareMpiBackend::Impl {
  CommConfig config{};
  MPI_Comm comm{MPI_COMM_NULL};
  bool owns_mpi_init{false};
  int rank{-1};
  int nranks{0};

  // Per-tag pending lists — each in-flight Isend owns its serialized buffer
  // and must outlive the request. Stored as std::list for iterator stability
  // across reap-and-erase passes in progress().
  struct PendingSend {
    std::vector<std::uint8_t> buffer;
    MPI_Request request{MPI_REQUEST_NULL};
  };
  std::list<PendingSend> pending_temporal;
  std::list<PendingSend> pending_halo;

  std::uint64_t dropped_crc{0};
  std::uint64_t dropped_version{0};
  std::uint64_t dropped_halo_crc{0};
  std::uint64_t dropped_halo_version{0};

  static void reap(std::list<PendingSend>& pending) {
    for (auto it = pending.begin(); it != pending.end();) {
      int flag = 0;
      MPI_Test(&it->request, &flag, MPI_STATUS_IGNORE);
      if (flag) {
        it = pending.erase(it);
      } else {
        ++it;
      }
    }
  }
};

GpuAwareMpiBackend::GpuAwareMpiBackend() : impl_(std::make_unique<Impl>()) {
  // Hard refusal at construction: callers (engine preflight in T7.9) catch
  // this and dispatch the fallback backend. SPEC §6.2 explicitly requires
  // the probe to gate construction.
  if (!is_cuda_aware_mpi()) {
    throw std::runtime_error(
        "GpuAwareMpiBackend: CUDA-aware MPI not detected at runtime "
        "(MPIX_Query_cuda_support() == 0 and no env override). Falling back "
        "to MpiHostStagingBackend is the engine's responsibility.");
  }
}

GpuAwareMpiBackend::~GpuAwareMpiBackend() {
  if (!impl_) {
    return;  // moved-from
  }
  // Best-effort cleanup; mirrors MpiHostStagingBackend dtor.
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

GpuAwareMpiBackend::GpuAwareMpiBackend(GpuAwareMpiBackend&&) noexcept = default;
GpuAwareMpiBackend& GpuAwareMpiBackend::operator=(GpuAwareMpiBackend&&) noexcept = default;

void GpuAwareMpiBackend::initialize(const CommConfig& config) {
  impl_->config = config;

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
}

void GpuAwareMpiBackend::shutdown() {
  for (auto& s : impl_->pending_temporal) {
    MPI_Wait(&s.request, MPI_STATUS_IGNORE);
  }
  impl_->pending_temporal.clear();
  for (auto& s : impl_->pending_halo) {
    MPI_Wait(&s.request, MPI_STATUS_IGNORE);
  }
  impl_->pending_halo.clear();

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

void GpuAwareMpiBackend::send_temporal_packet(const TemporalPacket& packet, int dest_rank) {
  Impl::reap(impl_->pending_temporal);

  Impl::PendingSend pending;
  pending.buffer = pack_temporal_packet(packet);
  impl_->pending_temporal.push_back(std::move(pending));
  auto& back = impl_->pending_temporal.back();
  MPI_Isend(back.buffer.data(),
            static_cast<int>(back.buffer.size()),
            MPI_BYTE,
            dest_rank,
            kTagTemporal,
            impl_->comm,
            &back.request);
}

std::vector<TemporalPacket> GpuAwareMpiBackend::drain_arrived_temporal() {
  std::vector<TemporalPacket> out;
  for (;;) {
    int flag = 0;
    MPI_Status status;
    MPI_Iprobe(MPI_ANY_SOURCE, kTagTemporal, impl_->comm, &flag, &status);
    if (!flag) {
      break;
    }
    int byte_count = 0;
    MPI_Get_count(&status, MPI_BYTE, &byte_count);
    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(byte_count));
    MPI_Recv(buffer.data(),
             byte_count,
             MPI_BYTE,
             status.MPI_SOURCE,
             status.MPI_TAG,
             impl_->comm,
             MPI_STATUS_IGNORE);
    auto result = unpack_temporal_packet(buffer);
    if (!result.ok()) {
      if (result.error.find("CRC") != std::string::npos) {
        ++impl_->dropped_crc;
      } else if (result.error.find("protocol version") != std::string::npos) {
        ++impl_->dropped_version;
      } else {
        ++impl_->dropped_crc;
      }
      continue;
    }
    out.push_back(std::move(*result.packet));
  }
  return out;
}

void GpuAwareMpiBackend::send_subdomain_halo(const HaloPacket& packet, int dest_subdomain) {
  Impl::reap(impl_->pending_halo);

  Impl::PendingSend pending;
  pending.buffer = pack_halo_packet(packet);
  impl_->pending_halo.push_back(std::move(pending));
  auto& back = impl_->pending_halo.back();
  // CUDA-aware MPI fast-path becomes meaningful once the engine wires the
  // halo snapshot builder to write directly into pinned (or device) memory
  // (T7.9). Until then the buffer is a host vector — the MPI library
  // accepts it identically regardless of CUDA support.
  MPI_Isend(back.buffer.data(),
            static_cast<int>(back.buffer.size()),
            MPI_BYTE,
            dest_subdomain,
            kTagHalo,
            impl_->comm,
            &back.request);
}

std::vector<HaloPacket> GpuAwareMpiBackend::drain_arrived_halo() {
  std::vector<HaloPacket> out;
  for (;;) {
    int flag = 0;
    MPI_Status status;
    MPI_Iprobe(MPI_ANY_SOURCE, kTagHalo, impl_->comm, &flag, &status);
    if (!flag) {
      break;
    }
    int byte_count = 0;
    MPI_Get_count(&status, MPI_BYTE, &byte_count);
    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(byte_count));
    MPI_Recv(buffer.data(),
             byte_count,
             MPI_BYTE,
             status.MPI_SOURCE,
             status.MPI_TAG,
             impl_->comm,
             MPI_STATUS_IGNORE);
    auto result = unpack_halo_packet(buffer);
    if (!result.ok()) {
      if (result.error.find("CRC") != std::string::npos) {
        ++impl_->dropped_halo_crc;
      } else if (result.error.find("protocol version") != std::string::npos) {
        ++impl_->dropped_halo_version;
      } else {
        ++impl_->dropped_halo_crc;
      }
      continue;
    }
    out.push_back(std::move(*result.packet));
  }
  return out;
}

void GpuAwareMpiBackend::send_migration_packet(const MigrationPacket&, int) {
  // Migration over outer comm is wired in T7.9 (engine Pattern 2). For T7.3
  // the backend ships ready-shaped — a no-op send that progress()/drain()
  // surface uniformly with the halo path. SPEC §6.2 reserves migration
  // routing as part of the Hybrid composition (T7.5).
}

std::vector<MigrationPacket> GpuAwareMpiBackend::drain_arrived_migrations() {
  return {};
}

double GpuAwareMpiBackend::global_sum_double(double local) {
  if (impl_->config.use_deterministic_reductions) {
    return deterministic_sum_double(local, impl_->comm);
  }
  double result = 0.0;
  MPI_Allreduce(&local, &result, 1, MPI_DOUBLE, MPI_SUM, impl_->comm);
  return result;
}

double GpuAwareMpiBackend::global_max_double(double local) {
  double result = 0.0;
  MPI_Allreduce(&local, &result, 1, MPI_DOUBLE, MPI_MAX, impl_->comm);
  return result;
}

void GpuAwareMpiBackend::barrier() {
  MPI_Barrier(impl_->comm);
}

void GpuAwareMpiBackend::progress() {
  Impl::reap(impl_->pending_temporal);
  Impl::reap(impl_->pending_halo);
}

BackendInfo GpuAwareMpiBackend::info() const {
  BackendInfo i;
  i.name = "GpuAwareMpiBackend";
  i.protocol_version = kCommProtocolVersion;
  i.capabilities = {BackendCapability::GpuAwarePointers, BackendCapability::CollectiveOptimized};
  return i;
}

int GpuAwareMpiBackend::rank() const {
  return impl_->rank;
}

int GpuAwareMpiBackend::nranks() const {
  return impl_->nranks;
}

std::uint64_t GpuAwareMpiBackend::dropped_crc_count() const noexcept {
  return impl_ ? impl_->dropped_crc : 0;
}
std::uint64_t GpuAwareMpiBackend::dropped_version_count() const noexcept {
  return impl_ ? impl_->dropped_version : 0;
}
std::uint64_t GpuAwareMpiBackend::dropped_halo_crc_count() const noexcept {
  return impl_ ? impl_->dropped_halo_crc : 0;
}
std::uint64_t GpuAwareMpiBackend::dropped_halo_version_count() const noexcept {
  return impl_ ? impl_->dropped_halo_version : 0;
}

}  // namespace tdmd::comm

#else  // TDMD_ENABLE_MPI

// Stub implementation for the no-MPI build flavor: the constructor always
// throws, mirroring the CUDA-aware-failure refusal. Engine preflight
// (T7.9) treats both as "fall back to host-staging" — but host-staging
// itself is also unavailable without MPI, so a no-MPI build that asks for
// GpuAwareMpiBackend is necessarily a configuration error worth surfacing.

#include <stdexcept>

namespace tdmd::comm {

struct GpuAwareMpiBackend::Impl {};

GpuAwareMpiBackend::GpuAwareMpiBackend() : impl_(nullptr) {
  throw std::runtime_error(
      "GpuAwareMpiBackend: built without TDMD_ENABLE_MPI; "
      "rebuild with -DTDMD_ENABLE_MPI=ON to use this backend.");
}

GpuAwareMpiBackend::~GpuAwareMpiBackend() = default;
GpuAwareMpiBackend::GpuAwareMpiBackend(GpuAwareMpiBackend&&) noexcept = default;
GpuAwareMpiBackend& GpuAwareMpiBackend::operator=(GpuAwareMpiBackend&&) noexcept = default;

void GpuAwareMpiBackend::initialize(const CommConfig&) {}
void GpuAwareMpiBackend::shutdown() {}
void GpuAwareMpiBackend::send_temporal_packet(const TemporalPacket&, int) {}
std::vector<TemporalPacket> GpuAwareMpiBackend::drain_arrived_temporal() {
  return {};
}
void GpuAwareMpiBackend::send_subdomain_halo(const HaloPacket&, int) {}
std::vector<HaloPacket> GpuAwareMpiBackend::drain_arrived_halo() {
  return {};
}
void GpuAwareMpiBackend::send_migration_packet(const MigrationPacket&, int) {}
std::vector<MigrationPacket> GpuAwareMpiBackend::drain_arrived_migrations() {
  return {};
}
double GpuAwareMpiBackend::global_sum_double(double local) {
  return local;
}
double GpuAwareMpiBackend::global_max_double(double local) {
  return local;
}
void GpuAwareMpiBackend::barrier() {}
void GpuAwareMpiBackend::progress() {}
BackendInfo GpuAwareMpiBackend::info() const {
  BackendInfo i;
  i.name = "GpuAwareMpiBackend(disabled)";
  return i;
}
int GpuAwareMpiBackend::rank() const {
  return -1;
}
int GpuAwareMpiBackend::nranks() const {
  return 0;
}
std::uint64_t GpuAwareMpiBackend::dropped_crc_count() const noexcept {
  return 0;
}
std::uint64_t GpuAwareMpiBackend::dropped_version_count() const noexcept {
  return 0;
}
std::uint64_t GpuAwareMpiBackend::dropped_halo_crc_count() const noexcept {
  return 0;
}
std::uint64_t GpuAwareMpiBackend::dropped_halo_version_count() const noexcept {
  return 0;
}

}  // namespace tdmd::comm

#endif  // TDMD_ENABLE_MPI
