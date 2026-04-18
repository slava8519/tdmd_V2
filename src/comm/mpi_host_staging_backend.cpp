// SPEC: docs/specs/comm/SPEC.md §6.1, §7
// Exec pack: docs/development/m5_execution_pack.md T5.4

#include "tdmd/comm/mpi_host_staging_backend.hpp"

#if defined(TDMD_ENABLE_MPI) && TDMD_ENABLE_MPI

#include "tdmd/comm/deterministic_reduction.hpp"
#include "tdmd/comm/packet_serializer.hpp"

#include <stdexcept>
#include <utility>

namespace tdmd::comm {

namespace {

// Distinct MPI tags per packet class avoid collisions when scheduler +
// halo + migration traffic coexist on the same communicator (M7+). Values
// are part of the wire contract — see comm/SPEC §4 note on tag layout.
constexpr int kTagTemporal = 1001;

}  // namespace

MpiHostStagingBackend::~MpiHostStagingBackend() {
  // Best-effort cleanup in case the caller forgot to call shutdown(). We
  // deliberately swallow MPI errors here: a destructor that throws during
  // stack unwinding would std::terminate the process, and there is no
  // meaningful recovery for an MPI state already this far gone.
  if (comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&comm_);
    comm_ = MPI_COMM_NULL;
  }
  if (owns_mpi_init_) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
    owns_mpi_init_ = false;
  }
}

void MpiHostStagingBackend::initialize(const CommConfig& config) {
  config_ = config;

  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited) {
    // We'd prefer MPI_THREAD_SINGLE; M5 scheduler is single-threaded per
    // D-M5-5. M6+ can upgrade to SERIALIZED / MULTIPLE as needed.
    int provided = 0;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SINGLE, &provided);
    owns_mpi_init_ = true;
  }

  MPI_Comm_dup(MPI_COMM_WORLD, &comm_);
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &nranks_);
}

void MpiHostStagingBackend::shutdown() {
  // Drain in-flight sends — callers are expected to have already completed
  // their protocol-level handshake, but we'd rather wait than leak MPI
  // requests and crash in MPI_Finalize.
  for (auto& send : pending_sends_) {
    MPI_Wait(&send.request, MPI_STATUS_IGNORE);
  }
  pending_sends_.clear();

  if (comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&comm_);
    comm_ = MPI_COMM_NULL;
  }
  if (owns_mpi_init_) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
    owns_mpi_init_ = false;
  }
}

void MpiHostStagingBackend::reap_completed_sends() {
  for (auto it = pending_sends_.begin(); it != pending_sends_.end();) {
    int flag = 0;
    MPI_Test(&it->request, &flag, MPI_STATUS_IGNORE);
    if (flag) {
      it = pending_sends_.erase(it);
    } else {
      ++it;
    }
  }
}

void MpiHostStagingBackend::send_temporal_packet(const TemporalPacket& packet, int dest_rank) {
  reap_completed_sends();

  PendingSend pending;
  pending.buffer = pack_temporal_packet(packet);

  // MPI_Isend with the buffer owned by PendingSend; the buffer is stable
  // (std::list never invalidates pointers to its nodes) until progress()
  // reaps the request.
  pending_sends_.push_back(std::move(pending));
  auto& back = pending_sends_.back();
  MPI_Isend(back.buffer.data(),
            static_cast<int>(back.buffer.size()),
            MPI_BYTE,
            dest_rank,
            kTagTemporal,
            comm_,
            &back.request);
}

std::vector<TemporalPacket> MpiHostStagingBackend::drain_arrived_temporal() {
  std::vector<TemporalPacket> out;

  // Poll Iprobe in a loop — every call to this method drains whatever has
  // already arrived. Non-blocking: breaks as soon as no match is pending.
  for (;;) {
    int flag = 0;
    MPI_Status status;
    MPI_Iprobe(MPI_ANY_SOURCE, kTagTemporal, comm_, &flag, &status);
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
             comm_,
             MPI_STATUS_IGNORE);

    auto result = unpack_temporal_packet(buffer);
    if (!result.ok()) {
      // comm/SPEC §4.4: CRC failure is not fatal — drop + count. Likewise
      // protocol-version mismatch per §4.3. Classifying the error via
      // substring match is adequate here because the serializer is the
      // only thing producing these strings; a richer tag would be premature.
      if (result.error.find("CRC") != std::string::npos) {
        ++dropped_crc_;
      } else if (result.error.find("protocol version") != std::string::npos) {
        ++dropped_version_;
      } else {
        // Other corruption (truncated, payload-size mismatch). Treat as
        // CRC-class failure — the buffer is unusable.
        ++dropped_crc_;
      }
      continue;
    }
    out.push_back(std::move(*result.packet));
  }

  return out;
}

void MpiHostStagingBackend::send_subdomain_halo(const HaloPacket&, int) {
  // Pattern 2 stub (M7+). Intentionally does nothing.
}

std::vector<HaloPacket> MpiHostStagingBackend::drain_arrived_halo() {
  return {};
}

void MpiHostStagingBackend::send_migration_packet(const MigrationPacket&, int) {
  // Pattern 2 stub (M7+). Intentionally does nothing.
}

std::vector<MigrationPacket> MpiHostStagingBackend::drain_arrived_migrations() {
  return {};
}

double MpiHostStagingBackend::global_sum_double(double local) {
  if (config_.use_deterministic_reductions) {
    return deterministic_sum_double(local, comm_);
  }
  // Fast-profile path (M8+). In M5 this branch is unreachable under
  // Reference config but wiring it keeps the Fast-on test surface small
  // when it lands.
  double result = 0.0;
  MPI_Allreduce(&local, &result, 1, MPI_DOUBLE, MPI_SUM, comm_);
  return result;
}

double MpiHostStagingBackend::global_max_double(double local) {
  // MAX is associative and ordering-invariant — deterministic under any
  // MPI implementation. No need for the Kahan-ordered path.
  double result = 0.0;
  MPI_Allreduce(&local, &result, 1, MPI_DOUBLE, MPI_MAX, comm_);
  return result;
}

void MpiHostStagingBackend::barrier() {
  MPI_Barrier(comm_);
}

void MpiHostStagingBackend::progress() {
  reap_completed_sends();
}

BackendInfo MpiHostStagingBackend::info() const {
  BackendInfo i;
  i.name = "MpiHostStagingBackend";
  i.protocol_version = kCommProtocolVersion;
  // Capabilities: CPU-only staging, no direct RDMA or GPU-awareness.
  i.capabilities = {BackendCapability::CollectiveOptimized};
  return i;
}

int MpiHostStagingBackend::rank() const {
  return rank_;
}

int MpiHostStagingBackend::nranks() const {
  return nranks_;
}

}  // namespace tdmd::comm

#endif  // TDMD_ENABLE_MPI
