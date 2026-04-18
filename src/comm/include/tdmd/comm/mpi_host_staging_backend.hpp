#pragma once

// SPEC: docs/specs/comm/SPEC.md §6.1 (MpiHostStagingBackend),
//       §2.2 (CommBackend interface), §7 (collectives)
// Master spec: §10 (parallel model), §D.14 (MPI guarantees)
// Exec pack: docs/development/m5_execution_pack.md T5.4
//
// Universal MPI fallback backend. CPU-only: send path serializes into a
// host buffer (T5.3 packet serializer) then posts MPI_Isend; receive path
// polls MPI_Iprobe + MPI_Mrecv, deserializes, validates CRC, and returns
// any packets ready for the caller. Non-blocking by construction — the
// only synchronous call on this interface is `barrier()`.
//
// Concurrency model (M5): MPI_THREAD_SINGLE. The backend does not spawn
// progress threads; scheduler calls `progress()` once per iteration, which
// runs MPI_Test on outstanding send requests to let the MPI library reclaim
// buffers. Multi-threaded progression — M6+.
//
// Determinism: `global_sum_double` in Reference profile goes through
// `deterministic_sum_double` (D-M5-9). Raw MPI_Allreduce is not used for
// summation. `global_max_double` uses MPI_Allreduce(MAX) — max is
// associative + ordering-invariant, so this is safe.

#include "tdmd/comm/comm_backend.hpp"
#include "tdmd/comm/comm_config.hpp"
#include "tdmd/comm/types.hpp"

#include <cstdint>
#include <list>
#include <vector>

#if defined(TDMD_ENABLE_MPI) && TDMD_ENABLE_MPI
#include <mpi.h>
#endif

namespace tdmd::comm {

#if defined(TDMD_ENABLE_MPI) && TDMD_ENABLE_MPI

class MpiHostStagingBackend : public CommBackend {
public:
  MpiHostStagingBackend() = default;

  // Movable but not copyable — MPI handles have unique ownership semantics.
  MpiHostStagingBackend(const MpiHostStagingBackend&) = delete;
  MpiHostStagingBackend& operator=(const MpiHostStagingBackend&) = delete;
  MpiHostStagingBackend(MpiHostStagingBackend&&) = default;
  MpiHostStagingBackend& operator=(MpiHostStagingBackend&&) = default;

  ~MpiHostStagingBackend() override;

  void initialize(const CommConfig& config) override;
  void shutdown() override;

  void send_temporal_packet(const TemporalPacket& packet, int dest_rank) override;
  std::vector<TemporalPacket> drain_arrived_temporal() override;

  // HaloPacket / MigrationPacket — Pattern 2 (M7+). Stubbed in M5: send is
  // a no-op, drain returns empty. Tests in M7 will lift these; M5 callers
  // never invoke them.
  void send_subdomain_halo(const HaloPacket& packet, int dest_subdomain) override;
  std::vector<HaloPacket> drain_arrived_halo() override;
  void send_migration_packet(const MigrationPacket& packet, int dest_subdomain) override;
  std::vector<MigrationPacket> drain_arrived_migrations() override;

  double global_sum_double(double local) override;
  double global_max_double(double local) override;
  void barrier() override;
  void progress() override;

  BackendInfo info() const override;
  int rank() const override;
  int nranks() const override;

  // Observability — telemetry hooks read these counters. Non-virtual; not
  // part of the abstract interface.
  std::uint64_t dropped_crc_count() const noexcept { return dropped_crc_; }
  std::uint64_t dropped_version_count() const noexcept { return dropped_version_; }

private:
  // Each in-flight send owns its serialized byte buffer; the buffer lives
  // until MPI_Test reports the request complete. Using std::list so
  // iterator stability across progress() pruning.
  struct PendingSend {
    std::vector<std::uint8_t> buffer;
    MPI_Request request{MPI_REQUEST_NULL};
  };

  // Reap completed sends from `pending_sends_`. Called automatically on
  // every send + progress + drain.
  void reap_completed_sends();

  CommConfig config_{};
  MPI_Comm comm_ = MPI_COMM_NULL;
  bool owns_mpi_init_ = false;  // true iff this backend called MPI_Init
  int rank_ = -1;
  int nranks_ = 0;

  std::list<PendingSend> pending_sends_;

  std::uint64_t dropped_crc_ = 0;
  std::uint64_t dropped_version_ = 0;
};

#endif  // TDMD_ENABLE_MPI

}  // namespace tdmd::comm
