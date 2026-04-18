#pragma once

// SPEC: docs/specs/comm/SPEC.md §6.5 (RingBackend), §7.1 (collectives)
// Master spec: §13.3 (anchor-test premise)
// Exec pack: docs/development/m5_execution_pack.md T5.5
//
// Legacy ring-topology backend. The abstract interface allows any dest; this
// concrete class restricts `send_temporal_packet(pkt, dest)` to the single
// legal next-hop `(rank + 1) mod nranks`. Used **only** for anchor-test
// §13.3 (reproduction of Andreev's TIME-MD ring experiment). Production
// Pattern 1 runs continue to use MpiHostStagingBackend (mesh).
//
// Reduction contract (comm/SPEC §7.1): `global_sum_double` performs a
// sequential Kahan add around the ring followed by a ring-broadcast. This
// matches the dissertation's collective pattern — and because the addition
// order is fixed (rank 0 + rank 1 + … + rank P-1), the result is bit-exact
// across all ranks and across repeated invocations.

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

class RingBackend : public CommBackend {
public:
  RingBackend() = default;

  RingBackend(const RingBackend&) = delete;
  RingBackend& operator=(const RingBackend&) = delete;
  RingBackend(RingBackend&&) = default;
  RingBackend& operator=(RingBackend&&) = default;

  ~RingBackend() override;

  void initialize(const CommConfig& config) override;
  void shutdown() override;

  // Hard assert: dest_rank MUST equal (rank + 1) % nranks. Any violation
  // in Reference profile is a programmer error — the backend refuses to
  // send and aborts so the bug surfaces immediately.
  void send_temporal_packet(const TemporalPacket& packet, int dest_rank) override;
  std::vector<TemporalPacket> drain_arrived_temporal() override;

  // Pattern 2 — not implemented (no-op/empty).
  void send_subdomain_halo(const HaloPacket& packet, int dest_subdomain) override;
  std::vector<HaloPacket> drain_arrived_halo() override;
  void send_migration_packet(const MigrationPacket& packet, int dest_subdomain) override;
  std::vector<MigrationPacket> drain_arrived_migrations() override;

  // Ring-sum + ring-broadcast (comm/SPEC §7.1). Bit-exact under the fixed
  // rank-ascending addition order.
  double global_sum_double(double local) override;
  double global_max_double(double local) override;
  void barrier() override;
  void progress() override;

  BackendInfo info() const override;
  int rank() const override;
  int nranks() const override;

  // Non-virtual predicate used by unit tests to exercise the guardrail
  // without tripping the abort that `send_temporal_packet` would raise.
  bool is_ring_next(int dest) const noexcept;

  std::uint64_t dropped_crc_count() const noexcept { return dropped_crc_; }
  std::uint64_t dropped_version_count() const noexcept { return dropped_version_; }

private:
  struct PendingSend {
    std::vector<std::uint8_t> buffer;
    MPI_Request request{MPI_REQUEST_NULL};
  };

  void reap_completed_sends();

  CommConfig config_{};
  MPI_Comm comm_ = MPI_COMM_NULL;
  bool owns_mpi_init_ = false;
  int rank_ = -1;
  int nranks_ = 0;

  std::list<PendingSend> pending_sends_;

  std::uint64_t dropped_crc_ = 0;
  std::uint64_t dropped_version_ = 0;
};

#endif  // TDMD_ENABLE_MPI

}  // namespace tdmd::comm
