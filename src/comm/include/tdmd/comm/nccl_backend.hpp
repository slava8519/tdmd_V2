#pragma once

// SPEC: docs/specs/comm/SPEC.md §6.3 (NcclBackend), §2.3
// Master spec: §14 M7, §12.6
// Exec pack: docs/development/m7_execution_pack.md T7.4
//
// NCCL backend for intra-node TD temporal packets + allreduce collectives.
// Pattern 2's inner level owns NCCL; inter-node halos stay on
// GpuAwareMpiBackend (or MpiHostStaging fallback). Construction probes
// `is_nccl_available()` and throws `std::runtime_error` when NCCL is absent —
// the engine's preflight (T7.9) catches the throw and routes the inner level
// to GpuAwareMpi or MpiHostStaging.
//
// PIMPL firewall: public header pulls only TDMD types + abstract
// `CommBackend`. `<nccl.h>` / `<mpi.h>` live in the `.cpp` TU so downstream
// targets that include this header don't force an NCCL/MPI transitive dep.
//
// Determinism (D-M5-9 + D-M7-4): NCCL is transport-only. `global_sum_double`
// uses `ncclAllGather` to collect per-rank doubles, then folds them through
// the existing host-side Kahan reduction (`deterministic_sum_double`). That
// preserves byte-exact equivalence with `MpiHostStagingBackend` on the M5
// thermo smoke and keeps the D-M5-12 chain intact through the NCCL path.

#include "tdmd/comm/comm_backend.hpp"
#include "tdmd/comm/comm_config.hpp"
#include "tdmd/comm/types.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace tdmd::comm {

class NcclBackend : public CommBackend {
public:
  NcclBackend();
  ~NcclBackend() override;

  NcclBackend(const NcclBackend&) = delete;
  NcclBackend& operator=(const NcclBackend&) = delete;
  NcclBackend(NcclBackend&&) noexcept;
  NcclBackend& operator=(NcclBackend&&) noexcept;

  void initialize(const CommConfig& config) override;
  void shutdown() override;

  void send_temporal_packet(const TemporalPacket& packet, int dest_rank) override;
  std::vector<TemporalPacket> drain_arrived_temporal() override;

  // Halo + migration stay on the outer transport (GpuAwareMpi / MpiHostStaging)
  // per HybridBackend's routing matrix (SPEC §6.4). NcclBackend surfaces them
  // as no-ops so the interface is complete; actual dispatch lives in T7.5.
  void send_subdomain_halo(const HaloPacket& packet, int dest_subdomain) override;
  std::vector<HaloPacket> drain_arrived_halo() override;
  void send_migration_packet(const MigrationPacket& packet, int dest_subdomain) override;
  std::vector<MigrationPacket> drain_arrived_migrations() override;

  double global_sum_double(double local) override;
  double global_max_double(double local) override;
  void barrier() override;
  void progress() override;

  [[nodiscard]] BackendInfo info() const override;
  [[nodiscard]] int rank() const override;
  [[nodiscard]] int nranks() const override;

  // Telemetry shaped to match GpuAwareMpiBackend — intentional so
  // HybridBackend (T7.5) can aggregate uniformly across both backends.
  [[nodiscard]] std::uint64_t dropped_crc_count() const noexcept;
  [[nodiscard]] std::uint64_t dropped_version_count() const noexcept;
  [[nodiscard]] std::uint64_t dropped_halo_crc_count() const noexcept;
  [[nodiscard]] std::uint64_t dropped_halo_version_count() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tdmd::comm
