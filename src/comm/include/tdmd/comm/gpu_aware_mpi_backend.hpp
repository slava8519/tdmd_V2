#pragma once

// SPEC: docs/specs/comm/SPEC.md §6.2 (GpuAwareMpiBackend), §2.2
// Master spec: §10, §12.6
// Exec pack: docs/development/m7_execution_pack.md T7.3
//
// CUDA-aware MPI backend. Same lifecycle as `MpiHostStagingBackend` but
// declares the `GpuAwarePointers` capability and is permitted by the
// HybridBackend (T7.5) to handle inter-subdomain halos for multi-node
// runs. The constructor calls `is_cuda_aware_mpi()` and refuses to
// construct (throws `std::runtime_error`) if the runtime probe reports
// no CUDA-aware support — the engine's preflight (T7.9) catches the
// throw and falls back to `MpiHostStagingBackend` with a warning.
//
// PIMPL firewall: the public header pulls in only TDMD types and the
// abstract `CommBackend`. MPI / CUDA headers live entirely in the .cpp
// translation unit so downstream targets that include this header do not
// need MPI configured.
//
// Inner-level temporal packets currently flow through the same host-staged
// path as MpiHostStagingBackend — the inner-TD optimization belongs to
// `NcclBackend` (T7.4). For outer-level halos the path is different: when
// CUDA-aware MPI is available, the underlying `MPI_Isend` accepts a
// device pointer directly. In M7 the HaloPacket payload is still a host
// byte vector (the in-engine device→host copy lives in the engine wire
// of T7.9), so the immediate behavior is "host-byte send with the right
// MPI stack"; the device-pointer fast-path is exercised once T7.9 wires
// the snapshot builder to write into pinned host memory.

#include "tdmd/comm/comm_backend.hpp"
#include "tdmd/comm/comm_config.hpp"
#include "tdmd/comm/types.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace tdmd::comm {

class GpuAwareMpiBackend : public CommBackend {
public:
  GpuAwareMpiBackend();
  ~GpuAwareMpiBackend() override;

  GpuAwareMpiBackend(const GpuAwareMpiBackend&) = delete;
  GpuAwareMpiBackend& operator=(const GpuAwareMpiBackend&) = delete;
  GpuAwareMpiBackend(GpuAwareMpiBackend&&) noexcept;
  GpuAwareMpiBackend& operator=(GpuAwareMpiBackend&&) noexcept;

  void initialize(const CommConfig& config) override;
  void shutdown() override;

  void send_temporal_packet(const TemporalPacket& packet, int dest_rank) override;
  std::vector<TemporalPacket> drain_arrived_temporal() override;

  void send_subdomain_halo(const HaloPacket& packet, int dest_subdomain) override;
  std::vector<HaloPacket> drain_arrived_halo() override;

  // Migration packets ride the same outer transport in M7. T7.9 may
  // reroute them; for now they are no-ops so HybridBackend doesn't need
  // a third backend just to dispatch them.
  void send_migration_packet(const MigrationPacket& packet, int dest_subdomain) override;
  std::vector<MigrationPacket> drain_arrived_migrations() override;

  double global_sum_double(double local) override;
  double global_max_double(double local) override;
  void barrier() override;
  void progress() override;

  [[nodiscard]] BackendInfo info() const override;
  [[nodiscard]] int rank() const override;
  [[nodiscard]] int nranks() const override;

  // Telemetry — same shape as MpiHostStagingBackend so cross-backend
  // dashboards work uniformly.
  [[nodiscard]] std::uint64_t dropped_crc_count() const noexcept;
  [[nodiscard]] std::uint64_t dropped_version_count() const noexcept;
  [[nodiscard]] std::uint64_t dropped_halo_crc_count() const noexcept;
  [[nodiscard]] std::uint64_t dropped_halo_version_count() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tdmd::comm
