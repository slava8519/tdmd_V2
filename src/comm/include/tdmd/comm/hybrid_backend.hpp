#pragma once

// SPEC: docs/specs/comm/SPEC.md §6.4 (HybridBackend), §3.4 (topology)
// Master spec: §14 M7 (Pattern 2 default backend)
// Exec pack: docs/development/m7_execution_pack.md T7.5
//
// HybridBackend — composition (not duplication) of two backends:
//
//   inner: NCCL preferred, GpuAwareMpi fallback, MpiHostStaging last resort.
//          Carries `send_temporal_packet` (TD intra-subdomain) and the
//          collective reductions (`global_sum_double`, `global_max_double`,
//          `barrier`). NCCL-on-NVLink wins the latency competition for
//          high-frequency intra-subdomain TD packets.
//
//   outer: GpuAwareMpi preferred, MpiHostStaging fallback. Carries
//          `send_subdomain_halo` and `send_migration_packet` — inter-
//          subdomain SD traffic. MPI is the only universal cross-node
//          transport at v1; NCCL multi-node requires NVSwitch (deferred to
//          NvshmemBackend research, comm/SPEC §6.6).
//
// Routing matrix (comm/SPEC §6.4):
//
//   send_temporal_packet         → inner
//   drain_arrived_temporal       → inner
//   send_subdomain_halo          → outer
//   drain_arrived_halo           → outer
//   send_migration_packet        → outer
//   drain_arrived_migrations     → outer
//   global_sum_double            → inner (preserves D-M5-12 thermo chain)
//   global_max_double            → inner
//   barrier                      → inner
//   progress                     → both (one tick each)
//
// Single-subdomain Pattern 1 runs (the dissertation anchor-test geometry +
// every M5 smoke) construct HybridBackend without ever touching the outer
// path: with `peer_neighbors(sd)` returning empty, no halo or migration
// send is ever issued, so the wrapped behavior is bit-identical to the
// inner backend alone.
//
// Construction policy: caller passes already-constructed inner + outer
// backends via unique_ptr. Discovery / fallback chain (try NCCL, fall back
// to GpuAwareMpi, etc.) lives in the engine preflight (T7.9), not here —
// HybridBackend itself is policy-free transport.

#include "tdmd/comm/comm_backend.hpp"
#include "tdmd/comm/comm_config.hpp"
#include "tdmd/comm/topology_resolver.hpp"
#include "tdmd/comm/types.hpp"

#include <memory>
#include <vector>

namespace tdmd::comm {

class HybridBackend : public CommBackend {
public:
  // Both backends must be non-null. They share the same world rank space
  // (D-M7-2: 1:1 subdomain↔rank), so `inner->rank()` == `outer->rank()`
  // and likewise for `nranks()`. The constructor doesn't assert this
  // because backends might not yet be initialized; `initialize()` does.
  HybridBackend(std::unique_ptr<CommBackend> inner,
                std::unique_ptr<CommBackend> outer,
                CartesianGrid grid);

  ~HybridBackend() override;

  HybridBackend(const HybridBackend&) = delete;
  HybridBackend& operator=(const HybridBackend&) = delete;
  HybridBackend(HybridBackend&&) noexcept;
  HybridBackend& operator=(HybridBackend&&) noexcept;

  void initialize(const CommConfig& config) override;
  void shutdown() override;

  void send_temporal_packet(const TemporalPacket& packet, int dest_rank) override;
  std::vector<TemporalPacket> drain_arrived_temporal() override;

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

  // Topology accessor — exposed so OuterSdCoordinator (T7.6) can ask
  // "who do I halo-send to?" without re-deriving the grid.
  [[nodiscard]] const TopologyResolver& topology() const noexcept { return topology_; }

  // Backend accessors — used by tests and by the engine preflight to
  // surface telemetry separately for inner vs outer paths.
  [[nodiscard]] const CommBackend& inner() const noexcept { return *inner_; }
  [[nodiscard]] const CommBackend& outer() const noexcept { return *outer_; }

private:
  std::unique_ptr<CommBackend> inner_;
  std::unique_ptr<CommBackend> outer_;
  TopologyResolver topology_;
};

}  // namespace tdmd::comm
