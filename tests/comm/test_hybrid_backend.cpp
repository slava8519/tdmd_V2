// SPEC: docs/specs/comm/SPEC.md §6.4 (HybridBackend), §3.4 (topology)
// Master spec: §14 M7 (Pattern 2 default backend)
// Exec pack: docs/development/m7_execution_pack.md T7.5
//
// Unit tests for HybridBackend's routing matrix. Uses a `SpyBackend` mock
// (records every call) so dispatch correctness can be verified without
// needing NCCL / CUDA-aware MPI / MPI at all — runs in the always-built
// `test_comm` binary on every CI flavor.
//
// Coverage:
//   - send_temporal_packet, drain_arrived_temporal       → inner
//   - send_subdomain_halo,  drain_arrived_halo            → outer
//   - send_migration_packet, drain_arrived_migrations     → outer
//   - global_sum_double / global_max_double / barrier     → inner
//   - progress                                            → both
//   - initialize delegates to both; rank/nranks mismatch  → throws
//   - subdomain count mismatch (D-M7-2 violation)         → throws
//   - 4-rank 2×2 grid: corner subdomain peer_neighbors    → 3 in 2D
//   - Pattern 1 (single subdomain): peer_neighbors empty,
//     halos / migrations never get sent end-to-end (no outer traffic)
//   - info() composes inner+outer name and union of capabilities
//   - shutdown order: outer first, inner second

#include "tdmd/comm/comm_backend.hpp"
#include "tdmd/comm/comm_config.hpp"
#include "tdmd/comm/hybrid_backend.hpp"
#include "tdmd/comm/topology_resolver.hpp"
#include "tdmd/comm/types.hpp"

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tc = tdmd::comm;

namespace {

// SpyBackend — records every call. Configurable rank/nranks/info.name so
// tests can simulate a 4-rank world without spinning up real MPI.
class SpyBackend : public tc::CommBackend {
public:
  std::string name = "Spy";
  int my_rank = 0;
  int world_size = 1;
  std::vector<tc::BackendCapability> caps;
  bool initialized = false;
  bool shut = false;
  int init_order = 0;
  int shutdown_order = 0;
  static inline int next_init_seq = 0;
  static inline int next_shutdown_seq = 0;

  int sent_temporal = 0;
  int drained_temporal = 0;
  int sent_halo = 0;
  int drained_halo = 0;
  int sent_migration = 0;
  int drained_migration = 0;
  int sum_calls = 0;
  int max_calls = 0;
  int barrier_calls = 0;
  int progress_calls = 0;
  std::vector<int> sent_temporal_dest;
  std::vector<int> sent_halo_dest;

  void initialize(const tc::CommConfig& /*config*/) override {
    initialized = true;
    init_order = ++next_init_seq;
  }
  void shutdown() override {
    shut = true;
    shutdown_order = ++next_shutdown_seq;
  }
  void send_temporal_packet(const tc::TemporalPacket& /*packet*/, int dest_rank) override {
    ++sent_temporal;
    sent_temporal_dest.push_back(dest_rank);
  }
  std::vector<tc::TemporalPacket> drain_arrived_temporal() override {
    ++drained_temporal;
    return {};
  }
  void send_subdomain_halo(const tc::HaloPacket& /*packet*/, int dest_subdomain) override {
    ++sent_halo;
    sent_halo_dest.push_back(dest_subdomain);
  }
  std::vector<tc::HaloPacket> drain_arrived_halo() override {
    ++drained_halo;
    return {};
  }
  void send_migration_packet(const tc::MigrationPacket& /*packet*/,
                             int /*dest_subdomain*/) override {
    ++sent_migration;
  }
  std::vector<tc::MigrationPacket> drain_arrived_migrations() override {
    ++drained_migration;
    return {};
  }
  double global_sum_double(double local) override {
    ++sum_calls;
    return local;
  }
  double global_max_double(double local) override {
    ++max_calls;
    return local;
  }
  void barrier() override { ++barrier_calls; }
  void progress() override { ++progress_calls; }

  [[nodiscard]] tc::BackendInfo info() const override {
    tc::BackendInfo i;
    i.name = name;
    i.capabilities = caps;
    return i;
  }
  [[nodiscard]] int rank() const override { return my_rank; }
  [[nodiscard]] int nranks() const override { return world_size; }
};

[[nodiscard]] std::unique_ptr<SpyBackend> make_spy(std::string name,
                                                   int rank,
                                                   int world,
                                                   std::vector<tc::BackendCapability> caps = {}) {
  auto s = std::make_unique<SpyBackend>();
  s->name = std::move(name);
  s->my_rank = rank;
  s->world_size = world;
  s->caps = std::move(caps);
  return s;
}

}  // namespace

TEST_CASE("HybridBackend — constructor rejects null backends", "[comm][hybrid]") {
  REQUIRE_THROWS_AS(
      (tc::HybridBackend{nullptr, std::make_unique<SpyBackend>(), tc::CartesianGrid{}}),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      (tc::HybridBackend{std::make_unique<SpyBackend>(), nullptr, tc::CartesianGrid{}}),
      std::invalid_argument);
}

TEST_CASE("HybridBackend — initialize delegates and validates rank parity", "[comm][hybrid]") {
  SECTION("matched ranks → initializes both") {
    auto inner = make_spy("Inner", 0, 1);
    auto outer = make_spy("Outer", 0, 1);
    auto* inner_raw = inner.get();
    auto* outer_raw = outer.get();
    tc::HybridBackend backend(std::move(inner), std::move(outer), tc::CartesianGrid{1, 1, 1});
    SpyBackend::next_init_seq = 0;
    REQUIRE_NOTHROW(backend.initialize(tc::CommConfig{}));
    REQUIRE(inner_raw->initialized);
    REQUIRE(outer_raw->initialized);
    // Inner first, then outer.
    REQUIRE(inner_raw->init_order < outer_raw->init_order);
  }
  SECTION("mismatched ranks → throws") {
    auto inner = make_spy("Inner", 0, 4);
    auto outer = make_spy("Outer", 1, 4);
    tc::HybridBackend backend(std::move(inner), std::move(outer), tc::CartesianGrid{4, 1, 1});
    REQUIRE_THROWS_AS(backend.initialize(tc::CommConfig{}), std::runtime_error);
  }
  SECTION("nranks doesn't match topology subdomain count → throws") {
    auto inner = make_spy("Inner", 0, 3);
    auto outer = make_spy("Outer", 0, 3);
    // 4-subdomain grid but 3 ranks — D-M7-2 violation (1:1 binding broken).
    tc::HybridBackend backend(std::move(inner), std::move(outer), tc::CartesianGrid{2, 2, 1});
    REQUIRE_THROWS_AS(backend.initialize(tc::CommConfig{}), std::runtime_error);
  }
}

TEST_CASE("HybridBackend — routing matrix (4-rank 2×2 grid)", "[comm][hybrid]") {
  auto inner = make_spy("Inner", 0, 4, {tc::BackendCapability::CollectiveOptimized});
  auto outer = make_spy("Outer", 0, 4, {tc::BackendCapability::GpuAwarePointers});
  auto* inner_raw = inner.get();
  auto* outer_raw = outer.get();
  // 2×2×1 Cartesian grid → 4 subdomains.
  tc::HybridBackend backend(std::move(inner), std::move(outer), tc::CartesianGrid{2, 2, 1});
  backend.initialize(tc::CommConfig{});

  SECTION("send_temporal → inner only") {
    backend.send_temporal_packet(tc::TemporalPacket{}, /*dest_rank=*/2);
    REQUIRE(inner_raw->sent_temporal == 1);
    REQUIRE(inner_raw->sent_temporal_dest == std::vector<int>{2});
    REQUIRE(outer_raw->sent_temporal == 0);
  }
  SECTION("drain_temporal → inner only") {
    (void) backend.drain_arrived_temporal();
    REQUIRE(inner_raw->drained_temporal == 1);
    REQUIRE(outer_raw->drained_temporal == 0);
  }
  SECTION("send_subdomain_halo → outer only") {
    backend.send_subdomain_halo(tc::HaloPacket{}, /*dest_subdomain=*/3);
    REQUIRE(outer_raw->sent_halo == 1);
    REQUIRE(outer_raw->sent_halo_dest == std::vector<int>{3});
    REQUIRE(inner_raw->sent_halo == 0);
  }
  SECTION("drain_halo → outer only") {
    (void) backend.drain_arrived_halo();
    REQUIRE(outer_raw->drained_halo == 1);
    REQUIRE(inner_raw->drained_halo == 0);
  }
  SECTION("send_migration → outer only") {
    backend.send_migration_packet(tc::MigrationPacket{}, /*dest_subdomain=*/1);
    REQUIRE(outer_raw->sent_migration == 1);
    REQUIRE(inner_raw->sent_migration == 0);
  }
  SECTION("drain_migrations → outer only") {
    (void) backend.drain_arrived_migrations();
    REQUIRE(outer_raw->drained_migration == 1);
    REQUIRE(inner_raw->drained_migration == 0);
  }
  SECTION("global_sum / global_max / barrier → inner only") {
    (void) backend.global_sum_double(1.5);
    (void) backend.global_max_double(2.25);
    backend.barrier();
    REQUIRE(inner_raw->sum_calls == 1);
    REQUIRE(inner_raw->max_calls == 1);
    REQUIRE(inner_raw->barrier_calls == 1);
    REQUIRE(outer_raw->sum_calls == 0);
    REQUIRE(outer_raw->max_calls == 0);
    REQUIRE(outer_raw->barrier_calls == 0);
  }
  SECTION("progress → both, once each per call") {
    backend.progress();
    backend.progress();
    REQUIRE(inner_raw->progress_calls == 2);
    REQUIRE(outer_raw->progress_calls == 2);
  }
  SECTION("info() composes name and unions capabilities") {
    const tc::BackendInfo i = backend.info();
    REQUIRE(i.name.find("HybridBackend") != std::string::npos);
    REQUIRE(i.name.find("Inner") != std::string::npos);
    REQUIRE(i.name.find("Outer") != std::string::npos);
    bool has_collective = false;
    bool has_gpu_aware = false;
    for (const auto& c : i.capabilities) {
      if (c == tc::BackendCapability::CollectiveOptimized) {
        has_collective = true;
      }
      if (c == tc::BackendCapability::GpuAwarePointers) {
        has_gpu_aware = true;
      }
    }
    REQUIRE(has_collective);
    REQUIRE(has_gpu_aware);
  }
  SECTION("rank / nranks delegate to inner") {
    REQUIRE(backend.rank() == 0);
    REQUIRE(backend.nranks() == 4);
  }
  SECTION("topology() exposes the configured grid") {
    REQUIRE(backend.topology().grid().total() == 4);
    // 2×2 corner subdomain (0,0,0) — 3 neighbors in 2D.
    REQUIRE(backend.topology().peer_neighbors(0).size() == 3);
  }
}

TEST_CASE("HybridBackend — Pattern 1 single-subdomain has empty peer set", "[comm][hybrid]") {
  // 1×1×1 grid with periodic wrap — Pattern 1 from the dissertation anchor
  // and every M5 smoke. peer_neighbors must be empty so the engine never
  // issues an outer halo send.
  auto inner = make_spy("Inner", 0, 1);
  auto outer = make_spy("Outer", 0, 1);
  tc::HybridBackend backend(std::move(inner),
                            std::move(outer),
                            tc::CartesianGrid{1, 1, 1, true, true, true});
  backend.initialize(tc::CommConfig{});
  REQUIRE(backend.topology().peer_neighbors(0).empty());
}

TEST_CASE("HybridBackend — shutdown order: outer first, inner second", "[comm][hybrid]") {
  auto inner = make_spy("Inner", 0, 1);
  auto outer = make_spy("Outer", 0, 1);
  auto* inner_raw = inner.get();
  auto* outer_raw = outer.get();
  tc::HybridBackend backend(std::move(inner), std::move(outer), tc::CartesianGrid{1, 1, 1});
  backend.initialize(tc::CommConfig{});
  SpyBackend::next_shutdown_seq = 0;
  backend.shutdown();
  REQUIRE(outer_raw->shut);
  REQUIRE(inner_raw->shut);
  REQUIRE(outer_raw->shutdown_order < inner_raw->shutdown_order);
}
