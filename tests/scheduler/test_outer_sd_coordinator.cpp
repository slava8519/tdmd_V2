// SPEC: docs/specs/scheduler/SPEC.md §2.4 (OC-1..OC-6), §4.6 (HA-1..HA-5)
// Master spec: §12.7a
// Exec pack: docs/development/m7_execution_pack.md T7.6
//
// Unit tests for the Pattern 2 outer coordinator + ring-buffer halo
// archive. Tests are grouped by SPEC contract row (OC-N / HA-N) for easy
// audit against the SPEC table. The cross-system byte-exact-vs-M6 trace
// comparison is deferred to T7.9 (SimulationEngine Pattern 2 wire).

#include "tdmd/scheduler/concrete_outer_sd_coordinator.hpp"
#include "tdmd/scheduler/halo_snapshot.hpp"
#include "tdmd/scheduler/outer_sd_coordinator.hpp"
#include "tdmd/scheduler/subdomain_grid.hpp"
#include "tdmd/state/box.hpp"

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <type_traits>

namespace ts = tdmd::scheduler;

namespace {

ts::SubdomainGrid make_grid_2x1x1() {
  ts::SubdomainGrid g;
  g.n_subdomains = {2, 1, 1};
  g.subdomain_boxes.resize(2);
  g.subdomain_boxes[0] = tdmd::Box{0.0, 5.0, 0.0, 10.0, 0.0, 10.0, true, true, true};
  g.subdomain_boxes[1] = tdmd::Box{5.0, 10.0, 0.0, 10.0, 0.0, 10.0, true, true, true};
  g.rank_of_subdomain = {0, 1};
  return g;
}

ts::HaloSnapshot make_peer_snap(std::uint32_t source_sub,
                                ts::ZoneId source_zone,
                                ts::TimeLevel level,
                                ts::Version ver = 1,
                                std::uint32_t atom_count = 8) {
  ts::HaloSnapshot s;
  s.source_subdomain_id = source_sub;
  s.source_zone_id = source_zone;
  s.time_level = level;
  s.source_version = ver;
  s.atom_count = atom_count;
  s.payload.assign(atom_count * 64, std::uint8_t{0xCC});  // 64 B/atom SoA
  return s;
}

}  // namespace

TEST_CASE("HaloSnapshot — default-constructible, value-type layout", "[scheduler][t7_6][types]") {
  ts::HaloSnapshot s;
  REQUIRE(s.source_subdomain_id == 0);
  REQUIRE(s.source_zone_id == 0);
  REQUIRE(s.time_level == 0);
  REQUIRE(s.source_version == 0);
  REQUIRE(s.atom_count == 0);
  REQUIRE(s.payload.empty());
  REQUIRE(s.received_seq == 0);
}

TEST_CASE("SubdomainGrid — total_subdomains matches product", "[scheduler][t7_6][grid]") {
  const auto g = make_grid_2x1x1();
  REQUIRE(g.total_subdomains() == 2);
  REQUIRE(g.subdomain_boxes.size() == 2);
  REQUIRE(g.rank_of_subdomain.size() == 2);
}

TEST_CASE("ConcreteOuterSdCoordinator — initialize rejects zero K_max", "[scheduler][t7_6][init]") {
  ts::ConcreteOuterSdCoordinator coord;
  const auto g = make_grid_2x1x1();
  REQUIRE_THROWS_AS(coord.initialize(g, 0), std::logic_error);
}

TEST_CASE("ConcreteOuterSdCoordinator — initialize rejects malformed grid",
          "[scheduler][t7_6][init]") {
  ts::ConcreteOuterSdCoordinator coord;
  ts::SubdomainGrid g;
  g.n_subdomains = {2, 1, 1};
  g.subdomain_boxes.resize(1);  // wrong arity (expected 2)
  g.rank_of_subdomain.resize(2);
  REQUIRE_THROWS_AS(coord.initialize(g, 4), std::logic_error);
}

TEST_CASE(
    "ConcreteOuterSdCoordinator — OC-1 can_advance vacuously true with no "
    "registered peer deps",
    "[scheduler][t7_6][oc1]") {
  ts::ConcreteOuterSdCoordinator coord;
  coord.initialize(make_grid_2x1x1(), 4);
  // Local zone 7 has no registered peer deps → can advance freely.
  REQUIRE(coord.can_advance_boundary_zone(7, 1));
  REQUIRE(coord.can_advance_boundary_zone(7, 100));
}

TEST_CASE("ConcreteOuterSdCoordinator — OC-1 can_advance flips to true on archive",
          "[scheduler][t7_6][oc1]") {
  ts::ConcreteOuterSdCoordinator coord;
  coord.initialize(make_grid_2x1x1(), 4);
  // Local zone 3 depends on (peer_sub=1, peer_zone=5).
  coord.register_zone_peer_dependency(3, ts::PeerKey{1, 5});
  // No archive yet → cannot advance to level 1 (needs peer level 0).
  REQUIRE_FALSE(coord.can_advance_boundary_zone(3, 1));
  // Archive peer snapshot at level 0; now level 1 can advance.
  coord.archive_peer_snapshot(make_peer_snap(1, 5, 0));
  REQUIRE(coord.can_advance_boundary_zone(3, 1));
}

TEST_CASE(
    "ConcreteOuterSdCoordinator — OC-1 can_advance idempotent, no side "
    "effect on result",
    "[scheduler][t7_6][oc1]") {
  ts::ConcreteOuterSdCoordinator coord;
  coord.initialize(make_grid_2x1x1(), 4);
  coord.register_zone_peer_dependency(3, ts::PeerKey{1, 5});
  // Repeated calls without archive — result unchanged.
  REQUIRE_FALSE(coord.can_advance_boundary_zone(3, 1));
  REQUIRE_FALSE(coord.can_advance_boundary_zone(3, 1));
  REQUIRE_FALSE(coord.can_advance_boundary_zone(3, 1));
  coord.archive_peer_snapshot(make_peer_snap(1, 5, 0));
  // Repeated calls after archive — still consistent.
  REQUIRE(coord.can_advance_boundary_zone(3, 1));
  REQUIRE(coord.can_advance_boundary_zone(3, 1));
}

TEST_CASE("ConcreteOuterSdCoordinator — OC-1 target_level 0 trivially advanceable",
          "[scheduler][t7_6][oc1]") {
  ts::ConcreteOuterSdCoordinator coord;
  coord.initialize(make_grid_2x1x1(), 4);
  coord.register_zone_peer_dependency(0, ts::PeerKey{1, 0});
  // Step 0 — no peer history needed.
  REQUIRE(coord.can_advance_boundary_zone(0, 0));
}

TEST_CASE(
    "ConcreteOuterSdCoordinator — OC-2 register_boundary_snapshot collision "
    "is hard error in Reference",
    "[scheduler][t7_6][oc2][ha2]") {
  ts::ConcreteOuterSdCoordinator coord{ts::ConcreteOuterSdCoordinator::Mode::kReference};
  coord.initialize(make_grid_2x1x1(), 4);
  const auto snap = make_peer_snap(/*src_sub=*/0, /*src_zone=*/9, /*lvl=*/2);
  REQUIRE_NOTHROW(coord.register_boundary_snapshot(9, 2, snap));
  REQUIRE_THROWS_AS(coord.register_boundary_snapshot(9, 2, snap), std::logic_error);
  REQUIRE(coord.register_collisions_total() == 1);
}

TEST_CASE(
    "ConcreteOuterSdCoordinator — OC-2 Production mode logs but doesn't "
    "throw on collision",
    "[scheduler][t7_6][oc2]") {
  ts::ConcreteOuterSdCoordinator coord{ts::ConcreteOuterSdCoordinator::Mode::kProduction};
  coord.initialize(make_grid_2x1x1(), 4);
  const auto snap = make_peer_snap(0, 9, 2);
  REQUIRE_NOTHROW(coord.register_boundary_snapshot(9, 2, snap));
  REQUIRE_NOTHROW(coord.register_boundary_snapshot(9, 2, snap));
  REQUIRE(coord.register_collisions_total() == 1);
}

TEST_CASE(
    "ConcreteOuterSdCoordinator — OC-3 + HA-1 ring evicts oldest at "
    "K_max+1 insert",
    "[scheduler][t7_6][oc3][ha1]") {
  ts::ConcreteOuterSdCoordinator coord;
  const std::uint32_t k_max = 4;
  coord.initialize(make_grid_2x1x1(), k_max);
  const ts::PeerKey key{1, 7};
  for (ts::TimeLevel t = 0; t < k_max; ++t) {
    coord.archive_peer_snapshot(make_peer_snap(key.peer_subdomain, key.peer_zone, t));
  }
  REQUIRE(coord.peer_slot_count(key) == k_max);
  // K_max+1: must evict level 0 (oldest); level k_max becomes newest.
  coord.archive_peer_snapshot(make_peer_snap(key.peer_subdomain, key.peer_zone, k_max));
  REQUIRE(coord.peer_slot_count(key) == k_max);
  // Level 0 evicted → fetch returns nullopt + telemetry.
  const auto got = coord.fetch_peer_snapshot(key.peer_subdomain, key.peer_zone, 0);
  REQUIRE_FALSE(got.has_value());
  REQUIRE(coord.snapshot_too_old_total() == 1);
  // Level k_max present.
  const auto top = coord.fetch_peer_snapshot(key.peer_subdomain, key.peer_zone, k_max);
  REQUIRE(top.has_value());
}

TEST_CASE(
    "ConcreteOuterSdCoordinator — OC-4 fetch_peer_snapshot returns nullopt "
    "for unknown peer",
    "[scheduler][t7_6][oc4]") {
  ts::ConcreteOuterSdCoordinator coord;
  coord.initialize(make_grid_2x1x1(), 4);
  REQUIRE_FALSE(coord.fetch_peer_snapshot(99, 99, 5).has_value());
  REQUIRE(coord.snapshot_too_old_total() == 0);
}

TEST_CASE(
    "ConcreteOuterSdCoordinator — HA-3 fetch below oldest_level → nullopt + "
    "telemetry",
    "[scheduler][t7_6][ha3]") {
  ts::ConcreteOuterSdCoordinator coord;
  coord.initialize(make_grid_2x1x1(), 2);
  coord.archive_peer_snapshot(make_peer_snap(1, 5, 10));
  coord.archive_peer_snapshot(make_peer_snap(1, 5, 11));
  // Now insert level 12: ring full → evict 10. Fetching level 9 (which
  // was never present) and level 10 (evicted) both look "too old".
  coord.archive_peer_snapshot(make_peer_snap(1, 5, 12));
  REQUIRE_FALSE(coord.fetch_peer_snapshot(1, 5, 10).has_value());
  REQUIRE(coord.snapshot_too_old_total() == 1);
  REQUIRE_FALSE(coord.fetch_peer_snapshot(1, 5, 9).has_value());
  REQUIRE(coord.snapshot_too_old_total() == 2);
}

TEST_CASE(
    "ConcreteOuterSdCoordinator — HA-4 fetch above newest_level → nullopt, "
    "no telemetry",
    "[scheduler][t7_6][ha4]") {
  ts::ConcreteOuterSdCoordinator coord;
  coord.initialize(make_grid_2x1x1(), 4);
  coord.archive_peer_snapshot(make_peer_snap(1, 5, 0));
  coord.archive_peer_snapshot(make_peer_snap(1, 5, 1));
  REQUIRE_FALSE(coord.fetch_peer_snapshot(1, 5, 5).has_value());
  REQUIRE(coord.snapshot_too_old_total() == 0);  // HA-4 is normal stall
}

TEST_CASE("ConcreteOuterSdCoordinator — HA-5 eviction blocked while use_count > 0",
          "[scheduler][t7_6][ha5]") {
  ts::ConcreteOuterSdCoordinator coord;
  const std::uint32_t k_max = 2;
  coord.initialize(make_grid_2x1x1(), k_max);
  const ts::PeerKey key{1, 7};
  coord.archive_peer_snapshot(make_peer_snap(key.peer_subdomain, key.peer_zone, 0));
  coord.archive_peer_snapshot(make_peer_snap(key.peer_subdomain, key.peer_zone, 1));
  // Pin oldest slot via fetch.
  REQUIRE(coord.fetch_peer_snapshot(key.peer_subdomain, key.peer_zone, 0).has_value());
  // Eviction attempt fails because oldest slot is in use.
  REQUIRE_THROWS_AS(
      coord.archive_peer_snapshot(make_peer_snap(key.peer_subdomain, key.peer_zone, 2)),
      std::logic_error);
  // Release pin → eviction now succeeds.
  REQUIRE(coord.release_snapshot(key.peer_subdomain, key.peer_zone, 0));
  REQUIRE_NOTHROW(
      coord.archive_peer_snapshot(make_peer_snap(key.peer_subdomain, key.peer_zone, 2)));
  REQUIRE(coord.peer_slot_count(key) == k_max);
}

TEST_CASE(
    "ConcreteOuterSdCoordinator — HA-2 incoming peer snapshot at duplicate "
    "level rejected (Reference)",
    "[scheduler][t7_6][ha2]") {
  ts::ConcreteOuterSdCoordinator coord;
  coord.initialize(make_grid_2x1x1(), 4);
  coord.archive_peer_snapshot(make_peer_snap(1, 5, 3));
  REQUIRE_THROWS_AS(coord.archive_peer_snapshot(make_peer_snap(1, 5, 3)), std::logic_error);
  REQUIRE(coord.register_collisions_total() == 1);
}

TEST_CASE(
    "ConcreteOuterSdCoordinator — Reference received_seq deterministic + "
    "monotonic",
    "[scheduler][t7_6][determinism]") {
  ts::ConcreteOuterSdCoordinator coord;
  coord.initialize(make_grid_2x1x1(), 4);
  coord.archive_peer_snapshot(make_peer_snap(1, 5, 0));
  coord.archive_peer_snapshot(make_peer_snap(1, 5, 1));
  coord.archive_peer_snapshot(make_peer_snap(1, 6, 0));

  const auto a = coord.fetch_peer_snapshot(1, 5, 0);
  const auto b = coord.fetch_peer_snapshot(1, 5, 1);
  const auto c = coord.fetch_peer_snapshot(1, 6, 0);
  REQUIRE(a.has_value());
  REQUIRE(b.has_value());
  REQUIRE(c.has_value());
  REQUIRE(a->received_seq == 0);
  REQUIRE(b->received_seq == 1);
  REQUIRE(c->received_seq == 2);
}

TEST_CASE(
    "ConcreteOuterSdCoordinator — OC-5 set_global_frontier rejects backward "
    "motion",
    "[scheduler][t7_6][oc5]") {
  ts::ConcreteOuterSdCoordinator coord;
  coord.initialize(make_grid_2x1x1(), 4);
  REQUIRE(coord.global_frontier_min() == 0);
  REQUIRE(coord.global_frontier_max() == 0);
  REQUIRE_NOTHROW(coord.set_global_frontier(2, 5));
  REQUIRE(coord.global_frontier_min() == 2);
  REQUIRE(coord.global_frontier_max() == 5);
  REQUIRE_NOTHROW(coord.set_global_frontier(2, 5));  // idempotent
  REQUIRE_NOTHROW(coord.set_global_frontier(3, 6));
  REQUIRE_THROWS_AS(coord.set_global_frontier(2, 6), std::logic_error);
  REQUIRE_THROWS_AS(coord.set_global_frontier(3, 5), std::logic_error);
  REQUIRE_THROWS_AS(coord.set_global_frontier(8, 4), std::logic_error);
}

TEST_CASE(
    "ConcreteOuterSdCoordinator — OC-6 stall watchdog emits report past "
    "T_stall_max",
    "[scheduler][t7_6][oc6]") {
  ts::ConcreteOuterSdCoordinator coord;
  coord.initialize(make_grid_2x1x1(), 4);
  coord.register_zone_peer_dependency(3, ts::PeerKey{1, 5});

  // First poll → records waiting_since.
  REQUIRE_FALSE(coord.can_advance_boundary_zone(3, 1));
  REQUIRE(coord.drain_stall_reports().empty());

  // T_stall_max immediately → no stall yet (elapsed is ~0ms).
  coord.check_stall_boundaries(std::chrono::milliseconds{1'000});
  REQUIRE(coord.drain_stall_reports().empty());

  // Sleep just over T_stall_max=10ms and re-check.
  std::this_thread::sleep_for(std::chrono::milliseconds{20});
  coord.check_stall_boundaries(std::chrono::milliseconds{10});
  const auto reports = coord.drain_stall_reports();
  REQUIRE(reports.size() == 1);
  REQUIRE(reports[0].local_zone == 3);
  REQUIRE(reports[0].waiting_for_level == 1);
  REQUIRE(reports[0].peer.peer_subdomain == 1);
  REQUIRE(reports[0].peer.peer_zone == 5);

  // Once peer arrives, repeating can_advance clears the wait, so a
  // subsequent watchdog tick yields no report.
  coord.archive_peer_snapshot(make_peer_snap(1, 5, 0));
  REQUIRE(coord.can_advance_boundary_zone(3, 1));
  std::this_thread::sleep_for(std::chrono::milliseconds{20});
  coord.check_stall_boundaries(std::chrono::milliseconds{10});
  REQUIRE(coord.drain_stall_reports().empty());
}

TEST_CASE(
    "ConcreteOuterSdCoordinator — release_snapshot returns false for "
    "unknown / non-pinned slot",
    "[scheduler][t7_6][ha5]") {
  ts::ConcreteOuterSdCoordinator coord;
  coord.initialize(make_grid_2x1x1(), 4);
  REQUIRE_FALSE(coord.release_snapshot(99, 99, 0));
  coord.archive_peer_snapshot(make_peer_snap(1, 5, 0));
  REQUIRE_FALSE(coord.release_snapshot(1, 5, 0));  // never fetched
  REQUIRE(coord.fetch_peer_snapshot(1, 5, 0).has_value());
  REQUIRE(coord.release_snapshot(1, 5, 0));
  REQUIRE_FALSE(coord.release_snapshot(1, 5, 0));  // already at 0
}

TEST_CASE(
    "ConcreteOuterSdCoordinator — out-of-order archive insertion preserves "
    "ascending order",
    "[scheduler][t7_6][ordering]") {
  ts::ConcreteOuterSdCoordinator coord;
  coord.initialize(make_grid_2x1x1(), 4);
  // Insert levels in scrambled order.
  coord.archive_peer_snapshot(make_peer_snap(1, 5, 2));
  coord.archive_peer_snapshot(make_peer_snap(1, 5, 0));
  coord.archive_peer_snapshot(make_peer_snap(1, 5, 1));
  REQUIRE(coord.fetch_peer_snapshot(1, 5, 0).has_value());
  REQUIRE(coord.fetch_peer_snapshot(1, 5, 1).has_value());
  REQUIRE(coord.fetch_peer_snapshot(1, 5, 2).has_value());
  REQUIRE(coord.snapshot_too_old_total() == 0);
}
