// SPEC: docs/specs/scheduler/SPEC.md §2.4 (OC-1..OC-6), §2.5 (dep_mask bit 4),
//       §5.1 Pattern-2 boundary branch
// Master spec: §6.3, §12.7a
// Exec pack: docs/development/m7_execution_pack.md T7.7
//
// Pattern 2 boundary-dependency wiring tests. Verifies the
// CausalWavefrontScheduler hooks into OuterSdCoordinator at three points:
//
//   1. select_ready_tasks — gates boundary zones on
//      can_advance_boundary_zone (SPEC §5.1 Pattern-2 branch).
//   2. commit_completed — Phase B hook invokes register_boundary_snapshot
//      via the user-supplied builder closure (SPEC §6.2 + §2.4 OC-2).
//   3. check_deadlock — forwards check_stall_boundaries to the coordinator
//      with the boundary-specific threshold (SPEC §2.4 OC-6).
//
// Pattern 1 byte-exactness is preserved by two independent guards:
//   (a) outer_coord_ == nullptr (no coordinator attached), or
//   (b) is_boundary_zone(z) == false (empty flags vector or per-zone false).
// Either is sufficient to skip every Pattern 2 hook.

#include "tdmd/scheduler/causal_wavefront_scheduler.hpp"
#include "tdmd/scheduler/concrete_outer_sd_coordinator.hpp"
#include "tdmd/scheduler/halo_snapshot.hpp"
#include "tdmd/scheduler/outer_sd_coordinator.hpp"
#include "tdmd/scheduler/policy.hpp"
#include "tdmd/scheduler/subdomain_grid.hpp"
#include "tdmd/state/box.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ts = tdmd::scheduler;
namespace tz = tdmd::zoning;

namespace {

tz::ZoningPlan make_plan(std::uint32_t nx, std::uint32_t ny = 1, std::uint32_t nz = 1) {
  tz::ZoningPlan plan;
  plan.scheme = tz::ZoningScheme::Linear1D;
  plan.n_zones = {nx, ny, nz};
  plan.zone_size = {1.0, 1.0, 1.0};
  plan.cutoff = 0.5;
  plan.skin = 0.5;
  plan.buffer_width = {0.5, 0.5, 0.5};
  const auto total = static_cast<tz::ZoneId>(nx * ny * nz);
  plan.canonical_order.reserve(total);
  for (tz::ZoneId z = 0; z < total; ++z) {
    plan.canonical_order.push_back(z);
  }
  plan.n_min_per_rank = 1;
  plan.optimal_rank_count = total;
  return plan;
}

ts::SubdomainGrid make_grid_2x1x1() {
  ts::SubdomainGrid g;
  g.n_subdomains = {2, 1, 1};
  g.subdomain_boxes.resize(2);
  g.subdomain_boxes[0] = tdmd::Box{0.0, 5.0, 0.0, 10.0, 0.0, 10.0, true, true, true};
  g.subdomain_boxes[1] = tdmd::Box{5.0, 10.0, 0.0, 10.0, 0.0, 10.0, true, true, true};
  g.rank_of_subdomain = {0, 1};
  return g;
}

// Always-safe certificate input source. Mirrors test_select_ready_tasks.cpp.
struct SafeSource : ts::CertificateInputSource {
  void fill_inputs(ts::ZoneId zone,
                   ts::TimeLevel time_level,
                   ts::CertificateInputs& out) const override {
    out.zone_id = zone;
    out.time_level = time_level;
    out.v_max_zone = 0.1;
    out.a_max_zone = 0.2;
    out.dt_candidate = 0.001;
    out.buffer_width = 1.0;
    out.skin_remaining = 1.0;
    out.frontier_margin = 1.0;
    out.neighbor_valid_until_step = time_level + 100;
    out.halo_valid_until_step = time_level + 100;
  }
};

ts::SchedulerPolicy ref_policy(std::uint32_t cap = 16, std::uint32_t k_max = 1) {
  auto p = ts::PolicyFactory::for_reference();
  p.max_tasks_per_iteration = cap;
  p.k_max_pipeline_depth = k_max;
  return p;
}

void prime_all_zones(ts::CausalWavefrontScheduler& sched, ts::TimeLevel start_level = 0) {
  for (std::size_t z = 0; z < sched.total_zones(); ++z) {
    sched.on_zone_data_arrived(static_cast<ts::ZoneId>(z), start_level, 0);
  }
}

// Minimal observability shim — counts forwarded calls to the OC-* methods so
// tests can assert the scheduler dispatched into the coordinator without
// relying on side effects of the concrete implementation.
class CountingOuterCoord : public ts::OuterSdCoordinator {
public:
  void initialize(const ts::SubdomainGrid&, std::uint32_t) override {}
  bool can_advance_boundary_zone(ts::ZoneId zone, ts::TimeLevel level) override {
    last_can_advance_zone = zone;
    last_can_advance_level = level;
    ++can_advance_calls;
    return can_advance_result;
  }
  void register_boundary_snapshot(ts::ZoneId zone,
                                  ts::TimeLevel level,
                                  const ts::HaloSnapshot& snap) override {
    last_register_zone = zone;
    last_register_level = level;
    last_register_atom_count = snap.atom_count;
    ++register_calls;
  }
  std::optional<ts::HaloSnapshot> fetch_peer_snapshot(std::uint32_t,
                                                      ts::ZoneId,
                                                      ts::TimeLevel) override {
    return std::nullopt;
  }
  void check_stall_boundaries(std::chrono::milliseconds t) override {
    last_stall_threshold = t;
    ++check_stall_calls;
  }
  ts::TimeLevel global_frontier_min() const override { return 0; }
  ts::TimeLevel global_frontier_max() const override { return 0; }

  bool can_advance_result = true;
  std::uint64_t can_advance_calls = 0;
  ts::ZoneId last_can_advance_zone = 0xFFFFFFFFU;
  ts::TimeLevel last_can_advance_level = 0;
  std::uint64_t register_calls = 0;
  ts::ZoneId last_register_zone = 0xFFFFFFFFU;
  ts::TimeLevel last_register_level = 0;
  std::uint32_t last_register_atom_count = 0;
  std::uint64_t check_stall_calls = 0;
  std::chrono::milliseconds last_stall_threshold{0};
};

}  // namespace

TEST_CASE("Pattern 1 byte-exact — no outer coord attached, all hooks dormant",
          "[scheduler][t7_7][pattern1]") {
  ts::CausalWavefrontScheduler sched{ref_policy()};
  sched.initialize(make_plan(2));
  SafeSource src;
  sched.set_certificate_input_source(&src);
  prime_all_zones(sched);
  sched.refresh_certificates();

  // No coordinator → boundary state must be at-rest defaults.
  REQUIRE(sched.outer_coordinator() == nullptr);
  REQUIRE(sched.boundary_gates_blocked_count() == 0);
  REQUIRE(sched.boundary_registers_emitted_count() == 0);
  REQUIRE(sched.boundary_register_skips_count() == 0);

  // is_boundary_zone defaults to all-false on empty flags.
  for (ts::ZoneId z = 0; z < sched.total_zones(); ++z) {
    REQUIRE_FALSE(sched.is_boundary_zone(z));
  }

  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 2);
  REQUIRE(sched.boundary_gates_blocked_count() == 0);
}

TEST_CASE("Coord attached but no boundary flags — Pattern 1 byte-exact preserved",
          "[scheduler][t7_7][pattern1]") {
  ts::CausalWavefrontScheduler sched{ref_policy()};
  sched.initialize(make_plan(2));
  SafeSource src;
  sched.set_certificate_input_source(&src);

  CountingOuterCoord coord;
  coord.can_advance_result = false;  // would block IF queried
  sched.attach_outer_coordinator(&coord);

  prime_all_zones(sched);
  sched.refresh_certificates();

  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 2);
  // Coord must NOT be queried — both zones report is_boundary_zone == false.
  REQUIRE(coord.can_advance_calls == 0);
  REQUIRE(sched.boundary_gates_blocked_count() == 0);
}

TEST_CASE("set_boundary_zone_flags rejects size mismatch", "[scheduler][t7_7][api]") {
  ts::CausalWavefrontScheduler sched{ref_policy()};
  sched.initialize(make_plan(4));
  REQUIRE_THROWS_AS(sched.set_boundary_zone_flags(std::vector<bool>(3, true)), std::logic_error);
  REQUIRE_THROWS_AS(sched.set_boundary_zone_flags(std::vector<bool>(5, false)), std::logic_error);
  // Correct size must succeed.
  REQUIRE_NOTHROW(sched.set_boundary_zone_flags(std::vector<bool>(4, true)));
  REQUIRE(sched.is_boundary_zone(0));
  REQUIRE(sched.is_boundary_zone(3));
}

TEST_CASE("is_boundary_zone — out-of-range or empty flags reports false",
          "[scheduler][t7_7][api]") {
  ts::CausalWavefrontScheduler sched{ref_policy()};
  sched.initialize(make_plan(2));
  // Empty flags before any set_boundary_zone_flags call.
  REQUIRE_FALSE(sched.is_boundary_zone(0));
  REQUIRE_FALSE(sched.is_boundary_zone(1));
  REQUIRE_FALSE(sched.is_boundary_zone(99));  // out of range
}

TEST_CASE("Boundary gate blocks zone when coord can_advance returns false",
          "[scheduler][t7_7][gate]") {
  ts::CausalWavefrontScheduler sched{ref_policy(/*cap=*/16, /*k_max=*/1)};
  sched.initialize(make_plan(2));
  SafeSource src;
  sched.set_certificate_input_source(&src);

  CountingOuterCoord coord;
  coord.can_advance_result = false;
  sched.attach_outer_coordinator(&coord);

  // Zone 0 boundary, zone 1 interior.
  sched.set_boundary_zone_flags({true, false});

  prime_all_zones(sched);
  sched.refresh_certificates();

  const auto tasks = sched.select_ready_tasks();
  // Only zone 1 (interior) emits; zone 0 is gated.
  REQUIRE(tasks.size() == 1);
  REQUIRE(tasks[0].zone_id == 1);
  // Coord queried exactly once for zone 0 at level 1.
  REQUIRE(coord.can_advance_calls == 1);
  REQUIRE(coord.last_can_advance_zone == 0);
  REQUIRE(coord.last_can_advance_level == 1);
  REQUIRE(sched.boundary_gates_blocked_count() == 1);
}

TEST_CASE("Boundary gate unblocks zone when coord can_advance returns true",
          "[scheduler][t7_7][gate]") {
  ts::CausalWavefrontScheduler sched{ref_policy(/*cap=*/16, /*k_max=*/1)};
  sched.initialize(make_plan(2));
  SafeSource src;
  sched.set_certificate_input_source(&src);

  CountingOuterCoord coord;
  coord.can_advance_result = true;
  sched.attach_outer_coordinator(&coord);
  sched.set_boundary_zone_flags({true, true});

  prime_all_zones(sched);
  sched.refresh_certificates();

  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 2);
  REQUIRE(coord.can_advance_calls == 2);
  REQUIRE(sched.boundary_gates_blocked_count() == 0);
}

TEST_CASE("Phase B hook fires register_boundary_snapshot via builder",
          "[scheduler][t7_7][phaseB]") {
  ts::CausalWavefrontScheduler sched{ref_policy(/*cap=*/16, /*k_max=*/1)};
  sched.initialize(make_plan(2));
  SafeSource src;
  sched.set_certificate_input_source(&src);

  CountingOuterCoord coord;
  coord.can_advance_result = true;
  sched.attach_outer_coordinator(&coord);
  sched.set_boundary_zone_flags({true, false});  // zone 0 boundary only

  // Builder returns a snapshot stamped with atom_count = 99 + level so we
  // can assert the hook actually consumed our closure.
  sched.set_boundary_snapshot_builder([](ts::ZoneId z, ts::TimeLevel t) {
    ts::HaloSnapshot s;
    s.source_zone_id = z;
    s.time_level = t;
    s.atom_count = 99u + static_cast<std::uint32_t>(t);
    return s;
  });

  prime_all_zones(sched);
  sched.refresh_certificates();
  auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 2);

  // Drive both zones through Phase A.
  for (const auto& t : tasks) {
    sched.mark_computing(t);
    sched.mark_completed(t);
  }
  sched.commit_completed();

  // Exactly one register call (only zone 0 is boundary). Atom_count proves
  // the builder closure was invoked with the right (zone, level).
  REQUIRE(coord.register_calls == 1);
  REQUIRE(coord.last_register_zone == 0);
  REQUIRE(coord.last_register_level == 1);
  REQUIRE(coord.last_register_atom_count == 100u);  // 99 + 1
  REQUIRE(sched.boundary_registers_emitted_count() == 1);
  REQUIRE(sched.boundary_register_skips_count() == 0);
}

TEST_CASE("Phase B hook skips register when no builder is set", "[scheduler][t7_7][phaseB]") {
  ts::CausalWavefrontScheduler sched{ref_policy(/*cap=*/16, /*k_max=*/1)};
  sched.initialize(make_plan(2));
  SafeSource src;
  sched.set_certificate_input_source(&src);

  CountingOuterCoord coord;
  coord.can_advance_result = true;
  sched.attach_outer_coordinator(&coord);
  sched.set_boundary_zone_flags({true, true});  // both boundary

  // Note: NO snapshot builder set.

  prime_all_zones(sched);
  sched.refresh_certificates();
  auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 2);
  for (const auto& t : tasks) {
    sched.mark_computing(t);
    sched.mark_completed(t);
  }
  sched.commit_completed();

  // No register calls reach the coord; both boundary zones counted as skips.
  REQUIRE(coord.register_calls == 0);
  REQUIRE(sched.boundary_registers_emitted_count() == 0);
  REQUIRE(sched.boundary_register_skips_count() == 2);
}

TEST_CASE("Phase B hook leaves non-boundary zones alone", "[scheduler][t7_7][phaseB]") {
  ts::CausalWavefrontScheduler sched{ref_policy(/*cap=*/16, /*k_max=*/1)};
  sched.initialize(make_plan(2));
  SafeSource src;
  sched.set_certificate_input_source(&src);

  CountingOuterCoord coord;
  coord.can_advance_result = true;
  sched.attach_outer_coordinator(&coord);
  sched.set_boundary_zone_flags({false, false});  // neither boundary

  // Builder is set but should not be invoked.
  std::uint64_t builder_calls = 0;
  sched.set_boundary_snapshot_builder([&builder_calls](ts::ZoneId, ts::TimeLevel) {
    ++builder_calls;
    return ts::HaloSnapshot{};
  });

  prime_all_zones(sched);
  sched.refresh_certificates();
  auto tasks = sched.select_ready_tasks();
  for (const auto& t : tasks) {
    sched.mark_computing(t);
    sched.mark_completed(t);
  }
  sched.commit_completed();

  REQUIRE(builder_calls == 0);
  REQUIRE(coord.register_calls == 0);
  REQUIRE(sched.boundary_registers_emitted_count() == 0);
  REQUIRE(sched.boundary_register_skips_count() == 0);
}

TEST_CASE("check_deadlock forwards check_stall_boundaries to coord",
          "[scheduler][t7_7][watchdog]") {
  ts::CausalWavefrontScheduler sched{ref_policy()};
  sched.initialize(make_plan(2));
  SafeSource src;
  sched.set_certificate_input_source(&src);
  sched.set_target_time_level(100);  // keep finished()==false

  CountingOuterCoord coord;
  coord.can_advance_result = true;
  sched.attach_outer_coordinator(&coord);

  // Per-boundary threshold defaults to 0 → falls back to t_watchdog.
  sched.check_deadlock(std::chrono::milliseconds{60'000});
  REQUIRE(coord.check_stall_calls == 1);
  REQUIRE(coord.last_stall_threshold == std::chrono::milliseconds{60'000});

  // Setting an explicit threshold uses it instead of t_watchdog.
  sched.set_boundary_stall_max(std::chrono::milliseconds{250});
  sched.check_deadlock(std::chrono::milliseconds{60'000});
  REQUIRE(coord.check_stall_calls == 2);
  REQUIRE(coord.last_stall_threshold == std::chrono::milliseconds{250});
}

TEST_CASE("check_deadlock skips watchdog forwarding when no coord attached",
          "[scheduler][t7_7][watchdog]") {
  ts::CausalWavefrontScheduler sched{ref_policy()};
  sched.initialize(make_plan(2));
  SafeSource src;
  sched.set_certificate_input_source(&src);
  sched.set_target_time_level(100);  // finished()==false

  // No coord attached. Must not throw or touch any null pointer.
  REQUIRE_NOTHROW(sched.check_deadlock(std::chrono::milliseconds{60'000}));
  REQUIRE(sched.outer_coordinator() == nullptr);
}

TEST_CASE("shutdown clears all T7.7 boundary state", "[scheduler][t7_7][lifecycle]") {
  ts::CausalWavefrontScheduler sched{ref_policy()};
  sched.initialize(make_plan(2));
  SafeSource src;
  sched.set_certificate_input_source(&src);

  CountingOuterCoord coord;
  coord.can_advance_result = false;
  sched.attach_outer_coordinator(&coord);
  sched.set_boundary_zone_flags({true, true});
  sched.set_boundary_snapshot_builder([](ts::ZoneId, ts::TimeLevel) { return ts::HaloSnapshot{}; });
  sched.set_boundary_stall_max(std::chrono::milliseconds{1});

  prime_all_zones(sched);
  sched.refresh_certificates();
  (void) sched.select_ready_tasks();  // increments boundary_gates_blocked_
  REQUIRE(sched.boundary_gates_blocked_count() > 0);

  sched.shutdown();

  // Re-initialize; boundary state must be back at defaults.
  sched.initialize(make_plan(2));
  REQUIRE(sched.outer_coordinator() == nullptr);
  REQUIRE(sched.boundary_gates_blocked_count() == 0);
  REQUIRE(sched.boundary_registers_emitted_count() == 0);
  REQUIRE(sched.boundary_register_skips_count() == 0);
  for (ts::ZoneId z = 0; z < sched.total_zones(); ++z) {
    REQUIRE_FALSE(sched.is_boundary_zone(z));
  }
}

TEST_CASE("End-to-end: coord blocks, peer arrives via archive, zone unblocks",
          "[scheduler][t7_7][e2e]") {
  // Use the real ConcreteOuterSdCoordinator to exercise the round-trip
  // semantics (peer dependency + archive insertion). Two-zone single-rank
  // plan; we stand up a 2-subdomain grid only because the coord requires
  // one. Zone 0 declared boundary with one peer dependency that is not yet
  // satisfied → gated. After archive_peer_snapshot at level 0, the zone
  // can advance to t=1.

  ts::CausalWavefrontScheduler sched{ref_policy(/*cap=*/16, /*k_max=*/1)};
  sched.initialize(make_plan(2));
  SafeSource src;
  sched.set_certificate_input_source(&src);

  ts::ConcreteOuterSdCoordinator coord;
  coord.initialize(make_grid_2x1x1(), /*k_max=*/4);
  // Local zone 0 depends on (peer_subdomain=1, peer_zone=0).
  coord.register_zone_peer_dependency(0, ts::PeerKey{1, 0});

  sched.attach_outer_coordinator(&coord);
  sched.set_boundary_zone_flags({true, false});

  prime_all_zones(sched);
  sched.refresh_certificates();

  // First pass: zone 0 gated (no peer snap at level 0); zone 1 emits.
  {
    const auto tasks = sched.select_ready_tasks();
    REQUIRE(tasks.size() == 1);
    REQUIRE(tasks[0].zone_id == 1);
    REQUIRE(sched.boundary_gates_blocked_count() == 1);
  }

  // Inject the missing peer snapshot at level 0.
  ts::HaloSnapshot snap;
  snap.source_subdomain_id = 1;
  snap.source_zone_id = 0;
  snap.time_level = 0;
  snap.source_version = 1;
  snap.atom_count = 4;
  snap.payload.assign(4 * 64, std::uint8_t{0xAA});
  coord.archive_peer_snapshot(std::move(snap));

  // Second pass: zone 0 should now be selectable. Zone 1 already Ready, so
  // skip if its state machine has moved on; we re-check the gate by hand.
  REQUIRE(coord.can_advance_boundary_zone(/*local_zone=*/0, /*target_level=*/1));
}
