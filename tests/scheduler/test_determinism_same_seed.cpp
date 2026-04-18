// Exec pack: docs/development/m4_execution_pack.md T4.10
// SPEC: docs/specs/scheduler/SPEC.md §12.3 (determinism — Level 1)
// Master spec: §7.3, §13.4
//
// T4.10 (a) — same-seed same-binary: two engines, identical YAML, 10 steps
// each → byte-exact final atom SoA + byte-exact scheduler event log.
//
// This extends the T4.9 thermo byte-exact check to the full integrator state
// (positions, velocities, forces) and the scheduler event history. Level 1
// determinism (SPEC §12.3) says: on a fixed binary/flavor, the same input
// produces identical bits. Cross-compiler / cross-arch parity is Level 2+
// (M5 anchor-test) and not in M4 scope.
//
// Why directly compare AtomSoA bits and EventLog: thermo aggregates
// information — two runs can hit identical thermo while silently diverging
// on per-atom state. Atom-level byte-exactness is the actual D-M4-9 premise.
// Event log byte-exactness ensures the scheduler's decision sequence is also
// reproducible — a prerequisite for M5 anchor-test reproducibility claims.

#include "tdmd/io/yaml_config.hpp"
#include "tdmd/runtime/simulation_engine.hpp"
#include "tdmd/scheduler/causal_wavefront_scheduler.hpp"
#include "tdmd/scheduler/event_log.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#ifndef TDMD_IO_FIXTURES_DIR
#error "TDMD_IO_FIXTURES_DIR must be defined by the build system"
#endif

namespace {

// Mirror of test_td_mode_smoke.cpp config — shrunken Morse so the 8.1 Å box
// satisfies the cutoff stencil.
tdmd::io::YamlConfig load_td_config() {
  const std::string path = std::string(TDMD_IO_FIXTURES_DIR) + "/configs/valid_nve_al.yaml";
  auto config = tdmd::io::parse_yaml_config(path);
  config.potential.morse.r0 = 2.0;
  config.potential.morse.cutoff = 2.5;
  config.potential.morse.alpha = 2.0;
  config.neighbor.skin = 0.1;
  config.scheduler.td_mode = true;
  return config;
}

std::string valid_config_dir() {
  return std::string(TDMD_IO_FIXTURES_DIR) + "/configs";
}

// Compare two double arrays bit-for-bit (memcmp of the underlying bytes).
// The alternative, double equality, treats NaN as unequal, which is what we
// want for a byte-exact test anyway.
bool bitwise_equal(const double* a, const double* b, std::size_t n) {
  return std::memcmp(a, b, n * sizeof(double)) == 0;
}

}  // namespace

TEST_CASE("T4.10 (a) — same-seed same-binary: atom SoA byte-exact across two TD runs",
          "[scheduler][determinism][same-seed]") {
  constexpr std::uint64_t kSteps = 10;

  tdmd::SimulationEngine engine_a;
  engine_a.init(load_td_config(), valid_config_dir());
  std::ostringstream thermo_a;
  (void) engine_a.run(kSteps, &thermo_a);

  tdmd::SimulationEngine engine_b;
  engine_b.init(load_td_config(), valid_config_dir());
  std::ostringstream thermo_b;
  (void) engine_b.run(kSteps, &thermo_b);

  const auto& a = engine_a.atoms();
  const auto& b = engine_b.atoms();

  REQUIRE(a.size() == b.size());
  const std::size_t n = a.size();

  // Positions, velocities, forces — every SoA column must match bit-for-bit.
  CHECK(bitwise_equal(a.x.data(), b.x.data(), n));
  CHECK(bitwise_equal(a.y.data(), b.y.data(), n));
  CHECK(bitwise_equal(a.z.data(), b.z.data(), n));
  CHECK(bitwise_equal(a.vx.data(), b.vx.data(), n));
  CHECK(bitwise_equal(a.vy.data(), b.vy.data(), n));
  CHECK(bitwise_equal(a.vz.data(), b.vz.data(), n));
  CHECK(bitwise_equal(a.fx.data(), b.fx.data(), n));
  CHECK(bitwise_equal(a.fy.data(), b.fy.data(), n));
  CHECK(bitwise_equal(a.fz.data(), b.fz.data(), n));

  // Thermo stream byte-exact (also covered by T4.9 smoke, asserted here for
  // completeness — a Level 1 determinism failure might be subtler than
  // per-column memcmp can catch alone).
  CHECK(thermo_a.str() == thermo_b.str());

  engine_a.finalize();
  engine_b.finalize();
}

TEST_CASE("T4.10 (a) — scheduler event log byte-exact across two TD runs",
          "[scheduler][determinism][same-seed][event-log]") {
  constexpr std::uint64_t kSteps = 10;

  tdmd::SimulationEngine engine_a;
  engine_a.init(load_td_config(), valid_config_dir());
  std::ostringstream sink_a;
  (void) engine_a.run(kSteps, &sink_a);

  tdmd::SimulationEngine engine_b;
  engine_b.init(load_td_config(), valid_config_dir());
  std::ostringstream sink_b;
  (void) engine_b.run(kSteps, &sink_b);

  const auto* sched_a = engine_a.td_scheduler_for_testing();
  const auto* sched_b = engine_b.td_scheduler_for_testing();
  REQUIRE(sched_a != nullptr);
  REQUIRE(sched_b != nullptr);

  const auto events_a = sched_a->event_log().snapshot();
  const auto events_b = sched_b->event_log().snapshot();

  REQUIRE(events_a.size() == events_b.size());
  REQUIRE_FALSE(events_a.empty());

  // Timestamps are steady_clock — non-deterministic by definition. Compare
  // only the decision fields (kind, zone_id, time_level, count).
  for (std::size_t i = 0; i < events_a.size(); ++i) {
    INFO("event index " << i);
    CHECK(events_a[i].kind == events_b[i].kind);
    CHECK(events_a[i].zone_id == events_b[i].zone_id);
    CHECK(events_a[i].time_level == events_b[i].time_level);
    CHECK(events_a[i].count == events_b[i].count);
  }

  engine_a.finalize();
  engine_b.finalize();
}

TEST_CASE("T4.10 (a) — event log not bound by diagnostic window", "[scheduler][event-log]") {
  // OQ-M4-4: event log capacity is 1024; diagnostic dump surfaces 100.
  // Assert the wider buffer is observable via the public event_log() hook.
  constexpr std::uint64_t kSteps = 5;

  tdmd::SimulationEngine engine;
  engine.init(load_td_config(), valid_config_dir());
  std::ostringstream sink;
  (void) engine.run(kSteps, &sink);

  const auto* sched = engine.td_scheduler_for_testing();
  REQUIRE(sched != nullptr);
  const auto& log = sched->event_log();

  CHECK(tdmd::scheduler::EventLog::capacity() == 1024);
  CHECK(log.size() <= log.capacity());
  CHECK(log.size() > 0);

  engine.finalize();
}

TEST_CASE("T4.11 — TD scheduler counter sanity in K=1 single-rank",
          "[scheduler][td-mode][counters]") {
  // T4.11 acceptance proxy for "scheduler telemetry counters match expected":
  //   - every zone commits once per step → count(CommitCompleted) ≥ n_steps
  //   - no cert-invalidation rollbacks (always-safe stub — can't trip)
  //   - no deadlock events fired
  //   - K=1: current_pipeline_depth() == 1 after init (D-M4-1)
  constexpr std::uint64_t kSteps = 5;

  tdmd::SimulationEngine engine;
  engine.init(load_td_config(), valid_config_dir());
  std::ostringstream sink;
  (void) engine.run(kSteps, &sink);

  const auto* sched = engine.td_scheduler_for_testing();
  REQUIRE(sched != nullptr);

  // CommitCompleted is emitted once per commit_completed() call with
  // `count` carrying the batch size. Sum the counts across all events
  // of that kind and assert ≥ kSteps × total_zones committed overall.
  // (Each step in K=1 drives a full Ready → Committed sweep.)
  const auto events = sched->event_log().snapshot();
  std::size_t commit_events = 0;
  std::uint64_t total_committed = 0;
  std::size_t deadlock_events = 0;
  std::size_t rollback_events = 0;
  for (const auto& e : events) {
    switch (e.kind) {
      case tdmd::scheduler::SchedulerEvent::CommitCompleted:
        ++commit_events;
        total_committed += e.count;
        break;
      case tdmd::scheduler::SchedulerEvent::DeadlockFired:
        ++deadlock_events;
        break;
      case tdmd::scheduler::SchedulerEvent::CertInvalidatedRollback:
        ++rollback_events;
        break;
      default:
        break;
    }
  }

  // Single-rank K=1: every step commits the full set of resident zones,
  // so total_committed == total_zones × kSteps. We assert ≥ to tolerate
  // the extra commit that may fire at initialize() priming.
  CHECK(commit_events >= kSteps);
  CHECK(total_committed >= kSteps * sched->total_zones());

  CHECK(deadlock_events == 0);  // SPEC §8.1 watchdog must NOT trip in happy path
  CHECK(rollback_events == 0);  // AlwaysSafeCertificateInputSource → no invalidations

  // D-M4-1: pipeline depth is 1 in M4. `current_pipeline_depth()`
  // reports frontier_max - frontier_min; after init+N steps with all
  // zones advanced uniformly, the depth is 0 when quiesced and ≤ 1
  // when one arrival has happened without a matching commit.
  CHECK(sched->current_pipeline_depth() <= 1u);

  engine.finalize();
}
