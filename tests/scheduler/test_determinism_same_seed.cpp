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
