// T7.9 — SimulationEngine Pattern 2 wire smoke.
//
// Acceptance (exec pack §T7.9):
//   - Pattern 1 regression: default YAML (no `zoning.subdomains`) leaves
//     `outer_sd_coordinator_for_testing()` == nullptr — M1..M6 legacy path
//     byte-for-byte identical.
//   - Pattern 2 opt-in: `zoning.subdomains: [2, 1, 1]` constructs a live
//     `ConcreteOuterSdCoordinator` owned by the engine, initialized with
//     `k_max = scheduler.pipeline_depth_cap`, and attached to the inner
//     scheduler when `td_mode: true`.
//   - Preflight rejection: malformed subdomains (e.g. `[2, 0, 1]`) fail the
//     schema preflight with a clear `zoning.subdomains` key path.
//
// In-process test (no MPI); CLI-level HybridBackend construction + transport
// probe chain belongs to T7.14. We exercise the engine-internal wire only,
// which is what unblocks T7.10 / T7.11 / T7.12.

#include "tdmd/io/preflight.hpp"
#include "tdmd/io/yaml_config.hpp"
#include "tdmd/runtime/simulation_engine.hpp"
#include "tdmd/scheduler/outer_sd_coordinator.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <string>

#ifndef TDMD_IO_FIXTURES_DIR
#error "TDMD_IO_FIXTURES_DIR must be defined by the build system"
#endif

namespace {

// Reuse the T1.4 fixture (32-atom Al FCC Morse). For a Pattern 2 wire test the
// physics doesn't matter — only that init() reaches the coord-construction
// code without exploding. Matching test_simulation_engine.cpp's toy-parameter
// override so the 8.1 Å box satisfies the 3-cells-per-axis stencil.
tdmd::io::YamlConfig load_base_config() {
  const std::string path = std::string(TDMD_IO_FIXTURES_DIR) + "/configs/valid_nve_al.yaml";
  auto config = tdmd::io::parse_yaml_config(path);
  config.potential.morse.r0 = 2.0;
  config.potential.morse.cutoff = 2.5;
  config.potential.morse.alpha = 2.0;
  config.neighbor.skin = 0.1;
  return config;
}

std::string base_config_dir() {
  return std::string(TDMD_IO_FIXTURES_DIR) + "/configs";
}

bool any_error_at(const std::vector<tdmd::io::PreflightError>& errors, const char* key) {
  return std::any_of(errors.begin(), errors.end(), [&](const tdmd::io::PreflightError& e) {
    return e.severity == tdmd::io::PreflightSeverity::Error && e.key_path == key;
  });
}

}  // namespace

TEST_CASE("SimulationEngine Pattern 1 regression — no coordinator by default",
          "[runtime][engine][pattern2][t79]") {
  auto config = load_base_config();
  REQUIRE(config.zoning.subdomains[0] == 1U);
  REQUIRE(config.zoning.subdomains[1] == 1U);
  REQUIRE(config.zoning.subdomains[2] == 1U);

  tdmd::SimulationEngine engine;
  engine.init(config, base_config_dir());

  // D-M7-2 contract: default config leaves engine in Pattern 1 — coord null.
  REQUIRE(engine.outer_sd_coordinator_for_testing() == nullptr);

  engine.finalize();
}

TEST_CASE("SimulationEngine Pattern 2 opt-in — coordinator constructed + attached",
          "[runtime][engine][pattern2][t79]") {
  auto config = load_base_config();
  // Opt-in to Pattern 2 along x. [2,1,1] is the minimum non-trivial case and
  // matches the M7 2-rank smoke geometry the exec pack calls out.
  config.zoning.subdomains = {2U, 1U, 1U};

  SECTION("td_mode=false — coord owned by engine, scheduler path untouched") {
    config.scheduler.td_mode = false;

    tdmd::SimulationEngine engine;
    engine.init(config, base_config_dir());

    const auto* coord = engine.outer_sd_coordinator_for_testing();
    REQUIRE(coord != nullptr);
    // OC-5: post-initialize global frontier starts at 0. Both endpoints are
    // monotonic non-decreasing and the coordinator is freshly built, so the
    // zeros are the expected ground state.
    REQUIRE(coord->global_frontier_min() == 0);
    REQUIRE(coord->global_frontier_max() == 0);

    // No scheduler in legacy mode — coord should not be attached anywhere.
    REQUIRE(engine.td_scheduler_for_testing() == nullptr);

    engine.finalize();
  }

  SECTION("td_mode=true — coord attached to inner scheduler") {
    config.scheduler.td_mode = true;
    config.scheduler.pipeline_depth_cap = 2U;  // K=2 → k_max threaded through
                                               // to ConcreteOuterSdCoordinator.

    tdmd::SimulationEngine engine;
    engine.init(config, base_config_dir());

    const auto* coord = engine.outer_sd_coordinator_for_testing();
    REQUIRE(coord != nullptr);
    REQUIRE(engine.td_scheduler_for_testing() != nullptr);

    // Live Pattern 2 + TD mode: schedule loop should still step without
    // tripping on the attached coord (every zone is internal w.r.t. the
    // 1-subdomain-equivalent boundary set the wire produces — future
    // T7.14 smoke exercises real cross-subdomain halo traffic).
    const auto final_row = engine.run(2, nullptr);
    REQUIRE(final_row.step == 2U);

    engine.finalize();
  }
}

TEST_CASE("Preflight rejects zoning.subdomains with a zero axis",
          "[runtime][preflight][pattern2][t79]") {
  auto config = load_base_config();
  config.zoning.subdomains = {2U, 0U, 1U};

  const auto errors = tdmd::io::preflight(config);
  REQUIRE_FALSE(tdmd::io::preflight_passes(errors));
  REQUIRE(any_error_at(errors, "zoning.subdomains"));
}

TEST_CASE("Preflight rejects comm.backend=hybrid on a single-subdomain run",
          "[runtime][preflight][pattern2][t79]") {
  auto config = load_base_config();
  config.zoning.subdomains = {1U, 1U, 1U};  // Pattern 1 — hybrid is wrong here.
  config.comm.backend = tdmd::io::CommBackendKind::Hybrid;

  const auto errors = tdmd::io::preflight(config);
  REQUIRE_FALSE(tdmd::io::preflight_passes(errors));
  REQUIRE(any_error_at(errors, "comm.backend"));
}
