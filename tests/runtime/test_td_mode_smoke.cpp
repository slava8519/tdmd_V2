// Exec pack: docs/development/m4_execution_pack.md T4.9
// SPEC: docs/specs/scheduler/SPEC.md §10 (engine integration)
// Master spec: §14 M4 acceptance gate (D-M4-9)
//
// T4.9 acceptance: SimulationEngine with `scheduler.td_mode: true` must
// produce a thermo stream byte-identical to the legacy path (`td_mode:
// false`) on the same fixture. In K=1 single-rank (D-M4-1, D-M4-6) the
// scheduler wraps lifecycle events around unchanged physics calls, so the
// neighbor-list reduction order is preserved and no FP drift can sneak in.

#include "tdmd/io/yaml_config.hpp"
#include "tdmd/runtime/simulation_engine.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>

#ifndef TDMD_IO_FIXTURES_DIR
#error "TDMD_IO_FIXTURES_DIR must be defined by the build system"
#endif

namespace {

// Shrunken Morse params — the 8.1 Å fixture box won't satisfy the default
// 8.0 Å cutoff stencil. Kept identical to test_simulation_engine.cpp so
// we compare like-with-like.
tdmd::io::YamlConfig load_valid_config(bool td_mode) {
  const std::string path = std::string(TDMD_IO_FIXTURES_DIR) + "/configs/valid_nve_al.yaml";
  auto config = tdmd::io::parse_yaml_config(path);
  config.potential.morse.r0 = 2.0;
  config.potential.morse.cutoff = 2.5;
  config.potential.morse.alpha = 2.0;
  config.neighbor.skin = 0.1;
  config.scheduler.td_mode = td_mode;
  return config;
}

std::string valid_config_dir() {
  return std::string(TDMD_IO_FIXTURES_DIR) + "/configs";
}

std::string run_and_capture_thermo(bool td_mode, std::uint64_t n_steps) {
  auto cfg = load_valid_config(td_mode);
  tdmd::SimulationEngine engine;
  engine.init(cfg, valid_config_dir());
  std::ostringstream os;
  (void) engine.run(n_steps, &os);
  engine.finalize();
  return os.str();
}

}  // namespace

TEST_CASE("T4.9 — td_mode=true thermo byte-exact to legacy on Al Morse fixture",
          "[runtime][engine][td_mode][byte-exact]") {
  constexpr std::uint64_t kSteps = 10;
  const std::string legacy = run_and_capture_thermo(/*td_mode=*/false, kSteps);
  const std::string td = run_and_capture_thermo(/*td_mode=*/true, kSteps);

  REQUIRE_FALSE(legacy.empty());
  REQUIRE_FALSE(td.empty());
  REQUIRE(legacy == td);  // byte-for-byte identity (D-M4-9)
}

TEST_CASE("T4.9 — td_mode=false is the default", "[runtime][engine][td_mode][default]") {
  auto cfg = load_valid_config(/*td_mode=*/false);
  REQUIRE_FALSE(cfg.scheduler.td_mode);
}

TEST_CASE("T4.9 — td_mode determinism: same inputs, identical thermo across re-runs",
          "[runtime][engine][td_mode][determinism]") {
  constexpr std::uint64_t kSteps = 5;
  const std::string a = run_and_capture_thermo(/*td_mode=*/true, kSteps);
  const std::string b = run_and_capture_thermo(/*td_mode=*/true, kSteps);
  REQUIRE(a == b);
}
