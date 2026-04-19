// Exec pack: docs/development/m5_execution_pack.md T5.8
// SPEC: scheduler/SPEC.md §10 (engine integration)
// Master spec: §14 M5 acceptance gate D-M5-12
//
// D-M5-12 byte-exact regression: K=1 P=1 td_mode with the T5.8 CommBackend
// hook present (but null, single-rank) must produce the same thermo stream
// as the legacy `td_mode=false` path. This complements test_td_mode_smoke's
// D-M4-9 check by asserting that adding the comm-backend pointer to the
// engine does NOT perturb the reduction path when it's inactive.
//
// No MPI linkage here — the TDMD_HAVE_MPI guard in run_command / engine
// short-circuits cleanly when MPI is not initialised (or not compiled in).
// This test runs in both MPI-on and MPI-off builds.

#include "tdmd/io/yaml_config.hpp"
#include "tdmd/runtime/simulation_engine.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <sstream>
#include <string>

#ifndef TDMD_IO_FIXTURES_DIR
#error "TDMD_IO_FIXTURES_DIR must be defined by the build system"
#endif

namespace {

tdmd::io::YamlConfig load_shrunken_morse(bool td_mode) {
  const std::string path = std::string(TDMD_IO_FIXTURES_DIR) + "/configs/valid_nve_al.yaml";
  auto config = tdmd::io::parse_yaml_config(path);
  config.potential.morse.r0 = 2.0;
  config.potential.morse.cutoff = 2.5;
  config.potential.morse.alpha = 2.0;
  config.neighbor.skin = 0.1;
  config.scheduler.td_mode = td_mode;
  config.scheduler.pipeline_depth_cap = 1;
  return config;
}

std::string fixture_dir() {
  return std::string(TDMD_IO_FIXTURES_DIR) + "/configs";
}

std::string run_and_capture(bool td_mode, std::uint64_t n_steps) {
  auto cfg = load_shrunken_morse(td_mode);
  tdmd::SimulationEngine engine;
  // Backend pointer stays null — T5.8's injection point is never exercised.
  // The engine should therefore behave byte-for-byte like the pre-T5.8 code.
  engine.init(cfg, fixture_dir());
  std::ostringstream os;
  (void) engine.run(n_steps, &os);
  engine.finalize();
  return os.str();
}

}  // namespace

TEST_CASE("D-M5-12 — K=1 P=1 td_mode thermo byte-exact to legacy NVE on Al Morse fixture",
          "[runtime][engine][td_mode][d-m5-12]") {
  constexpr std::uint64_t kSteps = 10;
  const std::string legacy = run_and_capture(/*td_mode=*/false, kSteps);
  const std::string td = run_and_capture(/*td_mode=*/true, kSteps);

  REQUIRE_FALSE(legacy.empty());
  REQUIRE_FALSE(td.empty());
  REQUIRE(legacy == td);
}

TEST_CASE("D-M5-12 — K=1 td_mode reproducibility across back-to-back runs",
          "[runtime][engine][td_mode][d-m5-12][determinism]") {
  constexpr std::uint64_t kSteps = 5;
  const std::string a = run_and_capture(/*td_mode=*/true, kSteps);
  const std::string b = run_and_capture(/*td_mode=*/true, kSteps);
  REQUIRE(a == b);
}
