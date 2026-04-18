// T1.9 — SimulationEngine lifecycle + integration tests.
//
// Covers three layers:
//   (1) FSM — init → run → finalize order is strict; double-init, out-of-order
//       run / finalize all throw SimulationEngineStateError.
//   (2) Integration — a 10-step NVE run on a tiny Al FCC config completes,
//       emits the expected thermo format, and respects thermo.every.
//   (3) Determinism — same config → byte-identical thermo stream on reruns.
//
// We reuse the hermetic fixtures shipped for T1.3 / T1.4 (32-atom Al FCC data
// + 100-step tdmd.yaml) so this test does not depend on the `examples/` dir
// which is a user-facing surface that can drift.

#include "tdmd/io/yaml_config.hpp"
#include "tdmd/runtime/simulation_engine.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <string>
#include <string_view>

#ifndef TDMD_IO_FIXTURES_DIR
#error "TDMD_IO_FIXTURES_DIR must be defined by the build system"
#endif

namespace {

// The T1.4 valid_nve_al fixture points at `../al_fcc_small.data` (32 atoms,
// 8.1 A cubic box) relative to the `configs/` subdirectory. We pass the
// `configs/` path as `config_dir` so the relative atoms.path resolves
// correctly regardless of the test CWD.
//
// The fixture's physical Morse cutoff (8.0 A) would require a 24.9 A box to
// satisfy the 3-cells-per-axis stencil, so for these lifecycle tests we
// shrink r0/cutoff/skin to fit the 8.1 A fixture. These are toy values — the
// real Girifalco-Weizer parameters are exercised by the T1.11 differential
// harness on a larger cell.
tdmd::io::YamlConfig load_valid_config() {
  const std::string path = std::string(TDMD_IO_FIXTURES_DIR) + "/configs/valid_nve_al.yaml";
  auto config = tdmd::io::parse_yaml_config(path);
  config.potential.morse.r0 = 2.0;
  config.potential.morse.cutoff = 2.5;
  config.potential.morse.alpha = 2.0;
  config.neighbor.skin = 0.1;
  return config;
}

std::string valid_config_dir() {
  return std::string(TDMD_IO_FIXTURES_DIR) + "/configs";
}

// Strip the leading two status lines and the trailing status line the CLI
// emits around the thermo stream. The engine itself does not produce these —
// only run_command does — so for direct-engine tests the stream is just
// header + rows.
std::size_t count_non_comment_lines(const std::string& s) {
  std::size_t n = 0;
  std::size_t pos = 0;
  while (pos < s.size()) {
    auto end = s.find('\n', pos);
    std::string_view line(s.data() + pos, (end == std::string::npos ? s.size() : end) - pos);
    if (!line.empty() && line.front() != '#') {
      ++n;
    }
    if (end == std::string::npos) {
      break;
    }
    pos = end + 1;
  }
  return n;
}

}  // namespace

TEST_CASE("SimulationEngine: lifecycle is strict", "[runtime][engine][lifecycle]") {
  const auto config = load_valid_config();

  SECTION("init → run → finalize accepts valid sequence") {
    tdmd::SimulationEngine engine;
    REQUIRE_FALSE(engine.is_initialised());
    engine.init(config, valid_config_dir());
    REQUIRE(engine.is_initialised());
    std::ostringstream thermo;
    (void) engine.run(5, &thermo);
    engine.finalize();
    // Engine leaves "Initialised" only on transition to Finalised — any
    // subsequent run() must throw.
    REQUIRE_THROWS_AS(engine.run(1, nullptr), tdmd::SimulationEngineStateError);
  }

  SECTION("double init rejected") {
    tdmd::SimulationEngine engine;
    engine.init(config, valid_config_dir());
    REQUIRE_THROWS_AS(engine.init(config, valid_config_dir()), tdmd::SimulationEngineStateError);
  }

  SECTION("run before init rejected") {
    tdmd::SimulationEngine engine;
    REQUIRE_THROWS_AS(engine.run(1, nullptr), tdmd::SimulationEngineStateError);
  }

  SECTION("finalize before init rejected") {
    tdmd::SimulationEngine engine;
    REQUIRE_THROWS_AS(engine.finalize(), tdmd::SimulationEngineStateError);
  }

  SECTION("run after finalize rejected") {
    tdmd::SimulationEngine engine;
    engine.init(config, valid_config_dir());
    engine.finalize();
    REQUIRE_THROWS_AS(engine.run(1, nullptr), tdmd::SimulationEngineStateError);
  }
}

TEST_CASE("SimulationEngine: thermo stream layout", "[runtime][engine][thermo]") {
  auto config = load_valid_config();
  // Trim the step count — the happy-path test runs fast.
  config.run.n_steps = 10;
  config.thermo.every = 5;

  tdmd::SimulationEngine engine;
  engine.init(config, valid_config_dir());
  std::ostringstream thermo;
  const auto final_row = engine.run(config.run.n_steps, &thermo);
  engine.finalize();

  const std::string out = thermo.str();
  // Header row present.
  REQUIRE_THAT(out, Catch::Matchers::StartsWith("# step temp pe ke etotal press"));

  // step 0 + step 5 + step 10 = 3 data rows, plus a header line.
  const auto data_rows = count_non_comment_lines(out);
  REQUIRE(data_rows == 3);

  // Final row step matches `run`'s N.
  REQUIRE(final_row.step == 10U);
}

TEST_CASE("SimulationEngine: final step row is always emitted", "[runtime][engine][thermo]") {
  auto config = load_valid_config();
  // 7 steps, thermo every 4 → expect rows at 0, 4, 7 (7 is the forced final).
  config.run.n_steps = 7;
  config.thermo.every = 4;

  tdmd::SimulationEngine engine;
  engine.init(config, valid_config_dir());
  std::ostringstream thermo;
  (void) engine.run(config.run.n_steps, &thermo);
  engine.finalize();

  // step 0 (initial), step 4 (modulo hit), step 7 (forced final) = 3 rows.
  REQUIRE(count_non_comment_lines(thermo.str()) == 3);
}

TEST_CASE("SimulationEngine: thermo stream is bit-exact deterministic",
          "[runtime][engine][determinism]") {
  auto config = load_valid_config();
  config.run.n_steps = 20;
  config.thermo.every = 5;

  auto run_once = [&](std::string& out) {
    tdmd::SimulationEngine engine;
    engine.init(config, valid_config_dir());
    std::ostringstream thermo;
    (void) engine.run(config.run.n_steps, &thermo);
    engine.finalize();
    out = thermo.str();
  };

  std::string first;
  std::string second;
  run_once(first);
  run_once(second);

  // Bit-exact — the entire stream, not just a checksum. Makes this test fail
  // loudly if any nondeterministic source (unordered containers, threading,
  // uninitialised memory) sneaks into the hot path.
  REQUIRE(first == second);
  REQUIRE(first.find("# step temp pe ke etotal press") == 0);
}

TEST_CASE("SimulationEngine: accessors reflect loaded config", "[runtime][engine][accessors]") {
  const auto config = load_valid_config();
  tdmd::SimulationEngine engine;
  engine.init(config, valid_config_dir());

  // T1.3 fixture is a 32-atom Al FCC cube.
  REQUIRE(engine.atoms().size() == 32U);
  REQUIRE(engine.thermo_every() == config.thermo.every);
  REQUIRE(engine.current_step() == 0U);

  (void) engine.run(3, nullptr);
  REQUIRE(engine.current_step() == 3U);
  engine.finalize();
}
