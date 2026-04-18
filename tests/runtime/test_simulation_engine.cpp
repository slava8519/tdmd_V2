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
#include <unordered_map>

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

// ---------------------------------------------------------------------------
// T2.2 — lj ingest integration: identity-reference round-trip.
//
// With LjReference{σ=1, ε=1, m=1}, every lj→metal scaling factor for LENGTH,
// ENERGY, MASS collapses to 1.0 exactly. Since the fixture has no velocities
// (vx/vy/vz all zero) and identity velocity conversion multiplies zero by the
// factor, velocities also come out bitwise zero. The post-ingest state
// (atoms, box, species masses) must therefore be bitwise identical to the
// metal path. This locks in the invariant that the lj branch doesn't drift
// numerically when the reference is the physical identity.
// ---------------------------------------------------------------------------

TEST_CASE("SimulationEngine: metal ↔ lj identity-reference round-trip (positions/box/mass)",
          "[runtime][engine][lj]") {
  // Mirror the metal config as lj with identity (σ=ε=m=1). For LENGTH, ENERGY,
  // and MASS the conversion collapses to 1.0 exactly, so positions, box
  // bounds, and species masses are bitwise identical. Velocities and time are
  // NOT bitwise identical — even with σ=ε=m=1 the velocity factor carries a
  // 1/sqrt(mvv2e) ≈ 98.23 scaling (LAMMPS `metal` mvv2e is a pure numerical
  // constant, not a dimensional σ/ε/m combination). So the bitwise claim only
  // holds for the pure-scalar dimensions — this is the test's scope.
  auto metal_cfg = load_valid_config();
  auto lj_cfg = metal_cfg;
  lj_cfg.simulation.units = tdmd::io::UnitsKind::Lj;
  lj_cfg.simulation.reference = tdmd::LjReference{.sigma = 1.0, .epsilon = 1.0, .mass = 1.0};

  tdmd::SimulationEngine metal_engine;
  metal_engine.init(metal_cfg, valid_config_dir());
  tdmd::SimulationEngine lj_engine;
  lj_engine.init(lj_cfg, valid_config_dir());

  REQUIRE(metal_engine.atoms().size() == lj_engine.atoms().size());
  const auto& ma = metal_engine.atoms();
  const auto& la = lj_engine.atoms();
  for (std::size_t i = 0; i < ma.size(); ++i) {
    REQUIRE(ma.x[i] == la.x[i]);
    REQUIRE(ma.y[i] == la.y[i]);
    REQUIRE(ma.z[i] == la.z[i]);
    REQUIRE(ma.type[i] == la.type[i]);
  }

  const auto& mb = metal_engine.box();
  const auto& lb = lj_engine.box();
  REQUIRE(mb.xlo == lb.xlo);
  REQUIRE(mb.xhi == lb.xhi);
  REQUIRE(mb.ylo == lb.ylo);
  REQUIRE(mb.yhi == lb.yhi);
  REQUIRE(mb.zlo == lb.zlo);
  REQUIRE(mb.zhi == lb.zhi);

  REQUIRE(metal_engine.species().count() == lj_engine.species().count());
  for (std::size_t t = 0; t < metal_engine.species().count(); ++t) {
    const auto id = static_cast<tdmd::SpeciesId>(t);
    REQUIRE(metal_engine.species().get_info(id).mass == lj_engine.species().get_info(id).mass);
  }

  metal_engine.finalize();
  lj_engine.finalize();
}

TEST_CASE("SimulationEngine: lj non-identity reference scales positions by σ",
          "[runtime][engine][lj]") {
  // With σ=2.0, every lj position `p` in the .data file becomes `2p` in
  // internal metal representation. To keep both engines physically
  // constructable on the same underlying .data file we halve the lj-config
  // cutoff / r0 / skin so their post-conversion metal values match the metal
  // baseline (and the cell-grid 3-cells-per-axis constraint is satisfied on
  // both sides — the lj-engine metal box is σ · metal_box = 16.2 Å, which
  // comfortably fits 3·(2.5+0.1) = 7.8 Å). Because the lj-engine uses a
  // different (larger) box, its cell-grid stable-reorder permutation can
  // differ from the metal baseline's, so we look up atoms by `AtomId` — those
  // IDs are preserved through reorder and uniquely identify the same
  // physical atom.
  auto metal_cfg = load_valid_config();
  auto lj_cfg = metal_cfg;
  lj_cfg.simulation.units = tdmd::io::UnitsKind::Lj;
  lj_cfg.simulation.reference = tdmd::LjReference{.sigma = 2.0, .epsilon = 1.0, .mass = 1.0};
  lj_cfg.potential.morse.cutoff *= 0.5;  // converted metal cutoff == metal baseline cutoff
  lj_cfg.potential.morse.r0 *= 0.5;
  lj_cfg.neighbor.skin *= 0.5;

  tdmd::SimulationEngine metal_engine;
  metal_engine.init(metal_cfg, valid_config_dir());
  tdmd::SimulationEngine lj_engine;
  lj_engine.init(lj_cfg, valid_config_dir());

  REQUIRE(metal_engine.atoms().size() == lj_engine.atoms().size());
  const auto& ma = metal_engine.atoms();
  const auto& la = lj_engine.atoms();

  // Build id → index maps; the two engines may have been reordered
  // independently (different box size → different cell grid).
  std::unordered_map<tdmd::AtomId, std::size_t> m_idx;
  std::unordered_map<tdmd::AtomId, std::size_t> l_idx;
  m_idx.reserve(ma.size());
  l_idx.reserve(la.size());
  for (std::size_t i = 0; i < ma.size(); ++i) {
    m_idx.emplace(ma.id[i], i);
  }
  for (std::size_t i = 0; i < la.size(); ++i) {
    l_idx.emplace(la.id[i], i);
  }
  REQUIRE(m_idx.size() == ma.size());
  REQUIRE(l_idx.size() == la.size());

  for (const auto& [id, mi] : m_idx) {
    const auto it = l_idx.find(id);
    REQUIRE(it != l_idx.end());
    const std::size_t li = it->second;
    REQUIRE(la.x[li] == ma.x[mi] * 2.0);
    REQUIRE(la.y[li] == ma.y[mi] * 2.0);
    REQUIRE(la.z[li] == ma.z[mi] * 2.0);
  }

  const auto& mb = metal_engine.box();
  const auto& lb = lj_engine.box();
  REQUIRE(lb.xhi == mb.xhi * 2.0);
  REQUIRE(lb.yhi == mb.yhi * 2.0);
  REQUIRE(lb.zhi == mb.zhi * 2.0);

  metal_engine.finalize();
  lj_engine.finalize();
}
