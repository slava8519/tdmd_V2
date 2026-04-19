// T1.4 — `tdmd.yaml` parser tests (structural / schema layer).
//
// Preflight (semantic layer) has its own test file — keeping them split lets
// parse-stage errors be exercised against an isolated YamlConfig without
// filesystem side effects, and keeps a clean boundary between "malformed
// config" and "config asks for something that cannot be done".

#include "tdmd/io/yaml_config.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#ifndef TDMD_TEST_FIXTURES_DIR
#error "TDMD_TEST_FIXTURES_DIR must be defined by the build system"
#endif

namespace {

std::string fixture_path(const std::string& rel) {
  return std::string(TDMD_TEST_FIXTURES_DIR) + "/configs/" + rel;
}

// Tiny helper so tests read as `load("simulation:\n  units: metal\n...")`.
// Matches the shape of the `read_lammps_data` tests.
tdmd::io::YamlConfig parse_string(std::string_view yaml) {
  return tdmd::io::parse_yaml_config_string(yaml, "<inline>");
}

}  // namespace

TEST_CASE("parse_yaml_config: valid fixture populates every field", "[io][yaml]") {
  const auto cfg = tdmd::io::parse_yaml_config(fixture_path("valid_nve_al.yaml"));

  CHECK(cfg.simulation.units == tdmd::io::UnitsKind::Metal);
  CHECK(cfg.simulation.seed == 42U);

  CHECK(cfg.atoms.source == tdmd::io::AtomsSource::LammpsData);
  CHECK(cfg.atoms.path == "../al_fcc_small.data");

  CHECK(cfg.potential.style == tdmd::io::PotentialStyle::Morse);
  CHECK(cfg.potential.morse.D == 0.2703);
  CHECK(cfg.potential.morse.alpha == 1.1646);
  CHECK(cfg.potential.morse.r0 == 3.253);
  CHECK(cfg.potential.morse.cutoff == 8.0);
  CHECK(cfg.potential.morse.cutoff_strategy == tdmd::io::MorseCutoffStrategy::ShiftedForce);

  CHECK(cfg.integrator.style == tdmd::io::IntegratorStyle::VelocityVerlet);
  CHECK(cfg.integrator.dt == 0.001);

  CHECK(cfg.neighbor.skin == 0.3);
  CHECK(cfg.run.n_steps == 100U);
}

TEST_CASE("parse_yaml_config: defaults fill in for omitted optional fields", "[io][yaml]") {
  // No `simulation.seed`, no `neighbor`, no `thermo` block, no explicit
  // `cutoff_strategy` — defaults should be applied.
  constexpr std::string_view yaml = R"(
simulation:
  units: metal
atoms:
  source: lammps_data
  path: ./some.data
potential:
  style: morse
  params:
    D: 0.1
    alpha: 1.0
    r0: 3.0
    cutoff: 7.0
integrator:
  style: velocity_verlet
  dt: 0.001
run:
  n_steps: 1
)";
  const auto cfg = parse_string(yaml);
  CHECK(cfg.simulation.seed == 12345U);                // default
  CHECK(cfg.neighbor.skin == 0.3);                     // default
  CHECK(cfg.thermo.every == 100U);                     // default
  CHECK(cfg.potential.morse.cutoff_strategy ==         //
        tdmd::io::MorseCutoffStrategy::ShiftedForce);  // default
}

TEST_CASE("parse_yaml_config: missing required top-level block raises", "[io][yaml]") {
  constexpr std::string_view yaml = R"(
simulation:
  units: metal
# no atoms / potential / integrator / run
)";
  REQUIRE_THROWS_AS(parse_string(yaml), tdmd::io::YamlParseError);
  try {
    parse_string(yaml);
  } catch (const tdmd::io::YamlParseError& e) {
    // First missing required block is `atoms`; message should name it.
    CHECK_THAT(std::string(e.what()), Catch::Matchers::ContainsSubstring("atoms"));
  }
}

TEST_CASE("parse_yaml_config: missing required leaf raises with key path", "[io][yaml]") {
  const auto assert_missing = [](std::string_view yaml, std::string_view expected_path) {
    try {
      parse_string(yaml);
      FAIL("expected YamlParseError");
    } catch (const tdmd::io::YamlParseError& e) {
      CHECK(e.key_path() == std::string(expected_path));
    }
  };

  // simulation.units missing.
  assert_missing(R"(
simulation:
  seed: 1
atoms: { source: lammps_data, path: ./a.data }
potential: { style: morse, params: { D: 0.1, alpha: 1.0, r0: 3.0, cutoff: 7.0 } }
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1 }
)",
                 "simulation.units");

  // atoms.path missing.
  assert_missing(R"(
simulation: { units: metal }
atoms: { source: lammps_data }
potential: { style: morse, params: { D: 0.1, alpha: 1.0, r0: 3.0, cutoff: 7.0 } }
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1 }
)",
                 "atoms.path");

  // potential.params.alpha missing.
  assert_missing(R"(
simulation: { units: metal }
atoms: { source: lammps_data, path: ./a.data }
potential: { style: morse, params: { D: 0.1, r0: 3.0, cutoff: 7.0 } }
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1 }
)",
                 "potential.params.alpha");

  // integrator.dt missing.
  assert_missing(R"(
simulation: { units: metal }
atoms: { source: lammps_data, path: ./a.data }
potential: { style: morse, params: { D: 0.1, alpha: 1.0, r0: 3.0, cutoff: 7.0 } }
integrator: { style: velocity_verlet }
run: { n_steps: 1 }
)",
                 "integrator.dt");

  // run.n_steps missing.
  assert_missing(R"(
simulation: { units: metal }
atoms: { source: lammps_data, path: ./a.data }
potential: { style: morse, params: { D: 0.1, alpha: 1.0, r0: 3.0, cutoff: 7.0 } }
integrator: { style: velocity_verlet, dt: 0.001 }
run: {}
)",
                 "run.n_steps");
}

TEST_CASE("parse_yaml_config: units=lj parses (schema layer is permissive)", "[io][yaml]") {
  // Schema-layer test: lj is now a valid literal (M2). Cross-field rules
  // (lj requires reference, etc.) live in preflight, not here — so a config
  // with `units: lj` and no reference parses cleanly and fails preflight.
  constexpr std::string_view yaml_lj = R"(
simulation: { units: lj }
atoms: { source: lammps_data, path: ./a.data }
potential: { style: morse, params: { D: 0.1, alpha: 1.0, r0: 3.0, cutoff: 7.0 } }
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1 }
)";
  const auto cfg = parse_string(yaml_lj);
  CHECK(cfg.simulation.units == tdmd::io::UnitsKind::Lj);
  CHECK_FALSE(cfg.simulation.reference.has_value());
}

TEST_CASE("parse_yaml_config: simulation.reference block populates LjReference", "[io][yaml]") {
  constexpr std::string_view yaml = R"(
simulation:
  units: lj
  reference:
    sigma: 3.405
    epsilon: 0.0104
    mass: 39.948
atoms: { source: lammps_data, path: ./a.data }
potential: { style: morse, params: { D: 0.1, alpha: 1.0, r0: 3.0, cutoff: 7.0 } }
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1 }
)";
  const auto cfg = parse_string(yaml);
  REQUIRE(cfg.simulation.reference.has_value());
  CHECK(cfg.simulation.reference->sigma == 3.405);
  CHECK(cfg.simulation.reference->epsilon == 0.0104);
  CHECK(cfg.simulation.reference->mass == 39.948);
}

TEST_CASE("parse_yaml_config: simulation.reference missing sub-keys reject", "[io][yaml]") {
  // Every field in `reference` is required at the schema layer (LjReference
  // has no partial form). Dropping any one must raise YamlParseError.
  const auto make_yaml = [](std::string_view ref_body) {
    return std::string(R"(
simulation:
  units: lj
  reference:
)") + std::string(ref_body) +
           R"(
atoms: { source: lammps_data, path: ./a.data }
potential: { style: morse, params: { D: 0.1, alpha: 1.0, r0: 3.0, cutoff: 7.0 } }
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1 }
)";
  };
  REQUIRE_THROWS_AS(parse_string(make_yaml("    epsilon: 0.0104\n    mass: 39.948")),
                    tdmd::io::YamlParseError);
  REQUIRE_THROWS_AS(parse_string(make_yaml("    sigma: 3.405\n    mass: 39.948")),
                    tdmd::io::YamlParseError);
  REQUIRE_THROWS_AS(parse_string(make_yaml("    sigma: 3.405\n    epsilon: 0.0104")),
                    tdmd::io::YamlParseError);
}

TEST_CASE("parse_yaml_config: unsupported units literals reject", "[io][yaml]") {
  // Garbage literal — not metal, not lj.
  constexpr std::string_view yaml_garbage = R"(
simulation: { units: furlongs }
atoms: { source: lammps_data, path: ./a.data }
potential: { style: morse, params: { D: 0.1, alpha: 1.0, r0: 3.0, cutoff: 7.0 } }
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1 }
)";
  REQUIRE_THROWS_AS(parse_string(yaml_garbage), tdmd::io::YamlParseError);
}

TEST_CASE("parse_yaml_config: unsupported atoms.source rejects", "[io][yaml]") {
  constexpr std::string_view yaml = R"(
simulation: { units: metal }
atoms: { source: inline, path: ./a.data }
potential: { style: morse, params: { D: 0.1, alpha: 1.0, r0: 3.0, cutoff: 7.0 } }
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1 }
)";
  REQUIRE_THROWS_AS(parse_string(yaml), tdmd::io::YamlParseError);
}

TEST_CASE("parse_yaml_config: unsupported potential.style rejects", "[io][yaml]") {
  // `eam/alloy` landed in T2.9; a still-unsupported style (SNAP) stands in for
  // the "future style" regression we want this test to catch.
  constexpr std::string_view yaml = R"(
simulation: { units: metal }
atoms: { source: lammps_data, path: ./a.data }
potential: { style: snap, params: { file: ./Al.snap } }
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1 }
)";
  REQUIRE_THROWS_AS(parse_string(yaml), tdmd::io::YamlParseError);
}

TEST_CASE("parse_yaml_config: eam/alloy style accepted with file param", "[io][yaml]") {
  constexpr std::string_view yaml = R"(
simulation: { units: metal }
atoms: { source: lammps_data, path: ./a.data }
potential: { style: eam/alloy, params: { file: ./Ni-Al.eam.alloy } }
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1 }
)";
  const auto cfg = parse_string(yaml);
  REQUIRE(cfg.potential.style == tdmd::io::PotentialStyle::EamAlloy);
  REQUIRE(cfg.potential.eam_alloy.file == "./Ni-Al.eam.alloy");
}

TEST_CASE("parse_yaml_config: eam/alloy rejects empty file string", "[io][yaml]") {
  constexpr std::string_view yaml = R"(
simulation: { units: metal }
atoms: { source: lammps_data, path: ./a.data }
potential: { style: eam/alloy, params: { file: "" } }
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1 }
)";
  REQUIRE_THROWS_AS(parse_string(yaml), tdmd::io::YamlParseError);
}

TEST_CASE("parse_yaml_config: eam/alloy rejects missing file param", "[io][yaml]") {
  constexpr std::string_view yaml = R"(
simulation: { units: metal }
atoms: { source: lammps_data, path: ./a.data }
potential: { style: eam/alloy, params: {} }
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1 }
)";
  REQUIRE_THROWS_AS(parse_string(yaml), tdmd::io::YamlParseError);
}

TEST_CASE("parse_yaml_config: cutoff_strategy accepts both M1 literals", "[io][yaml]") {
  const auto make_yaml = [](std::string_view strategy) {
    return std::string(R"(
simulation: { units: metal }
atoms: { source: lammps_data, path: ./a.data }
potential:
  style: morse
  params:
    D: 0.1
    alpha: 1.0
    r0: 3.0
    cutoff: 7.0
    cutoff_strategy: )") +
           std::string(strategy) + R"(
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1 }
)";
  };

  const auto sf = parse_string(make_yaml("shifted_force"));
  CHECK(sf.potential.morse.cutoff_strategy == tdmd::io::MorseCutoffStrategy::ShiftedForce);

  const auto hc = parse_string(make_yaml("hard_cutoff"));
  CHECK(hc.potential.morse.cutoff_strategy == tdmd::io::MorseCutoffStrategy::HardCutoff);

  REQUIRE_THROWS_AS(parse_string(make_yaml("smoothed")), tdmd::io::YamlParseError);
}

TEST_CASE("parse_yaml_config: unknown keys at recognised blocks reject (typo guard)",
          "[io][yaml]") {
  // `simluation` — typo of `simulation` at top level.
  constexpr std::string_view yaml_top = R"(
simluation: { units: metal }
simulation: { units: metal }
atoms: { source: lammps_data, path: ./a.data }
potential: { style: morse, params: { D: 0.1, alpha: 1.0, r0: 3.0, cutoff: 7.0 } }
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1 }
)";
  REQUIRE_THROWS_AS(parse_string(yaml_top), tdmd::io::YamlParseError);

  // `n_stepss` inside `run`.
  constexpr std::string_view yaml_leaf = R"(
simulation: { units: metal }
atoms: { source: lammps_data, path: ./a.data }
potential: { style: morse, params: { D: 0.1, alpha: 1.0, r0: 3.0, cutoff: 7.0 } }
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1, n_stepss: 2 }
)";
  REQUIRE_THROWS_AS(parse_string(yaml_leaf), tdmd::io::YamlParseError);
}

TEST_CASE("parse_yaml_config: malformed YAML reports line number", "[io][yaml]") {
  // Deliberate tab-indent error + unterminated flow.
  constexpr std::string_view yaml = R"(
simulation: { units: metal
atoms: broken
)";
  try {
    parse_string(yaml);
    FAIL("expected YamlParseError on malformed YAML");
  } catch (const tdmd::io::YamlParseError& e) {
    CHECK(e.line() > 0);
  }
}

TEST_CASE("parse_yaml_config: empty document rejects", "[io][yaml]") {
  REQUIRE_THROWS_AS(parse_string(""), tdmd::io::YamlParseError);
  REQUIRE_THROWS_AS(parse_string("\n# just a comment\n"), tdmd::io::YamlParseError);
}

TEST_CASE("parse_yaml_config: type mismatch on numeric field names the field", "[io][yaml]") {
  constexpr std::string_view yaml = R"(
simulation: { units: metal }
atoms: { source: lammps_data, path: ./a.data }
potential: { style: morse, params: { D: "hello", alpha: 1.0, r0: 3.0, cutoff: 7.0 } }
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1 }
)";
  try {
    parse_string(yaml);
    FAIL("expected YamlParseError on non-numeric D");
  } catch (const tdmd::io::YamlParseError& e) {
    CHECK(e.key_path() == "potential.params.D");
  }
}

TEST_CASE("parse_yaml_config: file not found throws runtime_error, not YamlParseError",
          "[io][yaml]") {
  // File-open failure is distinct from malformed YAML — caller shouldn't
  // confuse "config missing" with "config malformed".
  REQUIRE_THROWS_AS(tdmd::io::parse_yaml_config("/tmp/tdmd_this_file_does_not_exist.yaml"),
                    std::runtime_error);
}

TEST_CASE("parse_yaml_config: required top-level block missing names it", "[io][yaml]") {
  // Each of these drops one required block — verify the parser names the
  // first one in canonical order (simulation → atoms → potential → integrator → run).
  const auto assert_rejected = [](std::string_view yaml, std::string_view expected_block) {
    try {
      parse_string(yaml);
      FAIL("expected YamlParseError");
    } catch (const tdmd::io::YamlParseError& e) {
      CHECK_THAT(std::string(e.what()),
                 Catch::Matchers::ContainsSubstring(std::string(expected_block)));
    }
  };

  assert_rejected(R"(
atoms: { source: lammps_data, path: ./a.data }
potential: { style: morse, params: { D: 0.1, alpha: 1.0, r0: 3.0, cutoff: 7.0 } }
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1 }
)",
                  "simulation");

  assert_rejected(R"(
simulation: { units: metal }
potential: { style: morse, params: { D: 0.1, alpha: 1.0, r0: 3.0, cutoff: 7.0 } }
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1 }
)",
                  "atoms");

  assert_rejected(R"(
simulation: { units: metal }
atoms: { source: lammps_data, path: ./a.data }
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1 }
)",
                  "potential");

  assert_rejected(R"(
simulation: { units: metal }
atoms: { source: lammps_data, path: ./a.data }
potential: { style: morse, params: { D: 0.1, alpha: 1.0, r0: 3.0, cutoff: 7.0 } }
run: { n_steps: 1 }
)",
                  "integrator");

  assert_rejected(R"(
simulation: { units: metal }
atoms: { source: lammps_data, path: ./a.data }
potential: { style: morse, params: { D: 0.1, alpha: 1.0, r0: 3.0, cutoff: 7.0 } }
integrator: { style: velocity_verlet, dt: 0.001 }
)",
                  "run");
}

TEST_CASE("zoning block: defaults to Auto when omitted", "[io][yaml][zoning]") {
  // T5.9 — no zoning: key means M3 auto-select (Hilbert/Decomp2D/Linear1D
  // per §3.4 aspect-ratio tree). This preserves M3/M4 regressions.
  constexpr std::string_view yaml = R"(
simulation: { units: metal }
atoms: { source: lammps_data, path: ./a.data }
potential: { style: morse, params: { D: 0.1, alpha: 1.0, r0: 3.0, cutoff: 7.0 } }
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1 }
)";
  const auto cfg = parse_string(yaml);
  CHECK(cfg.zoning.scheme == tdmd::io::ZoningSchemeKind::Auto);
}

TEST_CASE("zoning.scheme parses the three allowed literals", "[io][yaml][zoning]") {
  constexpr std::string_view head = R"(
simulation: { units: metal }
atoms: { source: lammps_data, path: ./a.data }
potential: { style: morse, params: { D: 0.1, alpha: 1.0, r0: 3.0, cutoff: 7.0 } }
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1 }
zoning: { scheme: )";

  {
    const auto cfg = parse_string(std::string(head) + "auto }");
    CHECK(cfg.zoning.scheme == tdmd::io::ZoningSchemeKind::Auto);
  }
  {
    const auto cfg = parse_string(std::string(head) + "hilbert }");
    CHECK(cfg.zoning.scheme == tdmd::io::ZoningSchemeKind::Hilbert);
  }
  {
    const auto cfg = parse_string(std::string(head) + "linear_1d }");
    CHECK(cfg.zoning.scheme == tdmd::io::ZoningSchemeKind::Linear1D);
  }
}

TEST_CASE("zoning.scheme rejects unknown literals and unknown keys", "[io][yaml][zoning]") {
  constexpr std::string_view base = R"(
simulation: { units: metal }
atoms: { source: lammps_data, path: ./a.data }
potential: { style: morse, params: { D: 0.1, alpha: 1.0, r0: 3.0, cutoff: 7.0 } }
integrator: { style: velocity_verlet, dt: 0.001 }
run: { n_steps: 1 }
)";

  REQUIRE_THROWS_AS(parse_string(std::string(base) + "zoning: { scheme: auto_magic }"),
                    tdmd::io::YamlParseError);
  REQUIRE_THROWS_AS(parse_string(std::string(base) + "zoning: { schema: linear_1d }"),
                    tdmd::io::YamlParseError);
}

TEST_CASE("missing_units fixture reproduces on disk", "[io][yaml]") {
  // Sanity: fixture + parser agree on what "required units" means.
  REQUIRE_THROWS_AS(tdmd::io::parse_yaml_config(fixture_path("missing_units.yaml")),
                    tdmd::io::YamlParseError);
}

TEST_CASE("missing_atoms_source fixture reproduces on disk", "[io][yaml]") {
  REQUIRE_THROWS_AS(tdmd::io::parse_yaml_config(fixture_path("missing_atoms_source.yaml")),
                    tdmd::io::YamlParseError);
}
