// T1.4 — preflight (semantic) tests.
//
// Preflight accumulates; it never fail-fasts. These tests verify that
// (a) a valid YamlConfig + existing on-disk file returns an empty vector,
// (b) each semantic rule produces exactly the expected key_path when violated,
// (c) multi-error accumulation preserves order (determinism requirement
//     from exec pack T1.4: "same config → same error list, same order").

#include "tdmd/io/preflight.hpp"
#include "tdmd/io/yaml_config.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <filesystem>
#include <string>

#ifndef TDMD_TEST_FIXTURES_DIR
#error "TDMD_TEST_FIXTURES_DIR must be defined by the build system"
#endif

namespace {

// Valid M1 YamlConfig — baseline that preflight should accept. Each test
// tweaks one field to assert a specific rule without cross-contamination.
// `atoms.path` defaults to the 32-atom .data fixture from T1.3.
tdmd::io::YamlConfig valid_cfg() {
  tdmd::io::YamlConfig cfg{};
  cfg.simulation.units = tdmd::io::UnitsKind::Metal;
  cfg.simulation.seed = 42;
  cfg.atoms.source = tdmd::io::AtomsSource::LammpsData;
  cfg.atoms.path = std::string(TDMD_TEST_FIXTURES_DIR) + "/al_fcc_small.data";
  cfg.potential.style = tdmd::io::PotentialStyle::Morse;
  cfg.potential.morse = {.D = 0.2703,
                         .alpha = 1.1646,
                         .r0 = 3.253,
                         .cutoff = 8.0,
                         .cutoff_strategy = tdmd::io::MorseCutoffStrategy::ShiftedForce};
  cfg.integrator.style = tdmd::io::IntegratorStyle::VelocityVerlet;
  cfg.integrator.dt = 0.001;
  cfg.neighbor.skin = 0.3;
  cfg.thermo.every = 100;
  cfg.run.n_steps = 100;
  return cfg;
}

// True iff `errors` has exactly one entry with the given key_path.
bool has_single_error_at(const std::vector<tdmd::io::PreflightError>& errors,
                         const std::string& key_path) {
  if (errors.size() != 1) {
    return false;
  }
  return errors[0].severity == tdmd::io::PreflightSeverity::Error && errors[0].key_path == key_path;
}

}  // namespace

TEST_CASE("preflight: valid M1 config returns empty vector", "[io][preflight]") {
  const auto cfg = valid_cfg();
  const auto errs = tdmd::io::preflight(cfg);
  REQUIRE(errs.empty());
  CHECK(tdmd::io::preflight_passes(errs));
}

TEST_CASE("preflight: atoms.path missing file reports error", "[io][preflight]") {
  auto cfg = valid_cfg();
  cfg.atoms.path = "/tmp/tdmd_definitely_not_a_real_file_xyz.data";
  const auto errs = tdmd::io::preflight(cfg);
  CHECK(has_single_error_at(errs, "atoms.path"));
  CHECK_FALSE(tdmd::io::preflight_passes(errs));
}

TEST_CASE("preflight: empty atoms.path reports error", "[io][preflight]") {
  auto cfg = valid_cfg();
  cfg.atoms.path.clear();
  const auto errs = tdmd::io::preflight(cfg);
  CHECK(has_single_error_at(errs, "atoms.path"));
}

TEST_CASE("preflight: atoms.path that is a directory rejects", "[io][preflight]") {
  // fixtures dir exists and is a directory — perfect `not a regular file` case.
  auto cfg = valid_cfg();
  cfg.atoms.path = TDMD_TEST_FIXTURES_DIR;
  const auto errs = tdmd::io::preflight(cfg);
  CHECK(has_single_error_at(errs, "atoms.path"));
}

TEST_CASE("preflight: Morse D <= 0 rejects", "[io][preflight]") {
  auto cfg = valid_cfg();
  cfg.potential.morse.D = 0.0;
  const auto errs = tdmd::io::preflight(cfg);
  CHECK(has_single_error_at(errs, "potential.params.D"));

  cfg.potential.morse.D = -0.1;
  const auto errs2 = tdmd::io::preflight(cfg);
  CHECK(has_single_error_at(errs2, "potential.params.D"));
}

TEST_CASE("preflight: Morse alpha <= 0 rejects", "[io][preflight]") {
  auto cfg = valid_cfg();
  cfg.potential.morse.alpha = 0.0;
  const auto errs = tdmd::io::preflight(cfg);
  CHECK(has_single_error_at(errs, "potential.params.alpha"));
}

TEST_CASE("preflight: Morse r0 <= 0 rejects", "[io][preflight]") {
  auto cfg = valid_cfg();
  cfg.potential.morse.r0 = 0.0;
  const auto errs = tdmd::io::preflight(cfg);
  // When r0 fails, we must also skip the `cutoff <= r0` check (would
  // generate a spurious second error). Verify exactly one error.
  CHECK(has_single_error_at(errs, "potential.params.r0"));
}

TEST_CASE("preflight: Morse cutoff <= r0 rejects", "[io][preflight]") {
  auto cfg = valid_cfg();
  cfg.potential.morse.r0 = 5.0;
  cfg.potential.morse.cutoff = 4.0;  // < r0
  const auto errs = tdmd::io::preflight(cfg);
  CHECK(has_single_error_at(errs, "potential.params.cutoff"));

  // cutoff == r0 is also invalid (strict >).
  cfg.potential.morse.cutoff = 5.0;
  const auto errs2 = tdmd::io::preflight(cfg);
  CHECK(has_single_error_at(errs2, "potential.params.cutoff"));
}

TEST_CASE("preflight: non-finite Morse params rejects", "[io][preflight]") {
  auto cfg = valid_cfg();
  cfg.potential.morse.D = std::nan("");
  const auto errs = tdmd::io::preflight(cfg);
  CHECK(has_single_error_at(errs, "potential.params.D"));
}

TEST_CASE("preflight: integrator.dt <= 0 rejects", "[io][preflight]") {
  auto cfg = valid_cfg();
  cfg.integrator.dt = 0.0;
  CHECK(has_single_error_at(tdmd::io::preflight(cfg), "integrator.dt"));

  cfg.integrator.dt = -0.001;
  CHECK(has_single_error_at(tdmd::io::preflight(cfg), "integrator.dt"));
}

TEST_CASE("preflight: non-finite integrator.dt rejects", "[io][preflight]") {
  auto cfg = valid_cfg();
  cfg.integrator.dt = std::numeric_limits<double>::infinity();
  CHECK(has_single_error_at(tdmd::io::preflight(cfg), "integrator.dt"));
}

TEST_CASE("preflight: neighbor.skin <= 0 rejects", "[io][preflight]") {
  auto cfg = valid_cfg();
  cfg.neighbor.skin = 0.0;
  CHECK(has_single_error_at(tdmd::io::preflight(cfg), "neighbor.skin"));
}

TEST_CASE("preflight: run.n_steps == 0 rejects", "[io][preflight]") {
  auto cfg = valid_cfg();
  cfg.run.n_steps = 0;
  CHECK(has_single_error_at(tdmd::io::preflight(cfg), "run.n_steps"));
}

TEST_CASE("preflight: multi-error accumulation in canonical order", "[io][preflight]") {
  // Three simultaneous problems — atoms file, Morse D, integrator dt.
  auto cfg = valid_cfg();
  cfg.atoms.path = "/tmp/tdmd_missing_x.data";
  cfg.potential.morse.D = -1.0;
  cfg.integrator.dt = 0.0;

  const auto errs = tdmd::io::preflight(cfg);
  REQUIRE(errs.size() == 3);
  CHECK(errs[0].key_path == "atoms.path");
  CHECK(errs[1].key_path == "potential.params.D");
  CHECK(errs[2].key_path == "integrator.dt");
  CHECK_FALSE(tdmd::io::preflight_passes(errs));
}

TEST_CASE("preflight: determinism — same config returns identical vector twice",
          "[io][preflight]") {
  auto cfg = valid_cfg();
  cfg.atoms.path = "/tmp/tdmd_missing_y.data";
  cfg.potential.morse.cutoff = 1.0;  // < r0
  cfg.run.n_steps = 0;

  const auto a = tdmd::io::preflight(cfg);
  const auto b = tdmd::io::preflight(cfg);
  REQUIRE(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    CHECK(a[i].severity == b[i].severity);
    CHECK(a[i].key_path == b[i].key_path);
    CHECK(a[i].message == b[i].message);
  }
}

TEST_CASE("preflight: passes on the T1.3 al_fcc_small.data path", "[io][preflight]") {
  // End-to-end sanity: the fixture we ship for T1.3 is still reachable +
  // readable, which is what we care about for the atoms.path rule.
  const auto cfg = valid_cfg();
  CHECK(std::filesystem::exists(cfg.atoms.path));
  CHECK(tdmd::io::preflight(cfg).empty());
}
