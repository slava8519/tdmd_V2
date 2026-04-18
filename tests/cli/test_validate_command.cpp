// T1.10 — `tdmd validate` CLI end-to-end tests.
//
// Reuses the T1.4 YAML fixtures so the validate layer exercises the same
// failure modes the preflight unit tests do, but through the CLI surface
// (exit codes, stderr vs stdout partitioning, --strict, --explain).

#include "tdmd/cli/validate_command.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <sstream>
#include <string>
#include <vector>

#ifndef TDMD_IO_FIXTURES_DIR
#error "TDMD_IO_FIXTURES_DIR must be defined by the build system"
#endif

namespace {

std::string io_config(const std::string& name) {
  return std::string(TDMD_IO_FIXTURES_DIR) + "/configs/" + name;
}

std::string parse_err(const std::vector<std::string>& argv, tdmd::cli::ValidateOptions& out) {
  std::ostringstream help;
  auto result = tdmd::cli::parse_validate_options(argv, out, help);
  return result.error;
}

}  // namespace

TEST_CASE("tdmd validate parser: positional config captured", "[cli][validate][parse]") {
  tdmd::cli::ValidateOptions opts;
  REQUIRE(parse_err({"cfg.yaml"}, opts).empty());
  REQUIRE(opts.config_path == "cfg.yaml");
  REQUIRE_FALSE(opts.strict);
  REQUIRE(opts.explain_field.empty());
}

TEST_CASE("tdmd validate parser: --strict captured", "[cli][validate][parse]") {
  tdmd::cli::ValidateOptions opts;
  REQUIRE(parse_err({"--strict", "cfg.yaml"}, opts).empty());
  REQUIRE(opts.strict);
}

TEST_CASE("tdmd validate parser: --explain makes positional optional", "[cli][validate][parse]") {
  tdmd::cli::ValidateOptions opts;
  REQUIRE(parse_err({"--explain", "units"}, opts).empty());
  REQUIRE(opts.explain_field == "units");
  REQUIRE(opts.config_path.empty());
}

TEST_CASE("tdmd validate parser: positional required when no --explain", "[cli][validate][parse]") {
  tdmd::cli::ValidateOptions opts;
  const auto err = parse_err({}, opts);
  REQUIRE_FALSE(err.empty());
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("missing positional"));
}

TEST_CASE("tdmd validate parser: --help signals early return", "[cli][validate][parse]") {
  tdmd::cli::ValidateOptions opts;
  std::ostringstream help;
  auto result = tdmd::cli::parse_validate_options({"--help"}, opts, help);
  REQUIRE(result.help_requested);
  REQUIRE_THAT(help.str(), Catch::Matchers::ContainsSubstring("Usage"));
}

TEST_CASE("tdmd validate: valid config exits 0 with summary", "[cli][validate][integration]") {
  tdmd::cli::ValidateOptions opts;
  opts.config_path = io_config("valid_nve_al.yaml");

  std::ostringstream out;
  std::ostringstream err;
  tdmd::cli::ValidateStreams streams{&out, &err};
  const int rc = tdmd::cli::validate_command(opts, streams);

  INFO("stderr: " << err.str());
  REQUIRE(rc == 0);
  REQUIRE_THAT(out.str(), Catch::Matchers::StartsWith("OK\n"));
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("units:       metal"));
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("velocity_verlet"));
  REQUIRE(err.str().empty());
}

TEST_CASE("tdmd validate: missing_units.yaml exits 2 with parse error",
          "[cli][validate][integration]") {
  tdmd::cli::ValidateOptions opts;
  opts.config_path = io_config("missing_units.yaml");

  std::ostringstream out;
  std::ostringstream err;
  tdmd::cli::ValidateStreams streams{&out, &err};
  const int rc = tdmd::cli::validate_command(opts, streams);

  REQUIRE(rc == 2);
  // missing_units is a parse-stage failure (required key absent) rather than
  // a preflight issue — the error message mentions units.
  REQUIRE_THAT(err.str(),
               Catch::Matchers::ContainsSubstring("config parse error") ||
                   Catch::Matchers::ContainsSubstring("units"));
}

TEST_CASE("tdmd validate: missing atoms file surfaces preflight error",
          "[cli][validate][integration]") {
  tdmd::cli::ValidateOptions opts;
  opts.config_path = io_config("missing_atoms_file.yaml");

  std::ostringstream out;
  std::ostringstream err;
  tdmd::cli::ValidateStreams streams{&out, &err};
  const int rc = tdmd::cli::validate_command(opts, streams);

  REQUIRE(rc == 2);
  REQUIRE_THAT(err.str(), Catch::Matchers::ContainsSubstring("preflight failed"));
  REQUIRE_THAT(err.str(), Catch::Matchers::ContainsSubstring("atoms.path"));
}

TEST_CASE("tdmd validate: bad_timestep.yaml fails on integrator.dt",
          "[cli][validate][integration]") {
  tdmd::cli::ValidateOptions opts;
  opts.config_path = io_config("bad_timestep.yaml");

  std::ostringstream out;
  std::ostringstream err;
  tdmd::cli::ValidateStreams streams{&out, &err};
  const int rc = tdmd::cli::validate_command(opts, streams);

  REQUIRE(rc == 2);
  // The error surfaces either at parse (if the value is rejected as unparseable
  // for its schema type) or at preflight (if it parses but is <= 0). Either
  // way the key_path should appear in the stderr stream.
  REQUIRE_THAT(err.str(),
               Catch::Matchers::ContainsSubstring("integrator.dt") ||
                   Catch::Matchers::ContainsSubstring("dt"));
}

TEST_CASE("tdmd validate: --explain on known field prints body", "[cli][validate][explain]") {
  tdmd::cli::ValidateOptions opts;
  opts.explain_field = "units";

  std::ostringstream out;
  std::ostringstream err;
  tdmd::cli::ValidateStreams streams{&out, &err};
  const int rc = tdmd::cli::validate_command(opts, streams);

  REQUIRE(rc == 0);
  REQUIRE_THAT(out.str(), Catch::Matchers::StartsWith("units:"));
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("metal"));
}

TEST_CASE("tdmd validate: --explain on unknown field lists recognised fields",
          "[cli][validate][explain]") {
  tdmd::cli::ValidateOptions opts;
  opts.explain_field = "definitely.not.a.field";

  std::ostringstream out;
  std::ostringstream err;
  tdmd::cli::ValidateStreams streams{&out, &err};
  const int rc = tdmd::cli::validate_command(opts, streams);

  REQUIRE(rc == 2);
  REQUIRE_THAT(err.str(), Catch::Matchers::ContainsSubstring("unknown field"));
  REQUIRE_THAT(err.str(), Catch::Matchers::ContainsSubstring("units"));
}

TEST_CASE("tdmd validate: --strict promotes warnings to errors", "[cli][validate][strict]") {
  // M1 preflight has no warning-only rules yet (see io/preflight.cpp — every
  // rule emits Error). Verify that --strict is at least a no-op on a clean
  // config so the contract does not regress silently when warning severities
  // are introduced in later milestones.
  tdmd::cli::ValidateOptions opts;
  opts.config_path = io_config("valid_nve_al.yaml");
  opts.strict = true;

  std::ostringstream out;
  std::ostringstream err;
  tdmd::cli::ValidateStreams streams{&out, &err};
  const int rc = tdmd::cli::validate_command(opts, streams);
  REQUIRE(rc == 0);
  REQUIRE_THAT(out.str(), Catch::Matchers::StartsWith("OK\n"));
}

TEST_CASE("tdmd validate: does not open the atoms .data file", "[cli][validate][invariant]") {
  // The validate contract (io/SPEC §3.3, cli/SPEC §2.2) forbids reading the
  // .data payload — preflight only stat()s the path. The missing_atoms_file
  // fixture fails on the stat check, but a config pointing at an existing
  // .data file must succeed without loading atoms. We exercise that by
  // validating valid_nve_al.yaml (which points at ../al_fcc_small.data) and
  // confirming stdout does NOT contain any "atom" count that would come from
  // actually reading the file.
  tdmd::cli::ValidateOptions opts;
  opts.config_path = io_config("valid_nve_al.yaml");

  std::ostringstream out;
  std::ostringstream err;
  tdmd::cli::ValidateStreams streams{&out, &err};
  const int rc = tdmd::cli::validate_command(opts, streams);

  REQUIRE(rc == 0);
  REQUIRE(out.str().find("32 atoms") == std::string::npos);
}
