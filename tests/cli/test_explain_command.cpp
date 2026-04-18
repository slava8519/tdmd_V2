// T2.11 — `tdmd explain --perf` CLI tests.
//
// Covers:
//   1. Argument parsing surface (--perf, --format, --field, --help, positional).
//   2. Dispatch gating (--runtime / --scheduler deferred, --format json rejected).
//   3. End-to-end perf prediction on the shared T1.4 morse fixture.
//   4. EAM-path code (synthesised minimal config + .data header).
//   5. LAMMPS `.data` header peek edge cases (minimal file, comments, `atom
//      types` false-match, missing file).
//   6. Field documentation mode + known/unknown field symmetry with validate.

#include "tdmd/cli/explain_command.hpp"
#include "tdmd/cli/validate_command.hpp"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <filesystem>
#include <fstream>
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

std::string parse_err(const std::vector<std::string>& argv, tdmd::cli::ExplainOptions& out) {
  std::ostringstream help;
  auto result = tdmd::cli::parse_explain_options(argv, out, help);
  return result.error;
}

// Unique temp path per call — parallel test runs would otherwise clash on the
// fixed fixture name. `std::atomic` because Catch2 may run sections in
// parallel with per-case fixtures.
std::filesystem::path unique_temp(const std::string& suffix) {
  static std::atomic<int> counter{0};
  const int n = counter.fetch_add(1);
  return std::filesystem::temp_directory_path() /
         ("tdmd_explain_test_" + std::to_string(n) + "_" + suffix);
}

std::filesystem::path write_file(const std::filesystem::path& p, const std::string& content) {
  std::ofstream out(p);
  out << content;
  out.close();
  return p;
}

}  // namespace

// -----------------------------------------------------------------------------
// 1. Argument parser surface.
// -----------------------------------------------------------------------------

TEST_CASE("tdmd explain parser: positional config captured", "[cli][explain][parse]") {
  tdmd::cli::ExplainOptions opts;
  REQUIRE(parse_err({"cfg.yaml", "--perf"}, opts).empty());
  REQUIRE(opts.config_path == "cfg.yaml");
  REQUIRE(opts.perf);
  REQUIRE_FALSE(opts.runtime);
  REQUIRE(opts.format == "human");
}

TEST_CASE("tdmd explain parser: --format captured", "[cli][explain][parse]") {
  tdmd::cli::ExplainOptions opts;
  REQUIRE(parse_err({"cfg.yaml", "--perf", "--format", "json"}, opts).empty());
  REQUIRE(opts.format == "json");
}

TEST_CASE("tdmd explain parser: --zoning flag captured", "[cli][explain][parse]") {
  tdmd::cli::ExplainOptions opts;
  REQUIRE(parse_err({"cfg.yaml", "--zoning"}, opts).empty());
  REQUIRE(opts.zoning);
  REQUIRE_FALSE(opts.perf);
  REQUIRE(opts.config_path == "cfg.yaml");
}

TEST_CASE("tdmd explain parser: --field makes positional optional", "[cli][explain][parse]") {
  tdmd::cli::ExplainOptions opts;
  REQUIRE(parse_err({"--field", "units"}, opts).empty());
  REQUIRE(opts.explain_field == "units");
  REQUIRE(opts.config_path.empty());
}

TEST_CASE("tdmd explain parser: positional required when no --field", "[cli][explain][parse]") {
  tdmd::cli::ExplainOptions opts;
  const auto err = parse_err({"--perf"}, opts);
  REQUIRE_FALSE(err.empty());
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("missing positional"));
}

TEST_CASE("tdmd explain parser: --field without argument is a parse error",
          "[cli][explain][parse]") {
  tdmd::cli::ExplainOptions opts;
  const auto err = parse_err({"--field"}, opts);
  REQUIRE_FALSE(err.empty());
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("argument parse error"));
}

TEST_CASE("tdmd explain parser: --help signals early return", "[cli][explain][parse]") {
  tdmd::cli::ExplainOptions opts;
  std::ostringstream help;
  auto result = tdmd::cli::parse_explain_options({"--help"}, opts, help);
  REQUIRE(result.help_requested);
  REQUIRE_THAT(help.str(), Catch::Matchers::ContainsSubstring("Usage"));
  REQUIRE_THAT(help.str(), Catch::Matchers::ContainsSubstring("--perf"));
}

// -----------------------------------------------------------------------------
// 2. Dispatch gating for M3.
// -----------------------------------------------------------------------------

TEST_CASE("tdmd explain: missing focus flag is rejected on M3", "[cli][explain][gate]") {
  tdmd::cli::ExplainOptions opts;
  opts.config_path = io_config("valid_nve_al.yaml");

  std::ostringstream out, err;
  tdmd::cli::ExplainStreams streams{&out, &err};
  const int rc = tdmd::cli::explain_command(opts, streams);
  REQUIRE(rc == 2);
  REQUIRE_THAT(err.str(), Catch::Matchers::ContainsSubstring("--perf"));
  REQUIRE_THAT(err.str(), Catch::Matchers::ContainsSubstring("--zoning"));
}

TEST_CASE("tdmd explain: --perf and --zoning are mutually exclusive", "[cli][explain][gate]") {
  tdmd::cli::ExplainOptions opts;
  opts.config_path = io_config("valid_nve_al.yaml");
  opts.perf = true;
  opts.zoning = true;

  std::ostringstream out, err;
  tdmd::cli::ExplainStreams streams{&out, &err};
  const int rc = tdmd::cli::explain_command(opts, streams);
  REQUIRE(rc == 2);
  REQUIRE_THAT(err.str(), Catch::Matchers::ContainsSubstring("mutually exclusive"));
}

TEST_CASE("tdmd explain: --runtime focus is deferred to M4+", "[cli][explain][gate]") {
  tdmd::cli::ExplainOptions opts;
  opts.config_path = io_config("valid_nve_al.yaml");
  opts.runtime = true;

  std::ostringstream out, err;
  tdmd::cli::ExplainStreams streams{&out, &err};
  const int rc = tdmd::cli::explain_command(opts, streams);
  REQUIRE(rc == 2);
  REQUIRE_THAT(err.str(), Catch::Matchers::ContainsSubstring("M4+"));
}

TEST_CASE("tdmd explain: --scheduler focus is deferred to M4+", "[cli][explain][gate]") {
  tdmd::cli::ExplainOptions opts;
  opts.config_path = io_config("valid_nve_al.yaml");
  opts.scheduler = true;

  std::ostringstream out, err;
  tdmd::cli::ExplainStreams streams{&out, &err};
  const int rc = tdmd::cli::explain_command(opts, streams);
  REQUIRE(rc == 2);
  REQUIRE_THAT(err.str(), Catch::Matchers::ContainsSubstring("M4+"));
}

TEST_CASE("tdmd explain: --format json is deferred to M4+", "[cli][explain][gate]") {
  tdmd::cli::ExplainOptions opts;
  opts.config_path = io_config("valid_nve_al.yaml");
  opts.perf = true;
  opts.format = "json";

  std::ostringstream out, err;
  tdmd::cli::ExplainStreams streams{&out, &err};
  const int rc = tdmd::cli::explain_command(opts, streams);
  REQUIRE(rc == 2);
  REQUIRE_THAT(err.str(), Catch::Matchers::ContainsSubstring("M4+"));
  REQUIRE_THAT(err.str(), Catch::Matchers::ContainsSubstring("human"));
}

TEST_CASE("tdmd explain: --format human is accepted silently", "[cli][explain][gate]") {
  tdmd::cli::ExplainOptions opts;
  opts.config_path = io_config("valid_nve_al.yaml");
  opts.perf = true;
  opts.format = "human";

  std::ostringstream out, err;
  tdmd::cli::ExplainStreams streams{&out, &err};
  const int rc = tdmd::cli::explain_command(opts, streams);
  INFO("stderr: " << err.str());
  REQUIRE(rc == 0);
  REQUIRE(err.str().empty());
}

// -----------------------------------------------------------------------------
// 3. End-to-end perf prediction on the shared T1.4 morse fixture.
// -----------------------------------------------------------------------------

TEST_CASE("tdmd explain --perf: morse config emits Pattern 1 + Pattern 3 + M4 caveat",
          "[cli][explain][integration]") {
  tdmd::cli::ExplainOptions opts;
  opts.config_path = io_config("valid_nve_al.yaml");
  opts.perf = true;

  std::ostringstream out, err;
  tdmd::cli::ExplainStreams streams{&out, &err};
  const int rc = tdmd::cli::explain_command(opts, streams);
  INFO("stderr: " << err.str());
  INFO("stdout: " << out.str());

  REQUIRE(rc == 0);
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("Performance Prediction"));
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("Pattern1_TD"));
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("Pattern3_SD"));
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("morse"));
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("32"));
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("Caveats"));
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("until M4"));
  REQUIRE_THAT(out.str(), !Catch::Matchers::ContainsSubstring("until M5"));

  // Line-count budget per the T2.11 design note (≤ 40 lines of output).
  std::size_t line_count = 1;
  for (char c : out.str()) {
    if (c == '\n') {
      ++line_count;
    }
  }
  REQUIRE(line_count <= 40);
}

// -----------------------------------------------------------------------------
// 3b. End-to-end zoning plan on a synthesised Morse config.
// -----------------------------------------------------------------------------

TEST_CASE("tdmd explain --zoning: cubic box picks Hilbert3D with correct N_min",
          "[cli][explain][zoning][integration]") {
  // Cubic 40 Å box with a Morse cutoff of 5 Å + 0.3 Å skin → 7 zones per
  // axis (5.3 Å zone width). xy=49 ≥ 16, aspect=1 ≤ 3 → §3.4 cubic branch
  // picks Hilbert3D; N_min = 4·max(7·7, 7·7, 7·7) = 196.
  const auto data_path = unique_temp("zoning_cubic.data");
  write_file(data_path,
             "LAMMPS fixture for explain --zoning test\n"
             "\n"
             "1 atoms\n"
             "1 atom types\n"
             "0.0 40.0 xlo xhi\n"
             "0.0 40.0 ylo yhi\n"
             "0.0 40.0 zlo zhi\n"
             "\n"
             "Masses\n"
             "\n"
             "1 26.98\n"
             "\n"
             "Atoms\n"
             "\n"
             "1 1 0 0 0\n");

  const auto config_path = unique_temp("zoning_cubic.yaml");
  std::ostringstream yaml;
  yaml << "simulation:\n  units: metal\n"
       << "atoms:\n  source: lammps_data\n  path: " << data_path.string() << "\n"
       << "potential:\n  style: morse\n  params:\n    D: 0.2703\n    alpha: 1.1646\n"
       << "    r0: 3.253\n    cutoff: 5.0\n"
       << "integrator:\n  style: velocity_verlet\n  dt: 0.001\n"
       << "neighbor:\n  skin: 0.3\n"
       << "run:\n  n_steps: 1\n";
  write_file(config_path, yaml.str());

  tdmd::cli::ExplainOptions opts;
  opts.config_path = config_path.string();
  opts.zoning = true;

  std::ostringstream out, err;
  tdmd::cli::ExplainStreams streams{&out, &err};
  const int rc = tdmd::cli::explain_command(opts, streams);
  INFO("stderr: " << err.str());
  INFO("stdout: " << out.str());

  REQUIRE(rc == 0);
  REQUIRE(err.str().empty());
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("TDMD Zoning Plan"));
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("Scheme:    Hilbert3D"));
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("(7, 7, 7)"));
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("N_min per rank:    196"));
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("Canonical order:   343"));
}

TEST_CASE("tdmd explain --zoning: too-small box rejected with planner error",
          "[cli][explain][zoning][error]") {
  // 8.1 Å box with Morse cutoff 8.0 Å + skin 0.3 Å → 0 zones per axis
  // → total < 3 → ZoningPlanError surfaces as exit code 2 with an
  // actionable message naming the SD-vacuum fallback.
  const auto data_path = unique_temp("zoning_tiny.data");
  write_file(data_path,
             "LAMMPS fixture for tiny-box rejection\n"
             "\n"
             "1 atoms\n"
             "1 atom types\n"
             "0.0 8.1 xlo xhi\n"
             "0.0 8.1 ylo yhi\n"
             "0.0 8.1 zlo zhi\n"
             "\n"
             "Masses\n"
             "\n"
             "1 26.98\n"
             "\n"
             "Atoms\n"
             "\n"
             "1 1 0 0 0\n");

  const auto config_path = unique_temp("zoning_tiny.yaml");
  std::ostringstream yaml;
  yaml << "simulation:\n  units: metal\n"
       << "atoms:\n  source: lammps_data\n  path: " << data_path.string() << "\n"
       << "potential:\n  style: morse\n  params:\n    D: 0.2703\n    alpha: 1.1646\n"
       << "    r0: 3.253\n    cutoff: 8.0\n"
       << "integrator:\n  style: velocity_verlet\n  dt: 0.001\n"
       << "run:\n  n_steps: 1\n";
  write_file(config_path, yaml.str());

  tdmd::cli::ExplainOptions opts;
  opts.config_path = config_path.string();
  opts.zoning = true;

  std::ostringstream out, err;
  tdmd::cli::ExplainStreams streams{&out, &err};
  const int rc = tdmd::cli::explain_command(opts, streams);
  INFO("stderr: " << err.str());

  REQUIRE(rc == 2);
  REQUIRE_THAT(err.str(), Catch::Matchers::ContainsSubstring("fewer than 3 zones"));
}

// -----------------------------------------------------------------------------
// 4. EAM-path code.
// -----------------------------------------------------------------------------

TEST_CASE("tdmd explain --perf: eam_alloy config shows M5 caveat and EAM note",
          "[cli][explain][integration][eam]") {
  const auto data_path = unique_temp("eam.data");
  write_file(data_path,
             "LAMMPS fixture for explain EAM test\n"
             "\n"
             "100 atoms\n"
             "2 atom types\n"
             "0.0 10.0 xlo xhi\n"
             "0.0 10.0 ylo yhi\n"
             "0.0 10.0 zlo zhi\n"
             "\n"
             "Masses\n"
             "\n"
             "1 58.69\n"
             "2 26.98\n");

  // Synthetic eam_alloy config. The `params.file` path is not required to
  // exist for `explain --perf` — we skip preflight. It's included only so
  // the YAML parser sees a well-formed schema.
  const auto eam_stub = unique_temp("stub.eam.alloy");
  write_file(eam_stub, "");

  const auto config_path = unique_temp("eam.yaml");
  std::ostringstream yaml;
  yaml << "simulation:\n  units: metal\n"
       << "atoms:\n  source: lammps_data\n  path: " << data_path.string() << "\n"
       << "potential:\n  style: eam/alloy\n  params:\n    file: " << eam_stub.string() << "\n"
       << "integrator:\n  style: velocity_verlet\n  dt: 0.001\n"
       << "run:\n  n_steps: 1\n";
  write_file(config_path, yaml.str());

  tdmd::cli::ExplainOptions opts;
  opts.config_path = config_path.string();
  opts.perf = true;

  std::ostringstream out, err;
  tdmd::cli::ExplainStreams streams{&out, &err};
  const int rc = tdmd::cli::explain_command(opts, streams);
  INFO("stderr: " << err.str());
  INFO("stdout: " << out.str());
  REQUIRE(rc == 0);

  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("eam_alloy"));
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("100"));
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("until M5"));
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("eam/fs calibration"));
}

// -----------------------------------------------------------------------------
// 5. `.data` header peek edge cases — driven through full dispatch with
//    synthesised fixture pairs.
// -----------------------------------------------------------------------------

namespace {

// Helper: build a minimal YAML that points at the given data file. Wrapped
// so each edge-case test stays focused on the `.data` header content.
std::filesystem::path build_morse_config_for(const std::filesystem::path& data_path) {
  auto config_path = unique_temp("morse.yaml");
  std::ostringstream yaml;
  yaml << "simulation:\n  units: metal\n"
       << "atoms:\n  source: lammps_data\n  path: " << data_path.string() << "\n"
       << "potential:\n  style: morse\n  params:\n    D: 0.2703\n    alpha: 1.1646\n"
       << "    r0: 3.253\n    cutoff: 8.0\n"
       << "integrator:\n  style: velocity_verlet\n  dt: 0.001\n"
       << "run:\n  n_steps: 1\n";
  write_file(config_path, yaml.str());
  return config_path;
}

int run_explain_with(const std::filesystem::path& config,
                     std::ostringstream& out,
                     std::ostringstream& err) {
  tdmd::cli::ExplainOptions opts;
  opts.config_path = config.string();
  opts.perf = true;
  tdmd::cli::ExplainStreams streams{&out, &err};
  return tdmd::cli::explain_command(opts, streams);
}

}  // namespace

TEST_CASE("tdmd explain: .data header with only 'N atoms' line works",
          "[cli][explain][data-header]") {
  const auto data_path = unique_temp("minimal.data");
  write_file(data_path, "title comment\n\n7 atoms\n");
  const auto config_path = build_morse_config_for(data_path);

  std::ostringstream out, err;
  const int rc = run_explain_with(config_path, out, err);
  INFO("stderr: " << err.str());
  REQUIRE(rc == 0);
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("Atoms:     7"));
}

TEST_CASE("tdmd explain: .data header with # comments before atoms line works",
          "[cli][explain][data-header]") {
  const auto data_path = unique_temp("commented.data");
  write_file(data_path,
             "title comment\n"
             "\n"
             "# leading annotation from a downstream tool\n"
             "   # indented comment\n"
             "\n"
             "  13 atoms\n"
             "1 atom types\n");
  const auto config_path = build_morse_config_for(data_path);

  std::ostringstream out, err;
  const int rc = run_explain_with(config_path, out, err);
  INFO("stderr: " << err.str());
  REQUIRE(rc == 0);
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("Atoms:     13"));
}

TEST_CASE("tdmd explain: .data header with only 'N atom types' is rejected",
          "[cli][explain][data-header]") {
  // The "atom types" header alone (no preceding "atoms" line) must NOT
  // false-match as an atom count. We expect the scan to fall through to the
  // end-of-header section and raise "missing 'N atoms'".
  const auto data_path = unique_temp("onlytypes.data");
  write_file(data_path,
             "title\n"
             "\n"
             "1 atom types\n"
             "0.0 10.0 xlo xhi\n"
             "\n"
             "Masses\n"
             "\n"
             "1 26.98\n");
  const auto config_path = build_morse_config_for(data_path);

  std::ostringstream out, err;
  const int rc = run_explain_with(config_path, out, err);
  REQUIRE(rc == 2);
  REQUIRE_THAT(err.str(), Catch::Matchers::ContainsSubstring("missing 'N atoms'"));
}

TEST_CASE("tdmd explain: missing .data file produces a clear error",
          "[cli][explain][data-header]") {
  const auto data_path = unique_temp("does-not-exist.data");  // not written.
  const auto config_path = build_morse_config_for(data_path);

  std::ostringstream out, err;
  const int rc = run_explain_with(config_path, out, err);
  REQUIRE(rc == 2);
  REQUIRE_THAT(err.str(), Catch::Matchers::ContainsSubstring("cannot open"));
  REQUIRE_THAT(err.str(), Catch::Matchers::ContainsSubstring(data_path.string()));
}

// -----------------------------------------------------------------------------
// 6. Field-documentation mode + symmetry with validate --explain.
// -----------------------------------------------------------------------------

TEST_CASE("tdmd explain --field: known field prints body", "[cli][explain][field]") {
  tdmd::cli::ExplainOptions opts;
  opts.explain_field = "units";

  std::ostringstream out, err;
  tdmd::cli::ExplainStreams streams{&out, &err};
  const int rc = tdmd::cli::explain_command(opts, streams);
  REQUIRE(rc == 0);
  REQUIRE_THAT(out.str(), Catch::Matchers::StartsWith("units:"));
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("metal"));
}

TEST_CASE("tdmd explain --field: unknown field lists recognised fields", "[cli][explain][field]") {
  tdmd::cli::ExplainOptions opts;
  opts.explain_field = "no.such.field";

  std::ostringstream out, err;
  tdmd::cli::ExplainStreams streams{&out, &err};
  const int rc = tdmd::cli::explain_command(opts, streams);
  REQUIRE(rc == 2);
  REQUIRE_THAT(err.str(), Catch::Matchers::ContainsSubstring("unknown field"));
  REQUIRE_THAT(err.str(), Catch::Matchers::ContainsSubstring("units"));
}

TEST_CASE("tdmd explain --field and validate --explain produce identical bodies",
          "[cli][explain][field][regression]") {
  // Symmetry guarantee from the T2.11 design: both peer paths share a single
  // `config_field_descriptions()` table. A regression here would indicate
  // someone duplicated the table back into one of the files.
  const std::string field = "integrator.dt";

  tdmd::cli::ExplainOptions explain_opts;
  explain_opts.explain_field = field;
  std::ostringstream explain_out, explain_err;
  const int explain_rc = tdmd::cli::explain_command(explain_opts, {&explain_out, &explain_err});
  REQUIRE(explain_rc == 0);

  tdmd::cli::ValidateOptions validate_opts;
  validate_opts.explain_field = field;
  std::ostringstream validate_out, validate_err;
  const int validate_rc =
      tdmd::cli::validate_command(validate_opts, {&validate_out, &validate_err});
  REQUIRE(validate_rc == 0);

  REQUIRE(explain_out.str() == validate_out.str());
}
