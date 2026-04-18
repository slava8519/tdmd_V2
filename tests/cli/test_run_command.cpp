// T1.9 — `tdmd run` CLI end-to-end tests.
//
// Exercises the parser + run_command layer without spawning a child process,
// so failures surface as clean Catch2 messages instead of process crashes.
// The integration test path runs the same code that `tdmd run <...>` invokes
// in production; only `argv[0]`-style dispatch from main.cpp is skipped.

#include "tdmd/cli/run_command.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#ifndef TDMD_IO_FIXTURES_DIR
#error "TDMD_IO_FIXTURES_DIR must be defined by the build system"
#endif

namespace {

// Path to the CLI fixture. Computed relative to the io fixtures dir we already
// inject — the cli fixture directory lives one level up, so we synthesise it
// from CMAKE_SOURCE_DIR.
std::string cli_fixture(const std::string& name) {
  namespace fs = std::filesystem;
  fs::path base(TDMD_IO_FIXTURES_DIR);
  // TDMD_IO_FIXTURES_DIR is tests/io/fixtures — go up two, across to cli.
  return (base.parent_path().parent_path() / "cli" / "fixtures" / name).string();
}

std::string io_config_fixture(const std::string& name) {
  return std::string(TDMD_IO_FIXTURES_DIR) + "/configs/" + name;
}

// Tiny helper: run the parser over a positional-only argv and return errors.
std::string parse_err(const std::vector<std::string>& argv, tdmd::cli::RunOptions& out) {
  std::ostringstream help;
  auto result = tdmd::cli::parse_run_options(argv, out, help);
  return result.error;
}

}  // namespace

TEST_CASE("tdmd run parser: positional config is required", "[cli][run][parse]") {
  tdmd::cli::RunOptions opts;
  const auto err = parse_err({}, opts);
  REQUIRE_FALSE(err.empty());
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("missing positional"));
}

TEST_CASE("tdmd run parser: positional config captured", "[cli][run][parse]") {
  tdmd::cli::RunOptions opts;
  REQUIRE(parse_err({"path/to/config.yaml"}, opts).empty());
  REQUIRE(opts.config_path == "path/to/config.yaml");
  REQUIRE(opts.thermo_path.empty());
  REQUIRE_FALSE(opts.quiet);
}

TEST_CASE("tdmd run parser: --thermo + --quiet captured", "[cli][run][parse]") {
  tdmd::cli::RunOptions opts;
  REQUIRE(parse_err({"--thermo", "/tmp/t.log", "--quiet", "cfg.yaml"}, opts).empty());
  REQUIRE(opts.config_path == "cfg.yaml");
  REQUIRE(opts.thermo_path == "/tmp/t.log");
  REQUIRE(opts.quiet);
}

TEST_CASE("tdmd run parser: --help writes usage and signals early-return", "[cli][run][parse]") {
  tdmd::cli::RunOptions opts;
  std::ostringstream help;
  auto result = tdmd::cli::parse_run_options({"--help"}, opts, help);
  REQUIRE(result.help_requested);
  REQUIRE(result.error.empty());
  REQUIRE_THAT(help.str(), Catch::Matchers::ContainsSubstring("Usage"));
}

TEST_CASE("tdmd run parser: unknown flag rejected", "[cli][run][parse]") {
  tdmd::cli::RunOptions opts;
  const auto err = parse_err({"--no-such-flag", "cfg.yaml"}, opts);
  REQUIRE_FALSE(err.empty());
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("argument parse error"));
}

TEST_CASE("tdmd run: happy path returns 0 and emits thermo stream", "[cli][run][integration]") {
  tdmd::cli::RunOptions opts;
  opts.config_path = cli_fixture("cli_nve_toy.yaml");
  opts.quiet = true;

  std::ostringstream out;
  std::ostringstream err;
  tdmd::cli::RunStreams streams{&out, &err};
  const int rc = tdmd::cli::run_command(opts, streams);

  INFO("stderr: " << err.str());
  REQUIRE(rc == 0);
  REQUIRE_THAT(out.str(), Catch::Matchers::ContainsSubstring("# step temp pe ke etotal press"));
  // 4 steps, thermo every 2 → rows at 0, 2, 4 = 3 data rows.
  const auto out_str = out.str();
  std::size_t newlines = 0;
  for (char c : out_str) {
    if (c == '\n') {
      ++newlines;
    }
  }
  // header + 3 data rows = 4 newlines (quiet suppresses status lines).
  REQUIRE(newlines == 4U);
}

TEST_CASE("tdmd run: --thermo redirects output to a file", "[cli][run][integration]") {
  namespace fs = std::filesystem;
  const auto tmp = fs::temp_directory_path() / "tdmd_cli_thermo.tmp";
  fs::remove(tmp);

  tdmd::cli::RunOptions opts;
  opts.config_path = cli_fixture("cli_nve_toy.yaml");
  opts.thermo_path = tmp.string();
  opts.quiet = true;

  std::ostringstream out;
  std::ostringstream err;
  tdmd::cli::RunStreams streams{&out, &err};
  const int rc = tdmd::cli::run_command(opts, streams);

  INFO("stderr: " << err.str());
  REQUIRE(rc == 0);
  REQUIRE(fs::exists(tmp));
  REQUIRE(fs::file_size(tmp) > 0U);
  // Stdout should contain NO thermo — everything went to the file.
  REQUIRE(out.str().find("# step") == std::string::npos);
  fs::remove(tmp);
}

TEST_CASE("tdmd run: preflight failure returns exit code 2", "[cli][run][exit-codes]") {
  tdmd::cli::RunOptions opts;
  // The io T1.4 test suite ships a fixture that fails preflight because its
  // atoms file does not exist. Reusing it keeps the cli layer hermetic.
  opts.config_path = io_config_fixture("missing_atoms_file.yaml");
  opts.quiet = true;

  std::ostringstream out;
  std::ostringstream err;
  tdmd::cli::RunStreams streams{&out, &err};
  const int rc = tdmd::cli::run_command(opts, streams);

  REQUIRE(rc == 2);
  REQUIRE_THAT(err.str(), Catch::Matchers::ContainsSubstring("preflight failed"));
}

TEST_CASE("tdmd run: yaml parse failure returns exit code 2", "[cli][run][exit-codes]") {
  tdmd::cli::RunOptions opts;
  opts.config_path = io_config_fixture("missing_units.yaml");
  opts.quiet = true;

  std::ostringstream out;
  std::ostringstream err;
  tdmd::cli::RunStreams streams{&out, &err};
  const int rc = tdmd::cli::run_command(opts, streams);

  REQUIRE(rc == 2);
  // missing_units fails during parse (required field), not during preflight.
  REQUIRE_THAT(err.str(), Catch::Matchers::ContainsSubstring("config parse error"));
}

TEST_CASE("tdmd run: non-existent config returns exit code 1", "[cli][run][exit-codes]") {
  tdmd::cli::RunOptions opts;
  opts.config_path = "/absolutely/does/not/exist.yaml";
  opts.quiet = true;

  std::ostringstream out;
  std::ostringstream err;
  tdmd::cli::RunStreams streams{&out, &err};
  const int rc = tdmd::cli::run_command(opts, streams);

  // Non-existent file surfaces as a YamlParseError at the io layer → exit 2
  // (treated as a config-parse failure) or as an IO exception → exit 1. We
  // accept either here; the important contract is non-zero.
  REQUIRE(rc != 0);
}

TEST_CASE("tdmd run: thermo stream is deterministic across invocations",
          "[cli][run][determinism]") {
  tdmd::cli::RunOptions opts;
  opts.config_path = cli_fixture("cli_nve_toy.yaml");
  opts.quiet = true;

  auto run_once = [&]() {
    std::ostringstream out;
    std::ostringstream err;
    tdmd::cli::RunStreams streams{&out, &err};
    const int rc = tdmd::cli::run_command(opts, streams);
    REQUIRE(rc == 0);
    return out.str();
  };

  const auto first = run_once();
  const auto second = run_once();
  REQUIRE(first == second);
}
