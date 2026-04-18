#pragma once

// SPEC: docs/specs/cli/SPEC.md §2.1 (tdmd run),
//       docs/specs/runtime/SPEC.md §2.2 (SimulationEngine lifecycle)
// Exec pack: docs/development/m1_execution_pack.md T1.9
//
// Entry point for `tdmd run <config.yaml>`. The CLI layer is a thin shell:
// parse args → load YAML → preflight → drive `SimulationEngine`. It owns the
// process exit code contract for this subcommand:
//
//   0 — run completed successfully
//   1 — runtime error (IO, physics divergence, unexpected exception)
//   2 — preflight failure (schema / semantic validation rejected the config)
//
// `run_command` is split out from `main` so the integration test can exercise
// the full subcommand end-to-end without shelling out to an installed binary.

#include <iosfwd>
#include <string>
#include <vector>

namespace tdmd::cli {

// Options recognised by `tdmd run`. Populated by `parse_run_options` or filled
// directly in tests.
struct RunOptions {
  std::string config_path;  // positional: path to tdmd.yaml
  std::string thermo_path;  // optional --thermo <file>; empty = stdout
  bool quiet = false;       // --quiet suppresses non-thermo stdout
};

// Streams the CLI layer writes into. Tests inject std::ostringstream here so
// the subcommand is hermetic; `main` wires these to `std::cout` / `std::cerr`.
struct RunStreams {
  std::ostream* out = nullptr;  // human-readable progress
  std::ostream* err = nullptr;  // errors (preflight + runtime)
};

// Returned exit code follows the contract above.
[[nodiscard]] int run_command(const RunOptions& options, const RunStreams& streams);

// Argument parser for `tdmd run`. Populates `out_options` on success. Returns
// a non-empty error string on parse failure (missing positional, unknown flag,
// conflicting options). `--help` is handled by the parser by writing usage to
// `help_out` and returning a sentinel (see `RunParseResult`).
struct RunParseResult {
  bool help_requested = false;  // --help / -h was seen
  std::string error;            // non-empty on parse failure
};

[[nodiscard]] RunParseResult parse_run_options(const std::vector<std::string>& argv,
                                               RunOptions& out_options,
                                               std::ostream& help_out);

}  // namespace tdmd::cli
