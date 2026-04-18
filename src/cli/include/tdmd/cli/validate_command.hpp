#pragma once

// SPEC: docs/specs/cli/SPEC.md §2.2 (tdmd validate),
//       docs/specs/io/SPEC.md §3.3 (preflight)
// Exec pack: docs/development/m1_execution_pack.md T1.10
//
// `tdmd validate <config.yaml>` runs the full parse + preflight pipeline
// without touching `.data` files or launching the simulation. It is the
// user's pre-flight check before committing to a long run.
//
// Exit code contract (matches `tdmd run` so batch drivers can dispatch on a
// single switch):
//   0 — config parses and preflight passes (no errors; warnings are
//       allowed unless --strict).
//   1 — internal / IO error (e.g. file not readable for a reason other than
//       missing).
//   2 — config parse error OR preflight found errors OR --strict promoted a
//       warning to an error.
//
// --explain <field> is a documentation hook orthogonal to validation: it
// prints a short human-readable description for the named field and exits 0.
// The M1 field set is intentionally small (cli/SPEC §2.2 hint: only `units`
// is mandatory), but we include the other top-level keys a first-time user
// is likely to ask about.

#include <iosfwd>
#include <string>
#include <vector>

namespace tdmd::cli {

// Options recognised by `tdmd validate`. Populated by `parse_validate_options`
// or filled directly in tests.
struct ValidateOptions {
  std::string config_path;    // positional; empty when --explain drives the call.
  std::string explain_field;  // --explain <field>; empty → validation mode.
  bool strict = false;        // --strict promotes preflight warnings to errors.
};

// Streams the validate subcommand writes into — structurally identical to the
// run streams so `main` can wire them the same way.
struct ValidateStreams {
  std::ostream* out = nullptr;  // summaries, `--explain` body
  std::ostream* err = nullptr;  // parse / preflight errors
};

// Per the contract above.
[[nodiscard]] int validate_command(const ValidateOptions& options, const ValidateStreams& streams);

// Argument parser. `help_out` receives the usage string on `--help`.
struct ValidateParseResult {
  bool help_requested = false;
  std::string error;  // non-empty on parse failure
};

[[nodiscard]] ValidateParseResult parse_validate_options(const std::vector<std::string>& argv,
                                                         ValidateOptions& out_options,
                                                         std::ostream& help_out);

}  // namespace tdmd::cli
