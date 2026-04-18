#pragma once

// SPEC: docs/specs/cli/SPEC.md §5 (tdmd explain),
//       docs/specs/perfmodel/SPEC.md §6.1 (explain format)
// Exec pack: docs/development/m2_execution_pack.md T2.11
//
// `tdmd explain <config.yaml>` — pedagogical view of what TDMD will do with a
// given config. M2 ships only the `--perf` focus; `--runtime` / `--scheduler`
// are accepted by the parser but rejected at dispatch with a clear "M3+" hint,
// because zoning plans (M3) and the TdScheduler (M4) don't exist yet.
//
// M2 surface:
//   --perf                  Required for prediction mode. Emits Pattern 1 / 3
//                           analytic breakdown from `PerfModel` with the
//                           `HardwareProfile::modern_x86_64()` profile.
//   --format human          Only human format is implemented; json / markdown
//                           carry-forward to M3+.
//   --field <name>          Field-documentation mode (orthogonal to --perf).
//                           Same table as `tdmd validate --explain` — kept
//                           symmetric because cli/SPEC §5 makes `tdmd explain`
//                           the canonical top-level command.
//
// Exit-code contract (matches `tdmd run` / `tdmd validate`):
//   0 — explain produced and printed successfully.
//   1 — internal / IO error (e.g. PerfModel sanity check or unexpected state).
//   2 — bad argv (missing --perf, unsupported format, M3+ focus requested,
//       missing positional, config parse failure, missing .data file header).

#include <iosfwd>
#include <string>
#include <vector>

namespace tdmd::cli {

struct ExplainOptions {
  std::string config_path;    // positional; empty when --field drives the call.
  std::string explain_field;  // --field <name>; empty → perf mode.
  std::string format = "human";
  bool perf = false;
  bool runtime = false;    // M3+ — accepted by parser, rejected at dispatch.
  bool scheduler = false;  // M4+ — same.
  bool verbose = false;    // M2 no-op; accepted so scripts don't break.
};

struct ExplainStreams {
  std::ostream* out = nullptr;  // human-readable explain output
  std::ostream* err = nullptr;  // usage / parse / missing-input errors
};

[[nodiscard]] int explain_command(const ExplainOptions& options, const ExplainStreams& streams);

struct ExplainParseResult {
  bool help_requested = false;
  std::string error;  // non-empty on parse failure
};

[[nodiscard]] ExplainParseResult parse_explain_options(const std::vector<std::string>& argv,
                                                       ExplainOptions& out_options,
                                                       std::ostream& help_out);

}  // namespace tdmd::cli
