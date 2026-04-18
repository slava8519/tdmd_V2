#pragma once

// SPEC: docs/specs/cli/SPEC.md §5 (tdmd explain),
//       docs/specs/perfmodel/SPEC.md §6.1 (explain --perf format),
//       docs/specs/zoning/SPEC.md §3.4, §4.3 (ZoningPlan + rationale)
// Exec pack: docs/development/m2_execution_pack.md T2.11,
//            docs/development/m3_execution_pack.md T3.9
//
// `tdmd explain <config.yaml>` — pedagogical view of what TDMD will do with a
// given config. M3 ships `--perf` and `--zoning`; `--scheduler` is accepted
// by the parser but rejected at dispatch with a clear "M4+" hint because
// the TdScheduler doesn't exist yet.
//
// M3 surface:
//   --perf                  Emits Pattern 1 / 3 analytic breakdown from
//                           `PerfModel` with the modern_x86_64 HW profile.
//   --zoning                Emits the zoning plan rationale: auto-selected
//                           scheme (Linear1D / Decomp2D / Hilbert3D), per-axis
//                           zone counts, zone size, N_min per rank, optimal
//                           rank count, canonical order length, and advisories
//                           from `DefaultZoningPlanner`. Cutoff is read from
//                           the potential (Morse: params.cutoff; EAM/alloy:
//                           parsed from the setfl file header); skin from
//                           `neighbor.skin`. Mutually exclusive with --perf.
//   --format human          Only human format is implemented; json / markdown
//                           carry-forward to M4+.
//   --field <name>          Field-documentation mode (orthogonal to --perf /
//                           --zoning). Same table as `tdmd validate --explain`.
//
// Exit-code contract (matches `tdmd run` / `tdmd validate`):
//   0 — explain produced and printed successfully.
//   1 — internal / IO error (PerfModel sanity, setfl parse, unexpected state).
//   2 — bad argv (no focus, multiple focuses, unsupported format, M4+ focus
//       requested, missing positional, config parse failure, missing .data
//       file header).

#include <iosfwd>
#include <string>
#include <vector>

namespace tdmd::cli {

struct ExplainOptions {
  std::string config_path;    // positional; empty when --field drives the call.
  std::string explain_field;  // --field <name>; empty → perf mode.
  std::string format = "human";
  bool perf = false;
  bool zoning = false;     // M3 — ZoningPlan rationale from DefaultZoningPlanner.
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
