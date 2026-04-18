#pragma once

// SPEC: docs/specs/io/SPEC.md §3.3 (preflight semantic validation)
//       docs/specs/cli/SPEC.md §4.3 (tdmd validate check categories)
// Exec pack: docs/development/m1_execution_pack.md T1.4
//
// Semantic validation pass — takes a parsed `YamlConfig` and returns the
// accumulated list of issues. Multi-error by design (exec pack T1.4
// "multi-error mode — collect все issues, report все сразу"): a malformed
// config might violate three invariants, and we want the user to see all
// three on one CI run instead of fixing them one at a time.
//
// Pure function: given the same `YamlConfig` value and the same filesystem
// view (e.g. whether `atoms.path` exists), returns the same vector in the same
// order. Parse-stage YamlParseError is orthogonal — structural schema
// problems are already caught and never reach here.

#include "tdmd/io/yaml_config.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace tdmd::io {

enum class PreflightSeverity : std::uint8_t {
  Error,    // blocks `tdmd run`; exit code 2 for `tdmd validate`.
  Warning,  // does not block `tdmd run`; exit code 1 for `tdmd validate`.
};

struct PreflightError {
  PreflightSeverity severity = PreflightSeverity::Error;
  // Dotted path to the offending field, e.g. `integrator.dt` or
  // `potential.params.cutoff`. Empty for whole-config problems.
  std::string key_path;
  // Human-readable message — should include the offending value + the
  // expectation (see cli/SPEC §4.4 examples).
  std::string message;
};

// Runs every M1 semantic rule and returns the accumulated issues. Empty vector
// → config is valid for `tdmd run`.
//
// Rules checked (per exec pack T1.4 + io/SPEC §3.3):
//   - simulation.seed is finite (always true for uint64; kept for symmetry).
//   - atoms.path file exists and is readable.
//   - potential.morse: D > 0, alpha > 0, r0 > 0, cutoff > r0 — all finite.
//   - integrator.dt > 0, finite.
//   - neighbor.skin > 0, finite.
//   - run.n_steps >= 1.
[[nodiscard]] std::vector<PreflightError> preflight(const YamlConfig& config);

// Convenience: true iff no entries have severity == Error. Warnings do not
// block execution (matches cli/SPEC §4.6 exit-code mapping).
[[nodiscard]] bool preflight_passes(const std::vector<PreflightError>& errors) noexcept;

}  // namespace tdmd::io
