#pragma once

// SPEC: docs/specs/cli/SPEC.md §4 (validate), §5 (explain)
// Exec pack: docs/development/m2_execution_pack.md T2.11
//
// Short-form descriptions of top-level tdmd.yaml fields, shared between
// `tdmd validate --explain <field>` (T1.10) and `tdmd explain --field <field>`
// (T2.11). Keeping the table in one place guarantees symmetric output on both
// peer paths — the CLI SPEC §5 promises `tdmd explain` as the canonical form,
// but `validate --explain` is kept working indefinitely for the M1 users who
// already scripted against it.

#include <map>
#include <string>

namespace tdmd::cli {

// Returns a stable reference to the shared field → body map. Entries are
// two-or-three-sentence summaries of the io/SPEC §3 block descriptions;
// master-spec §5.3 is the source for unit-system semantics.
[[nodiscard]] const std::map<std::string, std::string>& config_field_descriptions();

}  // namespace tdmd::cli
