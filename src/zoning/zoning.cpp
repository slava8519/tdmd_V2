// SPEC: docs/specs/zoning/SPEC.md
// Exec pack: docs/development/m3_execution_pack.md T3.2
//
// Placeholder translation unit — the interface header is all-inline for M3's
// skeleton. Scheme implementations (T3.3..T3.5) will each add their own
// .cpp files to this target. Keeping one live `.cpp` here ensures CMake
// always produces a non-empty static archive that downstream targets can
// link against before any scheme lands.

#include "tdmd/zoning/zoning.hpp"

#include "tdmd/zoning/planner.hpp"

namespace tdmd::zoning {

// Intentionally empty. The module is header-only at T3.2; this file exists
// so the static library has at least one object file and so that later
// tasks have a natural landing spot for shared helpers (e.g. axis-selection
// helpers used by both Linear1D and Decomp2D).

}  // namespace tdmd::zoning
