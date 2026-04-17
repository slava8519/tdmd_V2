#include "tdmd/state/atom_soa.hpp"

// TODO(M1): AtomSoA implementation per state/SPEC.md.
// M0 skeleton — header-only behavior is sufficient for smoke tests.

namespace tdmd {

// Intentional TU so tdmd_state has at least one object file.
// Remove when real implementation lands and pulls in its own cpp's.
extern const char* const kStateModuleTag;
const char* const kStateModuleTag = "tdmd::state (M0 skeleton)";

}  // namespace tdmd
