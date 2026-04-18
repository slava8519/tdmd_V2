// SPEC: docs/specs/scheduler/SPEC.md
// Exec pack: docs/development/m4_execution_pack.md T4.2
//
// Placeholder translation unit — the module is mostly header-only at T4.2.
// Concrete implementations (SafetyCertificate, ZoneStateMachine,
// CausalWavefrontScheduler) land in T4.3–T4.8 and each adds its own .cpp
// to this target. This file ensures the static archive is non-empty so
// downstream targets (tests, runtime) can link against tdmd_scheduler
// before any concrete implementation lands.

#include "tdmd/scheduler/td_scheduler.hpp"
#include "tdmd/scheduler/types.hpp"

namespace tdmd::scheduler {

// Intentionally empty.

}  // namespace tdmd::scheduler
