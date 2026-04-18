#pragma once

// SPEC: docs/specs/scheduler/SPEC.md §4.2 (cert input provenance)
// Master spec: §6.4
// Exec pack: docs/development/m4_execution_pack.md T4.5, T4.9
//
// Pluggable source of the state-owned / neighbor-owned inputs that go into
// a SafetyCertificate (see types.hpp::CertificateInputs). The scheduler
// holds a pointer to one; `refresh_certificates()` dereferences it per
// zone to populate v_max / a_max / skin_remaining / ...
//
// In T4.5 the scheduler accepts a nullptr source (no state wiring yet) —
// refresh_certificates then builds cert entries with zero physics inputs.
// That's fine for T4.5's acceptance (cert count + ordering), useless for
// T4.6's safety check. T4.9 wires a concrete source backed by the
// SimulationEngine's state + neighbor modules.
//
// The interface is minimal by design — one call per zone, out-parameter
// aggregate — so concrete sources stay easy to mock in tests.

#include "tdmd/scheduler/safety_certificate.hpp"
#include "tdmd/scheduler/types.hpp"

namespace tdmd::scheduler {

class CertificateInputSource {
public:
  virtual ~CertificateInputSource() = default;

  // Fill `out` with the zone's current physics/neighbor snapshot for the
  // candidate step `time_level`. Must be pure (no side effects) and
  // thread-safe-for-read in M5+; M4 calls this serially.
  //
  // Implementations are free to compute `dt_candidate`, `frontier_margin`,
  // and the validity windows; the scheduler still overrides
  // `mode_policy_tag` to its own store's tag on insert.
  virtual void fill_inputs(ZoneId zone, TimeLevel time_level, CertificateInputs& out) const = 0;
};

}  // namespace tdmd::scheduler
