#pragma once

// SPEC: docs/specs/scheduler/SPEC.md §4 (safety certificate)
// Master spec: §6.4 (displacement bound + I7 monotonicity)
// Exec pack: docs/development/m4_execution_pack.md T4.3
//
// Pure math + factory for SafetyCertificate. No state owned here — the
// CertificateStore owns storage + cert_id allocation (certificate_store.hpp).
//
// The math is intentionally platform-agnostic double arithmetic: in Reference
// profile the cert math MUST be identical across architectures (SPEC §4.5),
// so we avoid FMA intrinsics and any fast-math flags in this TU.

#include "tdmd/scheduler/types.hpp"

#include <cstdint>

namespace tdmd::scheduler {

// Inputs to certificate construction. One aggregate rather than a 13-arg
// factory — call sites read better with designated initializers, and the
// struct doubles as the fuzzer's random-sample record.
struct CertificateInputs {
  ZoneId zone_id = 0;
  TimeLevel time_level = 0;
  Version version = 0;

  double v_max_zone = 0.0;    // Å / ps
  double a_max_zone = 0.0;    // Å / ps²
  double dt_candidate = 0.0;  // ps

  double buffer_width = 0.0;     // Å — zoning-owned skin buffer
  double skin_remaining = 0.0;   // Å — neighbor-owned unused skin
  double frontier_margin = 0.0;  // Å — K_max·dt − (t − frontier_min)·dt

  TimeLevel neighbor_valid_until_step = 0;
  TimeLevel halo_valid_until_step = 0;

  std::uint64_t mode_policy_tag = 0;
};

// δ(dt) = v·dt + 0.5·a·dt². Returns NaN if any input is NaN; returns a finite
// value for finite inputs (cannot overflow at metal-unit scales). Pure; no
// clamping — the safe() predicate handles defensive clamping.
double compute_displacement_bound(double v_max, double a_max, double dt) noexcept;

// safe ⇔ all of { v, a, dt, buffer, skin, frontier, delta } finite AND
//                  v, a, dt, buffer, skin, frontier ≥ 0 AND
//                  delta < min(buffer, skin, frontier).
//
// Defensive policy (R-M4-2): any non-finite or negative numeric input → false.
// Degenerate dt ≤ 0 → false (can't integrate zero/negative time).
// This policy is load-bearing for I7 monotonicity — see build_certificate().
bool compute_safe(double v_max,
                  double a_max,
                  double dt,
                  double buffer,
                  double skin,
                  double frontier) noexcept;

// Factory. Pure, stateless. `cert_id` is injected (the store owns allocation).
// Fills every field of SafetyCertificate from the inputs; `safe` and
// `displacement_bound` are computed via the functions above.
SafetyCertificate build_certificate(std::uint64_t cert_id, const CertificateInputs& in) noexcept;

}  // namespace tdmd::scheduler
