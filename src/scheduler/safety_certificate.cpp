// SPEC: docs/specs/scheduler/SPEC.md §4
// Master spec: §6.4
// Exec pack: docs/development/m4_execution_pack.md T4.3

#include "tdmd/scheduler/safety_certificate.hpp"

#include <cmath>

namespace tdmd::scheduler {

double compute_displacement_bound(double v_max, double a_max, double dt) noexcept {
  // δ(dt) = v·dt + 0.5·a·dt². Plain arithmetic — no fma() call. Reference
  // profile demands identical results across x86/ARM/GPU; fma() is not
  // portably guaranteed to produce the same bit pattern on every backend,
  // so we expand the expression and accept the extra ULP.
  return v_max * dt + 0.5 * a_max * dt * dt;
}

bool compute_safe(double v_max,
                  double a_max,
                  double dt,
                  double buffer,
                  double skin,
                  double frontier) noexcept {
  // Defensive guards (R-M4-2). Any of the six inputs being non-finite,
  // negative, or (for dt) non-positive collapses the predicate to false.
  // This is what preserves I7 monotonicity on pathological inputs — if
  // safe(C[dt₂]) = false by defense, there is no obligation on dt₁.
  if (!std::isfinite(v_max) || !std::isfinite(a_max) || !std::isfinite(dt) ||
      !std::isfinite(buffer) || !std::isfinite(skin) || !std::isfinite(frontier)) {
    return false;
  }
  if (v_max < 0.0 || a_max < 0.0 || buffer < 0.0 || skin < 0.0 || frontier < 0.0) {
    return false;
  }
  if (dt <= 0.0) {
    return false;
  }
  const double delta = compute_displacement_bound(v_max, a_max, dt);
  if (!std::isfinite(delta)) {
    return false;
  }
  double threshold = buffer;
  if (skin < threshold) {
    threshold = skin;
  }
  if (frontier < threshold) {
    threshold = frontier;
  }
  return delta < threshold;
}

SafetyCertificate build_certificate(std::uint64_t cert_id, const CertificateInputs& in) noexcept {
  SafetyCertificate c;
  c.cert_id = cert_id;
  c.zone_id = in.zone_id;
  c.time_level = in.time_level;
  c.version = in.version;
  c.v_max_zone = in.v_max_zone;
  c.a_max_zone = in.a_max_zone;
  c.dt_candidate = in.dt_candidate;
  c.buffer_width = in.buffer_width;
  c.skin_remaining = in.skin_remaining;
  c.frontier_margin = in.frontier_margin;
  c.neighbor_valid_until_step = in.neighbor_valid_until_step;
  c.halo_valid_until_step = in.halo_valid_until_step;
  c.mode_policy_tag = in.mode_policy_tag;
  c.displacement_bound = compute_displacement_bound(in.v_max_zone, in.a_max_zone, in.dt_candidate);
  c.safe = compute_safe(in.v_max_zone,
                        in.a_max_zone,
                        in.dt_candidate,
                        in.buffer_width,
                        in.skin_remaining,
                        in.frontier_margin);
  return c;
}

}  // namespace tdmd::scheduler
