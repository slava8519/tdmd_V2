// SPEC: docs/specs/zoning/SPEC.md §3.3, §4.2
// Exec pack: docs/development/m3_execution_pack.md T3.5
//
// Known open question — OQ-M3-6: exec pack line 313 and SPEC §8.3 row
// 3 both claim `3D Hilbert, 16³ → target n_opt ≈ 64`; the §3.3 formula
// `n_opt = min(Nx,Ny,Nz)/4` gives 4 for 16³ and only reaches 64 at
// 256³. Likely a SPEC §8.3 row-label confusion — same fix pattern as
// OQ-M3-5. T3.5 implements §3.3 as written; T3.7 resolves via SPEC delta.

#include "tdmd/zoning/hilbert3d.hpp"

#include "tdmd/zoning/hilbert.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <string>

namespace tdmd::zoning {

namespace {

std::uint32_t n_zones_along(double length, double cutoff, double skin) noexcept {
  const double w = cutoff + skin;
  if (w <= 0.0 || length < w) {
    return 0;
  }
  return static_cast<std::uint32_t>(std::floor(length / w));
}

// Smallest power of 2 that is ≥ n and ≥ 1. For n=0 returns 1 (pad
// nothing meaningfully; the planner will have thrown earlier on
// unusably small boxes).
std::uint32_t next_pow2(std::uint32_t n) noexcept {
  if (n <= 1) {
    return 1;
  }
  return 1u << (32 - std::countl_zero(n - 1u));
}

int bits_for(std::uint32_t pad) noexcept {
  if (pad <= 1) {
    return 0;
  }
  return 32 - std::countl_zero(pad - 1u);
}

}  // namespace

ZoningPlan Hilbert3DZoningPlanner::plan(const tdmd::Box& box,
                                        double cutoff,
                                        double skin,
                                        std::uint64_t n_ranks,
                                        const PerformanceHint& /*hint*/) const {
  if (cutoff <= 0.0 || skin < 0.0) {
    throw ZoningPlanError("Hilbert3D: cutoff must be > 0 and skin >= 0");
  }
  const std::array<std::uint32_t, 3> n = {
      n_zones_along(box.lx(), cutoff, skin),
      n_zones_along(box.ly(), cutoff, skin),
      n_zones_along(box.lz(), cutoff, skin),
  };
  const std::uint64_t total = static_cast<std::uint64_t>(n[0]) * n[1] * n[2];
  if (total < 2) {
    throw ZoningPlanError(
        "Hilbert3D: box admits fewer than 2 zones total — use SD-vacuum or "
        "enlarge the box");
  }

  const std::uint32_t pad_max = std::max({n[0], n[1], n[2]});
  const std::uint32_t pad = next_pow2(pad_max);
  const int bits = bits_for(pad);
  // bits==0 happens only for pad==1 i.e. total==1; caught above.

  ZoningPlan plan;
  plan.scheme = ZoningScheme::Hilbert3D;
  plan.n_zones = n;

  const double w = cutoff + skin;
  plan.zone_size[0] = n[0] > 0 ? box.lx() / n[0] : box.lx();
  plan.zone_size[1] = n[1] > 0 ? box.ly() / n[1] : box.ly();
  plan.zone_size[2] = n[2] > 0 ? box.lz() / n[2] : box.lz();
  for (int ax = 0; ax < 3; ++ax) {
    if (plan.zone_size[static_cast<std::size_t>(ax)] < w && n[static_cast<std::size_t>(ax)] >= 1) {
      plan.zone_size[static_cast<std::size_t>(ax)] = w;
    }
  }

  // N_min per §3.3 formula: `4·max(Nx·Ny, Ny·Nz, Nx·Nz)`. For a cubic
  // N×N×N, this is 4N² — inside the empirical [3N², 6N²] envelope
  // used by the property tests.
  const std::uint64_t xy = static_cast<std::uint64_t>(n[0]) * n[1];
  const std::uint64_t yz = static_cast<std::uint64_t>(n[1]) * n[2];
  const std::uint64_t xz = static_cast<std::uint64_t>(n[0]) * n[2];
  const std::uint64_t face_max = std::max({xy, yz, xz});
  plan.n_min_per_rank = 4ull * face_max;
  plan.optimal_rank_count = total / plan.n_min_per_rank;
  if (plan.optimal_rank_count == 0) {
    plan.optimal_rank_count = 1;
  }

  // Walk the padded Hilbert curve. For each index d in [0, pad³),
  // decode (hx, hy, hz); if in-box, push the lex flat ZoneId.
  const std::uint64_t pad_total = static_cast<std::uint64_t>(pad) * pad * pad;
  plan.canonical_order.reserve(static_cast<std::size_t>(total));
  const std::uint64_t stride_y = n[0];
  const std::uint64_t stride_z = static_cast<std::uint64_t>(n[0]) * n[1];
  for (std::uint64_t d = 0; d < pad_total; ++d) {
    std::uint32_t hx = 0, hy = 0, hz = 0;
    hilbert::d_to_xyz(static_cast<std::uint32_t>(d), bits, hx, hy, hz);
    if (hx < n[0] && hy < n[1] && hz < n[2]) {
      const std::uint64_t flat = hx + hy * stride_y + hz * stride_z;
      plan.canonical_order.push_back(static_cast<ZoneId>(flat));
    }
  }

  plan.buffer_width = {skin, skin, skin};
  plan.cutoff = cutoff;
  plan.skin = skin;

  (void) n_ranks;
  return plan;
}

ZoningPlan Hilbert3DZoningPlanner::plan_with_scheme(const tdmd::Box& box,
                                                    double cutoff,
                                                    double skin,
                                                    ZoningScheme forced_scheme,
                                                    const PerformanceHint& hint) const {
  if (forced_scheme != ZoningScheme::Hilbert3D) {
    throw ZoningPlanError("Hilbert3DZoningPlanner cannot satisfy non-Hilbert3D forced scheme");
  }
  return plan(box, cutoff, skin, /*n_ranks=*/1, hint);
}

std::uint64_t Hilbert3DZoningPlanner::estimate_n_min(ZoningScheme scheme,
                                                     const tdmd::Box& box,
                                                     double cutoff,
                                                     double skin) const {
  if (scheme != ZoningScheme::Hilbert3D) {
    throw ZoningPlanError("Hilbert3DZoningPlanner::estimate_n_min called with other scheme");
  }
  const std::uint32_t nx = n_zones_along(box.lx(), cutoff, skin);
  const std::uint32_t ny = n_zones_along(box.ly(), cutoff, skin);
  const std::uint32_t nz = n_zones_along(box.lz(), cutoff, skin);
  const std::uint64_t xy = static_cast<std::uint64_t>(nx) * ny;
  const std::uint64_t yz = static_cast<std::uint64_t>(ny) * nz;
  const std::uint64_t xz = static_cast<std::uint64_t>(nx) * nz;
  return 4ull * std::max({xy, yz, xz});
}

std::uint64_t Hilbert3DZoningPlanner::estimate_optimal_ranks(ZoningScheme scheme,
                                                             const tdmd::Box& box,
                                                             double cutoff,
                                                             double skin) const {
  if (scheme != ZoningScheme::Hilbert3D) {
    throw ZoningPlanError(
        "Hilbert3DZoningPlanner::estimate_optimal_ranks called with other scheme");
  }
  const std::uint32_t nx = n_zones_along(box.lx(), cutoff, skin);
  const std::uint32_t ny = n_zones_along(box.ly(), cutoff, skin);
  const std::uint32_t nz = n_zones_along(box.lz(), cutoff, skin);
  const std::uint64_t total = static_cast<std::uint64_t>(nx) * ny * nz;
  const std::uint64_t nmin = estimate_n_min(scheme, box, cutoff, skin);
  if (nmin == 0) {
    return 1;
  }
  const auto opt = total / nmin;
  return opt == 0 ? 1ull : opt;
}

bool Hilbert3DZoningPlanner::validate_manual_plan(const ZoningPlan& /*plan*/,
                                                  std::string& reason_if_invalid) const {
  reason_if_invalid = "validate_manual_plan: not implemented in M3 (SPEC §10 OQ-5)";
  return false;
}

}  // namespace tdmd::zoning
