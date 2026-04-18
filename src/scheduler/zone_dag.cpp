// SPEC: docs/specs/scheduler/SPEC.md §2.3
// Master spec: §6.3
// Exec pack: docs/development/m4_execution_pack.md T4.5

#include "tdmd/scheduler/zone_dag.hpp"

#include "tdmd/zoning/zoning.hpp"

#include <cstdint>
#include <stdexcept>

// Scheduler operates on scheduler::ZoneId; zoning emits zoning::ZoneId.
// M4 keeps them at the same underlying type so masks can round-trip both
// sides without a cast noise in every function.
static_assert(std::is_same_v<tdmd::scheduler::ZoneId, tdmd::zoning::ZoneId>,
              "scheduler::ZoneId and zoning::ZoneId must share the same underlying type in M4");

namespace tdmd::scheduler {

std::array<std::uint32_t, 3> unravel_zone_index(ZoneId id,
                                                std::array<std::uint32_t, 3> n_zones) noexcept {
  const std::uint32_t nx = n_zones[0];
  const std::uint32_t ny = n_zones[1];
  const std::uint32_t plane = nx * ny;
  const std::uint32_t z = plane == 0 ? 0 : (id / plane);
  const std::uint32_t r = plane == 0 ? id : (id % plane);
  const std::uint32_t y = nx == 0 ? 0 : (r / nx);
  const std::uint32_t x = nx == 0 ? 0 : (r % nx);
  return {x, y, z};
}

std::vector<ZoneDepMask> compute_spatial_dependencies(const tdmd::zoning::ZoningPlan& plan,
                                                      double radius) {
  const auto total = plan.total_zones();
  if (total > 64) {
    throw std::runtime_error(
        "compute_spatial_dependencies: >64 zones not supported in M4 "
        "(see OQ-M4-1); got " +
        std::to_string(total));
  }

  std::vector<ZoneDepMask> masks(total, 0);
  if (total < 2) {
    return masks;
  }

  // Eps on the squared distance: scale to the radius magnitude so we admit
  // exact-equality face-adjacent pairs (centre distance == radius at the
  // minimum zoning constraint) without also admitting edge-diagonals at
  // radius·√2.
  const double r2 = radius * radius;
  const double r2_eps = r2 * 1e-9;

  const auto& s = plan.zone_size;

  for (ZoneId a = 0; a + 1 < total; ++a) {
    const auto [ax, ay, az] = unravel_zone_index(a, plan.n_zones);
    for (ZoneId b = a + 1; b < total; ++b) {
      const auto [bx, by, bz] = unravel_zone_index(b, plan.n_zones);
      const double dx = (static_cast<double>(ax) - static_cast<double>(bx)) * s[0];
      const double dy = (static_cast<double>(ay) - static_cast<double>(by)) * s[1];
      const double dz = (static_cast<double>(az) - static_cast<double>(bz)) * s[2];
      const double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 <= r2 + r2_eps) {
        masks[a] |= (static_cast<ZoneDepMask>(1) << b);
        masks[b] |= (static_cast<ZoneDepMask>(1) << a);
      }
    }
  }
  return masks;
}

}  // namespace tdmd::scheduler
