#pragma once

// SPEC: docs/specs/scheduler/SPEC.md §2.3, §5.1 (spatial_peers check)
// Master spec: §6.3
// Exec pack: docs/development/m4_execution_pack.md T4.5
//
// Spatial dependency DAG for the scheduler. Each zone carries a 64-bit
// `ZoneDepMask` naming which other zones it must keep consistent with — i.e.
// zones whose atoms can, within the current `r_c + r_skin` budget, reach
// into the reference zone. The scheduler uses this mask in T4.6's
// `select_ready_tasks` to gate progress by spatial-peer completion.
//
// M4 metric: two zones are dependents iff their grid centres are within
// `radius·(1+1e-9)` (unit-tolerant equality). For a uniform grid with
// `zone_size == radius`, this selects **face-adjacent** zones only —
// diagonal pairs are at distance `radius·√2 > radius·(1+ε)`. The 1e-9
// tolerance is there to absorb fp round-off on exact-equality corners; it
// is tight enough not to accidentally include edge/corner neighbours.
//
// Zone index convention: row-major `(x + nx·y + nx·ny·z)`. Tests construct
// plans matching this layout; the zoning module's default planners emit
// `canonical_order` that may permute this index (Hilbert3D does), but the
// DAG is built on the underlying grid coordinates, not the canonical walk.

#include "tdmd/scheduler/types.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace tdmd::zoning {
struct ZoningPlan;
}

namespace tdmd::scheduler {

// 64-bit bitset indexed by ZoneId. Master spec § 12.4 (dep_mask on ZoneTask)
// uses the same width; OQ-M4-1 documents the std::bitset escalation path
// once a subdomain exceeds 64 zones (M7+).
using ZoneDepMask = std::uint64_t;

// Unpack a flat ZoneId into (x, y, z) grid coordinates using row-major
// ordering `(x + nx·y + nx·ny·z)`. Public so tests can verify the mapping
// matches the one used inside compute_spatial_dependencies.
std::array<std::uint32_t, 3> unravel_zone_index(ZoneId id,
                                                std::array<std::uint32_t, 3> n_zones) noexcept;

// Build per-zone dependency masks. Throws std::runtime_error if total
// zone count exceeds 64 (see OQ-M4-1). `radius` is `cutoff + skin` from
// the plan. Returned vector is sized plan.total_zones(); bit `j` in
// entry `i` is set iff zones `i` and `j` are spatial peers (i≠j).
std::vector<ZoneDepMask> compute_spatial_dependencies(const tdmd::zoning::ZoningPlan& plan,
                                                      double radius);

}  // namespace tdmd::scheduler
