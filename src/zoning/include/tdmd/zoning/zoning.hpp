#pragma once

// SPEC: docs/specs/zoning/SPEC.md §2.1 (core types)
// Master spec: §12.3
// Exec pack: docs/development/m3_execution_pack.md T3.2
//
// Core value types produced / consumed by the zoning planner. This header
// has no dependency on state/, neighbor/, or runtime/ — zoning is a pure
// math module (SPEC §1.2) and its types stay at a low strata of the build
// graph so that perfmodel/, scheduler/ (M4+), and cli/ can all consume them
// without triggering a runtime rebuild cascade.

#include "tdmd/state/box.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace tdmd::zoning {

// Stable zone index across the whole box. 32-bit is enough for any v1
// deployment: a 10³ cube at 3 Å per zone → 10⁹ zones would overflow, but
// realistic TD systems stay well below 2³² (master spec §6.1). Widening to
// uint64 later is a source-compatible change.
using ZoneId = std::uint32_t;

// Scheme tag — which decomposition algorithm the planner selected. `Manual`
// is a v9+ stub (SPEC §2.1) used only when the user supplies an explicit
// plan; for v1 the planner always fills one of the first three.
enum class ZoningScheme : std::uint8_t {
  Linear1D,
  Decomp2D,
  Hilbert3D,
  Manual,
};

// Complete plan output. All fields are populated by `ZoningPlanner::plan()`
// in a single call — the plan is an immutable snapshot at that moment, not
// a live object. Copying is cheap (the only heap allocation is
// `canonical_order`, which tests frequently sort / iterate).
//
// The struct is layout-stable across CMake configurations because it has
// no platform-dependent types; the SPEC documents this as an implicit
// invariant (canonical_order bit-match is part of the reference profile
// contract in §4.3).
struct ZoningPlan {
  ZoningScheme scheme = ZoningScheme::Linear1D;

  // Zone counts per axis (x, y, z). Product equals canonical_order.size()
  // except for Hilbert3D where the padded walk may generate a slightly
  // larger internal array — canonical_order is always the filtered,
  // in-bounds sequence.
  std::array<std::uint32_t, 3> n_zones{0, 0, 0};

  // Physical zone dimensions in Å (same units as Box). Constrained to
  // `zone_size[i] >= cutoff + skin` — enforced by property tests §8.2.
  std::array<double, 3> zone_size{0.0, 0.0, 0.0};

  // N_min from Andreev eq. 35 / 43 or the Hilbert-3D empirical envelope.
  // Minimum number of zones one rank must hold to sustain a TD pipeline.
  std::uint64_t n_min_per_rank = 1;

  // floor(n_zones / n_min_per_rank) — ranks at which TD scales linearly.
  // Beyond this, Pattern 2 is recommended (M7+).
  std::uint64_t optimal_rank_count = 1;

  // Ordered zone traversal. Length == product(n_zones). Every ZoneId in
  // [0, product(n_zones)) appears exactly once (permutation property,
  // SPEC §4.3).
  std::vector<ZoneId> canonical_order;

  // Skin buffer carried on each zone face. Defaults to `skin` along each
  // axis; Production profile (§5.2) may grow this based on observed v_max.
  std::array<double, 3> buffer_width{0.0, 0.0, 0.0};

  // Copies of the input constraints so downstream consumers don't need to
  // track them separately.
  double cutoff = 0.0;
  double skin = 0.0;

  // Pattern-2 awareness (SPEC §7). Always nullopt in M3 (Pattern 1 only);
  // M7 populates when nested inside an `OuterSdCoordinator`.
  std::optional<tdmd::Box> subdomain_box;

  [[nodiscard]] std::uint64_t total_zones() const noexcept {
    return static_cast<std::uint64_t>(n_zones[0]) * n_zones[1] * n_zones[2];
  }
};

// Performance hint fed into `plan()`. M3 ships a default-constructed hint
// (all zeros); M4 scheduler will source real numbers from perfmodel's
// `HardwareProfile`. Shape is frozen here so M4/M7 callsites don't churn.
//
// Every field is advisory — the planner may ignore them without correctness
// consequences; they only affect heuristic branch choices (SPEC §3.4).
struct PerformanceHint {
  // Wall-time per force evaluation on a single atom, in seconds. Used to
  // estimate whether TD's peer-to-peer traffic would be CPU- or
  // bandwidth-bound.
  double cost_per_force_evaluation_seconds = 0.0;

  // Aggregate p2p bandwidth between ranks on the chosen interconnect.
  double bandwidth_peer_to_peer_bytes_per_sec = 0.0;

  // Bytes per atom in the temporal packet payload. Default 32 B matches
  // the position+type layout used by perfmodel Pattern 1 (M2 D-M2-6).
  double atom_record_size_bytes = 32.0;

  // Scheduler's preferred pipeline depth. Zoning uses this to rank
  // alternative schemes in borderline aspect ratios; it is **not**
  // a contract — the planner may choose a different K.
  std::uint32_t preferred_K_pipeline = 1;
};

}  // namespace tdmd::zoning
