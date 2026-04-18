#pragma once

// SPEC: docs/specs/scheduler/SPEC.md §2.1 (core types), §3 (state machine),
//       §4 (safety certificate)
// Master spec: §12.4, §13.4
// Exec pack: docs/development/m4_execution_pack.md T4.2
//
// Core value types for the scheduler module. Like zoning/, this header has
// no dependency on state/, neighbor/, or runtime/ — the scheduler is a
// pure policy+data module. Concrete behaviour (state-machine enforcement,
// safety-certificate math, task selection) ships in T4.3–T4.7.

#include <array>
#include <chrono>
#include <cstdint>

namespace tdmd::scheduler {

// Stable zone index across the whole subdomain. Re-declared here (rather
// than imported from tdmd::zoning) to keep the scheduler module free of a
// build-time dependency on zoning/ — the scheduler reads a ZoningPlan at
// runtime but its public API operates purely on IDs. The two ZoneId
// aliases are guaranteed to be the same underlying type by a static_assert
// where the boundaries meet (T4.5).
using ZoneId = std::uint32_t;

// Monotonic per-zone time step counter. 64-bit to accommodate ns-scale
// trajectories: 10¹⁸ steps at 1 fs/step = 10³ seconds of wall-clock
// simulation before overflow, far beyond any practical run.
using TimeLevel = std::uint64_t;

// Monotonic per-zone state revision. Bumped on every mark_completed
// (§6.1 Phase A). Used by the certificate store and temporal packets to
// discard stale data.
using Version = std::uint64_t;

// Zone lifecycle per SPEC §3.1 (legal transitions) and master spec §6.2.
// In M4 single-rank Pattern 1, PackedForSend / InFlight become no-ops for
// internal zones (all zones are internal until M5 multi-rank), but the
// states are defined here so the state-machine table in T4.4 covers the
// full shape. See D-M4-6.
enum class ZoneState : std::uint8_t {
  Empty,          // memory allocated, no data
  ResidentPrev,   // holds data from the previous step
  Ready,          // cert issued, deps satisfied
  Computing,      // force + integrate in flight
  Completed,      // compute done, awaiting commit (Phase A)
  PackedForSend,  // packed into a TemporalPacket (M5+)
  InFlight,       // MPI/NCCL transfer in progress (M5+)
  Committed,      // acknowledged by receiver, releasable
};

// Safety certificate per SPEC §4.1 and master spec §6.4. Built by the
// scheduler from (v_max_zone, a_max_zone) reductions, neighbor skin
// remaining, frontier margin, and the candidate dt. `safe` is the
// precomputed result of the predicate
//
//     δ(dt) = v·dt + 0.5·a·dt²
//     safe ⇔ δ < min(buffer_width, skin_remaining, frontier_margin)
//
// T4.3 implements build()/safe() and proves I7 monotonicity:
//   safe(C[dt_hi]) ∧ dt_lo < dt_hi  ⟹  safe(C[dt_lo])
struct SafetyCertificate {
  bool safe = false;
  std::uint64_t cert_id = 0;
  ZoneId zone_id = 0;
  TimeLevel time_level = 0;
  Version version = 0;

  double v_max_zone = 0.0;  // Å / ps (metal units)
  double a_max_zone = 0.0;  // Å / ps²

  double dt_candidate = 0.0;        // ps
  double displacement_bound = 0.0;  // Å — δ(dt_candidate)
  double buffer_width = 0.0;        // Å — zoning-owned skin buffer
  double skin_remaining = 0.0;      // Å — neighbor-owned unused skin
  double frontier_margin = 0.0;     // Å — K_max · dt - (t - frontier_min)·dt

  // Temporal validity windows. Certificate invalid beyond either.
  TimeLevel neighbor_valid_until_step = 0;
  TimeLevel halo_valid_until_step = 0;  // ∞-equivalent in Pattern 1

  // Compile-time policy fingerprint (BuildFlavor + ExecProfile). In M4
  // Reference it is a single constant (set by PolicyFactory::for_reference).
  // Scheduler rejects certificates whose mode_policy_tag mismatches its own.
  std::uint64_t mode_policy_tag = 0;
};

// Task ticket handed to the runtime for one (zone, time_level, version)
// triple. `dep_mask` encodes the bit set of spatial peers whose completion
// is required — 64-bit covers up to 64 zones per subdomain (OQ-M4-1
// documents the escalation path to a std::bitset alias if widened).
struct ZoneTask {
  ZoneId zone_id = 0;
  TimeLevel time_level = 0;
  Version local_state_version = 0;

  std::uint64_t dep_mask = 0;
  std::uint64_t certificate_version = 0;

  std::uint32_t priority = 0;
  std::uint32_t mode_policy_tag = 0;
};

}  // namespace tdmd::scheduler
