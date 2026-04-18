#pragma once

// SPEC: docs/specs/scheduler/SPEC.md §5.2 (tie-break), §9 (queue semantics)
// Master spec: §6.7, §13.4
// Exec pack: docs/development/m4_execution_pack.md T4.6
//
// Minimal queue-support types for the M4 CausalWavefrontScheduler.
//
// The scheduler's ready_queue / blocked_queue / inflight_queue / completed_queue
// are *derivable* from ZoneMeta.state in M4 single-thread Pattern 1; we don't
// materialize them as persistent containers. What we DO materialize here is
// the priority ordering contract from SPEC §5.2:
//
//     Reference: (time_level_asc, canonical_zone_order_asc, version_asc)
//
// Exposed as a strict weak ordering functor so it can be unit-tested
// independently and reused by T4.7 (retry_queue ordering) and T4.10
// (determinism tests).

#include "tdmd/scheduler/types.hpp"

#include <cstddef>
#include <cstdint>

namespace tdmd::scheduler {

// One entry in the ready_queue priority-ordering decision. Carries the four
// fields that the Reference tie-break consults — the scheduler fills it from
// ZoneMeta + CertificateStore at select_ready_tasks() time.
struct TaskCandidate {
  ZoneId zone_id = 0;
  TimeLevel time_level = 0;
  std::size_t canonical_index = 0;  // position of zone in ZoningPlan::canonical_order
  Version version = 0;
  std::uint64_t cert_id = 0;  // retained for trace / ZoneTask.certificate_version
};

// Strict weak ordering implementing the SPEC §5.2 Reference tie-break.
// Stable with std::sort; fuzzer T4.10 relies on this.
struct ReferenceTaskCompare {
  [[nodiscard]] bool operator()(const TaskCandidate& a, const TaskCandidate& b) const noexcept {
    if (a.time_level != b.time_level) {
      return a.time_level < b.time_level;
    }
    if (a.canonical_index != b.canonical_index) {
      return a.canonical_index < b.canonical_index;
    }
    return a.version < b.version;
  }
};

}  // namespace tdmd::scheduler
