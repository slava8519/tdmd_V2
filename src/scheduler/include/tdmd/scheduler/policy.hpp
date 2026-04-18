#pragma once

// SPEC: docs/specs/scheduler/SPEC.md §11 (policy plumbing)
// Exec pack: docs/development/m4_execution_pack.md T4.2, D-M4-3, D-M4-7
//
// SchedulerPolicy is compile-time-set for a given ExecProfile. In M4 only
// `for_reference()` has a live body; the Production and FastExperimental
// factories are declared (so callsites compile) but throw — the profiles
// don't land until M6+ when there's a GPU baseline worth tuning.

#include <chrono>
#include <cstdint>
#include <stdexcept>

namespace tdmd::scheduler {

struct SchedulerPolicy {
  // Frontier control:
  std::uint32_t k_max_pipeline_depth = 1;     // D-M4-1: K=1 in M4
  std::uint32_t max_tasks_per_iteration = 1;  // single-thread in M4

  // Priority:
  bool use_canonical_tie_break = true;  // Reference: always true
  bool allow_task_stealing = false;     // Fast only (M8+)

  // Certificate:
  bool allow_adaptive_buffer = false;        // Production/Fast only
  bool deterministic_reduction_cert = true;  // Reference: required

  // Watchdog:
  std::chrono::milliseconds t_watchdog{30'000};  // D-M4-7: 30s default

  // Retry (D-M4-13):
  std::uint32_t max_retries_per_task = 3;
  bool exponential_backoff = false;  // Reference: canonical counts

  // Commit:
  bool two_phase_commit = true;  // always — documented invariant

  // Fingerprint of (BuildFlavor × ExecProfile). SafetyCertificate::mode_policy_tag
  // must equal this; mismatches reject certificates (SPEC §11.2).
  std::uint64_t mode_policy_tag = 0;
};

class PolicyFactory {
public:
  static SchedulerPolicy for_reference();

  // Stubs until M6+ (D-M4-3). Throwing keeps the signatures compilable by
  // call-site scaffolding without silently promoting an unsupported policy.
  [[noreturn]] static SchedulerPolicy for_production();
  [[noreturn]] static SchedulerPolicy for_fast_experimental();
};

}  // namespace tdmd::scheduler
