// SPEC: docs/specs/scheduler/SPEC.md §11.1
// Exec pack: docs/development/m4_execution_pack.md T4.2, D-M4-3, D-M4-10
//
// Reference-profile policy construction. The tag `mode_policy_tag` is a
// compile-time-ish fingerprint — in M4 a single constant distinguishes
// "Fp64ReferenceBuild × Reference ExecProfile" from the unimplemented
// production / fast profiles. The concrete value is arbitrary (the
// scheduler only cares about equality), fixed here for reproducibility.

#include "tdmd/scheduler/policy.hpp"

#include <stdexcept>

namespace tdmd::scheduler {

namespace {
// "TDMD_REF" packed into 8 bytes — memorable fingerprint for the Reference
// policy. Any scheduler / certificate whose mode_policy_tag doesn't match
// this will be rejected at validation time (T4.3+).
constexpr std::uint64_t kReferencePolicyTag = 0x544D44'5F52'4546ULL;  // "TDMD_REF"
}  // namespace

SchedulerPolicy PolicyFactory::for_reference() {
  SchedulerPolicy p;
  p.k_max_pipeline_depth = 1;     // D-M4-1
  p.max_tasks_per_iteration = 1;  // single-thread in M4
  p.use_canonical_tie_break = true;
  p.allow_task_stealing = false;
  p.allow_adaptive_buffer = false;
  p.deterministic_reduction_cert = true;
  p.t_watchdog = std::chrono::milliseconds{30'000};  // D-M4-7
  p.max_retries_per_task = 3;                        // D-M4-13
  p.exponential_backoff = false;
  p.two_phase_commit = true;
  p.mode_policy_tag = kReferencePolicyTag;
  return p;
}

SchedulerPolicy PolicyFactory::for_production() {
  throw std::logic_error{
      "scheduler::PolicyFactory::for_production is not implemented in M4 "
      "(D-M4-3); Production ExecProfile lands in M6+"};
}

SchedulerPolicy PolicyFactory::for_fast_experimental() {
  throw std::logic_error{
      "scheduler::PolicyFactory::for_fast_experimental is not implemented in "
      "M4 (D-M4-3); FastExperimental ExecProfile lands in M8+"};
}

}  // namespace tdmd::scheduler
