#pragma once

// SPEC: docs/specs/scheduler/SPEC.md §7 (retry policy), §11.1 (D-M4-13)
// Master spec: §6.6, §13.4
// Exec pack: docs/development/m4_execution_pack.md T4.7
//
// RetryTracker — per-(zone, time_level) retry counter with a canonical
// ceiling. Deterministic by construction: increments happen only at the
// scheduler's cert-invalidation rollback path and counts reset on the
// zone's next successful Phase A (mark_completed) or on release into a
// fresh time step. There is NO random backoff; ordering is set by the
// scheduler's canonical task queue (SPEC §5.2, §7.2).
//
// D-M4-13: `max_retries_per_task = 3`. Exceeding the ceiling is a hard
// failure — the tracker throws `RetryExhaustedError` so the scheduler can
// propagate it to the engine's diagnostic dump path (T4.8).

#include "tdmd/scheduler/types.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace tdmd::scheduler {

class RetryExhaustedError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class RetryTracker {
public:
  static constexpr std::uint32_t kDefaultMaxRetries = 3;

  explicit RetryTracker(std::uint32_t max_retries = kDefaultMaxRetries) noexcept
      : max_retries_{max_retries} {}

  // Record a retry for (zone, level). Returns the new count. Throws
  // RetryExhaustedError if the new count strictly exceeds max_retries —
  // i.e. max_retries is the cap on the number of retries we tolerate, not
  // the number of attempts; count = 1 after the first retry, count = 3
  // after the third retry, count = 4 throws.
  std::uint32_t increment(ZoneId zone, TimeLevel level) {
    const auto new_count = ++counts_[key_of(zone, level)];
    if (new_count > max_retries_) {
      throw RetryExhaustedError("RetryTracker: retry count " + std::to_string(new_count) +
                                " exceeds max " + std::to_string(max_retries_) + " for zone " +
                                std::to_string(zone) + " at time_level " + std::to_string(level));
    }
    return new_count;
  }

  // Remove the counter for (zone, level). Called by the scheduler after a
  // clean Phase A completion — the retry budget is per-(zone, level), and
  // a successful compute at that level consumes no further retry slots.
  void reset(ZoneId zone, TimeLevel level) noexcept { counts_.erase(key_of(zone, level)); }

  // Remove every counter belonging to `zone`. Called on the zone's release
  // back to Empty so that the next cycle starts with a fresh budget.
  void reset_for(ZoneId zone) noexcept {
    for (auto it = counts_.begin(); it != counts_.end();) {
      if (static_cast<ZoneId>(it->first >> 32U) == zone) {
        it = counts_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Query current retry count for (zone, level). Returns 0 if untracked.
  [[nodiscard]] std::uint32_t count_of(ZoneId zone, TimeLevel level) const noexcept {
    const auto it = counts_.find(key_of(zone, level));
    return it == counts_.end() ? 0U : it->second;
  }

  [[nodiscard]] std::uint32_t max_retries() const noexcept { return max_retries_; }

  // Total number of tracked (zone, level) keys — test / telemetry only.
  [[nodiscard]] std::size_t tracked_count() const noexcept { return counts_.size(); }

  // Drop all counters. Scheduler::shutdown uses this to reset state.
  void clear() noexcept { counts_.clear(); }

private:
  // Pack (zone, level) into a single 64-bit key. ZoneId is 32-bit and
  // TimeLevel is 64-bit in M4, but in practice the OQ-M4-1 ceiling of 64
  // zones leaves zone in the high 32 bits and we truncate the time_level
  // to 32 bits — safe for 2^32 steps, which is far beyond the anchor-test
  // horizon. If we ever lift this, `Key` becomes a real struct.
  static std::uint64_t key_of(ZoneId zone, TimeLevel level) noexcept {
    return (static_cast<std::uint64_t>(zone) << 32U) |
           (static_cast<std::uint64_t>(level) & 0xFFFF'FFFFULL);
  }

  std::unordered_map<std::uint64_t, std::uint32_t> counts_;
  std::uint32_t max_retries_;
};

}  // namespace tdmd::scheduler
