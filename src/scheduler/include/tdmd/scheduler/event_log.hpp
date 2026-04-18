#pragma once

// SPEC: docs/specs/scheduler/SPEC.md §8.3 (diagnostic dump event ring)
// Master spec: §6.6
// Exec pack: docs/development/m4_execution_pack.md T4.10 (+ OQ-M4-4 resolution)
//
// Fixed-capacity, allocation-free circular event buffer. Promoted out of
// CausalWavefrontScheduler at T4.10 so determinism tests can snapshot the
// full event history without reaching into scheduler internals.
//
// OQ-M4-4: capacity = 1024 events. DiagnosticReport (T4.8 / SPEC §8.3) still
// reports only the last 100 via `snapshot_last(100)` — the extra buffer is
// diagnostic headroom for tests and post-mortem tooling.

#include "tdmd/scheduler/diagnostic_dump.hpp"
#include "tdmd/scheduler/types.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tdmd::scheduler {

class EventLog {
public:
  // OQ-M4-4: 1024 entries. Diagnostic dump surfaces a tail of 100 by design.
  static constexpr std::size_t kCapacity = 1024;

  void push(SchedulerEvent kind,
            ZoneId zone_id = 0xFFFFFFFFU,
            TimeLevel time_level = 0,
            std::uint32_t count = 0) noexcept;

  // Oldest → newest copy of every live entry (up to `size()`).
  [[nodiscard]] std::vector<EventRecord> snapshot() const;

  // Oldest → newest copy of the last `n` live entries. `n` is clamped to
  // `size()`. Used by DiagnosticReport.recent_events (n=100) and by tests
  // that want a bounded comparison window.
  [[nodiscard]] std::vector<EventRecord> snapshot_last(std::size_t n) const;

  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] static constexpr std::size_t capacity() noexcept { return kCapacity; }

  void clear() noexcept {
    head_ = 0;
    count_ = 0;
  }

private:
  std::array<EventRecord, kCapacity> ring_{};
  std::size_t head_ = 0;   // next write slot
  std::size_t count_ = 0;  // live entries, capped at kCapacity
};

}  // namespace tdmd::scheduler
