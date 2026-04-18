// SPEC: docs/specs/scheduler/SPEC.md §8.3 (diagnostic dump event ring)
// Exec pack: docs/development/m4_execution_pack.md T4.10

#include "tdmd/scheduler/event_log.hpp"

#include <algorithm>

namespace tdmd::scheduler {

void EventLog::push(SchedulerEvent kind,
                    ZoneId zone_id,
                    TimeLevel time_level,
                    std::uint32_t count) noexcept {
  EventRecord& slot = ring_[head_];
  slot.timestamp = std::chrono::steady_clock::now();
  slot.kind = kind;
  slot.zone_id = zone_id;
  slot.time_level = time_level;
  slot.count = count;
  head_ = (head_ + 1) % kCapacity;
  if (count_ < kCapacity) {
    ++count_;
  }
}

std::vector<EventRecord> EventLog::snapshot() const {
  std::vector<EventRecord> out;
  out.reserve(count_);
  const std::size_t start = (head_ + kCapacity - count_) % kCapacity;
  for (std::size_t i = 0; i < count_; ++i) {
    const std::size_t idx = (start + i) % kCapacity;
    out.push_back(ring_[idx]);
  }
  return out;
}

std::vector<EventRecord> EventLog::snapshot_last(std::size_t n) const {
  const std::size_t take = std::min(n, count_);
  std::vector<EventRecord> out;
  out.reserve(take);
  // Walk from (head - take) mod capacity forward `take` slots.
  const std::size_t start = (head_ + kCapacity - take) % kCapacity;
  for (std::size_t i = 0; i < take; ++i) {
    const std::size_t idx = (start + i) % kCapacity;
    out.push_back(ring_[idx]);
  }
  return out;
}

}  // namespace tdmd::scheduler
