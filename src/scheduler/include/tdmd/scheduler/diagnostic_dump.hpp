#pragma once

// SPEC: docs/specs/scheduler/SPEC.md §8 (deadlock detection), §8.3 (dump)
// Master spec: §6.6
// Exec pack: docs/development/m4_execution_pack.md T4.8
//
// DiagnosticReport — what a deadlocked scheduler snapshots before it
// throws. Structured so tests can assert individual fields and so the
// operator-facing ToString() stays grep-friendly.
//
// Event ring buffer: the scheduler records a SchedulerEvent every time a
// public event handler or state transition runs. On deadlock, the last
// kEventRingCapacity (100) events ship with the report so the operator
// can see what was happening before the stall.

#include "tdmd/scheduler/types.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace tdmd::scheduler {

class DeadlockError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

enum class SchedulerEvent : std::uint8_t {
  Initialize,
  Shutdown,
  RefreshCertificates,
  InvalidateCertificatesFor,
  InvalidateAllCertificates,
  SelectReadyTasks,
  MarkComputing,
  MarkCompleted,
  MarkPacked,
  MarkInflight,
  MarkCommitted,
  CommitCompleted,
  ReleaseCommitted,
  ZoneDataArrived,
  HaloArrived,
  NeighborRebuildCompleted,
  CertInvalidatedRollback,
  DeadlockFired,
};

[[nodiscard]] std::string to_string(SchedulerEvent e);

struct EventRecord {
  std::chrono::steady_clock::time_point timestamp{};
  SchedulerEvent kind{SchedulerEvent::Initialize};
  // 0xFFFFFFFF when the event is zone-agnostic (e.g. RefreshCertificates).
  ZoneId zone_id{0xFFFFFFFFU};
  TimeLevel time_level{0};
  // Auxiliary count: batch size for SelectReadyTasks / CommitCompleted,
  // affected-zones count for NeighborRebuildCompleted. Zero otherwise.
  std::uint32_t count{0};
};

struct DiagnosticReport {
  // Zone state histogram — index by static_cast<std::size_t>(ZoneState).
  // 8 slots covers every enumerator in zone_state.hpp.
  std::array<std::size_t, 8> state_counts{};
  std::size_t ready_queue_count{0};
  std::size_t inflight_queue_count{0};

  TimeLevel frontier_min{0};
  TimeLevel frontier_max{0};

  std::chrono::milliseconds idle_duration{0};
  std::chrono::milliseconds t_watchdog{0};

  std::size_t total_zones{0};
  std::size_t tracked_retry_counters{0};

  // Events ordered oldest → newest. At most kEventRingCapacity entries.
  std::vector<EventRecord> recent_events;

  // Operator-facing single-string dump. Multi-line, grep-able labels:
  //   "deadlock:", "frontier_min=", "frontier_max=", "zones:",
  //   "queues:", "events:".
  [[nodiscard]] std::string to_string() const;
};

}  // namespace tdmd::scheduler
