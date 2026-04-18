// SPEC: docs/specs/scheduler/SPEC.md §8.3
// Exec pack: docs/development/m4_execution_pack.md T4.8

#include "tdmd/scheduler/diagnostic_dump.hpp"

#include <sstream>

namespace tdmd::scheduler {

std::string to_string(SchedulerEvent e) {
  switch (e) {
    case SchedulerEvent::Initialize:
      return "Initialize";
    case SchedulerEvent::Shutdown:
      return "Shutdown";
    case SchedulerEvent::RefreshCertificates:
      return "RefreshCertificates";
    case SchedulerEvent::InvalidateCertificatesFor:
      return "InvalidateCertificatesFor";
    case SchedulerEvent::InvalidateAllCertificates:
      return "InvalidateAllCertificates";
    case SchedulerEvent::SelectReadyTasks:
      return "SelectReadyTasks";
    case SchedulerEvent::MarkComputing:
      return "MarkComputing";
    case SchedulerEvent::MarkCompleted:
      return "MarkCompleted";
    case SchedulerEvent::MarkPacked:
      return "MarkPacked";
    case SchedulerEvent::MarkInflight:
      return "MarkInflight";
    case SchedulerEvent::MarkCommitted:
      return "MarkCommitted";
    case SchedulerEvent::CommitCompleted:
      return "CommitCompleted";
    case SchedulerEvent::ReleaseCommitted:
      return "ReleaseCommitted";
    case SchedulerEvent::ZoneDataArrived:
      return "ZoneDataArrived";
    case SchedulerEvent::HaloArrived:
      return "HaloArrived";
    case SchedulerEvent::NeighborRebuildCompleted:
      return "NeighborRebuildCompleted";
    case SchedulerEvent::CertInvalidatedRollback:
      return "CertInvalidatedRollback";
    case SchedulerEvent::DeadlockFired:
      return "DeadlockFired";
  }
  return "Unknown";
}

namespace {

const char* state_name(std::size_t idx) {
  switch (idx) {
    case static_cast<std::size_t>(ZoneState::Empty):
      return "Empty";
    case static_cast<std::size_t>(ZoneState::ResidentPrev):
      return "ResidentPrev";
    case static_cast<std::size_t>(ZoneState::Ready):
      return "Ready";
    case static_cast<std::size_t>(ZoneState::Computing):
      return "Computing";
    case static_cast<std::size_t>(ZoneState::Completed):
      return "Completed";
    case static_cast<std::size_t>(ZoneState::PackedForSend):
      return "PackedForSend";
    case static_cast<std::size_t>(ZoneState::InFlight):
      return "InFlight";
    case static_cast<std::size_t>(ZoneState::Committed):
      return "Committed";
    default:
      return "Unknown";
  }
}

}  // namespace

std::string DiagnosticReport::to_string() const {
  std::ostringstream os;
  os << "deadlock: idle_for=" << idle_duration.count() << "ms"
     << " t_watchdog=" << t_watchdog.count() << "ms\n";
  os << "frontier_min=" << frontier_min << " frontier_max=" << frontier_max << '\n';
  os << "total_zones=" << total_zones << " tracked_retry_counters=" << tracked_retry_counters
     << '\n';

  os << "zones:";
  for (std::size_t i = 0; i < state_counts.size(); ++i) {
    os << ' ' << state_name(i) << '=' << state_counts[i];
  }
  os << '\n';

  os << "queues: ready=" << ready_queue_count << " inflight=" << inflight_queue_count << '\n';

  os << "events: count=" << recent_events.size() << '\n';
  // Print events oldest → newest with elapsed ms from the newest event.
  // (Ring buffer gives us chronological order already.)
  const auto anchor = recent_events.empty() ? std::chrono::steady_clock::time_point{}
                                            : recent_events.back().timestamp;
  for (const auto& e : recent_events) {
    const auto delta_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(anchor - e.timestamp).count();
    os << "  [-" << delta_ms << "ms] " << scheduler::to_string(e.kind);
    if (e.zone_id != 0xFFFFFFFFU) {
      os << " zone=" << e.zone_id;
    }
    os << " t=" << e.time_level;
    if (e.count != 0) {
      os << " n=" << e.count;
    }
    os << '\n';
  }
  os << "advice: check K_max, cert_source safety, neighbor rebuild cadence; verify no circular "
        "peer wait.\n";
  return os.str();
}

}  // namespace tdmd::scheduler
