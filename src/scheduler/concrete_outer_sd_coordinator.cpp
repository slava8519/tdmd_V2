// SPEC: docs/specs/scheduler/SPEC.md §2.4 (OC-1..OC-6), §4.6 (HA-1..HA-5)
// Master spec: §12.7a
// Exec pack: docs/development/m7_execution_pack.md T7.6

#include "tdmd/scheduler/concrete_outer_sd_coordinator.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace tdmd::scheduler {

void ConcreteOuterSdCoordinator::initialize(const SubdomainGrid& grid, std::uint32_t k_max) {
  if (k_max == 0) {
    hard_error("OuterSdCoordinator::initialize — k_max must be ≥ 1");
  }
  if (grid.subdomain_boxes.size() != grid.total_subdomains() ||
      grid.rank_of_subdomain.size() != grid.total_subdomains()) {
    hard_error(
        "OuterSdCoordinator::initialize — grid arity mismatch "
        "(subdomain_boxes / rank_of_subdomain vs n_subdomains product)");
  }
  grid_ = grid;
  k_max_ = k_max;
  local_registry_.clear();
  peer_archives_.clear();
  zone_peer_deps_.clear();
  watch_.clear();
  stall_reports_.clear();
  global_frontier_min_ = 0;
  global_frontier_max_ = 0;
  next_seq_ = 0;
  snapshot_too_old_total_ = 0;
  register_collisions_total_ = 0;
}

bool ConcreteOuterSdCoordinator::can_advance_boundary_zone(ZoneId local_zone,
                                                           TimeLevel target_level) {
  // OC-1: idempotent + non-blocking. No mutation of archive state.
  // The watchdog book-keeping side-effects are limited to recording the
  // first time we observed a wait — the predicate result is unchanged.
  const auto it = zone_peer_deps_.find(local_zone);
  if (it == zone_peer_deps_.end() || it->second.empty()) {
    // No registered peer dependencies → vacuously satisfiable. T7.7
    // promotes "no deps" into a real "interior zone" flag; for now this
    // keeps Pattern 1 compatible.
    return true;
  }
  if (target_level == 0) {
    // Step 0 has no peer history — initial state is trivially advanceable.
    return true;
  }
  const TimeLevel needed_level = target_level - 1;
  bool all_present = true;
  for (const auto& peer : it->second) {
    const auto archive_it = peer_archives_.find(peer);
    if (archive_it == peer_archives_.end() ||
        find_level_slot(archive_it->second, needed_level) == kNpos) {
      note_waiting(local_zone, peer, target_level);
      all_present = false;
    } else {
      clear_waiting(local_zone, peer);
    }
  }
  return all_present;
}

void ConcreteOuterSdCoordinator::register_boundary_snapshot(ZoneId local_zone,
                                                            TimeLevel level,
                                                            const HaloSnapshot& snap) {
  // OC-2: exactly one snapshot per (local_zone, level).
  const auto key = std::make_pair(local_zone, level);
  const auto [_, inserted] = local_registry_.try_emplace(key, snap);
  if (!inserted) {
    register_collisions_total_++;
    if (mode_ == Mode::kReference) {
      hard_error(
          "OuterSdCoordinator::register_boundary_snapshot — HA-2: "
          "double registration for (local_zone, level)");
    }
    // Production / Fast: keep existing snapshot, log advisory.
    return;
  }
  // Note: the SPEC's archive stores **incoming peer** snapshots; outgoing
  // local snapshots are tracked here only for HA-2 enforcement and are
  // packed for transport by the comm backend (T7.5/T7.7).
}

std::optional<HaloSnapshot> ConcreteOuterSdCoordinator::fetch_peer_snapshot(
    std::uint32_t peer_subdomain,
    ZoneId peer_zone,
    TimeLevel level) {
  // OC-4: non-blocking; returns nullopt on miss.
  const PeerKey key{peer_subdomain, peer_zone};
  const auto archive_it = peer_archives_.find(key);
  if (archive_it == peer_archives_.end() || archive_it->second.slots.empty()) {
    return std::nullopt;  // HA-4 normal stall — peer hasn't arrived yet.
  }
  const auto& slots = archive_it->second.slots;
  const TimeLevel oldest_level = slots.front().snap.time_level;
  const TimeLevel newest_level = slots.back().snap.time_level;
  if (level < oldest_level) {
    // HA-3: too old — peer level evicted. Telemetry counter + nullopt.
    snapshot_too_old_total_++;
    return std::nullopt;
  }
  if (level > newest_level) {
    return std::nullopt;  // HA-4: peer hasn't reached this level yet.
  }
  const std::size_t slot = find_level_slot(archive_it->second, level);
  if (slot == kNpos) {
    return std::nullopt;
  }
  archive_it->second.slots[slot].use_count++;
  return archive_it->second.slots[slot].snap;
}

void ConcreteOuterSdCoordinator::check_stall_boundaries(std::chrono::milliseconds t_stall_max) {
  // OC-6: independent of inner deadlock detector. We walk recorded waits
  // and emit reports for any that have exceeded the threshold.
  const auto now = std::chrono::steady_clock::now();
  for (const auto& [key, watch] : watch_) {
    if (!watch.active) {
      continue;
    }
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - watch.waiting_since);
    if (elapsed > t_stall_max) {
      stall_reports_.push_back(OuterStallReport{
          /*local_zone=*/key.first,
          /*waiting_for_level=*/watch.waiting_for_level,
          /*peer=*/watch.peer,
          /*waiting_since=*/watch.waiting_since,
      });
    }
  }
}

void ConcreteOuterSdCoordinator::archive_peer_snapshot(HaloSnapshot snap) {
  // SPEC §4.6 lifecycle: assign deterministic received_seq, then insert
  // into the per-(peer_subdomain, peer_zone) ring buffer.
  const PeerKey key{snap.source_subdomain_id, snap.source_zone_id};
  snap.received_seq = next_seq_++;
  insert_into_archive(peer_archives_[key], std::move(snap));
}

void ConcreteOuterSdCoordinator::register_zone_peer_dependency(ZoneId local_zone, PeerKey peer) {
  auto& peers = zone_peer_deps_[local_zone];
  if (std::find(peers.begin(), peers.end(), peer) == peers.end()) {
    peers.push_back(peer);
    std::sort(peers.begin(), peers.end());  // determinism in iteration
  }
}

bool ConcreteOuterSdCoordinator::release_snapshot(std::uint32_t peer_subdomain,
                                                  ZoneId peer_zone,
                                                  TimeLevel level) {
  const PeerKey key{peer_subdomain, peer_zone};
  const auto it = peer_archives_.find(key);
  if (it == peer_archives_.end()) {
    return false;
  }
  const std::size_t slot = find_level_slot(it->second, level);
  if (slot == kNpos) {
    return false;
  }
  if (it->second.slots[slot].use_count == 0) {
    return false;
  }
  it->second.slots[slot].use_count--;
  return true;
}

void ConcreteOuterSdCoordinator::set_global_frontier(TimeLevel new_min, TimeLevel new_max) {
  // OC-5: monotonic non-decreasing.
  if (new_min < global_frontier_min_ || new_max < global_frontier_max_) {
    hard_error(
        "OuterSdCoordinator::set_global_frontier — OC-5 violation: "
        "frontier moved backwards");
  }
  if (new_max < new_min) {
    hard_error("OuterSdCoordinator::set_global_frontier — max < min");
  }
  global_frontier_min_ = new_min;
  global_frontier_max_ = new_max;
}

std::vector<OuterStallReport> ConcreteOuterSdCoordinator::drain_stall_reports() {
  std::vector<OuterStallReport> out;
  out.swap(stall_reports_);
  return out;
}

std::size_t ConcreteOuterSdCoordinator::peer_slot_count(const PeerKey& peer) const {
  const auto it = peer_archives_.find(peer);
  return it == peer_archives_.end() ? 0 : it->second.slots.size();
}

std::size_t ConcreteOuterSdCoordinator::find_level_slot(const PeerArchive& arch,
                                                        TimeLevel level) const {
  for (std::size_t i = 0; i < arch.slots.size(); ++i) {
    if (arch.slots[i].snap.time_level == level) {
      return i;
    }
  }
  return kNpos;
}

void ConcreteOuterSdCoordinator::insert_into_archive(PeerArchive& arch, HaloSnapshot snap) {
  // HA-1: ring capacity. If at capacity, evict oldest (HA-5 gated).
  if (arch.slots.size() >= static_cast<std::size_t>(k_max_)) {
    if (arch.slots.front().use_count != 0) {
      hard_error(
          "OuterSdCoordinator: HA-5 violation — eviction blocked by "
          "outstanding fetch on oldest slot");
    }
    arch.slots.erase(arch.slots.begin());
  }
  // HA-2 already enforced for incoming peer archive: reject duplicate
  // (peer_zone, level) pair.
  for (const auto& existing : arch.slots) {
    if (existing.snap.time_level == snap.time_level) {
      register_collisions_total_++;
      if (mode_ == Mode::kReference) {
        hard_error(
            "OuterSdCoordinator: HA-2 — duplicate peer snapshot at same "
            "time_level");
      }
      return;
    }
  }
  // Maintain ascending order by time_level. New levels are typically
  // monotonic, so back-insert is the common case.
  if (arch.slots.empty() || arch.slots.back().snap.time_level < snap.time_level) {
    arch.slots.push_back(ArchiveSlot{std::move(snap), 0u});
  } else {
    auto pos = std::upper_bound(
        arch.slots.begin(),
        arch.slots.end(),
        snap.time_level,
        [](TimeLevel level, const ArchiveSlot& s) { return level < s.snap.time_level; });
    arch.slots.insert(pos, ArchiveSlot{std::move(snap), 0u});
  }
}

void ConcreteOuterSdCoordinator::note_waiting(ZoneId local_zone,
                                              const PeerKey& peer,
                                              TimeLevel target_level) {
  const auto key = std::make_pair(local_zone, peer);
  auto& w = watch_[key];
  if (!w.active || w.waiting_for_level != target_level) {
    w.active = true;
    w.waiting_for_level = target_level;
    w.peer = peer;
    w.waiting_since = std::chrono::steady_clock::now();
  }
}

void ConcreteOuterSdCoordinator::clear_waiting(ZoneId local_zone, const PeerKey& peer) {
  const auto key = std::make_pair(local_zone, peer);
  auto it = watch_.find(key);
  if (it != watch_.end()) {
    it->second.active = false;
  }
}

[[noreturn]] void ConcreteOuterSdCoordinator::hard_error(const std::string& what) const {
  throw std::logic_error(what);
}

}  // namespace tdmd::scheduler
