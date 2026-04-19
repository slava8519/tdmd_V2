#pragma once

// SPEC: docs/specs/scheduler/SPEC.md §2.4 (OC-1..OC-6), §4.6 (HA-1..HA-5)
// Master spec: §12.7a
// Exec pack: docs/development/m7_execution_pack.md T7.6
//
// Reference implementation of `OuterSdCoordinator`. Single-threaded model
// for the M7 baseline (OQ-M7-3 race window deferred). Container choice is
// `std::map` for deterministic iteration order in Reference profile.
//
// The boundary-topology source (which (peer_subdomain, peer_zone) pairs
// each local zone depends on) is not available from `SubdomainGrid` alone;
// T7.7 wires it from the zone DAG. T7.6 exposes
// `register_zone_peer_dependency()` so unit tests + T7.7 can populate it
// out-of-band. With no peers registered, `can_advance` returns true —
// Pattern 1 nullable shape stays consistent.

#include "tdmd/scheduler/outer_sd_coordinator.hpp"

#include <chrono>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tdmd::scheduler {

struct PeerKey {
  std::uint32_t peer_subdomain = 0;
  ZoneId peer_zone = 0;

  bool operator<(const PeerKey& o) const noexcept {
    return std::tie(peer_subdomain, peer_zone) < std::tie(o.peer_subdomain, o.peer_zone);
  }
  bool operator==(const PeerKey& o) const noexcept {
    return peer_subdomain == o.peer_subdomain && peer_zone == o.peer_zone;
  }
};

// One stalled-boundary report. `check_stall_boundaries` returns the list
// of triggers since last call (and resets the in-coordinator queue). Real
// telemetry sink wiring lands in T7.7; T7.6 just exposes the queue.
struct OuterStallReport {
  ZoneId local_zone = 0;
  TimeLevel waiting_for_level = 0;
  PeerKey peer{};
  std::chrono::steady_clock::time_point waiting_since{};
};

class ConcreteOuterSdCoordinator final : public OuterSdCoordinator {
public:
  // Reference / non-Reference behaviour for HA-2 collisions. Reference =
  // throw on double-register; Production / Fast = log advisory and keep
  // existing snapshot. T7.6 ships only the Reference flavour live.
  enum class Mode : std::uint8_t {
    kReference,
    kProduction,
  };

  explicit ConcreteOuterSdCoordinator(Mode mode = Mode::kReference) noexcept : mode_{mode} {}

  // OuterSdCoordinator interface — see SPEC §2.4.
  void initialize(const SubdomainGrid& grid, std::uint32_t k_max) override;
  [[nodiscard]] bool can_advance_boundary_zone(ZoneId local_zone, TimeLevel target_level) override;
  void register_boundary_snapshot(ZoneId local_zone,
                                  TimeLevel level,
                                  const HaloSnapshot& snap) override;
  [[nodiscard]] std::optional<HaloSnapshot> fetch_peer_snapshot(std::uint32_t peer_subdomain,
                                                                ZoneId peer_zone,
                                                                TimeLevel level) override;
  void check_stall_boundaries(std::chrono::milliseconds t_stall_max) override;
  [[nodiscard]] TimeLevel global_frontier_min() const override { return global_frontier_min_; }
  [[nodiscard]] TimeLevel global_frontier_max() const override { return global_frontier_max_; }

  // T7.6 helpers — not part of the abstract contract. T7.7 wires from
  // zoning; tests call directly.

  // Ingest an unpacked peer snapshot into the ring buffer (per SPEC §4.6
  // lifecycle: HaloPacket → unpack → archive). T7.5/T7.7 wires the comm
  // path; T7.6 unit tests call this directly. Assigns `received_seq`
  // from the deterministic per-coordinator counter and inserts in
  // ascending `time_level` order. Throws (Reference) on HA-2 collision.
  void archive_peer_snapshot(HaloSnapshot snap);

  // Declares that `local_zone` requires a snapshot from `peer` at
  // `target_level - 1` to advance to `target_level`. Idempotent.
  void register_zone_peer_dependency(ZoneId local_zone, PeerKey peer);

  // Releases an outstanding fetch; decrements `use_count` on the slot.
  // Returns false if no in-flight fetch is on record.
  bool release_snapshot(std::uint32_t peer_subdomain, ZoneId peer_zone, TimeLevel level);

  // Advances global frontier estimates. Asserts monotonicity (OC-5).
  // T7.7 / runtime calls this after collectives roll up local frontiers.
  void set_global_frontier(TimeLevel new_min, TimeLevel new_max);

  // Drain stall reports accumulated by `check_stall_boundaries`.
  std::vector<OuterStallReport> drain_stall_reports();

  // Test / telemetry introspection.
  [[nodiscard]] std::uint64_t snapshot_too_old_total() const noexcept {
    return snapshot_too_old_total_;
  }
  [[nodiscard]] std::uint64_t register_collisions_total() const noexcept {
    return register_collisions_total_;
  }
  [[nodiscard]] std::uint32_t k_max() const noexcept { return k_max_; }
  [[nodiscard]] std::size_t archive_size() const noexcept { return peer_archives_.size(); }
  // Slot count for a specific peer key (HA-1 capacity assertion in tests).
  [[nodiscard]] std::size_t peer_slot_count(const PeerKey& peer) const;

private:
  // Per (peer_subdomain, peer_zone) ring buffer entry.
  struct ArchiveSlot {
    HaloSnapshot snap;
    std::uint32_t use_count = 0;
  };

  // Ring buffer of size K_max — std::vector held with a manual head/size
  // for O(1) eviction by oldest_level. K_max is small (≤ 8 typically), so
  // linear scans for level lookup are fine.
  struct PeerArchive {
    std::vector<ArchiveSlot> slots;  // ordered by ascending time_level
  };

  // Per-boundary outgoing-zone tracking (HA-2 + watchdog).
  struct LocalBoundaryWatch {
    TimeLevel waiting_for_level = 0;
    PeerKey peer{};
    std::chrono::steady_clock::time_point waiting_since{};
    bool active = false;
  };

  Mode mode_;
  SubdomainGrid grid_{};
  std::uint32_t k_max_ = 0;

  // Locally registered outgoing snapshots: (local_zone, level) → snapshot
  // (kept for HA-2 double-register detection only).
  std::map<std::pair<ZoneId, TimeLevel>, HaloSnapshot> local_registry_;

  // Incoming archive: per (peer_subdomain, peer_zone) → ring buffer.
  std::map<PeerKey, PeerArchive> peer_archives_;

  // Boundary topology: local_zone → set of peer keys it depends on.
  std::map<ZoneId, std::vector<PeerKey>> zone_peer_deps_;

  // Watchdog: per (local_zone, peer) waiting state.
  std::map<std::pair<ZoneId, PeerKey>, LocalBoundaryWatch> watch_;
  std::vector<OuterStallReport> stall_reports_;

  TimeLevel global_frontier_min_ = 0;
  TimeLevel global_frontier_max_ = 0;

  std::uint64_t next_seq_ = 0;
  std::uint64_t snapshot_too_old_total_ = 0;
  std::uint64_t register_collisions_total_ = 0;

  // Returns the slot index in `peer.slots` for a given level, or npos.
  static constexpr std::size_t kNpos = static_cast<std::size_t>(-1);
  std::size_t find_level_slot(const PeerArchive& arch, TimeLevel level) const;

  // Inserts the snapshot, evicting oldest if HA-1 would be violated.
  // Throws (Reference) if eviction is blocked by HA-5.
  void insert_into_archive(PeerArchive& arch, HaloSnapshot snap);

  void note_waiting(ZoneId local_zone, const PeerKey& peer, TimeLevel target_level);
  void clear_waiting(ZoneId local_zone, const PeerKey& peer);

  [[noreturn]] void hard_error(const std::string& what) const;
};

}  // namespace tdmd::scheduler
