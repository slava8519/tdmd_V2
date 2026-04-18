#pragma once

// SPEC: docs/specs/neighbor/SPEC.md §2.2 (NeighborManager API),
//       §5 (skin tracker), §6 (rebuild policy)
// Exec pack: docs/development/m3_execution_pack.md T3.8
//
// `DisplacementTracker` implements the skin-tracker half of the neighbor
// rebuild dance. It owns:
//   1. A baseline copy of positions (set at every `init` / `execute_rebuild`);
//   2. The rolling `max_displacement` since the last baseline, in minimum-
//      image convention (periodic wraps do not register as huge jumps);
//   3. A pending-rebuild flag + short reason string, accepting multiple
//      trigger sources (skin exceeded, migration, potential-changed, manual);
//   4. A monotone `build_version` — bumped once per `execute_rebuild`,
//      consumed by scheduler/SPEC §4 for temporal packet consistency.
//
// Trigger rule (SPEC §5.2): `skin_exceeded() == max_displacement > threshold`
// with default `threshold = r_skin / 2`. Conservative because the pair
// closure bound is `d_i + d_j ≤ 2·max_displacement`.

#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tdmd {

class DisplacementTracker {
public:
  // Records the current positions as the new build baseline, zeros the
  // displacement. Does NOT reset the threshold (call `set_threshold` to
  // change that) or the build_version counter (call `execute_rebuild` to
  // bump it). Use `init` for the very first baseline of a run and leave
  // `execute_rebuild` for every subsequent rebuild — the semantic split
  // keeps the post-init build_version==0 invariant.
  void init(const AtomSoA& atoms);

  // Recomputes `max_displacement_` against the stored baseline using
  // minimum-image convention. Throws `std::logic_error` if the atom count
  // has changed since the last baseline (migration / add / remove happened
  // without a corresponding `execute_rebuild`).
  void update_displacement(const AtomSoA& atoms, const Box& box);

  // Setter for the trigger threshold; negative values rejected. Conventional
  // value is `r_skin / 2`.
  void set_threshold(double threshold);

  [[nodiscard]] double threshold() const noexcept { return threshold_; }
  [[nodiscard]] double max_displacement() const noexcept { return max_displacement_; }
  [[nodiscard]] std::size_t size() const noexcept { return x_at_build_.size(); }
  [[nodiscard]] bool empty() const noexcept { return x_at_build_.empty(); }

  // Conservative skin check: true iff the tracker believes the neighbor
  // list is stale. May trigger slightly before strictly required
  // (SPEC §5.2 r_skin/2 threshold), never after.
  [[nodiscard]] bool skin_exceeded() const noexcept { return max_displacement_ > threshold_; }

  // --- Rebuild orchestration (SPEC §6) ---

  [[nodiscard]] std::uint64_t build_version() const noexcept { return build_version_; }
  [[nodiscard]] bool rebuild_pending() const noexcept { return rebuild_pending_; }
  [[nodiscard]] const std::string& rebuild_reason() const noexcept { return rebuild_reason_; }

  // Flags that a rebuild is wanted. Idempotent: multiple calls between
  // two `execute_rebuild`s accumulate reasons (the most recent wins for
  // telemetry; the pending flag was already true). Empty reason is
  // permitted but discouraged — a short tag like "skin" / "migration"
  // helps operators trace why the tracker fired.
  void request_rebuild(std::string reason);

  // Atomically: re-baselines positions to `atoms`, zeros max_displacement,
  // clears the pending flag, increments `build_version`. The scheduler
  // calls this after the neighbor list has actually been rebuilt so
  // subsequent `update_displacement` runs measure drift from the new
  // baseline.
  void execute_rebuild(const AtomSoA& atoms);

private:
  std::vector<double> x_at_build_;
  std::vector<double> y_at_build_;
  std::vector<double> z_at_build_;
  double max_displacement_ = 0.0;
  double threshold_ = 0.0;
  std::uint64_t build_version_ = 0;
  bool rebuild_pending_ = false;
  std::string rebuild_reason_;
};

}  // namespace tdmd
