#pragma once

// SPEC: docs/specs/neighbor/SPEC.md §2.1 (DisplacementTracker), §5 (skin tracker)
// Exec pack: docs/development/m1_execution_pack.md T1.6
//
// Tracks how far each atom has moved since the last neighbor rebuild, in
// minimum-image convention, and reports whether the list has gone stale.
//
// Trigger rule (SPEC §5.2): rebuild is requested once `max_displacement >
// skin / 2`. This is conservative — a pair's relative displacement is at
// most `d_i + d_j ≤ 2·max_displacement`, so keeping `2·max_displacement <
// skin` guarantees no pair leaves / enters the cutoff without a rebuild.
//
// M1: global tracker (single max across all atoms). Per-zone tracking is a
// v2+ optimization per SPEC §5.3.

#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <cstddef>
#include <vector>

namespace tdmd {

class DisplacementTracker {
public:
  // Records the current positions as the new build baseline and zeros the
  // displacement. Threshold stays at its previous value unless
  // `set_threshold` is called afterwards.
  void reset(const AtomSoA& atoms);

  // Recomputes `max_displacement_` against the stored baseline. Uses
  // minimum-image convention so that a periodic wrap does not register as
  // a huge displacement. Throws `std::logic_error` if atom count changed
  // since the last `reset` (rebuild is required before further tracking).
  void update(const AtomSoA& atoms, const Box& box);

  // Setter for the trigger threshold. Conventional value is `skin / 2`.
  // Negative thresholds are rejected.
  void set_threshold(double threshold);

  [[nodiscard]] double threshold() const noexcept { return threshold_; }
  [[nodiscard]] double max_displacement() const noexcept { return max_displacement_; }
  [[nodiscard]] std::size_t size() const noexcept { return x_at_build_.size(); }
  [[nodiscard]] bool empty() const noexcept { return x_at_build_.empty(); }

  // True iff the tracker believes the neighbor list is stale. Conservative:
  // may return true slightly before strictly required, never after.
  [[nodiscard]] bool needs_rebuild() const noexcept { return max_displacement_ > threshold_; }

private:
  std::vector<double> x_at_build_;
  std::vector<double> y_at_build_;
  std::vector<double> z_at_build_;
  double max_displacement_ = 0.0;
  double threshold_ = 0.0;
};

}  // namespace tdmd
