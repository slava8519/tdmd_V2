#include "tdmd/neighbor/displacement_tracker.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace tdmd {

void DisplacementTracker::init(const AtomSoA& atoms) {
  x_at_build_.assign(atoms.x.begin(), atoms.x.end());
  y_at_build_.assign(atoms.y.begin(), atoms.y.end());
  z_at_build_.assign(atoms.z.begin(), atoms.z.end());
  max_displacement_ = 0.0;
  rebuild_pending_ = false;
  rebuild_reason_.clear();
  // Note: `build_version_` stays at whatever it was (0 on first init; the
  // tracker object may be reused across restart cycles).
}

void DisplacementTracker::update_displacement(const AtomSoA& atoms, const Box& box) {
  if (atoms.size() != x_at_build_.size()) {
    throw std::logic_error(
        "DisplacementTracker::update_displacement: atom count changed since the last "
        "baseline (migration / add / remove occurred without execute_rebuild)");
  }

  double max_d_sq = 0.0;
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const auto delta = box.unwrap_minimum_image(atoms.x[i] - x_at_build_[i],
                                                atoms.y[i] - y_at_build_[i],
                                                atoms.z[i] - z_at_build_[i]);
    const double d_sq = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
    if (d_sq > max_d_sq) {
      max_d_sq = d_sq;
    }
  }
  max_displacement_ = std::sqrt(max_d_sq);
}

void DisplacementTracker::set_threshold(double threshold) {
  if (!(threshold >= 0.0)) {
    throw std::invalid_argument(
        "DisplacementTracker::set_threshold: threshold must be non-negative");
  }
  threshold_ = threshold;
}

void DisplacementTracker::request_rebuild(std::string reason) {
  rebuild_pending_ = true;
  rebuild_reason_ = std::move(reason);
}

void DisplacementTracker::execute_rebuild(const AtomSoA& atoms) {
  x_at_build_.assign(atoms.x.begin(), atoms.x.end());
  y_at_build_.assign(atoms.y.begin(), atoms.y.end());
  z_at_build_.assign(atoms.z.begin(), atoms.z.end());
  max_displacement_ = 0.0;
  rebuild_pending_ = false;
  rebuild_reason_.clear();
  ++build_version_;
}

}  // namespace tdmd
