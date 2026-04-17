#include "tdmd/neighbor/displacement_tracker.hpp"

#include <cmath>
#include <stdexcept>

namespace tdmd {

void DisplacementTracker::reset(const AtomSoA& atoms) {
  const std::size_t n = atoms.size();
  x_at_build_.assign(atoms.x.begin(), atoms.x.end());
  y_at_build_.assign(atoms.y.begin(), atoms.y.end());
  z_at_build_.assign(atoms.z.begin(), atoms.z.end());
  max_displacement_ = 0.0;
  (void) n;
}

void DisplacementTracker::update(const AtomSoA& atoms, const Box& box) {
  if (atoms.size() != x_at_build_.size()) {
    throw std::logic_error(
        "DisplacementTracker::update: atom count changed since last reset "
        "(migration / add / remove occurred without a rebuild)");
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

}  // namespace tdmd
