#include "tdmd/state/box.hpp"

#include <cassert>
#include <cmath>

namespace tdmd {

namespace {

inline void wrap_axis(double& coord,
                      std::int32_t& image,
                      double lo,
                      double hi,
                      bool periodic) noexcept {
  if (!periodic)
    return;
  const double len = hi - lo;
  if (!(len > 0.0))
    return;  // degenerate box: no-op.
  while (coord >= hi) {
    coord -= len;
    ++image;
  }
  while (coord < lo) {
    coord += len;
    --image;
  }
}

inline double minimum_image_axis(double delta, double len, bool periodic) noexcept {
  if (!periodic)
    return delta;
  if (!(len > 0.0))
    return delta;
  const double half = 0.5 * len;
  if (delta > half) {
    delta -= len * std::ceil((delta - half) / len);
  } else if (delta < -half) {
    delta += len * std::ceil((-delta - half) / len);
  }
  return delta;
}

}  // namespace

double Box::length(int axis) const noexcept {
  assert(axis >= 0 && axis < 3 && "Box::length — axis must be 0|1|2");
  switch (axis) {
    case 0:
      return lx();
    case 1:
      return ly();
    case 2:
      return lz();
    default:
      return 0.0;
  }
}

bool Box::is_valid_m1() const noexcept {
  return lx() > 0.0 && ly() > 0.0 && lz() > 0.0 && tilt_xy == 0.0 && tilt_xz == 0.0 &&
         tilt_yz == 0.0;
}

void Box::wrap(double& x,
               double& y,
               double& z,
               std::int32_t& image_x,
               std::int32_t& image_y,
               std::int32_t& image_z) const noexcept {
  wrap_axis(x, image_x, xlo, xhi, periodic_x);
  wrap_axis(y, image_y, ylo, yhi, periodic_y);
  wrap_axis(z, image_z, zlo, zhi, periodic_z);
}

void Box::wrap(double& x, double& y, double& z) const noexcept {
  std::int32_t dummy_ix = 0;
  std::int32_t dummy_iy = 0;
  std::int32_t dummy_iz = 0;
  wrap(x, y, z, dummy_ix, dummy_iy, dummy_iz);
}

std::array<double, 3> Box::unwrap_minimum_image(double dx, double dy, double dz) const noexcept {
  return {
      minimum_image_axis(dx, lx(), periodic_x),
      minimum_image_axis(dy, ly(), periodic_y),
      minimum_image_axis(dz, lz(), periodic_z),
  };
}

}  // namespace tdmd
