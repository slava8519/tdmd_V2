#include "tdmd/potentials/tabulated.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Algorithm reference: LAMMPS `pair_eam.cpp::interpolate()` and
// `PairEAM::single()` (GPL-2.0). Math is reimplemented fresh here to keep
// TDMD BSD-3; the FP operation sequence is preserved to meet invariant
// D-M2-1 (LAMMPS-bit-match, potentials/SPEC §4.4). See the header for the
// full algorithm write-up.

namespace tdmd::potentials {

namespace {

// The 5-point interior stencil needs two valid neighbours on each side, so
// n >= 5 is the minimum that produces at least one interior cell.
constexpr std::size_t kMinPoints = 5;

// Uniform-grid tolerance — relative gap between observed dx and the
// extrapolated grid step. 1e-12 matches potentials/SPEC §4.4 (uniform-grid
// contract) and leaves plenty of headroom for writer FP noise.
constexpr double kUniformGridRelTol = 1.0e-12;

[[noreturn]] void throw_invalid(const std::string& msg) {
  throw std::invalid_argument("TabulatedFunction: " + msg);
}

void validate_uniform_grid(const std::vector<double>& x_grid) {
  const std::size_t n = x_grid.size();
  if (n < kMinPoints) {
    std::ostringstream msg;
    msg << "x_grid must have at least " << kMinPoints << " points (got " << n << ").";
    throw_invalid(msg.str());
  }
  const double dx = x_grid[1] - x_grid[0];
  if (!(dx > 0.0)) {
    std::ostringstream msg;
    msg << "x_grid must be strictly increasing (dx = " << dx << ").";
    throw_invalid(msg.str());
  }
  for (std::size_t i = 2; i < n; ++i) {
    const double step = x_grid[i] - x_grid[i - 1];
    const double residual = std::abs(step - dx) / dx;
    if (residual > kUniformGridRelTol) {
      std::ostringstream msg;
      msg << "x_grid not uniform at index " << i << " (step=" << step << " vs dx=" << dx
          << ", rel=" << residual << ").";
      throw_invalid(msg.str());
    }
  }
}

}  // namespace

TabulatedFunction::TabulatedFunction(double x0, double dx, std::vector<double> y_values)
    : n_(y_values.size()), x0_(x0), dx_(dx), rdx_(0.0) {
  if (n_ < kMinPoints) {
    std::ostringstream msg;
    msg << "y_values must have at least " << kMinPoints << " entries (got " << n_ << ").";
    throw_invalid(msg.str());
  }
  if (!(dx_ > 0.0)) {
    std::ostringstream msg;
    msg << "dx must be strictly positive (got " << dx_ << ").";
    throw_invalid(msg.str());
  }
  rdx_ = 1.0 / dx_;
  coeffs_.resize(n_);
  build_spline_(y_values);
}

TabulatedFunction::TabulatedFunction(const std::vector<double>& x_grid,
                                     const std::vector<double>& y_values)
    : n_(y_values.size()), x0_(0.0), dx_(0.0), rdx_(0.0) {
  if (x_grid.size() != y_values.size()) {
    std::ostringstream msg;
    msg << "x_grid (" << x_grid.size() << ") and y_values (" << y_values.size()
        << ") must have identical length.";
    throw_invalid(msg.str());
  }
  validate_uniform_grid(x_grid);
  x0_ = x_grid.front();
  dx_ = x_grid[1] - x_grid[0];
  rdx_ = 1.0 / dx_;
  coeffs_.resize(n_);
  build_spline_(y_values);
}

// Build the per-cell spline. LAMMPS uses 1-based indexing with the sentinel
// cell at index n; TDMD uses 0-based storage so every LAMMPS `m` maps to
// `m - 1` here. To keep the FP operation sequence identical we still expose
// the "s values" as a single contiguous array the size of the grid and
// consume it in the same order LAMMPS does.
void TabulatedFunction::build_spline_(std::span<const double> y) {
  // Ctors validate n_ ≥ kMinPoints before calling us; the bounds assertion
  // documents the invariant and helps the optimiser elide redundant checks.
  assert(y.size() == n_ && n_ >= kMinPoints);
  std::vector<double> s(n_);

// gcc 13 -Wnull-dereference fires on `s[…]` right after the vector
// resize because it loses track of the dynamic size guarantee across the
// allocator boundary. The vector is non-empty by construction (n_ ≥ 5),
// so the deref is safe; silence the diagnostic just for the spline body.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

  // Boundary slopes (LAMMPS m = 1, 2, n-1, n). The 5-point stencil in the
  // interior loop uses these, so they must be written first.
  s[0] = y[1] - y[0];
  s[1] = 0.5 * (y[2] - y[0]);
  s[n_ - 2] = 0.5 * (y[n_ - 1] - y[n_ - 3]);
  s[n_ - 1] = y[n_ - 1] - y[n_ - 2];

  // Interior 4th-order slopes (LAMMPS m = 3 … n-2). Operation order preserved
  // per D-M2-1: parenthesised difference of far-neighbours first, then the
  // weighted inner difference, then the /12 divide.
  for (std::size_t i = 2; i + 2 < n_; ++i) {
    s[i] = ((y[i - 2] - y[i + 2]) + 8.0 * (y[i + 1] - y[i - 1])) / 12.0;
  }

  // Cubic-Hermite cell coefficients (LAMMPS m = 1 … n-1, TDMD i = 0 … n-2).
  // coeffs_[i][6] — y value at left cell node.
  // coeffs_[i][5] — slope s[i].
  // coeffs_[i][4] — c (Horner's quadratic coefficient).
  // coeffs_[i][3] — b (Horner's cubic coefficient).
  for (std::size_t i = 0; i + 1 < n_; ++i) {
    const double fi = y[i];
    const double fi1 = y[i + 1];
    coeffs_[i][6] = fi;
    coeffs_[i][5] = s[i];
    coeffs_[i][4] = 3.0 * (fi1 - fi) - 2.0 * s[i] - s[i + 1];
    coeffs_[i][3] = s[i] + s[i + 1] - 2.0 * (fi1 - fi);
  }

  // Sentinel cell (LAMMPS m = n). Carries y and s but no cubic — an eval
  // targeting this cell would hit the right clamp which resolves to cell
  // n-2 at p = 1, so the cubic terms are never read. Zeroing them keeps
  // fixture comparisons exact and avoids any stale-read heisenbugs.
  coeffs_[n_ - 1][6] = y[n_ - 1];
  coeffs_[n_ - 1][5] = s[n_ - 1];
  coeffs_[n_ - 1][4] = 0.0;
  coeffs_[n_ - 1][3] = 0.0;

  // Pre-divided derivative coefficients, one multiply saved per eval.
  for (std::size_t i = 0; i < n_; ++i) {
    coeffs_[i][2] = coeffs_[i][5] * rdx_;
    coeffs_[i][1] = 2.0 * coeffs_[i][4] * rdx_;
    coeffs_[i][0] = 3.0 * coeffs_[i][3] * rdx_;
  }

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
}

// Cell selection + local-coordinate reduction, shared between eval() and
// derivative(). Produces (i, p) such that coeffs_[i] is the cubic to
// evaluate at local coordinate p ∈ [0, 1]. The clamp follows LAMMPS:
//   * m = int(p_lammps) clamped to [1, n-1]  (LAMMPS 1-based)
//   * p reduced to [0, 1]
// so for x below x0 the first cell is used at p = 0 (returns y[0]), and for
// x above the right end the last-interior cell is used at p = 1 (returns
// y[n-1]). No actual extrapolation happens — hot-path safe.
namespace {

struct CellPoint {
  std::size_t i;  // 0-based cell index
  double p;       // local coord in [0, 1]
};

CellPoint locate(double x, double x0, double rdx, std::size_t n) noexcept {
  // LAMMPS pattern: p_raw = (x - x0) * rdx + 1.0, m = int(p_raw).
  // +1.0 so the 1-based cell index emerges directly; subtract m to get the
  // local coord. We carry the LAMMPS-style m through to stay bit-equivalent
  // in the occasional tie-breaking case at cell boundaries.
  const double p_lammps = (x - x0) * rdx + 1.0;
  long long m = static_cast<long long>(p_lammps);

  // Clamp m into [1, n-1] (LAMMPS 1-based cell range). n-1 here is the
  // last interior cell; the sentinel cell m = n is never selected.
  const long long m_min = 1;
  const long long m_max = static_cast<long long>(n) - 1;
  if (m < m_min) {
    m = m_min;
  } else if (m > m_max) {
    m = m_max;
  }

  double p = p_lammps - static_cast<double>(m);
  if (p > 1.0) {
    p = 1.0;
  } else if (p < 0.0) {
    // Only possible when x < x0 (m was clamped up to 1). Pin to the left
    // cell boundary so the eval at p = 0 returns y[0] exactly.
    p = 0.0;
  }

  return {static_cast<std::size_t>(m - 1), p};
}

}  // namespace

double TabulatedFunction::eval(double x) const noexcept {
  const auto [i, p] = locate(x, x0_, rdx_, n_);
  const auto& c = coeffs_[i];
  // Horner: ((b·p + c)·p + s)·p + f. Parentheses matter — preserves the
  // LAMMPS FP operation sequence used by `single()`.
  return ((c[3] * p + c[4]) * p + c[5]) * p + c[6];
}

double TabulatedFunction::derivative(double x) const noexcept {
  const auto [i, p] = locate(x, x0_, rdx_, n_);
  const auto& c = coeffs_[i];
  // (3b/δ · p + 2c/δ) · p + s/δ.
  return (c[0] * p + c[1]) * p + c[2];
}

const std::array<double, TabulatedFunction::kCoefficientsPerCell>& TabulatedFunction::coeffs(
    std::size_t cell_one_based) const {
  if (cell_one_based < 1 || cell_one_based > n_) {
    std::ostringstream msg;
    msg << "TabulatedFunction::coeffs: cell index " << cell_one_based << " out of range [1, " << n_
        << "].";
    throw std::out_of_range(msg.str());
  }
  return coeffs_[cell_one_based - 1];
}

}  // namespace tdmd::potentials
