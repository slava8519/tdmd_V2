// SPEC: docs/specs/potentials/SPEC.md §4.4 (tabulated primitive, LAMMPS
// bit-match requirement D-M2-1).
// Exec pack: docs/development/m2_execution_pack.md T2.5.
//
// Tests for TabulatedFunction (uniform-grid cubic-Hermite spline). Covers:
//   1. Input-validation throws.
//   2. Grid-point exactness (eval(x[i]) == y[i] bitwise).
//   3. Cubic reproduction at deep-interior cells (4-point FD is exact for
//      polynomials of degree ≤ 3 — residual driven only by FP rounding in
//      the FD/Hermite chain).
//   4. sin / cos accuracy on a fine grid (convergence-rate spot-check).
//   5. Clamp-to-boundary extrapolation (hot-path safety).
//   6. Hand-derived LAMMPS-match fixture for y = i² (bit-exact coefficients).
//   7. `coeffs()` bounds-check.
//   8. 10 000-trial property sweep over random cubics.

#include "tdmd/potentials/tabulated.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

using tdmd::potentials::TabulatedFunction;

namespace {

std::vector<double>
sample_cubic(double a, double b, double c, double d, double x0, double dx, std::size_t n) {
  std::vector<double> y(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double x = x0 + static_cast<double>(i) * dx;
    y[i] = ((d * x + c) * x + b) * x + a;
  }
  return y;
}

}  // namespace

TEST_CASE("TabulatedFunction: constructor rejects too-few points", "[potentials][tabulated]") {
  // kMinPoints = 5; four entries must be rejected by both ctors.
  std::vector<double> y = {1.0, 2.0, 3.0, 4.0};
  REQUIRE_THROWS_AS(TabulatedFunction(0.0, 1.0, y), std::invalid_argument);

  std::vector<double> x = {0.0, 1.0, 2.0, 3.0};
  REQUIRE_THROWS_AS(TabulatedFunction(x, y), std::invalid_argument);
}

TEST_CASE("TabulatedFunction: constructor rejects non-positive dx", "[potentials][tabulated]") {
  std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0};
  REQUIRE_THROWS_AS(TabulatedFunction(0.0, 0.0, y), std::invalid_argument);
  REQUIRE_THROWS_AS(TabulatedFunction(0.0, -1.0, y), std::invalid_argument);
}

TEST_CASE("TabulatedFunction: x_grid ctor rejects non-uniform grid", "[potentials][tabulated]") {
  std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.1};  // last gap is 1.1
  std::vector<double> y = {0.0, 1.0, 2.0, 3.0, 4.0};
  REQUIRE_THROWS_AS(TabulatedFunction(x, y), std::invalid_argument);
}

TEST_CASE("TabulatedFunction: x_grid ctor rejects length mismatch", "[potentials][tabulated]") {
  std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> y = {0.0, 1.0, 2.0, 3.0, 4.0};
  REQUIRE_THROWS_AS(TabulatedFunction(x, y), std::invalid_argument);
}

TEST_CASE("TabulatedFunction: eval at grid points is bitwise exact", "[potentials][tabulated]") {
  // Hermite form with the f-term written first → eval(x[i]) reduces to y[i]
  // with zero rounding (((b·0 + c)·0 + s)·0 + f). Hold the line: a single
  // lost bit here means the coeff layout drifted.
  const double x0 = 1.0;
  const double dx = 0.5;
  std::vector<double> y = {1.0, 2.5, 4.2, 3.7, 5.9, 6.1, 4.4, 3.3};
  TabulatedFunction tab(x0, dx, y);
  for (std::size_t i = 0; i < y.size(); ++i) {
    const double x = x0 + static_cast<double>(i) * dx;
    REQUIRE(tab.eval(x) == y[i]);
  }
}

TEST_CASE("TabulatedFunction: cubic exact at deep-interior cells", "[potentials][tabulated]") {
  // y(x) = a + b·x + c·x² + d·x³. The 4-point central FD is mathematically
  // exact for degree ≤ 3, so deep-interior cells (both endpoint slopes use
  // the 4-point stencil) reproduce y exactly up to rounding in the FD/Hermite
  // /Horner chain — ~1e-13 rel is comfortable headroom.
  const double a = 1.0, b = 2.0, c = 3.0, d = 4.0;
  const double x0 = 0.0;
  const double dx = 0.1;
  const std::size_t n = 21;
  auto y = sample_cubic(a, b, c, d, x0, dx, n);
  TabulatedFunction tab(x0, dx, y);

  // Deep-interior cells: i ∈ [2, n-4] so that both s[i] and s[i+1] are
  // 4-point-FD estimates.
  for (std::size_t cell = 2; cell + 3 < n; ++cell) {
    for (double p : {0.0, 0.25, 0.5, 0.75, 1.0}) {
      const double x = x0 + (static_cast<double>(cell) + p) * dx;
      const double expected = ((d * x + c) * x + b) * x + a;
      const double actual = tab.eval(x);
      const double scale = std::max({std::fabs(expected), 1.0});
      REQUIRE(std::fabs(actual - expected) < 1.0e-13 * scale);
    }
  }
}

TEST_CASE("TabulatedFunction: sin / cos accuracy", "[potentials][tabulated]") {
  // Grid: 1000 points, dx = 0.01, span [0, 9.99]. Cubic Hermite with 4-point
  // slopes is O(dx^4) for the value and O(dx^3) for the derivative — so 1e-6
  // / 1e-4 is loose but catches order-of-magnitude regressions.
  const std::size_t n = 1000;
  const double dx = 0.01;
  const double x0 = 0.0;
  std::vector<double> y(n);
  for (std::size_t i = 0; i < n; ++i) {
    y[i] = std::sin(x0 + static_cast<double>(i) * dx);
  }
  TabulatedFunction tab(x0, dx, y);

  std::mt19937 rng(0xC01DBEEFu);
  std::uniform_real_distribution<double> sample(0.1, 9.88);
  for (int trial = 0; trial < 200; ++trial) {
    const double x = sample(rng);
    REQUIRE(std::fabs(tab.eval(x) - std::sin(x)) < 1.0e-6);
    REQUIRE(std::fabs(tab.derivative(x) - std::cos(x)) < 1.0e-4);
  }
}

TEST_CASE("TabulatedFunction: clamp-to-boundary extrapolation", "[potentials][tabulated]") {
  // Out-of-range x must pin to the nearest boundary. At p = 0 the cell
  // returns f (= y at left node) and at p = 1 the last interior cell
  // returns f + s + c + b = y[n-1] (Hermite endpoint condition).
  const double x0 = 1.0;
  const double dx = 0.25;
  std::vector<double> y = {3.1, 4.2, 5.3, 6.4, 7.5, 8.6};
  TabulatedFunction tab(x0, dx, y);

  REQUIRE(tab.eval(-10.0) == y.front());
  REQUIRE(tab.eval(0.0) == y.front());

  const double x_max = x0 + static_cast<double>(y.size() - 1) * dx;
  REQUIRE(tab.eval(x_max + 1.0) == y.back());
  REQUIRE(tab.eval(1.0e9) == y.back());

  // Derivative clamps identically (same locate() path).
  REQUIRE(std::isfinite(tab.derivative(-1.0e9)));
  REQUIRE(std::isfinite(tab.derivative(+1.0e9)));
}

TEST_CASE("TabulatedFunction: LAMMPS-match fixture (y = i²)", "[potentials][tabulated][fixture]") {
  // Hand derivation for y = {0, 1, 4, 9, 16, 25}, dx = 1:
  //   s[0] = 1-0 = 1
  //   s[1] = (4-0)/2 = 2
  //   s[2] = ((0-16) + 8·(9-1))/12 = 4
  //   s[3] = ((1-25) + 8·(16-4))/12 = 6
  //   s[4] = (25-9)/2 = 8
  //   s[5] = 25-16 = 9
  // Cell m=3 (0-based 2): f=4, s=4, s_next=6 →
  //   c = 3·(9-4) - 2·4 - 6 = 1
  //   b = 4 + 6 - 2·(9-4) = 0
  // LAMMPS 7-tuple [3b/δ, 2c/δ, s/δ, b, c, s, f] with δ = 1 is verified below.
  std::vector<double> y = {0.0, 1.0, 4.0, 9.0, 16.0, 25.0};
  TabulatedFunction tab(0.0, 1.0, y);

  const auto& cell3 = tab.coeffs(3);  // LAMMPS 1-based m = 3
  REQUIRE(cell3[0] == 0.0);
  REQUIRE(cell3[1] == 2.0);
  REQUIRE(cell3[2] == 4.0);
  REQUIRE(cell3[3] == 0.0);
  REQUIRE(cell3[4] == 1.0);
  REQUIRE(cell3[5] == 4.0);
  REQUIRE(cell3[6] == 4.0);

  // Sentinel cell m = n carries b = c = 0 and the final y value; the
  // derivative pre-dividers degenerate to 0 with b = c = 0.
  const auto& sentinel = tab.coeffs(y.size());
  REQUIRE(sentinel[3] == 0.0);
  REQUIRE(sentinel[4] == 0.0);
  REQUIRE(sentinel[6] == 25.0);
}

TEST_CASE("TabulatedFunction: coeffs() bounds check", "[potentials][tabulated]") {
  TabulatedFunction tab(0.0, 1.0, std::vector<double>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0});
  REQUIRE_THROWS_AS(tab.coeffs(0), std::out_of_range);
  REQUIRE_THROWS_AS(tab.coeffs(7), std::out_of_range);
  REQUIRE_NOTHROW(tab.coeffs(1));
  REQUIRE_NOTHROW(tab.coeffs(6));
}

TEST_CASE("TabulatedFunction: property — 10 000 random cubics",
          "[potentials][tabulated][property]") {
  // Random cubics on random uniform grids, sampled at random deep-interior p.
  // 4-point FD is exact on cubics → residual is FP rounding only. The
  // 1e-11 rel ceiling is ~10⁵ eps, comfortably above the observed ~1e-13
  // noise on an 8-cell chain but far below algorithmic error levels.
  std::mt19937 rng(0xFEEDF00Du);
  std::uniform_real_distribution<double> coef(-2.0, 2.0);
  std::uniform_real_distribution<double> dx_dist(0.1, 0.5);
  std::uniform_real_distribution<double> p_dist(0.0, 1.0);

  constexpr int kTrials = 10000;
  const std::size_t n = 30;
  for (int trial = 0; trial < kTrials; ++trial) {
    const double a = coef(rng);
    const double b = coef(rng);
    const double c = coef(rng);
    const double d = coef(rng);
    const double dx = dx_dist(rng);
    const double x0 = coef(rng);
    auto y = sample_cubic(a, b, c, d, x0, dx, n);
    TabulatedFunction tab(x0, dx, y);

    // Deep-interior cells: [2, n-4] — cycle through them so the sampling
    // covers the full valid range.
    const std::size_t cell = 2 + static_cast<std::size_t>(trial) % (n - 5);
    const double p = p_dist(rng);
    const double x = x0 + (static_cast<double>(cell) + p) * dx;
    const double expected = ((d * x + c) * x + b) * x + a;
    const double actual = tab.eval(x);

    const double scale = std::max({std::fabs(expected), std::fabs(actual), 1.0});
    const double rel = std::fabs(actual - expected) / scale;
    REQUIRE(rel < 1.0e-11);
  }
}
