#pragma once

// SPEC: docs/specs/potentials/SPEC.md §4.4 (tabulated primitive, LAMMPS
// bit-match requirement); master spec §14 M2 (EAM enablement).
// Exec pack: docs/development/m2_execution_pack.md T2.5.
//
// `TabulatedFunction` — uniform-grid cubic interpolant used by the EAM family
// (frho, rhor, z2r) and reserved for MEAM / SNAP callers in later milestones.
//
// --------------------------------------------------------------------------
// Algorithm (fresh reimplementation — not a copy)
// --------------------------------------------------------------------------
// The coefficient build and the Horner-form evaluator mirror the numerical
// scheme in LAMMPS `pair_eam.cpp::interpolate()` and `PairEAM::single()`
// (GPL-2.0). To keep invariant D-M2-1 (LAMMPS-bit-match, potentials/SPEC §4.4)
// realisable we preserve LAMMPS's exact sequence of FP additions and
// multiplications — IEEE 754 is non-associative so the bit-match depends on
// operation order, not just formula equivalence. Code layout, variable
// naming, ownership and error handling are original TDMD.
//
// Per-cell storage follows LAMMPS's 7-tuple convention so fixture data can
// be compared element-by-element:
//
//   coeffs[m] = [ 3·b/δ,  2·c/δ,  s/δ,  b,  c,  s,  f ]   // indices 0..6
//
// where δ = dx, f = y at the left cell node, s = "slope" (a finite-difference
// estimate of δ·y'), and b, c are cubic-Hermite coefficients. The
// low-numbered triple (indices 0..2) feeds the derivative evaluator; the
// high triple (3..5) feeds the value evaluator; index 6 is the raw y value
// kept for direct spline inspection and for the Horner constant term.
//
// Slope estimator:
//   * interior (m = 3 … n-2): 4th-order central FD
//       s[m] = ( (y[m-2] − y[m+2]) + 8·(y[m+1] − y[m-1]) ) / 12
//     which is exact for any polynomial of degree ≤ 3.
//   * near-boundary (m = 2, m = n-1): 2-point central FD, 2nd-order.
//   * boundary (m = 1, m = n): 1-sided forward / backward FD, 1st-order.
//
// Cubic-Hermite coefficients per cell (m = 1 … n-1):
//   c  = 3·(y[m+1] − y[m]) − 2·s[m] − s[m+1]
//   b  = s[m] + s[m+1] − 2·(y[m+1] − y[m])
// so  y(p) = ((b·p + c)·p + s)·p + f   for p ∈ [0, 1].
// Sentinel cell m = n carries b = c = 0 so the derivative of the last
// interior cell at p = 1 still evaluates cleanly (it doesn't own a cubic).
//
// --------------------------------------------------------------------------
// Public API contract
// --------------------------------------------------------------------------
//   * Uniform grid only. Non-uniform callers must resample upstream (master
//     spec §5.3). The `from_grid` factory verifies uniformity at 1e-12 rel.
//   * Minimum 5 grid points (the 5-point stencil needs ≥ 2 interior cells).
//     Smaller inputs throw `std::invalid_argument`.
//   * `eval` / `derivative` are `noexcept`, hot-path safe, `const`.
//   * Out-of-range `x` is clamped to the boundary cell (i.e. for x ≥ x_max
//     the evaluator returns the rightmost cell's polynomial at p = 1 — which
//     equals y[n-1] by construction of the Hermite form). This matches
//     LAMMPS's `m = MIN(m, nr-1); p = MIN(p, 1.0)` clamp and protects hot
//     loops (EAM rho can slightly exceed tabulated rhomax during MD).
//   * Coefficient layout is part of the public API for fixture testing; see
//     `coeffs()` accessor.

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace tdmd::potentials {

class TabulatedFunction {
public:
  static constexpr std::size_t kCoefficientsPerCell = 7;

  // Uniform-grid ctor: y_values[i] is sampled at x0 + i·dx, i ∈ [0, y.size()).
  //
  // Throws `std::invalid_argument` if dx ≤ 0 or if y has fewer than 5 entries.
  TabulatedFunction(double x0, double dx, std::vector<double> y_values);

  // x_grid ctor: validates uniform spacing to 1e-12 rel before delegating.
  // x_grid and y_values must have identical length.
  TabulatedFunction(const std::vector<double>& x_grid, const std::vector<double>& y_values);

  // y(x). Outside [x0, x0 + (n-1)·dx] the result is clamped to the nearest
  // boundary cell (not extrapolated). Safe to call with any finite double.
  [[nodiscard]] double eval(double x) const noexcept;

  // dy/dx at x. Same clamp-to-boundary semantics as `eval`.
  [[nodiscard]] double derivative(double x) const noexcept;

  // Number of grid points.
  [[nodiscard]] std::size_t size() const noexcept { return n_; }

  [[nodiscard]] double x0() const noexcept { return x0_; }
  [[nodiscard]] double dx() const noexcept { return dx_; }

  // Coefficient accessor. `cell_one_based` uses LAMMPS-style 1..n indexing
  // so the array indices in fixture tests line up directly with LAMMPS
  // `spline[m][k]`. Throws `std::out_of_range` for invalid cell indices.
  [[nodiscard]] const std::array<double, kCoefficientsPerCell>& coeffs(
      std::size_t cell_one_based) const;

private:
  void build_spline_(std::span<const double> y);

  std::size_t n_;  // number of grid points (== number of LAMMPS cells)
  double x0_;      // first grid point position
  double dx_;      // uniform spacing; guaranteed > 0
  double rdx_;     // 1 / dx_; cached for hot-path eval

  // coeffs_[i] corresponds to LAMMPS spline[i+1]; size == n_. The last
  // entry (i = n_-1 ⇔ LAMMPS m = n) is the sentinel cell with b = c = 0.
  std::vector<std::array<double, kCoefficientsPerCell>> coeffs_;
};

}  // namespace tdmd::potentials
