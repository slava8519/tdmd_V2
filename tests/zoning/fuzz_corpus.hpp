#pragma once

// SPEC: docs/specs/zoning/SPEC.md §8.2 (property tests)
// Exec pack: docs/development/m3_execution_pack.md T3.7, D-M3-8
//
// Deterministic fuzz-case generators. A single `std::mt19937_64` with a
// frozen seed feeds per-scheme box generators tuned to bias into the
// regions of the SPEC §3.4 decision tree that each concrete planner
// actually handles — so 100k Linear1D cases don't get "wasted" on boxes
// that would trigger Decomp2D in Default routing.
//
// The generators produce boxes in metal units (Å). Cutoff + skin together
// give w ∈ [2, 4] Å — loosely Lennard-Jones-ish so `floor(length/w)` yields
// a meaningful zone count.

#include "tdmd/state/box.hpp"

#include <cstdint>
#include <random>

namespace tdmd::zoning::fuzz {

// Frozen seed — D-M3-8 reproducibility. Value is the first 64 bits of
// md5("tdmd/M3/zoning-property-seed-v1") truncated; any re-seed requires
// an exec-pack entry + CI baseline refresh.
inline constexpr std::uint64_t kFixedSeed = 0xD3ADB33FFC0FFEEuLL;

struct FuzzCase {
  tdmd::Box box;
  double cutoff;
  double skin;
  std::uint64_t n_ranks;
};

inline tdmd::Box make_box(double lx, double ly, double lz) {
  tdmd::Box b;
  b.xlo = 0.0;
  b.xhi = lx;
  b.ylo = 0.0;
  b.yhi = ly;
  b.zlo = 0.0;
  b.zhi = lz;
  b.periodic_x = b.periodic_y = b.periodic_z = true;
  return b;
}

class FuzzGenerator {
public:
  explicit FuzzGenerator(std::uint64_t seed = kFixedSeed) : rng_(seed) {}

  // Needle geometry: one long axis with 4..20 zones, two trivial. The
  // DefaultZoningPlanner picks Linear1D when aspect > 10 and min_ax < 4;
  // we bias toward aspect ≥ 12 so the invariant `N_min == 2` always holds.
  FuzzCase next_linear1d() {
    const double c = sample_cutoff();
    const double s = sample_skin();
    const double w = c + s;
    const auto n_long = std::uniform_int_distribution<std::uint32_t>{4, 20}(rng_);
    // Box length = n·w + tiny jitter inside the last zone. Other two axes
    // strictly < w so n_y = n_z = 0 — wait, that would fail Linear1D's
    // "box has usable extent" check. Use exactly one zone on y,z: length
    // in [w, 1.5·w).
    const double lx = n_long * w + 0.1 * w;
    const double ly = w + 0.3 * w * unit_real_();
    const double lz = w + 0.3 * w * unit_real_();
    return {make_box(lx, ly, lz), c, s, sample_n_ranks(n_long)};
  }

  // Thin slab — 2 axes with 2..8 zones, third trivial. Matches Decomp2D's
  // "single choice of outer+inner axis" path. Avoid the aspect > 10 branch
  // of SPEC §3.4 so DefaultZoningPlanner wouldn't re-route to Linear1D —
  // that's irrelevant to the concrete planner's invariants, but biasing
  // here keeps the statistical shape of the corpus interpretable.
  FuzzCase next_decomp2d() {
    const double c = sample_cutoff();
    const double s = sample_skin();
    const double w = c + s;
    const auto na = std::uniform_int_distribution<std::uint32_t>{2, 8}(rng_);
    const auto nb = std::uniform_int_distribution<std::uint32_t>{2, 8}(rng_);
    const double la = na * w + 0.1 * w;
    const double lb = nb * w + 0.1 * w;
    const double trivial = w + 0.3 * w * unit_real_();
    // Randomize which axis is trivial (0/1/2) so the canonical_order /
    // zigzag logic gets exercised in all three orientations.
    switch (std::uniform_int_distribution<int>{0, 2}(rng_)) {
      case 0:
        return {make_box(trivial, la, lb), c, s, sample_n_ranks(na * nb / 4)};
      case 1:
        return {make_box(la, trivial, lb), c, s, sample_n_ranks(na * nb / 4)};
      default:
        return {make_box(la, lb, trivial), c, s, sample_n_ranks(na * nb / 4)};
    }
  }

  // Cube-ish — all three axes with 3..7 zones. Guarantees aspect ≤ ~2.33
  // so SPEC §3.4 routes to Hilbert3D; also keeps pad³ ≤ 8³ so property-
  // test runtime stays well inside the 10 s CI budget.
  FuzzCase next_hilbert3d() {
    const double c = sample_cutoff();
    const double s = sample_skin();
    const double w = c + s;
    const auto nx = std::uniform_int_distribution<std::uint32_t>{3, 7}(rng_);
    const auto ny = std::uniform_int_distribution<std::uint32_t>{3, 7}(rng_);
    const auto nz = std::uniform_int_distribution<std::uint32_t>{3, 7}(rng_);
    const double lx = nx * w + 0.1 * w;
    const double ly = ny * w + 0.1 * w;
    const double lz = nz * w + 0.1 * w;
    return {make_box(lx, ly, lz), c, s, sample_n_ranks(nx * ny * nz / 16)};
  }

private:
  double sample_cutoff() { return 2.0 + 2.0 * unit_real_(); }  // [2, 4] Å
  double sample_skin() { return 0.2 + 0.8 * unit_real_(); }    // [0.2, 1.0] Å

  // n_ranks exercises both the "fits" and "exceeds n_opt" branches; bias
  // toward modest counts so the advisory path fires without dominating.
  std::uint64_t sample_n_ranks(std::uint32_t hint) {
    const std::uint32_t lo = hint == 0 ? 1 : hint;
    const std::uint32_t hi = std::max<std::uint32_t>(lo + 1, 2 * hint);
    return std::uniform_int_distribution<std::uint32_t>{lo, hi}(rng_);
  }

  double unit_real_() { return std::uniform_real_distribution<double>{0.0, 1.0}(rng_); }

  std::mt19937_64 rng_;
};

}  // namespace tdmd::zoning::fuzz
