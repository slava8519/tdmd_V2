// SPEC: docs/specs/zoning/SPEC.md §8.3 (dissertation anchor table)
// Exec pack: docs/development/m3_execution_pack.md T3.7, D-M3-10
//
// **M3 artifact gate.** The dissertation anchor table asserts that each
// concrete planner produces the N_min / n_opt values derived in SPEC §3.x
// (which in turn trace back to Andreev's 2007 dissertation). Any failure
// here blocks the M3 release — either the planner regressed, or the SPEC
// needs a delta.
//
// The values here reflect the T3.7 SPEC delta resolving OQ-M3-5 /
// OQ-M3-6 (see zoning/SPEC.md §8.3 changelog). Prior to T3.7 the table
// in §8.3 carried transcription errors ("2D 16×5 → n_opt=13" and
// "3D Hilbert 16³ → n_opt=64") that did not follow from the §3.2 / §3.3
// formulas; T3.4 and T3.5 had tagged the conflicting assertions
// `[!mayfail]` pending resolution.

#include "tdmd/state/box.hpp"
#include "tdmd/zoning/decomp2d.hpp"
#include "tdmd/zoning/hilbert3d.hpp"
#include "tdmd/zoning/linear1d.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

namespace tz = tdmd::zoning;

namespace {

tdmd::Box make_box(double lx, double ly, double lz) {
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

}  // namespace

TEST_CASE("Anchor — Linear1D 1D×16 zones → n_opt = 8 (Andreev §2.2, eq. 35)",
          "[zoning][anchor][gate]") {
  // cutoff+skin = 3 Å; long axis 48.1 Å → floor(48.1/3) = 16 zones.
  // N_min = 2 (eq. 35) → n_opt = floor(16/2) = 8.
  tz::Linear1DZoningPlanner p;
  tz::PerformanceHint hint;
  auto plan = p.plan(make_box(48.1, 3.1, 3.1), 2.5, 0.5, 1, hint);
  REQUIRE(plan.scheme == tz::ZoningScheme::Linear1D);
  REQUIRE(plan.n_zones[0] == 16);
  REQUIRE(plan.n_min_per_rank == 2);
  REQUIRE(plan.optimal_rank_count == 8);
}

TEST_CASE("Anchor — Decomp2D 16×5 zones → n_opt = 6 (SPEC §3.2 formula, OQ-M3-5 resolved)",
          "[zoning][anchor][gate]") {
  // OQ-M3-5 resolution: SPEC §8.3 originally claimed n_opt=13; the
  // formula N_min = 2·(N_inner+1) = 2·(5+1) = 12 gives n_opt =
  // floor(80/12) = 6. Dissertation eq. 45 actually evaluates to 14
  // (verified during T3.4), so the prior §8.3 value was a transcription
  // error — not a re-derivation. The formula is now the source of truth.
  tz::Decomp2DZoningPlanner p;
  tz::PerformanceHint hint;
  auto plan = p.plan(make_box(48.1, 15.1, 3.1), 2.5, 0.5, 1, hint);
  REQUIRE(plan.scheme == tz::ZoningScheme::Decomp2D);
  REQUIRE(plan.n_zones[0] == 16);
  REQUIRE(plan.n_zones[1] == 5);
  REQUIRE(plan.n_zones[2] == 1);
  REQUIRE(plan.n_min_per_rank == 12);
  REQUIRE(plan.optimal_rank_count == 6);
}

TEST_CASE("Anchor — Hilbert3D 16³ zones → n_opt = 4 (SPEC §3.3 formula, OQ-M3-6 resolved)",
          "[zoning][anchor][gate]") {
  // OQ-M3-6 resolution: SPEC §8.3 originally claimed n_opt=64 for 16³,
  // which actually corresponds to a 256³ box under the §3.3 formula
  // (N_min = 4·256² = 262144; 16777216/262144 = 64) — a row-label
  // mis-attribution. For 16³: N_min = 4·16² = 1024; n_opt =
  // floor(4096/1024) = 4. Corrected to the formula-consistent value.
  tz::Hilbert3DZoningPlanner p;
  tz::PerformanceHint hint;
  auto plan = p.plan(make_box(48.1, 48.1, 48.1), 2.5, 0.5, 1, hint);
  REQUIRE(plan.scheme == tz::ZoningScheme::Hilbert3D);
  REQUIRE(plan.n_zones[0] == 16);
  REQUIRE(plan.n_zones[1] == 16);
  REQUIRE(plan.n_zones[2] == 16);
  REQUIRE(plan.canonical_order.size() == 16ull * 16 * 16);
  REQUIRE(plan.n_min_per_rank == 4ull * 16 * 16);
  REQUIRE(plan.optimal_rank_count == 4);
}

TEST_CASE("Anchor — Hilbert3D scaling: 256³ zones → n_opt = 64 (the number §8.3 originally wanted)",
          "[zoning][anchor][gate]") {
  // Sanity check that the §8.3 original "64" isn't magic — it's just the
  // 256³ row. Running plan() on a 256³ box is prohibitive (16M zones,
  // many seconds), so we call estimate_* which computes n_opt directly
  // without walking the Hilbert curve.
  tz::Hilbert3DZoningPlanner p;
  auto box = make_box(768.1, 768.1, 768.1);  // cutoff+skin=3 → 768/3 = 256
  REQUIRE(p.estimate_n_min(tz::ZoningScheme::Hilbert3D, box, 2.5, 0.5) == 4ull * 256 * 256);
  REQUIRE(p.estimate_optimal_ranks(tz::ZoningScheme::Hilbert3D, box, 2.5, 0.5) == 64u);
}
