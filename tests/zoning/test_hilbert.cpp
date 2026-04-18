// SPEC: docs/specs/zoning/SPEC.md §3.3
// Exec pack: docs/development/m3_execution_pack.md T3.5

#include "tdmd/zoning/hilbert.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstdlib>
#include <set>

namespace h = tdmd::zoning::hilbert;

TEST_CASE("Hilbert — round-trip d ↔ (x,y,z) on small cubes", "[zoning][hilbert]") {
  // Exhaustive check at orders 1..4. 4^3 bits per dim → 64^3 = 262144
  // combinations at order 6; we cap at 4 to keep the test fast.
  for (int bits : {1, 2, 3, 4}) {
    const std::uint32_t total = 1u << (3 * bits);
    for (std::uint32_t d = 0; d < total; ++d) {
      std::uint32_t x = 0, y = 0, z = 0;
      h::d_to_xyz(d, bits, x, y, z);
      REQUIRE(x < (1u << bits));
      REQUIRE(y < (1u << bits));
      REQUIRE(z < (1u << bits));
      REQUIRE(h::xyz_to_d(x, y, z, bits) == d);
    }
  }
}

TEST_CASE("Hilbert — d_to_xyz is a permutation of the cube", "[zoning][hilbert]") {
  for (int bits : {1, 2, 3, 4}) {
    const std::uint32_t total = 1u << (3 * bits);
    std::set<std::uint64_t> seen;
    for (std::uint32_t d = 0; d < total; ++d) {
      std::uint32_t x = 0, y = 0, z = 0;
      h::d_to_xyz(d, bits, x, y, z);
      const std::uint64_t packed =
          (static_cast<std::uint64_t>(z) << 40) | (static_cast<std::uint64_t>(y) << 20) | x;
      seen.insert(packed);
    }
    REQUIRE(seen.size() == total);
  }
}

TEST_CASE("Hilbert — locality: consecutive d differ by one axis ±1", "[zoning][hilbert]") {
  // The defining Hilbert property: ‖coord(d+1) − coord(d)‖₁ = 1.
  // Equivalent: exactly one axis changes, and by ±1.
  for (int bits : {1, 2, 3, 4}) {
    const std::uint32_t total = 1u << (3 * bits);
    std::uint32_t px = 0, py = 0, pz = 0;
    h::d_to_xyz(0, bits, px, py, pz);
    for (std::uint32_t d = 1; d < total; ++d) {
      std::uint32_t x = 0, y = 0, z = 0;
      h::d_to_xyz(d, bits, x, y, z);
      const int dx = std::abs(static_cast<int>(x) - static_cast<int>(px));
      const int dy = std::abs(static_cast<int>(y) - static_cast<int>(py));
      const int dz = std::abs(static_cast<int>(z) - static_cast<int>(pz));
      const int sum = dx + dy + dz;
      const int max_axis = std::max({dx, dy, dz});
      REQUIRE(sum == 1);
      REQUIRE(max_axis == 1);
      px = x;
      py = y;
      pz = z;
    }
  }
}

TEST_CASE("Hilbert — d=0 maps to origin (Skilling canonical)", "[zoning][hilbert]") {
  // Skilling's convention: d=0 is (0,0,0); other conventions exist
  // (rotate / reflect), but we're frozen on this one for canonical
  // ordering stability.
  for (int bits : {1, 2, 3, 4}) {
    std::uint32_t x = 0, y = 0, z = 0;
    h::d_to_xyz(0, bits, x, y, z);
    REQUIRE(x == 0u);
    REQUIRE(y == 0u);
    REQUIRE(z == 0u);
  }
}

TEST_CASE("Hilbert — order=1 reference table (8 cells)", "[zoning][hilbert]") {
  // Any reasonable 3D Hilbert implementation visits the 8 corners of
  // the unit cube in a Gray-code-like order. We pin our specific
  // Skilling-convention sequence here so regressions catch silent
  // convention drift. The sequence below is what this implementation
  // produces; any change requires bumping the zoning module major
  // version per SPEC §8.4.
  const std::uint32_t expected[8][3] = {
      {0, 0, 0},  // d=0
      {0, 0, 1},  // d=1
      {0, 1, 1},  // d=2
      {0, 1, 0},  // d=3
      {1, 1, 0},  // d=4
      {1, 1, 1},  // d=5
      {1, 0, 1},  // d=6
      {1, 0, 0},  // d=7
  };
  for (std::uint32_t d = 0; d < 8; ++d) {
    std::uint32_t x = 0, y = 0, z = 0;
    h::d_to_xyz(d, 1, x, y, z);
    REQUIRE(x == expected[d][0]);
    REQUIRE(y == expected[d][1]);
    REQUIRE(z == expected[d][2]);
  }
}
