#include "tdmd/state/box.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <random>

namespace {

tdmd::Box make_cubic_box(double side, bool periodic = true) {
  tdmd::Box b;
  b.xlo = 0.0;
  b.xhi = side;
  b.ylo = 0.0;
  b.yhi = side;
  b.zlo = 0.0;
  b.zhi = side;
  b.periodic_x = periodic;
  b.periodic_y = periodic;
  b.periodic_z = periodic;
  return b;
}

}  // namespace

TEST_CASE("Box::lx/ly/lz/volume", "[state][box]") {
  tdmd::Box b;
  b.xlo = -1.0;
  b.xhi = 4.0;  // lx = 5
  b.ylo = 0.0;
  b.yhi = 3.0;  // ly = 3
  b.zlo = 2.0;
  b.zhi = 10.0;  // lz = 8

  REQUIRE(b.lx() == 5.0);
  REQUIRE(b.ly() == 3.0);
  REQUIRE(b.lz() == 8.0);
  REQUIRE(b.length(0) == 5.0);
  REQUIRE(b.length(1) == 3.0);
  REQUIRE(b.length(2) == 8.0);
  REQUIRE(b.volume() == 5.0 * 3.0 * 8.0);
}

TEST_CASE("Box::is_valid_m1 rejects triclinic tilt", "[state][box]") {
  auto b = make_cubic_box(10.0);
  REQUIRE(b.is_valid_m1());
  b.tilt_xy = 0.1;
  REQUIRE_FALSE(b.is_valid_m1());
  b.tilt_xy = 0.0;
  b.tilt_xz = 0.1;
  REQUIRE_FALSE(b.is_valid_m1());
}

TEST_CASE("Box::wrap is no-op inside primary image", "[state][box][wrap]") {
  auto b = make_cubic_box(10.0);
  double x = 1.0, y = 5.0, z = 9.999;
  std::int32_t ix = 0, iy = 0, iz = 0;
  b.wrap(x, y, z, ix, iy, iz);
  REQUIRE(x == 1.0);
  REQUIRE(y == 5.0);
  REQUIRE(z == 9.999);
  REQUIRE(ix == 0);
  REQUIRE(iy == 0);
  REQUIRE(iz == 0);
}

TEST_CASE("Box::wrap positive crossing increments image", "[state][box][wrap]") {
  auto b = make_cubic_box(10.0);
  double x = 10.5, y = 21.0, z = 0.0;
  std::int32_t ix = 0, iy = 0, iz = 0;
  b.wrap(x, y, z, ix, iy, iz);
  REQUIRE(x == 0.5);
  REQUIRE(ix == 1);
  REQUIRE(y == 1.0);
  REQUIRE(iy == 2);
  REQUIRE(z == 0.0);
  REQUIRE(iz == 0);
}

TEST_CASE("Box::wrap negative crossing decrements image", "[state][box][wrap]") {
  auto b = make_cubic_box(10.0);
  double x = -0.5, y = -12.0, z = 5.0;
  std::int32_t ix = 0, iy = 0, iz = 0;
  b.wrap(x, y, z, ix, iy, iz);
  REQUIRE(x == 9.5);
  REQUIRE(ix == -1);
  REQUIRE(y == 8.0);
  REQUIRE(iy == -2);
  REQUIRE(z == 5.0);
  REQUIRE(iz == 0);
}

TEST_CASE("Box::wrap skips non-periodic axes", "[state][box][wrap]") {
  auto b = make_cubic_box(10.0, /*periodic=*/false);
  double x = 12.5, y = -5.0, z = 99.0;
  std::int32_t ix = 7, iy = -3, iz = 1;  // pre-existing image counts preserved
  b.wrap(x, y, z, ix, iy, iz);
  REQUIRE(x == 12.5);
  REQUIRE(y == -5.0);
  REQUIRE(z == 99.0);
  REQUIRE(ix == 7);
  REQUIRE(iy == -3);
  REQUIRE(iz == 1);
}

TEST_CASE("Box::wrap idempotent after first call (exhaustive)", "[state][box][wrap]") {
  auto b = make_cubic_box(10.0);
  // Deterministic seed — keep failures reproducible.
  std::mt19937_64 rng(0xA110C8ULL);
  std::uniform_real_distribution<double> coord(-123.0, 123.0);

  constexpr int kIter = 10'000;
  for (int k = 0; k < kIter; ++k) {
    double x = coord(rng);
    double y = coord(rng);
    double z = coord(rng);
    std::int32_t ix = 0, iy = 0, iz = 0;
    b.wrap(x, y, z, ix, iy, iz);

    // After one wrap, coordinates are in [lo, hi).
    REQUIRE(x >= b.xlo);
    REQUIRE(x < b.xhi);
    REQUIRE(y >= b.ylo);
    REQUIRE(y < b.yhi);
    REQUIRE(z >= b.zlo);
    REQUIRE(z < b.zhi);

    // Second call must not change anything.
    const double x1 = x, y1 = y, z1 = z;
    const std::int32_t ix1 = ix, iy1 = iy, iz1 = iz;
    b.wrap(x, y, z, ix, iy, iz);
    REQUIRE(x == x1);
    REQUIRE(y == y1);
    REQUIRE(z == z1);
    REQUIRE(ix == ix1);
    REQUIRE(iy == iy1);
    REQUIRE(iz == iz1);
  }
}

TEST_CASE("Box::unwrap_minimum_image keeps |Δ| ≤ L/2 on each periodic axis",
          "[state][box][minimum_image]") {
  auto b = make_cubic_box(10.0);
  const double half = 0.5 * b.lx();

  std::mt19937_64 rng(0xDEADBEEFULL);
  std::uniform_real_distribution<double> delta(-500.0, 500.0);

  constexpr int kIter = 10'000;
  for (int k = 0; k < kIter; ++k) {
    const auto r = b.unwrap_minimum_image(delta(rng), delta(rng), delta(rng));
    REQUIRE(std::abs(r[0]) <= half + 1e-12);
    REQUIRE(std::abs(r[1]) <= half + 1e-12);
    REQUIRE(std::abs(r[2]) <= half + 1e-12);
  }
}

TEST_CASE("Box::unwrap_minimum_image preserves Δ on non-periodic axes",
          "[state][box][minimum_image]") {
  auto b = make_cubic_box(10.0, /*periodic=*/false);
  const auto r = b.unwrap_minimum_image(42.0, -77.5, 1e6);
  REQUIRE(r[0] == 42.0);
  REQUIRE(r[1] == -77.5);
  REQUIRE(r[2] == 1e6);
}

TEST_CASE("Box::unwrap_minimum_image canonical single-axis reductions",
          "[state][box][minimum_image]") {
  auto b = make_cubic_box(10.0);
  SECTION("Δ = 6 (just over half): -4") {
    const auto r = b.unwrap_minimum_image(6.0, 0.0, 0.0);
    REQUIRE(r[0] == -4.0);
  }
  SECTION("Δ = -6: +4") {
    const auto r = b.unwrap_minimum_image(-6.0, 0.0, 0.0);
    REQUIRE(r[0] == 4.0);
  }
  SECTION("Δ = 5 (exactly half): +5 preserved (boundary)") {
    // |Δ| ≤ L/2 permits Δ = +L/2; algorithm should not shift.
    const auto r = b.unwrap_minimum_image(5.0, 0.0, 0.0);
    REQUIRE(std::abs(r[0]) <= 5.0 + 1e-12);
  }
  SECTION("Δ = 15.5 (1.5 box lengths + half): -4.5") {
    const auto r = b.unwrap_minimum_image(15.5, 0.0, 0.0);
    REQUIRE(std::abs(r[0]) <= 5.0 + 1e-12);
  }
}
