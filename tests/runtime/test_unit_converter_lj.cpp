#include "tdmd/integrator/velocity_verlet.hpp"
#include "tdmd/runtime/physical_constants.hpp"
#include "tdmd/runtime/unit_converter.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>

using tdmd::EnergyQ;
using tdmd::ForceQ;
using tdmd::kBoltzmann_eV_per_K;
using tdmd::kMetalMvv2e;
using tdmd::LengthQ;
using tdmd::LjReference;
using tdmd::MassQ;
using tdmd::PressureQ;
using tdmd::TemperatureQ;
using tdmd::TimeQ;
using tdmd::UnitConverter;
using tdmd::VelocityQ;

namespace {

// ULP distance between two finite FP64 values with matching signs. Standard
// bit-pattern trick: type-punning through memcpy (endianness-safe) yields
// lexicographic ordering of the magnitudes; the integer difference is the ULP
// count. Meant for tolerance assertions in the 0–16 ulp range; rolls over for
// values straddling zero or with different signs, so we require those
// separately in callers.
std::uint64_t ulp_distance(double a, double b) {
  std::uint64_t ai = 0;
  std::uint64_t bi = 0;
  std::memcpy(&ai, &a, sizeof(double));
  std::memcpy(&bi, &b, sizeof(double));
  return ai > bi ? (ai - bi) : (bi - ai);
}

bool close_ulp(double a, double b, std::uint64_t tol) {
  if (a == b) {
    return true;
  }
  if (std::signbit(a) != std::signbit(b)) {
    // Values of opposite sign: only equal when both are zero (handled above).
    return false;
  }
  return ulp_distance(a, b) <= tol;
}

constexpr LjReference kIdentity{1.0, 1.0, 1.0};

// Lennard-Jones Ar parameters from Rahman (1964): σ=3.405 Å, ε=0.0104 eV
// (≈ 120 K · kB), m=39.948 g/mol. Used as a non-trivial physical reference
// for round-trip robustness.
constexpr LjReference kAr{3.405, 0.0104, 39.948};

}  // namespace

// ---------------------------------------------------------------------------
// Section 1 — Formula spot checks against LAMMPS docs.
// ---------------------------------------------------------------------------

TEST_CASE("from_lj with identity reference gives unit-factor-only metal values",
          "[runtime][units][lj]") {
  // With σ=ε=m=1 and the standard LAMMPS formulas, from_lj reduces to a pure
  // unit-conversion-factor operation. length/energy/mass/force are factor-of-1;
  // pressure, time, velocity, temperature carry the conversion constant.
  REQUIRE(UnitConverter::length_from_lj(2.5, kIdentity).metal_angstroms == 2.5);
  REQUIRE(UnitConverter::energy_from_lj(-0.3, kIdentity).metal_eV == -0.3);
  REQUIRE(UnitConverter::mass_from_lj(4.0, kIdentity).metal_g_per_mol == 4.0);
  REQUIRE(UnitConverter::force_from_lj(7.0, kIdentity).metal_eV_per_A == 7.0);

  // Non-trivial factors: compare to direct CODATA-derived formula.
  REQUIRE(close_ulp(UnitConverter::pressure_from_lj(1.0, kIdentity).metal_bar, 1.602176634e6, 2));
  REQUIRE(close_ulp(UnitConverter::temperature_from_lj(1.0, kIdentity).metal_K,
                    1.0 / kBoltzmann_eV_per_K,
                    2));
  REQUIRE(
      close_ulp(UnitConverter::time_from_lj(1.0, kIdentity).metal_ps, std::sqrt(kMetalMvv2e), 2));
  REQUIRE(close_ulp(UnitConverter::velocity_from_lj(1.0, kIdentity).metal_A_per_ps,
                    1.0 / std::sqrt(kMetalMvv2e),
                    2));
}

TEST_CASE("from_lj with Ar reference follows LAMMPS formulas", "[runtime][units][lj]") {
  // length_metal = lj · σ
  REQUIRE(close_ulp(UnitConverter::length_from_lj(1.0, kAr).metal_angstroms, kAr.sigma, 2));
  // energy_metal = lj · ε
  REQUIRE(close_ulp(UnitConverter::energy_from_lj(1.0, kAr).metal_eV, kAr.epsilon, 2));
  // mass_metal = lj · m_ref
  REQUIRE(close_ulp(UnitConverter::mass_from_lj(1.0, kAr).metal_g_per_mol, kAr.mass, 2));
  // force_metal = lj · ε/σ
  REQUIRE(
      close_ulp(UnitConverter::force_from_lj(1.0, kAr).metal_eV_per_A, kAr.epsilon / kAr.sigma, 2));
  // pressure_metal = lj · (ε/σ³) · 1.602176634e6 bar/(eV/Å³)
  const double sigma3 = kAr.sigma * kAr.sigma * kAr.sigma;
  REQUIRE(close_ulp(UnitConverter::pressure_from_lj(1.0, kAr).metal_bar,
                    kAr.epsilon / sigma3 * 1.602176634e6,
                    4));
  // temperature_metal = lj · ε/kB
  REQUIRE(close_ulp(UnitConverter::temperature_from_lj(1.0, kAr).metal_K,
                    kAr.epsilon / kBoltzmann_eV_per_K,
                    2));
  // time_metal = lj · sqrt(m·σ²/ε) · sqrt(kMetalMvv2e)
  REQUIRE(
      close_ulp(UnitConverter::time_from_lj(1.0, kAr).metal_ps,
                std::sqrt(kAr.mass * kAr.sigma * kAr.sigma / kAr.epsilon) * std::sqrt(kMetalMvv2e),
                4));
  // velocity_metal = lj · sqrt(ε/m) / sqrt(kMetalMvv2e)
  REQUIRE(close_ulp(UnitConverter::velocity_from_lj(1.0, kAr).metal_A_per_ps,
                    std::sqrt(kAr.epsilon / kAr.mass) / std::sqrt(kMetalMvv2e),
                    4));
}

// ---------------------------------------------------------------------------
// Section 2 — Round-trip on identity reference.
//   length / energy / mass / force: bitwise (pure factor of 1)
//   time / pressure / velocity / temperature: ≤ 2 ulp (factor ≠ 1)
// ---------------------------------------------------------------------------

TEST_CASE("round-trip on identity ref: length/energy/mass/force bitwise",
          "[runtime][units][lj][roundtrip]") {
  const double samples[] = {0.0,
                            1.0,
                            -1.0,
                            26.9815,
                            1.602176634e-19,
                            1e12,
                            -1e-12,
                            std::numeric_limits<double>::min(),
                            std::numeric_limits<double>::max(),
                            std::numeric_limits<double>::denorm_min()};
  for (double x : samples) {
    REQUIRE(UnitConverter::length_to_lj(UnitConverter::length_from_lj(x, kIdentity), kIdentity) ==
            x);
    REQUIRE(UnitConverter::energy_to_lj(UnitConverter::energy_from_lj(x, kIdentity), kIdentity) ==
            x);
    REQUIRE(UnitConverter::mass_to_lj(UnitConverter::mass_from_lj(x, kIdentity), kIdentity) == x);
    REQUIRE(UnitConverter::force_to_lj(UnitConverter::force_from_lj(x, kIdentity), kIdentity) == x);
  }
}

TEST_CASE("round-trip on identity ref: time/pressure/velocity/temperature within 2 ulp",
          "[runtime][units][lj][roundtrip]") {
  const double samples[] = {1.0, -1.0, 26.9815, 1.602176634e-19, 1e12, -1e-12};
  for (double x : samples) {
    REQUIRE(
        close_ulp(UnitConverter::time_to_lj(UnitConverter::time_from_lj(x, kIdentity), kIdentity),
                  x,
                  2));
    REQUIRE(close_ulp(
        UnitConverter::pressure_to_lj(UnitConverter::pressure_from_lj(x, kIdentity), kIdentity),
        x,
        2));
    REQUIRE(close_ulp(
        UnitConverter::velocity_to_lj(UnitConverter::velocity_from_lj(x, kIdentity), kIdentity),
        x,
        2));
    REQUIRE(
        close_ulp(UnitConverter::temperature_to_lj(UnitConverter::temperature_from_lj(x, kIdentity),
                                                   kIdentity),
                  x,
                  2));
  }
}

TEST_CASE("round-trip on Ar reference: all 8 dims within 2 ulp",
          "[runtime][units][lj][roundtrip]") {
  const double samples[] = {1.0, -1.0, 0.5, 26.9815, 1.602176634e-19, 1e12, -1e-12};
  for (double x : samples) {
    REQUIRE(
        close_ulp(UnitConverter::length_to_lj(UnitConverter::length_from_lj(x, kAr), kAr), x, 2));
    REQUIRE(
        close_ulp(UnitConverter::energy_to_lj(UnitConverter::energy_from_lj(x, kAr), kAr), x, 2));
    REQUIRE(close_ulp(UnitConverter::mass_to_lj(UnitConverter::mass_from_lj(x, kAr), kAr), x, 2));
    REQUIRE(close_ulp(UnitConverter::force_to_lj(UnitConverter::force_from_lj(x, kAr), kAr), x, 2));
    REQUIRE(close_ulp(UnitConverter::pressure_to_lj(UnitConverter::pressure_from_lj(x, kAr), kAr),
                      x,
                      2));
    REQUIRE(close_ulp(UnitConverter::time_to_lj(UnitConverter::time_from_lj(x, kAr), kAr), x, 2));
    REQUIRE(close_ulp(UnitConverter::velocity_to_lj(UnitConverter::velocity_from_lj(x, kAr), kAr),
                      x,
                      2));
    REQUIRE(
        close_ulp(UnitConverter::temperature_to_lj(UnitConverter::temperature_from_lj(x, kAr), kAr),
                  x,
                  2));
  }
}

// ---------------------------------------------------------------------------
// Section 3 — Internal consistency between time and velocity factors.
// ---------------------------------------------------------------------------

TEST_CASE("lj time and velocity factors are exact reciprocals (identity ref)",
          "[runtime][units][lj]") {
  // metal_time_per_lj · metal_velocity_per_lj = 1.0 when σ=ε=m=1. This is the
  // invariant that time_factor = sqrt(mvv2e), velocity_factor = 1/time_factor.
  const double t = UnitConverter::time_from_lj(1.0, kIdentity).metal_ps;
  const double v = UnitConverter::velocity_from_lj(1.0, kIdentity).metal_A_per_ps;
  REQUIRE(close_ulp(t * v, 1.0, 2));
}

// ---------------------------------------------------------------------------
// Section 4 — Invalid-reference handling.
// ---------------------------------------------------------------------------

TEST_CASE("invalid reference (σ or ε or m ≤ 0) throws std::invalid_argument",
          "[runtime][units][lj][invalid]") {
  const LjReference bad_sigma{-1.0, 1.0, 1.0};
  const LjReference zero_sigma{0.0, 1.0, 1.0};
  const LjReference bad_eps{1.0, -1.0, 1.0};
  const LjReference zero_eps{1.0, 0.0, 1.0};
  const LjReference bad_mass{1.0, 1.0, -1.0};
  const LjReference zero_mass{1.0, 1.0, 0.0};

  REQUIRE_THROWS_AS(UnitConverter::length_from_lj(1.0, bad_sigma), std::invalid_argument);
  REQUIRE_THROWS_AS(UnitConverter::length_from_lj(1.0, zero_sigma), std::invalid_argument);
  REQUIRE_THROWS_AS(UnitConverter::energy_from_lj(1.0, bad_eps), std::invalid_argument);
  REQUIRE_THROWS_AS(UnitConverter::energy_from_lj(1.0, zero_eps), std::invalid_argument);
  REQUIRE_THROWS_AS(UnitConverter::mass_from_lj(1.0, bad_mass), std::invalid_argument);
  REQUIRE_THROWS_AS(UnitConverter::mass_from_lj(1.0, zero_mass), std::invalid_argument);

  // to_lj path also validates.
  REQUIRE_THROWS_AS(UnitConverter::length_to_lj(LengthQ{1.0}, bad_sigma), std::invalid_argument);
  REQUIRE_THROWS_AS(UnitConverter::energy_to_lj(EnergyQ{1.0}, bad_eps), std::invalid_argument);
  REQUIRE_THROWS_AS(UnitConverter::mass_to_lj(MassQ{1.0}, bad_mass), std::invalid_argument);
}

TEST_CASE("NaN-valued reference rejected on every dimension", "[runtime][units][lj][invalid]") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const LjReference nan_sigma{nan, 1.0, 1.0};
  const LjReference nan_eps{1.0, nan, 1.0};
  const LjReference nan_mass{1.0, 1.0, nan};

  // NaN comparisons are unordered, so `!(x > 0)` catches them.
  REQUIRE_THROWS_AS(UnitConverter::length_from_lj(1.0, nan_sigma), std::invalid_argument);
  REQUIRE_THROWS_AS(UnitConverter::energy_from_lj(1.0, nan_eps), std::invalid_argument);
  REQUIRE_THROWS_AS(UnitConverter::mass_from_lj(1.0, nan_mass), std::invalid_argument);
  REQUIRE_THROWS_AS(UnitConverter::time_from_lj(1.0, nan_mass), std::invalid_argument);
  REQUIRE_THROWS_AS(UnitConverter::pressure_from_lj(1.0, nan_sigma), std::invalid_argument);
  REQUIRE_THROWS_AS(UnitConverter::velocity_from_lj(1.0, nan_eps), std::invalid_argument);
  REQUIRE_THROWS_AS(UnitConverter::temperature_from_lj(1.0, nan_eps), std::invalid_argument);
  REQUIRE_THROWS_AS(UnitConverter::force_from_lj(1.0, nan_sigma), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Section 5 — Property test: random (x, ref) pairs, round-trip ≤ 4 ulp.
// Seed is fixed so the same inputs run every time (CI reproducibility).
// ---------------------------------------------------------------------------

TEST_CASE("property: round-trip within 4 ulp on 10^4 random inputs",
          "[runtime][units][lj][property]") {
  std::mt19937_64 rng{0xC0FFEEULL};  // fixed seed; same inputs on every run
  std::uniform_real_distribution<double> x_dist(-1e6, 1e6);
  std::uniform_real_distribution<double> sigma_dist(0.1, 10.0);     // Å
  std::uniform_real_distribution<double> epsilon_dist(1e-4, 10.0);  // eV
  std::uniform_real_distribution<double> mass_dist(1.0, 300.0);     // g/mol

  constexpr std::uint64_t kUlpBudget = 4;
  constexpr std::size_t kCases = 10000;
  for (std::size_t i = 0; i < kCases; ++i) {
    const double x = x_dist(rng);
    const LjReference ref{sigma_dist(rng), epsilon_dist(rng), mass_dist(rng)};

    REQUIRE(close_ulp(UnitConverter::length_to_lj(UnitConverter::length_from_lj(x, ref), ref),
                      x,
                      kUlpBudget));
    REQUIRE(close_ulp(UnitConverter::energy_to_lj(UnitConverter::energy_from_lj(x, ref), ref),
                      x,
                      kUlpBudget));
    REQUIRE(close_ulp(UnitConverter::mass_to_lj(UnitConverter::mass_from_lj(x, ref), ref),
                      x,
                      kUlpBudget));
    REQUIRE(close_ulp(UnitConverter::force_to_lj(UnitConverter::force_from_lj(x, ref), ref),
                      x,
                      kUlpBudget));
    REQUIRE(close_ulp(UnitConverter::pressure_to_lj(UnitConverter::pressure_from_lj(x, ref), ref),
                      x,
                      kUlpBudget));
    REQUIRE(close_ulp(UnitConverter::time_to_lj(UnitConverter::time_from_lj(x, ref), ref),
                      x,
                      kUlpBudget));
    REQUIRE(close_ulp(UnitConverter::velocity_to_lj(UnitConverter::velocity_from_lj(x, ref), ref),
                      x,
                      kUlpBudget));
    REQUIRE(
        close_ulp(UnitConverter::temperature_to_lj(UnitConverter::temperature_from_lj(x, ref), ref),
                  x,
                  kUlpBudget));
  }
}
