#include "tdmd/runtime/unit_converter.hpp"

#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <type_traits>

using tdmd::EnergyQ;
using tdmd::IncompatibleUnitError;
using tdmd::LengthQ;
using tdmd::UnitConverter;
using tdmd::UnitSystem;

// Detailed lj ↔ metal coverage lives in test_unit_converter_lj.cpp (T2.1).
// This file carries the metal identity path + `validate_supported_for_input`
// behaviour — the tests that existed before lj was implemented.

TEST_CASE("UnitConverter internal_system is metal", "[runtime][units]") {
  STATIC_REQUIRE(UnitConverter::internal_system() == UnitSystem::Metal);
}

TEST_CASE("UnitSystem to_string names", "[runtime][units]") {
  REQUIRE(tdmd::to_string(UnitSystem::Metal) == "metal");
  REQUIRE(tdmd::to_string(UnitSystem::Lj) == "lj");
  REQUIRE(tdmd::to_string(UnitSystem::Real) == "real");
  REQUIRE(tdmd::to_string(UnitSystem::Cgs) == "cgs");
  REQUIRE(tdmd::to_string(UnitSystem::Si) == "si");
}

TEST_CASE("metal ↔ metal is bitwise identity for all 8 dimensions", "[runtime][units][identity]") {
  constexpr double samples[] = {0.0,
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
    REQUIRE(UnitConverter::length_to_metal(UnitConverter::length_from_metal(x)) == x);
    REQUIRE(UnitConverter::energy_to_metal(UnitConverter::energy_from_metal(x)) == x);
    REQUIRE(UnitConverter::time_to_metal(UnitConverter::time_from_metal(x)) == x);
    REQUIRE(UnitConverter::mass_to_metal(UnitConverter::mass_from_metal(x)) == x);
    REQUIRE(UnitConverter::force_to_metal(UnitConverter::force_from_metal(x)) == x);
    REQUIRE(UnitConverter::pressure_to_metal(UnitConverter::pressure_from_metal(x)) == x);
    REQUIRE(UnitConverter::velocity_to_metal(UnitConverter::velocity_from_metal(x)) == x);
    REQUIRE(UnitConverter::temperature_to_metal(UnitConverter::temperature_from_metal(x)) == x);
  }
}

TEST_CASE("metal strong typedefs carry unit-labeled storage", "[runtime][units]") {
  const auto l = UnitConverter::length_from_metal(3.0);
  const auto e = UnitConverter::energy_from_metal(3.0);

  REQUIRE(l.metal_angstroms == 3.0);
  REQUIRE(e.metal_eV == 3.0);
  STATIC_REQUIRE(!std::is_same_v<LengthQ, EnergyQ>);
  STATIC_REQUIRE(!std::is_convertible_v<LengthQ, EnergyQ>);
  STATIC_REQUIRE(!std::is_convertible_v<double, LengthQ>);
  STATIC_REQUIRE(!std::is_convertible_v<LengthQ, double>);
}

TEST_CASE("validate_supported_for_input accepts metal and lj", "[runtime][units]") {
  REQUIRE_NOTHROW(UnitConverter::validate_supported_for_input(UnitSystem::Metal));
  REQUIRE_NOTHROW(UnitConverter::validate_supported_for_input(UnitSystem::Lj));
}

TEST_CASE("validate_supported_for_input rejects real/cgs/si as incompatible", "[runtime][units]") {
  REQUIRE_THROWS_AS(UnitConverter::validate_supported_for_input(UnitSystem::Real),
                    IncompatibleUnitError);
  REQUIRE_THROWS_AS(UnitConverter::validate_supported_for_input(UnitSystem::Cgs),
                    IncompatibleUnitError);
  REQUIRE_THROWS_AS(UnitConverter::validate_supported_for_input(UnitSystem::Si),
                    IncompatibleUnitError);
}

TEST_CASE("metal identity is constexpr — usable in compile-time contexts",
          "[runtime][units][constexpr]") {
  constexpr auto q = UnitConverter::length_from_metal(2.5);
  constexpr double back = UnitConverter::length_to_metal(q);
  STATIC_REQUIRE(q.metal_angstroms == 2.5);
  STATIC_REQUIRE(back == 2.5);
}
