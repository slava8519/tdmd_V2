#include "tdmd/runtime/unit_converter.hpp"

#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <type_traits>

using tdmd::EnergyQ;
using tdmd::ForceQ;
using tdmd::IncompatibleUnitError;
using tdmd::LengthQ;
using tdmd::LjReference;
using tdmd::MassQ;
using tdmd::NotImplementedInM1Error;
using tdmd::PressureQ;
using tdmd::TemperatureQ;
using tdmd::TimeQ;
using tdmd::UnitConverter;
using tdmd::UnitSystem;
using tdmd::VelocityQ;

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

TEST_CASE("lj conversions throw NotImplementedInM1Error (all 8 dims)",
          "[runtime][units][lj_stub]") {
  const LjReference ref{1.0, 1.0, 1.0};

  REQUIRE_THROWS_AS(UnitConverter::length_from_lj(1.0, ref), NotImplementedInM1Error);
  REQUIRE_THROWS_AS(UnitConverter::length_to_lj(LengthQ{1.0}, ref), NotImplementedInM1Error);

  REQUIRE_THROWS_AS(UnitConverter::energy_from_lj(1.0, ref), NotImplementedInM1Error);
  REQUIRE_THROWS_AS(UnitConverter::energy_to_lj(EnergyQ{1.0}, ref), NotImplementedInM1Error);

  REQUIRE_THROWS_AS(UnitConverter::time_from_lj(1.0, ref), NotImplementedInM1Error);
  REQUIRE_THROWS_AS(UnitConverter::time_to_lj(TimeQ{1.0}, ref), NotImplementedInM1Error);

  REQUIRE_THROWS_AS(UnitConverter::mass_from_lj(1.0, ref), NotImplementedInM1Error);
  REQUIRE_THROWS_AS(UnitConverter::mass_to_lj(MassQ{1.0}, ref), NotImplementedInM1Error);

  REQUIRE_THROWS_AS(UnitConverter::force_from_lj(1.0, ref), NotImplementedInM1Error);
  REQUIRE_THROWS_AS(UnitConverter::force_to_lj(ForceQ{1.0}, ref), NotImplementedInM1Error);

  REQUIRE_THROWS_AS(UnitConverter::pressure_from_lj(1.0, ref), NotImplementedInM1Error);
  REQUIRE_THROWS_AS(UnitConverter::pressure_to_lj(PressureQ{1.0}, ref), NotImplementedInM1Error);

  REQUIRE_THROWS_AS(UnitConverter::velocity_from_lj(1.0, ref), NotImplementedInM1Error);
  REQUIRE_THROWS_AS(UnitConverter::velocity_to_lj(VelocityQ{1.0}, ref), NotImplementedInM1Error);

  REQUIRE_THROWS_AS(UnitConverter::temperature_from_lj(1.0, ref), NotImplementedInM1Error);
  REQUIRE_THROWS_AS(UnitConverter::temperature_to_lj(TemperatureQ{1.0}, ref),
                    NotImplementedInM1Error);
}

TEST_CASE("NotImplementedInM1Error carries actionable message", "[runtime][units][lj_stub]") {
  try {
    [[maybe_unused]] const auto q = UnitConverter::length_from_lj(1.0, LjReference{});
    FAIL("expected throw");
  } catch (const NotImplementedInM1Error& err) {
    const std::string what(err.what());
    REQUIRE(what.find("M1") != std::string::npos);
    REQUIRE(what.find("M2") != std::string::npos);
    REQUIRE(what.find("length") != std::string::npos);
  }
}

TEST_CASE("validate_supported_for_input accepts metal", "[runtime][units]") {
  REQUIRE_NOTHROW(UnitConverter::validate_supported_for_input(UnitSystem::Metal));
}

TEST_CASE("validate_supported_for_input rejects lj with NotImplementedInM1Error",
          "[runtime][units]") {
  REQUIRE_THROWS_AS(UnitConverter::validate_supported_for_input(UnitSystem::Lj),
                    NotImplementedInM1Error);
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
