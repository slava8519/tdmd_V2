#include "tdmd/runtime/unit_converter.hpp"

#include <sstream>

namespace tdmd {

std::string_view to_string(UnitSystem s) noexcept {
  switch (s) {
    case UnitSystem::Metal:
      return "metal";
    case UnitSystem::Lj:
      return "lj";
    case UnitSystem::Real:
      return "real";
    case UnitSystem::Cgs:
      return "cgs";
    case UnitSystem::Si:
      return "si";
  }
  return "<unknown>";
}

namespace {

[[noreturn]] void throw_lj_stub(std::string_view dimension) {
  std::ostringstream msg;
  msg << "UnitConverter::" << dimension << "_{from,to}_lj: not implemented in M1 (D-M1-6). "
      << "Scheduled for M2 — see docs/development/m1_execution_pack.md T1.2.";
  throw NotImplementedInM1Error(msg.str());
}

}  // namespace

LengthQ UnitConverter::length_from_lj(double /*value*/, const LjReference& /*ref*/) {
  throw_lj_stub("length");
}
double UnitConverter::length_to_lj(LengthQ /*q*/, const LjReference& /*ref*/) {
  throw_lj_stub("length");
}

EnergyQ UnitConverter::energy_from_lj(double /*value*/, const LjReference& /*ref*/) {
  throw_lj_stub("energy");
}
double UnitConverter::energy_to_lj(EnergyQ /*q*/, const LjReference& /*ref*/) {
  throw_lj_stub("energy");
}

TimeQ UnitConverter::time_from_lj(double /*value*/, const LjReference& /*ref*/) {
  throw_lj_stub("time");
}
double UnitConverter::time_to_lj(TimeQ /*q*/, const LjReference& /*ref*/) {
  throw_lj_stub("time");
}

MassQ UnitConverter::mass_from_lj(double /*value*/, const LjReference& /*ref*/) {
  throw_lj_stub("mass");
}
double UnitConverter::mass_to_lj(MassQ /*q*/, const LjReference& /*ref*/) {
  throw_lj_stub("mass");
}

ForceQ UnitConverter::force_from_lj(double /*value*/, const LjReference& /*ref*/) {
  throw_lj_stub("force");
}
double UnitConverter::force_to_lj(ForceQ /*q*/, const LjReference& /*ref*/) {
  throw_lj_stub("force");
}

PressureQ UnitConverter::pressure_from_lj(double /*value*/, const LjReference& /*ref*/) {
  throw_lj_stub("pressure");
}
double UnitConverter::pressure_to_lj(PressureQ /*q*/, const LjReference& /*ref*/) {
  throw_lj_stub("pressure");
}

VelocityQ UnitConverter::velocity_from_lj(double /*value*/, const LjReference& /*ref*/) {
  throw_lj_stub("velocity");
}
double UnitConverter::velocity_to_lj(VelocityQ /*q*/, const LjReference& /*ref*/) {
  throw_lj_stub("velocity");
}

TemperatureQ UnitConverter::temperature_from_lj(double /*value*/, const LjReference& /*ref*/) {
  throw_lj_stub("temperature");
}
double UnitConverter::temperature_to_lj(TemperatureQ /*q*/, const LjReference& /*ref*/) {
  throw_lj_stub("temperature");
}

void UnitConverter::validate_supported_for_input(UnitSystem s) {
  switch (s) {
    case UnitSystem::Metal:
      return;
    case UnitSystem::Lj:
      throw NotImplementedInM1Error(
          "UnitConverter: 'lj' input system is not implemented in M1 (D-M1-6). "
          "Scheduled for M2.");
    case UnitSystem::Real:
      throw IncompatibleUnitError(
          "UnitConverter: 'real' unit system is recognized but not supported in v1 "
          "(master spec §5.3). Converter path: post-v1.");
    case UnitSystem::Cgs:
      throw IncompatibleUnitError(
          "UnitConverter: 'cgs' unit system is recognized but not supported in v1.");
    case UnitSystem::Si:
      throw IncompatibleUnitError(
          "UnitConverter: 'si' unit system is never supported (master spec §5.3 "
          "policy).");
  }
  throw IncompatibleUnitError("UnitConverter: unknown UnitSystem value");
}

}  // namespace tdmd
