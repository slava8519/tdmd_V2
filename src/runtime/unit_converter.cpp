#include "tdmd/runtime/unit_converter.hpp"

#include "tdmd/integrator/velocity_verlet.hpp"
#include "tdmd/runtime/physical_constants.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

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

// ---------------------------------------------------------------------------
// LJ ↔ metal conversion constants.
//
// LAMMPS's `units lj` treats every quantity as dimensionless relative to
// (σ, ε, m). TDMD's internal representation is always `metal` (master spec
// §5.3), so every ingest/emit path through lj passes through these factors.
//
// length_metal    = l_lj · σ                           [Å]
// energy_metal    = E_lj · ε                           [eV]
// mass_metal      = m_lj · m_ref                       [g/mol]
// force_metal     = f_lj · (ε/σ)                       [eV/Å]
// temperature_metal = T_lj · (ε/kB)                    [K]
// pressure_metal  = P_lj · (ε/σ³) · EV_PER_A3_TO_BAR   [bar]
// time_metal      = t_lj · sqrt(m·σ²/ε) · LJ_TIME      [ps]
// velocity_metal  = v_lj · sqrt(ε/m)    · LJ_VEL       [Å/ps]
//
// LJ_TIME and LJ_VEL are exact inverses of each other. Their value derives
// from the LAMMPS `metal` mvv2e factor (see `integrator/velocity_verlet.hpp`),
// so a round-trip lj → metal → lj through TDMD reuses the same numerical
// constants the integrator already uses internally.
// ---------------------------------------------------------------------------
namespace {

// Bar per (eV/Å³). Derived from CODATA 2019 exact:
//   1 eV = 1.602176634e-19 J, 1 Å = 1e-10 m, 1 bar = 1e5 Pa.
//   => 1 eV/Å³ = 1.602176634e11 Pa = 1.602176634e6 bar (exact).
constexpr double kPressure_eV_per_A3_to_bar = 1.602176634e6;

// Time factor: ps per sqrt((g/mol)·Å²/eV).
// Equal to sqrt(kMetalMvv2e) where kMetalMvv2e = 1.0364269e-4 (LAMMPS metal).
// Computed at program startup rather than constexpr because std::sqrt is not
// constexpr until C++26; the `static const` storage guarantees a single
// initialization with thread-safe semantics (C++11 [stmt.dcl]/4).
static const double kLjTimeFactor_ps = std::sqrt(kMetalMvv2e);

// Velocity factor: (Å/ps) per sqrt(eV/(g/mol)).
// Exact inverse of kLjTimeFactor_ps (verified in unit tests).
static const double kLjVelocityFactor_APerPs = 1.0 / kLjTimeFactor_ps;

void validate_reference(const LjReference& ref) {
  if (!(ref.sigma > 0.0)) {
    std::ostringstream msg;
    msg << "UnitConverter: LjReference.sigma must be > 0 (got " << ref.sigma << ").";
    throw std::invalid_argument(msg.str());
  }
  if (!(ref.epsilon > 0.0)) {
    std::ostringstream msg;
    msg << "UnitConverter: LjReference.epsilon must be > 0 (got " << ref.epsilon << ").";
    throw std::invalid_argument(msg.str());
  }
  if (!(ref.mass > 0.0)) {
    std::ostringstream msg;
    msg << "UnitConverter: LjReference.mass must be > 0 (got " << ref.mass << ").";
    throw std::invalid_argument(msg.str());
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// length: metal = lj · σ
// ---------------------------------------------------------------------------
LengthQ UnitConverter::length_from_lj(double value, const LjReference& ref) {
  validate_reference(ref);
  return {value * ref.sigma};
}
double UnitConverter::length_to_lj(LengthQ q, const LjReference& ref) {
  validate_reference(ref);
  return q.metal_angstroms / ref.sigma;
}

// ---------------------------------------------------------------------------
// energy: metal = lj · ε
// ---------------------------------------------------------------------------
EnergyQ UnitConverter::energy_from_lj(double value, const LjReference& ref) {
  validate_reference(ref);
  return {value * ref.epsilon};
}
double UnitConverter::energy_to_lj(EnergyQ q, const LjReference& ref) {
  validate_reference(ref);
  return q.metal_eV / ref.epsilon;
}

// ---------------------------------------------------------------------------
// time: metal = lj · sqrt(m·σ²/ε) · kLjTimeFactor_ps
// ---------------------------------------------------------------------------
TimeQ UnitConverter::time_from_lj(double value, const LjReference& ref) {
  validate_reference(ref);
  const double tau = std::sqrt(ref.mass * ref.sigma * ref.sigma / ref.epsilon);
  return {value * tau * kLjTimeFactor_ps};
}
double UnitConverter::time_to_lj(TimeQ q, const LjReference& ref) {
  validate_reference(ref);
  const double tau = std::sqrt(ref.mass * ref.sigma * ref.sigma / ref.epsilon);
  return q.metal_ps / (tau * kLjTimeFactor_ps);
}

// ---------------------------------------------------------------------------
// mass: metal = lj · m_ref
// ---------------------------------------------------------------------------
MassQ UnitConverter::mass_from_lj(double value, const LjReference& ref) {
  validate_reference(ref);
  return {value * ref.mass};
}
double UnitConverter::mass_to_lj(MassQ q, const LjReference& ref) {
  validate_reference(ref);
  return q.metal_g_per_mol / ref.mass;
}

// ---------------------------------------------------------------------------
// force: metal = lj · (ε/σ)
// No unit-conversion factor: eV/Å is already the metal force unit.
// ---------------------------------------------------------------------------
ForceQ UnitConverter::force_from_lj(double value, const LjReference& ref) {
  validate_reference(ref);
  return {value * ref.epsilon / ref.sigma};
}
double UnitConverter::force_to_lj(ForceQ q, const LjReference& ref) {
  validate_reference(ref);
  return q.metal_eV_per_A * ref.sigma / ref.epsilon;
}

// ---------------------------------------------------------------------------
// pressure: metal = lj · (ε/σ³) · kPressure_eV_per_A3_to_bar
// ---------------------------------------------------------------------------
PressureQ UnitConverter::pressure_from_lj(double value, const LjReference& ref) {
  validate_reference(ref);
  const double sigma3 = ref.sigma * ref.sigma * ref.sigma;
  return {value * ref.epsilon / sigma3 * kPressure_eV_per_A3_to_bar};
}
double UnitConverter::pressure_to_lj(PressureQ q, const LjReference& ref) {
  validate_reference(ref);
  const double sigma3 = ref.sigma * ref.sigma * ref.sigma;
  return q.metal_bar * sigma3 / (ref.epsilon * kPressure_eV_per_A3_to_bar);
}

// ---------------------------------------------------------------------------
// velocity: metal = lj · sqrt(ε/m) · kLjVelocityFactor_APerPs
// ---------------------------------------------------------------------------
VelocityQ UnitConverter::velocity_from_lj(double value, const LjReference& ref) {
  validate_reference(ref);
  const double v_scale = std::sqrt(ref.epsilon / ref.mass);
  return {value * v_scale * kLjVelocityFactor_APerPs};
}
double UnitConverter::velocity_to_lj(VelocityQ q, const LjReference& ref) {
  validate_reference(ref);
  const double v_scale = std::sqrt(ref.epsilon / ref.mass);
  return q.metal_A_per_ps / (v_scale * kLjVelocityFactor_APerPs);
}

// ---------------------------------------------------------------------------
// temperature: metal = lj · (ε/kB)
// Uses TDMD's CODATA-2019 kB; consistent with the rest of TDMD physics.
// ---------------------------------------------------------------------------
TemperatureQ UnitConverter::temperature_from_lj(double value, const LjReference& ref) {
  validate_reference(ref);
  return {value * ref.epsilon / kBoltzmann_eV_per_K};
}
double UnitConverter::temperature_to_lj(TemperatureQ q, const LjReference& ref) {
  validate_reference(ref);
  return q.metal_K * kBoltzmann_eV_per_K / ref.epsilon;
}

void UnitConverter::validate_supported_for_input(UnitSystem s) {
  switch (s) {
    case UnitSystem::Metal:
    case UnitSystem::Lj:
      return;
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
