#pragma once

// SPEC: TDMD master spec §5.3 (unit system support policy)
// Exec pack: docs/development/m1_execution_pack.md T1.2 (D-M1-6)
// Module home: src/runtime/ (see docs/specs/runtime/SPEC.md §2)
//
// `UnitConverter` is the single conversion point between external unit
// systems (LAMMPS `metal`, `lj`, ...) and TDMD's internal native
// representation. Per master spec §5.3 the internal representation is
// **always `metal`** (Å / eV / g·mol⁻¹ / ps); no other module does raw unit
// arithmetic.
//
// Strong-typed quantities (`LengthQ`, `EnergyQ`, ...) prevent cross-dimension
// mistakes at compile time: a function that takes `LengthQ` cannot be called
// with `EnergyQ` or with a bare `double`.
//
// M1 scope (this file):
//   - full `metal` support for 8 dimensions;
//   - `lj` functions are API-stable stubs that throw `NotImplementedInM1Error`.
// M2 will add real `lj` conversions without changing any signature.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace tdmd {

enum class UnitSystem : std::uint8_t {
  Metal,  // LAMMPS metal: Å, eV, g/mol, ps
  Lj,     // LAMMPS lj:    reduced (sigma, epsilon, mass) — M2 impl
  Real,   // LAMMPS real:  recognized, not supported in v1
  Cgs,    // cgs:          recognized, not supported in v1
  Si,     // SI:           never supported (master spec §5.3 policy)
};

[[nodiscard]] std::string_view to_string(UnitSystem s) noexcept;

// Strong typedefs for dimensional safety. The member `metal_<unit>` stores the
// quantity in native (metal) units. Default-constructed values are zero; the
// wrappers are trivially copyable and can be used in aggregates.
struct LengthQ {
  double metal_angstroms = 0.0;
};
struct EnergyQ {
  double metal_eV = 0.0;
};
struct TimeQ {
  double metal_ps = 0.0;
};
struct MassQ {
  double metal_g_per_mol = 0.0;
};
struct ForceQ {
  double metal_eV_per_A = 0.0;
};
struct PressureQ {
  double metal_bar = 0.0;
};
struct VelocityQ {
  double metal_A_per_ps = 0.0;
};
struct TemperatureQ {
  double metal_K = 0.0;
};

// Reference parameters required for LJ ↔ metal conversion (LAMMPS convention:
// length = sigma·Å, energy = epsilon·eV, mass = mass·g/mol). Used by M2's lj
// adapter; in M1 the stub functions still accept this struct for API stability.
struct LjReference {
  double sigma = 1.0;    // Å
  double epsilon = 1.0;  // eV
  double mass = 1.0;     // g/mol
};

// Thrown by `UnitConverter::*_from_lj / *_to_lj` in M1. Upgraded to real
// conversion in M2 (exec pack D-M1-6).
class NotImplementedInM1Error : public std::runtime_error {
public:
  explicit NotImplementedInM1Error(std::string_view what_arg)
      : std::runtime_error(std::string(what_arg)) {}
};

// Thrown on dimensional violations we cannot catch at compile time (e.g.
// runtime-selected unit systems that don't apply to the requested quantity).
class IncompatibleUnitError : public std::invalid_argument {
public:
  explicit IncompatibleUnitError(std::string_view what_arg)
      : std::invalid_argument(std::string(what_arg)) {}
};

class UnitConverter {
public:
  UnitConverter() = default;

  // Internal representation of TDMD. Master spec §5.3 fixes this to `Metal`
  // for v1; the accessor exists so later milestones can change the internal
  // representation without touching callers (not expected in v1).
  [[nodiscard]] static constexpr UnitSystem internal_system() noexcept { return UnitSystem::Metal; }

  // ---------------------------------------------------------------------------
  // metal ↔ internal — identity in M1. constexpr for zero-overhead calls from
  // hot paths and to allow compile-time checks.
  // ---------------------------------------------------------------------------
  [[nodiscard]] static constexpr LengthQ length_from_metal(double v) noexcept { return {v}; }
  [[nodiscard]] static constexpr double length_to_metal(LengthQ q) noexcept {
    return q.metal_angstroms;
  }

  [[nodiscard]] static constexpr EnergyQ energy_from_metal(double v) noexcept { return {v}; }
  [[nodiscard]] static constexpr double energy_to_metal(EnergyQ q) noexcept { return q.metal_eV; }

  [[nodiscard]] static constexpr TimeQ time_from_metal(double v) noexcept { return {v}; }
  [[nodiscard]] static constexpr double time_to_metal(TimeQ q) noexcept { return q.metal_ps; }

  [[nodiscard]] static constexpr MassQ mass_from_metal(double v) noexcept { return {v}; }
  [[nodiscard]] static constexpr double mass_to_metal(MassQ q) noexcept {
    return q.metal_g_per_mol;
  }

  [[nodiscard]] static constexpr ForceQ force_from_metal(double v) noexcept { return {v}; }
  [[nodiscard]] static constexpr double force_to_metal(ForceQ q) noexcept {
    return q.metal_eV_per_A;
  }

  [[nodiscard]] static constexpr PressureQ pressure_from_metal(double v) noexcept { return {v}; }
  [[nodiscard]] static constexpr double pressure_to_metal(PressureQ q) noexcept {
    return q.metal_bar;
  }

  [[nodiscard]] static constexpr VelocityQ velocity_from_metal(double v) noexcept { return {v}; }
  [[nodiscard]] static constexpr double velocity_to_metal(VelocityQ q) noexcept {
    return q.metal_A_per_ps;
  }

  [[nodiscard]] static constexpr TemperatureQ temperature_from_metal(double v) noexcept {
    return {v};
  }
  [[nodiscard]] static constexpr double temperature_to_metal(TemperatureQ q) noexcept {
    return q.metal_K;
  }

  // ---------------------------------------------------------------------------
  // lj ↔ internal — API stable stubs. Every call throws NotImplementedInM1Error
  // in M1. Signatures will not change in M2. Reference parameters are accepted
  // today so callers written now compile and run correctly once M2 lands.
  // ---------------------------------------------------------------------------
  [[nodiscard]] static LengthQ length_from_lj(double value, const LjReference& ref);
  [[nodiscard]] static double length_to_lj(LengthQ q, const LjReference& ref);

  [[nodiscard]] static EnergyQ energy_from_lj(double value, const LjReference& ref);
  [[nodiscard]] static double energy_to_lj(EnergyQ q, const LjReference& ref);

  [[nodiscard]] static TimeQ time_from_lj(double value, const LjReference& ref);
  [[nodiscard]] static double time_to_lj(TimeQ q, const LjReference& ref);

  [[nodiscard]] static MassQ mass_from_lj(double value, const LjReference& ref);
  [[nodiscard]] static double mass_to_lj(MassQ q, const LjReference& ref);

  [[nodiscard]] static ForceQ force_from_lj(double value, const LjReference& ref);
  [[nodiscard]] static double force_to_lj(ForceQ q, const LjReference& ref);

  [[nodiscard]] static PressureQ pressure_from_lj(double value, const LjReference& ref);
  [[nodiscard]] static double pressure_to_lj(PressureQ q, const LjReference& ref);

  [[nodiscard]] static VelocityQ velocity_from_lj(double value, const LjReference& ref);
  [[nodiscard]] static double velocity_to_lj(VelocityQ q, const LjReference& ref);

  [[nodiscard]] static TemperatureQ temperature_from_lj(double value, const LjReference& ref);
  [[nodiscard]] static double temperature_to_lj(TemperatureQ q, const LjReference& ref);

  // Runtime dimensional check — throws IncompatibleUnitError if the requested
  // unit system does not apply to this module (e.g. asking for SI). Used by
  // io/preflight before committing to a unit system choice.
  static void validate_supported_for_input(UnitSystem s);
};

// Compile-time sanity: strong typedefs must be distinct types and not
// implicitly convertible to each other or to `double`.
static_assert(!std::is_same_v<LengthQ, EnergyQ>);
static_assert(!std::is_convertible_v<LengthQ, EnergyQ>);
static_assert(!std::is_convertible_v<LengthQ, double>);
static_assert(!std::is_convertible_v<double, LengthQ>);
static_assert(std::is_trivially_copyable_v<LengthQ>);

}  // namespace tdmd
