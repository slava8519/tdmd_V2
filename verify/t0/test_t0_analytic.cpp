// T0 morse-analytic benchmark — pure-C++ Catch2 test.
//
// Unlike T1 (which shells out to a Python driver because it needs LAMMPS as
// an oracle), T0's ground truth is a closed-form formula. No external process,
// no .log parsing. The test loads both the metal and the lj variant of the
// benchmark, inits a `SimulationEngine` in-process, reads forces directly from
// `engine.atoms()` (exposed by T1.9), and compares against the analytic Morse
// force / PE.
//
// The closed-form Morse pair potential, in the LAMMPS convention that TDMD
// matches (see src/potentials/morse.cpp line 113), has U(r0) = -D:
//
//     delta = r - r0
//     e     = exp(-alpha * delta)
//     U(r)  = D * (1 - e)^2 - D      = D * [e^2 - 2*e]
//     |F|   = -dU/dr = 2 * D * alpha * (1 - e) * e
//
// The "(1-e)^2" textbook form (with U(r0)=0) differs by the additive constant
// D; forces are identical. Getting the form right matters here because the
// test asserts PE at 1e-12 rel.
//
// SPEC: docs/specs/verify/SPEC.md §4.1 (T0 overview), §6 (reference data),
//       §7 (harness). Exec pack: docs/development/m2_execution_pack.md T2.3.

#include "tdmd/io/yaml_config.hpp"
#include "tdmd/runtime/simulation_engine.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <string>

#ifndef TDMD_REPO_ROOT
#error "TDMD_REPO_ROOT must be defined by the build system"
#endif

namespace {

struct AnalyticReference {
  double r;          // Å
  double force_mag;  // eV/Å, |F| on either atom (signed by +/- x̂ below)
  double pe;         // eV
};

// Closed-form Morse for a single pair. Keep in one place so the test and the
// threshold rationale agree on exactly which formula is canonical.
AnalyticReference compute_analytic(double D, double alpha, double r0, double r) {
  AnalyticReference out{};
  const double delta = r - r0;
  const double e = std::exp(-alpha * delta);
  const double one_minus_e = 1.0 - e;
  out.r = r;
  out.force_mag = 2.0 * D * alpha * one_minus_e * e;
  // LAMMPS/TDMD convention: U(r0) = -D. Under hard_cutoff the shift terms
  // (e_pair_rc, g_rc) are identically zero, so this is the whole story.
  out.pe = D * one_minus_e * one_minus_e - D;
  return out;
}

std::filesystem::path benchmark_dir() {
  namespace fs = std::filesystem;
  return fs::path(TDMD_REPO_ROOT) / "verify" / "benchmarks" / "t0_morse_analytic";
}

// Run a single variant (metal or lj config) and return the forces + PE. The
// engine is local to this helper — each call starts from a fresh state so the
// two variants don't share hidden coupling.
struct VariantResult {
  double fx0, fy0, fz0;
  double fx1, fy1, fz1;
  double x0, y0, z0;
  double x1, y1, z1;
  double mass0, mass1;
  double pe;
};

VariantResult run_variant(const std::string& config_name) {
  namespace fs = std::filesystem;
  const fs::path cfg_path = benchmark_dir() / config_name;
  auto config = tdmd::io::parse_yaml_config(cfg_path.string());

  tdmd::SimulationEngine engine;
  engine.init(config, benchmark_dir().string());

  const auto row = engine.run(0, nullptr);  // n_steps=0 → just returns the initial thermo row.

  const auto& atoms = engine.atoms();
  REQUIRE(atoms.size() == 2);

  // Atom indices after init may not match the .data file order (cell-grid
  // reorder). Sort by id so tests are independent of internal layout.
  std::size_t i0 = 0;
  std::size_t i1 = 1;
  if (atoms.id[0] > atoms.id[1]) {
    i0 = 1;
    i1 = 0;
  }

  VariantResult r{};
  r.fx0 = atoms.fx[i0];
  r.fy0 = atoms.fy[i0];
  r.fz0 = atoms.fz[i0];
  r.fx1 = atoms.fx[i1];
  r.fy1 = atoms.fy[i1];
  r.fz1 = atoms.fz[i1];
  r.x0 = atoms.x[i0];
  r.y0 = atoms.y[i0];
  r.z0 = atoms.z[i0];
  r.x1 = atoms.x[i1];
  r.y1 = atoms.y[i1];
  r.z1 = atoms.z[i1];
  r.mass0 = engine.species().get_info(atoms.type[i0]).mass;
  r.mass1 = engine.species().get_info(atoms.type[i1]).mass;
  r.pe = row.potential_energy;
  return r;
}

// Thin wrapper over Catch2's matchers so residuals print the actual relative
// error instead of just a bool.
void expect_rel(double actual, double expected, double rel_tol, const char* label) {
  const double denom = std::abs(expected) > 0 ? std::abs(expected) : 1.0;
  const double residual = std::abs(actual - expected) / denom;
  INFO(label << ": actual=" << actual << " expected=" << expected << " rel=" << residual);
  REQUIRE(residual <= rel_tol);
}

}  // namespace

TEST_CASE("T0 morse-analytic: metal variant matches closed-form F + PE",
          "[verify][analytic][t0][metal]") {
  // Physical parameters — kept in lockstep with config_metal.yaml. If the
  // YAML drifts, the analytic check silently re-centers; this is intentional:
  // the tolerance is small enough (1e-12) that a real param mismatch still
  // registers as a failure, but the test remains readable.
  constexpr double D = 1.0;
  constexpr double alpha = 1.0;
  constexpr double r0 = 3.0;
  constexpr double expected_r = 3.5;
  constexpr double rel_tol = 1.0e-12;

  const auto result = run_variant("config_metal.yaml");
  const auto ref = compute_analytic(D, alpha, r0, expected_r);

  const double r_actual = result.x1 - result.x0;
  REQUIRE_THAT(r_actual, Catch::Matchers::WithinAbs(expected_r, 1e-12));

  expect_rel(result.fx0, +ref.force_mag, rel_tol, "F_x(atom 0)");
  expect_rel(result.fx1, -ref.force_mag, rel_tol, "F_x(atom 1)");
  REQUIRE(result.fy0 == 0.0);
  REQUIRE(result.fz0 == 0.0);
  REQUIRE(result.fy1 == 0.0);
  REQUIRE(result.fz1 == 0.0);

  expect_rel(result.pe, ref.pe, rel_tol, "potential_energy");
}

TEST_CASE("T0 morse-analytic: lj variant (identity reference) matches closed-form",
          "[verify][analytic][t0][lj]") {
  // With sigma=epsilon=mass=1, every metal value survives the conversion
  // bitwise (length/energy/mass identity). Velocity/time carry sqrt(mvv2e)
  // but are irrelevant at n_steps=0 with v=0.
  constexpr double D = 1.0;
  constexpr double alpha = 1.0;
  constexpr double r0 = 3.0;
  constexpr double expected_r = 3.5;
  constexpr double rel_tol = 1.0e-12;

  const auto result = run_variant("config_lj.yaml");
  const auto ref = compute_analytic(D, alpha, r0, expected_r);

  expect_rel(result.fx0, +ref.force_mag, rel_tol, "F_x(atom 0)");
  expect_rel(result.fx1, -ref.force_mag, rel_tol, "F_x(atom 1)");
  REQUIRE(result.fy0 == 0.0);
  REQUIRE(result.fz0 == 0.0);
  REQUIRE(result.fy1 == 0.0);
  REQUIRE(result.fz1 == 0.0);

  expect_rel(result.pe, ref.pe, rel_tol, "potential_energy");
}

TEST_CASE("T0 morse-analytic: metal and lj-identity variants match bitwise",
          "[verify][analytic][t0][crosscheck]") {
  const auto m = run_variant("config_metal.yaml");
  const auto l = run_variant("config_lj.yaml");

  // Positions, masses, forces, and PE must be bitwise-identical. `==` on
  // doubles is deliberate here — the whole point of identity-reference lj is
  // that the conversion is a mathematical no-op, not "close enough".
  REQUIRE(m.x0 == l.x0);
  REQUIRE(m.y0 == l.y0);
  REQUIRE(m.z0 == l.z0);
  REQUIRE(m.x1 == l.x1);
  REQUIRE(m.y1 == l.y1);
  REQUIRE(m.z1 == l.z1);

  REQUIRE(m.mass0 == l.mass0);
  REQUIRE(m.mass1 == l.mass1);

  REQUIRE(m.fx0 == l.fx0);
  REQUIRE(m.fy0 == l.fy0);
  REQUIRE(m.fz0 == l.fz0);
  REQUIRE(m.fx1 == l.fx1);
  REQUIRE(m.fy1 == l.fy1);
  REQUIRE(m.fz1 == l.fz1);

  REQUIRE(m.pe == l.pe);
}
