#include "tdmd/integrator/velocity_verlet.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/species.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <vector>

namespace {

tdmd::SpeciesId register_al(tdmd::SpeciesRegistry& species) {
  tdmd::SpeciesInfo info;
  info.name = "Al";
  info.mass = 26.98;  // g/mol
  info.atomic_number = 13;
  return species.register_species(info);
}

// Zero all forces. Tests that don't use a force field still rely on this to
// keep the (uninitialised-after-drift) fx/fy/fz invariant — VV requires f(t+dt).
void zero_forces(tdmd::AtomSoA& atoms) {
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    atoms.fx[i] = 0.0;
    atoms.fy[i] = 0.0;
    atoms.fz[i] = 0.0;
  }
}

// Hand-rolled linear spring toward origin: f = -k · r. Used to exercise the
// integrator against a closed-form oscillator; `k` is in eV/Å².
void apply_spring_force(tdmd::AtomSoA& atoms, double k) {
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    atoms.fx[i] = -k * atoms.x[i];
    atoms.fy[i] = -k * atoms.y[i];
    atoms.fz[i] = -k * atoms.z[i];
  }
}

// Potential energy of the spring (for energy-conservation checks): 0.5 · k · |r|².
double spring_potential_energy(const tdmd::AtomSoA& atoms, double k) {
  double u = 0.0;
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    u += 0.5 * k * (atoms.x[i] * atoms.x[i] + atoms.y[i] * atoms.y[i] + atoms.z[i] * atoms.z[i]);
  }
  return u;
}

// Hand-rolled two-body Morse along x-axis (pair potential):
//   U(r) = D · [1 - exp(-α(r - r0))]²
//   F on atom i = +dU/dr · (r_j - r_i)/r
// Forces on two atoms are equal and opposite (Newton's third law).
// Parameters chosen so that effective relative-motion ω·dt is small — VV's peak
// energy oscillation scales with (dt·ω)², and the energy-conservation test asks
// for < 1e-5. For a pair, reduced mass μ = m/2 controls the relative motion:
// k_eff = 2·D·α², ω = √(k_eff·ftm2v/μ). With D=0.02, α=0.6 on Al,
// ω ≈ 3.2 ps⁻¹ → dt=0.001 ps → (dt·ω)² ≈ 1e-5 → expected peak drift ≈ 3e-6.
struct MorseParams {
  double D = 0.02;     // eV
  double alpha = 0.6;  // 1/Å
  double r0 = 2.5;     // Å
};

double morse_pair(tdmd::AtomSoA& atoms, const MorseParams& p) {
  const double dx = atoms.x[1] - atoms.x[0];
  const double dy = atoms.y[1] - atoms.y[0];
  const double dz = atoms.z[1] - atoms.z[0];
  const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
  const double e = std::exp(-p.alpha * (r - p.r0));
  const double one_minus_e = 1.0 - e;
  const double u = p.D * one_minus_e * one_minus_e;
  const double dU_dr = 2.0 * p.D * p.alpha * e * one_minus_e;

  // f_i = +dU/dr · hat(r_ij)  (attractive for r>r0, repulsive for r<r0)
  const double fx = dU_dr * dx / r;
  const double fy = dU_dr * dy / r;
  const double fz = dU_dr * dz / r;

  atoms.fx[0] = fx;
  atoms.fy[0] = fy;
  atoms.fz[0] = fz;
  atoms.fx[1] = -fx;
  atoms.fy[1] = -fy;
  atoms.fz[1] = -fz;
  return u;
}

}  // namespace

TEST_CASE("VelocityVerlet dt validation rejects non-positive and non-finite",
          "[integrator][vv][validation]") {
  tdmd::SpeciesRegistry species;
  const auto id = register_al(species);
  tdmd::AtomSoA atoms;
  atoms.add_atom(id, 0.0, 0.0, 0.0);
  zero_forces(atoms);

  tdmd::VelocityVerletIntegrator vv;
  REQUIRE_THROWS_AS(vv.pre_force_step(atoms, species, 0.0), std::invalid_argument);
  REQUIRE_THROWS_AS(vv.pre_force_step(atoms, species, -0.001), std::invalid_argument);
  REQUIRE_THROWS_AS(vv.pre_force_step(atoms, species, std::nan("")), std::invalid_argument);
  REQUIRE_THROWS_AS(vv.pre_force_step(atoms, species, std::numeric_limits<double>::infinity()),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(vv.post_force_step(atoms, species, 0.0), std::invalid_argument);
  REQUIRE_THROWS_AS(vv.post_force_step(atoms, species, -0.001), std::invalid_argument);
  REQUIRE_NOTHROW(vv.pre_force_step(atoms, species, 0.001));
  REQUIRE_NOTHROW(vv.post_force_step(atoms, species, 0.001));
}

TEST_CASE("VelocityVerlet free flight: zero force gives linear trajectory",
          "[integrator][vv][free]") {
  tdmd::SpeciesRegistry species;
  const auto id = register_al(species);
  tdmd::AtomSoA atoms;
  atoms.add_atom(id, 1.0, -2.0, 3.0, /*vx=*/0.5, /*vy=*/-0.25, /*vz=*/0.1);
  zero_forces(atoms);

  tdmd::VelocityVerletIntegrator vv;
  constexpr double dt = 0.01;
  constexpr int n_steps = 100;
  for (int s = 0; s < n_steps; ++s) {
    vv.pre_force_step(atoms, species, dt);
    // Zero force at the new position (free flight).
    zero_forces(atoms);
    vv.post_force_step(atoms, species, dt);
  }

  const double t = n_steps * dt;
  REQUIRE(atoms.x[0] == Catch::Approx(1.0 + 0.5 * t).epsilon(1e-14));
  REQUIRE(atoms.y[0] == Catch::Approx(-2.0 + -0.25 * t).epsilon(1e-14));
  REQUIRE(atoms.z[0] == Catch::Approx(3.0 + 0.1 * t).epsilon(1e-14));
  REQUIRE(atoms.vx[0] == Catch::Approx(0.5).epsilon(1e-14));
  REQUIRE(atoms.vy[0] == Catch::Approx(-0.25).epsilon(1e-14));
  REQUIRE(atoms.vz[0] == Catch::Approx(0.1).epsilon(1e-14));
}

TEST_CASE("VelocityVerlet harmonic oscillator conserves energy and matches analytic period",
          "[integrator][vv][harmonic]") {
  tdmd::SpeciesRegistry species;
  const auto id = register_al(species);
  tdmd::AtomSoA atoms;
  atoms.add_atom(id, 0.1, 0.0, 0.0);  // displacement along x, zero initial v

  const double m = species.get_info(id).mass;
  // Soft spring: k=0.05 eV/Å² gives ω ≈ 4.23 ps⁻¹ on Al, so with dt=0.001 ps
  // the VV peak energy oscillation ≈ (dt·ω)²/12 ≈ 1.5e-6 — safely below 1e-5.
  const double k = 0.05;  // eV/Å²
  // Effective ω² = k · ftm2v / m → analytic period T_analytic.
  const double omega = std::sqrt(k * tdmd::kMetalFtm2v / m);
  const double T_analytic = 2.0 * std::numbers::pi / omega;
  constexpr double dt = 0.001;
  const int steps_per_period = static_cast<int>(T_analytic / dt + 0.5);
  REQUIRE(steps_per_period > 100);  // Sanity: enough resolution for VV.

  tdmd::VelocityVerletIntegrator vv;
  apply_spring_force(atoms, k);
  const double E0 = tdmd::kinetic_energy(atoms, species) + spring_potential_energy(atoms, k);

  // Integrate for 5 periods and track zero-crossings of x (positive → negative).
  int crossings = 0;
  double t_last_crossing = 0.0;
  double t_first_crossing = -1.0;
  double prev_x = atoms.x[0];
  double max_energy_drift = 0.0;
  const int total_steps = 5 * steps_per_period;
  for (int s = 1; s <= total_steps; ++s) {
    vv.pre_force_step(atoms, species, dt);
    apply_spring_force(atoms, k);
    vv.post_force_step(atoms, species, dt);

    const double E = tdmd::kinetic_energy(atoms, species) + spring_potential_energy(atoms, k);
    const double drift = std::abs(E - E0) / std::abs(E0);
    max_energy_drift = std::max(max_energy_drift, drift);

    const double t = s * dt;
    if (prev_x > 0.0 && atoms.x[0] <= 0.0) {
      // Linear-interpolate the crossing time.
      const double t_cross = t - dt * atoms.x[0] / (atoms.x[0] - prev_x);
      if (t_first_crossing < 0.0) {
        t_first_crossing = t_cross;
      }
      t_last_crossing = t_cross;
      ++crossings;
    }
    prev_x = atoms.x[0];
  }

  // After 5 periods, we expect 5 "x goes from + to −" crossings. Period is the
  // spacing between crossings.
  REQUIRE(crossings >= 4);
  const double T_measured = (t_last_crossing - t_first_crossing) / (crossings - 1);
  REQUIRE(std::abs(T_measured - T_analytic) / T_analytic < 0.01);  // 1%
  REQUIRE(max_energy_drift < 1e-5);
}

TEST_CASE("VelocityVerlet two-body Morse: energy drift < 1e-5 over 10⁴ steps",
          "[integrator][vv][morse][energy]") {
  tdmd::SpeciesRegistry species;
  const auto id = register_al(species);
  tdmd::AtomSoA atoms;
  // Start slightly displaced from equilibrium (r₀=2.5) so the dimer oscillates;
  // small displacement keeps anharmonic corrections well below the drift budget.
  atoms.add_atom(id, 0.0, 0.0, 0.0);
  atoms.add_atom(id, 2.45, 0.0, 0.0);

  const MorseParams p;
  double u = morse_pair(atoms, p);
  const double E0 = tdmd::kinetic_energy(atoms, species) + u;

  tdmd::VelocityVerletIntegrator vv;
  constexpr double dt = 0.001;
  constexpr int n_steps = 10000;
  double max_drift = 0.0;
  for (int s = 0; s < n_steps; ++s) {
    vv.pre_force_step(atoms, species, dt);
    u = morse_pair(atoms, p);
    vv.post_force_step(atoms, species, dt);

    const double E = tdmd::kinetic_energy(atoms, species) + u;
    max_drift = std::max(max_drift, std::abs(E - E0) / std::abs(E0));
  }
  REQUIRE(max_drift < 1e-5);
}

TEST_CASE("VelocityVerlet time reversal: velocity negation returns to initial state",
          "[integrator][vv][reversal]") {
  tdmd::SpeciesRegistry species;
  const auto id = register_al(species);
  tdmd::AtomSoA atoms;
  atoms.add_atom(id, 0.0, 0.0, 0.0);
  atoms.add_atom(id, 2.4, 0.0, 0.0);

  const MorseParams p;
  std::vector<double> x0, y0, z0;
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    x0.push_back(atoms.x[i]);
    y0.push_back(atoms.y[i]);
    z0.push_back(atoms.z[i]);
  }
  morse_pair(atoms, p);

  tdmd::VelocityVerletIntegrator vv;
  constexpr double dt = 0.001;
  constexpr int n_steps = 500;

  for (int s = 0; s < n_steps; ++s) {
    vv.pre_force_step(atoms, species, dt);
    morse_pair(atoms, p);
    vv.post_force_step(atoms, species, dt);
  }

  // Negate velocities; forces at current positions re-evaluated in next iter.
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    atoms.vx[i] = -atoms.vx[i];
    atoms.vy[i] = -atoms.vy[i];
    atoms.vz[i] = -atoms.vz[i];
  }

  for (int s = 0; s < n_steps; ++s) {
    vv.pre_force_step(atoms, species, dt);
    morse_pair(atoms, p);
    vv.post_force_step(atoms, species, dt);
  }

  for (std::size_t i = 0; i < atoms.size(); ++i) {
    REQUIRE(std::abs(atoms.x[i] - x0[i]) < 1e-10);
    REQUIRE(std::abs(atoms.y[i] - y0[i]) < 1e-10);
    REQUIRE(std::abs(atoms.z[i] - z0[i]) < 1e-10);
  }
}

TEST_CASE("VelocityVerlet is deterministic: same input yields bit-identical output",
          "[integrator][vv][determinism]") {
  auto run = []() {
    tdmd::SpeciesRegistry species;
    const auto id = register_al(species);
    tdmd::AtomSoA atoms;
    atoms.add_atom(id, 0.0, 0.0, 0.0);
    atoms.add_atom(id, 2.3, 0.0, 0.0);
    atoms.add_atom(id, 0.0, 2.6, 0.0);

    const MorseParams p;
    morse_pair(atoms, p);

    tdmd::VelocityVerletIntegrator vv;
    constexpr double dt = 0.0005;
    for (int s = 0; s < 200; ++s) {
      vv.pre_force_step(atoms, species, dt);
      morse_pair(atoms, p);
      vv.post_force_step(atoms, species, dt);
    }
    std::vector<double> snapshot;
    for (std::size_t i = 0; i < atoms.size(); ++i) {
      snapshot.push_back(atoms.x[i]);
      snapshot.push_back(atoms.y[i]);
      snapshot.push_back(atoms.z[i]);
      snapshot.push_back(atoms.vx[i]);
      snapshot.push_back(atoms.vy[i]);
      snapshot.push_back(atoms.vz[i]);
    }
    return snapshot;
  };

  const auto a = run();
  const auto b = run();
  REQUIRE(a == b);
}

TEST_CASE("kinetic_energy returns 0 for empty system", "[integrator][vv][ke]") {
  tdmd::SpeciesRegistry species;
  (void) register_al(species);
  tdmd::AtomSoA atoms;
  REQUIRE(tdmd::kinetic_energy(atoms, species) == 0.0);
}

TEST_CASE("kinetic_energy matches hand calculation in metal units", "[integrator][vv][ke]") {
  tdmd::SpeciesRegistry species;
  const auto id = register_al(species);
  tdmd::AtomSoA atoms;
  atoms.add_atom(id, 0.0, 0.0, 0.0, /*vx=*/1.0, /*vy=*/2.0, /*vz=*/3.0);
  atoms.add_atom(id, 1.0, 0.0, 0.0, /*vx=*/0.0, /*vy=*/0.0, /*vz=*/0.0);

  // KE = 0.5 · 26.98 · (1+4+9) · mvv2e = 0.5 · 26.98 · 14 · 1.0364269e-4 eV.
  const double expected = 0.5 * 26.98 * 14.0 * tdmd::kMetalMvv2e;
  REQUIRE(tdmd::kinetic_energy(atoms, species) == Catch::Approx(expected).epsilon(1e-14));
}
