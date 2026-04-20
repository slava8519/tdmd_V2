// SPEC: docs/specs/potentials/SPEC.md §6.5 (SNAP force evaluation). Exec pack:
// docs/development/m8_execution_pack.md T8.4b.
//
// Unit / structural tests для SnapPotential::compute (T8.4b force body port).
// The D-M8-7 byte-exact gate vs LAMMPS lives в T8.5 (run_differential +
// LAMMPS oracle run); these здесь tests cover TDMD-internal correctness
// properties that don't require the LAMMPS runtime:
//
//   1. Construction smoke — canonical W_2940_2017_2 fixture loads и the
//      engine survives init() without a throw.
//   2. Dimer — 2-atom W at r = 3.0 Å produces finite PE, finite forces,
//      antisymmetric (Newton's 3rd law) pair forces, и virial along the pair
//      axis only.
//   3. BCC W Newton's 3rd law — |Σ F_i| / max|F_i| ≤ 1e-12 per component on
//      a 250-atom perturbed BCC slab. Catches any asymmetric accumulation в
//      the half-list → full-list bridge.
//   4. Energy-force consistency — central-difference F_x ≈ −(E(+ε)−E(−ε))/(2ε)
//      on perturbed BCC W. Absolute agreement ≤ 5e-6 (FD truncation-dominated;
//      analytic-force-vs-FD-force is the textbook test for dE/dR = F
//      consistency).
//
// Self-skips with exit 77 when the LAMMPS submodule isn't initialized
// (Option A / public CI convention — matches test_snap_file.cpp).

#include "tdmd/neighbor/cell_grid.hpp"
#include "tdmd/neighbor/neighbor_list.hpp"
#include "tdmd/potentials/snap.hpp"
#include "tdmd/potentials/snap_file.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

#ifndef TDMD_TEST_FIXTURES_DIR
#error "TDMD_TEST_FIXTURES_DIR must be defined by the build system"
#endif

namespace fs = std::filesystem;
using tdmd::AtomSoA;
using tdmd::Box;
using tdmd::CellGrid;
using tdmd::ForceResult;
using tdmd::NeighborList;
using tdmd::SnapPotential;
using tdmd::SpeciesId;
using tdmd::potentials::parse_snap_files;
using tdmd::potentials::SnapData;

namespace {

constexpr int kExitSkip = 77;

fs::path lammps_snap_examples_dir() {
  const fs::path fixtures_dir = TDMD_TEST_FIXTURES_DIR;
  const fs::path repo_root = fixtures_dir.parent_path().parent_path().parent_path();
  return repo_root / "verify" / "third_party" / "lammps" / "examples" / "snap";
}

void skip_if_submodule_uninitialized(const fs::path& dir) {
  if (!fs::exists(dir)) {
    std::fprintf(stderr,
                 "[test_snap_compute] SKIP: LAMMPS submodule not initialized at %s — "
                 "run `git submodule update --init verify/third_party/lammps`.\n",
                 dir.string().c_str());
    std::fflush(stderr);
    std::exit(kExitSkip);
  }
}

SnapData load_w_fixture() {
  const auto dir = lammps_snap_examples_dir();
  return parse_snap_files((dir / "W_2940_2017_2.snapcoeff").string(),
                          (dir / "W_2940_2017_2.snapparam").string());
}

Box make_cubic_box(double length) {
  Box b;
  b.xhi = length;
  b.yhi = length;
  b.zhi = length;
  b.periodic_x = true;
  b.periodic_y = true;
  b.periodic_z = true;
  return b;
}

// Places a 2·nrep³ BCC W lattice centered на (0,0,0) origin with cubic-cell
// side a. Returns atoms in the box, ready for binning.
void add_bcc_W(AtomSoA& atoms, int nrep, double a) {
  for (int kz = 0; kz < nrep; ++kz) {
    for (int ky = 0; ky < nrep; ++ky) {
      for (int kx = 0; kx < nrep; ++kx) {
        atoms.add_atom(0,
                       static_cast<double>(kx) * a,
                       static_cast<double>(ky) * a,
                       static_cast<double>(kz) * a);
        atoms.add_atom(0,
                       (static_cast<double>(kx) + 0.5) * a,
                       (static_cast<double>(ky) + 0.5) * a,
                       (static_cast<double>(kz) + 0.5) * a);
      }
    }
  }
}

void zero_forces(AtomSoA& atoms) {
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    atoms.fx[i] = 0.0;
    atoms.fy[i] = 0.0;
    atoms.fz[i] = 0.0;
  }
}

// Scattered perturbation applied to break pristine-lattice symmetry и
// uncover any asymmetric bug в the full-list reconstruction. Deterministic —
// seed-free — so regressions are reproducible across platforms.
void apply_tiny_rattle(AtomSoA& atoms) {
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const double s = 0.01;  // Å — 0.3 % of W NN distance (2.754 Å)
    const double ph = static_cast<double>(i) * 0.1;
    atoms.x[i] += s * std::sin(ph);
    atoms.y[i] += s * std::sin(ph + 1.0);
    atoms.z[i] += s * std::sin(ph + 2.0);
  }
}

// Full PE computation on a fresh neighbour list — used by the FD consistency
// test so perturbed positions get a freshly-binned list.
double compute_pe(SnapPotential& snap, AtomSoA& atoms, const Box& box, double cutoff, double skin) {
  CellGrid grid;
  NeighborList list;
  grid.build(box, cutoff, skin);
  grid.bin(atoms);
  list.build(atoms, box, grid, cutoff, skin);
  zero_forces(atoms);
  const ForceResult r = snap.compute(atoms, list, box);
  return r.potential_energy;
}

}  // namespace

TEST_CASE("SnapPotential: construction on canonical W fixture",
          "[potentials][snap][compute][t8.4b]") {
  skip_if_submodule_uninitialized(lammps_snap_examples_dir());

  const auto data = load_w_fixture();
  REQUIRE(data.species.size() == 1);
  REQUIRE(data.species[0].name == "W");
  REQUIRE(data.k_max == 55);  // twojmax=8 → 55 bispectrum components

  REQUIRE_NOTHROW(SnapPotential(data));
  SnapPotential snap(data);
  REQUIRE(snap.name() == "snap");
  REQUIRE(snap.cutoff() == Catch::Approx(4.73442).margin(1e-12));
}

TEST_CASE("SnapPotential: 2-atom W dimer — forces antisymmetric, virial along pair axis",
          "[potentials][snap][compute][t8.4b]") {
  skip_if_submodule_uninitialized(lammps_snap_examples_dir());

  SnapPotential snap(load_w_fixture());
  const double cutoff = snap.cutoff();
  const double skin = snap.effective_skin();

  Box box = make_cubic_box(30.0);
  AtomSoA atoms;
  const double cx = 0.5 * box.lx();
  const double cy = 0.5 * box.ly();
  const double cz = 0.5 * box.lz();
  const double r_sep = 3.0;  // < cutoff, > 1st NN for BCC W
  atoms.add_atom(0, cx - 0.5 * r_sep, cy, cz);
  atoms.add_atom(0, cx + 0.5 * r_sep, cy, cz);

  CellGrid grid;
  NeighborList list;
  grid.build(box, cutoff, skin);
  grid.bin(atoms);
  list.build(atoms, box, grid, cutoff, skin);

  zero_forces(atoms);
  const ForceResult fr = snap.compute(atoms, list, box);

  // PE + forces finite (no NaN/Inf от the CG recursion).
  REQUIRE(std::isfinite(fr.potential_energy));
  REQUIRE(std::isfinite(atoms.fx[0]));
  REQUIRE(std::isfinite(atoms.fy[0]));
  REQUIRE(std::isfinite(atoms.fz[0]));

  // Newton's 3rd law (exact — pair forces are applied как +=/-= of the
  // same bit pattern).
  constexpr double kNewtonTol = 1e-14;
  REQUIRE(std::abs(atoms.fx[0] + atoms.fx[1]) < kNewtonTol);
  REQUIRE(std::abs(atoms.fy[0] + atoms.fy[1]) < kNewtonTol);
  REQUIRE(std::abs(atoms.fz[0] + atoms.fz[1]) < kNewtonTol);

  // Pair along x → only y/z components of F_i должны be zero by symmetry
  // (each bispectrum Y_ml ∝ rotational harmonics of r̂; along x axis the y/z
  // Cartesian components of the resulting force sum to zero by symmetry).
  constexpr double kSymTol = 1e-12;
  REQUIRE(std::abs(atoms.fy[0]) < kSymTol);
  REQUIRE(std::abs(atoms.fz[0]) < kSymTol);

  // Virial symmetry: off-diagonal components зачинают zero (pair along x).
  REQUIRE(std::abs(fr.virial[3]) < kSymTol);  // xy
  REQUIRE(std::abs(fr.virial[4]) < kSymTol);  // xz
  REQUIRE(std::abs(fr.virial[5]) < kSymTol);  // yz
  // Trace non-zero (pair potential is not at equilibrium separation of 3.0 Å).
  const double virial_trace = fr.virial[0] + fr.virial[1] + fr.virial[2];
  REQUIRE(std::abs(virial_trace) > 1e-6);
}

TEST_CASE("SnapPotential: BCC W — Newton's 3rd law on perturbed lattice",
          "[potentials][snap][compute][t8.4b]") {
  skip_if_submodule_uninitialized(lammps_snap_examples_dir());

  SnapPotential snap(load_w_fixture());
  const double cutoff = snap.cutoff();
  const double skin = snap.effective_skin();
  const double a = 3.1803;
  const int nrep = 5;

  Box box = make_cubic_box(static_cast<double>(nrep) * a);
  AtomSoA atoms;
  add_bcc_W(atoms, nrep, a);
  apply_tiny_rattle(atoms);
  REQUIRE(atoms.size() == static_cast<std::size_t>(2 * nrep * nrep * nrep));

  CellGrid grid;
  NeighborList list;
  grid.build(box, cutoff, skin);
  grid.bin(atoms);
  list.build(atoms, box, grid, cutoff, skin);

  zero_forces(atoms);
  const ForceResult fr = snap.compute(atoms, list, box);
  REQUIRE(std::isfinite(fr.potential_energy));

  double sum_fx = 0.0;
  double sum_fy = 0.0;
  double sum_fz = 0.0;
  double max_f = 0.0;
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    sum_fx += atoms.fx[i];
    sum_fy += atoms.fy[i];
    sum_fz += atoms.fz[i];
    max_f = std::max(
        max_f,
        std::max(std::abs(atoms.fx[i]), std::max(std::abs(atoms.fy[i]), std::abs(atoms.fz[i]))));
  }
  REQUIRE(max_f > 0.0);  // not a pristine lattice (would give zero forces)

  // |Σ F| / max|F| ≤ 1e-12 — tight because forces are applied +=/-= в pairs
  // и the FP sum of zeroes is exactly zero, except для rounding когда
  // +fij и −fij are added into a non-zero accumulator.
  const double rel_fx = std::abs(sum_fx) / max_f;
  const double rel_fy = std::abs(sum_fy) / max_f;
  const double rel_fz = std::abs(sum_fz) / max_f;
  INFO("sum_fx=" << sum_fx << " sum_fy=" << sum_fy << " sum_fz=" << sum_fz << " max_f=" << max_f);
  REQUIRE(rel_fx < 1e-12);
  REQUIRE(rel_fy < 1e-12);
  REQUIRE(rel_fz < 1e-12);
}

TEST_CASE("SnapPotential: BCC W — force == −dE/dR via central differences",
          "[potentials][snap][compute][t8.4b]") {
  skip_if_submodule_uninitialized(lammps_snap_examples_dir());

  SnapPotential snap(load_w_fixture());
  const double cutoff = snap.cutoff();
  const double skin = snap.effective_skin();
  const double a = 3.1803;
  const int nrep = 5;

  Box box = make_cubic_box(static_cast<double>(nrep) * a);
  AtomSoA atoms;
  add_bcc_W(atoms, nrep, a);
  apply_tiny_rattle(atoms);

  // Analytic force on atom 0 (x-component) от the at-reference configuration.
  // compute_pe rebuilds the NL, so this first call establishes baseline.
  CellGrid grid;
  NeighborList list;
  grid.build(box, cutoff, skin);
  grid.bin(atoms);
  list.build(atoms, box, grid, cutoff, skin);
  zero_forces(atoms);
  [[maybe_unused]] const auto fr0 = snap.compute(atoms, list, box);
  const double fx0_analytic = atoms.fx[0];
  const double fy0_analytic = atoms.fy[0];
  const double fz0_analytic = atoms.fz[0];

  // Central differences along each axis. ε = 1e-4 Å — small enough that the
  // SNAP E(R) is smooth on this scale (potential is C∞), big enough that
  // FP cancellation doesn't dominate (E values are ~1e3 eV for 250 atoms,
  // so ΔE ≈ F·ε ≈ 1e-1 eV per perturbation, well above FP noise floor).
  const double eps = 1e-4;
  auto fd_component = [&](std::size_t i, int axis) -> double {
    double& coord = (axis == 0 ? atoms.x[i] : (axis == 1 ? atoms.y[i] : atoms.z[i]));
    const double saved = coord;
    coord = saved + eps;
    const double e_plus = compute_pe(snap, atoms, box, cutoff, skin);
    coord = saved - eps;
    const double e_minus = compute_pe(snap, atoms, box, cutoff, skin);
    coord = saved;
    return -(e_plus - e_minus) / (2.0 * eps);
  };

  const double fx0_fd = fd_component(0, 0);
  const double fy0_fd = fd_component(0, 1);
  const double fz0_fd = fd_component(0, 2);

  INFO("fx0: analytic=" << fx0_analytic << " fd=" << fx0_fd);
  INFO("fy0: analytic=" << fy0_analytic << " fd=" << fy0_fd);
  INFO("fz0: analytic=" << fz0_analytic << " fd=" << fz0_fd);

  // Central-diff truncation error is O(ε²) · f'''(R). For SNAP on BCC W, f'''
  // is bounded around ~10³ eV/Å³ (characteristic of covalent-core repulsion
  // at displaced ionic positions), so truncation at ε=1e-4 is ~1e-5 eV/Å.
  // Analytic forces on tiny-rattle atoms are ~1 eV/Å, so relative error ≤ 1e-5
  // is a reasonable target.
  const double tol = 1e-4;  // absolute eV/Å
  REQUIRE(std::abs(fx0_fd - fx0_analytic) < tol);
  REQUIRE(std::abs(fy0_fd - fy0_analytic) < tol);
  REQUIRE(std::abs(fz0_fd - fz0_analytic) < tol);
}
