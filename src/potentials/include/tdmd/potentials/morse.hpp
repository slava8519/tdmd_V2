#pragma once

// SPEC: docs/specs/potentials/SPEC.md §3 (Morse), §2.4 (cutoff treatment)
// Exec pack: docs/development/m1_execution_pack.md T1.8
//
// Morse pair potential — first TDMD potential, reference for the M1 differential
// test vs LAMMPS `pair_style morse`. Single-species in M1 (per exec pack scope);
// multi-species dispatch arrives with T1.3 config parsing and T2.* EAM work.
//
// Potential form (SPEC §3.1):
//   E_pair(r) = D · [1 - exp(-α·(r - r₀))]² - D
// so the well bottom is at r = r₀ with E_pair(r₀) = -D. The "-D" shift is a
// conventional offset that leaves the force unchanged; it makes E(∞) = 0.
//
// Force: the analytic radial scalar dE/dr is
//   G(r) = dE/dr = 2·D·α·[1 - exp(-α·(r - r₀))]·exp(-α·(r - r₀))
// and the force on atom i from pair (i, j) is
//   F_i = (G(r) / r) · (r_j - r_i)     (attractive for r > r₀)
// with F_j = -F_i (Newton's 3rd law).
//
// Cutoff treatment: Strategy C ("shifted-force") per SPEC §2.4.2 — both E and F
// are continuous at r = r_c:
//   G_shifted(r)  = G(r) - G(r_c)
//   E_shifted(r)  = E_pair(r) - E_pair(r_c) - (r - r_c) · G(r_c)
// for r < r_c, else 0. G(r_c) is precomputed at construction.
//
// The SPEC §2.4.2 matrix lists Strategy A (hard cutoff) for the 2-atom analytic
// unit test and Strategy C for production canonical benchmarks; this
// implementation is Strategy C. For an apples-to-apples LAMMPS diff at T1.11
// configure LAMMPS with an equivalent shifted-force convention.

#include "tdmd/neighbor/neighbor_list.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace tdmd {

class MorsePotential {
public:
  // Per-pair parameters. Single-species in M1 means one struct per MorsePotential
  // instance. Multi-species M2 will switch to `std::vector<PairParams>` indexed
  // by (type_i, type_j).
  struct PairParams {
    double D = 0.0;       // Well depth, eV.
    double alpha = 0.0;   // Width parameter, 1/Å.
    double r0 = 0.0;      // Equilibrium distance, Å.
    double cutoff = 0.0;  // Interaction cutoff, Å. Must be > r0; caller enforces.
  };

  // Aggregate returned from `compute`. Per-atom forces are written in-place into
  // `atoms.fx / fy / fz`; they are NOT in this struct.
  struct Result {
    double potential_energy = 0.0;      // eV, summed over pairs
    std::array<double, 6> virial = {};  // (xx, yy, zz, xy, xz, yz), eV·Å (Voigt)
  };

  // Cutoff treatment per SPEC §2.4.1/§2.4.2:
  //   HardCutoff    (Strategy A) — E(r) = E_pair(r), F(r) = F_pair(r); zero past r_c.
  //                                Discontinuous at r_c; NOT for production.
  //                                Used by the analytic-reference / unit-test path
  //                                where "F(r₀) = 0 exactly" must hold.
  //   ShiftedForce  (Strategy C) — production default. Continuous E and F at r_c.
  //                                F(r₀) = -G(r_c) (small — ~G(r_c) ≈ 2·D·α·e_c).
  enum class CutoffStrategy : std::uint8_t {
    HardCutoff,
    ShiftedForce,
  };

  // Constructs with the given pair parameters. Throws std::invalid_argument if
  // `params.D ≤ 0`, `params.alpha ≤ 0`, `params.r0 ≤ 0`, `params.cutoff ≤ params.r0`,
  // or any value is not finite. Default strategy matches SPEC §2.4.2 Morse row
  // (production: shifted-force).
  explicit MorsePotential(const PairParams& params,
                          CutoffStrategy strategy = CutoffStrategy::ShiftedForce);

  // Accumulates pair forces into `atoms.fx / fy / fz` and returns total PE and
  // virial (both in eV units consistent with forces in eV/Å). The caller MUST
  // zero `atoms.fx / fy / fz` before invocation — this is additive.
  //
  // Uses the CSR half-list `(j > i, newton on)`: each pair entry contributes
  // +F to atom i and -F to atom j. The `neigh_r2` cache is used to skip the
  // sqrt when r² > cutoff²; this mirrors the cutoff treatment in §3.4 of the
  // SPEC and avoids work for pairs that fell inside the skin but outside the
  // interaction cutoff.
  //
  // Does not allocate on the hot path — only reads / writes into the caller's
  // SoA.
  [[nodiscard]] Result compute(AtomSoA& atoms, const NeighborList& neighbors, const Box& box) const;

  [[nodiscard]] double cutoff() const noexcept { return params_.cutoff; }
  [[nodiscard]] const PairParams& params() const noexcept { return params_; }
  [[nodiscard]] CutoffStrategy strategy() const noexcept { return strategy_; }

  // Recommended skin: a conservative 5% of the cutoff. Neighbor module may
  // override this from user config; this value is a safe default for M1.
  [[nodiscard]] double effective_skin() const noexcept { return 0.05 * params_.cutoff; }

  [[nodiscard]] std::string name() const { return "morse"; }

private:
  PairParams params_{};
  CutoffStrategy strategy_ = CutoffStrategy::ShiftedForce;
  // Precomputed G(r_c) — the force-shift constant. Zero for Strategy A.
  double g_at_rc_ = 0.0;
  // Precomputed E_pair(r_c), used by Strategy C to shift energy so that
  // E_shifted(r_c) = 0. Zero for Strategy A (hard cutoff uses raw E_pair).
  double e_pair_at_rc_ = 0.0;
};

}  // namespace tdmd
