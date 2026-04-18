#pragma once

// SPEC: docs/specs/potentials/SPEC.md §4.1–§4.3 (EAM/alloy form and two-pass
// force algorithm), §2.4.2 (cutoff policy — hard + natural tail in M2).
// Exec pack: docs/development/m2_execution_pack.md T2.7.
//
// EAM/alloy potential — single per-species electron-density function `ρ_α(r)`,
// per-species embedding `F_α(ρ)`, and lower-triangular pair part `φ_{αβ}(r)`
// stored as `z2r(r) = r · φ(r)`. Two-pass force evaluation matches LAMMPS
// `pair_eam_alloy.cpp::compute` — see eam_alloy.cpp for the FP operation
// sequence that makes the bit-match achievable (T2.9 differential harness).
//
// Species mapping. Atom's `AtomSoA::type[i]` is used directly as the index
// into `EamAlloyData::species_names` / `F_rho` / `rho_r`. The caller must
// ensure the data file's species ordering matches the per-atom `SpeciesId`s
// produced by the data importer (T2.2 maps LAMMPS `type` 1..N to
// SpeciesId 0..N-1 via `SpeciesRegistry`). A debug-only range check catches
// index-out-of-table silently during tests.
//
// Cutoff. M2 uses hard cutoff only (Strategy A per SPEC §2.4); `ρ(r_c)` and
// `φ(r_c)` are zero by LAMMPS file convention so the discontinuity at `r_c`
// is numerically negligible for a well-formed parameter file. Shifted-force
// smoothing arrives with T3.*.
//
// Newton's 3rd law. The half-list traversal visits each pair once. Density
// contributions are symmetric: `ρ[i] += ρ_β(r)` and `ρ[j] += ρ_α(r)` from the
// same pair. Forces are applied `F_i += Δ`, `F_j -= Δ` per pair.
//
// Hot-path allocation. Per-atom `ρ` and `F'(ρ)` buffers are held as member
// state and `resize`d at each `compute()` entry, not allocated. Grown-once,
// reused-across-steps — matches the exec-pack invariant "No hidden
// allocations in compute()".

#include "tdmd/potentials/eam_file.hpp"
#include "tdmd/potentials/potential.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace tdmd {

class EamAlloyPotential final : public Potential {
public:
  // Takes ownership of the parsed EAM parameters. Validates that the data
  // struct is internally consistent (table counts match N_species,
  // cutoff > 0); throws `std::invalid_argument` on any structural mismatch.
  explicit EamAlloyPotential(potentials::EamAlloyData data);

  [[nodiscard]] ForceResult compute(AtomSoA& atoms,
                                    const NeighborList& neighbors,
                                    const Box& box) override;

  [[nodiscard]] double cutoff() const noexcept override { return data_.cutoff; }
  [[nodiscard]] std::string name() const override { return "eam/alloy"; }

  // Parameter-struct accessor — used by telemetry/explain and by the
  // differential harness to cross-check table contents against LAMMPS.
  [[nodiscard]] const potentials::EamAlloyData& data() const noexcept { return data_; }

  // Post-compute accessors (tests only). `density()[i]` holds ρ_i from the
  // most recent `compute()` call; `dF_drho()[i]` holds F'(ρ_i). Both are
  // empty before the first `compute()` invocation.
  [[nodiscard]] const std::vector<double>& density() const noexcept { return density_; }
  [[nodiscard]] const std::vector<double>& dF_drho() const noexcept { return dF_drho_; }

  // Recommended skin (conservative 5 % of cutoff, same heuristic as Morse).
  [[nodiscard]] double effective_skin() const noexcept { return 0.05 * data_.cutoff; }

private:
  potentials::EamAlloyData data_;
  std::vector<double> density_;
  std::vector<double> dF_drho_;
};

}  // namespace tdmd
