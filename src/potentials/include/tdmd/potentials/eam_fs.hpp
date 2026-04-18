#pragma once

// SPEC: docs/specs/potentials/SPEC.md §4.2 (Finnis-Sinclair variant — per-pair
// ρ_{αβ}(r) tables), §4.3 (shared two-pass algorithm). Exec pack: T2.7.
//
// EAM/FS potential — same two-pass force math as EAM/alloy, but the density
// contribution from a single neighbour depends on the ordered (α, β) species
// pair. `EamFsData::rho_ij[α·N + β]` is the density seen at an atom of
// species α from a single neighbour of species β. In the general FS case
// `ρ_{αβ} ≠ ρ_{βα}`, so the parser stores N × N tables (row-major) rather
// than the N tables an alloy file carries.
//
// Everything else (F(ρ), z2r pair part, cutoff policy, Newton 3rd law,
// scratch-buffer ownership) matches `EamAlloyPotential` — see that header
// for the shared rationale. The duplication of the compute() body between
// the two classes is intentional: the inner loops touch different index
// patterns (`rho_r[β]` vs `rho_ij[α·N + β]`) and a templated dispatch would
// add indirection for no algorithmic gain in M2.

#include "tdmd/potentials/eam_file.hpp"
#include "tdmd/potentials/potential.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace tdmd {

class EamFsPotential final : public Potential {
public:
  explicit EamFsPotential(potentials::EamFsData data);

  [[nodiscard]] ForceResult compute(AtomSoA& atoms,
                                    const NeighborList& neighbors,
                                    const Box& box) override;

  [[nodiscard]] double cutoff() const noexcept override { return data_.cutoff; }
  [[nodiscard]] std::string name() const override { return "eam/fs"; }

  [[nodiscard]] const potentials::EamFsData& data() const noexcept { return data_; }
  [[nodiscard]] const std::vector<double>& density() const noexcept { return density_; }
  [[nodiscard]] const std::vector<double>& dF_drho() const noexcept { return dF_drho_; }

  [[nodiscard]] double effective_skin() const noexcept { return 0.05 * data_.cutoff; }

private:
  potentials::EamFsData data_;
  std::vector<double> density_;
  std::vector<double> dF_drho_;
};

}  // namespace tdmd
