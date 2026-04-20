#pragma once

// SPEC: docs/specs/potentials/SPEC.md §6 (SNAP — Spectral Neighbor Analysis
// Potential). Exec pack: docs/development/m8_execution_pack.md T8.4a/T8.4b.
//
// Linear SNAP (Thompson et al., JCP 2015; Wood & Thompson 2017). In M8 T8.4a
// only the types + parser + skeleton land; the force-evaluation body (three-
// pass kernel port от LAMMPS USER-SNAP `pair_snap.cpp` + `sna.cpp`) is
// scheduled for T8.4b, и `SnapPotential::compute` throws `std::logic_error`
// until then.
//
// Algorithmic outline (§6.5 of the potentials SPEC):
//   Pass 1: accumulate per-atom bispectrum components B_{k,i} over half-list
//           neighbours.
//   Pass 2: contract B_{k,i} с per-species β_k to get per-atom energy E_i и
//           its derivative dE_i / dB_{k,i}.
//   Pass 3: revisit pairs to convert dE/dB into per-pair forces through the
//           Clebsch-Gordan chain rule; accumulate into atoms.fx/fy/fz и the
//           virial tensor.
//
// Reference implementation. LAMMPS USER-SNAP (GPLv2) is the byte-for-byte
// oracle for the CPU FP64 path (D-M8-7 ≤1e-12 rel). The port preserves the
// FP summation ordering from `SNA::compute_bi` / `SNA::compute_dbidrj`
// verbatim so that MixedFast swap-in later on behaves as a precision policy
// over a shared core, not a parallel implementation.

#include "tdmd/potentials/potential.hpp"
#include "tdmd/potentials/snap_file.hpp"

#include <string>

namespace tdmd {

class SnapPotential final : public Potential {
public:
  // Takes ownership of the parsed SNAP parameter + coefficient set. Validates
  // basic structural consistency (species count > 0, twojmax even and ≥ 0,
  // β coefficient count matches `k_max + 1` для linear SNAP); throws
  // `std::invalid_argument` on any mismatch.
  explicit SnapPotential(potentials::SnapData data);

  // T8.4a skeleton: throws `std::logic_error` — full force body lands in T8.4b.
  [[nodiscard]] ForceResult compute(AtomSoA& atoms,
                                    const NeighborList& neighbors,
                                    const Box& box) override;

  // Maximum pairwise cutoff across species pairs (rcutfac·(R_α + R_β)).
  [[nodiscard]] double cutoff() const noexcept override { return data_.max_pairwise_cutoff(); }

  [[nodiscard]] std::string name() const override { return "snap"; }

  // Parameter accessor — used by telemetry/explain и the T8.5 differential
  // harness to cross-check table contents against LAMMPS.
  [[nodiscard]] const potentials::SnapData& data() const noexcept { return data_; }

  // Recommended neighbor-list skin. Matches the EAM / Morse heuristic (5 %
  // of cutoff) until the T8.10 benchmark measurements suggest a SNAP-specific
  // value.
  [[nodiscard]] double effective_skin() const noexcept {
    return 0.05 * data_.max_pairwise_cutoff();
  }

private:
  potentials::SnapData data_;
};

}  // namespace tdmd
