#pragma once

// SPEC: docs/specs/potentials/SPEC.md §6 (SNAP — Spectral Neighbor Analysis
// Potential). Exec pack: docs/development/m8_execution_pack.md T8.4a/T8.4b.
//
// Linear SNAP (Thompson et al., JCP 2015; Wood & Thompson 2017). Force body
// is a verbatim port of LAMMPS USER-SNAP `pair_snap.cpp::compute()` plus
// `sna.cpp` (see snap/sna_engine.hpp). TDMD Fp64Reference ≡ LAMMPS FP64
// ≤ 1e-12 rel is the M8 acceptance gate (D-M8-7).
//
// Algorithmic outline (§6.5 of the potentials SPEC):
//   Pass 1: per-atom bispectrum B_{k,i} accumulated over full-list neighbours.
//   Pass 2: contract B_{k,i} с per-species β_k to get per-atom energy E_i и
//           its derivative dE_i/dB_{k,i}.
//   Pass 3: revisit pairs to convert dE/dB into per-pair forces через the
//           Clebsch-Gordan chain rule; accumulate into atoms.fx/fy/fz и the
//           virial tensor.
//
// Half-list → full-list bridge. LAMMPS SNAP uses a full neighbor list
// (`REQ_FULL` — each pair (i,j) appears twice: once в firstneigh[i], once
// в firstneigh[j]); TDMD's NeighborList is half-list (newton on, j > i).
// `compute()` materialises a symmetric full-list scratch from the half list
// so the SnaEngine outer loop matches upstream line-for-line. Scratch is
// held as member state (grow-once, reused-across-steps) per the potentials
// SPEC "no hidden allocations in compute()" invariant.

#include "tdmd/potentials/potential.hpp"
#include "tdmd/potentials/snap_file.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tdmd {

// Forward-declare the SnaEngine pimpl so snap.hpp consumers don't pull in
// the full LAMMPS-port surface (snap_detail::SnaEngine has ~30 member fields
// и mirrors upstream private layout).
namespace snap_detail {
class SnaEngine;
}

class SnapPotential final : public Potential {
public:
  // Takes ownership of the parsed SNAP parameter + coefficient set. Validates
  // structural consistency (species count > 0, twojmax even и ≥ 0, β
  // coefficient count matches `k_max + 1` для linear SNAP); throws
  // `std::invalid_argument` on any mismatch. Constructs the internal SnaEngine
  // и calls its `init()` (Clebsch-Gordan и root-pq tables) eagerly so that
  // compute() has no first-call latency spike.
  explicit SnapPotential(potentials::SnapData data);

  // Out-of-line destructor — required because `engine_` is a
  // `unique_ptr<SnaEngine>` где `SnaEngine` is only forward-declared здесь.
  ~SnapPotential() override;

  // Non-copyable (Potential deletes copy); move out-of-line для pimpl.
  SnapPotential(SnapPotential&&) noexcept;
  SnapPotential& operator=(SnapPotential&&) noexcept;

  [[nodiscard]] ForceResult compute(AtomSoA& atoms,
                                    const NeighborList& neighbors,
                                    const Box& box) override;

  // Maximum pairwise cutoff across species pairs (rcutfac·(R_α + R_β)).
  [[nodiscard]] double cutoff() const noexcept override { return data_.max_pairwise_cutoff(); }

  [[nodiscard]] std::string name() const override { return "snap"; }

  // Parameter accessor — used by telemetry/explain и the T8.5 differential
  // harness to cross-check table contents против LAMMPS.
  [[nodiscard]] const potentials::SnapData& data() const noexcept { return data_; }

  // Recommended neighbor-list skin. Matches the EAM / Morse heuristic (5 %
  // of cutoff) until the T8.10 benchmark measurements suggest a SNAP-specific
  // value.
  [[nodiscard]] double effective_skin() const noexcept {
    return 0.05 * data_.max_pairwise_cutoff();
  }

private:
  potentials::SnapData data_;
  std::unique_ptr<snap_detail::SnaEngine> engine_;

  // Full-list scratch derived из half-list each compute(). CSR-style.
  // Grown-once / reused-across-steps — match the potentials SPEC invariant.
  std::vector<std::uint64_t> full_offsets_;
  std::vector<std::uint32_t> full_ids_;
  std::vector<std::uint32_t> full_cursor_;
};

}  // namespace tdmd
