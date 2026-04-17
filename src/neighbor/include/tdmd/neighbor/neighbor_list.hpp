#pragma once

// SPEC: docs/specs/neighbor/SPEC.md §2.1, §4 (neighbor list)
// Exec pack: docs/development/m1_execution_pack.md T1.6
//
// Half-list (`newton on`) pair list built over a `CellGrid`. For every pair
// (i, j) with |r_ij| ≤ cutoff + skin and j > i we store one entry: the local
// index `j` and the squared separation r_ij² (reused by pair-force kernels to
// avoid recomputation).
//
// Storage is CSR: `page_offsets[i]..page_offsets[i+1]` slices `neigh_ids`
// and `neigh_r2`. This matches neighbor/SPEC §2.1 and keeps cache behaviour
// predictable. M1 uses local atom indices (uint32_t), not AtomIds — the list
// is valid only within a single `build_version`; any reorder/migration
// invalidates indices by construction.
//
// Determinism: inputs (AtomSoA, Box, CellGrid) → `neigh_ids` sequence is
// bit-for-bit reproducible (ordered iteration over i, then over stencil
// cells, then over the stable cell_atoms CSR).

#include "tdmd/neighbor/cell_grid.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <cstdint>
#include <vector>

namespace tdmd {

class NeighborList {
public:
  // Builds the list. Requires `grid.bin(atoms)` already executed and `box`
  // to match the one passed to `grid.build`. `cutoff` must match the one
  // used for `grid.build` (fatal if mismatched — preflight in caller).
  void build(const AtomSoA& atoms,
             const Box& box,
             const CellGrid& grid,
             double cutoff,
             double skin);

  // Clears internal buffers. Keeps cutoff/skin/build_version unchanged.
  void clear() noexcept;

  [[nodiscard]] std::size_t atom_count() const noexcept {
    return page_offsets_.empty() ? 0 : page_offsets_.size() - 1;
  }

  // Number of (j > i) pair entries across the whole list.
  [[nodiscard]] std::size_t pair_count() const noexcept { return neigh_ids_.size(); }

  // CSR slice for atom `i`: half-list neighbors (only j > i).
  [[nodiscard]] std::uint64_t page_begin(std::size_t i) const noexcept { return page_offsets_[i]; }
  [[nodiscard]] std::uint64_t page_end(std::size_t i) const noexcept {
    return page_offsets_[i + 1];
  }

  [[nodiscard]] const std::vector<std::uint64_t>& page_offsets() const noexcept {
    return page_offsets_;
  }
  [[nodiscard]] const std::vector<std::uint32_t>& neigh_ids() const noexcept { return neigh_ids_; }
  [[nodiscard]] const std::vector<double>& neigh_r2() const noexcept { return neigh_r2_; }

  [[nodiscard]] double cutoff() const noexcept { return cutoff_; }
  [[nodiscard]] double skin() const noexcept { return skin_; }
  [[nodiscard]] std::uint64_t build_version() const noexcept { return build_version_; }

private:
  std::vector<std::uint64_t> page_offsets_;  // size atom_count + 1
  std::vector<std::uint32_t> neigh_ids_;
  std::vector<double> neigh_r2_;

  double cutoff_ = 0.0;
  double skin_ = 0.0;
  std::uint64_t build_version_ = 0;
};

}  // namespace tdmd
