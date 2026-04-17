#include "tdmd/neighbor/neighbor_list.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace tdmd {

void NeighborList::clear() noexcept {
  page_offsets_.clear();
  neigh_ids_.clear();
  neigh_r2_.clear();
}

void NeighborList::build(const AtomSoA& atoms,
                         const Box& box,
                         const CellGrid& grid,
                         double cutoff,
                         double skin) {
  const double reach = cutoff + skin;
  const double reach_sq = reach * reach;
  const std::size_t n = atoms.size();

  page_offsets_.assign(n + 1, 0);
  neigh_ids_.clear();
  neigh_r2_.clear();

  const auto& cell_offsets = grid.cell_offsets();
  const auto& cell_atoms = grid.cell_atoms();

  for (std::size_t i = 0; i < n; ++i) {
    page_offsets_[i] = neigh_ids_.size();
    const std::size_t ci = grid.cell_of(atoms.x[i], atoms.y[i], atoms.z[i]);
    const std::array<std::size_t, 27> stencil = grid.neighbor_cells(ci);

    for (std::size_t cj : stencil) {
      const std::uint64_t begin = cell_offsets[cj];
      const std::uint64_t end = cell_offsets[cj + 1];
      for (std::uint64_t k = begin; k < end; ++k) {
        const std::uint32_t j = cell_atoms[k];
        if (static_cast<std::size_t>(j) <= i) {
          continue;
        }
        const auto delta = box.unwrap_minimum_image(atoms.x[j] - atoms.x[i],
                                                    atoms.y[j] - atoms.y[i],
                                                    atoms.z[j] - atoms.z[i]);
        const double r2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
        if (r2 <= reach_sq) {
          neigh_ids_.push_back(j);
          neigh_r2_.push_back(r2);
        }
      }
    }
  }
  page_offsets_[n] = neigh_ids_.size();

  cutoff_ = cutoff;
  skin_ = skin;
  ++build_version_;
}

}  // namespace tdmd
