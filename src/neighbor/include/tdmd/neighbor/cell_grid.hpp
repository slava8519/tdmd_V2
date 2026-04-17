#pragma once

// SPEC: docs/specs/neighbor/SPEC.md
//
// CellGrid partitions the simulation box into cells for O(N) neighbor search.
// Full definition (cell sizing, ghost atom handling, rebuild policy) lives in
// neighbor/SPEC.md §2.1 and §3; this file is a skeleton for M0.

#include <cstddef>

namespace tdmd {

struct CellGrid {
  // TODO(M1): fields per neighbor/SPEC.md §2.1 — cell extents, strides,
  // ghost cell padding, atom-to-cell lookup table.

  [[nodiscard]] std::size_t cell_count() const noexcept { return cell_count_; }
  [[nodiscard]] bool empty() const noexcept { return cell_count_ == 0; }

 private:
  std::size_t cell_count_ = 0;
};

}  // namespace tdmd
