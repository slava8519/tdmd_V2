#pragma once

// SPEC: docs/specs/neighbor/SPEC.md §2.1, §3 (cell grid), §8 (stable reorder)
// Exec pack: docs/development/m1_execution_pack.md T1.5
//
// `CellGrid` partitions the simulation box into a 3D grid of cells of size
// ≥ (cutoff + skin) on each axis, giving O(N) neighbor search. In M1 we also
// own `compute_stable_reorder` — the permutation that groups atoms of the same
// cell contiguously for cache locality. Applying the permutation to `AtomSoA`
// fields is done by a free function (`apply_reorder`) so that `CellGrid`
// itself never mutates state, preserving the ownership invariant of master
// spec §8.2.
//
// M1 constraints:
// - Orthogonal box only (triclinic tilt = 0); enforced via `Box::is_valid_m1`.
// - At least 3 cells per axis (periodic 3×3×3 stencil requires this for
//   unique-cell guarantees). Preflight error otherwise.
// - Stable ordering: within each cell atoms retain their prior `AtomSoA`
//   index order, and the same `(AtomSoA, CellGrid)` input produces bit-exact
//   identical `cell_atoms` / `ReorderMap` outputs across runs.

#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tdmd {

// Thrown by `CellGrid::build` when inputs violate M1 invariants (cutoff/skin
// non-positive, box too small for 3-cells-per-axis, triclinic, etc.).
class InvalidCellGridError : public std::invalid_argument {
public:
  using std::invalid_argument::invalid_argument;
};

// Permutation that reorders `AtomSoA` so atoms of the same cell are
// contiguous. `old_to_new[i] = j` means atom at old index `i` lands at new
// index `j`; `new_to_old` is the inverse. Both arrays have `atoms.size()`
// entries. Construction is deterministic (stable sort by cell index).
struct ReorderMap {
  std::vector<std::uint64_t> old_to_new;
  std::vector<std::uint64_t> new_to_old;

  [[nodiscard]] std::size_t size() const noexcept { return old_to_new.size(); }
  [[nodiscard]] bool empty() const noexcept { return old_to_new.empty(); }
};

class CellGrid {
public:
  // Sizes the grid for `(cutoff + skin)` and remembers the box. No binning
  // happens here — call `bin()` afterwards. Throws `InvalidCellGridError` if
  // the box cannot host ≥3 cells per axis at the requested cutoff/skin, or if
  // cutoff/skin are non-positive, or if the box is not M1-valid.
  void build(const Box& box, double cutoff, double skin);

  // Populates `cell_offsets` (CSR prefix sum, size `nx*ny*nz + 1`) and
  // `cell_atoms` (atom indices, sorted ascending by original atom index
  // within each cell). O(N). Requires `build()` to have been called first and
  // the box used for build to still be the source of truth for `atoms`.
  void bin(const AtomSoA& atoms);

  // Cell index for a point already inside the box (caller must wrap via
  // `Box::wrap` beforehand — on-axis periodic wrap is *not* repeated here).
  // Floor-biases to lower cell at the high boundary via clamp to `n-1`.
  [[nodiscard]] std::size_t cell_of(double x, double y, double z) const noexcept;

  // Twenty-seven neighbor cells (including self) around `cell_idx`, with
  // periodic wrap on each axis. Unique-cell guarantee: for nx, ny, nz >= 3
  // each returned cell index is distinct (see exec pack T1.5).
  [[nodiscard]] std::array<std::size_t, 27> neighbor_cells(std::size_t cell_idx) const noexcept;

  // Stable-sort atom indices by cell index and return the resulting
  // permutation. Pure function of `(atoms, *this)` — no state mutation.
  [[nodiscard]] ReorderMap compute_stable_reorder(const AtomSoA& atoms) const;

  [[nodiscard]] std::size_t cell_count() const noexcept {
    return static_cast<std::size_t>(nx_) * ny_ * nz_;
  }
  [[nodiscard]] bool empty() const noexcept { return cell_count() == 0; }

  [[nodiscard]] std::uint32_t nx() const noexcept { return nx_; }
  [[nodiscard]] std::uint32_t ny() const noexcept { return ny_; }
  [[nodiscard]] std::uint32_t nz() const noexcept { return nz_; }
  [[nodiscard]] double cell_x() const noexcept { return cell_x_; }
  [[nodiscard]] double cell_y() const noexcept { return cell_y_; }
  [[nodiscard]] double cell_z() const noexcept { return cell_z_; }

  // CSR slice for cell `cell_idx`: atoms are at
  // `cell_atoms_[cell_offsets_[cell_idx] .. cell_offsets_[cell_idx + 1])`.
  [[nodiscard]] const std::vector<std::uint32_t>& cell_offsets() const noexcept {
    return cell_offsets_;
  }
  [[nodiscard]] const std::vector<std::uint32_t>& cell_atoms() const noexcept {
    return cell_atoms_;
  }

  [[nodiscard]] std::uint64_t build_version() const noexcept { return build_version_; }

private:
  [[nodiscard]] std::size_t linear_index(int ix, int iy, int iz) const noexcept;

  Box box_{};
  double cutoff_ = 0.0;
  double skin_ = 0.0;

  double cell_x_ = 0.0;
  double cell_y_ = 0.0;
  double cell_z_ = 0.0;
  std::uint32_t nx_ = 0;
  std::uint32_t ny_ = 0;
  std::uint32_t nz_ = 0;

  std::vector<std::uint32_t> cell_offsets_;
  std::vector<std::uint32_t> cell_atoms_;

  std::uint64_t build_version_ = 0;
};

// Permutes every field of `AtomSoA` by `map.new_to_old`. After this call
// `atoms.id[j]` equals the id that was at `atoms.id[map.new_to_old[j]]`
// before, and likewise for all SoA fields. AtomId values are preserved —
// only indices change (master spec §8.2 atom-identity invariant).
void apply_reorder(AtomSoA& atoms, const ReorderMap& map);

}  // namespace tdmd
