#include "tdmd/neighbor/cell_grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <utility>

namespace tdmd {

namespace {

constexpr std::uint32_t kMinCellsPerAxis = 3;

[[nodiscard]] std::uint32_t cells_for_axis(double length, double w) {
  if (length <= 0.0 || w <= 0.0) {
    return 0;
  }
  const auto raw = static_cast<std::uint32_t>(std::floor(length / w));
  return raw < kMinCellsPerAxis ? kMinCellsPerAxis : raw;
}

[[nodiscard]] std::size_t cell_index_axis(double coord,
                                          double lo,
                                          double cell_size,
                                          std::uint32_t n) noexcept {
  if (cell_size <= 0.0 || n == 0) {
    return 0;
  }
  const double local = coord - lo;
  auto idx = static_cast<std::int64_t>(std::floor(local / cell_size));
  if (idx < 0) {
    idx = 0;
  } else if (idx >= static_cast<std::int64_t>(n)) {
    idx = static_cast<std::int64_t>(n) - 1;
  }
  return static_cast<std::size_t>(idx);
}

[[nodiscard]] std::uint32_t wrap_axis(std::int32_t idx, std::uint32_t n) noexcept {
  const auto ni = static_cast<std::int32_t>(n);
  std::int32_t w = idx % ni;
  if (w < 0) {
    w += ni;
  }
  return static_cast<std::uint32_t>(w);
}

template <typename Vec>
void apply_permutation(Vec& field, const std::vector<std::uint64_t>& new_to_old) {
  Vec scratch(field.size());
  for (std::size_t i = 0; i < field.size(); ++i) {
    scratch[i] = field[new_to_old[i]];
  }
  field.swap(scratch);
}

}  // namespace

void CellGrid::build(const Box& box, double cutoff, double skin) {
  if (!box.is_valid_m1()) {
    throw InvalidCellGridError(
        "CellGrid::build: box is not M1-valid (non-positive length or non-zero tilt)");
  }
  if (!(cutoff > 0.0)) {
    throw InvalidCellGridError("CellGrid::build: cutoff must be > 0");
  }
  if (skin < 0.0) {
    throw InvalidCellGridError("CellGrid::build: skin must be >= 0");
  }

  const double w = cutoff + skin;
  const std::uint32_t nx = cells_for_axis(box.lx(), w);
  const std::uint32_t ny = cells_for_axis(box.ly(), w);
  const std::uint32_t nz = cells_for_axis(box.lz(), w);

  const bool too_small_x = (box.lx() / w) < static_cast<double>(kMinCellsPerAxis);
  const bool too_small_y = (box.ly() / w) < static_cast<double>(kMinCellsPerAxis);
  const bool too_small_z = (box.lz() / w) < static_cast<double>(kMinCellsPerAxis);
  if (too_small_x || too_small_y || too_small_z) {
    throw InvalidCellGridError(
        "CellGrid::build: box too small for 3-cells-per-axis stencil; required "
        "L_axis >= 3 * (cutoff + skin) = " +
        std::to_string(3.0 * w));
  }

  box_ = box;
  cutoff_ = cutoff;
  skin_ = skin;
  nx_ = nx;
  ny_ = ny;
  nz_ = nz;
  cell_x_ = box.lx() / static_cast<double>(nx);
  cell_y_ = box.ly() / static_cast<double>(ny);
  cell_z_ = box.lz() / static_cast<double>(nz);

  const std::size_t total = cell_count();
  cell_offsets_.assign(total + 1, 0);
  cell_atoms_.clear();
  ++build_version_;
}

void CellGrid::bin(const AtomSoA& atoms) {
  const std::size_t total = cell_count();
  cell_offsets_.assign(total + 1, 0);
  cell_atoms_.assign(atoms.size(), 0);

  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const std::size_t cid = cell_of(atoms.x[i], atoms.y[i], atoms.z[i]);
    ++cell_offsets_[cid + 1];
  }

  for (std::size_t c = 0; c < total; ++c) {
    cell_offsets_[c + 1] += cell_offsets_[c];
  }

  std::vector<std::uint32_t> cursor(
      cell_offsets_.begin(),
      std::next(cell_offsets_.begin(), static_cast<std::ptrdiff_t>(total)));
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const std::size_t cid = cell_of(atoms.x[i], atoms.y[i], atoms.z[i]);
    cell_atoms_[cursor[cid]++] = static_cast<std::uint32_t>(i);
  }
}

std::size_t CellGrid::cell_of(double x, double y, double z) const noexcept {
  const std::size_t ix = cell_index_axis(x, box_.xlo, cell_x_, nx_);
  const std::size_t iy = cell_index_axis(y, box_.ylo, cell_y_, ny_);
  const std::size_t iz = cell_index_axis(z, box_.zlo, cell_z_, nz_);
  return linear_index(static_cast<int>(ix), static_cast<int>(iy), static_cast<int>(iz));
}

std::size_t CellGrid::linear_index(int ix, int iy, int iz) const noexcept {
  return static_cast<std::size_t>(ix) +
         nx_ * (static_cast<std::size_t>(iy) +
                static_cast<std::size_t>(ny_) * static_cast<std::size_t>(iz));
}

std::array<std::size_t, 27> CellGrid::neighbor_cells(std::size_t cell_idx) const noexcept {
  std::array<std::size_t, 27> out{};
  if (cell_count() == 0) {
    return out;
  }
  const auto iz = static_cast<std::int32_t>(cell_idx / (static_cast<std::size_t>(nx_) * ny_));
  const auto rem = cell_idx - static_cast<std::size_t>(iz) * nx_ * ny_;
  const auto iy = static_cast<std::int32_t>(rem / nx_);
  const auto ix = static_cast<std::int32_t>(rem % nx_);

  std::size_t k = 0;
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const std::uint32_t jx = wrap_axis(ix + dx, nx_);
        const std::uint32_t jy = wrap_axis(iy + dy, ny_);
        const std::uint32_t jz = wrap_axis(iz + dz, nz_);
        out[k++] = linear_index(static_cast<int>(jx), static_cast<int>(jy), static_cast<int>(jz));
      }
    }
  }
  return out;
}

ReorderMap CellGrid::compute_stable_reorder(const AtomSoA& atoms) const {
  ReorderMap map;
  const std::size_t n = atoms.size();
  map.old_to_new.assign(n, 0);
  map.new_to_old.assign(n, 0);
  if (n == 0) {
    return map;
  }

  std::vector<std::uint64_t> indices(n);
  std::iota(indices.begin(), indices.end(), std::uint64_t{0});

  std::vector<std::size_t> keys(n);
  for (std::size_t i = 0; i < n; ++i) {
    keys[i] = cell_of(atoms.x[i], atoms.y[i], atoms.z[i]);
  }

  std::stable_sort(indices.begin(), indices.end(), [&keys](std::uint64_t a, std::uint64_t b) {
    return keys[a] < keys[b];
  });

  for (std::size_t new_idx = 0; new_idx < n; ++new_idx) {
    const std::uint64_t old_idx = indices[new_idx];
    map.new_to_old[new_idx] = old_idx;
    map.old_to_new[old_idx] = new_idx;
  }
  return map;
}

void apply_reorder(AtomSoA& atoms, const ReorderMap& map) {
  if (atoms.size() != map.size()) {
    throw std::invalid_argument("apply_reorder: size mismatch between atoms and map");
  }
  if (atoms.empty()) {
    return;
  }
  apply_permutation(atoms.id, map.new_to_old);
  apply_permutation(atoms.type, map.new_to_old);
  apply_permutation(atoms.x, map.new_to_old);
  apply_permutation(atoms.y, map.new_to_old);
  apply_permutation(atoms.z, map.new_to_old);
  apply_permutation(atoms.vx, map.new_to_old);
  apply_permutation(atoms.vy, map.new_to_old);
  apply_permutation(atoms.vz, map.new_to_old);
  apply_permutation(atoms.fx, map.new_to_old);
  apply_permutation(atoms.fy, map.new_to_old);
  apply_permutation(atoms.fz, map.new_to_old);
  apply_permutation(atoms.image_x, map.new_to_old);
  apply_permutation(atoms.image_y, map.new_to_old);
  apply_permutation(atoms.image_z, map.new_to_old);
  apply_permutation(atoms.flags, map.new_to_old);
}

}  // namespace tdmd
