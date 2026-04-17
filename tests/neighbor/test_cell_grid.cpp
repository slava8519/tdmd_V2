#include "tdmd/neighbor/cell_grid.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

namespace {

tdmd::Box make_cubic_box(double length, bool periodic = true) {
  tdmd::Box box;
  box.xlo = 0.0;
  box.ylo = 0.0;
  box.zlo = 0.0;
  box.xhi = length;
  box.yhi = length;
  box.zhi = length;
  box.periodic_x = periodic;
  box.periodic_y = periodic;
  box.periodic_z = periodic;
  return box;
}

tdmd::AtomSoA make_al_fcc_500(double a = 4.05) {
  tdmd::AtomSoA atoms;
  for (int k = 0; k < 5; ++k) {
    for (int j = 0; j < 5; ++j) {
      for (int i = 0; i < 5; ++i) {
        const double x0 = i * a;
        const double y0 = j * a;
        const double z0 = k * a;
        atoms.add_atom(0, x0, y0, z0);
        atoms.add_atom(0, x0 + 0.5 * a, y0 + 0.5 * a, z0);
        atoms.add_atom(0, x0 + 0.5 * a, y0, z0 + 0.5 * a);
        atoms.add_atom(0, x0, y0 + 0.5 * a, z0 + 0.5 * a);
      }
    }
  }
  return atoms;
}

double min_image_delta(double d, double L) {
  while (d > 0.5 * L) {
    d -= L;
  }
  while (d < -0.5 * L) {
    d += L;
  }
  return d;
}

}  // namespace

TEST_CASE("CellGrid default-constructs as empty", "[neighbor][smoke]") {
  tdmd::CellGrid grid;
  REQUIRE(grid.empty());
  REQUIRE(grid.cell_count() == 0);
}

TEST_CASE("CellGrid is nothrow default-constructible", "[neighbor][smoke]") {
  STATIC_REQUIRE(std::is_nothrow_default_constructible_v<tdmd::CellGrid>);
}

TEST_CASE("CellGrid::build rejects invalid inputs", "[neighbor][cell_grid]") {
  tdmd::CellGrid grid;
  const auto box = make_cubic_box(20.25);

  REQUIRE_THROWS_AS(grid.build(box, -1.0, 0.3), tdmd::InvalidCellGridError);
  REQUIRE_THROWS_AS(grid.build(box, 6.0, -0.1), tdmd::InvalidCellGridError);

  const auto small = make_cubic_box(10.0);
  REQUIRE_THROWS_AS(grid.build(small, 6.0, 0.3), tdmd::InvalidCellGridError);

  tdmd::Box triclinic = make_cubic_box(20.25);
  triclinic.tilt_xy = 0.1;
  REQUIRE_THROWS_AS(grid.build(triclinic, 6.0, 0.3), tdmd::InvalidCellGridError);
}

TEST_CASE("CellGrid::build sizes Al-FCC 5x5x5 box to expected cells", "[neighbor][cell_grid]") {
  tdmd::CellGrid grid;
  const auto box = make_cubic_box(20.25);
  grid.build(box, 6.0, 0.3);

  REQUIRE(grid.nx() == 3);
  REQUIRE(grid.ny() == 3);
  REQUIRE(grid.nz() == 3);
  REQUIRE(grid.cell_count() == 27);
  REQUIRE(grid.cell_x() >= 6.3);
  REQUIRE(grid.cell_y() >= 6.3);
  REQUIRE(grid.cell_z() >= 6.3);
}

TEST_CASE("CellGrid::bin on Al-FCC 500 atoms: all atoms accounted for", "[neighbor][cell_grid]") {
  tdmd::CellGrid grid;
  const auto box = make_cubic_box(20.25);
  grid.build(box, 6.0, 0.3);

  auto atoms = make_al_fcc_500();
  REQUIRE(atoms.size() == 500);
  grid.bin(atoms);

  const auto& offsets = grid.cell_offsets();
  REQUIRE(offsets.size() == grid.cell_count() + 1);
  REQUIRE(offsets.front() == 0);
  REQUIRE(offsets.back() == atoms.size());

  std::vector<bool> seen(atoms.size(), false);
  const auto& cell_atoms = grid.cell_atoms();
  REQUIRE(cell_atoms.size() == atoms.size());
  for (auto idx : cell_atoms) {
    REQUIRE(idx < atoms.size());
    REQUIRE_FALSE(seen[idx]);
    seen[idx] = true;
  }
}

TEST_CASE("CellGrid::bin is stable — within-cell order matches input index",
          "[neighbor][cell_grid][stable]") {
  tdmd::CellGrid grid;
  const auto box = make_cubic_box(20.25);
  grid.build(box, 6.0, 0.3);
  auto atoms = make_al_fcc_500();
  grid.bin(atoms);

  const auto& offsets = grid.cell_offsets();
  const auto& cell_atoms = grid.cell_atoms();
  for (std::size_t c = 0; c < grid.cell_count(); ++c) {
    for (std::size_t k = offsets[c] + 1; k < offsets[c + 1]; ++k) {
      REQUIRE(cell_atoms[k - 1] < cell_atoms[k]);
    }
  }
}

TEST_CASE("CellGrid::neighbor_cells visits 27 unique cells for nx,ny,nz>=3",
          "[neighbor][cell_grid][stencil]") {
  tdmd::CellGrid grid;
  const auto box = make_cubic_box(20.25);
  grid.build(box, 6.0, 0.3);

  for (std::size_t c = 0; c < grid.cell_count(); ++c) {
    const auto stencil = grid.neighbor_cells(c);
    std::set<std::size_t> unique(stencil.begin(), stencil.end());
    REQUIRE(unique.size() == 27);
    REQUIRE(unique.contains(c));
    for (auto idx : stencil) {
      REQUIRE(idx < grid.cell_count());
    }
  }
}

TEST_CASE("CellGrid::neighbor_cells handles tiny 3x3x3 box without duplication",
          "[neighbor][cell_grid][edge]") {
  tdmd::CellGrid grid;
  const auto box = make_cubic_box(18.9001);
  grid.build(box, 6.0, 0.3);
  REQUIRE(grid.nx() == 3);
  REQUIRE(grid.ny() == 3);
  REQUIRE(grid.nz() == 3);

  const auto stencil = grid.neighbor_cells(0);
  std::set<std::size_t> unique(stencil.begin(), stencil.end());
  REQUIRE(unique.size() == 27);
}

TEST_CASE("CellGrid neighbor stencil covers all pairs within cutoff+skin",
          "[neighbor][cell_grid][property]") {
  std::mt19937_64 rng(0xC3110C0U);
  std::uniform_real_distribution<double> uni_pos(0.0, 18.0);
  std::uniform_real_distribution<double> uni_disp(-0.5, 0.5);

  const double cutoff = 5.0;
  const double skin = 0.3;
  const double reach = cutoff + skin;
  const double reach_sq = reach * reach;

  constexpr int kTrials = 10000;
  constexpr int kAtomsPerTrial = 8;
  for (int t = 0; t < kTrials; ++t) {
    tdmd::CellGrid grid;
    const double L = 18.0 + uni_disp(rng);
    const auto box = make_cubic_box(L);
    grid.build(box, cutoff, skin);

    tdmd::AtomSoA atoms;
    for (int k = 0; k < kAtomsPerTrial; ++k) {
      const double x = uni_pos(rng);
      const double y = uni_pos(rng);
      const double z = uni_pos(rng);
      atoms.add_atom(0, x, y, z);
    }
    grid.bin(atoms);

    for (std::size_t i = 0; i < atoms.size(); ++i) {
      const std::size_t ci = grid.cell_of(atoms.x[i], atoms.y[i], atoms.z[i]);
      const auto stencil = grid.neighbor_cells(ci);
      std::set<std::size_t> stencil_set(stencil.begin(), stencil.end());
      for (std::size_t j = 0; j < atoms.size(); ++j) {
        if (i == j) {
          continue;
        }
        double dx = min_image_delta(atoms.x[i] - atoms.x[j], L);
        double dy = min_image_delta(atoms.y[i] - atoms.y[j], L);
        double dz = min_image_delta(atoms.z[i] - atoms.z[j], L);
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 <= reach_sq) {
          const std::size_t cj = grid.cell_of(atoms.x[j], atoms.y[j], atoms.z[j]);
          REQUIRE(stencil_set.contains(cj));
        }
      }
    }
  }
}

TEST_CASE("CellGrid::compute_stable_reorder is deterministic — same input → same map",
          "[neighbor][cell_grid][determinism][property]") {
  std::mt19937_64 rng(0xDEADB3A7U);
  std::uniform_real_distribution<double> uni(0.0, 20.0);

  const double L = 20.25;
  const auto box = make_cubic_box(L);
  constexpr int kTrials = 10000;
  for (int t = 0; t < kTrials; ++t) {
    tdmd::AtomSoA atoms;
    for (int k = 0; k < 16; ++k) {
      atoms.add_atom(0, uni(rng), uni(rng), uni(rng));
    }

    tdmd::CellGrid grid_a;
    grid_a.build(box, 6.0, 0.3);
    grid_a.bin(atoms);
    const auto map_a = grid_a.compute_stable_reorder(atoms);

    tdmd::CellGrid grid_b;
    grid_b.build(box, 6.0, 0.3);
    grid_b.bin(atoms);
    const auto map_b = grid_b.compute_stable_reorder(atoms);

    REQUIRE(map_a.old_to_new == map_b.old_to_new);
    REQUIRE(map_a.new_to_old == map_b.new_to_old);
  }
}

TEST_CASE("ReorderMap round-trip preserves AtomSoA fields exactly",
          "[neighbor][cell_grid][reorder]") {
  tdmd::AtomSoA atoms = make_al_fcc_500();
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    atoms.vx[i] = static_cast<double>(i) * 0.125;
    atoms.vy[i] = -static_cast<double>(i) * 0.5;
    atoms.vz[i] = 1.0;
    atoms.fx[i] = 0.1 * static_cast<double>(i);
    atoms.image_x[i] = static_cast<std::int32_t>(i % 7) - 3;
    atoms.flags[i] = static_cast<std::uint32_t>(i * 3u);
  }

  std::vector<tdmd::AtomId> id_before(atoms.id.begin(), atoms.id.end());
  std::vector<double> x_before(atoms.x.begin(), atoms.x.end());
  std::vector<double> vx_before(atoms.vx.begin(), atoms.vx.end());
  std::vector<std::int32_t> image_x_before(atoms.image_x.begin(), atoms.image_x.end());
  std::vector<std::uint32_t> flags_before(atoms.flags.begin(), atoms.flags.end());

  tdmd::CellGrid grid;
  grid.build(make_cubic_box(20.25), 6.0, 0.3);
  grid.bin(atoms);
  const auto map = grid.compute_stable_reorder(atoms);
  tdmd::apply_reorder(atoms, map);

  REQUIRE(atoms.invariants_hold());
  REQUIRE(atoms.size() == id_before.size());

  for (std::size_t old_idx = 0; old_idx < id_before.size(); ++old_idx) {
    const std::size_t new_idx = map.old_to_new[old_idx];
    REQUIRE(atoms.id[new_idx] == id_before[old_idx]);
    REQUIRE(atoms.x[new_idx] == x_before[old_idx]);
    REQUIRE(atoms.vx[new_idx] == vx_before[old_idx]);
    REQUIRE(atoms.image_x[new_idx] == image_x_before[old_idx]);
    REQUIRE(atoms.flags[new_idx] == flags_before[old_idx]);
  }
}

TEST_CASE("ReorderMap groups same-cell atoms contiguously after apply",
          "[neighbor][cell_grid][reorder]") {
  tdmd::AtomSoA atoms = make_al_fcc_500();
  tdmd::CellGrid grid;
  grid.build(make_cubic_box(20.25), 6.0, 0.3);
  grid.bin(atoms);
  const auto map = grid.compute_stable_reorder(atoms);
  tdmd::apply_reorder(atoms, map);

  grid.bin(atoms);
  const auto& cell_atoms = grid.cell_atoms();
  for (std::size_t k = 1; k < cell_atoms.size(); ++k) {
    REQUIRE(cell_atoms[k - 1] + 1 == cell_atoms[k]);
  }
}

TEST_CASE("apply_reorder composed with inverse recovers original order",
          "[neighbor][cell_grid][reorder]") {
  tdmd::AtomSoA atoms = make_al_fcc_500();
  std::vector<tdmd::AtomId> id_before(atoms.id.begin(), atoms.id.end());
  std::vector<double> x_before(atoms.x.begin(), atoms.x.end());

  tdmd::CellGrid grid;
  grid.build(make_cubic_box(20.25), 6.0, 0.3);
  grid.bin(atoms);
  const auto fwd = grid.compute_stable_reorder(atoms);
  tdmd::apply_reorder(atoms, fwd);

  tdmd::ReorderMap inverse;
  inverse.old_to_new = fwd.new_to_old;
  inverse.new_to_old = fwd.old_to_new;
  tdmd::apply_reorder(atoms, inverse);

  REQUIRE(atoms.size() == id_before.size());
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    REQUIRE(atoms.id[i] == id_before[i]);
    REQUIRE(atoms.x[i] == x_before[i]);
  }
}
