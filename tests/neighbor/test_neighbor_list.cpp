#include "tdmd/neighbor/cell_grid.hpp"
#include "tdmd/neighbor/neighbor_list.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <utility>
#include <vector>

namespace {

tdmd::Box make_cubic_box(double length, bool periodic = true) {
  tdmd::Box box;
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

// Brute-force O(N^2) reference half-list used to check completeness of the
// cell-grid-accelerated build.
std::set<std::pair<std::uint32_t, std::uint32_t>> brute_force_half_pairs(const tdmd::AtomSoA& atoms,
                                                                         const tdmd::Box& box,
                                                                         double reach) {
  std::set<std::pair<std::uint32_t, std::uint32_t>> pairs;
  const double reach_sq = reach * reach;
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    for (std::size_t j = i + 1; j < atoms.size(); ++j) {
      const auto d = box.unwrap_minimum_image(atoms.x[j] - atoms.x[i],
                                              atoms.y[j] - atoms.y[i],
                                              atoms.z[j] - atoms.z[i]);
      const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      if (r2 <= reach_sq) {
        pairs.emplace(static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(j));
      }
    }
  }
  return pairs;
}

std::set<std::pair<std::uint32_t, std::uint32_t>> list_pairs(const tdmd::NeighborList& list) {
  std::set<std::pair<std::uint32_t, std::uint32_t>> pairs;
  const auto& offsets = list.page_offsets();
  const auto& ids = list.neigh_ids();
  for (std::size_t i = 0; i < list.atom_count(); ++i) {
    for (std::uint64_t k = offsets[i]; k < offsets[i + 1]; ++k) {
      pairs.emplace(static_cast<std::uint32_t>(i), ids[k]);
    }
  }
  return pairs;
}

}  // namespace

TEST_CASE("NeighborList builds with CSR layout and self-consistent offsets", "[neighbor][list]") {
  auto atoms = make_al_fcc_500();
  const auto box = make_cubic_box(20.25);
  tdmd::CellGrid grid;
  grid.build(box, 6.0, 0.3);
  grid.bin(atoms);

  tdmd::NeighborList list;
  list.build(atoms, box, grid, 6.0, 0.3);

  REQUIRE(list.atom_count() == atoms.size());
  const auto& off = list.page_offsets();
  REQUIRE(off.size() == atoms.size() + 1);
  REQUIRE(off.front() == 0);
  REQUIRE(off.back() == list.pair_count());
  for (std::size_t i = 0; i + 1 < off.size(); ++i) {
    REQUIRE(off[i] <= off[i + 1]);
  }
  REQUIRE(list.cutoff() == 6.0);
  REQUIRE(list.skin() == 0.3);
  REQUIRE(list.build_version() == 1);
}

TEST_CASE("NeighborList half-list invariant: every pair has j > i", "[neighbor][list][half]") {
  auto atoms = make_al_fcc_500();
  const auto box = make_cubic_box(20.25);
  tdmd::CellGrid grid;
  grid.build(box, 6.0, 0.3);
  grid.bin(atoms);

  tdmd::NeighborList list;
  list.build(atoms, box, grid, 6.0, 0.3);

  const auto& off = list.page_offsets();
  const auto& ids = list.neigh_ids();
  for (std::size_t i = 0; i < list.atom_count(); ++i) {
    for (std::uint64_t k = off[i]; k < off[i + 1]; ++k) {
      REQUIRE(ids[k] > i);
      REQUIRE(ids[k] < atoms.size());
    }
  }
}

TEST_CASE("NeighborList has no duplicate pairs", "[neighbor][list][dupes]") {
  auto atoms = make_al_fcc_500();
  const auto box = make_cubic_box(20.25);
  tdmd::CellGrid grid;
  grid.build(box, 6.0, 0.3);
  grid.bin(atoms);

  tdmd::NeighborList list;
  list.build(atoms, box, grid, 6.0, 0.3);

  const auto& off = list.page_offsets();
  const auto& ids = list.neigh_ids();
  for (std::size_t i = 0; i < list.atom_count(); ++i) {
    std::set<std::uint32_t> seen;
    for (std::uint64_t k = off[i]; k < off[i + 1]; ++k) {
      REQUIRE(seen.insert(ids[k]).second);
    }
  }
}

TEST_CASE("NeighborList r² matches true squared distance", "[neighbor][list][physics]") {
  auto atoms = make_al_fcc_500();
  const auto box = make_cubic_box(20.25);
  tdmd::CellGrid grid;
  grid.build(box, 6.0, 0.3);
  grid.bin(atoms);

  tdmd::NeighborList list;
  list.build(atoms, box, grid, 6.0, 0.3);

  const auto& off = list.page_offsets();
  const auto& ids = list.neigh_ids();
  const auto& r2 = list.neigh_r2();
  const double reach = 6.3;
  const double reach_sq = reach * reach;
  for (std::size_t i = 0; i < list.atom_count(); ++i) {
    for (std::uint64_t k = off[i]; k < off[i + 1]; ++k) {
      const std::size_t j = ids[k];
      const auto d = box.unwrap_minimum_image(atoms.x[j] - atoms.x[i],
                                              atoms.y[j] - atoms.y[i],
                                              atoms.z[j] - atoms.z[i]);
      const double expected = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      REQUIRE(r2[k] == expected);
      REQUIRE(r2[k] <= reach_sq);
    }
  }
}

TEST_CASE("NeighborList Al-FCC 500 yields expected average neighbors",
          "[neighbor][list][physics]") {
  auto atoms = make_al_fcc_500();
  const auto box = make_cubic_box(20.25);
  tdmd::CellGrid grid;
  grid.build(box, 6.0, 0.3);
  grid.bin(atoms);

  tdmd::NeighborList list;
  list.build(atoms, box, grid, 6.0, 0.3);

  const double avg_half_neighbors =
      static_cast<double>(list.pair_count()) / static_cast<double>(atoms.size());
  // FCC has 12 nearest at a/sqrt(2)≈2.86Å, 6 at a≈4.05Å, 24 at ~4.96Å (+12
  // more inside 6.3). Full neighbor count ≈ 54; half-list ≈ 27 on average.
  REQUIRE(avg_half_neighbors > 20.0);
  REQUIRE(avg_half_neighbors < 45.0);
}

TEST_CASE("NeighborList completeness — brute-force property test (10⁴ cases)",
          "[neighbor][list][property]") {
  std::mt19937_64 rng(0xFAB1ED1U);
  std::uniform_real_distribution<double> uni(0.0, 18.5);

  constexpr int kTrials = 10000;
  const double cutoff = 5.0;
  const double skin = 0.3;
  const double reach = cutoff + skin;
  const double L = 18.5;

  const auto box = make_cubic_box(L);
  for (int t = 0; t < kTrials; ++t) {
    tdmd::AtomSoA atoms;
    for (int k = 0; k < 10; ++k) {
      atoms.add_atom(0, uni(rng), uni(rng), uni(rng));
    }
    tdmd::CellGrid grid;
    grid.build(box, cutoff, skin);
    grid.bin(atoms);

    tdmd::NeighborList list;
    list.build(atoms, box, grid, cutoff, skin);

    const auto expected = brute_force_half_pairs(atoms, box, reach);
    const auto actual = list_pairs(list);
    REQUIRE(actual == expected);
  }
}

TEST_CASE("NeighborList is deterministic — same input yields byte-identical list",
          "[neighbor][list][determinism]") {
  std::mt19937_64 rng(0xC0FFEEU);
  std::uniform_real_distribution<double> uni(0.0, 20.0);

  const auto box = make_cubic_box(20.25);
  tdmd::AtomSoA atoms;
  for (int k = 0; k < 64; ++k) {
    atoms.add_atom(0, uni(rng), uni(rng), uni(rng));
  }

  tdmd::CellGrid grid_a;
  grid_a.build(box, 6.0, 0.3);
  grid_a.bin(atoms);
  tdmd::NeighborList list_a;
  list_a.build(atoms, box, grid_a, 6.0, 0.3);

  tdmd::CellGrid grid_b;
  grid_b.build(box, 6.0, 0.3);
  grid_b.bin(atoms);
  tdmd::NeighborList list_b;
  list_b.build(atoms, box, grid_b, 6.0, 0.3);

  REQUIRE(list_a.page_offsets() == list_b.page_offsets());
  REQUIRE(list_a.neigh_ids() == list_b.neigh_ids());
  REQUIRE(list_a.neigh_r2() == list_b.neigh_r2());
}

TEST_CASE("NeighborList clear() zeroes buffers but keeps type invariants",
          "[neighbor][list][lifecycle]") {
  auto atoms = make_al_fcc_500();
  const auto box = make_cubic_box(20.25);
  tdmd::CellGrid grid;
  grid.build(box, 6.0, 0.3);
  grid.bin(atoms);

  tdmd::NeighborList list;
  list.build(atoms, box, grid, 6.0, 0.3);
  REQUIRE(list.pair_count() > 0);

  list.clear();
  REQUIRE(list.pair_count() == 0);
  REQUIRE(list.atom_count() == 0);
}
