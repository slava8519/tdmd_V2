#include "tdmd/state/atom_soa.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace {

bool is_aligned(const void* ptr, std::size_t alignment) {
  return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}

}  // namespace

TEST_CASE("AtomSoA default-constructs and reports empty", "[state][smoke]") {
  tdmd::AtomSoA atoms;
  REQUIRE(atoms.empty());
  REQUIRE(atoms.size() == 0);
  REQUIRE(atoms.invariants_hold());
}

TEST_CASE("AtomSoA is nothrow default-constructible", "[state][smoke]") {
  STATIC_REQUIRE(std::is_nothrow_default_constructible_v<tdmd::AtomSoA>);
}

TEST_CASE("AlignedAllocator reports 64-byte alignment", "[state][alignment]") {
  STATIC_REQUIRE(tdmd::AlignedAllocator<double, 64>::alignment == 64);
  STATIC_REQUIRE(tdmd::kSoaAlignment == 64);
}

TEST_CASE("AtomSoA buffers are 64-byte aligned after reserve", "[state][alignment]") {
  tdmd::AtomSoA atoms;
  atoms.reserve(128);
  atoms.resize(64);

  REQUIRE(is_aligned(atoms.id.data(), 64));
  REQUIRE(is_aligned(atoms.type.data(), 64));
  REQUIRE(is_aligned(atoms.x.data(), 64));
  REQUIRE(is_aligned(atoms.y.data(), 64));
  REQUIRE(is_aligned(atoms.z.data(), 64));
  REQUIRE(is_aligned(atoms.vx.data(), 64));
  REQUIRE(is_aligned(atoms.vy.data(), 64));
  REQUIRE(is_aligned(atoms.vz.data(), 64));
  REQUIRE(is_aligned(atoms.fx.data(), 64));
  REQUIRE(is_aligned(atoms.fy.data(), 64));
  REQUIRE(is_aligned(atoms.fz.data(), 64));
  REQUIRE(is_aligned(atoms.image_x.data(), 64));
  REQUIRE(is_aligned(atoms.image_y.data(), 64));
  REQUIRE(is_aligned(atoms.image_z.data(), 64));
  REQUIRE(is_aligned(atoms.flags.data(), 64));
}

TEST_CASE("AtomSoA add_atom appends fields consistently", "[state][add_remove]") {
  tdmd::AtomSoA atoms;
  const auto id1 = atoms.add_atom(/*type=*/1, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
  const auto id2 = atoms.add_atom(/*type=*/2, 4.0, 5.0, 6.0);

  REQUIRE(atoms.size() == 2);
  REQUIRE(atoms.invariants_hold());
  REQUIRE(id1 == 1);
  REQUIRE(id2 == 2);

  REQUIRE(atoms.id[0] == 1);
  REQUIRE(atoms.id[1] == 2);
  REQUIRE(atoms.type[0] == 1);
  REQUIRE(atoms.type[1] == 2);
  REQUIRE(atoms.x[0] == 1.0);
  REQUIRE(atoms.x[1] == 4.0);
  REQUIRE(atoms.vx[0] == 0.1);
  REQUIRE(atoms.vx[1] == 0.0);

  // Forces and images start at zero.
  REQUIRE(atoms.fx[0] == 0.0);
  REQUIRE(atoms.fy[0] == 0.0);
  REQUIRE(atoms.fz[0] == 0.0);
  REQUIRE(atoms.image_x[0] == 0);
  REQUIRE(atoms.image_y[0] == 0);
  REQUIRE(atoms.image_z[0] == 0);
  REQUIRE(atoms.flags[0] == 0u);
}

TEST_CASE("AtomSoA assigns monotonic never-reused AtomIds", "[state][identity]") {
  tdmd::AtomSoA atoms;
  const auto id1 = atoms.add_atom(0, 0.0, 0.0, 0.0);
  const auto id2 = atoms.add_atom(0, 0.0, 0.0, 0.0);
  atoms.remove_atom(0);
  const auto id3 = atoms.add_atom(0, 0.0, 0.0, 0.0);

  REQUIRE(id1 == 1);
  REQUIRE(id2 == 2);
  // id3 must not collide with previously-issued ids.
  REQUIRE(id3 == 3);
  REQUIRE(id3 != id1);
  REQUIRE(id3 != id2);
}

TEST_CASE("AtomSoA remove_atom uses swap-and-pop semantics", "[state][add_remove]") {
  tdmd::AtomSoA atoms;
  for (int i = 0; i < 5; ++i) {
    atoms.add_atom(/*type=*/static_cast<tdmd::SpeciesId>(i), static_cast<double>(i), 0.0, 0.0);
  }
  REQUIRE(atoms.size() == 5);

  const auto last_id = atoms.id.back();
  const auto last_type = atoms.type.back();
  const auto last_x = atoms.x.back();

  // Remove index 1. By swap-and-pop, what was at index 4 (last) moves to 1.
  atoms.remove_atom(1);

  REQUIRE(atoms.size() == 4);
  REQUIRE(atoms.invariants_hold());
  REQUIRE(atoms.id[1] == last_id);
  REQUIRE(atoms.type[1] == last_type);
  REQUIRE(atoms.x[1] == last_x);
}

TEST_CASE("AtomSoA remove_atom on last index just pops", "[state][add_remove]") {
  tdmd::AtomSoA atoms;
  atoms.add_atom(0, 1.0, 0.0, 0.0);
  atoms.add_atom(0, 2.0, 0.0, 0.0);
  atoms.remove_atom(1);

  REQUIRE(atoms.size() == 1);
  REQUIRE(atoms.x[0] == 1.0);
  REQUIRE(atoms.invariants_hold());
}

TEST_CASE("AtomSoA reserve keeps size at zero", "[state][capacity]") {
  tdmd::AtomSoA atoms;
  atoms.reserve(100);
  REQUIRE(atoms.size() == 0);
  REQUIRE(atoms.capacity() >= 100);
}

TEST_CASE("AtomSoA resize grows and zero-inits new atoms", "[state][resize]") {
  tdmd::AtomSoA atoms;
  atoms.resize(10);
  REQUIRE(atoms.size() == 10);
  REQUIRE(atoms.invariants_hold());
  for (std::size_t i = 0; i < 10; ++i) {
    REQUIRE(atoms.x[i] == 0.0);
    REQUIRE(atoms.fx[i] == 0.0);
    REQUIRE(atoms.image_x[i] == 0);
  }
  atoms.resize(3);
  REQUIRE(atoms.size() == 3);
  REQUIRE(atoms.invariants_hold());
}

TEST_CASE("AtomSoA clear keeps capacity", "[state][capacity]") {
  tdmd::AtomSoA atoms;
  atoms.reserve(64);
  atoms.add_atom(0, 1.0, 2.0, 3.0);
  atoms.add_atom(0, 4.0, 5.0, 6.0);
  const auto cap = atoms.capacity();
  atoms.clear();
  REQUIRE(atoms.empty());
  REQUIRE(atoms.capacity() == cap);
}

TEST_CASE("AtomSoA bulk add_remove stress stays consistent", "[state][stress]") {
  tdmd::AtomSoA atoms;
  constexpr std::size_t kN = 1024;
  for (std::size_t i = 0; i < kN; ++i) {
    atoms.add_atom(/*type=*/static_cast<tdmd::SpeciesId>(i % 4), static_cast<double>(i), 0.0, 0.0);
  }
  REQUIRE(atoms.size() == kN);
  REQUIRE(atoms.invariants_hold());

  // Remove 256 atoms from arbitrary positions; size must decrease by exactly 1
  // per call, invariants hold at every step.
  for (std::size_t k = 0; k < 256; ++k) {
    const std::size_t idx = (k * 7) % atoms.size();
    const std::size_t size_before = atoms.size();
    atoms.remove_atom(idx);
    REQUIRE(atoms.size() == size_before - 1);
    REQUIRE(atoms.invariants_hold());
  }
  REQUIRE(atoms.size() == kN - 256);
}
