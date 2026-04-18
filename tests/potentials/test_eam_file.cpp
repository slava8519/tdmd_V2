// SPEC: docs/specs/potentials/SPEC.md §4.5 (EAM file format).
// Exec pack: docs/development/m2_execution_pack.md T2.6.
//
// Tests for parse_eam_alloy / parse_eam_fs against hand-authored fixtures
// (tests/potentials/fixtures/). Covers happy path for both variants,
// sensitive-to-mis-indexing multi-species layout (FS), and the malformed-
// input error reporting contract.

#include "tdmd/potentials/eam_file.hpp"
#include "tdmd/potentials/tabulated.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef TDMD_TEST_FIXTURES_DIR
#error "TDMD_TEST_FIXTURES_DIR must be defined by the build system"
#endif

using Catch::Matchers::ContainsSubstring;
using tdmd::potentials::EamAlloyData;
using tdmd::potentials::EamFsData;
using tdmd::potentials::parse_eam_alloy;
using tdmd::potentials::parse_eam_fs;

namespace {

std::filesystem::path fixture_path(const std::string& name) {
  return std::filesystem::path(TDMD_TEST_FIXTURES_DIR) / name;
}

// Write a string to a throwaway file in CMake's build tree so we can feed
// deliberately malformed inputs to the parser without polluting the
// committed fixtures.
std::filesystem::path write_tmp(const std::string& contents, const std::string& name) {
  const auto path = std::filesystem::temp_directory_path() / ("tdmd_eam_test_" + name);
  std::ofstream out(path);
  REQUIRE(out);
  out << contents;
  return path;
}

}  // namespace

TEST_CASE("parse_eam_alloy: Al single-species fixture", "[potentials][eam][file]") {
  const auto data = parse_eam_alloy(fixture_path("Al_small.eam.alloy").string());

  REQUIRE(data.species_names.size() == 1);
  REQUIRE(data.species_names[0] == "Al");
  REQUIRE(data.masses.size() == 1);
  REQUIRE(data.masses[0] == 26.98);

  REQUIRE(data.nrho == 10);
  REQUIRE(data.drho == 0.1);
  REQUIRE(data.nr == 10);
  REQUIRE(data.dr == 0.5);
  REQUIRE(data.cutoff == 5.0);

  // Grids: F(ρ) sampled on ρ = k·0.1; ρ(r) on r = k·0.5. Interior grid
  // points land at p=0 of their cell and return y[k] bitwise; the rightmost
  // node hits the clamp (cell n-2 at p=1) which is only mathematically —
  // not bit-exactly — equal to y[n-1] when the end cell has nonzero b/c.
  REQUIRE(data.F_rho.size() == 1);
  REQUIRE(data.F_rho[0].size() == 10);
  REQUIRE(data.F_rho[0].x0() == 0.0);
  REQUIRE(data.F_rho[0].dx() == 0.1);
  REQUIRE(data.F_rho[0].eval(0.0) == 0.0);   // interior grid node
  REQUIRE(data.F_rho[0].eval(0.4) == -0.4);  // interior grid node
  REQUIRE(data.F_rho[0].eval(0.5) == -0.5);  // interior grid node

  REQUIRE(data.rho_r.size() == 1);
  REQUIRE(data.rho_r[0].size() == 10);
  REQUIRE(data.rho_r[0].eval(0.0) == 1.0);  // interior grid node
  REQUIRE(data.rho_r[0].eval(1.0) == 0.8);  // interior grid node

  // Single species → 1 pair table (z2r_Al-Al).
  REQUIRE(data.z2r.size() == 1);
  REQUIRE(data.z2r[0].size() == 10);
  REQUIRE(data.z2r[0].eval(0.0) == 100.0);
  REQUIRE(data.z2r[0].eval(2.0) == 30.0);  // interior grid node

  REQUIRE(EamAlloyData::pair_index(0, 0) == 0);
}

TEST_CASE("parse_eam_fs: AlNi two-species fixture", "[potentials][eam][file]") {
  const auto data = parse_eam_fs(fixture_path("AlNi_small.eam.fs").string());

  REQUIRE(data.species_names == std::vector<std::string>{"Al", "Ni"});
  REQUIRE(data.masses.size() == 2);
  REQUIRE(data.masses[0] == 26.98);
  REQUIRE(data.masses[1] == 58.69);

  REQUIRE(data.nrho == 5);
  REQUIRE(data.drho == 0.2);
  REQUIRE(data.nr == 5);
  REQUIRE(data.dr == 1.0);
  REQUIRE(data.cutoff == 4.0);

  // Evaluate at interior grid nodes — rightmost (x = (n-1)*dx) hits the
  // boundary clamp and is only mathematically equal to y[n-1], so we
  // verify at ρ = 0 and ρ = 0.4 (both interior).
  REQUIRE(data.F_rho.size() == 2);
  REQUIRE(data.F_rho[0].eval(0.0) == -0.10);
  REQUIRE(data.F_rho[0].eval(0.4) == -0.30);
  REQUIRE(data.F_rho[1].eval(0.0) == -0.01);
  REQUIRE(data.F_rho[1].eval(0.4) == -0.03);

  // Finnis-Sinclair has N × N ρ_{αβ}(r) tables, packed row-major by (α, β).
  // The fixture was written so ρ_{Al←Al}, ρ_{Al←Ni}, ρ_{Ni←Al}, ρ_{Ni←Ni}
  // all have distinct leading coefficients so a mis-indexing mistake is
  // caught immediately.
  REQUIRE(data.rho_ij.size() == 4);
  const auto n = data.species_names.size();
  const auto at = [&](std::size_t a, std::size_t b, double r) {
    return data.rho_ij[EamFsData::rho_ij_index(a, b, n)].eval(r);
  };
  REQUIRE(at(0, 0, 0.0) == 1.10);  // Al ← Al
  REQUIRE(at(0, 1, 0.0) == 1.12);  // Al ← Ni
  REQUIRE(at(1, 0, 0.0) == 2.10);  // Ni ← Al
  REQUIRE(at(1, 1, 0.0) == 2.12);  // Ni ← Ni

  // 3 pair tables for N = 2 (Al-Al, Ni-Al, Ni-Ni; lower-triangular packing).
  REQUIRE(data.z2r.size() == 3);
  REQUIRE(data.z2r[EamFsData::pair_index(0, 0)].eval(0.0) == 10.0);
  REQUIRE(data.z2r[EamFsData::pair_index(1, 0)].eval(0.0) == 5.0);
  REQUIRE(data.z2r[EamFsData::pair_index(0, 1)].eval(0.0) == 5.0);  // symmetric
  REQUIRE(data.z2r[EamFsData::pair_index(1, 1)].eval(0.0) == 1.0);
}

TEST_CASE("parse_eam_alloy: missing file", "[potentials][eam][file]") {
  REQUIRE_THROWS_WITH(parse_eam_alloy("/nonexistent/path/to/file.eam.alloy"),
                      ContainsSubstring("cannot open"));
}

TEST_CASE("parse_eam_alloy: malformed N_species line", "[potentials][eam][file]") {
  // Mismatched name count — 2 declared, 1 name given.
  const std::string bad =
      "# comment 1\n"
      "# comment 2\n"
      "# comment 3\n"
      "2 Al\n"
      "10 0.1 10 0.5 5.0\n";
  const auto p = write_tmp(bad, "bad_species.eam.alloy");
  REQUIRE_THROWS_WITH(parse_eam_alloy(p.string()), ContainsSubstring(":4:"));
}

TEST_CASE("parse_eam_alloy: truncated array", "[potentials][eam][file]") {
  // Declares N_rho = 10, but only supplies 3 F values before EOF.
  const std::string bad =
      "# comment 1\n"
      "# comment 2\n"
      "# comment 3\n"
      "1 Al\n"
      "10 0.1 10 0.5 5.0\n"
      "13 26.98 4.05 FCC\n"
      "0.0 -0.1 -0.2\n";
  const auto p = write_tmp(bad, "truncated.eam.alloy");
  REQUIRE_THROWS_WITH(parse_eam_alloy(p.string()), ContainsSubstring("end of file"));
}

TEST_CASE("parse_eam_alloy: non-numeric where number expected", "[potentials][eam][file]") {
  const std::string bad =
      "# c1\n"
      "# c2\n"
      "# c3\n"
      "1 Al\n"
      "10 0.1 10 0.5 NOT_A_NUMBER\n";
  const auto p = write_tmp(bad, "bad_cutoff.eam.alloy");
  REQUIRE_THROWS_WITH(parse_eam_alloy(p.string()), ContainsSubstring(":5:"));
}

TEST_CASE("parse_eam_alloy: grid parameters out of range", "[potentials][eam][file]") {
  // N_rho < 5 rejected (spline minimum) — must be caught before the bulk
  // numeric read so malformed inputs don't wander into the array loop.
  const std::string bad =
      "# c1\n"
      "# c2\n"
      "# c3\n"
      "1 Al\n"
      "3 0.1 10 0.5 5.0\n";
  const auto p = write_tmp(bad, "small_nrho.eam.alloy");
  REQUIRE_THROWS_WITH(parse_eam_alloy(p.string()), ContainsSubstring("≥ 5"));
}

TEST_CASE("EamAlloyData::pair_index is symmetric", "[potentials][eam][file]") {
  // Packing must agree with the LAMMPS convention that the file lists
  // (i, j) pairs with i ≥ j. Symmetric access lets the force kernel look
  // up φ_{αβ} without pre-sorting the pair.
  for (std::size_t a = 0; a < 4; ++a) {
    for (std::size_t b = 0; b < 4; ++b) {
      REQUIRE(EamAlloyData::pair_index(a, b) == EamAlloyData::pair_index(b, a));
      REQUIRE(EamFsData::pair_index(a, b) == EamFsData::pair_index(b, a));
    }
  }
  // Ensure the packing is contiguous starting at 0 for lower-triangular.
  REQUIRE(EamAlloyData::pair_index(0, 0) == 0);
  REQUIRE(EamAlloyData::pair_index(1, 0) == 1);
  REQUIRE(EamAlloyData::pair_index(1, 1) == 2);
  REQUIRE(EamAlloyData::pair_index(2, 0) == 3);
  REQUIRE(EamAlloyData::pair_index(2, 2) == 5);
}
