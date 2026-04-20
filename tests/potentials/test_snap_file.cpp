// SPEC: docs/specs/potentials/SPEC.md §6.6 (SNAP parameter file format).
// Exec pack: docs/development/m8_execution_pack.md T8.4a.
//
// Unit tests for parse_snap_coeff / parse_snap_param / parse_snap_files
// + SnapPotential skeleton validator. Canonical fixture is the M1-landed
// LAMMPS submodule `verify/third_party/lammps/examples/snap/W_2940_2017_2.*`
// (chosen T8.2 — Wood & Thompson 2017, arXiv:1702.07042). Self-skips with
// exit 77 when the submodule isn't initialized (Option A / public CI).
//
// Hand-assembled malformed fixtures live в temporary files написанных
// через std::filesystem::temp_directory_path() so что they don't pollute
// the committed tree.

#include "tdmd/potentials/snap.hpp"
#include "tdmd/potentials/snap_file.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#ifndef TDMD_TEST_FIXTURES_DIR
#error "TDMD_TEST_FIXTURES_DIR must be defined by the build system"
#endif

namespace fs = std::filesystem;
using Catch::Matchers::ContainsSubstring;
using tdmd::potentials::parse_snap_coeff;
using tdmd::potentials::parse_snap_files;
using tdmd::potentials::parse_snap_param;
using tdmd::potentials::snap_k_max;
using tdmd::potentials::SnapData;
using tdmd::potentials::SnapParams;
using tdmd::potentials::SnapSpecies;

namespace {

constexpr int kExitSkip = 77;

fs::path lammps_snap_examples_dir() {
  const fs::path fixtures_dir = TDMD_TEST_FIXTURES_DIR;
  const fs::path repo_root = fixtures_dir.parent_path().parent_path().parent_path();
  return repo_root / "verify" / "third_party" / "lammps" / "examples" / "snap";
}

void skip_if_submodule_uninitialized(const fs::path& snap_dir) {
  if (!fs::exists(snap_dir)) {
    std::fprintf(stderr,
                 "[test_snap_file] SKIP: LAMMPS submodule not initialized at "
                 "%s — run `git submodule update --init "
                 "verify/third_party/lammps` locally pre-push.\n",
                 snap_dir.string().c_str());
    std::fflush(stderr);
    std::exit(kExitSkip);
  }
}

fs::path write_tmp(const std::string& contents, const std::string& name) {
  const auto path = fs::temp_directory_path() / ("tdmd_snap_test_" + name);
  std::ofstream out(path);
  REQUIRE(out);
  out << contents;
  return path;
}

}  // namespace

TEST_CASE("snap_k_max: matches LAMMPS SNA::compute_ncoeff for low twojmax",
          "[potentials][snap][file][t8.4a]") {
  // Values reproduced от running the LAMMPS SNA::compute_ncoeff formula
  // directly (same numbers appear в Thompson 2015 Table I / W. Wood fixture).
  // The twojmax=8 entry is cross-checked against the canonical W_2940_2017_2
  // .snapcoeff header (which declares 56 = k_max+1 coefficients).
  REQUIRE(snap_k_max(0) == 1);
  REQUIRE(snap_k_max(2) == 5);
  REQUIRE(snap_k_max(4) == 14);
  REQUIRE(snap_k_max(6) == 30);
  REQUIRE(snap_k_max(8) == 55);
  REQUIRE(snap_k_max(10) == 91);
  REQUIRE(snap_k_max(12) == 140);
}

TEST_CASE("parse_snap_param: canonical W_2940_2017_2 fixture", "[potentials][snap][file][t8.4a]") {
  const fs::path snap_dir = lammps_snap_examples_dir();
  skip_if_submodule_uninitialized(snap_dir);

  const auto params = parse_snap_param((snap_dir / "W_2940_2017_2.snapparam").string());

  REQUIRE(params.twojmax == 8);
  REQUIRE(params.rcutfac == 4.73442);
  REQUIRE(params.rfac0 == 0.99363);
  REQUIRE(params.rmin0 == 0.0);
  REQUIRE(params.bzeroflag == false);
  REQUIRE(params.quadraticflag == false);
  // Defaults preserved for keys absent from the fixture.
  REQUIRE(params.switchflag == true);
  REQUIRE(params.chemflag == false);
  REQUIRE(params.bnormflag == false);
  REQUIRE(params.wselfallflag == false);
  REQUIRE(params.switchinnerflag == false);
}

TEST_CASE("parse_snap_coeff: canonical W_2940_2017_2 fixture", "[potentials][snap][file][t8.4a]") {
  const fs::path snap_dir = lammps_snap_examples_dir();
  skip_if_submodule_uninitialized(snap_dir);

  const auto species = parse_snap_coeff((snap_dir / "W_2940_2017_2.snapcoeff").string());

  REQUIRE(species.size() == 1);
  REQUIRE(species[0].name == "W");
  REQUIRE(species[0].radius_elem == 0.5);
  REQUIRE(species[0].weight_elem == 1.0);
  // Header declares 56 coefficients = 1 (β_0) + 55 (k_max для twojmax=8).
  REQUIRE(species[0].beta.size() == 56);
  // First + last β reproduced bitwise from the upstream file (ASCII FP64
  // round-trip: these are decimal literals stored verbatim).
  REQUIRE(species[0].beta.front() == 0.781170857801);
  REQUIRE(species[0].beta.back() == -0.008314173699);
}

TEST_CASE("parse_snap_files: combined W fixture cross-checks k_max",
          "[potentials][snap][file][t8.4a]") {
  const fs::path snap_dir = lammps_snap_examples_dir();
  skip_if_submodule_uninitialized(snap_dir);

  const auto data = parse_snap_files((snap_dir / "W_2940_2017_2.snapcoeff").string(),
                                     (snap_dir / "W_2940_2017_2.snapparam").string());

  REQUIRE(data.k_max == 55);
  REQUIRE(data.species.size() == 1);
  REQUIRE(data.species[0].beta.size() == static_cast<std::size_t>(data.k_max) + 1);

  // rcut_sq_ab is a 1×1 matrix: rcutfac·(R_W + R_W) = 4.73442·(0.5 + 0.5) = 4.73442.
  REQUIRE(data.rcut_sq_ab.size() == 1);
  REQUIRE(data.rcut_sq_ab[0] == (4.73442 * 4.73442));
  REQUIRE(data.max_pairwise_cutoff() == 4.73442);

  // Checksum is stable — re-parse и compare.
  const auto data2 = parse_snap_files((snap_dir / "W_2940_2017_2.snapcoeff").string(),
                                      (snap_dir / "W_2940_2017_2.snapparam").string());
  REQUIRE(data2.checksum == data.checksum);
  REQUIRE(data.checksum != 0);
}

TEST_CASE("parse_snap_files: rejects coefficient-count mismatch",
          "[potentials][snap][file][t8.4a]") {
  // Param file declares twojmax=8 (k_max=55 → expect 56 coefs) but coefficient
  // file declares 10 coefficients — parse_snap_files must fail с a diagnostic
  // mentioning twojmax / k_max / expected vs got.
  const auto coeff_path = write_tmp(
      "1 10\n"
      "W 0.5 1\n"
      "0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n",
      "bad_ncoeff.snapcoeff");
  const auto param_path = write_tmp(
      "rcutfac 4.73442\n"
      "twojmax 8\n",
      "bad_ncoeff.snapparam");

  REQUIRE_THROWS_WITH(parse_snap_files(coeff_path.string(), param_path.string()),
                      ContainsSubstring("β coefficients but 56 expected"));
}

TEST_CASE("parse_snap_param: missing required key emits path-tagged diagnostic",
          "[potentials][snap][file][t8.4a]") {
  // rcutfac is required; drop it.
  const auto path = write_tmp("twojmax 8\n", "missing_rcutfac.snapparam");
  REQUIRE_THROWS_WITH(parse_snap_param(path.string()),
                      ContainsSubstring("missing required key 'rcutfac'"));
}

TEST_CASE("parse_snap_param: rejects odd twojmax с line-tagged diagnostic",
          "[potentials][snap][file][t8.4a]") {
  const auto path = write_tmp(
      "rcutfac 4.0\n"
      "twojmax 3\n",
      "odd_twojmax.snapparam");
  REQUIRE_THROWS_WITH(parse_snap_param(path.string()),
                      ContainsSubstring("twojmax must be non-negative and even"));
}

TEST_CASE("parse_snap_param: rejects chemflag=1 с a forward-looking message",
          "[potentials][snap][file][t8.4a]") {
  const auto path = write_tmp(
      "rcutfac 4.0\n"
      "twojmax 4\n"
      "chemflag 1\n",
      "chemflag_on.snapparam");
  REQUIRE_THROWS_WITH(parse_snap_param(path.string()), ContainsSubstring("chemflag=1"));
}

TEST_CASE("parse_snap_coeff: rejects invalid header", "[potentials][snap][file][t8.4a]") {
  const auto path = write_tmp("0 56\n", "zero_species.snapcoeff");
  REQUIRE_THROWS_WITH(parse_snap_coeff(path.string()),
                      ContainsSubstring("n_species must be positive"));
}

TEST_CASE("SnapPotential: constructor accepts parsed W fixture",
          "[potentials][snap][skeleton][t8.4a]") {
  const fs::path snap_dir = lammps_snap_examples_dir();
  skip_if_submodule_uninitialized(snap_dir);

  const auto data = parse_snap_files((snap_dir / "W_2940_2017_2.snapcoeff").string(),
                                     (snap_dir / "W_2940_2017_2.snapparam").string());

  tdmd::SnapPotential pot(data);
  REQUIRE(pot.name() == "snap");
  REQUIRE(pot.cutoff() == 4.73442);
  REQUIRE(pot.data().k_max == 55);
  REQUIRE(pot.effective_skin() == 0.05 * 4.73442);
}

// Note: the T8.4a "compute throws until T8.4b" skeleton test was retired when
// the force body landed в T8.4b (see tests/potentials/test_snap_compute.cpp для
// the real compute-path coverage).

TEST_CASE("SnapPotential: constructor rejects inconsistent β count",
          "[potentials][snap][skeleton][t8.4a]") {
  SnapData data;
  data.params.twojmax = 8;
  data.params.rcutfac = 4.73442;
  data.k_max = snap_k_max(8);
  SnapSpecies sp;
  sp.name = "W";
  sp.radius_elem = 0.5;
  sp.weight_elem = 1.0;
  sp.beta.assign(10, 0.0);  // wrong: expected k_max + 1 = 56
  data.species.push_back(std::move(sp));

  REQUIRE_THROWS_AS(tdmd::SnapPotential(data), std::invalid_argument);
}

TEST_CASE("SnapPotential: constructor rejects odd twojmax", "[potentials][snap][skeleton][t8.4a]") {
  SnapData data;
  data.params.twojmax = 3;  // odd
  data.params.rcutfac = 4.0;
  data.k_max = snap_k_max(3);
  SnapSpecies sp;
  sp.name = "W";
  sp.radius_elem = 0.5;
  sp.weight_elem = 1.0;
  sp.beta.assign(static_cast<std::size_t>(data.k_max) + 1, 0.0);
  data.species.push_back(std::move(sp));
  REQUIRE_THROWS_AS(tdmd::SnapPotential(data), std::invalid_argument);
}
