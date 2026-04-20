// SPEC: docs/specs/verify/SPEC.md §4.7 (T6 canonical fixture landed M8 T8.2).
// Exec pack: docs/development/m8_execution_pack.md T8.2 (verify LAMMPS SNAP
// subset + canonical W fixture choice).
//
// Minimal path-resolution gate для T6 canonical SNAP fixture. Asserts that
// the three fixture artefacts (`W_2940_2017_2.snap` include file, `.snapcoeff`
// coefficients, `.snapparam` hyperparameters) и the driver example
// (`in.snap.W.2940`) all resolve inside the M1-landed LAMMPS submodule. Runs
// на every CI lane — self-skips (exit 77) when the submodule is not initialized,
// which is the expected state on public runners per Option A (D-M6-6).
//
// This test does NOT run LAMMPS, does NOT load TDMD SnapPotential (that lands
// T8.4), and does NOT perform any numeric check. The full differential harness
// + thresholds lands at T8.5 (CPU) и T8.7 (GPU).

#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

#ifndef TDMD_TEST_FIXTURES_DIR
#error "TDMD_TEST_FIXTURES_DIR must be defined by the build system"
#endif

namespace fs = std::filesystem;

namespace {

constexpr int kExitSkip = 77;

// Resolve the LAMMPS submodule SNAP examples directory by walking up from
// the tests/potentials/fixtures/ path injected by CMake. The submodule lives
// at `verify/third_party/lammps/examples/snap/` relative to repo root.
fs::path lammps_snap_examples_dir() {
  const fs::path fixtures_dir = TDMD_TEST_FIXTURES_DIR;
  // tests/potentials/fixtures/ → tests/potentials/ → tests/ → repo root.
  const fs::path repo_root = fixtures_dir.parent_path().parent_path().parent_path();
  return repo_root / "verify" / "third_party" / "lammps" / "examples" / "snap";
}

void skip_if_submodule_uninitialized(const fs::path& snap_dir) {
  // Submodule init marker: the snap/ directory itself exists only after
  // `git submodule update --init`. On public CI (Option A) it will be absent.
  if (!fs::exists(snap_dir)) {
    std::fprintf(stderr,
                 "[test_lammps_oracle_snap_fixture] SKIP: LAMMPS submodule "
                 "not initialized at %s — run `git submodule update --init "
                 "verify/third_party/lammps` locally pre-push.\n",
                 snap_dir.string().c_str());
    std::fflush(stderr);
    std::exit(kExitSkip);
  }
}

}  // namespace

TEST_CASE("T6 canonical SNAP fixture: W_2940_2017_2 resolves inside LAMMPS submodule",
          "[potentials][snap][oracle][t8.2]") {
  const fs::path snap_dir = lammps_snap_examples_dir();
  skip_if_submodule_uninitialized(snap_dir);

  REQUIRE(fs::exists(snap_dir / "W_2940_2017_2.snap"));
  REQUIRE(fs::exists(snap_dir / "W_2940_2017_2.snapcoeff"));
  REQUIRE(fs::exists(snap_dir / "W_2940_2017_2.snapparam"));
  REQUIRE(fs::exists(snap_dir / "in.snap.W.2940"));
}

TEST_CASE("T6 canonical SNAP fixture: upstream reference log present",
          "[potentials][snap][oracle][t8.2]") {
  // The upstream 1-rank reference log is used for local sanity check — running
  // `in.snap.W.2940` against the M1-built LAMMPS should produce thermo output
  // byte-exactly matching this log to LAMMPS float precision. Not the TDMD
  // acceptance gate (D-M8-7/D-M8-8 handle that), but a useful confirmation
  // that the ML-SNAP subset compiled cleanly.
  const fs::path snap_dir = lammps_snap_examples_dir();
  skip_if_submodule_uninitialized(snap_dir);

  REQUIRE(fs::exists(snap_dir / "log.15Jun20.snap.W.2940.g++.1"));
}
