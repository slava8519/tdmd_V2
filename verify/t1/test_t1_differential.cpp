// T1 differential Catch2 wrapper.
//
// This test is a thin harness over `verify/t1/run_differential.py` (which
// does the heavy lifting — see docs/specs/verify/SPEC.md §7.1). Rationale:
// keeping the engine-vs-oracle comparison in Python keeps the build graph
// simple (no new C++ dependencies on YAML-parsing comparators) and lets the
// harness run standalone outside ctest during development.
//
// Exit-code translation:
//   0  → PASS (test body runs to completion).
//   1  → FAIL (thresholds violated): FAIL_CHECK with the captured stdout.
//   2  → harness setup error:        FAIL_CHECK with the captured stdout.
//  77  → LAMMPS unavailable:         SKIP (Catch2 translates to CTest "Skip").
//
// Required runtime injection (CMake):
//   - compile define TDMD_REPO_ROOT = absolute repo root.
//   - environment var TDMD_CLI_BIN  = absolute path to the `tdmd` binary
//     ($<TARGET_FILE:tdmd>, resolved at test invocation time).
//
// The LAMMPS binary is discovered via its canonical submodule-install path
// (see verify/SPEC.md §5). Absent binary ⇒ SKIP, not FAIL.

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>

#ifndef TDMD_REPO_ROOT
#error "TDMD_REPO_ROOT must be defined by the build system"
#endif

namespace {

constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitError = 2;
constexpr int kExitSkip = 77;

// Run `cmd` under a shell, capture combined stdout/stderr, return exit code.
// Uses popen — adequate for a test that shells out once per invocation.
int run_capture(const std::string& cmd, std::string& out) {
  std::array<char, 512> buffer{};
  // NOLINTNEXTLINE(cert-env33-c) — invoking python harness is the point of this test.
  FILE* pipe = popen(cmd.c_str(), "r");
  if (pipe == nullptr) {
    out = "popen failed";
    return kExitError;
  }
  while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    out.append(buffer.data());
  }
  const int status = pclose(pipe);
  if (status == -1) {
    return kExitError;
  }
  // Decode WEXITSTATUS — defined in <sys/wait.h>, but popen returns the same
  // encoding on POSIX systems: shift right by 8 to get the exit code.
  return (status & 0xff00) >> 8;
}

}  // namespace

TEST_CASE("T1 differential: Al FCC Morse NVE 500 atoms 100 steps vs LAMMPS",
          "[verify][differential][t1]") {
  namespace fs = std::filesystem;
  const fs::path repo_root = TDMD_REPO_ROOT;
  const fs::path harness = repo_root / "verify" / "t1" / "run_differential.py";
  const fs::path benchmark_dir = repo_root / "verify" / "benchmarks" / "t1_al_morse_500";
  const fs::path thresholds = repo_root / "verify" / "thresholds" / "thresholds.yaml";
  const fs::path lammps_bin =
      repo_root / "verify" / "third_party" / "lammps" / "install_tdmd" / "bin" / "lmp";
  const fs::path lammps_libdir =
      repo_root / "verify" / "third_party" / "lammps" / "install_tdmd" / "lib";

  // Runtime preconditions. Missing LAMMPS is a SKIP (Option A: public CI has
  // no LAMMPS; harness is run locally pre-push), not a FAIL.
  //
  // Exit directly with the SKIP_RETURN_CODE value rather than routing through
  // Catch2's SKIP() macro — Catch2 v3.5 returns a non-zero exit code when the
  // only test case is skipped (it treats "no test actually ran" as an error),
  // which then fails CTest's SKIP_RETURN_CODE match. A hard exit bypasses the
  // session's aggregate-return logic and hands CTest exactly the 77 it expects.
  if (!fs::exists(lammps_bin)) {
    std::fprintf(stderr,
                 "[test_t1_differential] SKIP: LAMMPS oracle missing at %s — run "
                 "tools/build_lammps.sh locally pre-push.\n",
                 lammps_bin.string().c_str());
    std::fflush(stderr);
    std::exit(kExitSkip);
  }

  const char* tdmd_bin = std::getenv("TDMD_CLI_BIN");
  REQUIRE(tdmd_bin != nullptr);
  REQUIRE(fs::exists(tdmd_bin));

  std::ostringstream cmd;
  cmd << "python3 " << harness                 //
      << " --benchmark " << benchmark_dir      //
      << " --tdmd " << tdmd_bin                //
      << " --lammps " << lammps_bin            //
      << " --lammps-libdir " << lammps_libdir  //
      << " --thresholds " << thresholds        //
      << " 2>&1";

  std::string captured;
  const int rc = run_capture(cmd.str(), captured);

  INFO("harness command: " << cmd.str());
  INFO("harness output:\n" << captured);

  if (rc == kExitSkip) {
    SKIP("Harness reported SKIP (see captured output in test log).");
  }
  REQUIRE(rc != kExitError);
  REQUIRE(rc == kExitPass);
}
