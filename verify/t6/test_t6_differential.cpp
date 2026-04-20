// T6 differential Catch2 wrapper.
//
// Thin shell around verify/t6/run_differential.py — mirrors T1/T4 so ctest
// surfaces the D-M8-7 byte-exact gate alongside unit tests without pulling
// YAML-parsing comparators into the C++ build graph.
//
// T6 is the **M8 SNAP proof-of-value gate on CPU FP64** (master spec §14 M8,
// verify/SPEC §4.7 and docs/development/m8_execution_pack.md T8.5). Missing
// LAMMPS → SKIP (Option A: no submodule build on public CI); any other
// non-zero exit → FAIL. Exit-code mapping mirrors T1/T4.

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
constexpr int kExitError = 2;
constexpr int kExitSkip = 77;

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
  return (status & 0xff00) >> 8;
}

}  // namespace

TEST_CASE("T6 differential: SNAP tungsten NVE 250 atoms 100 steps vs LAMMPS",
          "[verify][differential][t6]") {
  namespace fs = std::filesystem;
  const fs::path repo_root = TDMD_REPO_ROOT;
  const fs::path harness = repo_root / "verify" / "t6" / "run_differential.py";
  const fs::path benchmark_dir = repo_root / "verify" / "benchmarks" / "t6_snap_tungsten";
  const fs::path thresholds = repo_root / "verify" / "thresholds" / "thresholds.yaml";
  const fs::path lammps_bin =
      repo_root / "verify" / "third_party" / "lammps" / "install_tdmd" / "bin" / "lmp";
  const fs::path lammps_libdir =
      repo_root / "verify" / "third_party" / "lammps" / "install_tdmd" / "lib";

  if (!fs::exists(lammps_bin)) {
    std::fprintf(stderr,
                 "[test_t6_differential] SKIP: LAMMPS oracle missing at %s — run "
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
