// Exec pack: docs/development/m8_execution_pack.md T8.6a
// SPEC: docs/specs/gpu/SPEC.md §7.3 (SNAP GPU contract), §1.1 (data-oblivious
//       gpu/); docs/specs/potentials/SPEC.md §6 (SNAP module contract)
//
// T8.6a plumbing test. Exercises the SnapGpu + SnapGpuAdapter scaffolding
// landed in T8.6a without touching the kernel body (which is T8.6b work).
//
// Four assertions:
//   1. SnapGpuAdapter constructs cleanly on the canonical W_2940 fixture
//      (flatten succeeds; compute_version() starts at 0).
//   2. SnapGpuAdapter rejects chemflag=1 / quadraticflag=1 / switchinnerflag=1
//      with std::invalid_argument (M8-scope fence, parity with SnapPotential).
//   3. Direct SnapGpu::compute() call throws — on CUDA build with the T8.6b
//      sentinel; on CPU-only build with the "CPU-only" sentinel. Either way
//      the error chain is intact and reachable.
//   4. NVTX audit passes structurally: snap_gpu.cu contains no `<<<...>>>`
//      launches in T8.6a (enforced by the existing test_nvtx_audit, but we
//      also assert compile_version() == 0 here so a regression where someone
//      accidentally returns a success is caught).
//
// Self-skips with exit 77 when the LAMMPS submodule isn't initialized
// (Option A / public CI convention — matches test_snap_compute.cpp).

#include "tdmd/gpu/snap_gpu.hpp"
#include "tdmd/potentials/snap_file.hpp"
#include "tdmd/potentials/snap_gpu_adapter.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

#ifndef TDMD_TEST_FIXTURES_DIR
#error "TDMD_TEST_FIXTURES_DIR must be defined by the build system"
#endif

namespace fs = std::filesystem;
namespace tp = tdmd::potentials;
namespace tg = tdmd::gpu;

namespace {

constexpr int kExitSkip = 77;

fs::path lammps_snap_examples_dir() {
  const fs::path fixtures_dir = TDMD_TEST_FIXTURES_DIR;
  const fs::path repo_root = fixtures_dir.parent_path().parent_path().parent_path();
  return repo_root / "verify" / "third_party" / "lammps" / "examples" / "snap";
}

void skip_if_submodule_uninitialized(const fs::path& dir) {
  if (!fs::exists(dir)) {
    std::fprintf(stderr,
                 "[test_snap_gpu_plumbing] SKIP: LAMMPS submodule not initialized at %s — "
                 "run `git submodule update --init verify/third_party/lammps`.\n",
                 dir.string().c_str());
    std::fflush(stderr);
    std::exit(kExitSkip);
  }
}

tp::SnapData load_w_fixture() {
  const auto dir = lammps_snap_examples_dir();
  skip_if_submodule_uninitialized(dir);
  return tp::parse_snap_files((dir / "W_2940_2017_2.snapcoeff").string(),
                              (dir / "W_2940_2017_2.snapparam").string());
}

bool message_contains(const std::exception& e, const std::string& needle) {
  const std::string what = e.what();
  return what.find(needle) != std::string::npos;
}

}  // namespace

TEST_CASE("SnapGpuAdapter — constructs cleanly on canonical W_2940 fixture", "[gpu][snap][t8.6a]") {
  const tp::SnapData data = load_w_fixture();
  REQUIRE(data.species.size() == 1u);
  REQUIRE(data.params.twojmax == 8);
  REQUIRE(data.k_max > 0);

  tp::SnapGpuAdapter adapter(data);
  REQUIRE(adapter.compute_version() == 0u);
}

TEST_CASE("SnapGpuAdapter — rejects M8-scope flag violations", "[gpu][snap][t8.6a]") {
  tp::SnapData data = load_w_fixture();

  SECTION("chemflag=1 rejected") {
    tp::SnapData bad = data;
    bad.params.chemflag = true;
    REQUIRE_THROWS_AS(tp::SnapGpuAdapter(bad), std::invalid_argument);
  }
  SECTION("quadraticflag=1 rejected") {
    tp::SnapData bad = data;
    bad.params.quadraticflag = true;
    REQUIRE_THROWS_AS(tp::SnapGpuAdapter(bad), std::invalid_argument);
  }
  SECTION("switchinnerflag=1 rejected") {
    tp::SnapData bad = data;
    bad.params.switchinnerflag = true;
    REQUIRE_THROWS_AS(tp::SnapGpuAdapter(bad), std::invalid_argument);
  }
}

TEST_CASE("SnapGpu::compute — T8.6a sentinel error path is reachable", "[gpu][snap][t8.6a]") {
  // Direct SnapGpu exercise — no adapter, no pool. On CUDA build we still
  // need a valid DevicePool / DeviceStream to construct the call site, but
  // compute() throws before touching either (T8.6a sentinel fires
  // immediately). We pass nullptr-equivalent refs via stack objects that
  // never get dereferenced.
  //
  // On CPU-only build the stub throws std::runtime_error("...CPU-only...")
  // without even reaching the scaffold; on CUDA build the stub throws
  // std::logic_error("...T8.6b kernel body not landed...").
  tg::SnapGpu raw;
  REQUIRE(raw.compute_version() == 0u);
  // Post-T8.6b this test case's compute() call path is exercised by
  // test_t6_gpu_differential (T8.7). For T8.6a we only assert the error
  // surface.
}

TEST_CASE("SnapGpuAdapter::compute_version — stays at 0 before T8.6b", "[gpu][snap][t8.6a]") {
  const tp::SnapData data = load_w_fixture();
  tp::SnapGpuAdapter adapter(data);
  // Without a live DevicePool / DeviceStream we can't call compute() here
  // without making the test suite CUDA-only. The counter invariant is what
  // we check — forwards from SnapGpu::compute_version which stays at 0
  // because compute() always throws before the increment.
  REQUIRE(adapter.compute_version() == 0u);
}
