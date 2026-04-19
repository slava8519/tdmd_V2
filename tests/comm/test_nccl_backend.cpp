// SPEC: docs/specs/comm/SPEC.md §6.3, §7
// Master spec: §10, §12.6, §D.14
// Exec pack: docs/development/m7_execution_pack.md T7.4
//
// 2-rank NCCL backend test. Runtime SKIPs cleanly if NCCL isn't linked
// or no CUDA device is visible — same posture as test_gpu_aware_mpi_backend
// (Option A CI policy: the gate stays green on every CI node).
//
// Acceptance criteria (exec pack T7.4):
//   - NCCL allreduce bit-exact vs MpiHostStaging on the M5 smoke fixture.
//   - Construction throws cleanly when NCCL / CUDA are unavailable.
//
// On a machine with NCCL + ≥1 GPU visible, 2 ranks share a single GPU via
// round-robin cudaSetDevice — MPS-compatible and matches T6.7's 2-rank-on-
// 1-GPU pattern.

#include "tdmd/comm/comm_config.hpp"
#include "tdmd/comm/deterministic_reduction.hpp"
#include "tdmd/comm/nccl_backend.hpp"
#include "tdmd/comm/nccl_probe.hpp"

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <stdexcept>

#include <mpi.h>

namespace tc = tdmd::comm;

namespace {

std::unique_ptr<tc::NcclBackend> try_make_backend() {
  try {
    return std::make_unique<tc::NcclBackend>();
  } catch (const std::runtime_error&) {
    return nullptr;
  }
}

}  // namespace

TEST_CASE("NcclBackend — constructor refuses when NCCL absent", "[comm][nccl][probe_gate]") {
  const bool available = tc::is_nccl_available();
  if (available) {
    REQUIRE_NOTHROW(tc::NcclBackend{});
  } else {
    REQUIRE_THROWS_AS(tc::NcclBackend{}, std::runtime_error);
  }
}

TEST_CASE("NcclBackend — 2-rank allreduce bit-exact vs host Kahan", "[comm][nccl][2rank]") {
  int world_size = 0;
  int my_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  REQUIRE(world_size == 2);

  auto backend = try_make_backend();
  if (!backend) {
    SUCCEED("NCCL / CUDA unavailable; skipping NCCL 2-rank allreduce");
    return;
  }

  tc::CommConfig config;
  config.use_deterministic_reductions = true;

  // initialize() may still fail if CUDA devices aren't visible even though
  // the probe returned positive (e.g. nvidia-smi claims a GPU but the
  // process is sandboxed). Treat that the same as the probe being negative.
  try {
    backend->initialize(config);
  } catch (const std::runtime_error&) {
    SUCCEED("CUDA device not available to NCCL at initialize(); skipping");
    return;
  }
  REQUIRE(backend->nranks() == 2);

  // rank 0 contributes 1.5, rank 1 contributes 2.25 — non-trivial doubles
  // so bit-exact equality isn't coincidental.
  const double local = (backend->rank() == 0) ? 1.5 : 2.25;
  const double got = backend->global_sum_double(local);

  // Expected: rank-ordered Kahan fold of {1.5, 2.25}. Any deviation from the
  // host Kahan path means NCCL changed the semantics of the reduction —
  // which is the D-M5-9 / D-M5-12 invariant we refuse to break.
  const std::vector<double> rank_ordered{1.5, 2.25};
  const double expected = tc::kahan_sum_ordered(rank_ordered);
  REQUIRE(got == expected);

  // Max is associative — NCCL handles it directly.
  const double mx = backend->global_max_double(local);
  REQUIRE(mx == 2.25);

  backend->barrier();
  backend->shutdown();
}
