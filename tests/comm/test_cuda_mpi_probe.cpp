// SPEC: docs/specs/comm/SPEC.md §6.2
// Master spec: §12.6
// Exec pack: docs/development/m7_execution_pack.md T7.3
//
// Self-test for the CUDA-aware MPI runtime probe. The probe is required
// to NEVER abort and ALWAYS produce a clean boolean — calling it on a
// machine without CUDA-aware MPI must simply return false. Calling it
// repeatedly must produce the same answer (cached).

#include "tdmd/comm/cuda_mpi_probe.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>

namespace tc = tdmd::comm;

TEST_CASE("cuda_mpi_probe — never throws", "[comm][cuda_probe]") {
  REQUIRE_NOTHROW(tc::is_cuda_aware_mpi());
}

TEST_CASE("cuda_mpi_probe — repeat calls are stable (cached)", "[comm][cuda_probe]") {
  const bool a = tc::is_cuda_aware_mpi();
  const bool b = tc::is_cuda_aware_mpi();
  const bool c = tc::is_cuda_aware_mpi();
  REQUIRE(a == b);
  REQUIRE(b == c);
}

#if defined(TDMD_ENABLE_MPI) && TDMD_ENABLE_MPI
TEST_CASE("cuda_mpi_probe — env override flips the answer", "[comm][cuda_probe]") {
  // Reset the cache, set the OpenMPI env override, and verify the probe
  // honors it. The override is the canonical "manual yes" path on systems
  // where the symbol isn't surfaced. Only meaningful in MPI-enabled builds:
  // the no-MPI probe unconditionally returns false by construction.
  tc::reset_cuda_mpi_probe_cache_for_testing();

  // Pretend probe was originally false (or not-yet-evaluated); set the
  // env var and check we get a true. NB: the MPIX_Query path takes
  // priority over env vars, so on a machine where MPIX returns true
  // this assertion would still pass — that's fine, we're verifying the
  // env override doesn't break anything either way.
  setenv("OMPI_MCA_opal_cuda_support", "true", /*overwrite=*/1);
  const bool with_env = tc::is_cuda_aware_mpi();
  REQUIRE(with_env);

  // Reset and clear env: probe should fall back to its detection logic
  // (which on a non-CUDA-aware OpenMPI returns false).
  unsetenv("OMPI_MCA_opal_cuda_support");
  unsetenv("MV2_USE_CUDA");
  tc::reset_cuda_mpi_probe_cache_for_testing();
  // No assertion on the result here — depends on whether the underlying
  // MPI build is CUDA-aware. Just that the call completes cleanly.
  REQUIRE_NOTHROW(tc::is_cuda_aware_mpi());
}
#else
TEST_CASE("cuda_mpi_probe — no-MPI build returns false unconditionally", "[comm][cuda_probe]") {
  // When TDMD_ENABLE_MPI=OFF, the probe is a compile-time constant false.
  // Verify that even with env overrides set, the answer remains false.
  tc::reset_cuda_mpi_probe_cache_for_testing();
  setenv("OMPI_MCA_opal_cuda_support", "true", /*overwrite=*/1);
  REQUIRE_FALSE(tc::is_cuda_aware_mpi());
  unsetenv("OMPI_MCA_opal_cuda_support");
}
#endif
