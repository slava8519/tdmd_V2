// SPEC: docs/specs/comm/SPEC.md §6.3
// Master spec: §12.6
// Exec pack: docs/development/m7_execution_pack.md T7.4
//
// Self-test for the NCCL runtime probe. Like cuda_mpi_probe, the NCCL probe
// must NEVER abort and must produce a stable boolean.

#include "tdmd/comm/nccl_probe.hpp"

#include <catch2/catch_test_macros.hpp>

namespace tc = tdmd::comm;

TEST_CASE("nccl_probe — never throws", "[comm][nccl_probe]") {
  REQUIRE_NOTHROW(tc::is_nccl_available());
  REQUIRE_NOTHROW(tc::nccl_runtime_version());
}

TEST_CASE("nccl_probe — repeat calls are stable (cached)", "[comm][nccl_probe]") {
  const bool a = tc::is_nccl_available();
  const bool b = tc::is_nccl_available();
  const bool c = tc::is_nccl_available();
  REQUIRE(a == b);
  REQUIRE(b == c);
}

TEST_CASE("nccl_probe — version is non-negative", "[comm][nccl_probe]") {
  // Either NCCL isn't linked (version == 0) or ncclGetVersion returned a
  // positive encoded value. Negative is never a valid answer.
  REQUIRE(tc::nccl_runtime_version() >= 0);
  if (tc::is_nccl_available()) {
    REQUIRE(tc::nccl_runtime_version() > 0);
  } else {
    REQUIRE(tc::nccl_runtime_version() == 0);
  }
}

TEST_CASE("nccl_probe — reset cache lets the probe re-run", "[comm][nccl_probe]") {
  const bool before = tc::is_nccl_available();
  tc::reset_nccl_probe_cache_for_testing();
  const bool after = tc::is_nccl_available();
  // The answer must match — the underlying runtime hasn't changed mid-test.
  REQUIRE(before == after);
}
