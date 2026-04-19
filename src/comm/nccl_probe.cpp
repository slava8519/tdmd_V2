// SPEC: docs/specs/comm/SPEC.md §6.3 (NcclBackend probe)
// Master spec: §12.6
// Exec pack: docs/development/m7_execution_pack.md T7.4

#include "tdmd/comm/nccl_probe.hpp"

#include <atomic>

#if defined(TDMD_ENABLE_NCCL) && TDMD_ENABLE_NCCL
#include <nccl.h>
#endif

namespace tdmd::comm {

namespace {

// -1 unset, 0 false, otherwise a positive NCCL runtime version. Atomic so
// concurrent first-calls converge on the same answer without needing a lock.
std::atomic<int> cached_version_{-1};

int probe_uncached() noexcept {
#if defined(TDMD_ENABLE_NCCL) && TDMD_ENABLE_NCCL
  int version = 0;
  // ncclGetVersion is lightweight and doesn't require ncclCommInit, so it's
  // safe as a probe even before MPI_Init. On older NCCL it may return an
  // error code — treat anything non-ncclSuccess as "not available".
  const ncclResult_t rc = ncclGetVersion(&version);
  if (rc != ncclSuccess || version <= 0) {
    return 0;
  }
  return version;
#else
  return 0;
#endif
}

}  // namespace

bool is_nccl_available() noexcept {
  return nccl_runtime_version() > 0;
}

int nccl_runtime_version() noexcept {
  int cached = cached_version_.load(std::memory_order_acquire);
  if (cached >= 0) {
    return cached;
  }
  const int v = probe_uncached();
  cached_version_.store(v, std::memory_order_release);
  return v;
}

void reset_nccl_probe_cache_for_testing() noexcept {
  cached_version_.store(-1, std::memory_order_release);
}

}  // namespace tdmd::comm
