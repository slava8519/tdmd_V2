// SPEC: docs/specs/comm/SPEC.md §6.2 (GpuAwareMpiBackend probe)
// Master spec: §12.6 (comm interfaces)
// Exec pack: docs/development/m7_execution_pack.md T7.3

#include "tdmd/comm/cuda_mpi_probe.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>

#if defined(TDMD_ENABLE_MPI) && TDMD_ENABLE_MPI
#include <mpi.h>
// MPIX extensions live in mpi-ext.h on OpenMPI; not all MPI implementations
// ship it. Probe via __has_include so MVAPICH / Cray / Intel MPI builds
// still compile cleanly — the env-var fallback covers them.
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#define TDMD_HAS_MPIX_CUDA_SUPPORT 1
#else
#define TDMD_HAS_MPIX_CUDA_SUPPORT 0
#endif
#endif

namespace tdmd::comm {

namespace {

// Atomic so the cache is safe against concurrent first calls — even if two
// threads race the probe both will agree on the answer (MPI_Init is the
// real synchronization point; we just avoid double-querying the symbol).
std::atomic<int> cached_result_{-1};  // -1 unset, 0 false, 1 true

#if defined(TDMD_ENABLE_MPI) && TDMD_ENABLE_MPI

bool env_truthy(const char* name) noexcept {
  const char* v = std::getenv(name);
  if (v == nullptr) {
    return false;
  }
  // Accept "1", "true", "yes" (case-sensitive lower) — the OpenMPI
  // documentation lists "true" as the canonical value but field
  // experience says people set "1" too.
  return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 || std::strcmp(v, "yes") == 0;
}

#endif  // TDMD_ENABLE_MPI

bool probe_uncached() noexcept {
#if defined(TDMD_ENABLE_MPI) && TDMD_ENABLE_MPI
#if TDMD_HAS_MPIX_CUDA_SUPPORT && defined(MPIX_CUDA_AWARE_SUPPORT)
  // Compile-time macro tells us the build flag; runtime symbol confirms.
  // Only trust runtime — the build may have been linked against a stub
  // libmpi without CUDA symbols at runtime.
  if (MPIX_Query_cuda_support() == 1) {
    return true;
  }
#endif
  // Manual override / non-OpenMPI implementations.
  if (env_truthy("OMPI_MCA_opal_cuda_support")) {
    return true;
  }
  if (env_truthy("MV2_USE_CUDA")) {
    return true;
  }
#endif
  return false;
}

}  // namespace

bool is_cuda_aware_mpi() noexcept {
  int cached = cached_result_.load(std::memory_order_acquire);
  if (cached >= 0) {
    return cached == 1;
  }
  const bool result = probe_uncached();
  cached_result_.store(result ? 1 : 0, std::memory_order_release);
  return result;
}

void reset_cuda_mpi_probe_cache_for_testing() noexcept {
  cached_result_.store(-1, std::memory_order_release);
}

}  // namespace tdmd::comm
