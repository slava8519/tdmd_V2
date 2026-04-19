#pragma once

// SPEC: docs/specs/gpu/SPEC.md §12 (telemetry) + D-M6-14 (NVTX instrumentation)
// Exec pack: docs/development/m6_execution_pack.md T6.11
//
// TDMD_NVTX_RANGE(name) — RAII scoped range macro for Nsight Systems traces.
//
// Usage:
//   void my_gpu_op() {
//     TDMD_NVTX_RANGE("eam_density_kernel");
//     my_kernel<<<grid, block, 0, stream>>>(...);
//   }
//
// Semantics:
//   * When TDMD_BUILD_CUDA=ON the macro expands to an `nvtx3::scoped_range`
//     instance bound to the enclosing scope. Nsight Systems picks the range
//     up through libnvToolsExt.so at runtime with zero host-side setup.
//   * When TDMD_BUILD_CUDA=OFF the macro expands to `((void)0)` — a void
//     discard statement with no runtime or codegen footprint. CPU-only TUs
//     may freely sprinkle TDMD_NVTX_RANGE without paying any cost.
//
// Overhead target: D-M6-14 requires <1% steady-state overhead. The NVTX
// library fast-paths to a no-op when no profiler attaches, so the cost is
// essentially the two std::atomic loads inside nvtxDomainCreate / Destroy
// per scope — well under the budget at typical 100-kernel-per-step cadence.
//
// String literal convention: keep names **stable** so dashboards can match
// across runs. Pattern: "{subsystem}.{op}" — e.g. "nl.count_kernel",
// "eam.density_kernel", "vv.pre_force_kernel", "gpu.h2d.positions",
// "gpu.d2h.forces".
//
// The macro deliberately lives in telemetry/ (not gpu/) so non-GPU TUs such
// as comm/ can wrap MPI pack/unpack with the same surface if T6.11b decides
// to trace them. Including this header never pulls CUDA runtime headers
// outside TDMD_BUILD_CUDA sentinels, so PIMPL firewall contracts in
// gpu/SPEC §2.1 and D-M6-17 remain intact.

// TDMD_BUILD_CUDA is always defined by the build system (0 or 1) — consistent
// with gpu/*.cu conventions. Use `#if` not `#ifdef` so CPU-only builds that
// set TDMD_BUILD_CUDA=0 take the no-op path.
#if TDMD_BUILD_CUDA
#include <nvtx3/nvtx3.hpp>
#endif

// Token-pasting helpers — required so `__LINE__` expands before being
// concatenated with the variable prefix. Without the two-level indirection
// the identifier literally becomes `_tdmd_nvtx_scope___LINE__`.
#define TDMD_NVTX_CONCAT_INNER(a, b) a##b
#define TDMD_NVTX_CONCAT(a, b) TDMD_NVTX_CONCAT_INNER(a, b)

#if TDMD_BUILD_CUDA

// RAII range: begins on ctor, ends on dtor. Unique per expansion via
// `__LINE__` so multiple TDMD_NVTX_RANGE calls in the same enclosing scope
// do not collide. The scoped_range targets the default NVTX domain — this
// is what Nsight Systems displays under the process row.
#define TDMD_NVTX_RANGE(name)                                           \
  ::nvtx3::scoped_range TDMD_NVTX_CONCAT(_tdmd_nvtx_scope_, __LINE__) { \
    name                                                                \
  }

#else

// CPU-only build: zero-cost no-op. The `do { } while(0)` form would force a
// statement context at each call site; a plain `((void)0)` expression-style
// no-op lets the macro appear anywhere a statement or expression is valid
// without surprising the grammar.
#define TDMD_NVTX_RANGE(name) ((void) 0)

#endif  // TDMD_BUILD_CUDA
