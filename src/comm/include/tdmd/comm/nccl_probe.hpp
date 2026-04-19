// SPEC: docs/specs/comm/SPEC.md §6.3 (NcclBackend probe)
// Master spec: §12.6 (comm interfaces)
// Exec pack: docs/development/m7_execution_pack.md T7.4
//
// Runtime NCCL availability probe. Mirrors cuda_mpi_probe — caches a boolean,
// never aborts, env-overridable for testing. Used by NcclBackend to refuse
// construction cleanly when NCCL isn't linked, so the engine preflight (T7.9)
// can fall back to MpiHostStaging without a crash.

#pragma once

#include <cstdint>

namespace tdmd::comm {

// True iff NCCL is linked, usable, and reports a version via ncclGetVersion().
// The probe is cached on first call. Never throws. Returns false on:
//   - build without TDMD_ENABLE_NCCL (NCCL headers / library not found at
//     configure time),
//   - ncclGetVersion() returning a non-success status at runtime.
[[nodiscard]] bool is_nccl_available() noexcept;

// NCCL runtime version packed per the NCCL convention: MAJOR*1000 + MINOR*100 +
// PATCH. Returns 0 when the probe is negative. Used by the backend to emit the
// "NCCL < 2.18" warning the exec pack requires (D-M7-4).
[[nodiscard]] int nccl_runtime_version() noexcept;

// Reset the cache so tests can drive the probe path deterministically.
void reset_nccl_probe_cache_for_testing() noexcept;

}  // namespace tdmd::comm
