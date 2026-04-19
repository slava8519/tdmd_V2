#pragma once

// SPEC: docs/specs/gpu/SPEC.md §2 (PIMPL firewall)
// Exec pack: docs/development/m6_execution_pack.md T6.3, D-M6-17
//
// Internal header — NOT exposed via the public include path. Holds the
// CUDA-bound PIMPL body definitions for DeviceStream::Impl and
// DeviceEvent::Impl, and tiny free-function helpers shared by the gpu/
// implementation TUs (gpu_types.cpp, factories.cpp, device_pool.cpp).
//
// Only included when TDMD_BUILD_CUDA=1. Public headers (types.hpp, ...)
// must remain CUDA-free per D-M6-17.

#if !TDMD_BUILD_CUDA
#error "cuda_handles.hpp may only be included when TDMD_BUILD_CUDA=1"
#endif

#include "tdmd/gpu/types.hpp"

#include <cuda_runtime.h>

namespace tdmd::gpu {

struct DeviceStream::Impl {
  cudaStream_t stream = nullptr;
};

struct DeviceEvent::Impl {
  cudaEvent_t event = nullptr;
};

// Helper: extract the raw cudaStream_t from a DeviceStream, or nullptr
// (default / legacy stream) if the handle is invalid.
inline cudaStream_t raw_stream(const DeviceStream& s) noexcept {
  if (!s.valid()) {
    return nullptr;
  }
  return s.impl()->stream;
}

inline cudaEvent_t raw_event(const DeviceEvent& e) noexcept {
  if (!e.valid()) {
    return nullptr;
  }
  return e.impl()->event;
}

}  // namespace tdmd::gpu
