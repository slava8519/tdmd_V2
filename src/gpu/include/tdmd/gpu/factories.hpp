#pragma once

// SPEC: docs/specs/gpu/SPEC.md §2 (types), §4 (device probe + enumeration)
// Master spec: §14 M6, §15.2
// Exec pack: docs/development/m6_execution_pack.md T6.3
//
// Runtime factory functions that produce real CUDA handles. These are
// defined in factories.cpp when TDMD_BUILD_CUDA=1; on CPU-only builds
// they return empty / default-constructed values (valid()==false) so the
// dependent code path fails cleanly at the next real GPU op rather than
// at link time.

#include "tdmd/gpu/types.hpp"

#include <stdexcept>
#include <vector>

namespace tdmd::gpu {

// Enumerate visible CUDA devices. Returns empty vector on CPU-only builds
// or when cudaGetDeviceCount returns 0.
std::vector<DeviceInfo> probe_devices();

// Fetch DeviceInfo for a specific device ordinal.
// Throws std::runtime_error if id is out of range or CPU-only build.
DeviceInfo select_device(DeviceId id);

// Create a new CUDA stream bound to device_id. Stream is created with
// cudaStreamNonBlocking flag so it does not serialize against the legacy
// NULL stream — see gpu/SPEC §3.1.
// Throws std::runtime_error on failure or CPU-only build.
DeviceStream make_stream(DeviceId device_id);

// Create a new CUDA event bound to device_id. Event is created with
// cudaEventDisableTiming flag by default (pure sync barrier, not a timer).
// Throws std::runtime_error on failure or CPU-only build.
DeviceEvent make_event(DeviceId device_id);

}  // namespace tdmd::gpu
