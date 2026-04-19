// SPEC: docs/specs/gpu/SPEC.md §5 (memory model, abstract allocator)
// Exec pack: docs/development/m6_execution_pack.md T6.2, D-M6-12
//
// Out-of-line virtual destructor for DeviceAllocator — keeps the vtable
// anchored in this translation unit and avoids -Wweak-vtables warnings.
// Concrete implementations (DevicePool, PinnedHostPool) arrive in T6.3.

#include "tdmd/gpu/device_allocator.hpp"

namespace tdmd::gpu {

DeviceAllocator::~DeviceAllocator() = default;

}  // namespace tdmd::gpu
