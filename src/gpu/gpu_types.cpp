// SPEC: docs/specs/gpu/SPEC.md §2 (RAII types, PIMPL compile firewall)
// Exec pack: docs/development/m6_execution_pack.md T6.2
//
// T6.2 ships stub Impl definitions for DeviceStream and DeviceEvent — the
// structs are empty and hold no CUDA handles. T6.3 replaces this translation
// unit with CUDA-bound definitions (wrapping cudaStream_t / cudaEvent_t)
// living in a .cu file. Until that lands, the skeleton supports only
// null-construction and move semantics, which is sufficient for compile-time
// shape tests and the build-system smoke in tests/gpu/test_gpu_types.cpp.

#include "tdmd/gpu/types.hpp"

namespace tdmd::gpu {

struct DeviceStream::Impl {};
struct DeviceEvent::Impl {};

DeviceStream::DeviceStream() = default;
DeviceStream::DeviceStream(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
DeviceStream::~DeviceStream() = default;
DeviceStream::DeviceStream(DeviceStream&&) noexcept = default;
DeviceStream& DeviceStream::operator=(DeviceStream&&) noexcept = default;

DeviceEvent::DeviceEvent() = default;
DeviceEvent::DeviceEvent(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
DeviceEvent::~DeviceEvent() = default;
DeviceEvent::DeviceEvent(DeviceEvent&&) noexcept = default;
DeviceEvent& DeviceEvent::operator=(DeviceEvent&&) noexcept = default;

}  // namespace tdmd::gpu
