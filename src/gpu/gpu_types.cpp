// SPEC: docs/specs/gpu/SPEC.md §2 (RAII types, PIMPL compile firewall)
// Exec pack: docs/development/m6_execution_pack.md T6.2 (skeleton) + T6.3 (CUDA binding)
//
// This translation unit provides the PIMPL Impl struct definitions for
// DeviceStream and DeviceEvent. Two build paths:
//
//   TDMD_BUILD_CUDA=1 (default) — Impl binds cudaStream_t / cudaEvent_t.
//                                 Destructor calls cudaStreamDestroy /
//                                 cudaEventDestroy on non-null handles.
//                                 Factories in factories.cpp produce real
//                                 streams via cudaStreamCreateWithFlags.
//
//   TDMD_BUILD_CUDA=0          — Impl is an empty struct; the module still
//                                 supports null-construction and move
//                                 semantics (sufficient for CPU-only CI +
//                                 compile-time shape tests).
//
// The PIMPL split means public headers (types.hpp) compile on either
// path without touching CUDA headers — D-M6-17 compile firewall.

#include "tdmd/gpu/types.hpp"

#if TDMD_BUILD_CUDA
#include "cuda_handles.hpp"
#endif

namespace tdmd::gpu {

#if TDMD_BUILD_CUDA

// DeviceStream::Impl / DeviceEvent::Impl definitions live in cuda_handles.hpp
// so factories.cpp and device_pool.cpp can share the layout.

DeviceStream::DeviceStream() = default;

DeviceStream::DeviceStream(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

DeviceStream::~DeviceStream() {
  if (impl_ != nullptr && impl_->stream != nullptr) {
    // Destroy is best-effort: a failed destroy cannot be propagated from a
    // destructor. Errors end up in CUDA's sticky state and surface at the
    // next synchronising call; telemetry hook (T6.11) will log them.
    (void) cudaStreamDestroy(impl_->stream);
    impl_->stream = nullptr;
  }
}

DeviceStream::DeviceStream(DeviceStream&&) noexcept = default;
DeviceStream& DeviceStream::operator=(DeviceStream&&) noexcept = default;

DeviceEvent::DeviceEvent() = default;

DeviceEvent::DeviceEvent(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

DeviceEvent::~DeviceEvent() {
  if (impl_ != nullptr && impl_->event != nullptr) {
    (void) cudaEventDestroy(impl_->event);
    impl_->event = nullptr;
  }
}

DeviceEvent::DeviceEvent(DeviceEvent&&) noexcept = default;
DeviceEvent& DeviceEvent::operator=(DeviceEvent&&) noexcept = default;

#else  // CPU-only build

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

#endif  // TDMD_BUILD_CUDA

}  // namespace tdmd::gpu
