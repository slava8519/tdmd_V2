// SPEC: docs/specs/gpu/SPEC.md §2 (types), §4 (probe)
// Exec pack: docs/development/m6_execution_pack.md T6.3

#include "tdmd/gpu/factories.hpp"

#include <sstream>
#include <string>

#if TDMD_BUILD_CUDA
#include "cuda_handles.hpp"
#endif

namespace tdmd::gpu {

#if TDMD_BUILD_CUDA

namespace {

[[noreturn]] void throw_cuda_error(const char* op, cudaError_t err) {
  std::ostringstream oss;
  oss << "gpu::" << op << " failed: " << cudaGetErrorName(err) << " (" << cudaGetErrorString(err)
      << ")";
  throw std::runtime_error(oss.str());
}

DeviceInfo device_info_for(int id) {
  cudaDeviceProp prop{};
  cudaError_t err = cudaGetDeviceProperties(&prop, id);
  if (err != cudaSuccess) {
    throw_cuda_error("cudaGetDeviceProperties", err);
  }

  DeviceInfo info;
  info.device_id = id;
  info.name = prop.name;
  info.compute_capability_major = static_cast<std::uint32_t>(prop.major);
  info.compute_capability_minor = static_cast<std::uint32_t>(prop.minor);
  info.total_global_memory_bytes = prop.totalGlobalMem;
  info.multiprocessor_count = static_cast<std::uint32_t>(prop.multiProcessorCount);
  info.warp_size = static_cast<std::uint32_t>(prop.warpSize);
  info.max_threads_per_block = static_cast<std::uint32_t>(prop.maxThreadsPerBlock);

  // Capability inference from compute capability — see gpu/SPEC §2.2.
  // sm_80+ (Ampere and later) covers everything M6 targets (D-M6-1).
  if (prop.major >= 8) {
    info.capabilities.push_back(GpuCapability::TensorCores);
    info.capabilities.push_back(GpuCapability::AsyncMemcpy);
    info.capabilities.push_back(GpuCapability::L2CachePersistence);
  }
  if (prop.major >= 6) {
    info.capabilities.push_back(GpuCapability::CooperativeLaunch);
  }
  // cudaMallocAsync requires CUDA 11.2 runtime + sm_60+. We gate M6 on
  // CUDA 13.1 (D-M6-2) so the runtime is always recent enough; compute
  // capability ≥ 6 covers it. sm_80+ minimum per D-M6-1 guarantees this.
  if (prop.memoryPoolsSupported != 0) {
    info.capabilities.push_back(GpuCapability::CudaMallocAsync);
  }
  return info;
}

}  // namespace

std::vector<DeviceInfo> probe_devices() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver) {
    return {};
  }
  if (err != cudaSuccess) {
    throw_cuda_error("cudaGetDeviceCount", err);
  }
  std::vector<DeviceInfo> out;
  out.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    out.push_back(device_info_for(i));
  }
  return out;
}

DeviceInfo select_device(DeviceId id) {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    throw_cuda_error("cudaGetDeviceCount", err);
  }
  if (id < 0 || id >= count) {
    std::ostringstream oss;
    oss << "gpu::select_device: id=" << id << " out of range [0, " << count << ")";
    throw std::runtime_error(oss.str());
  }
  err = cudaSetDevice(id);
  if (err != cudaSuccess) {
    throw_cuda_error("cudaSetDevice", err);
  }
  return device_info_for(id);
}

DeviceStream make_stream(DeviceId device_id) {
  cudaError_t err = cudaSetDevice(device_id);
  if (err != cudaSuccess) {
    throw_cuda_error("cudaSetDevice", err);
  }
  cudaStream_t s = nullptr;
  // Non-blocking: does not serialise against legacy NULL stream — gpu/SPEC §3.1.
  err = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    throw_cuda_error("cudaStreamCreateWithFlags", err);
  }
  auto impl = std::make_unique<DeviceStream::Impl>();
  impl->stream = s;
  return DeviceStream(std::move(impl));
}

DeviceEvent make_event(DeviceId device_id) {
  cudaError_t err = cudaSetDevice(device_id);
  if (err != cudaSuccess) {
    throw_cuda_error("cudaSetDevice", err);
  }
  cudaEvent_t e = nullptr;
  // Disable timing: pure sync barrier, cheaper than a timed event.
  err = cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
  if (err != cudaSuccess) {
    throw_cuda_error("cudaEventCreateWithFlags", err);
  }
  auto impl = std::make_unique<DeviceEvent::Impl>();
  impl->event = e;
  return DeviceEvent(std::move(impl));
}

#else  // CPU-only build

std::vector<DeviceInfo> probe_devices() {
  return {};
}

DeviceInfo select_device(DeviceId /*id*/) {
  throw std::runtime_error(
      "gpu::select_device: CPU-only build (TDMD_BUILD_CUDA=0); no CUDA devices available");
}

DeviceStream make_stream(DeviceId /*device_id*/) {
  throw std::runtime_error(
      "gpu::make_stream: CPU-only build (TDMD_BUILD_CUDA=0); no CUDA runtime linked");
}

DeviceEvent make_event(DeviceId /*device_id*/) {
  throw std::runtime_error(
      "gpu::make_event: CPU-only build (TDMD_BUILD_CUDA=0); no CUDA runtime linked");
}

#endif  // TDMD_BUILD_CUDA

}  // namespace tdmd::gpu
