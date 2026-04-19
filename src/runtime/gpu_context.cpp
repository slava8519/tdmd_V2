// SPEC: docs/specs/runtime/SPEC.md §2.3; docs/specs/gpu/SPEC.md §9
// Exec pack: docs/development/m6_execution_pack.md T6.7

#include "tdmd/runtime/gpu_context.hpp"

#include "tdmd/gpu/factories.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace tdmd::runtime {

GpuContext::GpuContext(const tdmd::gpu::GpuConfig& cfg) {
  // `probe_devices` returns an empty vector on CPU-only builds, and
  // `select_device` / `make_stream` throw there — we surface those errors
  // unchanged so the engine can report a clean "CUDA not built in" message.
  const auto devices = tdmd::gpu::probe_devices();
  if (devices.empty()) {
    throw std::runtime_error(
        "tdmd GpuContext: no CUDA-capable device visible (rebuild with "
        "-DTDMD_BUILD_CUDA=ON or set runtime.backend=cpu)");
  }
  if (cfg.device_id < 0 || static_cast<std::size_t>(cfg.device_id) >= devices.size()) {
    throw std::runtime_error("tdmd GpuContext: gpu.device_id=" + std::to_string(cfg.device_id) +
                             " out of range; " + std::to_string(devices.size()) +
                             " device(s) visible");
  }

  device_info_ = tdmd::gpu::select_device(static_cast<tdmd::gpu::DeviceId>(cfg.device_id));
  pool_ = std::make_unique<tdmd::gpu::DevicePool>(cfg);
  compute_stream_ = tdmd::gpu::make_stream(device_info_.device_id);
  // D-M6-13: second non-blocking stream for copies that can execute alongside
  // compute. Real orchestration (cudaEventRecord + cudaStreamWaitEvent) lands
  // in T6.9b when there is overlap-able work; the stream exists at T6.9a so
  // that future adapter surfaces can borrow `mem_stream()` without re-opening
  // the GpuContext contract.
  mem_stream_ = tdmd::gpu::make_stream(device_info_.device_id);
}

GpuContext::~GpuContext() = default;

}  // namespace tdmd::runtime
