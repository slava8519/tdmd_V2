#pragma once

// SPEC: docs/specs/gpu/SPEC.md §11 (configuration / tuning)
// Master spec: §14 M6, §15.2
// Exec pack: docs/development/m6_execution_pack.md T6.2, D-M6-12, D-M6-13,
//            D-M6-14, D-M6-18
//
// Runtime-tunable knobs surfaced from tdmd.yaml `gpu:` block. Defaults are
// Reference-safe: device 0, two streams (compute + mem per D-M6-13), 256 MiB
// initial pool warm-up (D-M6-12), NVTX on (D-M6-14).

#include <cstdint>

namespace tdmd::gpu {

struct GpuConfig {
  // CUDA device ordinal; ranks override via CLI --gpu-device (D-M6-18).
  std::int32_t device_id = 0;

  // Streams owned per rank. 2 = (stream_compute, stream_mem) per D-M6-13.
  // Single-stream debug mode = 1; N-stream pipelining deferred to M7+.
  std::uint32_t streams = 2;

  // Initial size of DevicePool (cached device allocator, D-M6-12). Actual
  // pool grows on demand via cudaMallocAsync; this is the warm-up allocation
  // so the first kernel launch does not stall on allocator cold-start.
  std::uint32_t memory_pool_init_size_mib = 256;

  // Toggle NVTX_RANGE annotations. Overhead << 1 % per D-M6-14 — keep on.
  bool enable_nvtx = true;
};

}  // namespace tdmd::gpu
