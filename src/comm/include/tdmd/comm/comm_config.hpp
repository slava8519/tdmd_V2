#pragma once

// SPEC: docs/specs/comm/SPEC.md §3.3 (YAML config), §11 (configuration/tuning)
// Master spec: §10.1 (deployment patterns), §10.2 (topologies)
// Exec pack: docs/development/m5_execution_pack.md T5.2
//
// Backend-selection and tuning knobs surfaced from tdmd.yaml `comm:` block.
// Defaults are Reference-safe: deterministic reductions on, CRC validation
// on, MpiHostStaging backend (universal fallback, D-M5-2).

#include <cstdint>
#include <string>

namespace tdmd::comm {

// Which concrete CommBackend to instantiate at SimulationEngine init.
// `Auto` probes hardware and picks the best available (M7+ HybridBackend
// default). M5 ships MpiHostStaging + Ring only (D-M5-2).
enum class BackendKind : std::uint8_t {
  Auto,
  MpiHostStaging,
  Ring,
  GpuAwareMpi,  // reserved for M6
  Nccl,         // reserved for M6
  Hybrid,       // reserved for M7
  Nvshmem,      // reserved for v2+
};

// Inner-level topology (intra-subdomain, used by Pattern 1 + Pattern 2
// inner). Ring is required for anchor-test §13.3; Mesh is default for
// general Pattern 1 runs.
enum class InnerTopology : std::uint8_t {
  Auto,
  Ring,
  Mesh,
};

// Outer-level topology (inter-subdomain, Pattern 2 only). Declared in M5
// but unused until M7.
enum class OuterTopology : std::uint8_t {
  Auto,
  Mesh,
};

struct CommConfig {
  BackendKind backend = BackendKind::Auto;
  InnerTopology inner_topology = InnerTopology::Auto;
  OuterTopology outer_topology = OuterTopology::Auto;

  // Deterministic Kahan ring-sum reduction (comm/SPEC §7.2). Required ON
  // in Reference profile (D-M5-9). Fast profile (M8+) may opt out.
  bool use_deterministic_reductions = true;

  // CRC32 validation on receiver — comm/SPEC §4.4, §8.4. Always on in M5
  // (D-M5-11). Fast profile (M8+) may disable.
  bool use_crc32 = true;

  // Send-side buffer pool size (per packet type, per rank). Default 32 —
  // covers K=8 max with 4× redundancy for async overlap (OQ-M5-1).
  std::uint32_t send_buffer_pool_size = 32;

  // Reserved for M6 GpuAwareMpi probing — harmless knob in M5.
  bool use_gpu_aware = false;
  bool use_nccl_intranode = false;

  // When true, the backend attempts to benchmark its own latency + bandwidth
  // during initialize() and populates BackendInfo. Off by default in M5 —
  // keeps smoke tests fast and deterministic.
  bool auto_bench_on_init = false;

  // Optional explicit subdomain layout for Pattern 2 (M7+). Empty in M5.
  std::string subdomain_layout;
};

}  // namespace tdmd::comm
