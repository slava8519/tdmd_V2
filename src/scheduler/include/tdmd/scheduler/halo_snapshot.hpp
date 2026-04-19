#pragma once

// SPEC: docs/specs/scheduler/SPEC.md §4.6 (HaloSnapshot type + ring archive)
// Master spec: §12.7a (OuterSdCoordinator interface)
// Exec pack: docs/development/m7_execution_pack.md T7.6
//
// In-memory archive record produced by `OuterSdCoordinator::unpack_halo()`
// from a `tdmd::HaloPacket` (comm wire format, see comm/SPEC §4.2). The
// ownership boundary (master §8.2) puts wire format in comm/, unpack +
// archive in scheduler/. T7.6 lands the type and archive; the actual
// packet→snapshot conversion is wired in T7.5/T7.7.

#include "tdmd/scheduler/types.hpp"

#include <cstdint>
#include <vector>

namespace tdmd::scheduler {

struct HaloSnapshot {
  std::uint32_t source_subdomain_id = 0;
  // Peer's local zone id within its inner DAG (peer subdomain's namespace).
  ZoneId source_zone_id = 0;
  TimeLevel time_level = 0;
  // Peer's state version at the time of pack (for stale-discard).
  Version source_version = 0;
  std::uint32_t atom_count = 0;
  // Unpacked SoA payload matching state/SPEC §3.2 (FP64 in Reference).
  std::vector<std::uint8_t> payload;
  // Monotonic per-coordinator counter, deterministic in Reference profile.
  std::uint64_t received_seq = 0;
};

}  // namespace tdmd::scheduler
