#pragma once

// SPEC: docs/specs/comm/SPEC.md §2.1 (core types), §4.1 (wire format)
// Master spec: §10 (parallel model), §12.6 (comm interfaces)
// Exec pack: docs/development/m5_execution_pack.md T5.2
//
// Pure value types for the comm module. Like scheduler/ types, this header
// is dependency-free — no state/, no scheduler/, no potentials/. The comm
// module is a transport layer; packet payloads are opaque bytes from comm's
// perspective.
//
// Serialization of packet payloads (pack/unpack + CRC32) lands in T5.3.
// Concrete backend classes (MpiHostStaging, Ring) land in T5.4 / T5.5.

#include <cstdint>
#include <string>
#include <vector>

namespace tdmd::comm {

// Stable zone index across the whole subdomain. Re-declared here to keep
// the comm module free of a build-time dependency on scheduler/. The two
// ZoneId aliases are guaranteed to be the same underlying type by a
// static_assert at the boundary (T5.7 wires scheduler → comm).
using ZoneId = std::uint32_t;
using TimeLevel = std::uint64_t;
using Version = std::uint64_t;

// Protocol version per comm/SPEC §4.3. v1 is the initial release (D-M5-10).
// Incompatible wire-format changes (e.g. per-atom force in v1.1) bump this
// value; receivers reject packets whose version differs from their own.
inline constexpr std::uint16_t kCommProtocolVersion = 1;

// Box snapshot carried with each TemporalPacket so the receiver can apply
// periodic wrap consistently regardless of its own current box state.
// Matches comm/SPEC §4.1 (48 bytes = 6 × double).
struct Box {
  double xlo = 0.0;
  double xhi = 0.0;
  double ylo = 0.0;
  double yhi = 0.0;
  double zlo = 0.0;
  double zhi = 0.0;
};

// Inner-level transfer: per-zone atom data between ranks within a
// subdomain. See comm/SPEC §2.1 and master spec §10.4 (temporal packet
// protocol). Wire format specified by §4.1 — pack/unpack in T5.3.
struct TemporalPacket {
  std::uint16_t protocol_version = kCommProtocolVersion;
  ZoneId zone_id = 0;
  TimeLevel time_level = 0;
  Version version = 0;
  std::uint32_t atom_count = 0;
  Box box_snapshot{};
  std::uint64_t certificate_hash = 0;
  // Serialized per-atom binary data (76 bytes/atom per comm/SPEC §4.1).
  // Layout: id(8) | species(4) | pos(24) | vel(24) | image_flags(12) |
  // flags(4). T5.3 owns the pack/unpack path.
  std::vector<std::uint8_t> payload;
  std::uint32_t crc32 = 0;
};

// Outer-level transfer: halo exchange between subdomains (Pattern 2, M7+).
// Declared in M5 but unused — HaloPacket send/drain paths are no-op in the
// M5 backends. See comm/SPEC §4.2.
struct HaloPacket {
  std::uint16_t protocol_version = kCommProtocolVersion;
  std::uint32_t source_subdomain_id = 0;
  std::uint32_t dest_subdomain_id = 0;
  TimeLevel time_level = 0;
  std::uint32_t atom_count = 0;
  std::vector<std::uint8_t> payload;
  std::uint32_t crc32 = 0;
};

// Cross-subdomain atom migration (Pattern 2, M7+). In M5 Pattern 1, atom
// ownership follows zones and is carried inside TemporalPacket — see D-M5-8.
struct MigrationPacket {
  std::uint16_t protocol_version = kCommProtocolVersion;
  std::uint32_t source_subdomain_id = 0;
  std::uint32_t dest_subdomain_id = 0;
  std::uint32_t atom_count = 0;
  std::vector<std::uint8_t> payload;
  std::uint32_t crc32 = 0;
};

// Abstract addressing for collectives and send targets. Inner = ring/mesh
// peer within the current subdomain; Outer = neighbor subdomain; Global =
// reduction target. See comm/SPEC §3.2.
enum class CommEndpoint : std::uint8_t {
  InnerTdPeer,
  OuterSdPeer,
  GlobalRoot,
};

// Capabilities a backend advertises at init. Discovered programmatically
// by probes; used by HybridBackend (M7+) for routing decisions. See
// comm/SPEC §2.1.
enum class BackendCapability : std::uint8_t {
  GpuAwarePointers,
  RemoteDirectMemory,
  CollectiveOptimized,
  RingTopologyNative,
};

// Backend self-description — published via CommBackend::info() after init.
// `measured_bw_bytes_per_sec` and `measured_latency_us` are populated by
// the backend's auto-bench path during initialize(); values are advisory
// and may be zero if measurement is skipped or unavailable.
struct BackendInfo {
  std::string name;
  std::vector<BackendCapability> capabilities;
  std::uint64_t protocol_version = kCommProtocolVersion;
  double measured_bw_bytes_per_sec = 0.0;
  double measured_latency_us = 0.0;
};

}  // namespace tdmd::comm
