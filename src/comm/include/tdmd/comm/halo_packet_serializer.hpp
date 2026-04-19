#pragma once

// SPEC: docs/specs/comm/SPEC.md §4.2 (HaloPacket wire format),
//       §4.3 (protocol versioning), §4.4 (CRC32 policy)
// Master spec: §10 (parallel model), §12.6
// Exec pack: docs/development/m7_execution_pack.md T7.3
//
// Byte-level pack / unpack for HaloPacket. The wire layout mirrors the
// TemporalPacket layout (T5.3) so the receiver pipeline is structurally
// identical: header → opaque payload → CRC32 trailer.
//
// Wire format (fixed):
//
//   offset  size   field
//   0       2      protocol_version (uint16, big-endian)
//   2       4      source_subdomain_id
//   6       4      dest_subdomain_id
//   10      8      time_level
//   18      4      atom_count
//   22      4      payload_size
//   26      N      payload (atom_count × kHaloAtomRecordSize, or any opaque
//                  bytes — the comm module does not interpret them)
//   26+N    4      crc32 (covers all preceding bytes)
//
// Endian convention matches the TemporalPacket serializer: BE on the
// 2-byte protocol_version (network-sniffer friendly), LE on the rest
// (zero-cost on x86_64 / aarch64).
//
// Design note: HaloPacket payload IS a host byte vector even when the
// concrete backend is GPU-aware MPI. The CUDA-aware optimization applies
// only to the `MPI_Isend`'s data buffer pointer; the serialized format on
// the wire is the same regardless of backend.

#include "tdmd/comm/types.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tdmd::comm {

inline constexpr std::size_t kHaloHeaderSize = 26;
inline constexpr std::size_t kHaloCrcTrailerSize = 4;

// Per-atom record convention used when the HaloPacket payload was filled
// by the OuterSdCoordinator's snapshot builder. The comm module itself
// does not enforce or interpret this — it's exposed as a constant for the
// engine layer (T7.5/T7.9) to compute expected wire sizes.
inline constexpr std::size_t kHaloAtomRecordSize = 64;  // SoA pos+vel+id, 8 doubles

inline constexpr std::size_t halo_packet_wire_size(std::uint32_t payload_bytes) noexcept {
  return kHaloHeaderSize + static_cast<std::size_t>(payload_bytes) + kHaloCrcTrailerSize;
}

std::vector<std::uint8_t> pack_halo_packet(const HaloPacket& packet);

struct HaloUnpackResult {
  std::optional<HaloPacket> packet;
  std::string error;

  [[nodiscard]] bool ok() const noexcept { return packet.has_value(); }
};

// Validates: buffer length, protocol_version, header-payload consistency,
// CRC32. Any failure → HaloUnpackResult with .error populated and no packet.
HaloUnpackResult unpack_halo_packet(const std::uint8_t* data, std::size_t len);

inline HaloUnpackResult unpack_halo_packet(const std::vector<std::uint8_t>& bytes) {
  return unpack_halo_packet(bytes.data(), bytes.size());
}

}  // namespace tdmd::comm
