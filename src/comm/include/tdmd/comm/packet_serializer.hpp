#pragma once

// SPEC: docs/specs/comm/SPEC.md §4.1 (TemporalPacket wire format),
//       §4.3 (protocol versioning), §4.4 (CRC32 policy)
// Exec pack: docs/development/m5_execution_pack.md T5.3
//
// Byte-level pack / unpack for TemporalPacket. Wire layout (fixed per §4.1):
//
//   offset  size   field
//   0       2      protocol_version (uint16, big-endian)
//   2       4      zone_id
//   6       8      time_level
//   14      8      version
//   22      4      atom_count
//   26      48     box_snapshot (6 × double)
//   74      8      certificate_hash
//   82      4      payload_size
//   86      N      payload (atom_count × kAtomRecordSize bytes)
//   86+N    4      crc32 (covers all preceding bytes)
//
// Per-atom record (76 bytes):
//   0   8   atom_id
//   8   4   species
//   12  24  position (3 × double)
//   36  24  velocity (3 × double)
//   60  12  image_flags (3 × int32)
//   72  4   flags
//
// Endian convention (T5.3 scope): `protocol_version` big-endian per SPEC
// §4.1 explicit note; every other multi-byte field little-endian. Hosts
// TDMD ships on are little-endian (x86_64, aarch64 default), so LE fields
// cost zero byte-shuffle on send/receive. BE `protocol_version` was a
// deliberate SPEC choice — it makes the first 2 bytes of every packet
// network-order-recognizable by off-the-shelf network sniffers / logs.

#include "tdmd/comm/types.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tdmd::comm {

inline constexpr std::size_t kTemporalHeaderSize = 86;
inline constexpr std::size_t kAtomRecordSize = 76;
inline constexpr std::size_t kCrcTrailerSize = 4;

// Compute the exact serialized byte length of a TemporalPacket for a given
// atom_count. Handy for pre-allocating send buffers (T5.4 / T5.5).
inline constexpr std::size_t temporal_packet_wire_size(std::uint32_t atom_count) noexcept {
  return kTemporalHeaderSize + (static_cast<std::size_t>(atom_count) * kAtomRecordSize) +
         kCrcTrailerSize;
}

// Pack `packet` into a freshly allocated byte buffer. `packet.payload` is
// expected to already hold `atom_count × kAtomRecordSize` bytes (callers in
// T5.7 build it via the atom-record builder alongside the certificate emit).
// CRC32 is computed over the header + payload and written to the trailer.
// Returns the serialized buffer (length == temporal_packet_wire_size).
std::vector<std::uint8_t> pack_temporal_packet(const TemporalPacket& packet);

// Result of an unpack attempt. `packet` is present on success; `error`
// carries a human-readable message on failure (version mismatch, CRC
// mismatch, truncated buffer, malformed payload length).
struct UnpackResult {
  std::optional<TemporalPacket> packet;
  std::string error;

  bool ok() const noexcept { return packet.has_value(); }
};

// Unpack bytes into a TemporalPacket. Validates (in order):
//   1. buffer length is at least `kTemporalHeaderSize + kCrcTrailerSize`
//   2. protocol_version matches `kCommProtocolVersion` (D-M5-10)
//   3. payload_size == atom_count × kAtomRecordSize
//   4. buffer length matches header's declared wire size
//   5. CRC32 over (all bytes excluding trailer) == trailer
//
// Any failure yields UnpackResult{.error = "..."}. No partial packets are
// ever returned — success is all-or-nothing.
UnpackResult unpack_temporal_packet(const std::uint8_t* data, std::size_t len);

// Convenience overload.
inline UnpackResult unpack_temporal_packet(const std::vector<std::uint8_t>& bytes) {
  return unpack_temporal_packet(bytes.data(), bytes.size());
}

}  // namespace tdmd::comm
