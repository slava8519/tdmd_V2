// SPEC: docs/specs/comm/SPEC.md §4.2, §4.3, §4.4
// Master spec: §10
// Exec pack: docs/development/m7_execution_pack.md T7.3
//
// HaloPacket pack/unpack. Mirrors the TemporalPacket serializer (T5.3)
// in style — explicit per-byte writers, no struct memcpy shortcuts. The
// payload is opaque to comm: we write `packet.payload.size()` bytes and
// trust the caller to have filled it. The receiver sees `payload_size`
// in the header and reads exactly that many bytes back.

#include "tdmd/comm/halo_packet_serializer.hpp"

#include "tdmd/comm/crc32.hpp"

#include <cstring>

namespace tdmd::comm {

namespace {

// Same little-endian writers used by the temporal serializer. Inlined into
// each translation unit rather than shared via a private header — these
// are 4 lines each and a header would force every comm consumer to pull in
// the byte-twiddling decls just for backend internals.
void write_u32_le(std::uint8_t* dst, std::uint32_t v) noexcept {
  dst[0] = static_cast<std::uint8_t>(v & 0xFFu);
  dst[1] = static_cast<std::uint8_t>((v >> 8) & 0xFFu);
  dst[2] = static_cast<std::uint8_t>((v >> 16) & 0xFFu);
  dst[3] = static_cast<std::uint8_t>((v >> 24) & 0xFFu);
}

void write_u64_le(std::uint8_t* dst, std::uint64_t v) noexcept {
  for (int i = 0; i < 8; ++i) {
    dst[i] = static_cast<std::uint8_t>((v >> (8 * i)) & 0xFFu);
  }
}

void write_u16_be(std::uint8_t* dst, std::uint16_t v) noexcept {
  dst[0] = static_cast<std::uint8_t>((v >> 8) & 0xFFu);
  dst[1] = static_cast<std::uint8_t>(v & 0xFFu);
}

std::uint32_t read_u32_le(const std::uint8_t* src) noexcept {
  return static_cast<std::uint32_t>(src[0]) | (static_cast<std::uint32_t>(src[1]) << 8) |
         (static_cast<std::uint32_t>(src[2]) << 16) | (static_cast<std::uint32_t>(src[3]) << 24);
}

std::uint64_t read_u64_le(const std::uint8_t* src) noexcept {
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) {
    v |= static_cast<std::uint64_t>(src[i]) << (8 * i);
  }
  return v;
}

std::uint16_t read_u16_be(const std::uint8_t* src) noexcept {
  return static_cast<std::uint16_t>(static_cast<std::uint16_t>(src[1]) |
                                    (static_cast<std::uint16_t>(src[0]) << 8));
}

}  // namespace

std::vector<std::uint8_t> pack_halo_packet(const HaloPacket& packet) {
  const std::uint32_t payload_size = static_cast<std::uint32_t>(packet.payload.size());

  std::vector<std::uint8_t> out;
  out.resize(halo_packet_wire_size(payload_size));

  std::uint8_t* p = out.data();

  write_u16_be(p + 0, packet.protocol_version);
  write_u32_le(p + 2, packet.source_subdomain_id);
  write_u32_le(p + 6, packet.dest_subdomain_id);
  write_u64_le(p + 10, packet.time_level);
  write_u32_le(p + 18, packet.atom_count);
  write_u32_le(p + 22, payload_size);

  if (payload_size > 0) {
    std::memcpy(p + kHaloHeaderSize, packet.payload.data(), payload_size);
  }

  const std::size_t crc_region = kHaloHeaderSize + payload_size;
  const std::uint32_t crc = crc32(p, crc_region);
  write_u32_le(p + crc_region, crc);

  return out;
}

HaloUnpackResult unpack_halo_packet(const std::uint8_t* data, std::size_t len) {
  HaloUnpackResult r;
  if (len < kHaloHeaderSize + kHaloCrcTrailerSize) {
    r.error = "halo: buffer truncated below header+trailer minimum";
    return r;
  }

  HaloPacket pkt;
  pkt.protocol_version = read_u16_be(data + 0);
  if (pkt.protocol_version != kCommProtocolVersion) {
    r.error = "halo: protocol version mismatch";
    return r;
  }
  pkt.source_subdomain_id = read_u32_le(data + 2);
  pkt.dest_subdomain_id = read_u32_le(data + 6);
  pkt.time_level = read_u64_le(data + 10);
  pkt.atom_count = read_u32_le(data + 18);
  const std::uint32_t payload_size = read_u32_le(data + 22);

  const std::size_t expected_total = halo_packet_wire_size(payload_size);
  if (len != expected_total) {
    r.error = "halo: declared payload size doesn't match buffer length";
    return r;
  }

  // CRC validation BEFORE we copy the payload — a corrupted size header
  // could otherwise drive us to allocate a bogus vector.
  const std::size_t crc_region = kHaloHeaderSize + payload_size;
  const std::uint32_t computed = crc32(data, crc_region);
  const std::uint32_t carried = read_u32_le(data + crc_region);
  if (computed != carried) {
    r.error = "halo: CRC32 mismatch";
    return r;
  }

  pkt.payload.assign(data + kHaloHeaderSize, data + kHaloHeaderSize + payload_size);
  pkt.crc32 = carried;
  r.packet = std::move(pkt);
  return r;
}

}  // namespace tdmd::comm
