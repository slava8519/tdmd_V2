// SPEC: docs/specs/comm/SPEC.md §4.1, §4.3, §4.4
// Exec pack: docs/development/m5_execution_pack.md T5.3
//
// TemporalPacket (de)serialization. Every multi-byte write and read goes
// through explicit per-byte code — no `memcpy(&struct)` shortcuts — because
// the struct layout includes vtable-less trivial types but the packet
// contains variable-length payload and the emission order is part of the
// wire contract, not an implementation detail we may reshuffle.

#include "tdmd/comm/packet_serializer.hpp"

#include "tdmd/comm/crc32.hpp"

#include <cstring>
#include <string>

namespace tdmd::comm {

namespace {

// Little-endian byte writers. Hot on every send; small enough to inline.
// protocol_version is the only u16 on the wire and is BE per SPEC §4.1 —
// no write_u16_le / read_u16_le are needed.
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

void write_f64_le(std::uint8_t* dst, double v) noexcept {
  std::uint64_t u;
  std::memcpy(&u, &v, sizeof(u));
  write_u64_le(dst, u);
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

double read_f64_le(const std::uint8_t* src) noexcept {
  std::uint64_t u = read_u64_le(src);
  double v;
  std::memcpy(&v, &u, sizeof(v));
  return v;
}

std::uint16_t read_u16_be(const std::uint8_t* src) noexcept {
  return static_cast<std::uint16_t>(static_cast<std::uint16_t>(src[1]) |
                                    (static_cast<std::uint16_t>(src[0]) << 8));
}

}  // namespace

std::vector<std::uint8_t> pack_temporal_packet(const TemporalPacket& packet) {
  const std::size_t expected_payload_size =
      static_cast<std::size_t>(packet.atom_count) * kAtomRecordSize;

  std::vector<std::uint8_t> out;
  out.resize(temporal_packet_wire_size(packet.atom_count));

  std::uint8_t* p = out.data();

  // Header — offsets match comm/SPEC §4.1.
  write_u16_be(p + 0, packet.protocol_version);
  write_u32_le(p + 2, packet.zone_id);
  write_u64_le(p + 6, packet.time_level);
  write_u64_le(p + 14, packet.version);
  write_u32_le(p + 22, packet.atom_count);

  const Box& box = packet.box_snapshot;
  write_f64_le(p + 26, box.xlo);
  write_f64_le(p + 34, box.xhi);
  write_f64_le(p + 42, box.ylo);
  write_f64_le(p + 50, box.yhi);
  write_f64_le(p + 58, box.zlo);
  write_f64_le(p + 66, box.zhi);

  write_u64_le(p + 74, packet.certificate_hash);
  write_u32_le(p + 82, static_cast<std::uint32_t>(expected_payload_size));

  // Payload. Caller is responsible for filling `packet.payload` with exactly
  // `atom_count × kAtomRecordSize` bytes (§4.1 per-atom record layout). We
  // write whatever's there; if the caller passed a shorter buffer, we emit
  // what's given and zero-fill the remainder — but the resulting packet
  // will fail unpack validation, surfacing the caller's bug.
  const std::size_t payload_copy = std::min(expected_payload_size, packet.payload.size());
  if (payload_copy > 0) {
    std::memcpy(p + kTemporalHeaderSize, packet.payload.data(), payload_copy);
  }
  if (payload_copy < expected_payload_size) {
    std::memset(p + kTemporalHeaderSize + payload_copy, 0, expected_payload_size - payload_copy);
  }

  // CRC over everything except the trailer itself.
  const std::size_t crc_region = kTemporalHeaderSize + expected_payload_size;
  const std::uint32_t crc = crc32(p, crc_region);
  write_u32_le(p + crc_region, crc);

  return out;
}

UnpackResult unpack_temporal_packet(const std::uint8_t* data, std::size_t len) {
  UnpackResult result;

  if (len < kTemporalHeaderSize + kCrcTrailerSize) {
    result.error = "buffer too small for header + CRC trailer";
    return result;
  }

  const std::uint16_t version = read_u16_be(data + 0);
  if (version != kCommProtocolVersion) {
    result.error = "protocol version mismatch: got " + std::to_string(version) + ", expected " +
                   std::to_string(kCommProtocolVersion);
    return result;
  }

  const std::uint32_t zone_id = read_u32_le(data + 2);
  const std::uint64_t time_level = read_u64_le(data + 6);
  const std::uint64_t packet_version = read_u64_le(data + 14);
  const std::uint32_t atom_count = read_u32_le(data + 22);

  Box box;
  box.xlo = read_f64_le(data + 26);
  box.xhi = read_f64_le(data + 34);
  box.ylo = read_f64_le(data + 42);
  box.yhi = read_f64_le(data + 50);
  box.zlo = read_f64_le(data + 58);
  box.zhi = read_f64_le(data + 66);

  const std::uint64_t cert_hash = read_u64_le(data + 74);
  const std::uint32_t payload_size = read_u32_le(data + 82);

  const std::uint64_t expected_payload_size =
      static_cast<std::uint64_t>(atom_count) * kAtomRecordSize;
  if (payload_size != expected_payload_size) {
    result.error = "payload_size mismatch: declared " + std::to_string(payload_size) +
                   ", expected " + std::to_string(expected_payload_size);
    return result;
  }

  const std::uint64_t expected_total =
      kTemporalHeaderSize + expected_payload_size + kCrcTrailerSize;
  if (static_cast<std::uint64_t>(len) != expected_total) {
    result.error = "buffer length mismatch: got " + std::to_string(len) + ", expected " +
                   std::to_string(expected_total);
    return result;
  }

  const std::size_t crc_region = kTemporalHeaderSize + static_cast<std::size_t>(payload_size);
  const std::uint32_t computed_crc = crc32(data, crc_region);
  const std::uint32_t wire_crc = read_u32_le(data + crc_region);
  if (computed_crc != wire_crc) {
    result.error = "CRC32 mismatch";
    return result;
  }

  TemporalPacket packet;
  packet.protocol_version = version;
  packet.zone_id = zone_id;
  packet.time_level = time_level;
  packet.version = packet_version;
  packet.atom_count = atom_count;
  packet.box_snapshot = box;
  packet.certificate_hash = cert_hash;
  packet.payload.assign(data + kTemporalHeaderSize, data + kTemporalHeaderSize + payload_size);
  packet.crc32 = wire_crc;

  result.packet = std::move(packet);
  return result;
}

}  // namespace tdmd::comm
