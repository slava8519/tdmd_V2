// SPEC: docs/specs/comm/SPEC.md §4.2 (HaloPacket wire format), §4.3, §4.4
// Master spec: §10
// Exec pack: docs/development/m7_execution_pack.md T7.3
//
// Halo packet (de)serialization unit tests. Mirrors the TemporalPacket
// serializer test (T5.3) — round-trip integrity, CRC tampering rejection,
// truncation rejection, version mismatch.

#include "tdmd/comm/halo_packet_serializer.hpp"
#include "tdmd/comm/types.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>

namespace tc = tdmd::comm;

namespace {

tc::HaloPacket make_packet(std::uint32_t atom_count = 16, std::uint8_t fill = 0xCC) {
  tc::HaloPacket p;
  p.protocol_version = tc::kCommProtocolVersion;
  p.source_subdomain_id = 7;
  p.dest_subdomain_id = 11;
  p.time_level = 42;
  p.atom_count = atom_count;
  p.payload.assign(atom_count * tc::kHaloAtomRecordSize, fill);
  return p;
}

}  // namespace

TEST_CASE("halo serializer — round-trip preserves all fields", "[comm][halo][serializer]") {
  const auto src = make_packet(8, 0xAB);
  const auto bytes = tc::pack_halo_packet(src);
  REQUIRE(bytes.size() ==
          tc::halo_packet_wire_size(static_cast<std::uint32_t>(src.payload.size())));

  const auto r = tc::unpack_halo_packet(bytes);
  REQUIRE(r.ok());
  const auto& out = *r.packet;
  REQUIRE(out.protocol_version == src.protocol_version);
  REQUIRE(out.source_subdomain_id == src.source_subdomain_id);
  REQUIRE(out.dest_subdomain_id == src.dest_subdomain_id);
  REQUIRE(out.time_level == src.time_level);
  REQUIRE(out.atom_count == src.atom_count);
  REQUIRE(out.payload == src.payload);
}

TEST_CASE("halo serializer — empty payload round-trips", "[comm][halo][serializer]") {
  tc::HaloPacket src;
  src.source_subdomain_id = 0;
  src.dest_subdomain_id = 1;
  src.time_level = 0;
  src.atom_count = 0;
  // payload intentionally empty
  const auto bytes = tc::pack_halo_packet(src);
  REQUIRE(bytes.size() == tc::halo_packet_wire_size(0));

  const auto r = tc::unpack_halo_packet(bytes);
  REQUIRE(r.ok());
  REQUIRE(r.packet->payload.empty());
  REQUIRE(r.packet->source_subdomain_id == 0);
  REQUIRE(r.packet->dest_subdomain_id == 1);
}

TEST_CASE("halo serializer — CRC tamper detection", "[comm][halo][serializer]") {
  const auto src = make_packet(4, 0x33);
  auto bytes = tc::pack_halo_packet(src);
  REQUIRE(bytes.size() > tc::kHaloHeaderSize);
  // Flip a payload byte; CRC should reject.
  bytes[tc::kHaloHeaderSize] ^= 0x01;
  const auto r = tc::unpack_halo_packet(bytes);
  REQUIRE_FALSE(r.ok());
  REQUIRE(r.error.find("CRC") != std::string::npos);
}

TEST_CASE("halo serializer — truncated buffer rejected", "[comm][halo][serializer]") {
  const auto src = make_packet(4, 0x55);
  auto bytes = tc::pack_halo_packet(src);
  bytes.resize(tc::kHaloHeaderSize - 1);  // chop into the header
  const auto r = tc::unpack_halo_packet(bytes);
  REQUIRE_FALSE(r.ok());
  REQUIRE(r.error.find("truncated") != std::string::npos);
}

TEST_CASE("halo serializer — bad protocol version rejected", "[comm][halo][serializer]") {
  auto src = make_packet(2);
  src.protocol_version = static_cast<std::uint16_t>(tc::kCommProtocolVersion + 99);
  const auto bytes = tc::pack_halo_packet(src);
  const auto r = tc::unpack_halo_packet(bytes);
  REQUIRE_FALSE(r.ok());
  REQUIRE(r.error.find("protocol version") != std::string::npos);
}

TEST_CASE("halo serializer — payload-size mismatch rejected", "[comm][halo][serializer]") {
  const auto src = make_packet(4, 0x77);
  auto bytes = tc::pack_halo_packet(src);
  // Drop trailing CRC bytes — header still claims a payload size that
  // exceeds what's now in the buffer.
  bytes.resize(bytes.size() - 4);
  const auto r = tc::unpack_halo_packet(bytes);
  REQUIRE_FALSE(r.ok());
}

TEST_CASE("halo serializer — large payload (multi-KB) round-trips",
          "[comm][halo][serializer][large]") {
  // Stress the buffer growth path with 256 atoms × 64 B = 16 KB payload.
  const auto src = make_packet(256, 0x9A);
  const auto bytes = tc::pack_halo_packet(src);
  const auto r = tc::unpack_halo_packet(bytes);
  REQUIRE(r.ok());
  REQUIRE(r.packet->payload.size() == src.payload.size());
  REQUIRE(r.packet->payload == src.payload);
}
