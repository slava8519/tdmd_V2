// Exec pack: docs/development/m5_execution_pack.md T5.3
// SPEC: docs/specs/comm/SPEC.md §4.1 (wire format), §4.3 (versioning),
//       §4.4 (CRC32 policy)
//
// Byte-level validation for TemporalPacket serialization:
//   * RFC 1952 / zlib CRC32 anchored on "123456789" → 0xCBF43926
//   * Pack → bytes → unpack roundtrip across 0 / 1 / 10 / 100 / 1000 atoms
//   * CRC flip detection
//   * Protocol-version mismatch detection
//   * Header / per-atom / trailer size pins
//
// No MPI required — pure serialization.

#include "tdmd/comm/crc32.hpp"
#include "tdmd/comm/packet_serializer.hpp"
#include "tdmd/comm/types.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace tc = tdmd::comm;

namespace {

// Deterministic payload builder — lets the roundtrip test cover non-trivial
// byte patterns without depending on uninitialized memory. Seed is fixed so
// the test is byte-reproducible run-to-run (playbook Level 1 determinism).
std::vector<std::uint8_t> make_random_payload(std::uint32_t atom_count, std::uint64_t seed) {
  const std::size_t len = static_cast<std::size_t>(atom_count) * tc::kAtomRecordSize;
  std::vector<std::uint8_t> bytes(len);
  std::mt19937_64 rng(seed);
  for (std::size_t i = 0; i < len; ++i) {
    bytes[i] = static_cast<std::uint8_t>(rng() & 0xFFu);
  }
  return bytes;
}

tc::TemporalPacket make_packet(std::uint32_t atom_count, std::uint64_t seed) {
  tc::TemporalPacket pkt;
  pkt.zone_id = 7;
  pkt.time_level = 42;
  pkt.version = 1234;
  pkt.atom_count = atom_count;
  pkt.box_snapshot = tc::Box{0.0, 10.0, 0.0, 20.0, -5.0, 5.0};
  pkt.certificate_hash = 0xDEADBEEFCAFEBABEull;
  pkt.payload = make_random_payload(atom_count, seed);
  return pkt;
}

}  // namespace

TEST_CASE("CRC32 — known vector '123456789' == 0xCBF43926", "[comm][crc32]") {
  // RFC 1952 / zlib standard anchor. Any divergence from this value means
  // the reflected polynomial or the init/final XOR is wrong and wire
  // compatibility with every network stack TDMD peers with is broken.
  const std::string s = "123456789";
  const std::uint32_t c = tc::crc32(s.data(), s.size());
  REQUIRE(c == 0xCBF43926u);
}

TEST_CASE("CRC32 — empty input has identity checksum", "[comm][crc32]") {
  // zlib convention: CRC of empty buffer is 0.
  REQUIRE(tc::crc32(nullptr, 0) == 0u);
}

TEST_CASE("CRC32 — streaming update matches one-shot", "[comm][crc32]") {
  // Demonstrates the chained API works identically to a single call —
  // guarantees T5.4 can CRC a header + payload in two update() calls.
  const std::string a = "1234";
  const std::string b = "56789";

  const std::uint32_t one_shot = tc::crc32("123456789", 9);
  std::uint32_t streaming = tc::crc32_update(0u, a.data(), a.size());
  streaming = tc::crc32_update(streaming, b.data(), b.size());

  REQUIRE(streaming == one_shot);
}

TEST_CASE("wire-size pins — kAtomRecordSize and kTemporalHeaderSize", "[comm][serializer]") {
  // These constants are load-bearing for every downstream backend + the
  // anchor-test. Freeze them with static_assert so accidental edits surface
  // at compile time.
  STATIC_REQUIRE(tc::kAtomRecordSize == 76);
  STATIC_REQUIRE(tc::kTemporalHeaderSize == 86);
  STATIC_REQUIRE(tc::kCrcTrailerSize == 4);

  REQUIRE(tc::temporal_packet_wire_size(0) == 90);
  REQUIRE(tc::temporal_packet_wire_size(1) == 90 + 76);
  REQUIRE(tc::temporal_packet_wire_size(1000) == 90 + 1000 * 76);
}

TEST_CASE("TemporalPacket — empty atom_count=0 roundtrip", "[comm][serializer]") {
  tc::TemporalPacket pkt = make_packet(0, /*seed=*/1);
  const auto bytes = tc::pack_temporal_packet(pkt);
  REQUIRE(bytes.size() == tc::temporal_packet_wire_size(0));

  const auto result = tc::unpack_temporal_packet(bytes);
  REQUIRE(result.ok());
  REQUIRE(result.error.empty());

  const auto& back = *result.packet;
  REQUIRE(back.protocol_version == tc::kCommProtocolVersion);
  REQUIRE(back.zone_id == pkt.zone_id);
  REQUIRE(back.time_level == pkt.time_level);
  REQUIRE(back.version == pkt.version);
  REQUIRE(back.atom_count == 0u);
  REQUIRE(back.certificate_hash == pkt.certificate_hash);
  REQUIRE(back.payload.empty());
}

TEST_CASE("TemporalPacket — roundtrip byte-identical across atom counts", "[comm][serializer]") {
  for (std::uint32_t atom_count : {1u, 10u, 100u, 1000u}) {
    CAPTURE(atom_count);
    const auto seed = 0x1000u + atom_count;
    const auto pkt = make_packet(atom_count, seed);
    const auto bytes_1 = tc::pack_temporal_packet(pkt);

    const auto result = tc::unpack_temporal_packet(bytes_1);
    REQUIRE(result.ok());

    const auto& back = *result.packet;
    REQUIRE(back.protocol_version == tc::kCommProtocolVersion);
    REQUIRE(back.zone_id == pkt.zone_id);
    REQUIRE(back.time_level == pkt.time_level);
    REQUIRE(back.version == pkt.version);
    REQUIRE(back.atom_count == atom_count);
    REQUIRE(back.certificate_hash == pkt.certificate_hash);
    REQUIRE(back.box_snapshot.xlo == pkt.box_snapshot.xlo);
    REQUIRE(back.box_snapshot.xhi == pkt.box_snapshot.xhi);
    REQUIRE(back.box_snapshot.ylo == pkt.box_snapshot.ylo);
    REQUIRE(back.box_snapshot.yhi == pkt.box_snapshot.yhi);
    REQUIRE(back.box_snapshot.zlo == pkt.box_snapshot.zlo);
    REQUIRE(back.box_snapshot.zhi == pkt.box_snapshot.zhi);
    REQUIRE(back.payload == pkt.payload);

    // Second pack of the unpacked packet must yield identical bytes — the
    // strongest statement of byte-stability we can make without comparing
    // to a golden dump.
    const auto bytes_2 = tc::pack_temporal_packet(back);
    REQUIRE(bytes_1 == bytes_2);
  }
}

TEST_CASE("TemporalPacket — CRC mismatch rejected", "[comm][serializer][crc]") {
  auto pkt = make_packet(8, /*seed=*/999);
  auto bytes = tc::pack_temporal_packet(pkt);
  REQUIRE(tc::unpack_temporal_packet(bytes).ok());

  // Flip one byte inside the payload — CRC trailer must now mismatch.
  bytes[tc::kTemporalHeaderSize + 3] ^= 0xA5u;
  const auto result = tc::unpack_temporal_packet(bytes);
  REQUIRE_FALSE(result.ok());
  REQUIRE(result.error.find("CRC32") != std::string::npos);
}

TEST_CASE("TemporalPacket — protocol version mismatch rejected", "[comm][serializer][version]") {
  auto pkt = make_packet(4, /*seed=*/7);
  auto bytes = tc::pack_temporal_packet(pkt);

  // Wire format places protocol_version big-endian at offsets [0..1].
  // Writing 0x0002 (BE) simulates a v2 sender hitting a v1 receiver.
  bytes[0] = 0x00;
  bytes[1] = 0x02;
  const auto result = tc::unpack_temporal_packet(bytes);
  REQUIRE_FALSE(result.ok());
  REQUIRE(result.error.find("protocol version") != std::string::npos);
}

TEST_CASE("TemporalPacket — truncated buffer rejected", "[comm][serializer]") {
  auto pkt = make_packet(4, /*seed=*/3);
  auto bytes = tc::pack_temporal_packet(pkt);

  // Drop the CRC trailer + one payload byte — wire-size mismatch expected.
  bytes.resize(bytes.size() - (tc::kCrcTrailerSize + 1));
  const auto result = tc::unpack_temporal_packet(bytes);
  REQUIRE_FALSE(result.ok());
  REQUIRE_FALSE(result.error.empty());
}

TEST_CASE("TemporalPacket — buffer smaller than header rejected gracefully", "[comm][serializer]") {
  // No out-of-bounds reads on a tiny buffer.
  std::vector<std::uint8_t> tiny(10, 0u);
  const auto result = tc::unpack_temporal_packet(tiny);
  REQUIRE_FALSE(result.ok());
  REQUIRE(result.error.find("buffer too small") != std::string::npos);
}
