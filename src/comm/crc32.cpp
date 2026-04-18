// SPEC: docs/specs/comm/SPEC.md §4.4
// Exec pack: docs/development/m5_execution_pack.md T5.3
//
// Table-driven CRC32 (reflected IEEE polynomial 0xEDB88320) compatible with
// RFC 1952 / zlib. The 256-entry table is built at translation-unit load
// time — constexpr so it lives in .rodata without a runtime initializer.

#include "tdmd/comm/crc32.hpp"

#include <array>

namespace tdmd::comm {

namespace {

constexpr std::uint32_t kPolynomial = 0xEDB88320u;

constexpr std::array<std::uint32_t, 256> build_table() {
  std::array<std::uint32_t, 256> table{};
  for (std::uint32_t i = 0; i < 256; ++i) {
    std::uint32_t c = i;
    for (int k = 0; k < 8; ++k) {
      c = (c & 1u) ? (kPolynomial ^ (c >> 1)) : (c >> 1);
    }
    table[i] = c;
  }
  return table;
}

constexpr auto kTable = build_table();

}  // namespace

std::uint32_t crc32_update(std::uint32_t seed, const void* data, std::size_t len) noexcept {
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  std::uint32_t c = seed ^ 0xFFFFFFFFu;
  for (std::size_t i = 0; i < len; ++i) {
    c = kTable[(c ^ bytes[i]) & 0xFFu] ^ (c >> 8);
  }
  return c ^ 0xFFFFFFFFu;
}

std::uint32_t crc32(const void* data, std::size_t len) noexcept {
  return crc32_update(0u, data, len);
}

}  // namespace tdmd::comm
