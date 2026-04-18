#pragma once

// SPEC: docs/specs/comm/SPEC.md §4.4 (CRC32 policy)
// Exec pack: docs/development/m5_execution_pack.md T5.3
//
// RFC 1952 / zlib-compatible CRC32. Reflected polynomial 0xEDB88320.
// Table-driven 8-bit lookup — ~8 KiB code footprint, no external deps.
//
// Known test vector anchored by the M5 acceptance gate (D-M5-11):
//   crc32("123456789") == 0xCBF43926.
// Any change to the polynomial or reflection scheme that breaks this
// vector breaks wire compatibility with every network stack TDMD talks to.

#include <cstddef>
#include <cstdint>

namespace tdmd::comm {

// One-shot CRC32 over `len` bytes at `data`.
//
// For streaming use, initialise `seed = 0` and chain multiple calls via the
// three-argument overload; the second form XORs the current accumulator in
// before processing and XORs back out at the end (matches zlib `crc32()`).
std::uint32_t crc32(const void* data, std::size_t len) noexcept;

// Streaming variant: callers pre-seed with 0 for the first chunk, pass the
// returned value into the next call. Matches the zlib incremental API.
std::uint32_t crc32_update(std::uint32_t seed, const void* data, std::size_t len) noexcept;

}  // namespace tdmd::comm
