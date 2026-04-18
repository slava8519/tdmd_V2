#pragma once

// SPEC: docs/specs/zoning/SPEC.md §3.3 (Scheme C — Hilbert3D), §4.2
// Exec pack: docs/development/m3_execution_pack.md T3.5
// Reference: J. Skilling, "Programming the Hilbert curve", AIP Conf. Proc.
// 707, 381 (2004). Transpose-and-Gray-code algorithm, ported in-tree
// (no external library dependency per D-M3-1).
//
// Pure bit-twiddling — 3D Hilbert curve forward/inverse. `bits` is the
// order (bits per dimension); the curve visits [0, 2^bits)³. Callers
// that need non-power-of-2 cubes pad upward and filter.
//
// All routines assume `0 < bits <= 10` (gives 1024³ cube, comfortably
// within uint32_t). Caller is responsible for ensuring inputs fit.

#include <cstdint>

namespace tdmd::zoning::hilbert {

// Forward map: Hilbert index `d` in [0, 2^(3·bits)) → axes (x, y, z)
// each in [0, 2^bits). `d` is the 1D traversal index along the curve;
// consecutive `d` values produce coordinate triples that differ in
// exactly one component by ±1 (locality invariant).
void d_to_xyz(std::uint32_t d,
              int bits,
              std::uint32_t& x,
              std::uint32_t& y,
              std::uint32_t& z) noexcept;

// Inverse: (x, y, z) → Hilbert index. Satisfies
// `xyz_to_d(d_to_xyz(d, b), b) == d` for all d in [0, 2^(3·b)).
std::uint32_t xyz_to_d(std::uint32_t x, std::uint32_t y, std::uint32_t z, int bits) noexcept;

}  // namespace tdmd::zoning::hilbert
