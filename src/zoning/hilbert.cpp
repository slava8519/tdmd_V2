// SPEC: docs/specs/zoning/SPEC.md §3.3
// Exec pack: docs/development/m3_execution_pack.md T3.5
// Reference: Skilling 2004 (AIP Conf. Proc. 707, 381). The transpose-
// based algorithm. See the module header for the contract.

#include "tdmd/zoning/hilbert.hpp"

#include <array>
#include <cstdint>

namespace tdmd::zoning::hilbert {

namespace {

// Skilling's AxestoTranspose — operates on X[0..n-1] in-place.
// `bits` = dimension bit-width; `M = 1 << (bits-1)` is the high-order
// mask. After this call, X holds the *transposed* Hilbert representation
// where bit j of the Hilbert index is X[j % n] bit (j / n) — ready for
// bit-interleaving into a single scalar `d`.
void axes_to_transpose(std::array<std::uint32_t, 3>& X, int bits) noexcept {
  const std::uint32_t M = 1u << (bits - 1);
  // Inverse-undo pass: rotate / flip sub-cubes to express the coordinate
  // triple as the Gray-coded transpose of the Hilbert index.
  for (std::uint32_t Q = M; Q > 1; Q >>= 1) {
    const std::uint32_t P = Q - 1;
    for (std::size_t i = 0; i < 3; ++i) {
      if (X[i] & Q) {
        X[0] ^= P;
      } else {
        const std::uint32_t t = (X[0] ^ X[i]) & P;
        X[0] ^= t;
        X[i] ^= t;
      }
    }
  }
  // Gray-encode the columns so the bit-interleaving step below yields
  // the monotonic Hilbert index.
  for (std::size_t i = 1; i < 3; ++i) {
    X[i] ^= X[i - 1];
  }
  std::uint32_t t = 0;
  for (std::uint32_t Q = M; Q > 1; Q >>= 1) {
    if (X[2] & Q) {
      t ^= Q - 1;
    }
  }
  for (std::size_t i = 0; i < 3; ++i) {
    X[i] ^= t;
  }
}

// Inverse — takes transposed representation back to axes.
void transpose_to_axes(std::array<std::uint32_t, 3>& X, int bits) noexcept {
  const std::uint32_t N = 2u << (bits - 1);  // 1 << bits
  // Gray-decode
  const std::uint32_t t = X[2] >> 1;
  for (std::size_t i = 2; i > 0; --i) {
    X[i] ^= X[i - 1];
  }
  X[0] ^= t;
  // Undo the excess-work pass in reverse order of axes_to_transpose.
  for (std::uint32_t Q = 2; Q != N; Q <<= 1) {
    const std::uint32_t P = Q - 1;
    for (std::size_t i = 3; i-- > 0;) {
      if (X[i] & Q) {
        X[0] ^= P;
      } else {
        const std::uint32_t t2 = (X[0] ^ X[i]) & P;
        X[0] ^= t2;
        X[i] ^= t2;
      }
    }
  }
}

// Pack transposed X[0..2] into a single Hilbert index `d`. Bit j of `d`
// (0 = MSB) is taken from X[j % 3] bit (bits - 1 - j / 3).
std::uint32_t transpose_to_index(const std::array<std::uint32_t, 3>& X, int bits) noexcept {
  std::uint32_t d = 0;
  for (int i = 0; i < bits; ++i) {
    for (int dim = 0; dim < 3; ++dim) {
      const std::uint32_t src_mask = 1u << (bits - 1 - i);
      if (X[static_cast<std::size_t>(dim)] & src_mask) {
        const int out_bit = bits * 3 - 1 - (i * 3 + dim);
        d |= 1u << out_bit;
      }
    }
  }
  return d;
}

// Unpack single scalar Hilbert index `d` into transposed form X.
void index_to_transpose(std::uint32_t d, int bits, std::array<std::uint32_t, 3>& X) noexcept {
  X = {0u, 0u, 0u};
  for (int i = 0; i < bits * 3; ++i) {
    const std::uint32_t src_mask = 1u << (bits * 3 - 1 - i);
    if (d & src_mask) {
      const int dim = i % 3;
      const int out_bit = bits - 1 - (i / 3);
      X[static_cast<std::size_t>(dim)] |= 1u << out_bit;
    }
  }
}

}  // namespace

void d_to_xyz(std::uint32_t d,
              int bits,
              std::uint32_t& x,
              std::uint32_t& y,
              std::uint32_t& z) noexcept {
  std::array<std::uint32_t, 3> X{};
  index_to_transpose(d, bits, X);
  transpose_to_axes(X, bits);
  x = X[0];
  y = X[1];
  z = X[2];
}

std::uint32_t xyz_to_d(std::uint32_t x, std::uint32_t y, std::uint32_t z, int bits) noexcept {
  std::array<std::uint32_t, 3> X{x, y, z};
  axes_to_transpose(X, bits);
  return transpose_to_index(X, bits);
}

}  // namespace tdmd::zoning::hilbert
