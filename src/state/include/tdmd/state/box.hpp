#pragma once

// SPEC: docs/specs/state/SPEC.md §2.2 (Box), §4.2 (wrap), §4.4 (minimum image)
// Exec pack: docs/development/m1_execution_pack.md T1.1
//
// Orthogonal simulation box with per-axis periodic flags. Triclinic tilt
// factors are reserved as fields for forward API stability (M3+) and MUST be
// zero in M1 — construction / validation enforce this.
//
// Units: bounds are in `metal` Å. No unit conversion happens here; callers
// provide already-converted coordinates.

#include <array>
#include <cstdint>

namespace tdmd {

struct Box {
  double xlo = 0.0, xhi = 0.0;
  double ylo = 0.0, yhi = 0.0;
  double zlo = 0.0, zhi = 0.0;

  bool periodic_x = false;
  bool periodic_y = false;
  bool periodic_z = false;

  // Reserved for triclinic support (SPEC §2.2 — "only orthogonal в v1").
  // Must remain zero in M1; any non-zero value is an invariant violation.
  double tilt_xy = 0.0;
  double tilt_xz = 0.0;
  double tilt_yz = 0.0;

  [[nodiscard]] double lx() const noexcept { return xhi - xlo; }
  [[nodiscard]] double ly() const noexcept { return yhi - ylo; }
  [[nodiscard]] double lz() const noexcept { return zhi - zlo; }
  [[nodiscard]] double volume() const noexcept { return lx() * ly() * lz(); }

  // Length along a given axis (0=x, 1=y, 2=z). Debug-bounds-checked.
  [[nodiscard]] double length(int axis) const noexcept;

  // True iff bounds are positive on each axis and triclinic tilts are zero.
  [[nodiscard]] bool is_valid_m1() const noexcept;

  // Wraps (x, y, z) back to the primary image on each periodic axis, updating
  // the per-atom image counters accordingly (SPEC §4.2). On non-periodic axes
  // coordinates pass through unchanged.
  //
  // Idempotent: after one call the point lies in `[xlo, xhi)` on each periodic
  // axis and subsequent calls are no-ops.
  void wrap(double& x,
            double& y,
            double& z,
            std::int32_t& image_x,
            std::int32_t& image_y,
            std::int32_t& image_z) const noexcept;

  // Overload for callers that don't track image counts (e.g. one-off queries).
  void wrap(double& x, double& y, double& z) const noexcept;

  // Minimum-image convention on a separation vector (SPEC §4.4). Non-mutating;
  // takes a Δ on each axis and returns the version with `|Δ| ≤ L/2` on each
  // periodic axis. Called "unwrap_minimum_image" in the exec pack; keep that
  // spelling for contract continuity.
  [[nodiscard]] std::array<double, 3> unwrap_minimum_image(double dx,
                                                           double dy,
                                                           double dz) const noexcept;
};

}  // namespace tdmd
