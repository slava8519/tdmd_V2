#pragma once

// SPEC: docs/specs/state/SPEC.md §2.1 (AtomSoA), §4 (PBC/images), §9 (add/remove)
// Exec pack: docs/development/m1_execution_pack.md T1.1
//
// `AtomSoA` — the single owner of per-atom state on a rank (master spec §8.2).
// Fields are parallel vectors (Structure-of-Arrays) backed by 64-byte aligned
// allocations for future SIMD/GPU paths. Every vector has identical size().
//
// In M1 positions/velocities/forces live in `metal` units (Å, Å/ps, eV/Å). Unit
// conversion happens at IO boundaries via `UnitConverter` (T1.2); this module
// never performs unit math.

#include "tdmd/state/aligned_allocator.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tdmd {

using AtomId = std::uint64_t;
using SpeciesId = std::uint32_t;
using Version = std::uint64_t;

inline constexpr std::size_t kSoaAlignment = 64;

template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T, kSoaAlignment>>;

// Parameters for a single atom insertion. Velocities default to zero; forces
// and image counters are always zero-initialized by `add_atom`.
struct AtomInit {
  SpeciesId type = 0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double vx = 0.0;
  double vy = 0.0;
  double vz = 0.0;
};

// Structure-of-Arrays atom storage for a single MPI rank.
//
// Invariants (enforced by every mutator):
// - `id.size() == type.size() == x.size() == ... == flags.size()` (all fields
//   share the same logical length);
// - `&field[0] % kSoaAlignment == 0` whenever `size() > 0` (allocator-level
//   guarantee; exercised by the alignment tests);
// - AtomId monotonically increasing per instance, never reused after removal.
//
// Ownership (master spec §8.2): only `AtomSoA` (and, by extension,
// `StateManager` in T1.9) mutates these fields. Other modules hold read-only
// views.
struct AtomSoA {
  AlignedVector<AtomId> id;
  AlignedVector<SpeciesId> type;

  AlignedVector<double> x, y, z;
  AlignedVector<double> vx, vy, vz;
  AlignedVector<double> fx, fy, fz;

  AlignedVector<std::int32_t> image_x, image_y, image_z;

  AlignedVector<std::uint32_t> flags;

  [[nodiscard]] std::size_t size() const noexcept { return id.size(); }
  [[nodiscard]] bool empty() const noexcept { return id.empty(); }
  [[nodiscard]] std::size_t capacity() const noexcept { return id.capacity(); }

  // Pre-reserves storage across all SoA fields. Useful before bulk import to
  // avoid incremental reallocation.
  void reserve(std::size_t new_capacity);

  // Grows or shrinks all SoA fields to exactly `new_size`. Newly created atoms
  // are zero-initialized (including `id`, which must be rewritten by the
  // caller if real IDs are required; prefer `add_atom` for normal inserts).
  void resize(std::size_t new_size);

  // Drops all atoms; capacity is preserved (use `shrink_to_fit()` to release).
  void clear() noexcept;

  // Appends one atom. Forces and image counters are set to zero. Returns the
  // freshly minted AtomId.
  AtomId add_atom(const AtomInit& init);

  // Convenience overload covering the most common call site.
  AtomId add_atom(SpeciesId type,
                  double x,
                  double y,
                  double z,
                  double vx = 0.0,
                  double vy = 0.0,
                  double vz = 0.0);

  // Swap-and-pop removal (SPEC §9.2): at most one other atom's index changes
  // (the previous last atom moves into `atom_idx`). Bounds-checked in debug
  // builds via assert.
  void remove_atom(std::size_t atom_idx);

  // Checks all SoA vectors agree on size(). Intended for debug assertions and
  // property tests.
  [[nodiscard]] bool invariants_hold() const noexcept;

private:
  AtomId next_id_ = 1;  // 0 is reserved to mean "invalid / uninitialized".

  void resize_all_fields(std::size_t new_size);
  void reserve_all_fields(std::size_t new_capacity);
};

}  // namespace tdmd
