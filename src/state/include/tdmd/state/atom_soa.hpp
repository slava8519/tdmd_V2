#pragma once

// SPEC: docs/specs/state/SPEC.md
//
// AtomSoA holds per-atom Structure-of-Arrays data for a local MPI rank.
// Full definition (fields, layout invariants, ownership rules) lives in
// state/SPEC.md §2.1 and §4; this file is a skeleton for M0.

#include <cstddef>

namespace tdmd {

struct AtomSoA {
  // TODO(M1): fields per state/SPEC.md §2.1 — id, type, position x/y/z,
  // velocity vx/vy/vz, force fx/fy/fz, ghost/real flags, migration state.

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

private:
  std::size_t size_ = 0;
};

}  // namespace tdmd
