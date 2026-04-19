#pragma once

// SPEC: docs/specs/scheduler/SPEC.md §2.4 (SubdomainGrid in OuterSdCoordinator)
// Master spec: §12.7a
// Exec pack: docs/development/m7_execution_pack.md T7.6
//
// SD-level decomposition descriptor. Owned by `SimulationEngine` (master
// §12.8) in Pattern 2; absent in Pattern 1. Geometry (`subdomain_boxes`) is
// carried verbatim from the canonical interface — needed by the runtime
// halo-pack path, even though the coordinator itself only addresses
// subdomains by id.

#include "tdmd/state/box.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace tdmd::scheduler {

struct SubdomainGrid {
  // (P_space_x, P_space_y, P_space_z). Product == subdomain_boxes.size().
  std::array<std::uint32_t, 3> n_subdomains{1, 1, 1};
  // Per-subdomain orthogonal box. Indexed lexicographically:
  //   id = ix + nx * (iy + ny * iz)
  std::vector<tdmd::Box> subdomain_boxes;
  // MPI rank that owns each subdomain. size == subdomain_boxes.size().
  std::vector<int> rank_of_subdomain;

  [[nodiscard]] std::uint32_t total_subdomains() const noexcept {
    return n_subdomains[0] * n_subdomains[1] * n_subdomains[2];
  }
};

}  // namespace tdmd::scheduler
