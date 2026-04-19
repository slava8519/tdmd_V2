#pragma once

// SPEC: docs/specs/comm/SPEC.md §3.4 (TopologyResolver), §6.4 (HybridBackend)
// Master spec: §10.1-10.2 (topology), §14 M7 (Pattern 2)
// Exec pack: docs/development/m7_execution_pack.md T7.5
//
// Pure, stateless Cartesian-grid topology helper. Given a 3D subdomain grid
// (nx × ny × nz), answers two questions:
//
//   (a) What rank owns subdomain `sd`?
//       M7 assumes D-M7-2 — 1:1 subdomain↔rank binding. `owner_rank(sd)` is
//       simply the identity mapping. Future work (M8+) may split a subdomain
//       across multiple ranks; the API already returns a vector so that
//       extension doesn't change the interface.
//
//   (b) Who are subdomain `sd`'s Moore-neighborhood peers?
//       Up to 26 in 3D, 8 in 2D, 2 in 1D — non-periodic boundaries drop
//       out-of-grid neighbors. Periodic boundaries wrap; de-duplication
//       ensures the returned list doesn't contain `sd` itself or repeats
//       (relevant on tiny wrap-around grids like 2×2 periodic).
//
// Deterministic: the iteration order is fixed (z then y then x), so the same
// grid config always yields the same neighbor vector on every rank. This is
// critical for reproducibility of the halo dispatch pattern.

#include <cstdint>
#include <vector>

namespace tdmd::comm {

struct CartesianGrid {
  int nx{1};
  int ny{1};
  int nz{1};
  bool periodic_x{false};
  bool periodic_y{false};
  bool periodic_z{false};

  [[nodiscard]] int total() const noexcept { return nx * ny * nz; }
};

class TopologyResolver {
public:
  // Build the resolver for the given grid. Constant-time; no work deferred.
  explicit TopologyResolver(CartesianGrid grid);

  // Subdomain → linear index. (ix, iy, iz) must be in [0, n{x,y,z}).
  [[nodiscard]] int subdomain_id(int ix, int iy, int iz) const;

  // Linear subdomain id → grid coordinates. `sd` in [0, total()).
  void coords(int sd, int& ix, int& iy, int& iz) const;

  // D-M7-2: identity mapping between subdomains and ranks. Returned as a
  // vector to leave room for M8+ subdomain sharding; at M7 it's always
  // exactly one element.
  [[nodiscard]] std::vector<int> owner_ranks(int sd) const;

  // Moore-neighborhood peers in grid coordinates. Non-periodic boundaries
  // drop out-of-range neighbors. Periodic boundaries wrap. The returned
  // vector is sorted (ascending by subdomain id) and contains no duplicates
  // or `sd` itself. For a 3D interior subdomain the size is 26.
  [[nodiscard]] std::vector<int> peer_neighbors(int sd) const;

  [[nodiscard]] const CartesianGrid& grid() const noexcept { return grid_; }

private:
  CartesianGrid grid_;
};

}  // namespace tdmd::comm
