#include "tdmd/comm/topology_resolver.hpp"

// SPEC: docs/specs/comm/SPEC.md §3.4 (TopologyResolver), §6.4 (HybridBackend)
// Master spec: §10.1-10.2 (topology), §14 M7 (Pattern 2)
// Exec pack: docs/development/m7_execution_pack.md T7.5

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>

namespace tdmd::comm {

namespace {

// Apply Cartesian wrap on a single axis. Returns -1 if `coord` is out of
// range and the axis is non-periodic (caller skips that neighbor); otherwise
// returns the wrapped coordinate in [0, n).
[[nodiscard]] int wrap_axis(int coord, int n, bool periodic) {
  if (coord >= 0 && coord < n) {
    return coord;
  }
  if (!periodic) {
    return -1;
  }
  // Modular wrap that handles negative `coord` (e.g. -1 → n-1).
  int r = coord % n;
  if (r < 0) {
    r += n;
  }
  return r;
}

void validate_grid(const CartesianGrid& grid) {
  if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0) {
    throw std::invalid_argument("TopologyResolver: grid dimensions must be positive (got " +
                                std::to_string(grid.nx) + "×" + std::to_string(grid.ny) + "×" +
                                std::to_string(grid.nz) + ")");
  }
}

}  // namespace

TopologyResolver::TopologyResolver(CartesianGrid grid) : grid_(grid) {
  validate_grid(grid_);
}

int TopologyResolver::subdomain_id(int ix, int iy, int iz) const {
  if (ix < 0 || ix >= grid_.nx || iy < 0 || iy >= grid_.ny || iz < 0 || iz >= grid_.nz) {
    throw std::out_of_range("TopologyResolver::subdomain_id: coordinates out of range");
  }
  return ix + iy * grid_.nx + iz * grid_.nx * grid_.ny;
}

void TopologyResolver::coords(int sd, int& ix, int& iy, int& iz) const {
  if (sd < 0 || sd >= grid_.total()) {
    throw std::out_of_range("TopologyResolver::coords: subdomain id out of range");
  }
  const int xy = grid_.nx * grid_.ny;
  iz = sd / xy;
  const int rem = sd - iz * xy;
  iy = rem / grid_.nx;
  ix = rem - iy * grid_.nx;
}

std::vector<int> TopologyResolver::owner_ranks(int sd) const {
  if (sd < 0 || sd >= grid_.total()) {
    throw std::out_of_range("TopologyResolver::owner_ranks: subdomain id out of range");
  }
  // D-M7-2: identity mapping — one rank per subdomain in M7.
  return {sd};
}

std::vector<int> TopologyResolver::peer_neighbors(int sd) const {
  int ix = 0;
  int iy = 0;
  int iz = 0;
  coords(sd, ix, iy, iz);

  // std::set provides automatic dedup + sort. The dedup step matters on
  // tiny periodic grids (e.g. 2×2×2 periodic, where the same neighbor can
  // be reached via two different (dx,dy,dz) offsets) and the sort gives
  // deterministic ordering for byte-exact reproducibility (D-M5-12).
  std::set<int> peers;

  for (int dz = -1; dz <= 1; ++dz) {
    const int wz = wrap_axis(iz + dz, grid_.nz, grid_.periodic_z);
    if (wz < 0) {
      continue;
    }
    for (int dy = -1; dy <= 1; ++dy) {
      const int wy = wrap_axis(iy + dy, grid_.ny, grid_.periodic_y);
      if (wy < 0) {
        continue;
      }
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0) {
          continue;
        }
        const int wx = wrap_axis(ix + dx, grid_.nx, grid_.periodic_x);
        if (wx < 0) {
          continue;
        }
        const int peer = wx + wy * grid_.nx + wz * grid_.nx * grid_.ny;
        if (peer != sd) {
          peers.insert(peer);
        }
      }
    }
  }

  return {peers.begin(), peers.end()};
}

}  // namespace tdmd::comm
