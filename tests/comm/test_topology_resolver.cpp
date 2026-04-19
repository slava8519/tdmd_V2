// SPEC: docs/specs/comm/SPEC.md §3.4 (TopologyResolver)
// Master spec: §10.1-10.2 (topology), §14 M7
// Exec pack: docs/development/m7_execution_pack.md T7.5
//
// Pure-C++ unit tests for TopologyResolver — no MPI, runs in the always-built
// `test_comm` binary. Covers:
//   - 1D/2D/3D index round-trip (subdomain_id ↔ coords)
//   - Moore neighborhood interior counts (2 in 1D, 8 in 2D, 26 in 3D)
//   - Boundary subdomains: non-periodic drops out-of-grid neighbors;
//     periodic wraps and dedups on tiny grids
//   - D-M7-2: owner_ranks(sd) is identity { sd }
//   - Determinism: same grid → same neighbor vector across constructions
//   - Single-subdomain Pattern 1: peer_neighbors returns empty (no halos)

#include "tdmd/comm/topology_resolver.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

namespace tc = tdmd::comm;

namespace {

[[nodiscard]] bool sorted_unique(const std::vector<int>& v) {
  for (std::size_t i = 1; i < v.size(); ++i) {
    if (v[i] <= v[i - 1]) {
      return false;
    }
  }
  return true;
}

}  // namespace

TEST_CASE("TopologyResolver — constructor rejects zero/negative dims", "[comm][topology]") {
  // Extra parens wrap brace-init so its commas don't get parsed as macro args.
  REQUIRE_THROWS_AS((tc::TopologyResolver{tc::CartesianGrid{0, 1, 1}}), std::invalid_argument);
  REQUIRE_THROWS_AS((tc::TopologyResolver{tc::CartesianGrid{1, -1, 1}}), std::invalid_argument);
  REQUIRE_NOTHROW((tc::TopologyResolver{tc::CartesianGrid{1, 1, 1}}));
}

TEST_CASE("TopologyResolver — index round-trip 3D 4×3×2", "[comm][topology]") {
  tc::TopologyResolver topo{tc::CartesianGrid{4, 3, 2}};
  REQUIRE(topo.grid().total() == 24);
  for (int sd = 0; sd < topo.grid().total(); ++sd) {
    int ix = -1;
    int iy = -1;
    int iz = -1;
    topo.coords(sd, ix, iy, iz);
    REQUIRE(topo.subdomain_id(ix, iy, iz) == sd);
  }
}

TEST_CASE("TopologyResolver — out-of-range queries throw", "[comm][topology]") {
  tc::TopologyResolver topo{tc::CartesianGrid{2, 2, 2}};
  REQUIRE_THROWS_AS(topo.subdomain_id(2, 0, 0), std::out_of_range);
  int ix = 0;
  int iy = 0;
  int iz = 0;
  REQUIRE_THROWS_AS(topo.coords(8, ix, iy, iz), std::out_of_range);
  REQUIRE_THROWS_AS(topo.peer_neighbors(8), std::out_of_range);
  REQUIRE_THROWS_AS(topo.owner_ranks(-1), std::out_of_range);
}

TEST_CASE("TopologyResolver — D-M7-2 owner_ranks identity mapping", "[comm][topology]") {
  tc::TopologyResolver topo{tc::CartesianGrid{3, 2, 2}};
  for (int sd = 0; sd < topo.grid().total(); ++sd) {
    const auto owners = topo.owner_ranks(sd);
    REQUIRE(owners.size() == 1);
    REQUIRE(owners[0] == sd);
  }
}

TEST_CASE("TopologyResolver — single-subdomain Pattern 1 has no peers", "[comm][topology]") {
  tc::TopologyResolver topo{tc::CartesianGrid{1, 1, 1, /*px=*/true, /*py=*/true, /*pz=*/true}};
  // Even with periodic wrap, a 1×1×1 grid has no peers — every offset wraps
  // back onto sd itself, which is filtered out.
  REQUIRE(topo.peer_neighbors(0).empty());
}

TEST_CASE("TopologyResolver — 1D non-periodic boundary drops out-of-grid", "[comm][topology]") {
  tc::TopologyResolver topo{tc::CartesianGrid{4, 1, 1}};
  // Interior subdomain (id=1) sees both neighbors (0 and 2).
  REQUIRE(topo.peer_neighbors(1) == std::vector<int>{0, 2});
  // Boundary subdomain (id=0) sees only one neighbor (id=1).
  REQUIRE(topo.peer_neighbors(0) == std::vector<int>{1});
  REQUIRE(topo.peer_neighbors(3) == std::vector<int>{2});
}

TEST_CASE("TopologyResolver — 1D periodic wrap", "[comm][topology]") {
  tc::TopologyResolver topo{tc::CartesianGrid{4, 1, 1, /*px=*/true}};
  REQUIRE(topo.peer_neighbors(0) == std::vector<int>{1, 3});
  REQUIRE(topo.peer_neighbors(3) == std::vector<int>{0, 2});
}

TEST_CASE("TopologyResolver — 2D 3×3 non-periodic boundary counts", "[comm][topology]") {
  tc::TopologyResolver topo{tc::CartesianGrid{3, 3, 1}};
  // Interior (center, id=4) — 8 neighbors in 2D Moore.
  REQUIRE(topo.peer_neighbors(4).size() == 8);
  // Corner (id=0 = (0,0,0)) — 3 neighbors: (1,0), (0,1), (1,1).
  REQUIRE(topo.peer_neighbors(0).size() == 3);
  // Edge (id=1 = (1,0,0)) — 5 neighbors.
  REQUIRE(topo.peer_neighbors(1).size() == 5);
}

TEST_CASE("TopologyResolver — 3D interior subdomain has 26 neighbors", "[comm][topology]") {
  tc::TopologyResolver topo{tc::CartesianGrid{3, 3, 3}};
  // Center subdomain (1,1,1) → id = 1 + 1*3 + 1*9 = 13.
  const int center = topo.subdomain_id(1, 1, 1);
  REQUIRE(center == 13);
  const auto peers = topo.peer_neighbors(center);
  REQUIRE(peers.size() == 26);
  REQUIRE(sorted_unique(peers));
  // None of the peers is `center` itself.
  REQUIRE(std::find(peers.begin(), peers.end(), center) == peers.end());
}

TEST_CASE("TopologyResolver — 3D corner non-periodic has 7 neighbors", "[comm][topology]") {
  tc::TopologyResolver topo{tc::CartesianGrid{3, 3, 3}};
  // (0,0,0) corner — 7 neighbors (the 2×2×2 sub-cube minus self).
  const auto peers = topo.peer_neighbors(0);
  REQUIRE(peers.size() == 7);
  REQUIRE(sorted_unique(peers));
}

TEST_CASE("TopologyResolver — 2×2×2 fully periodic dedups to 7 unique peers", "[comm][topology]") {
  tc::TopologyResolver topo{tc::CartesianGrid{2, 2, 2, true, true, true}};
  // On 2×2×2 every Moore offset wraps to one of the other 7 subdomains. A
  // naive enumeration would yield 26 entries with duplicates; the resolver
  // must dedup down to 7 and never include `sd` itself.
  for (int sd = 0; sd < topo.grid().total(); ++sd) {
    const auto peers = topo.peer_neighbors(sd);
    REQUIRE(peers.size() == 7);
    REQUIRE(sorted_unique(peers));
    REQUIRE(std::find(peers.begin(), peers.end(), sd) == peers.end());
  }
}

TEST_CASE("TopologyResolver — deterministic ordering across re-constructions", "[comm][topology]") {
  tc::CartesianGrid g{3, 3, 3, true, true, true};
  tc::TopologyResolver topo_a{g};
  tc::TopologyResolver topo_b{g};
  for (int sd = 0; sd < g.total(); ++sd) {
    REQUIRE(topo_a.peer_neighbors(sd) == topo_b.peer_neighbors(sd));
  }
}
