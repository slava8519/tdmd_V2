// Exec pack: docs/development/m6_execution_pack.md T6.4
// SPEC: docs/specs/gpu/SPEC.md §7.1, §6.3 (D-M6-7 CPU↔GPU bit-exact)
// SPEC: docs/specs/neighbor/SPEC.md §2.1, §4
//
// T6.4 acceptance gate:
//   Build a half-neighbor-list on GPU for an Al FCC fixture, D2H-copy it,
//   then compare byte-for-byte against the CPU `NeighborList` built from
//   the same AtomSoA + CellGrid. `std::memcmp` over page_offsets, ids, and
//   r2 — matches D-M6-7 contract (IEEE754 byte equality, not `<epsilon`).
//
// Each case skips when CUDA is unavailable so the suite stays green on
// CPU-only builds + GPU-less CI runners.

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/factories.hpp"
#include "tdmd/gpu/gpu_config.hpp"
#include "tdmd/gpu/neighbor_list_gpu.hpp"
#include "tdmd/gpu/types.hpp"
#include "tdmd/neighbor/cell_grid.hpp"
#include "tdmd/neighbor/gpu_neighbor_builder.hpp"
#include "tdmd/neighbor/neighbor_list.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#if TDMD_BUILD_CUDA
#include <cuda_runtime.h>
#endif

namespace tg = tdmd::gpu;
namespace tn = tdmd::neighbor;

TEST_CASE("NeighborListGpu — CPU-only build throws", "[gpu][nl][cpu]") {
#if TDMD_BUILD_CUDA
  SUCCEED("CUDA build — CPU stub path exercised only on TDMD_BUILD_CUDA=OFF");
#else
  // Under CPU-only, DevicePool construction throws before we ever reach
  // the NL builder. A single catch is enough to confirm the stub path
  // isn't silently callable.
  tg::GpuConfig cfg;
  REQUIRE_THROWS_AS(tg::DevicePool(cfg), std::runtime_error);
#endif
}

#if TDMD_BUILD_CUDA

namespace {

bool cuda_device_available() noexcept {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  return err == cudaSuccess && count > 0;
}

// Builds a 6×6×6 Al FCC supercell (864 atoms, a=4.05 Å) inside a periodic
// orthogonal box. Chosen as the smallest FCC supercell that still satisfies
// the `≥3 cells per axis` CellGrid invariant at cutoff + skin = 7.0 Å.
struct AlFccFixture {
  tdmd::AtomSoA atoms;
  tdmd::Box box;
  static constexpr double kLatticeA = 4.05;
  static constexpr int kNx = 6;
  static constexpr int kNy = 6;
  static constexpr int kNz = 6;
  static constexpr double kCutoff = 6.5;
  static constexpr double kSkin = 0.5;
};

AlFccFixture make_al_fcc_fixture() {
  AlFccFixture fx;
  const double a = AlFccFixture::kLatticeA;
  fx.box.xlo = 0.0;
  fx.box.ylo = 0.0;
  fx.box.zlo = 0.0;
  fx.box.xhi = a * AlFccFixture::kNx;
  fx.box.yhi = a * AlFccFixture::kNy;
  fx.box.zhi = a * AlFccFixture::kNz;
  fx.box.periodic_x = fx.box.periodic_y = fx.box.periodic_z = true;

  // 4-atom FCC basis: (0,0,0), (a/2,a/2,0), (a/2,0,a/2), (0,a/2,a/2).
  const double basis[4][3] = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
  for (int iz = 0; iz < AlFccFixture::kNz; ++iz) {
    for (int iy = 0; iy < AlFccFixture::kNy; ++iy) {
      for (int ix = 0; ix < AlFccFixture::kNx; ++ix) {
        for (const auto& b : basis) {
          const double x = (ix + b[0]) * a;
          const double y = (iy + b[1]) * a;
          const double z = (iz + b[2]) * a;
          fx.atoms.add_atom(0U, x, y, z);
        }
      }
    }
  }
  return fx;
}

}  // namespace

TEST_CASE("NeighborListGpu — 864-atom Al FCC matches CPU bit-exact", "[gpu][nl]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }

  AlFccFixture fx = make_al_fcc_fixture();
  REQUIRE(fx.atoms.size() == 864u);

  tdmd::CellGrid grid;
  grid.build(fx.box, AlFccFixture::kCutoff, AlFccFixture::kSkin);
  grid.bin(fx.atoms);

  // --- CPU reference ---
  tdmd::NeighborList cpu_nl;
  cpu_nl.build(fx.atoms, fx.box, grid, AlFccFixture::kCutoff, AlFccFixture::kSkin);
  REQUIRE(cpu_nl.atom_count() == 864u);
  REQUIRE(cpu_nl.pair_count() > 0u);

  // --- GPU path via adapter ---
  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);

  tn::GpuNeighborBuilder gpu_builder(pool, stream);
  gpu_builder.build(fx.atoms, fx.box, grid, AlFccFixture::kCutoff, AlFccFixture::kSkin);

  const auto& gpu_nl = gpu_builder.neighbor_list();
  REQUIRE(gpu_nl.atom_count() == 864u);
  REQUIRE(gpu_nl.pair_count() == cpu_nl.pair_count());

  tg::NeighborListHostSnapshot snap = gpu_nl.download(stream);
  REQUIRE(snap.offsets.size() == cpu_nl.page_offsets().size());
  REQUIRE(snap.ids.size() == cpu_nl.neigh_ids().size());
  REQUIRE(snap.r2.size() == cpu_nl.neigh_r2().size());

  // D-M6-7: byte-equality, not `abs(a-b) < epsilon`. A single differing
  // byte fails the gate. Integer CSR (offsets, ids) is bit-exact across all
  // flavors because atom→cell binning uses only integer math. The r² FP
  // array is bit-exact only under Fp64ReferenceBuild (no FMA contraction);
  // non-Reference flavors run with --fmad=true and drift by a few ULPs.
  REQUIRE(std::memcmp(snap.offsets.data(),
                      cpu_nl.page_offsets().data(),
                      snap.offsets.size() * sizeof(std::uint64_t)) == 0);
  REQUIRE(std::memcmp(snap.ids.data(),
                      cpu_nl.neigh_ids().data(),
                      snap.ids.size() * sizeof(std::uint32_t)) == 0);
#ifdef TDMD_FLAVOR_FP64_REFERENCE
  REQUIRE(std::memcmp(snap.r2.data(), cpu_nl.neigh_r2().data(), snap.r2.size() * sizeof(double)) ==
          0);
#else
  // Non-Reference: assert r² agrees to within FMA-contraction drift (~few
  // ULPs per pair); D-M6-8-style loose check rather than D-M6-7 byte-equal.
  for (std::size_t k = 0; k < snap.r2.size(); ++k) {
    const double num = std::abs(snap.r2[k] - cpu_nl.neigh_r2()[k]);
    const double den = std::max(1.0, std::max(snap.r2[k], cpu_nl.neigh_r2()[k]));
    REQUIRE(num / den <= 1e-14);
  }
#endif
}

TEST_CASE("NeighborListGpu — rebuild increments build_version and stays bit-exact", "[gpu][nl]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }

  AlFccFixture fx = make_al_fcc_fixture();
  tdmd::CellGrid grid;
  grid.build(fx.box, AlFccFixture::kCutoff, AlFccFixture::kSkin);
  grid.bin(fx.atoms);

  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);

  tn::GpuNeighborBuilder builder(pool, stream);
  builder.build(fx.atoms, fx.box, grid, AlFccFixture::kCutoff, AlFccFixture::kSkin);
  const std::uint64_t v1 = builder.neighbor_list().build_version();
  const std::size_t pc1 = builder.neighbor_list().pair_count();

  // Second build on same inputs — version increments, pair count stable.
  builder.build(fx.atoms, fx.box, grid, AlFccFixture::kCutoff, AlFccFixture::kSkin);
  REQUIRE(builder.neighbor_list().build_version() == v1 + 1);
  REQUIRE(builder.neighbor_list().pair_count() == pc1);

  tdmd::NeighborList cpu_nl;
  cpu_nl.build(fx.atoms, fx.box, grid, AlFccFixture::kCutoff, AlFccFixture::kSkin);
  auto snap = builder.neighbor_list().download(stream);
  // Integer CSR ids are bit-exact across all flavors (atom binning is pure
  // integer math; no FMA contraction affects it).
  REQUIRE(std::memcmp(snap.ids.data(),
                      cpu_nl.neigh_ids().data(),
                      snap.ids.size() * sizeof(std::uint32_t)) == 0);
}

TEST_CASE("NeighborListGpu — empty atoms yields empty CSR", "[gpu][nl][edge]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }

  // Need a non-degenerate box for CellGrid::build but zero atoms.
  tdmd::AtomSoA empty;
  tdmd::Box box;
  box.xlo = box.ylo = box.zlo = 0.0;
  box.xhi = box.yhi = box.zhi = 24.3;
  box.periodic_x = box.periodic_y = box.periodic_z = true;

  tdmd::CellGrid grid;
  grid.build(box, 6.5, 0.5);
  grid.bin(empty);

  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);

  tn::GpuNeighborBuilder builder(pool, stream);
  builder.build(empty, box, grid, 6.5, 0.5);
  REQUIRE(builder.neighbor_list().atom_count() == 0u);
  REQUIRE(builder.neighbor_list().pair_count() == 0u);
  auto snap = builder.neighbor_list().download(stream);
  REQUIRE(snap.offsets.empty());
  REQUIRE(snap.ids.empty());
  REQUIRE(snap.r2.empty());
}

#endif  // TDMD_BUILD_CUDA
