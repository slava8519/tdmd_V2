// Bench: neighbor-list build — CPU vs GPU at 10⁴ and 10⁵ atoms.
// Exec pack: docs/development/m6_execution_pack.md T6.4 (micro-bench)
// SPEC: docs/specs/gpu/SPEC.md §7.1 + PerfModel calibration baseline for
//       T6.11 (§11.4).
//
// Builds an Al FCC supercell, times `NeighborList::build()` (CPU) vs
// `NeighborListGpu::build()` (GPU, including H2D + host scan + D2H
// offsets copy) — median of 5 repeats after 1 warmup.
//
// Not registered as a CTest — this is a manual local bench. GPU-only.

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/factories.hpp"
#include "tdmd/gpu/gpu_config.hpp"
#include "tdmd/gpu/types.hpp"
#include "tdmd/neighbor/cell_grid.hpp"
#include "tdmd/neighbor/gpu_neighbor_builder.hpp"
#include "tdmd/neighbor/neighbor_list.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

struct Fixture {
  tdmd::AtomSoA atoms;
  tdmd::Box box;
  double cutoff = 6.5;
  double skin = 0.5;
};

// Generates an Al FCC supercell of (Nx, Ny, Nz) unit cells = 4·Nx·Ny·Nz
// atoms. Lattice constant 4.05 Å. Periodic orthogonal box.
Fixture make_fcc(int nx, int ny, int nz) {
  Fixture fx;
  const double a = 4.05;
  fx.box.xlo = fx.box.ylo = fx.box.zlo = 0.0;
  fx.box.xhi = a * nx;
  fx.box.yhi = a * ny;
  fx.box.zhi = a * nz;
  fx.box.periodic_x = fx.box.periodic_y = fx.box.periodic_z = true;

  const double basis[4][3] = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
  fx.atoms.reserve(std::size_t{4} * static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                   static_cast<std::size_t>(nz));
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        for (const auto& b : basis) {
          fx.atoms.add_atom(0U, (ix + b[0]) * a, (iy + b[1]) * a, (iz + b[2]) * a);
        }
      }
    }
  }
  return fx;
}

// Chooses (nx, ny, nz) so 4·nx·ny·nz ≈ target_atoms, cubic shape.
struct GridDims {
  int nx, ny, nz;
};
GridDims pick_dims(std::size_t target_atoms) {
  const double side = std::cbrt(static_cast<double>(target_atoms) / 4.0);
  const int n = std::max(6, static_cast<int>(std::round(side)));
  return {n, n, n};
}

double median(std::vector<double>& v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

double time_cpu(const Fixture& fx, int repeats) {
  using clk = std::chrono::steady_clock;
  std::vector<double> ts;
  tdmd::CellGrid grid;
  grid.build(fx.box, fx.cutoff, fx.skin);
  grid.bin(fx.atoms);
  tdmd::NeighborList nl;
  nl.build(fx.atoms, fx.box, grid, fx.cutoff, fx.skin);  // warmup
  for (int k = 0; k < repeats; ++k) {
    auto t0 = clk::now();
    nl.build(fx.atoms, fx.box, grid, fx.cutoff, fx.skin);
    auto t1 = clk::now();
    ts.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  return median(ts);
}

double time_gpu(const Fixture& fx, int repeats) {
  using clk = std::chrono::steady_clock;
  std::vector<double> ts;

  tdmd::gpu::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 16;
  tdmd::gpu::DevicePool pool(cfg);
  tdmd::gpu::DeviceStream stream = tdmd::gpu::make_stream(cfg.device_id);
  tdmd::neighbor::GpuNeighborBuilder builder(pool, stream);

  tdmd::CellGrid grid;
  grid.build(fx.box, fx.cutoff, fx.skin);
  grid.bin(fx.atoms);
  builder.build(fx.atoms, fx.box, grid, fx.cutoff, fx.skin);  // warmup (alloc + pool prime)

  for (int k = 0; k < repeats; ++k) {
    auto t0 = clk::now();
    builder.build(fx.atoms, fx.box, grid, fx.cutoff, fx.skin);
    auto t1 = clk::now();
    ts.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  return median(ts);
}

}  // namespace

int main() {
  constexpr int kRepeats = 5;
  const std::size_t sizes[] = {10'000, 100'000};

  std::ofstream out("bench_results.txt");
  std::fprintf(stdout, "%-8s %12s %12s %10s\n", "size", "CPU_us", "GPU_us", "speedup");
  out << "# size\tCPU_us\tGPU_us\tspeedup\n";

  for (std::size_t target : sizes) {
    const GridDims d = pick_dims(target);
    const Fixture fx = make_fcc(d.nx, d.ny, d.nz);
    const std::size_t n = fx.atoms.size();

    const double cpu_us = time_cpu(fx, kRepeats);
    double gpu_us = 0.0;
    try {
      gpu_us = time_gpu(fx, kRepeats);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "GPU bench failed at n=%zu: %s\n", n, e.what());
      gpu_us = -1.0;
    }

    const double speedup = gpu_us > 0.0 ? cpu_us / gpu_us : 0.0;
    std::fprintf(stdout, "%-8zu %12.0f %12.0f %9.1fx\n", n, cpu_us, gpu_us, speedup);
    out << n << '\t' << cpu_us << '\t' << gpu_us << '\t' << speedup << '\n';
  }
  return 0;
}
