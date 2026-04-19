// Bench: EAM/alloy force compute — CPU vs GPU at 10⁴ and 10⁵ atoms.
// Exec pack: docs/development/m6_execution_pack.md T6.5 (micro-bench)
// SPEC: docs/specs/gpu/SPEC.md §7.2 + PerfModel calibration baseline for
//       T6.11 (§11.4).
//
// Single-species Al FCC with the test-fixture `Al_small.eam.alloy` tables —
// keeps the bench self-contained (no dependency on the large Mishin file).
// Times the compute() call on a pre-binned CellGrid, NL for CPU, + adapter
// translation + H2D + kernels + D2H for GPU. Median of 5 after 1 warmup.
//
// Not a CTest — manual local bench. GPU-only; compiles iff TDMD_BUILD_CUDA.

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/factories.hpp"
#include "tdmd/gpu/gpu_config.hpp"
#include "tdmd/gpu/types.hpp"
#include "tdmd/neighbor/cell_grid.hpp"
#include "tdmd/neighbor/neighbor_list.hpp"
#include "tdmd/potentials/eam_alloy.hpp"
#include "tdmd/potentials/eam_alloy_gpu_adapter.hpp"
#include "tdmd/potentials/eam_file.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef TDMD_BENCH_FIXTURES_DIR
#error "TDMD_BENCH_FIXTURES_DIR must be defined by the build system"
#endif

namespace {

struct Fixture {
  tdmd::AtomSoA atoms;
  tdmd::Box box;
  double cutoff = 5.0;
  double skin = 0.2;
};

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

void zero_forces(tdmd::AtomSoA& a) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    a.fx[i] = a.fy[i] = a.fz[i] = 0.0;
  }
}

double time_cpu(const Fixture& fx, tdmd::potentials::EamAlloyData data, int repeats) {
  using clk = std::chrono::steady_clock;
  std::vector<double> ts;
  tdmd::CellGrid grid;
  grid.build(fx.box, fx.cutoff, fx.skin);
  tdmd::AtomSoA atoms = fx.atoms;
  grid.bin(atoms);
  tdmd::NeighborList nl;
  nl.build(atoms, fx.box, grid, fx.cutoff, fx.skin);
  tdmd::EamAlloyPotential pot(std::move(data));
  zero_forces(atoms);
  (void) pot.compute(atoms, nl, fx.box);  // warmup
  for (int k = 0; k < repeats; ++k) {
    zero_forces(atoms);
    auto t0 = clk::now();
    (void) pot.compute(atoms, nl, fx.box);
    auto t1 = clk::now();
    ts.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  return median(ts);
}

double time_gpu(const Fixture& fx, const tdmd::potentials::EamAlloyData& data, int repeats) {
  using clk = std::chrono::steady_clock;
  std::vector<double> ts;

  tdmd::gpu::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 16;
  tdmd::gpu::DevicePool pool(cfg);
  tdmd::gpu::DeviceStream stream = tdmd::gpu::make_stream(cfg.device_id);
  tdmd::potentials::EamAlloyGpuAdapter adapter(data);

  tdmd::CellGrid grid;
  grid.build(fx.box, fx.cutoff, fx.skin);
  tdmd::AtomSoA atoms = fx.atoms;
  grid.bin(atoms);
  zero_forces(atoms);
  (void) adapter.compute(atoms, fx.box, grid, pool, stream);  // warmup

  for (int k = 0; k < repeats; ++k) {
    zero_forces(atoms);
    auto t0 = clk::now();
    (void) adapter.compute(atoms, fx.box, grid, pool, stream);
    auto t1 = clk::now();
    ts.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  return median(ts);
}

}  // namespace

int main() {
  const std::string pot_path =
      (std::filesystem::path(TDMD_BENCH_FIXTURES_DIR) / "Al_small.eam.alloy").string();
  const auto data = tdmd::potentials::parse_eam_alloy(pot_path);

  constexpr int kRepeats = 5;
  const std::size_t sizes[] = {10'000, 100'000};

  std::ofstream out("bench_results.txt");
  std::fprintf(stdout, "%-8s %12s %12s %10s\n", "size", "CPU_us", "GPU_us", "speedup");
  out << "# size\tCPU_us\tGPU_us\tspeedup\n";

  for (std::size_t target : sizes) {
    const GridDims d = pick_dims(target);
    Fixture fx = make_fcc(d.nx, d.ny, d.nz);
    fx.cutoff = data.cutoff;
    fx.skin = 0.2;
    const std::size_t n = fx.atoms.size();

    double cpu_us = 0.0;
    double gpu_us = 0.0;
    try {
      cpu_us = time_cpu(fx, data, kRepeats);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "CPU bench failed at n=%zu: %s\n", n, e.what());
      cpu_us = -1.0;
    }
    try {
      gpu_us = time_gpu(fx, data, kRepeats);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "GPU bench failed at n=%zu: %s\n", n, e.what());
      gpu_us = -1.0;
    }

    const double speedup = (cpu_us > 0.0 && gpu_us > 0.0) ? cpu_us / gpu_us : 0.0;
    std::fprintf(stdout, "%-8zu %12.0f %12.0f %9.1fx\n", n, cpu_us, gpu_us, speedup);
    out << n << '\t' << cpu_us << '\t' << gpu_us << '\t' << speedup << '\n';
  }
  return 0;
}
