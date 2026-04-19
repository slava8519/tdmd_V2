// Bench: VV NVE integrator — CPU vs GPU half-kick + drift at 10⁴ / 10⁵ atoms.
// Exec pack: docs/development/m6_execution_pack.md T6.6 (micro-bench)
// SPEC: docs/specs/gpu/SPEC.md §7.3 + PerfModel calibration baseline for
//       T6.11 (§11.4).
//
// Standard NVE step uses injected synthetic forces so the timing isolates the
// integrator kernel (no potential dependency). Median of 5 after 1 warmup.
//
// Not a CTest — manual local bench. GPU-only; compiles iff TDMD_BUILD_CUDA.

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/factories.hpp"
#include "tdmd/gpu/gpu_config.hpp"
#include "tdmd/gpu/types.hpp"
#include "tdmd/integrator/gpu_velocity_verlet.hpp"
#include "tdmd/integrator/velocity_verlet.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/species.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

namespace {

struct Fixture {
  tdmd::AtomSoA atoms;
  tdmd::SpeciesRegistry species;
};

Fixture make_lattice(std::size_t target_atoms) {
  Fixture fx;
  tdmd::SpeciesInfo info;
  info.name = "Al";
  info.mass = 26.98;
  info.atomic_number = 13;
  fx.species.register_species(info);

  const double side = std::cbrt(static_cast<double>(target_atoms));
  const std::size_t n = std::max<std::size_t>(8, static_cast<std::size_t>(std::round(side)));
  const double a = 3.0;
  fx.atoms.reserve(n * n * n);
  for (std::size_t iz = 0; iz < n; ++iz) {
    for (std::size_t iy = 0; iy < n; ++iy) {
      for (std::size_t ix = 0; ix < n; ++ix) {
        fx.atoms.add_atom(0U, a * ix, a * iy, a * iz);
      }
    }
  }
  for (std::size_t i = 0; i < fx.atoms.size(); ++i) {
    fx.atoms.vx[i] = 0.01 * std::sin(0.37 * (i + 1));
    fx.atoms.vy[i] = 0.01 * std::cos(0.19 * (i + 1));
    fx.atoms.vz[i] = 0.01 * std::sin(0.83 * (i + 1) + 1.1);
    fx.atoms.fx[i] = std::sin(0.11 * (i + 1));
    fx.atoms.fy[i] = std::cos(0.29 * (i + 1));
    fx.atoms.fz[i] = std::sin(0.53 * (i + 1));
  }
  return fx;
}

double median(std::vector<double>& v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

double time_cpu(const Fixture& fx, int repeats) {
  using clk = std::chrono::steady_clock;
  std::vector<double> ts;
  ts.reserve(repeats);
  tdmd::AtomSoA atoms = fx.atoms;
  tdmd::VelocityVerletIntegrator vv;
  const double dt = 0.001;

  vv.pre_force_step(atoms, fx.species, dt);  // warmup
  vv.post_force_step(atoms, fx.species, dt);

  for (int k = 0; k < repeats; ++k) {
    auto t0 = clk::now();
    vv.pre_force_step(atoms, fx.species, dt);
    vv.post_force_step(atoms, fx.species, dt);
    auto t1 = clk::now();
    ts.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  return median(ts);
}

double time_gpu(const Fixture& fx, int repeats) {
  using clk = std::chrono::steady_clock;
  std::vector<double> ts;
  ts.reserve(repeats);

  tdmd::gpu::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 16;
  tdmd::gpu::DevicePool pool(cfg);
  tdmd::gpu::DeviceStream stream = tdmd::gpu::make_stream(cfg.device_id);
  tdmd::GpuVelocityVerletIntegrator gpu(fx.species);

  tdmd::AtomSoA atoms = fx.atoms;
  const double dt = 0.001;

  gpu.pre_force_step(atoms, dt, pool, stream);  // warmup
  gpu.post_force_step(atoms, dt, pool, stream);

  for (int k = 0; k < repeats; ++k) {
    auto t0 = clk::now();
    gpu.pre_force_step(atoms, dt, pool, stream);
    gpu.post_force_step(atoms, dt, pool, stream);
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
    Fixture fx = make_lattice(target);
    const std::size_t n = fx.atoms.size();
    double cpu_us = -1.0;
    double gpu_us = -1.0;
    try {
      cpu_us = time_cpu(fx, kRepeats);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "CPU bench failed at n=%zu: %s\n", n, e.what());
    }
    try {
      gpu_us = time_gpu(fx, kRepeats);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "GPU bench failed at n=%zu: %s\n", n, e.what());
    }
    const double speedup = (cpu_us > 0.0 && gpu_us > 0.0) ? cpu_us / gpu_us : 0.0;
    std::fprintf(stdout, "%-8zu %12.0f %12.0f %9.1fx\n", n, cpu_us, gpu_us, speedup);
    out << n << '\t' << cpu_us << '\t' << gpu_us << '\t' << speedup << '\n';
  }
  return 0;
}
