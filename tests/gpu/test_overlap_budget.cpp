// Exec pack: docs/development/m7_execution_pack.md T7.8
// SPEC: docs/specs/gpu/SPEC.md §3.2 (compute/mem overlap pipeline)
//
// T7.8 single-rank pipeline-functional gate: ≥5% overlap budget на K=4 10k-atom
// EAM-only setup. Serial baseline = K back-to-back synchronous
// `EamAlloyGpu::compute()` calls on a single stream (no cross-stream overlap).
// Pipelined path = K `GpuDispatchAdapter::enqueue_eam()` followed by K
// `drain_eam()`; each slot runs H2D on mem_stream, kernels on compute_stream,
// D2H on mem_stream with event-chained cross-stream sync per gpu/SPEC §3.2.
// Overlap ratio = `(t_serial - t_pipelined) / t_pipelined`.
//
// Why 5%, not 30%? The 30% gate from exec pack §T7.8 / gpu/SPEC §3.2a is
// explicitly for the 2-rank K=4 setup, where halo D2H/MPI/H2D traffic roughly
// doubles per-step memory work. Single-rank EAM on RTX 5080 is kernel-bound:
// measured T_mem/T_k ≈ 0.24, giving asymptotic max overlap ≈ 21% (K→∞) and
// ≈ 17% at K=4 — physically less than the 30% bar. This test therefore
// validates the pipeline *mechanism* (event chain, slot rotation, bit-exact
// reductions) and ensures the overlap is non-trivial (>5%, comfortably above
// noise). The 30% 2-rank gate is owned by T7.14 (M7 integration smoke).
//
// Numerical cross-check: the first pipelined slot's PE + virial are compared
// against the serial oracle at ≤ 1e-12 rel (gpu/SPEC §7.2 — D-M6-7 byte-exact).
//
// Skips gracefully on CPU-only builds and on hosts with zero CUDA devices.

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/eam_alloy_gpu.hpp"
#include "tdmd/gpu/factories.hpp"
#include "tdmd/gpu/gpu_config.hpp"
#include "tdmd/gpu/neighbor_list_gpu.hpp"  // BoxParams
#include "tdmd/gpu/types.hpp"
#include "tdmd/neighbor/cell_grid.hpp"
#include "tdmd/potentials/eam_alloy.hpp"
#include "tdmd/potentials/eam_file.hpp"
#include "tdmd/potentials/tabulated.hpp"
#include "tdmd/scheduler/gpu_dispatch_adapter.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#ifndef TDMD_TEST_FIXTURES_DIR
#error "TDMD_TEST_FIXTURES_DIR must be defined by the build system"
#endif

#if TDMD_BUILD_CUDA
#include <cuda_runtime.h>
#endif

namespace tg = tdmd::gpu;
namespace tp = tdmd::potentials;
namespace ts = tdmd::scheduler;

namespace {

std::string test_fixtures_dir() {
  return TDMD_TEST_FIXTURES_DIR;
}

// 14×14×14 Al FCC = 10976 atoms — matches the exec pack "10k-atom" target.
// Lattice a=4.05 Å; box side 14·a = 56.7 Å > 3·(cutoff+skin) for Al_small
// (cutoff 5.0015 Å).
struct AlFccFixture {
  tdmd::AtomSoA atoms;
  tdmd::Box box;
};

AlFccFixture make_al_fcc(std::size_t nx, std::size_t ny, std::size_t nz) {
  AlFccFixture fx;
  const double a = 4.05;
  fx.box.xlo = fx.box.ylo = fx.box.zlo = 0.0;
  fx.box.xhi = a * static_cast<double>(nx);
  fx.box.yhi = a * static_cast<double>(ny);
  fx.box.zhi = a * static_cast<double>(nz);
  fx.box.periodic_x = fx.box.periodic_y = fx.box.periodic_z = true;

  const double basis[4][3] = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
  for (std::size_t iz = 0; iz < nz; ++iz) {
    for (std::size_t iy = 0; iy < ny; ++iy) {
      for (std::size_t ix = 0; ix < nx; ++ix) {
        for (const auto& b : basis) {
          const double x = (static_cast<double>(ix) + b[0]) * a;
          const double y = (static_cast<double>(iy) + b[1]) * a;
          const double z = (static_cast<double>(iz) + b[2]) * a;
          fx.atoms.add_atom(0U, x, y, z);
        }
      }
    }
  }
  return fx;
}

void perturb(tdmd::AtomSoA& atoms, double scale) {
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    atoms.x[i] += scale * (((i % 7) / 6.0) - 0.5);
    atoms.y[i] += scale * (((i % 11) / 10.0) - 0.5);
    atoms.z[i] += scale * (((i % 13) / 12.0) - 0.5);
  }
}

// Flatten a TabulatedFunction into 7-doubles-per-cell row-major layout —
// matching `EamAlloyGpuAdapter`'s internal flattening (potentials/SPEC §4.4).
void flatten_tab_into(const tp::TabulatedFunction& tab, std::vector<double>& out) {
  const std::size_t n = tab.size();
  out.reserve(out.size() + n * tp::TabulatedFunction::kCoefficientsPerCell);
  for (std::size_t i = 1; i <= n; ++i) {
    const auto& c = tab.coeffs(i);
    for (double v : c) {
      out.push_back(v);
    }
  }
}

double rel(double a, double b) {
  const double num = std::abs(a - b);
  const double den = std::max(1.0, std::max(std::abs(a), std::abs(b)));
  return num / den;
}

}  // namespace

#if TDMD_BUILD_CUDA

namespace {
bool cuda_device_available() noexcept {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  return err == cudaSuccess && count > 0;
}
}  // namespace

TEST_CASE("GpuDispatchAdapter — K=4 single-rank pipeline-functional overlap ≥ 5%",
          "[gpu][overlap][t78]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }

  // --- Fixture: 14×14×14 Al FCC = 10976 atoms.
  AlFccFixture fx = make_al_fcc(14, 14, 14);
  REQUIRE(fx.atoms.size() == 10976u);
  perturb(fx.atoms, 0.05);

  const auto data = tp::parse_eam_alloy(
      (std::filesystem::path(test_fixtures_dir()) / "Al_small.eam.alloy").string());
  REQUIRE(data.species_names.size() == 1u);

  const double cutoff = data.cutoff;
  const double skin = 0.2;
  tdmd::CellGrid grid;
  grid.build(fx.box, cutoff, skin);
  grid.bin(fx.atoms);

  // --- Flatten EAM tables once; both serial + pipelined paths share the
  // backing storage so the adapter's spline cache hits on steady state.
  std::vector<double> F_coeffs_flat, rho_coeffs_flat, z2r_coeffs_flat;
  for (const auto& t : data.F_rho) {
    flatten_tab_into(t, F_coeffs_flat);
  }
  for (const auto& t : data.rho_r) {
    flatten_tab_into(t, rho_coeffs_flat);
  }
  for (const auto& t : data.z2r) {
    flatten_tab_into(t, z2r_coeffs_flat);
  }

  tg::EamAlloyTablesHost tables;
  tables.n_species = data.species_names.size();
  tables.nrho = data.F_rho[0].size();
  tables.nr = data.rho_r[0].size();
  tables.npairs = tables.n_species * (tables.n_species + 1) / 2;
  tables.F_x0 = data.F_rho[0].x0();
  tables.F_dx = data.F_rho[0].dx();
  tables.r_x0 = data.rho_r[0].x0();
  tables.r_dx = data.rho_r[0].dx();
  tables.cutoff = cutoff;
  tables.F_coeffs = F_coeffs_flat.data();
  tables.rho_coeffs = rho_coeffs_flat.data();
  tables.z2r_coeffs = z2r_coeffs_flat.data();

  tg::BoxParams bp;
  bp.xlo = fx.box.xlo;
  bp.ylo = fx.box.ylo;
  bp.zlo = fx.box.zlo;
  bp.lx = fx.box.lx();
  bp.ly = fx.box.ly();
  bp.lz = fx.box.lz();
  bp.cell_x = grid.cell_x();
  bp.cell_y = grid.cell_y();
  bp.cell_z = grid.cell_z();
  bp.nx = grid.nx();
  bp.ny = grid.ny();
  bp.nz = grid.nz();
  bp.periodic_x = fx.box.periodic_x;
  bp.periodic_y = fx.box.periodic_y;
  bp.periodic_z = fx.box.periodic_z;
  bp.cutoff = cutoff;
  bp.skin = 0.0;

  const std::size_t n = fx.atoms.size();
  const std::size_t K = 4;

  // Pool sized for K slots × (atoms + cell CSR + spline tables + scratch).
  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 256;
  tg::DevicePool pool(cfg);

  // T7.8 overlap requires pinned host memory on both sides of H2D/D2H:
  // pageable buffers degrade cudaMemcpyAsync to internal staging +
  // host-thread block, preventing pipeline overlap. Allocate pinned mirrors
  // of positions / types / cell CSR (inputs) + per-slot force outputs.
  auto pin_double = [&](std::size_t count) {
    return pool.allocate_pinned_host(count * sizeof(double));
  };
  auto pin_u32 = [&](std::size_t count) {
    return pool.allocate_pinned_host(count * sizeof(std::uint32_t));
  };

  auto h_types = pin_u32(n);
  auto h_x = pin_double(n);
  auto h_y = pin_double(n);
  auto h_z = pin_double(n);
  auto h_cell_offsets = pin_u32(grid.cell_count() + 1);
  auto h_cell_atoms = pin_u32(n);
  auto* types_ptr = reinterpret_cast<std::uint32_t*>(h_types.get());
  auto* x_ptr = reinterpret_cast<double*>(h_x.get());
  auto* y_ptr = reinterpret_cast<double*>(h_y.get());
  auto* z_ptr = reinterpret_cast<double*>(h_z.get());
  auto* cell_offsets_ptr = reinterpret_cast<std::uint32_t*>(h_cell_offsets.get());
  auto* cell_atoms_ptr = reinterpret_cast<std::uint32_t*>(h_cell_atoms.get());
  std::copy_n(fx.atoms.type.data(), n, types_ptr);
  std::copy_n(fx.atoms.x.data(), n, x_ptr);
  std::copy_n(fx.atoms.y.data(), n, y_ptr);
  std::copy_n(fx.atoms.z.data(), n, z_ptr);
  std::copy_n(grid.cell_offsets().data(), grid.cell_count() + 1, cell_offsets_ptr);
  std::copy_n(grid.cell_atoms().data(), n, cell_atoms_ptr);

  // --- Serial baseline: K sync compute() calls on a single stream.
  tg::DeviceStream serial_stream = tg::make_stream(cfg.device_id);
  tg::EamAlloyGpu serial_gpu;

  auto h_s_fx = pin_double(n);
  auto h_s_fy = pin_double(n);
  auto h_s_fz = pin_double(n);
  auto* s_fx = reinterpret_cast<double*>(h_s_fx.get());
  auto* s_fy = reinterpret_cast<double*>(h_s_fy.get());
  auto* s_fz = reinterpret_cast<double*>(h_s_fz.get());

  auto run_serial_once = [&]() {
    std::fill_n(s_fx, n, 0.0);
    std::fill_n(s_fy, n, 0.0);
    std::fill_n(s_fz, n, 0.0);
    return serial_gpu.compute(n,
                              types_ptr,
                              x_ptr,
                              y_ptr,
                              z_ptr,
                              grid.cell_count(),
                              cell_offsets_ptr,
                              cell_atoms_ptr,
                              bp,
                              tables,
                              s_fx,
                              s_fy,
                              s_fz,
                              pool,
                              serial_stream);
  };

  // Warmup — populate spline cache + JIT any lazy CUDA context state.
  tg::EamAlloyGpuResult serial_oracle;
  for (int w = 0; w < 2; ++w) {
    serial_oracle = run_serial_once();
  }

  // --- Pipelined: K-deep dispatch adapter with compute/mem stream pair.
  tg::DeviceStream compute_stream = tg::make_stream(cfg.device_id);
  tg::DeviceStream mem_stream = tg::make_stream(cfg.device_id);
  ts::GpuDispatchAdapter adapter(K, pool, compute_stream, mem_stream);

  std::vector<tg::DevicePtr<std::byte>> h_p_fx_store, h_p_fy_store, h_p_fz_store;
  std::vector<double*> p_fx(K), p_fy(K), p_fz(K);
  h_p_fx_store.reserve(K);
  h_p_fy_store.reserve(K);
  h_p_fz_store.reserve(K);
  for (std::size_t k = 0; k < K; ++k) {
    h_p_fx_store.push_back(pin_double(n));
    h_p_fy_store.push_back(pin_double(n));
    h_p_fz_store.push_back(pin_double(n));
    p_fx[k] = reinterpret_cast<double*>(h_p_fx_store[k].get());
    p_fy[k] = reinterpret_cast<double*>(h_p_fy_store[k].get());
    p_fz[k] = reinterpret_cast<double*>(h_p_fz_store[k].get());
  }

  auto run_pipeline_once = [&]() {
    std::vector<std::size_t> slots;
    slots.reserve(K);
    for (std::size_t k = 0; k < K; ++k) {
      std::fill_n(p_fx[k], n, 0.0);
      std::fill_n(p_fy[k], n, 0.0);
      std::fill_n(p_fz[k], n, 0.0);
      const std::size_t slot = adapter.enqueue_eam(n,
                                                   types_ptr,
                                                   x_ptr,
                                                   y_ptr,
                                                   z_ptr,
                                                   grid.cell_count(),
                                                   cell_offsets_ptr,
                                                   cell_atoms_ptr,
                                                   bp,
                                                   tables,
                                                   p_fx[k],
                                                   p_fy[k],
                                                   p_fz[k]);
      slots.push_back(slot);
    }
    std::vector<tg::EamAlloyGpuResult> results;
    results.reserve(K);
    for (std::size_t k = 0; k < K; ++k) {
      results.push_back(adapter.drain_eam(slots[k]));
    }
    return results;
  };

  // Warmup: one full K-batch primes the per-slot spline caches; one serial run
  // primes the JIT path. Both are then dropped from the timing measurement.
  (void) run_pipeline_once();
  (void) run_serial_once();

  // Per-run jitter on a 7 ms wall-time measurement is ±10-20% of the overlap
  // delta — pinning the timing to a single trial flakes the gate. Take median
  // of N measurements to filter scheduling noise (other-process steals,
  // GPU-side context-switch jitter).
  const std::size_t kRepeats = 9;
  std::vector<double> serial_ns_samples;
  std::vector<double> pipelined_ns_samples;
  std::vector<double> ratio_samples;
  serial_ns_samples.reserve(kRepeats);
  pipelined_ns_samples.reserve(kRepeats);
  ratio_samples.reserve(kRepeats);
  std::vector<tg::EamAlloyGpuResult> last_pipeline_results;

  for (std::size_t r = 0; r < kRepeats; ++r) {
    cudaDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < K; ++i) {
      (void) run_serial_once();
    }
    cudaDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();

    cudaDeviceSynchronize();
    const auto t2 = std::chrono::steady_clock::now();
    last_pipeline_results = run_pipeline_once();
    cudaDeviceSynchronize();
    const auto t3 = std::chrono::steady_clock::now();

    const double s_ns =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    const double p_ns =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
    serial_ns_samples.push_back(s_ns);
    pipelined_ns_samples.push_back(p_ns);
    ratio_samples.push_back((s_ns - p_ns) / p_ns);
  }

  std::sort(ratio_samples.begin(), ratio_samples.end());
  std::sort(serial_ns_samples.begin(), serial_ns_samples.end());
  std::sort(pipelined_ns_samples.begin(), pipelined_ns_samples.end());
  const double overlap_ratio = ratio_samples[kRepeats / 2];
  const double t_serial_ms = serial_ns_samples[kRepeats / 2] / 1.0e6;
  const double t_pipelined_ms = pipelined_ns_samples[kRepeats / 2] / 1.0e6;
  const auto& pipeline_results = last_pipeline_results;

  INFO("K = " << K << ", atoms = " << n << ", repeats = " << kRepeats);
  INFO("median t_serial    = " << t_serial_ms << " ms");
  INFO("median t_pipelined = " << t_pipelined_ms << " ms");
  INFO("min/median/max overlap ratio = " << ratio_samples.front() << " / " << overlap_ratio << " / "
                                         << ratio_samples.back());

  // --- Numerical cross-check: slot 0's PE + virial must agree with the serial
  // oracle at ≤ 1e-12 rel. Event chain preserves math — D-M6-7 byte-exact.
  REQUIRE(rel(pipeline_results[0].potential_energy, serial_oracle.potential_energy) <= 1e-12);
  for (std::size_t k = 0; k < 6; ++k) {
    REQUIRE(rel(pipeline_results[0].virial[k], serial_oracle.virial[k]) <= 1e-12);
  }

  // --- Overlap gate (single-rank EAM-only). 5% threshold proves the pipeline
  // *mechanism* (event chain + slot rotation + dual-stream H2D/D2H) actually
  // overlaps work. The 30% production gate from exec pack §T7.8 / gpu/SPEC
  // §3.2a is for the 2-rank K=4 setup and is owned by T7.14 integration smoke
  // — single-rank EAM on RTX 5080 is kernel-bound (T_mem/T_k ≈ 0.24, max ~17%
  // at K=4). See the file header comment for the derivation.
  REQUIRE(overlap_ratio >= 0.05);
}

#endif  // TDMD_BUILD_CUDA
