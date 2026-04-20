// Exec pack: docs/development/m8_execution_pack.md T8.0 (T7.8b carry-forward)
// SPEC: docs/specs/gpu/SPEC.md §3.2c (2-rank overlap gate, hardware prerequisite)
// Master spec: §14 M7 carry-forward
// Decisions: D-M6-7 (bit-exact GPU vs CPU), D-M6-8 (dense-cutoff threshold)
//
// T8.0 2-rank pipeline-functional overlap gate: ≥30% overlap budget on K=4,
// 2-rank Pattern-2 emulation with synthetic halo D2H/MPI/H2D per slot.
//
// Why this test exists. T7.8 single-rank overlap is kernel-bound on a single
// GPU (T_mem/T_k ≈ 0.24 on RTX 5080), so the asymptotic max overlap is ~21%
// (K→∞) and ~17% at K=4. The 30% gate from exec pack §T7.8 / gpu/SPEC §3.2a
// is explicitly for the 2-rank K=4 setup, where halo traffic roughly doubles
// per-step memory work (T_mem/T_k ~0.55). This test ships the 2-rank
// infrastructure and the gate assertion.
//
// Hardware prerequisite (gpu/SPEC §3.2c). Measuring 2-rank overlap requires
// ≥2 physical GPUs so each rank owns a distinct device. On dev hosts with
// 1 GPU the test SKIPs with exit code 4 (Catch2 v3.5+ SKIP_RETURN_CODE), and
// CTest surfaces this as SKIPPED rather than FAIL. The 30% measurement is
// cloud-burst-gated and ties into T8.11 (TDMD vs LAMMPS scaling harness).
//
// Pipeline model. Each "TD step" in K=4 does EAM compute + halo exchange.
// Serial: K back-to-back `enqueue_eam + drain_eam + MPI_Sendrecv(halo)`,
// all synchronous. Pipelined: enqueue K async EAM dispatches into slot-
// rotated GpuDispatchAdapter, then drain in order with MPI_Sendrecv
// interleaved — slot k's mem_stream D2H + MPI Sendrecv overlaps with slot
// k+1's compute_stream kernel (gpu/SPEC §3.2).
//
// Numerical cross-check. Slot 0's PE + virial are compared against a serial
// oracle at ≤ 1e-12 rel (gpu/SPEC §7.2 — D-M6-7 byte-exact). Each rank
// evaluates the full 10 976-atom Al FCC locally; the halo exchange is a
// synthetic communication-volume model, not a true Pattern-2 coordinator.
// This isolates the overlap measurement from coordinator/migration cost.
//
// Skips gracefully on CPU-only builds, zero CUDA devices, or <2 devices.

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

#include <mpi.h>

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

// CPU-only build path: Catch2 demands ≥1 TEST_CASE per binary or it exits with
// code 2. Mirrors the T7.8 single-rank stub.
TEST_CASE("GpuDispatchAdapter — CPU-only build skips 2-rank overlap gate",
          "[gpu][overlap][t80][cpu]") {
#if TDMD_BUILD_CUDA
  SUCCEED("CUDA build — 2-rank overlap gate exercised in the TDMD_BUILD_CUDA test below");
#else
  SUCCEED("CPU-only build — 2-rank overlap pipeline test skipped (no CUDA runtime)");
#endif
}

#if TDMD_BUILD_CUDA

namespace {

int cuda_device_count() noexcept {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  return (err == cudaSuccess) ? count : 0;
}

}  // namespace

TEST_CASE("GpuDispatchAdapter — K=4 2-rank pipeline overlap ≥ 30% (T8.0/T7.8b)",
          "[gpu][overlap][t80][mpi][mpi2rank]") {
  // Hardware prerequisite (gpu/SPEC §3.2c): each rank needs its own GPU.
  const int devices = cuda_device_count();
  if (devices < 2) {
    SKIP(
        "need ≥ 2 CUDA devices for meaningful 2-rank overlap measurement "
        "(dev host = 1 GPU; 30% gate cloud-burst-gated, ties to T8.11)");
  }

  int rank = -1;
  int nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  REQUIRE(nranks == 2);

  // Per-rank device pinning so each rank owns a distinct GPU.
  REQUIRE(cudaSetDevice(rank % devices) == cudaSuccess);
  const int peer = 1 - rank;

  // --- Fixture: 14×14×14 Al FCC = 10976 atoms. Each rank evaluates the full
  // problem locally; halo exchange is modeled synthetically below.
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

  tg::GpuConfig cfg;
  cfg.device_id = rank % devices;
  cfg.memory_pool_init_size_mib = 256;
  tg::DevicePool pool(cfg);

  // Pinned host inputs (positions, types, cell CSR) — shared across all slots.
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

  // Synthetic per-slot halo buffer: ~1024 doubles = 8 KB, models the halo
  // surface volume that a real Pattern-2 P_space=2 split would exchange
  // between ranks (halo slab for a ~50 Å×50 Å contact face).
  constexpr std::size_t kHaloDoubles = 1024;
  std::vector<tg::DevicePtr<std::byte>> halo_send_store(K), halo_recv_store(K);
  std::vector<double*> halo_send(K), halo_recv(K);
  for (std::size_t k = 0; k < K; ++k) {
    halo_send_store[k] = pin_double(kHaloDoubles);
    halo_recv_store[k] = pin_double(kHaloDoubles);
    halo_send[k] = reinterpret_cast<double*>(halo_send_store[k].get());
    halo_recv[k] = reinterpret_cast<double*>(halo_recv_store[k].get());
    for (std::size_t i = 0; i < kHaloDoubles; ++i) {
      halo_send[k][i] = static_cast<double>(rank * 1000 + k * 10 + i);
    }
  }

  // --- Serial baseline: K iterations of {sync EAM compute, sync halo Sendrecv}.
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
    auto res = serial_gpu.compute(n,
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
    return res;
  };

  auto run_serial_step_with_halo = [&](std::size_t k) {
    auto res = run_serial_once();
    MPI_Sendrecv(halo_send[k],
                 kHaloDoubles,
                 MPI_DOUBLE,
                 peer,
                 /*sendtag=*/100 + static_cast<int>(k),
                 halo_recv[k],
                 kHaloDoubles,
                 MPI_DOUBLE,
                 peer,
                 /*recvtag=*/100 + static_cast<int>(k),
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    return res;
  };

  // Warmup — populate spline cache + JIT any lazy CUDA context state; also
  // prime the MPI channel so the first timed MPI_Sendrecv does not pay
  // connection-setup latency.
  tg::EamAlloyGpuResult serial_oracle;
  for (int w = 0; w < 2; ++w) {
    serial_oracle = run_serial_step_with_halo(0);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // --- Pipelined: K-deep adapter, K async enqueues, K drains interleaved
  // with MPI_Sendrecv. Slot k's D2H on mem_stream overlaps with slot k+1's
  // compute_stream kernel; the MPI_Sendrecv after drain(k) runs on the host
  // while slot k+1..K-1 GPU work is still in flight.
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
      MPI_Sendrecv(halo_send[k],
                   kHaloDoubles,
                   MPI_DOUBLE,
                   peer,
                   /*sendtag=*/200 + static_cast<int>(k),
                   halo_recv[k],
                   kHaloDoubles,
                   MPI_DOUBLE,
                   peer,
                   /*recvtag=*/200 + static_cast<int>(k),
                   MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
    }
    return results;
  };

  // Warmup.
  (void) run_pipeline_once();
  MPI_Barrier(MPI_COMM_WORLD);

  // Median-of-9 repeats to filter per-trial scheduling + MPI jitter. Matches
  // the single-rank T7.8 protocol.
  const std::size_t kRepeats = 9;
  std::vector<double> serial_ns_samples;
  std::vector<double> pipelined_ns_samples;
  std::vector<double> ratio_samples;
  serial_ns_samples.reserve(kRepeats);
  pipelined_ns_samples.reserve(kRepeats);
  ratio_samples.reserve(kRepeats);
  std::vector<tg::EamAlloyGpuResult> last_pipeline_results;

  for (std::size_t r = 0; r < kRepeats; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    cudaDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t k = 0; k < K; ++k) {
      (void) run_serial_step_with_halo(k);
    }
    cudaDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();

    MPI_Barrier(MPI_COMM_WORLD);
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

  INFO("rank = " << rank << ", K = " << K << ", atoms = " << n << ", repeats = " << kRepeats);
  INFO("median t_serial    = " << t_serial_ms << " ms");
  INFO("median t_pipelined = " << t_pipelined_ms << " ms");
  INFO("min/median/max overlap ratio = " << ratio_samples.front() << " / " << overlap_ratio << " / "
                                         << ratio_samples.back());

  // D-M6-7 bit-exact: slot 0 PE + virial ≡ serial oracle at ≤ 1e-12 rel.
  REQUIRE(rel(pipeline_results[0].potential_energy, serial_oracle.potential_energy) <= 1e-12);
  for (std::size_t k = 0; k < 6; ++k) {
    REQUIRE(rel(pipeline_results[0].virial[k], serial_oracle.virial[k]) <= 1e-12);
  }

  // 30% overlap gate — 2-rank K=4, halo traffic doubles memory work so
  // T_mem/T_k ~0.55 and asymptotic max overlap ≈ 36% (K→∞); the 30% bar is
  // achievable at K=4. See gpu/SPEC §3.2c for the derivation.
  REQUIRE(overlap_ratio >= 0.30);
}

#endif  // TDMD_BUILD_CUDA
