// Exec pack: docs/development/m8_execution_pack.md T8.9
// Pre-impl: docs/development/t8.9_pre_impl.md
// SPEC: docs/specs/gpu/SPEC.md §8 (MixedFast), §8.3 (D-M8-8 dense-cutoff),
//       docs/specs/verify/SPEC.md §4.7 (t6_snap_tungsten)
// Threshold registry: verify/thresholds/thresholds.yaml
//   gpu_mixed_fast_snap_only.snap.force_relative = 1.0e-5
//   gpu_mixed_fast_snap_only.snap.energy_relative = 1.0e-7
//
// T8.9 acceptance gate: Philosophy B SnapGpuMixed (narrow-FP32 pair-math)
// vs Fp64 Reference GPU on T6 tungsten 2000-atom rattled BCC.
//
// Why bypass the flavor-dispatched adapter: we want to measure SnapGpuMixed
// precision deterministically in every build-flavor CI pass (including
// MixedFastBuild where the adapter routes SNAP to FP64). Constructing
// SnapGpuMixed + SnapGpu directly pins both paths regardless of the flavor
// the test binary was compiled with.
//
// Gates (D-M8-8 dense-cutoff):
//   1. Per-atom force L∞ rel diff ≤ 1e-5
//   2. Total PE rel diff ≤ 1e-7
//   3. Virial Voigt rel-to-max ≤ 5e-6 (each of 6 components, rel to max
//      component magnitude on the Fp64 reference)
//
// Self-skips exit 77 on no-CUDA host or LAMMPS submodule uninitialized
// (Option A convention, matches test_snap_gpu_bit_exact).

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/factories.hpp"
#include "tdmd/gpu/gpu_config.hpp"
#include "tdmd/gpu/neighbor_list_gpu.hpp"  // BoxParams
#include "tdmd/gpu/snap_gpu.hpp"
#include "tdmd/gpu/snap_gpu_mixed.hpp"
#include "tdmd/neighbor/cell_grid.hpp"
#include "tdmd/potentials/snap_file.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

#ifndef TDMD_TEST_FIXTURES_DIR
#error "TDMD_TEST_FIXTURES_DIR must be defined by the build system"
#endif

#if TDMD_BUILD_CUDA
#include <cuda_runtime.h>
#endif

namespace fs = std::filesystem;
namespace tp = tdmd::potentials;
namespace tg = tdmd::gpu;

namespace {

constexpr int kExitSkip = 77;

fs::path lammps_snap_examples_dir() {
  const fs::path fixtures_dir = TDMD_TEST_FIXTURES_DIR;
  const fs::path repo_root = fixtures_dir.parent_path().parent_path().parent_path();
  return repo_root / "verify" / "third_party" / "lammps" / "examples" / "snap";
}

void skip_if_submodule_uninitialized(const fs::path& dir) {
  if (!fs::exists(dir)) {
    std::fprintf(stderr,
                 "[test_snap_mixed_fast_within_threshold] SKIP: LAMMPS submodule not initialized "
                 "at %s\n",
                 dir.string().c_str());
    std::fflush(stderr);
    std::exit(kExitSkip);
  }
}

tp::SnapData load_w_fixture() {
  const auto dir = lammps_snap_examples_dir();
  skip_if_submodule_uninitialized(dir);
  return tp::parse_snap_files((dir / "W_2940_2017_2.snapcoeff").string(),
                              (dir / "W_2940_2017_2.snapparam").string());
}

tdmd::Box make_cubic_box(double length) {
  tdmd::Box b;
  b.xhi = length;
  b.yhi = length;
  b.zhi = length;
  b.periodic_x = true;
  b.periodic_y = true;
  b.periodic_z = true;
  return b;
}

void add_bcc_W(tdmd::AtomSoA& atoms, int nrep, double a) {
  for (int kz = 0; kz < nrep; ++kz) {
    for (int ky = 0; ky < nrep; ++ky) {
      for (int kx = 0; kx < nrep; ++kx) {
        atoms.add_atom(0, kx * a, ky * a, kz * a);
        atoms.add_atom(0, (kx + 0.5) * a, (ky + 0.5) * a, (kz + 0.5) * a);
      }
    }
  }
}

void apply_tiny_rattle(tdmd::AtomSoA& atoms) {
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const double ph = static_cast<double>(i) * 0.1;
    atoms.x[i] += 0.01 * std::sin(ph);
    atoms.y[i] += 0.01 * std::sin(ph + 1.0);
    atoms.z[i] += 0.01 * std::sin(ph + 2.0);
  }
}

double rel(double a, double b) {
  const double num = std::abs(a - b);
  const double den = std::max(1.0, std::max(std::abs(a), std::abs(b)));
  return num / den;
}

struct ForceDiff {
  double worst = 0.0;
  std::size_t worst_atom = 0;
  int worst_k = 0;
};

ForceDiff max_rel_force_diff(const std::vector<double>& fx_a,
                             const std::vector<double>& fy_a,
                             const std::vector<double>& fz_a,
                             const std::vector<double>& fx_b,
                             const std::vector<double>& fy_b,
                             const std::vector<double>& fz_b) {
  ForceDiff d;
  for (std::size_t i = 0; i < fx_a.size(); ++i) {
    const double fa[3] = {fx_a[i], fy_a[i], fz_a[i]};
    const double fb[3] = {fx_b[i], fy_b[i], fz_b[i]};
    for (int k = 0; k < 3; ++k) {
      const double num = std::abs(fa[k] - fb[k]);
      const double den = std::max(1.0, std::max(std::abs(fa[k]), std::abs(fb[k])));
      const double r = num / den;
      if (r > d.worst) {
        d.worst = r;
        d.worst_atom = i;
        d.worst_k = k;
      }
    }
  }
  return d;
}

// Build BoxParams + flat SNAP tables from adapter-flattened buffers. Mirrors
// SnapGpuAdapter::compute but run at the gpu/ layer directly so both
// SnapGpu::compute and SnapGpuMixed::compute can be fed identical inputs.
struct DirectSnapInputs {
  tg::BoxParams bp;
  tg::SnapTablesHost tables;
  std::vector<double> radius_flat;
  std::vector<double> weight_flat;
  std::vector<double> beta_flat;
  // rcut_sq_ab is borrowed from SnapData (caller keeps SnapData alive).
};

DirectSnapInputs make_direct_inputs(const tp::SnapData& data,
                                    const tdmd::Box& box,
                                    const tdmd::CellGrid& grid) {
  DirectSnapInputs in;
  const std::size_t n_species = data.species.size();
  const std::size_t beta_stride = static_cast<std::size_t>(data.k_max) + 1U;

  in.radius_flat.resize(n_species);
  in.weight_flat.resize(n_species);
  for (std::size_t a = 0; a < n_species; ++a) {
    in.radius_flat[a] = data.species[a].radius_elem;
    in.weight_flat[a] = data.species[a].weight_elem;
  }
  in.beta_flat.resize(n_species * beta_stride);
  for (std::size_t a = 0; a < n_species; ++a) {
    const auto& src = data.species[a].beta;
    double* dst = in.beta_flat.data() + a * beta_stride;
    for (std::size_t k = 0; k < beta_stride; ++k) {
      dst[k] = src[k];
    }
  }

  in.bp.xlo = box.xlo;
  in.bp.ylo = box.ylo;
  in.bp.zlo = box.zlo;
  in.bp.lx = box.lx();
  in.bp.ly = box.ly();
  in.bp.lz = box.lz();
  in.bp.cell_x = grid.cell_x();
  in.bp.cell_y = grid.cell_y();
  in.bp.cell_z = grid.cell_z();
  in.bp.nx = grid.nx();
  in.bp.ny = grid.ny();
  in.bp.nz = grid.nz();
  in.bp.periodic_x = box.periodic_x;
  in.bp.periodic_y = box.periodic_y;
  in.bp.periodic_z = box.periodic_z;
  in.bp.cutoff = data.max_pairwise_cutoff();
  in.bp.skin = 0.0;

  in.tables.twojmax = data.params.twojmax;
  in.tables.rcutfac = data.params.rcutfac;
  in.tables.rfac0 = data.params.rfac0;
  in.tables.rmin0 = data.params.rmin0;
  in.tables.switchflag = data.params.switchflag ? 1 : 0;
  in.tables.bzeroflag = data.params.bzeroflag ? 1 : 0;
  in.tables.bnormflag = data.params.bnormflag ? 1 : 0;
  in.tables.wselfallflag = data.params.wselfallflag ? 1 : 0;
  in.tables.k_max = data.k_max;
  in.tables.idxb_max = 0;  // SnapGpu::compute re-derives from twojmax.
  in.tables.idxu_max = 0;
  in.tables.idxz_max = 0;
  in.tables.n_species = n_species;
  in.tables.beta_stride = beta_stride;
  in.tables.radius_elem = in.radius_flat.data();
  in.tables.weight_elem = in.weight_flat.data();
  in.tables.beta_coefficients = in.beta_flat.data();
  in.tables.rcut_sq_ab = data.rcut_sq_ab.data();
  return in;
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

TEST_CASE("SnapGpuMixed — T6 tungsten D-M8-8 gate: FP32 vs FP64 GPU within dense-cutoff budget",
          "[gpu][snap][t8.9][mixed_fast][dense_cutoff]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }

  const tp::SnapData data = load_w_fixture();

  // 10×10×10 BCC W = 2000 atoms (same fixture as T8.7 bit-exact test).
  constexpr double kLatticeA = 3.18;
  constexpr int kNrep = 10;
  tdmd::Box box = make_cubic_box(kLatticeA * kNrep);
  tdmd::AtomSoA base;
  add_bcc_W(base, kNrep, kLatticeA);
  REQUIRE(base.size() == 2000u);
  apply_tiny_rattle(base);

  const double cutoff = data.params.rcutfac;
  const double skin = 0.2;

  tdmd::CellGrid grid;
  grid.build(box, cutoff, skin);
  grid.bin(base);

  const std::size_t n = base.size();
  DirectSnapInputs in = make_direct_inputs(data, box, grid);

  // --- GPU FP64 reference (oracle for the 1e-5 / 1e-7 / 5e-6 budget) ---
  std::vector<double> fx_ref(n, 0.0), fy_ref(n, 0.0), fz_ref(n, 0.0);
  double pe_ref;
  double virial_ref[6];
  {
    tg::GpuConfig cfg;
    cfg.memory_pool_init_size_mib = 16;
    tg::DevicePool pool(cfg);
    tg::DeviceStream stream = tg::make_stream(cfg.device_id);
    tg::SnapGpu gpu_ref;
    const tg::SnapGpuResult r = gpu_ref.compute(n,
                                                base.type.data(),
                                                base.x.data(),
                                                base.y.data(),
                                                base.z.data(),
                                                grid.cell_count(),
                                                grid.cell_offsets().data(),
                                                grid.cell_atoms().data(),
                                                in.bp,
                                                in.tables,
                                                fx_ref.data(),
                                                fy_ref.data(),
                                                fz_ref.data(),
                                                pool,
                                                stream);
    pe_ref = r.potential_energy;
    for (std::size_t k = 0; k < 6; ++k) {
      virial_ref[k] = r.virial[k];
    }
  }

  // --- GPU FP32 pair-math (SnapGpuMixed) --- Fresh pool/stream so device
  // state between the two runs is isolated.
  std::vector<double> fx_mix(n, 0.0), fy_mix(n, 0.0), fz_mix(n, 0.0);
  double pe_mix;
  double virial_mix[6];
  {
    tg::GpuConfig cfg;
    cfg.memory_pool_init_size_mib = 16;
    tg::DevicePool pool(cfg);
    tg::DeviceStream stream = tg::make_stream(cfg.device_id);
    tg::SnapGpuMixed gpu_mix;
    const tg::SnapGpuResult r = gpu_mix.compute(n,
                                                base.type.data(),
                                                base.x.data(),
                                                base.y.data(),
                                                base.z.data(),
                                                grid.cell_count(),
                                                grid.cell_offsets().data(),
                                                grid.cell_atoms().data(),
                                                in.bp,
                                                in.tables,
                                                fx_mix.data(),
                                                fy_mix.data(),
                                                fz_mix.data(),
                                                pool,
                                                stream);
    pe_mix = r.potential_energy;
    for (std::size_t k = 0; k < 6; ++k) {
      virial_mix[k] = r.virial[k];
    }
  }

  // --- Gate 1: per-atom force L∞ rel ≤ 1e-5 ---
  constexpr double kTolForce = 1.0e-5;
  const ForceDiff fd = max_rel_force_diff(fx_ref, fy_ref, fz_ref, fx_mix, fy_mix, fz_mix);
  INFO("worst force rel = " << fd.worst << " at atom " << fd.worst_atom << " comp " << fd.worst_k
                            << " (tol " << kTolForce << ")");
  REQUIRE(fd.worst <= kTolForce);

  // --- Gate 2: PE rel ≤ 1e-7 ---
  constexpr double kTolEnergy = 1.0e-7;
  const double pe_rel = rel(pe_ref, pe_mix);
  INFO("PE ref=" << pe_ref << " mixed=" << pe_mix << " rel=" << pe_rel << " (tol " << kTolEnergy
                 << ")");
  REQUIRE(pe_rel <= kTolEnergy);

  // --- Gate 3: virial Voigt rel-to-max ≤ 5e-6 ---
  constexpr double kTolVirial = 5.0e-6;
  double max_abs = 1.0;
  for (std::size_t k = 0; k < 6; ++k) {
    max_abs = std::max(max_abs, std::abs(virial_ref[k]));
  }
  for (std::size_t k = 0; k < 6; ++k) {
    const double rel_to_max = std::abs(virial_ref[k] - virial_mix[k]) / max_abs;
    INFO("virial[" << k << "] ref=" << virial_ref[k] << " mixed=" << virial_mix[k]
                   << " rel_to_max=" << rel_to_max << " (tol " << kTolVirial << ")");
    REQUIRE(rel_to_max <= kTolVirial);
  }
}

#else  // !TDMD_BUILD_CUDA

TEST_CASE("SnapGpuMixed threshold gate — CPU-only build skips gracefully",
          "[gpu][snap][t8.9][cpu]") {
  SUCCEED("CPU-only build — SnapGpuMixed gate requires CUDA");
}

#endif  // TDMD_BUILD_CUDA
