// Exec pack: docs/development/m6_execution_pack.md T6.8 (shipped)
//            docs/development/m7_execution_pack.md T7.0 (D-M6-8 formalized)
// SPEC: docs/specs/gpu/SPEC.md §8.2 (MixedFast, Philosophy B), §8.3 (D-M6-8)
// SPEC: verify/thresholds/thresholds.yaml `gpu_mixed_fast.dense_cutoff`
// Master spec: §D.1 Philosophy B + Приложение C T7.0 addendum
//
// Single-step acceptance gate on Ni-Al EAM/alloy + Al FCC fixtures. Thresholds
// are the T7.0-canonical D-M6-8 **dense-cutoff** values (EAM-class potentials,
// ≥20 neighbors per atom typical):
//
//   * per-atom force rel-diff ≤ 1e-5   (floor: max(1.0, |f|))
//   * total PE         rel-diff ≤ 1e-7
//   * virial Voigt     rel-diff ≤ 5e-6  (max-component normalized)
//
// These are the **canonical** D-M6-8 values after the T7.0 SPEC delta, not
// a relaxation of an unreachable target — the 1e-6 / 1e-8 ambition that
// appeared in v1.0-v1.0.11 was an FP32 precision ceiling artifact, never
// achievable in Philosophy B MixedFast on dense-cutoff stencils. Rationale
// lives in gpu/SPEC §8.3 "Rationale для dense-cutoff ceiling" + memory
// `project_fp32_eam_ceiling.md`. Sparse-cutoff stencils (LJ/Morse) retain
// 1e-6/1e-8 as an M9+ ambition for when those styles land on GPU.
//
// Drift over time (NVE conservation per 1000 steps) is a separate gate
// covered by tests/gpu/test_t4_nve_drift.cpp — integrator-level, also
// MixedFast-only (Reference is byte-exact per D-M6-7 and drift is
// uninteresting there).
//
// All cases skip when no CUDA device is available so the suite stays green
// on CPU-only builds + GPU-less CI runners.

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/eam_alloy_gpu.hpp"
#include "tdmd/gpu/eam_alloy_gpu_mixed.hpp"
#include "tdmd/gpu/factories.hpp"
#include "tdmd/gpu/gpu_config.hpp"
#include "tdmd/gpu/neighbor_list_gpu.hpp"  // BoxParams
#include "tdmd/gpu/types.hpp"
#include "tdmd/neighbor/cell_grid.hpp"
#include "tdmd/potentials/eam_file.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#ifndef TDMD_TEST_FIXTURES_DIR
#error "TDMD_TEST_FIXTURES_DIR must be defined by the build system"
#endif
#ifndef TDMD_VERIFY_POTENTIALS_DIR
#error "TDMD_VERIFY_POTENTIALS_DIR must be defined by the build system"
#endif

#if TDMD_BUILD_CUDA
#include <cuda_runtime.h>
#endif

namespace tg = tdmd::gpu;
namespace tp = tdmd::potentials;

namespace {

std::string test_fixtures_dir() {
  return TDMD_TEST_FIXTURES_DIR;
}

std::string verify_potentials_dir() {
  return TDMD_VERIFY_POTENTIALS_DIR;
}

// Ni-Al B2 8×8×8 = 1024 atoms at a=2.88 Å — same fixture as
// test_eam_alloy_gpu.cpp so the Reference↔Mixed comparison is apples-to-apples.
struct NiAlFixture {
  tdmd::AtomSoA atoms;
  tdmd::Box box;
  static constexpr double kLatticeA = 2.88;
  static constexpr int kNx = 8;
  static constexpr int kNy = 8;
  static constexpr int kNz = 8;
};

NiAlFixture make_nial_b2() {
  NiAlFixture fx;
  const double a = NiAlFixture::kLatticeA;
  fx.box.xlo = 0.0;
  fx.box.ylo = 0.0;
  fx.box.zlo = 0.0;
  fx.box.xhi = a * NiAlFixture::kNx;
  fx.box.yhi = a * NiAlFixture::kNy;
  fx.box.zhi = a * NiAlFixture::kNz;
  fx.box.periodic_x = fx.box.periodic_y = fx.box.periodic_z = true;

  for (int iz = 0; iz < NiAlFixture::kNz; ++iz) {
    for (int iy = 0; iy < NiAlFixture::kNy; ++iy) {
      for (int ix = 0; ix < NiAlFixture::kNx; ++ix) {
        fx.atoms.add_atom(0U, ix * a, iy * a, iz * a);
        fx.atoms.add_atom(1U, (ix + 0.5) * a, (iy + 0.5) * a, (iz + 0.5) * a);
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

double rel(double a, double b) {
  const double num = std::abs(a - b);
  const double den = std::max(1.0, std::max(std::abs(a), std::abs(b)));
  return num / den;
}

// Flatten a single TabulatedFunction into consecutive 7-double cells,
// mirroring the adapter's layout (species-major for F_rho/rho_r, pair-major
// for z2r). Returns vector so lifetime persists across compute() calls.
void flatten_tab(const tdmd::potentials::TabulatedFunction& tab, std::vector<double>& out) {
  const std::size_t n = tab.size();
  for (std::size_t i = 1; i <= n; ++i) {
    const auto& c = tab.coeffs(i);
    for (double v : c) {
      out.push_back(v);
    }
  }
}

struct FlattenedTables {
  std::vector<double> F;
  std::vector<double> rho;
  std::vector<double> z2r;
};

FlattenedTables flatten_eam(const tdmd::potentials::EamAlloyData& data) {
  FlattenedTables out;
  for (const auto& t : data.F_rho) {
    flatten_tab(t, out.F);
  }
  for (const auto& t : data.rho_r) {
    flatten_tab(t, out.rho);
  }
  for (const auto& t : data.z2r) {
    flatten_tab(t, out.z2r);
  }
  return out;
}

tg::BoxParams make_boxparams(const tdmd::Box& box, const tdmd::CellGrid& grid, double cutoff) {
  tg::BoxParams bp;
  bp.xlo = box.xlo;
  bp.ylo = box.ylo;
  bp.zlo = box.zlo;
  bp.lx = box.lx();
  bp.ly = box.ly();
  bp.lz = box.lz();
  bp.cell_x = grid.cell_x();
  bp.cell_y = grid.cell_y();
  bp.cell_z = grid.cell_z();
  bp.nx = grid.nx();
  bp.ny = grid.ny();
  bp.nz = grid.nz();
  bp.periodic_x = box.periodic_x;
  bp.periodic_y = box.periodic_y;
  bp.periodic_z = box.periodic_z;
  bp.cutoff = cutoff;
  bp.skin = 0.0;
  return bp;
}

tg::EamAlloyTablesHost make_tables_host(const tdmd::potentials::EamAlloyData& data,
                                        const FlattenedTables& flat) {
  tg::EamAlloyTablesHost tables;
  tables.n_species = data.species_names.size();
  tables.nrho = data.F_rho[0].size();
  tables.nr = data.rho_r[0].size();
  tables.npairs = tables.n_species * (tables.n_species + 1) / 2;
  tables.F_x0 = data.F_rho[0].x0();
  tables.F_dx = data.F_rho[0].dx();
  tables.r_x0 = data.rho_r[0].x0();
  tables.r_dx = data.rho_r[0].dx();
  tables.cutoff = data.cutoff;
  tables.F_coeffs = flat.F.data();
  tables.rho_coeffs = flat.rho.data();
  tables.z2r_coeffs = flat.z2r.data();
  return tables;
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

TEST_CASE("EamAlloyGpuMixed — Ni-Al B2 within D-M6-8 thresholds vs Fp64 reference",
          "[gpu][eam][mixed]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }

  const auto nial_path =
      std::filesystem::path(verify_potentials_dir()) / "NiAl_Mishin_2004.eam.alloy";
  if (!std::filesystem::exists(nial_path)) {
    SKIP("NiAl_Mishin_2004.eam.alloy fixture not available");
  }
  const auto data = tp::parse_eam_alloy(nial_path.string());
  REQUIRE(data.species_names.size() == 2u);

  NiAlFixture fx = make_nial_b2();
  REQUIRE(fx.atoms.size() == 1024u);
  perturb(fx.atoms, 0.03);

  const double cutoff = data.cutoff;
  const double skin = 0.2;

  tdmd::CellGrid grid;
  grid.build(fx.box, cutoff, skin);
  grid.bin(fx.atoms);

  const FlattenedTables flat = flatten_eam(data);
  const tg::BoxParams bp = make_boxparams(fx.box, grid, cutoff);
  const tg::EamAlloyTablesHost tables_h = make_tables_host(data, flat);

  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);

  // --- Fp64 reference GPU ---
  tdmd::AtomSoA atoms_ref = fx.atoms;
  for (std::size_t i = 0; i < atoms_ref.size(); ++i) {
    atoms_ref.fx[i] = atoms_ref.fy[i] = atoms_ref.fz[i] = 0.0;
  }
  tg::EamAlloyGpu gpu_ref;
  tg::EamAlloyGpuResult r_ref = gpu_ref.compute(atoms_ref.size(),
                                                atoms_ref.type.data(),
                                                atoms_ref.x.data(),
                                                atoms_ref.y.data(),
                                                atoms_ref.z.data(),
                                                grid.cell_count(),
                                                grid.cell_offsets().data(),
                                                grid.cell_atoms().data(),
                                                bp,
                                                tables_h,
                                                atoms_ref.fx.data(),
                                                atoms_ref.fy.data(),
                                                atoms_ref.fz.data(),
                                                pool,
                                                stream);

  // --- MixedFast GPU ---
  tdmd::AtomSoA atoms_mix = fx.atoms;
  for (std::size_t i = 0; i < atoms_mix.size(); ++i) {
    atoms_mix.fx[i] = atoms_mix.fy[i] = atoms_mix.fz[i] = 0.0;
  }
  tg::EamAlloyGpuMixed gpu_mix;
  tg::EamAlloyGpuResult r_mix = gpu_mix.compute(atoms_mix.size(),
                                                atoms_mix.type.data(),
                                                atoms_mix.x.data(),
                                                atoms_mix.y.data(),
                                                atoms_mix.z.data(),
                                                grid.cell_count(),
                                                grid.cell_offsets().data(),
                                                grid.cell_atoms().data(),
                                                bp,
                                                tables_h,
                                                atoms_mix.fx.data(),
                                                atoms_mix.fy.data(),
                                                atoms_mix.fz.data(),
                                                pool,
                                                stream);

  // --- Force comparison: max over components ---
  double worst_force = 0.0;
  double worst_ref = 0.0;
  double worst_mix = 0.0;
  std::size_t worst_idx = 0;
  int worst_k = 0;
  for (std::size_t i = 0; i < atoms_ref.size(); ++i) {
    const double fr[3] = {atoms_ref.fx[i], atoms_ref.fy[i], atoms_ref.fz[i]};
    const double fm[3] = {atoms_mix.fx[i], atoms_mix.fy[i], atoms_mix.fz[i]};
    for (int k = 0; k < 3; ++k) {
      const double num = std::abs(fr[k] - fm[k]);
      const double den = std::max(1.0, std::max(std::abs(fr[k]), std::abs(fm[k])));
      const double re = num / den;
      if (re > worst_force) {
        worst_force = re;
        worst_idx = i;
        worst_k = k;
        worst_ref = fr[k];
        worst_mix = fm[k];
      }
    }
  }
  INFO("worst force rel-diff = " << worst_force << " at atom " << worst_idx << " k=" << worst_k
                                 << " ref=" << worst_ref << " mix=" << worst_mix);
  REQUIRE(worst_force <= 1e-5);  // T6.8a achievable threshold (see file header)

  // --- Energy comparison ---
  const double pe_rel = rel(r_ref.potential_energy, r_mix.potential_energy);
  INFO("PE ref=" << r_ref.potential_energy << " mix=" << r_mix.potential_energy
                 << " rel=" << pe_rel);
  REQUIRE(pe_rel <= 1e-7);

  // --- Virial Voigt comparison: normalize each component by the max
  //     component so off-diagonals that vanish by symmetry (B2 has only
  //     diagonal stress) are judged against the physical scale, not
  //     against their own near-zero magnitude.
  double virial_scale = 1.0;
  for (std::size_t k = 0; k < 6; ++k) {
    virial_scale = std::max(virial_scale, std::abs(r_ref.virial[k]));
  }
  for (std::size_t k = 0; k < 6; ++k) {
    const double vr = std::abs(r_ref.virial[k] - r_mix.virial[k]) / virial_scale;
    INFO("virial[" << k << "] ref=" << r_ref.virial[k] << " mix=" << r_mix.virial[k]
                   << " rel-to-scale=" << vr);
    REQUIRE(vr <= 5e-6);
  }
}

TEST_CASE("EamAlloyGpuMixed — Al FCC single-species within D-M6-8 thresholds",
          "[gpu][eam][mixed]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }

  const auto data = tp::parse_eam_alloy(
      (std::filesystem::path(test_fixtures_dir()) / "Al_small.eam.alloy").string());
  REQUIRE(data.species_names.size() == 1u);

  // 6×6×6 Al FCC = 864 atoms at a=4.05 Å.
  tdmd::AtomSoA atoms_tmpl;
  tdmd::Box box;
  const double a = 4.05;
  const int n = 6;
  box.xlo = box.ylo = box.zlo = 0.0;
  box.xhi = box.yhi = box.zhi = a * n;
  box.periodic_x = box.periodic_y = box.periodic_z = true;
  const double basis[4][3] = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
  for (int iz = 0; iz < n; ++iz) {
    for (int iy = 0; iy < n; ++iy) {
      for (int ix = 0; ix < n; ++ix) {
        for (const auto& b : basis) {
          atoms_tmpl.add_atom(0U, (ix + b[0]) * a, (iy + b[1]) * a, (iz + b[2]) * a);
        }
      }
    }
  }
  perturb(atoms_tmpl, 0.05);
  REQUIRE(atoms_tmpl.size() == 864u);

  const double cutoff = data.cutoff;
  const double skin = 0.2;

  tdmd::CellGrid grid;
  grid.build(box, cutoff, skin);
  grid.bin(atoms_tmpl);

  const FlattenedTables flat = flatten_eam(data);
  const tg::BoxParams bp = make_boxparams(box, grid, cutoff);
  const tg::EamAlloyTablesHost tables_h = make_tables_host(data, flat);

  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);

  tdmd::AtomSoA atoms_ref = atoms_tmpl;
  for (std::size_t i = 0; i < atoms_ref.size(); ++i) {
    atoms_ref.fx[i] = atoms_ref.fy[i] = atoms_ref.fz[i] = 0.0;
  }
  tg::EamAlloyGpu gpu_ref;
  tg::EamAlloyGpuResult r_ref = gpu_ref.compute(atoms_ref.size(),
                                                atoms_ref.type.data(),
                                                atoms_ref.x.data(),
                                                atoms_ref.y.data(),
                                                atoms_ref.z.data(),
                                                grid.cell_count(),
                                                grid.cell_offsets().data(),
                                                grid.cell_atoms().data(),
                                                bp,
                                                tables_h,
                                                atoms_ref.fx.data(),
                                                atoms_ref.fy.data(),
                                                atoms_ref.fz.data(),
                                                pool,
                                                stream);

  tdmd::AtomSoA atoms_mix = atoms_tmpl;
  for (std::size_t i = 0; i < atoms_mix.size(); ++i) {
    atoms_mix.fx[i] = atoms_mix.fy[i] = atoms_mix.fz[i] = 0.0;
  }
  tg::EamAlloyGpuMixed gpu_mix;
  tg::EamAlloyGpuResult r_mix = gpu_mix.compute(atoms_mix.size(),
                                                atoms_mix.type.data(),
                                                atoms_mix.x.data(),
                                                atoms_mix.y.data(),
                                                atoms_mix.z.data(),
                                                grid.cell_count(),
                                                grid.cell_offsets().data(),
                                                grid.cell_atoms().data(),
                                                bp,
                                                tables_h,
                                                atoms_mix.fx.data(),
                                                atoms_mix.fy.data(),
                                                atoms_mix.fz.data(),
                                                pool,
                                                stream);

  double worst_force = 0.0;
  for (std::size_t i = 0; i < atoms_ref.size(); ++i) {
    const double fr[3] = {atoms_ref.fx[i], atoms_ref.fy[i], atoms_ref.fz[i]};
    const double fm[3] = {atoms_mix.fx[i], atoms_mix.fy[i], atoms_mix.fz[i]};
    for (int k = 0; k < 3; ++k) {
      const double num = std::abs(fr[k] - fm[k]);
      const double den = std::max(1.0, std::max(std::abs(fr[k]), std::abs(fm[k])));
      worst_force = std::max(worst_force, num / den);
    }
  }
  INFO("worst force rel-diff = " << worst_force);
  REQUIRE(worst_force <= 1e-5);
  REQUIRE(rel(r_ref.potential_energy, r_mix.potential_energy) <= 1e-7);
}

TEST_CASE("EamAlloyGpuMixed — compute_version monotone on repeat calls", "[gpu][eam][mixed]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }

  const auto data = tp::parse_eam_alloy(
      (std::filesystem::path(test_fixtures_dir()) / "Al_small.eam.alloy").string());

  // 4×4×4 Al FCC = 256 atoms.
  tdmd::AtomSoA atoms;
  tdmd::Box box;
  const double a = 4.05;
  const int n = 4;
  box.xlo = box.ylo = box.zlo = 0.0;
  box.xhi = box.yhi = box.zhi = a * n;
  box.periodic_x = box.periodic_y = box.periodic_z = true;
  const double basis[4][3] = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
  for (int iz = 0; iz < n; ++iz) {
    for (int iy = 0; iy < n; ++iy) {
      for (int ix = 0; ix < n; ++ix) {
        for (const auto& b : basis) {
          atoms.add_atom(0U, (ix + b[0]) * a, (iy + b[1]) * a, (iz + b[2]) * a);
        }
      }
    }
  }
  REQUIRE(atoms.size() == 256u);
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    atoms.fx[i] = atoms.fy[i] = atoms.fz[i] = 0.0;
  }

  tdmd::CellGrid grid;
  grid.build(box, data.cutoff, 0.2);
  grid.bin(atoms);

  const FlattenedTables flat = flatten_eam(data);
  const tg::BoxParams bp = make_boxparams(box, grid, data.cutoff);
  const tg::EamAlloyTablesHost tables_h = make_tables_host(data, flat);

  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);

  tg::EamAlloyGpuMixed gpu;
  REQUIRE(gpu.compute_version() == 0u);

  (void) gpu.compute(atoms.size(),
                     atoms.type.data(),
                     atoms.x.data(),
                     atoms.y.data(),
                     atoms.z.data(),
                     grid.cell_count(),
                     grid.cell_offsets().data(),
                     grid.cell_atoms().data(),
                     bp,
                     tables_h,
                     atoms.fx.data(),
                     atoms.fy.data(),
                     atoms.fz.data(),
                     pool,
                     stream);
  REQUIRE(gpu.compute_version() == 1u);

  for (std::size_t i = 0; i < atoms.size(); ++i) {
    atoms.fx[i] = atoms.fy[i] = atoms.fz[i] = 0.0;
  }
  (void) gpu.compute(atoms.size(),
                     atoms.type.data(),
                     atoms.x.data(),
                     atoms.y.data(),
                     atoms.z.data(),
                     grid.cell_count(),
                     grid.cell_offsets().data(),
                     grid.cell_atoms().data(),
                     bp,
                     tables_h,
                     atoms.fx.data(),
                     atoms.fy.data(),
                     atoms.fz.data(),
                     pool,
                     stream);
  REQUIRE(gpu.compute_version() == 2u);
}

#else  // !TDMD_BUILD_CUDA

TEST_CASE("EamAlloyGpuMixed — CPU-only stub path", "[gpu][eam][mixed][cpu]") {
  // On CPU-only builds, the DevicePool ctor throws before we can reach the
  // mixed compute() — same behaviour as Fp64 reference on CPU-only. Covered
  // by test_eam_alloy_gpu.cpp. No duplicate assertion here.
  SUCCEED("CPU-only build — see test_eam_alloy_gpu.cpp for CPU-stub coverage");
}

#endif  // TDMD_BUILD_CUDA
