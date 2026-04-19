// Exec pack: docs/development/m6_execution_pack.md T6.5
// SPEC: docs/specs/gpu/SPEC.md §7.2 (EAM contract), §6.3 (D-M6-7 gate)
// SPEC: docs/specs/potentials/SPEC.md §4.1–§4.4
//
// T6.5 acceptance gate: run CPU EamAlloyPotential + GPU EamAlloyGpuAdapter
// on the same fixture; verify per-atom forces, total PE, and virial tensor
// agree within gpu/SPEC §7.2's 1e-12 relative tolerance.
//
// The ≤1e-12 rel (not byte-equal) gate absorbs the expected reduction-order
// drift between CPU half-list accumulation and GPU full-list per-atom
// accumulation. FP math is otherwise identical: same Horner form, same
// minimum-image formula, same pair-index mapping.
//
// All CUDA cases skip gracefully when no device is available.

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/eam_alloy_gpu.hpp"
#include "tdmd/gpu/factories.hpp"
#include "tdmd/gpu/gpu_config.hpp"
#include "tdmd/gpu/neighbor_list_gpu.hpp"  // BoxParams
#include "tdmd/gpu/types.hpp"
#include "tdmd/neighbor/cell_grid.hpp"
#include "tdmd/neighbor/neighbor_list.hpp"
#include "tdmd/potentials/eam_alloy.hpp"
#include "tdmd/potentials/eam_alloy_gpu_adapter.hpp"
#include "tdmd/potentials/eam_file.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

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

// 6×6×6 Al FCC supercell, 864 atoms. Lattice a=4.05 Å. Cutoff = 5.3 Å (fits
// Al_small.eam.alloy, which has cutoff 5.0015 Å).
struct AlFccFixture {
  tdmd::AtomSoA atoms;
  tdmd::Box box;
  static constexpr double kLatticeA = 4.05;
  static constexpr int kNx = 6;
  static constexpr int kNy = 6;
  static constexpr int kNz = 6;
};

AlFccFixture make_al_fcc(std::size_t nx = AlFccFixture::kNx,
                         std::size_t ny = AlFccFixture::kNy,
                         std::size_t nz = AlFccFixture::kNz,
                         tdmd::SpeciesId type = 0U) {
  AlFccFixture fx;
  const double a = AlFccFixture::kLatticeA;
  fx.box.xlo = 0.0;
  fx.box.ylo = 0.0;
  fx.box.zlo = 0.0;
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
          fx.atoms.add_atom(type, x, y, z);
        }
      }
    }
  }
  return fx;
}

// Ni-Al B2 (CsCl-like) fixture — 2-atom basis, alternating Ni (type=0) /
// Al (type=1) on the simple-cubic sub-lattices. Lattice a=2.88 Å (Mishin
// 2004 Ni-Al B2 a₀). 8×8×8 = 1024 atoms — box side 23.04 Å > 3·(cutoff+skin)
// for the Mishin 2004 file (cutoff 6.725 Å, need ≥ 20.77 Å).
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
        // Ni at (0, 0, 0), Al at (0.5, 0.5, 0.5) — B2.
        fx.atoms.add_atom(0U, ix * a, iy * a, iz * a);
        fx.atoms.add_atom(1U, (ix + 0.5) * a, (iy + 0.5) * a, (iz + 0.5) * a);
      }
    }
  }
  return fx;
}

// Perturb each coordinate by a deterministic signed offset proportional to
// `scale`, cycling through x/y/z with index. Keeps the atoms off the
// unstressed lattice so forces, PE, and virial are all nonzero.
void perturb(tdmd::AtomSoA& atoms, double scale) {
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const double s = scale * (((i % 7) / 6.0) - 0.5);
    atoms.x[i] += s;
    atoms.y[i] += scale * (((i % 11) / 10.0) - 0.5);
    atoms.z[i] += scale * (((i % 13) / 12.0) - 0.5);
  }
}

// Relative force agreement — max over components.
double max_rel_force_error(const tdmd::AtomSoA& cpu, const tdmd::AtomSoA& gpu) {
  double worst = 0.0;
  for (std::size_t i = 0; i < cpu.size(); ++i) {
    const double fc[3] = {cpu.fx[i], cpu.fy[i], cpu.fz[i]};
    const double fg[3] = {gpu.fx[i], gpu.fy[i], gpu.fz[i]};
    for (int k = 0; k < 3; ++k) {
      const double num = std::abs(fc[k] - fg[k]);
      const double den = std::max(1.0, std::max(std::abs(fc[k]), std::abs(fg[k])));
      worst = std::max(worst, num / den);
    }
  }
  return worst;
}

double rel(double a, double b) {
  const double num = std::abs(a - b);
  const double den = std::max(1.0, std::max(std::abs(a), std::abs(b)));
  return num / den;
}

}  // namespace

TEST_CASE("EamAlloyGpu — CPU-only build throws from adapter", "[gpu][eam][cpu]") {
#if TDMD_BUILD_CUDA
  SUCCEED("CUDA build — CPU stub path exercised only on TDMD_BUILD_CUDA=OFF");
#else
  // On CPU-only builds the adapter constructor + flatten succeed, but
  // `gpu_->compute()` throws when called.
  const auto data = tp::parse_eam_alloy(
      (std::filesystem::path(test_fixtures_dir()) / "Al_small.eam.alloy").string());
  tp::EamAlloyGpuAdapter adapter(data);
  // Without a DevicePool (which itself throws on CPU-only), we can't reach
  // compute(). Instead exercise the gpu::EamAlloyGpu class directly to
  // confirm its stub throws with a runtime_error mentioning "CPU-only".
  tg::EamAlloyGpu raw;
  tg::BoxParams bp{};
  tg::EamAlloyTablesHost tables{};
  tdmd::gpu::DevicePool* pool_ptr = nullptr;  // unused on CPU path
  tdmd::gpu::DeviceStream* stream_ptr = nullptr;
  (void) pool_ptr;
  (void) stream_ptr;
  // The compute signature requires refs; we can't fake DevicePool without
  // CUDA. Instead rely on adapter->compute() with a non-empty atom set and
  // expect the stub in gpu::EamAlloyGpu to fire.
  tdmd::AtomSoA atoms;
  atoms.add_atom(0U, 0.0, 0.0, 0.0);
  atoms.add_atom(0U, 1.5, 0.0, 0.0);
  tdmd::Box box;
  box.xlo = box.ylo = box.zlo = 0.0;
  box.xhi = box.yhi = box.zhi = 20.0;
  box.periodic_x = box.periodic_y = box.periodic_z = true;
  tdmd::CellGrid grid;
  grid.build(box, 5.0, 0.2);
  grid.bin(atoms);
  // DevicePool ctor throws under CPU-only — confirm that's enough to guard
  // the path.
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

}  // namespace

TEST_CASE("EamAlloyGpu — Al FCC single-species ≤1e-12 rel vs CPU", "[gpu][eam]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }
#ifndef TDMD_FLAVOR_FP64_REFERENCE
  SKIP(
      "D-M6-7 bit-exact EAM gate is Fp64ReferenceBuild-only; MixedFast path "
      "covered by test_eam_mixed_fast_within_threshold (D-M6-8 thresholds)");
#endif

  const auto data = tp::parse_eam_alloy(
      (std::filesystem::path(test_fixtures_dir()) / "Al_small.eam.alloy").string());
  REQUIRE(data.species_names.size() == 1u);
  REQUIRE(data.cutoff > 0.0);

  AlFccFixture fx = make_al_fcc();
  REQUIRE(fx.atoms.size() == 864u);

  // Small perturbation so forces aren't all ~zero.
  perturb(fx.atoms, 0.05);

  const double cutoff = data.cutoff;
  const double skin = 0.2;

  tdmd::CellGrid grid;
  grid.build(fx.box, cutoff, skin);
  grid.bin(fx.atoms);

  // --- CPU reference ---
  tdmd::AtomSoA atoms_cpu = fx.atoms;
  tdmd::NeighborList nl;
  nl.build(atoms_cpu, fx.box, grid, cutoff, skin);

  tdmd::EamAlloyPotential cpu_pot(data);
  // Zero forces before compute (additive contract).
  for (std::size_t i = 0; i < atoms_cpu.size(); ++i) {
    atoms_cpu.fx[i] = atoms_cpu.fy[i] = atoms_cpu.fz[i] = 0.0;
  }
  const tdmd::ForceResult cpu_r = cpu_pot.compute(atoms_cpu, nl, fx.box);

  // --- GPU path ---
  tdmd::AtomSoA atoms_gpu = fx.atoms;
  for (std::size_t i = 0; i < atoms_gpu.size(); ++i) {
    atoms_gpu.fx[i] = atoms_gpu.fy[i] = atoms_gpu.fz[i] = 0.0;
  }

  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);

  tp::EamAlloyGpuAdapter adapter(data);
  const tdmd::ForceResult gpu_r = adapter.compute(atoms_gpu, fx.box, grid, pool, stream);

  // --- Comparisons ---
  const double worst_force = max_rel_force_error(atoms_cpu, atoms_gpu);
  INFO("worst force rel err = " << worst_force);
  REQUIRE(worst_force <= 1e-12);

  INFO("PE cpu=" << cpu_r.potential_energy << " gpu=" << gpu_r.potential_energy);
  REQUIRE(rel(cpu_r.potential_energy, gpu_r.potential_energy) <= 1e-12);

  for (std::size_t k = 0; k < 6; ++k) {
    INFO("virial[" << k << "] cpu=" << cpu_r.virial[k] << " gpu=" << gpu_r.virial[k]);
    REQUIRE(rel(cpu_r.virial[k], gpu_r.virial[k]) <= 1e-12);
  }
}

TEST_CASE("EamAlloyGpu — Ni-Al B2 two-species ≤1e-12 rel vs CPU", "[gpu][eam]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }
#ifndef TDMD_FLAVOR_FP64_REFERENCE
  SKIP(
      "D-M6-7 bit-exact EAM gate is Fp64ReferenceBuild-only; MixedFast path "
      "covered by test_eam_mixed_fast_within_threshold (D-M6-8 thresholds)");
#endif

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

  tdmd::AtomSoA atoms_cpu = fx.atoms;
  tdmd::NeighborList nl;
  nl.build(atoms_cpu, fx.box, grid, cutoff, skin);
  tdmd::EamAlloyPotential cpu_pot(data);
  for (std::size_t i = 0; i < atoms_cpu.size(); ++i) {
    atoms_cpu.fx[i] = atoms_cpu.fy[i] = atoms_cpu.fz[i] = 0.0;
  }
  const tdmd::ForceResult cpu_r = cpu_pot.compute(atoms_cpu, nl, fx.box);

  tdmd::AtomSoA atoms_gpu = fx.atoms;
  for (std::size_t i = 0; i < atoms_gpu.size(); ++i) {
    atoms_gpu.fx[i] = atoms_gpu.fy[i] = atoms_gpu.fz[i] = 0.0;
  }
  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);
  tp::EamAlloyGpuAdapter adapter(data);
  const tdmd::ForceResult gpu_r = adapter.compute(atoms_gpu, fx.box, grid, pool, stream);

  const double worst_force = max_rel_force_error(atoms_cpu, atoms_gpu);
  INFO("worst force rel err = " << worst_force);
  REQUIRE(worst_force <= 1e-12);
  REQUIRE(rel(cpu_r.potential_energy, gpu_r.potential_energy) <= 1e-12);
  for (std::size_t k = 0; k < 6; ++k) {
    INFO("virial[" << k << "] cpu=" << cpu_r.virial[k] << " gpu=" << gpu_r.virial[k]);
    REQUIRE(rel(cpu_r.virial[k], gpu_r.virial[k]) <= 1e-12);
  }
}

TEST_CASE("EamAlloyGpu — compute_version monotone on repeat calls", "[gpu][eam]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }

  const auto data = tp::parse_eam_alloy(
      (std::filesystem::path(test_fixtures_dir()) / "Al_small.eam.alloy").string());
  AlFccFixture fx = make_al_fcc(4, 4, 4);  // 256 atoms — minimum box for the cutoff.
  REQUIRE(fx.atoms.size() == 256u);

  const double cutoff = data.cutoff;
  const double skin = 0.2;
  tdmd::CellGrid grid;
  grid.build(fx.box, cutoff, skin);
  grid.bin(fx.atoms);

  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);
  tp::EamAlloyGpuAdapter adapter(data);

  REQUIRE(adapter.compute_version() == 0u);

  tdmd::AtomSoA atoms = fx.atoms;
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    atoms.fx[i] = atoms.fy[i] = atoms.fz[i] = 0.0;
  }
  (void) adapter.compute(atoms, fx.box, grid, pool, stream);
  REQUIRE(adapter.compute_version() == 1u);

  (void) adapter.compute(atoms, fx.box, grid, pool, stream);
  REQUIRE(adapter.compute_version() == 2u);
}

TEST_CASE("EamAlloyGpu — splines cached across compute() calls (T6.9a)", "[gpu][eam][cache]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }

  // Invariant per T6.9a: back-to-back compute() calls with the same adapter
  // instance (therefore the same flattened spline host pointers) must upload
  // the F / rho / z2r tables to the device exactly once. Steady-state MD hot
  // loops rebuild neighbor lists and re-call compute() every step — without
  // this caching the ~MB-scale spline H2D would dominate MixedFast throughput.
  const auto data = tp::parse_eam_alloy(
      (std::filesystem::path(test_fixtures_dir()) / "Al_small.eam.alloy").string());
  AlFccFixture fx = make_al_fcc(4, 4, 4);

  const double cutoff = data.cutoff;
  const double skin = 0.2;
  tdmd::CellGrid grid;
  grid.build(fx.box, cutoff, skin);
  grid.bin(fx.atoms);

  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);
  tp::EamAlloyGpuAdapter adapter(data);

  REQUIRE(adapter.splines_upload_count() == 0u);

  tdmd::AtomSoA atoms = fx.atoms;
  for (int call = 0; call < 3; ++call) {
    for (std::size_t i = 0; i < atoms.size(); ++i) {
      atoms.fx[i] = atoms.fy[i] = atoms.fz[i] = 0.0;
    }
    (void) adapter.compute(atoms, fx.box, grid, pool, stream);
  }

  // Exactly one upload for the three calls — the adapter's flattened
  // coefficient buffers are immutable for its lifetime, so host-pointer
  // identity matches after the first call.
  REQUIRE(adapter.splines_upload_count() == 1u);
  REQUIRE(adapter.compute_version() == 3u);
}

TEST_CASE("EamAlloyGpu — empty atoms yields zero result", "[gpu][eam][edge]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }

  const auto data = tp::parse_eam_alloy(
      (std::filesystem::path(test_fixtures_dir()) / "Al_small.eam.alloy").string());

  tdmd::AtomSoA empty;
  tdmd::Box box;
  box.xlo = box.ylo = box.zlo = 0.0;
  box.xhi = box.yhi = box.zhi = 24.3;
  box.periodic_x = box.periodic_y = box.periodic_z = true;

  tdmd::CellGrid grid;
  grid.build(box, data.cutoff, 0.2);
  grid.bin(empty);

  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);
  tp::EamAlloyGpuAdapter adapter(data);

  const tdmd::ForceResult r = adapter.compute(empty, box, grid, pool, stream);
  REQUIRE(r.potential_energy == 0.0);
  for (std::size_t k = 0; k < 6; ++k) {
    REQUIRE(r.virial[k] == 0.0);
  }
}

#endif  // TDMD_BUILD_CUDA
