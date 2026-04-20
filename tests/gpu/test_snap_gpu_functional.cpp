// Exec pack: docs/development/m8_execution_pack.md T8.6b
// SPEC: docs/specs/gpu/SPEC.md §7.3 (SNAP GPU contract),
//       docs/specs/potentials/SPEC.md §6
//
// T8.6b acceptance gate (functional correctness). The D-M8-13 byte-exact
// 1e-12-rel GPU-vs-CPU gate is T8.7's job — this file covers what T8.6b
// actually promises:
//
//   1. W_2940 fixture + small perturbed BCC W slab: GPU compute() runs to
//      completion, PE is finite, all forces are finite, virial is finite.
//   2. Newton's 3rd law: |Σ F_i| per component ≤ 1e-9 of max|F_i|. Catches
//      asymmetric accumulation between `fij_own` и `fij_peer` (the full-list
//      bridge in pass 3 must replay the CPU's half-list Newton-3 sum).
//   3. compute_version monotone — first call 0→1, second call 1→2.
//   4. Empty atoms → zero result (matches EAM empty-atoms edge case).
//   5. GPU ≈ CPU within 1e-10 rel (loose tolerance; T8.7 tightens to 1e-12).
//      Serves as an early-warning signal that no kernel produces garbage
//      results в the middle of build matrix — catches sign errors, stride
//      mistakes, off-by-one on index tables.
//
// Self-skips with exit 77 когда нет CUDA device or LAMMPS submodule is
// uninitialized (Option A convention).

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/factories.hpp"
#include "tdmd/gpu/gpu_config.hpp"
#include "tdmd/neighbor/cell_grid.hpp"
#include "tdmd/neighbor/neighbor_list.hpp"
#include "tdmd/potentials/snap.hpp"
#include "tdmd/potentials/snap_file.hpp"
#include "tdmd/potentials/snap_gpu_adapter.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

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
                 "[test_snap_gpu_functional] SKIP: LAMMPS submodule not initialized at %s\n",
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

// 2·nrep³ BCC W lattice filling an nrep×a cubic box. W BCC NN = a·√3/2;
// a = 3.18 Å (LAMMPS W_2940 example default) gives NN = 2.754 Å < rcut = 4.73 Å.
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

void zero_forces(tdmd::AtomSoA& atoms) {
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    atoms.fx[i] = 0.0;
    atoms.fy[i] = 0.0;
    atoms.fz[i] = 0.0;
  }
}

// Deterministic scatter — breaks pristine-lattice symmetry so each atom
// sees a non-trivial neighbour cloud. Magnitude 0.01 Å (~0.3 % of NN).
void apply_tiny_rattle(tdmd::AtomSoA& atoms) {
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const double ph = static_cast<double>(i) * 0.1;
    atoms.x[i] += 0.01 * std::sin(ph);
    atoms.y[i] += 0.01 * std::sin(ph + 1.0);
    atoms.z[i] += 0.01 * std::sin(ph + 2.0);
  }
}

double max_abs_force(const tdmd::AtomSoA& atoms) {
  double m = 0.0;
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    m = std::max(m, std::abs(atoms.fx[i]));
    m = std::max(m, std::abs(atoms.fy[i]));
    m = std::max(m, std::abs(atoms.fz[i]));
  }
  return m;
}

struct Accum {
  double sx = 0.0;
  double sy = 0.0;
  double sz = 0.0;
};
Accum sum_forces(const tdmd::AtomSoA& atoms) {
  Accum s;
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    s.sx += atoms.fx[i];
    s.sy += atoms.fy[i];
    s.sz += atoms.fz[i];
  }
  return s;
}

double rel(double a, double b) {
  const double num = std::abs(a - b);
  const double den = std::max(1.0, std::max(std::abs(a), std::abs(b)));
  return num / den;
}

double max_rel_force_error(const tdmd::AtomSoA& a, const tdmd::AtomSoA& b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double fa[3] = {a.fx[i], a.fy[i], a.fz[i]};
    const double fb[3] = {b.fx[i], b.fy[i], b.fz[i]};
    for (int k = 0; k < 3; ++k) {
      const double num = std::abs(fa[k] - fb[k]);
      const double den = std::max(1.0, std::max(std::abs(fa[k]), std::abs(fb[k])));
      worst = std::max(worst, num / den);
    }
  }
  return worst;
}

}  // namespace

TEST_CASE("SnapGpu — CPU-only build throws from SnapGpu::compute stub", "[gpu][snap][t8.6b][cpu]") {
#if TDMD_BUILD_CUDA
  SUCCEED("CUDA build — CPU stub path exercised only on TDMD_BUILD_CUDA=OFF");
#else
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

TEST_CASE("SnapGpu — BCC W 3x3x3 (54 atoms) produces finite + Newton-3 forces",
          "[gpu][snap][t8.6b]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }

  const tp::SnapData data = load_w_fixture();

  constexpr double kLatticeA = 3.18;
  constexpr int kNrep = 5;  // box = 15.9 Å > 3·(rcut+skin) = 14.8 Å
  tdmd::Box box = make_cubic_box(kLatticeA * kNrep);
  tdmd::AtomSoA atoms;
  add_bcc_W(atoms, kNrep, kLatticeA);
  REQUIRE(atoms.size() == 250u);
  apply_tiny_rattle(atoms);
  zero_forces(atoms);

  const double cutoff = data.params.rcutfac;
  const double skin = 0.2;

  tdmd::CellGrid grid;
  grid.build(box, cutoff, skin);
  grid.bin(atoms);

  // Order matters: pool → stream → adapter so destruction runs
  // adapter → stream → pool and the adapter's DevicePtr deleters reach
  // the still-alive pool.
  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);
  tp::SnapGpuAdapter adapter(data);

  const tdmd::ForceResult r = adapter.compute(atoms, box, grid, pool, stream);

  // --- PE + forces + virial finite (gated behind a single REQUIRE so Catch2
  // doesn't allocate 750× expansion strings in a tight loop).
  REQUIRE(std::isfinite(r.potential_energy));
  bool forces_all_finite = true;
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    if (!std::isfinite(atoms.fx[i]) || !std::isfinite(atoms.fy[i]) || !std::isfinite(atoms.fz[i])) {
      forces_all_finite = false;
      break;
    }
  }
  REQUIRE(forces_all_finite);
  bool virial_finite = true;
  for (std::size_t k = 0; k < 6; ++k) {
    if (!std::isfinite(r.virial[k])) {
      virial_finite = false;
    }
  }
  REQUIRE(virial_finite);

  // --- Newton-3: Σ F_i ≈ 0 to 1e-9 of max|F| ---
  const Accum s = sum_forces(atoms);
  const double m = max_abs_force(atoms);
  REQUIRE(m > 0.0);  // Not all zero; we actually got forces.
  const double tol = 1e-9 * std::max(1.0, m);
  INFO("max|F| = " << m << ", Σ F = (" << s.sx << ", " << s.sy << ", " << s.sz << ")");
  REQUIRE(std::abs(s.sx) < tol);
  REQUIRE(std::abs(s.sy) < tol);
  REQUIRE(std::abs(s.sz) < tol);

  REQUIRE(adapter.compute_version() == 1u);
}

TEST_CASE("SnapGpu — CPU reference agrees within 1e-10 rel on BCC W", "[gpu][snap][t8.6b][diff]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }
#ifdef TDMD_FLAVOR_MIXED_FAST_SNAP_ONLY
  // D-M8-8 supersedes the 1e-10 loose diff gate — SNAP adapter dispatches to
  // SnapGpuMixed (narrow-FP32 pair-math) on this flavor, and the D-M8-8
  // 1e-5/1e-7/5e-6 envelope is exercised by
  // test_snap_mixed_fast_within_threshold.
  SUCCEED("MixedFastSnapOnlyBuild — D-M8-8 gate supersedes D-M8-13 for SNAP");
  return;
#else

  const tp::SnapData data = load_w_fixture();

  constexpr double kLatticeA = 3.18;
  constexpr int kNrep = 5;  // box = 15.9 Å > 3·(rcut+skin) = 14.8 Å
  tdmd::Box box = make_cubic_box(kLatticeA * kNrep);
  tdmd::AtomSoA base;
  add_bcc_W(base, kNrep, kLatticeA);
  apply_tiny_rattle(base);

  const double cutoff = data.params.rcutfac;
  const double skin = 0.2;

  tdmd::CellGrid grid;
  grid.build(box, cutoff, skin);
  grid.bin(base);

  // --- CPU reference ---
  tdmd::AtomSoA atoms_cpu = base;
  tdmd::NeighborList nl;
  nl.build(atoms_cpu, box, grid, cutoff, skin);
  tdmd::SnapPotential cpu_pot(data);
  zero_forces(atoms_cpu);
  const tdmd::ForceResult cpu_r = cpu_pot.compute(atoms_cpu, nl, box);

  // --- GPU ---
  tdmd::AtomSoA atoms_gpu = base;
  zero_forces(atoms_gpu);
  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);
  tp::SnapGpuAdapter adapter(data);
  const tdmd::ForceResult gpu_r = adapter.compute(atoms_gpu, box, grid, pool, stream);

  INFO("PE cpu=" << cpu_r.potential_energy << " gpu=" << gpu_r.potential_energy);
  REQUIRE(rel(cpu_r.potential_energy, gpu_r.potential_energy) <= 1e-10);

  const double worst_force = max_rel_force_error(atoms_cpu, atoms_gpu);
  INFO("worst force rel err = " << worst_force);
  REQUIRE(worst_force <= 1e-10);

  for (std::size_t k = 0; k < 6; ++k) {
    INFO("virial[" << k << "] cpu=" << cpu_r.virial[k] << " gpu=" << gpu_r.virial[k]);
    REQUIRE(rel(cpu_r.virial[k], gpu_r.virial[k]) <= 1e-10);
  }
#endif  // TDMD_FLAVOR_MIXED_FAST_SNAP_ONLY
}

TEST_CASE("SnapGpu — compute_version monotone across repeat calls", "[gpu][snap][t8.6b]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }

  const tp::SnapData data = load_w_fixture();

  constexpr double kLatticeA = 3.18;
  constexpr int kNrep = 5;  // box = 15.9 Å > 3·(rcut+skin) = 14.8 Å
  tdmd::Box box = make_cubic_box(kLatticeA * kNrep);
  tdmd::AtomSoA atoms;
  add_bcc_W(atoms, kNrep, kLatticeA);
  apply_tiny_rattle(atoms);

  const double cutoff = data.params.rcutfac;
  tdmd::CellGrid grid;
  grid.build(box, cutoff, 0.2);
  grid.bin(atoms);

  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);
  tp::SnapGpuAdapter adapter(data);

  REQUIRE(adapter.compute_version() == 0u);
  zero_forces(atoms);
  (void) adapter.compute(atoms, box, grid, pool, stream);
  REQUIRE(adapter.compute_version() == 1u);
  zero_forces(atoms);
  (void) adapter.compute(atoms, box, grid, pool, stream);
  REQUIRE(adapter.compute_version() == 2u);
}

TEST_CASE("SnapGpu — empty atoms yields zero result", "[gpu][snap][t8.6b][edge]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }

  const tp::SnapData data = load_w_fixture();

  tdmd::AtomSoA empty;
  tdmd::Box box = make_cubic_box(20.0);

  tdmd::CellGrid grid;
  grid.build(box, data.params.rcutfac, 0.2);
  grid.bin(empty);

  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);
  tp::SnapGpuAdapter adapter(data);

  const tdmd::ForceResult r = adapter.compute(empty, box, grid, pool, stream);
  REQUIRE(r.potential_energy == 0.0);
  for (std::size_t k = 0; k < 6; ++k) {
    REQUIRE(r.virial[k] == 0.0);
  }
}

#endif  // TDMD_BUILD_CUDA
