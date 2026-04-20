// Exec pack: docs/development/m8_execution_pack.md T8.7
// SPEC: docs/specs/gpu/SPEC.md §7.5 (SNAP GPU),
//       docs/specs/verify/SPEC.md §4.7 (t6_snap_tungsten)
// Threshold registry: verify/thresholds/thresholds.yaml
//   benchmarks.t6_snap_tungsten.gpu_fp64_vs_cpu_fp64.forces_relative = 1.0e-12
//
// T8.7 acceptance gate: D-M6-7 byte-exact chain extended to SNAP.
// TDMD SnapPotential GPU FP64 Reference ≡ TDMD SnapPotential CPU FP64 Reference
// at ≤ 1e-12 rel on T6 tungsten ~2000-atom rattled BCC. The T8.6b functional
// test already measured 1.3e-14 rel on 250 atoms; 2000 atoms preserves two
// decades of headroom per registry rationale (§4 of
// mixed_fast_snap_only_rationale.md + §6.7 of potentials/SPEC.md).
//
// Gate scope (what this file promises):
//   1. Per-atom force max L∞ rel diff ≤ 1e-12 (threshold registry).
//   2. Total PE rel diff ≤ 1e-12.
//   3. Total virial rel diff ≤ 1e-12 on all six Voigt components.
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
                 "[test_snap_gpu_bit_exact] SKIP: LAMMPS submodule not initialized at %s\n",
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

// 2·nrep³ BCC W lattice filling an nrep×a cubic box.
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

ForceDiff max_rel_force_diff(const tdmd::AtomSoA& a, const tdmd::AtomSoA& b) {
  ForceDiff d;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double fa[3] = {a.fx[i], a.fy[i], a.fz[i]};
    const double fb[3] = {b.fx[i], b.fy[i], b.fz[i]};
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

}  // namespace

#if TDMD_BUILD_CUDA

namespace {

bool cuda_device_available() noexcept {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  return err == cudaSuccess && count > 0;
}

}  // namespace

// D-M6-7 chain extension (D-M8-13). Threshold: 1.0e-12 rel on forces + PE +
// virial — per verify/thresholds/thresholds.yaml t6_snap_tungsten.gpu_fp64_vs_cpu_fp64.
TEST_CASE("SnapGpu — T6 tungsten bit-exact gate: GPU FP64 ≡ CPU FP64 within 1e-12 rel",
          "[gpu][snap][t8.7][bit_exact][t6]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }
#ifdef TDMD_FLAVOR_MIXED_FAST_SNAP_ONLY
  // D-M8-8 supersedes D-M8-13 on this flavor — SNAP adapter dispatches to
  // SnapGpuMixed (narrow-FP32 pair-math), so CPU FP64 ≡ GPU at 1e-12 is
  // structurally impossible. The D-M8-8 1e-5/1e-7/5e-6 gate is exercised by
  // test_snap_mixed_fast_within_threshold on this flavor.
  SUCCEED("MixedFastSnapOnlyBuild — D-M8-8 gate supersedes D-M8-13 for SNAP");
  return;
#else

  const tp::SnapData data = load_w_fixture();

  // 10×10×10 BCC = 2000 atoms. Spec §4.7 reference is "8×8×8 BCC (2048-atom)"
  // which is slightly ambiguous (8³·2 = 1024 atoms on canonical BCC counting);
  // 10³·2 = 2000 atoms sits well inside the T6 envelope and comfortably
  // exceeds 3·(rcut+skin) = 14.8 Å at box = 31.8 Å.
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

  // --- CPU FP64 reference ---
  tdmd::AtomSoA atoms_cpu = base;
  tdmd::NeighborList nl;
  nl.build(atoms_cpu, box, grid, cutoff, skin);
  tdmd::SnapPotential cpu_pot(data);
  zero_forces(atoms_cpu);
  const tdmd::ForceResult cpu_r = cpu_pot.compute(atoms_cpu, nl, box);

  // --- GPU FP64 --- (pool → stream → adapter lifetime order; gpu/SPEC §7.5).
  tdmd::AtomSoA atoms_gpu = base;
  zero_forces(atoms_gpu);
  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 16;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);
  tp::SnapGpuAdapter adapter(data);
  const tdmd::ForceResult gpu_r = adapter.compute(atoms_gpu, box, grid, pool, stream);

  constexpr double kTol = 1.0e-12;

  // --- Total PE ---
  const double pe_rel = rel(cpu_r.potential_energy, gpu_r.potential_energy);
  INFO("PE cpu=" << cpu_r.potential_energy << " gpu=" << gpu_r.potential_energy
                 << " rel=" << pe_rel);
  REQUIRE(pe_rel <= kTol);

  // --- Per-atom forces (L∞) ---
  const ForceDiff fd = max_rel_force_diff(atoms_cpu, atoms_gpu);
  INFO("worst force rel = " << fd.worst << " at atom " << fd.worst_atom << " comp " << fd.worst_k);
  REQUIRE(fd.worst <= kTol);

  // --- Total virial (six Voigt components) ---
  for (std::size_t k = 0; k < 6; ++k) {
    const double v_rel = rel(cpu_r.virial[k], gpu_r.virial[k]);
    INFO("virial[" << k << "] cpu=" << cpu_r.virial[k] << " gpu=" << gpu_r.virial[k]
                   << " rel=" << v_rel);
    REQUIRE(v_rel <= kTol);
  }
#endif  // TDMD_FLAVOR_MIXED_FAST_SNAP_ONLY
}

#else  // !TDMD_BUILD_CUDA

TEST_CASE("SnapGpu bit-exact gate — CPU-only build skips gracefully", "[gpu][snap][t8.7][cpu]") {
  SUCCEED("CPU-only build — gate requires CUDA");
}

#endif  // TDMD_BUILD_CUDA
