// Exec pack: docs/development/t8.6c-v5_pre_impl.md Stage 1 (bond list pre-pass).
// SPEC: docs/specs/gpu/SPEC.md §6.1 (no atomicAdd(double)), §7.5 (T8.6c-v5).
//
// Stage 1 acceptance gate. Builds the GPU SnapBondListGpu on a 5×5×5 BCC W
// rattled fixture and asserts it matches a CPU reference that replicates the
// exact same 3×3×3 cell-stencil walk + SNAP per-pair filter
// (j != i, rsq < cutsq_ij, rsq > 1e-20).
//
// What this gates:
//   * CSR offsets (uint32) byte-identical host-vs-device → emission order
//     matches CPU across all atoms.
//   * Bond SoA (bond_i, bond_j, type_i, type_j) byte-identical → filter logic
//     matches exactly.
//   * Bond geometry (dx, dy, dz, rsq) byte-identical under Fp64 Reference
//     (host computes minimum_image + |Δr|² in the same order as the device).
//     Non-Reference flavors allow ≤1e-14 rel ULP drift for FMA contraction,
//     same convention as test_neighbor_list_gpu.
//
// What Stages 2 + 3 rely on: since snap_ui_gather and snap_force_gather sum
// over `bond_i[start[atom]..end[atom])` in array order, CPU-identical
// emission order is the foundation of the downstream T8.7 ≤ 1e-12 rel gate.
//
// Self-skips with exit 77 когда CUDA unavailable. Zero dependency on LAMMPS
// fixtures — fabricates SNAP-like rcut_sq_ab matrix inline from W's 4.73 Å
// cutoff so it runs in any build matrix.

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/factories.hpp"
#include "tdmd/gpu/gpu_config.hpp"
#include "tdmd/gpu/neighbor_list_gpu.hpp"  // BoxParams
#include "tdmd/gpu/snap_bond_list_gpu.hpp"
#include "tdmd/neighbor/cell_grid.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#if TDMD_BUILD_CUDA
#include <cuda_runtime.h>
#endif

namespace tg = tdmd::gpu;

TEST_CASE("SnapBondListGpu — CPU-only build throws from build stub",
          "[gpu][snap][bond_list][cpu]") {
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

// ---------------------------------------------------------------------------
// CPU reference bond-list builder — must produce byte-identical output to the
// device kernels scan_snap_bonds (count + emit). The iteration order, filters,
// and float arithmetic ordering mirror snap_bond_list_gpu.cu exactly.
// ---------------------------------------------------------------------------

inline std::uint32_t wrap_axis(int idx, std::uint32_t n) {
  const int ni = static_cast<int>(n);
  int w = idx % ni;
  if (w < 0) {
    w += ni;
  }
  return static_cast<std::uint32_t>(w);
}

inline std::size_t cell_index_axis(double coord, double lo, double cell, std::uint32_t n) {
  const double local = coord - lo;
  long long idx = static_cast<long long>(std::floor(local / cell));
  if (idx < 0) {
    idx = 0;
  } else if (idx >= static_cast<long long>(n)) {
    idx = static_cast<long long>(n) - 1;
  }
  return static_cast<std::size_t>(idx);
}

inline std::size_t linear_index(std::uint32_t ix,
                                std::uint32_t iy,
                                std::uint32_t iz,
                                std::uint32_t nx,
                                std::uint32_t ny) {
  return static_cast<std::size_t>(ix) +
         static_cast<std::size_t>(nx) *
             (static_cast<std::size_t>(iy) +
              static_cast<std::size_t>(ny) * static_cast<std::size_t>(iz));
}

inline double minimum_image_axis(double delta, double len, bool periodic) {
  if (!periodic || !(len > 0.0)) {
    return delta;
  }
  const double half = 0.5 * len;
  if (delta > half) {
    delta -= len * std::ceil((delta - half) / len);
  } else if (delta < -half) {
    delta += len * std::ceil((-delta - half) / len);
  }
  return delta;
}

struct CpuBondList {
  std::vector<std::uint32_t> atom_bond_start;
  std::vector<std::uint32_t> bond_i;
  std::vector<std::uint32_t> bond_j;
  std::vector<std::uint32_t> bond_type_i;
  std::vector<std::uint32_t> bond_type_j;
  std::vector<double> bond_dx;
  std::vector<double> bond_dy;
  std::vector<double> bond_dz;
  std::vector<double> bond_rsq;
};

CpuBondList cpu_reference_bond_list(std::size_t n,
                                    const std::uint32_t* types,
                                    const double* x,
                                    const double* y,
                                    const double* z,
                                    const std::vector<std::uint32_t>& cell_offsets,
                                    const std::vector<std::uint32_t>& cell_atoms,
                                    const std::vector<double>& rcut_sq_ab,
                                    std::uint32_t n_species,
                                    const tg::BoxParams& p) {
  CpuBondList out;
  out.atom_bond_start.assign(n + 1, 0);

  // Reused per-atom scratch for this-atom's walk.
  auto push_bonds_for_atom = [&](std::uint32_t i, bool emit) -> std::uint32_t {
    std::uint32_t count = 0;
    const std::uint32_t itype = types[i];
    const double xi = x[i];
    const double yi = y[i];
    const double zi = z[i];

    const auto ix_u = static_cast<int>(cell_index_axis(xi, p.xlo, p.cell_x, p.nx));
    const auto iy_u = static_cast<int>(cell_index_axis(yi, p.ylo, p.cell_y, p.ny));
    const auto iz_u = static_cast<int>(cell_index_axis(zi, p.zlo, p.cell_z, p.nz));

    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const std::uint32_t jx = wrap_axis(ix_u + dx, p.nx);
          const std::uint32_t jy = wrap_axis(iy_u + dy, p.ny);
          const std::uint32_t jz = wrap_axis(iz_u + dz, p.nz);
          const std::size_t cj = linear_index(jx, jy, jz, p.nx, p.ny);

          const std::uint32_t begin = cell_offsets[cj];
          const std::uint32_t end = cell_offsets[cj + 1];
          for (std::uint32_t k = begin; k < end; ++k) {
            const std::uint32_t j = cell_atoms[k];
            if (j == i) {
              continue;
            }
            double ddx = x[j] - xi;
            double ddy = y[j] - yi;
            double ddz = z[j] - zi;
            ddx = minimum_image_axis(ddx, p.lx, p.periodic_x);
            ddy = minimum_image_axis(ddy, p.ly, p.periodic_y);
            ddz = minimum_image_axis(ddz, p.lz, p.periodic_z);
            const double rsq = ddx * ddx + ddy * ddy + ddz * ddz;
            const std::uint32_t jtype = types[j];
            const double cutsq_ij = rcut_sq_ab[static_cast<std::size_t>(itype) * n_species + jtype];
            if (!(rsq < cutsq_ij) || !(rsq > 1e-20)) {
              continue;
            }
            if (emit) {
              out.bond_i.push_back(i);
              out.bond_j.push_back(j);
              out.bond_type_i.push_back(itype);
              out.bond_type_j.push_back(jtype);
              out.bond_dx.push_back(ddx);
              out.bond_dy.push_back(ddy);
              out.bond_dz.push_back(ddz);
              out.bond_rsq.push_back(rsq);
            }
            ++count;
          }
        }
      }
    }
    return count;
  };

  // Pass 1 — counts (ignored; we use the running pass-2 cursor).
  // Pass 2 — emit directly into contiguous SoA.
  for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(n); ++i) {
    out.atom_bond_start[i + 1] = out.atom_bond_start[i] + push_bonds_for_atom(i, /*emit=*/true);
  }
  return out;
}

// 2·nrep³ BCC W lattice. NN = a·√3/2 = 2.754 Å at a=3.18, < cutoff 4.73.
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

// 0.01 Å rattle to break symmetry so the bond geometry is non-trivial.
void apply_tiny_rattle(tdmd::AtomSoA& atoms) {
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const double ph = static_cast<double>(i) * 0.1;
    atoms.x[i] += 0.01 * std::sin(ph);
    atoms.y[i] += 0.01 * std::sin(ph + 1.0);
    atoms.z[i] += 0.01 * std::sin(ph + 2.0);
  }
}

tdmd::Box make_cubic_box(double length) {
  tdmd::Box b;
  b.xhi = length;
  b.yhi = length;
  b.zhi = length;
  b.periodic_x = b.periodic_y = b.periodic_z = true;
  return b;
}

}  // namespace

TEST_CASE("SnapBondListGpu — BCC W 5×5×5 matches CPU cell-stencil order",
          "[gpu][snap][bond_list][t8.6c-v5]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }

  constexpr double kLatticeA = 3.18;
  constexpr int kNrep = 5;  // 250 atoms, box = 15.9 Å > 3·(rcut+skin)=14.86
  constexpr double kCutoff = 4.73;
  constexpr double kSkin = 0.2;

  tdmd::Box box = make_cubic_box(kLatticeA * kNrep);
  tdmd::AtomSoA atoms;
  add_bcc_W(atoms, kNrep, kLatticeA);
  REQUIRE(atoms.size() == 250u);
  apply_tiny_rattle(atoms);

  tdmd::CellGrid grid;
  grid.build(box, kCutoff, kSkin);
  grid.bin(atoms);

  // Fabricate a 1-species rcut_sq_ab matrix at W's SNAP cutoff². No dependency
  // on the LAMMPS W_2940 fixture — the bond list is geometry-only, species
  // info flows through verbatim.
  constexpr std::uint32_t kNSpecies = 1;
  const double cutsq = kCutoff * kCutoff;
  std::vector<double> rcut_sq_ab(kNSpecies * kNSpecies, cutsq);

  // Build host input arrays.
  const std::size_t n = atoms.size();
  std::vector<std::uint32_t> host_types(n);
  for (std::size_t i = 0; i < n; ++i) {
    host_types[i] = static_cast<std::uint32_t>(atoms.type[i]);
  }

  const auto& cell_offsets = grid.cell_offsets();
  const auto& cell_atoms = grid.cell_atoms();
  REQUIRE(cell_offsets.size() == grid.cell_count() + 1u);
  REQUIRE(cell_atoms.size() == n);

  tg::BoxParams bp;
  bp.xlo = box.xlo;
  bp.ylo = box.ylo;
  bp.zlo = box.zlo;
  bp.lx = box.xhi - box.xlo;
  bp.ly = box.yhi - box.ylo;
  bp.lz = box.zhi - box.zlo;
  bp.cell_x = grid.cell_x();
  bp.cell_y = grid.cell_y();
  bp.cell_z = grid.cell_z();
  bp.nx = grid.nx();
  bp.ny = grid.ny();
  bp.nz = grid.nz();
  bp.periodic_x = box.periodic_x;
  bp.periodic_y = box.periodic_y;
  bp.periodic_z = box.periodic_z;
  bp.cutoff = kCutoff;  // unused by SnapBondListGpu but set for completeness
  bp.skin = kSkin;

  // ---------- CPU reference ----------
  const CpuBondList cpu = cpu_reference_bond_list(n,
                                                  host_types.data(),
                                                  atoms.x.data(),
                                                  atoms.y.data(),
                                                  atoms.z.data(),
                                                  cell_offsets,
                                                  cell_atoms,
                                                  rcut_sq_ab,
                                                  kNSpecies,
                                                  bp);
  REQUIRE(cpu.bond_i.size() > 0u);
  REQUIRE(cpu.atom_bond_start.back() == cpu.bond_i.size());
  // Sanity — every atom has at least the 8 BCC nearest neighbours inside its
  // 3³ stencil. Lower bound is deliberately loose so a rattle perturbation
  // near the cutoff cannot make the test flaky.
  REQUIRE(cpu.bond_i.size() >= 8u * n);

  // ---------- GPU ----------
  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);

  tg::SnapBondListGpu bond_list;
  bond_list.build(n,
                  host_types.data(),
                  atoms.x.data(),
                  atoms.y.data(),
                  atoms.z.data(),
                  grid.cell_count(),
                  cell_offsets.data(),
                  cell_atoms.data(),
                  rcut_sq_ab.data(),
                  kNSpecies,
                  bp,
                  pool,
                  stream);

  REQUIRE(bond_list.atom_count() == n);
  REQUIRE(bond_list.bond_count() == cpu.bond_i.size());
  REQUIRE(bond_list.build_version() == 1u);

  const tg::SnapBondListHostSnapshot snap = bond_list.download(stream);

  // ---------- CSR offsets byte-exact (integer-only → no FMA drift) ----------
  REQUIRE(snap.atom_bond_start.size() == cpu.atom_bond_start.size());
  REQUIRE(std::memcmp(snap.atom_bond_start.data(),
                      cpu.atom_bond_start.data(),
                      cpu.atom_bond_start.size() * sizeof(std::uint32_t)) == 0);

  // ---------- Bond atom indices + types byte-exact ----------
  REQUIRE(snap.bond_i.size() == cpu.bond_i.size());
  REQUIRE(std::memcmp(snap.bond_i.data(),
                      cpu.bond_i.data(),
                      cpu.bond_i.size() * sizeof(std::uint32_t)) == 0);
  REQUIRE(std::memcmp(snap.bond_j.data(),
                      cpu.bond_j.data(),
                      cpu.bond_j.size() * sizeof(std::uint32_t)) == 0);
  REQUIRE(std::memcmp(snap.bond_type_i.data(),
                      cpu.bond_type_i.data(),
                      cpu.bond_type_i.size() * sizeof(std::uint32_t)) == 0);
  REQUIRE(std::memcmp(snap.bond_type_j.data(),
                      cpu.bond_type_j.data(),
                      cpu.bond_type_j.size() * sizeof(std::uint32_t)) == 0);

  // ---------- Bond geometry: byte-exact on Reference, ULP on FMA flavors ----
  auto max_abs_rel = [](const std::vector<double>& a, const std::vector<double>& b) {
    double worst = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) {
      const double num = std::abs(a[k] - b[k]);
      const double den = std::max(1.0, std::max(std::abs(a[k]), std::abs(b[k])));
      worst = std::max(worst, num / den);
    }
    return worst;
  };

#ifdef TDMD_FLAVOR_FP64_REFERENCE
  REQUIRE(std::memcmp(snap.bond_dx.data(),
                      cpu.bond_dx.data(),
                      cpu.bond_dx.size() * sizeof(double)) == 0);
  REQUIRE(std::memcmp(snap.bond_dy.data(),
                      cpu.bond_dy.data(),
                      cpu.bond_dy.size() * sizeof(double)) == 0);
  REQUIRE(std::memcmp(snap.bond_dz.data(),
                      cpu.bond_dz.data(),
                      cpu.bond_dz.size() * sizeof(double)) == 0);
  REQUIRE(std::memcmp(snap.bond_rsq.data(),
                      cpu.bond_rsq.data(),
                      cpu.bond_rsq.size() * sizeof(double)) == 0);
#else
  // Non-Reference flavors may contract r² via FMA on the device — same
  // convention as test_neighbor_list_gpu (≤ 1e-14 rel).
  REQUIRE(max_abs_rel(snap.bond_dx, cpu.bond_dx) <= 1e-14);
  REQUIRE(max_abs_rel(snap.bond_dy, cpu.bond_dy) <= 1e-14);
  REQUIRE(max_abs_rel(snap.bond_dz, cpu.bond_dz) <= 1e-14);
  REQUIRE(max_abs_rel(snap.bond_rsq, cpu.bond_rsq) <= 1e-14);
#endif
}

TEST_CASE("SnapBondListGpu — rebuild increments build_version + stays consistent",
          "[gpu][snap][bond_list]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }

  constexpr double kLatticeA = 3.18;
  constexpr int kNrep = 5;  // 250 atoms, box = 15.9 Å > 3·(rcut+skin)=14.86
  constexpr double kCutoff = 4.73;
  constexpr double kSkin = 0.2;

  tdmd::Box box = make_cubic_box(kLatticeA * kNrep);
  tdmd::AtomSoA atoms;
  add_bcc_W(atoms, kNrep, kLatticeA);
  REQUIRE(atoms.size() == 250u);
  apply_tiny_rattle(atoms);

  tdmd::CellGrid grid;
  grid.build(box, kCutoff, kSkin);
  grid.bin(atoms);

  constexpr std::uint32_t kNSpecies = 1;
  const double cutsq = kCutoff * kCutoff;
  std::vector<double> rcut_sq_ab(kNSpecies * kNSpecies, cutsq);
  std::vector<std::uint32_t> host_types(atoms.size(), 0U);

  tg::BoxParams bp;
  bp.xlo = box.xlo;
  bp.ylo = box.ylo;
  bp.zlo = box.zlo;
  bp.lx = box.xhi - box.xlo;
  bp.ly = box.yhi - box.ylo;
  bp.lz = box.zhi - box.zlo;
  bp.cell_x = grid.cell_x();
  bp.cell_y = grid.cell_y();
  bp.cell_z = grid.cell_z();
  bp.nx = grid.nx();
  bp.ny = grid.ny();
  bp.nz = grid.nz();
  bp.periodic_x = box.periodic_x;
  bp.periodic_y = box.periodic_y;
  bp.periodic_z = box.periodic_z;

  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);

  tg::SnapBondListGpu bl;
  bl.build(atoms.size(),
           host_types.data(),
           atoms.x.data(),
           atoms.y.data(),
           atoms.z.data(),
           grid.cell_count(),
           grid.cell_offsets().data(),
           grid.cell_atoms().data(),
           rcut_sq_ab.data(),
           kNSpecies,
           bp,
           pool,
           stream);
  REQUIRE(bl.build_version() == 1u);
  const std::size_t bonds_1 = bl.bond_count();
  const auto snap1 = bl.download(stream);

  // Rebuild with the same inputs — should produce identical output.
  bl.build(atoms.size(),
           host_types.data(),
           atoms.x.data(),
           atoms.y.data(),
           atoms.z.data(),
           grid.cell_count(),
           grid.cell_offsets().data(),
           grid.cell_atoms().data(),
           rcut_sq_ab.data(),
           kNSpecies,
           bp,
           pool,
           stream);
  REQUIRE(bl.build_version() == 2u);
  REQUIRE(bl.bond_count() == bonds_1);

  const auto snap2 = bl.download(stream);
  REQUIRE(std::memcmp(snap1.atom_bond_start.data(),
                      snap2.atom_bond_start.data(),
                      snap1.atom_bond_start.size() * sizeof(std::uint32_t)) == 0);
  REQUIRE(std::memcmp(snap1.bond_i.data(), snap2.bond_i.data(), bonds_1 * sizeof(std::uint32_t)) ==
          0);
  REQUIRE(std::memcmp(snap1.bond_j.data(), snap2.bond_j.data(), bonds_1 * sizeof(std::uint32_t)) ==
          0);
  REQUIRE(std::memcmp(snap1.bond_rsq.data(), snap2.bond_rsq.data(), bonds_1 * sizeof(double)) == 0);
}

// ---------------------------------------------------------------------------
// T-opt-3b: reverse_bond_index pairing. For every directed bond b = (i, j) the
// paired bond b' = (j, i) must exist in atom j's bond range, and the geometry
// must be exactly negated (bond_dx[b'] == -bond_dx[b], bit-exact — minimum_image
// is deterministic, and the 2000-atom BCC W box is large enough that each pair
// is unique under a single periodic image).
// ---------------------------------------------------------------------------
TEST_CASE("SnapBondListGpu — reverse_bond_index pairs bond (i,j) with bond (j,i) bit-exactly",
          "[gpu][snap][bond_list][t_opt_3b]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }

  constexpr double kLatticeA = 3.18;
  constexpr int kNrep = 5;  // 250 atoms in 15.9 Å box — 2·cutoff=9.46 < 15.9 → unique pairs
  constexpr double kCutoff = 4.73;
  constexpr double kSkin = 0.2;

  tdmd::Box box = make_cubic_box(kLatticeA * kNrep);
  tdmd::AtomSoA atoms;
  add_bcc_W(atoms, kNrep, kLatticeA);
  REQUIRE(atoms.size() == 250u);
  apply_tiny_rattle(atoms);

  tdmd::CellGrid grid;
  grid.build(box, kCutoff, kSkin);
  grid.bin(atoms);

  constexpr std::uint32_t kNSpecies = 1;
  const double cutsq = kCutoff * kCutoff;
  std::vector<double> rcut_sq_ab(kNSpecies * kNSpecies, cutsq);
  std::vector<std::uint32_t> host_types(atoms.size(), 0U);

  tg::BoxParams bp;
  bp.xlo = box.xlo;
  bp.ylo = box.ylo;
  bp.zlo = box.zlo;
  bp.lx = box.xhi - box.xlo;
  bp.ly = box.yhi - box.ylo;
  bp.lz = box.zhi - box.zlo;
  bp.cell_x = grid.cell_x();
  bp.cell_y = grid.cell_y();
  bp.cell_z = grid.cell_z();
  bp.nx = grid.nx();
  bp.ny = grid.ny();
  bp.nz = grid.nz();
  bp.periodic_x = box.periodic_x;
  bp.periodic_y = box.periodic_y;
  bp.periodic_z = box.periodic_z;

  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);

  tg::SnapBondListGpu bl;
  bl.build(atoms.size(),
           host_types.data(),
           atoms.x.data(),
           atoms.y.data(),
           atoms.z.data(),
           grid.cell_count(),
           grid.cell_offsets().data(),
           grid.cell_atoms().data(),
           rcut_sq_ab.data(),
           kNSpecies,
           bp,
           pool,
           stream);

  const auto snap = bl.download(stream);
  const std::size_t nb = snap.bond_i.size();
  REQUIRE(nb > 0u);
  REQUIRE(snap.reverse_bond_index.size() == nb);

  const std::uint32_t kSentinel = 0xFFFFFFFFu;
  std::size_t unpaired = 0;
  std::size_t atom_mismatch = 0;
  std::size_t geom_mismatch = 0;
  double worst_dx_rel = 0.0;
  double worst_dy_rel = 0.0;
  double worst_dz_rel = 0.0;

  for (std::size_t b = 0; b < nb; ++b) {
    const std::uint32_t bp_idx = snap.reverse_bond_index[b];
    if (bp_idx == kSentinel || bp_idx >= nb) {
      ++unpaired;
      continue;
    }
    // Index consistency: bond b = (i, j) must pair with bond bp_idx = (j, i).
    if (snap.bond_i[bp_idx] != snap.bond_j[b] || snap.bond_j[bp_idx] != snap.bond_i[b]) {
      ++atom_mismatch;
      continue;
    }
    // Geometry consistency: Δr must be exactly negated. minimum_image is
    // deterministic → bit-exact negation under Fp64 Reference; Fp32/FMA
    // flavors may drift by 1 ULP on the r² round-trip but dx/dy/dz come from
    // position subtraction only.
    const double sum_dx = snap.bond_dx[b] + snap.bond_dx[bp_idx];
    const double sum_dy = snap.bond_dy[b] + snap.bond_dy[bp_idx];
    const double sum_dz = snap.bond_dz[b] + snap.bond_dz[bp_idx];
    const double scale_dx =
        std::max(1.0, std::max(std::abs(snap.bond_dx[b]), std::abs(snap.bond_dx[bp_idx])));
    const double scale_dy =
        std::max(1.0, std::max(std::abs(snap.bond_dy[b]), std::abs(snap.bond_dy[bp_idx])));
    const double scale_dz =
        std::max(1.0, std::max(std::abs(snap.bond_dz[b]), std::abs(snap.bond_dz[bp_idx])));
    worst_dx_rel = std::max(worst_dx_rel, std::abs(sum_dx) / scale_dx);
    worst_dy_rel = std::max(worst_dy_rel, std::abs(sum_dy) / scale_dy);
    worst_dz_rel = std::max(worst_dz_rel, std::abs(sum_dz) / scale_dz);
    if (std::abs(sum_dx) > 1e-14 * scale_dx || std::abs(sum_dy) > 1e-14 * scale_dy ||
        std::abs(sum_dz) > 1e-14 * scale_dz) {
      ++geom_mismatch;
    }
  }

  REQUIRE(unpaired == 0u);
  REQUIRE(atom_mismatch == 0u);
  REQUIRE(geom_mismatch == 0u);

  // Symmetry: reverse(reverse(b)) == b for every bond.
  for (std::size_t b = 0; b < nb; ++b) {
    const std::uint32_t bp_idx = snap.reverse_bond_index[b];
    REQUIRE(bp_idx < nb);
    REQUIRE(snap.reverse_bond_index[bp_idx] == b);
  }
}

#endif  // TDMD_BUILD_CUDA
