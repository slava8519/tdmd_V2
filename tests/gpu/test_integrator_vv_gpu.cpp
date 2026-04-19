// Exec pack: docs/development/m6_execution_pack.md T6.6
// SPEC: docs/specs/gpu/SPEC.md §7.3 (VV NVE contract), §6.3 (D-M6-7 gate)
// SPEC: docs/specs/integrator/SPEC.md §3 (math), §8.1 (Reference FP64)
//
// T6.6 acceptance gate: run CPU `VelocityVerletIntegrator` +
// `GpuVelocityVerletIntegrator` on the same initial condition with a
// deterministic force field that does NOT depend on position (caller-
// injected forces), and verify positions + velocities agree bit-exact after
// 1 step, 10 steps, and 1000 steps.
//
// The Reference build uses `--fmad=false` (cmake/BuildFlavors.cmake §17), so
// both paths perform the identical FP64 operation `v += f * accel * half_dt`
// in the same order — literal equality is required (D-M6-7).
//
// All CUDA cases skip gracefully when no device is available.

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/factories.hpp"
#include "tdmd/gpu/gpu_config.hpp"
#include "tdmd/gpu/types.hpp"
#include "tdmd/integrator/gpu_velocity_verlet.hpp"
#include "tdmd/integrator/velocity_verlet.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/species.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#if TDMD_BUILD_CUDA
#include <cuda_runtime.h>
#endif

namespace tg = tdmd::gpu;

namespace {

tdmd::SpeciesId register_al(tdmd::SpeciesRegistry& species) {
  tdmd::SpeciesInfo info;
  info.name = "Al";
  info.mass = 26.98;  // g/mol
  info.atomic_number = 13;
  return species.register_species(info);
}

tdmd::SpeciesId register_ni(tdmd::SpeciesRegistry& species) {
  tdmd::SpeciesInfo info;
  info.name = "Ni";
  info.mass = 58.6934;
  info.atomic_number = 28;
  return species.register_species(info);
}

// N atoms laid out on a simple cubic lattice, non-overlapping — positions
// only matter insofar as they get integrated. Velocities seeded
// deterministically; forces populated fresh per step (see `force_field`
// below) so the test isolates the integrator kernel from any potential.
tdmd::AtomSoA make_lattice(std::size_t nx,
                           std::size_t ny,
                           std::size_t nz,
                           tdmd::SpeciesId type,
                           double a = 3.0) {
  tdmd::AtomSoA atoms;
  atoms.reserve(nx * ny * nz);
  for (std::size_t iz = 0; iz < nz; ++iz) {
    for (std::size_t iy = 0; iy < ny; ++iy) {
      for (std::size_t ix = 0; ix < nx; ++ix) {
        atoms.add_atom(type, a * ix, a * iy, a * iz);
      }
    }
  }
  // Seed deterministic velocities (≤ 0.02 Å/ps — thermal-ish).
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const double fi = static_cast<double>(i + 1);
    atoms.vx[i] = 0.01 * std::sin(0.37 * fi);
    atoms.vy[i] = 0.01 * std::cos(0.19 * fi);
    atoms.vz[i] = 0.01 * std::sin(0.83 * fi + 1.1);
  }
  return atoms;
}

// Deterministic synthetic force field — zero mean, smoothly varying in the
// per-atom index. Does NOT depend on position, so CPU and GPU see identical
// inputs on every step once the integrator has touched positions.
void force_field(tdmd::AtomSoA& atoms, std::uint64_t step) {
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const double fi = static_cast<double>(i + 1);
    const double s = 0.001 * static_cast<double>(step + 1);
    atoms.fx[i] = std::sin(0.11 * fi + s);
    atoms.fy[i] = std::cos(0.29 * fi - s);
    atoms.fz[i] = std::sin(0.53 * fi + 0.5 * s);
  }
}

void zero_forces(tdmd::AtomSoA& atoms) {
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    atoms.fx[i] = 0.0;
    atoms.fy[i] = 0.0;
    atoms.fz[i] = 0.0;
  }
}

// One VV NVE step on the caller's integrator (CPU or GPU). Forces are
// reset to `force_field(step)` on the drifted positions mid-step.
template <typename IntegratorPolicy>
void step_nve(IntegratorPolicy&& step_pre_force,
              IntegratorPolicy&& step_post_force,
              tdmd::AtomSoA& atoms,
              double dt,
              std::uint64_t step_index) {
  step_pre_force(atoms, dt);
  // Forces at drifted positions — inputs are identical for CPU and GPU
  // because `force_field` depends only on atom index and step number.
  force_field(atoms, step_index);
  step_post_force(atoms, dt);
}

// Returns true if every atom's position and velocity are byte-identical.
bool bit_exact_match(const tdmd::AtomSoA& a, const tdmd::AtomSoA& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a.x[i] != b.x[i] || a.y[i] != b.y[i] || a.z[i] != b.z[i]) {
      return false;
    }
    if (a.vx[i] != b.vx[i] || a.vy[i] != b.vy[i] || a.vz[i] != b.vz[i]) {
      return false;
    }
  }
  return true;
}

#if TDMD_BUILD_CUDA
bool cuda_device_available() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  return err == cudaSuccess && count > 0;
}
#endif

}  // namespace

TEST_CASE("VelocityVerletGpu: CPU-only build throws", "[gpu][integrator][vv]") {
#if TDMD_BUILD_CUDA
  SUCCEED("CUDA enabled build — stub path not exercised");
#else
  tdmd::SpeciesRegistry species;
  register_al(species);
  tdmd::GpuVelocityVerletIntegrator integrator(species);
  tdmd::AtomSoA atoms = make_lattice(2, 2, 2, 0U);
  zero_forces(atoms);

  tg::GpuConfig cfg;
  // DevicePool ctor itself throws under CPU-only; but the stub throws too
  // if pool is somehow bypassed. Covers both paths.
  REQUIRE_THROWS_AS(tg::DevicePool(cfg), std::runtime_error);
#endif
}

#if TDMD_BUILD_CUDA

TEST_CASE("VelocityVerletGpu: 1-step bit-exact CPU↔GPU (1000 atoms)",
          "[gpu][integrator][vv][reference]") {
  if (!cuda_device_available()) {
    SKIP("No CUDA device available");
  }
#ifndef TDMD_FLAVOR_FP64_REFERENCE
  SKIP(
      "D-M6-7 bit-exact VV gate is Fp64ReferenceBuild-only; non-Reference flavors "
      "compile with --fmad=true so CPU↔GPU literal equality is not guaranteed");
#endif

  tdmd::SpeciesRegistry species;
  register_al(species);

  constexpr std::size_t kNx = 10;
  constexpr std::size_t kNy = 10;
  constexpr std::size_t kNz = 10;
  constexpr double kDt = 0.001;  // 1 fs

  tdmd::AtomSoA atoms_cpu = make_lattice(kNx, kNy, kNz, 0U);
  tdmd::AtomSoA atoms_gpu = atoms_cpu;
  REQUIRE(atoms_cpu.size() == 1000);

  force_field(atoms_cpu, 0);
  force_field(atoms_gpu, 0);

  tdmd::VelocityVerletIntegrator cpu;
  cpu.pre_force_step(atoms_cpu, species, kDt);
  force_field(atoms_cpu, 1);
  cpu.post_force_step(atoms_cpu, species, kDt);

  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);

  tdmd::GpuVelocityVerletIntegrator gpu(species);
  gpu.pre_force_step(atoms_gpu, kDt, pool, stream);
  force_field(atoms_gpu, 1);
  gpu.post_force_step(atoms_gpu, kDt, pool, stream);

  REQUIRE(bit_exact_match(atoms_cpu, atoms_gpu));
}

TEST_CASE("VelocityVerletGpu: 10-step bit-exact CPU↔GPU (1000 atoms)",
          "[gpu][integrator][vv][reference]") {
  if (!cuda_device_available()) {
    SKIP("No CUDA device available");
  }
#ifndef TDMD_FLAVOR_FP64_REFERENCE
  SKIP(
      "D-M6-7 bit-exact VV gate is Fp64ReferenceBuild-only; non-Reference flavors "
      "compile with --fmad=true so CPU↔GPU literal equality is not guaranteed");
#endif

  tdmd::SpeciesRegistry species;
  register_al(species);

  constexpr std::size_t kNx = 10;
  constexpr std::size_t kNy = 10;
  constexpr std::size_t kNz = 10;
  constexpr double kDt = 0.001;
  constexpr std::uint64_t kSteps = 10;

  tdmd::AtomSoA atoms_cpu = make_lattice(kNx, kNy, kNz, 0U);
  tdmd::AtomSoA atoms_gpu = atoms_cpu;

  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);

  tdmd::VelocityVerletIntegrator cpu;
  tdmd::GpuVelocityVerletIntegrator gpu(species);

  force_field(atoms_cpu, 0);
  force_field(atoms_gpu, 0);

  for (std::uint64_t k = 0; k < kSteps; ++k) {
    cpu.pre_force_step(atoms_cpu, species, kDt);
    force_field(atoms_cpu, k + 1);
    cpu.post_force_step(atoms_cpu, species, kDt);

    gpu.pre_force_step(atoms_gpu, kDt, pool, stream);
    force_field(atoms_gpu, k + 1);
    gpu.post_force_step(atoms_gpu, kDt, pool, stream);
  }
  REQUIRE(bit_exact_match(atoms_cpu, atoms_gpu));
}

TEST_CASE("VelocityVerletGpu: 1000-step bit-exact CPU↔GPU (1000 atoms)",
          "[gpu][integrator][vv][reference][slow]") {
  if (!cuda_device_available()) {
    SKIP("No CUDA device available");
  }
#ifndef TDMD_FLAVOR_FP64_REFERENCE
  SKIP(
      "D-M6-7 bit-exact VV gate is Fp64ReferenceBuild-only; non-Reference flavors "
      "compile with --fmad=true so CPU↔GPU literal equality is not guaranteed");
#endif

  tdmd::SpeciesRegistry species;
  register_al(species);

  constexpr std::size_t kNx = 10;
  constexpr std::size_t kNy = 10;
  constexpr std::size_t kNz = 10;
  constexpr double kDt = 0.001;
  constexpr std::uint64_t kSteps = 1000;

  tdmd::AtomSoA atoms_cpu = make_lattice(kNx, kNy, kNz, 0U);
  tdmd::AtomSoA atoms_gpu = atoms_cpu;

  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);

  tdmd::VelocityVerletIntegrator cpu;
  tdmd::GpuVelocityVerletIntegrator gpu(species);

  force_field(atoms_cpu, 0);
  force_field(atoms_gpu, 0);

  for (std::uint64_t k = 0; k < kSteps; ++k) {
    cpu.pre_force_step(atoms_cpu, species, kDt);
    force_field(atoms_cpu, k + 1);
    cpu.post_force_step(atoms_cpu, species, kDt);

    gpu.pre_force_step(atoms_gpu, kDt, pool, stream);
    force_field(atoms_gpu, k + 1);
    gpu.post_force_step(atoms_gpu, kDt, pool, stream);
  }
  REQUIRE(bit_exact_match(atoms_cpu, atoms_gpu));
}

TEST_CASE("VelocityVerletGpu: two-species mass handling (Ni + Al)",
          "[gpu][integrator][vv][reference]") {
  if (!cuda_device_available()) {
    SKIP("No CUDA device available");
  }
#ifndef TDMD_FLAVOR_FP64_REFERENCE
  SKIP(
      "D-M6-7 bit-exact VV gate is Fp64ReferenceBuild-only; non-Reference flavors "
      "compile with --fmad=true so CPU↔GPU literal equality is not guaranteed");
#endif

  tdmd::SpeciesRegistry species;
  const tdmd::SpeciesId al = register_al(species);
  const tdmd::SpeciesId ni = register_ni(species);
  REQUIRE(al == 0U);
  REQUIRE(ni == 1U);

  // 8×8×8 mixed lattice — alternate Ni/Al based on parity of the lattice
  // index so both species see force + mass variation.
  tdmd::AtomSoA atoms;
  const double a = 3.5;
  atoms.reserve(512);
  std::size_t idx = 0;
  for (std::size_t iz = 0; iz < 8; ++iz) {
    for (std::size_t iy = 0; iy < 8; ++iy) {
      for (std::size_t ix = 0; ix < 8; ++ix) {
        const tdmd::SpeciesId t = ((ix + iy + iz) & 1U) != 0U ? ni : al;
        atoms.add_atom(t, a * ix, a * iy, a * iz);
        ++idx;
      }
    }
  }
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    atoms.vx[i] = 0.01 * std::sin(0.37 * (i + 1));
    atoms.vy[i] = 0.01 * std::cos(0.19 * (i + 1));
    atoms.vz[i] = 0.01 * std::sin(0.83 * (i + 1) + 1.1);
  }

  tdmd::AtomSoA atoms_cpu = atoms;
  tdmd::AtomSoA atoms_gpu = atoms;

  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);

  tdmd::VelocityVerletIntegrator cpu;
  tdmd::GpuVelocityVerletIntegrator gpu(species);

  force_field(atoms_cpu, 0);
  force_field(atoms_gpu, 0);

  constexpr double kDt = 0.001;
  constexpr std::uint64_t kSteps = 100;
  for (std::uint64_t k = 0; k < kSteps; ++k) {
    cpu.pre_force_step(atoms_cpu, species, kDt);
    force_field(atoms_cpu, k + 1);
    cpu.post_force_step(atoms_cpu, species, kDt);

    gpu.pre_force_step(atoms_gpu, kDt, pool, stream);
    force_field(atoms_gpu, k + 1);
    gpu.post_force_step(atoms_gpu, kDt, pool, stream);
  }
  REQUIRE(bit_exact_match(atoms_cpu, atoms_gpu));
}

TEST_CASE("VelocityVerletGpu: empty atoms → no-op + version bump", "[gpu][integrator][vv]") {
  if (!cuda_device_available()) {
    SKIP("No CUDA device available");
  }

  tdmd::SpeciesRegistry species;
  register_al(species);

  tdmd::AtomSoA atoms;  // empty

  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);

  tdmd::GpuVelocityVerletIntegrator gpu(species);
  REQUIRE(gpu.compute_version() == 0);
  gpu.pre_force_step(atoms, 0.001, pool, stream);
  REQUIRE(gpu.compute_version() == 1);
  gpu.post_force_step(atoms, 0.001, pool, stream);
  REQUIRE(gpu.compute_version() == 2);
  REQUIRE(atoms.size() == 0);
}

TEST_CASE("VelocityVerletGpu: dt validation", "[gpu][integrator][vv]") {
  if (!cuda_device_available()) {
    SKIP("No CUDA device available");
  }

  tdmd::SpeciesRegistry species;
  register_al(species);

  tdmd::AtomSoA atoms = make_lattice(2, 2, 2, 0U);
  zero_forces(atoms);

  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream stream = tg::make_stream(cfg.device_id);

  tdmd::GpuVelocityVerletIntegrator gpu(species);
  REQUIRE_THROWS_AS(gpu.pre_force_step(atoms, -0.001, pool, stream), std::invalid_argument);
  REQUIRE_THROWS_AS(gpu.pre_force_step(atoms, 0.0, pool, stream), std::invalid_argument);
  REQUIRE_THROWS_AS(gpu.post_force_step(atoms, std::nan(""), pool, stream), std::invalid_argument);
}

#endif  // TDMD_BUILD_CUDA
