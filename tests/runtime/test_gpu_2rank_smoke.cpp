// Exec pack: docs/development/m6_execution_pack.md T6.7
// SPEC: docs/specs/gpu/SPEC.md §9 (engine wire-up); docs/specs/comm/SPEC.md §7.2
// Decisions: D-M6-3 (host-staged MPI), D-M6-7 (CPU ≡ GPU bit-exact gate)
//
// T6.7 2-rank acceptance: SimulationEngine with `runtime.backend: gpu` +
// MpiHostStagingBackend on 2 ranks produces a thermo stream byte-equal to
// the 1-rank GPU path. This is the determinism chain extended from M5
// (D-M5-12 CPU K=1 P=2 ≡ K=1 P=1) to the GPU era:
//
//   CPU K=1 P=1  ≡  CPU K=1 P=2  (M5)
//         ≡
//   GPU K=1 P=1  ≡  GPU K=1 P=2  (this test + test_gpu_backend_smoke)
//
// The GPU force kernel is deterministic (D-M6-7 ≤1e-12 rel, T6.5) and the
// VV kernel is byte-equal to CPU VV under Reference (T6.6); the multi-rank
// reduction is Kahan-ring (T5.8). All three together preserve bit-exactness.

#include "tdmd/comm/mpi_host_staging_backend.hpp"
#include "tdmd/gpu/factories.hpp"
#include "tdmd/io/yaml_config.hpp"
#include "tdmd/runtime/simulation_engine.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <sstream>
#include <string>

#include <mpi.h>

#ifndef TDMD_REPO_ROOT
#error "TDMD_REPO_ROOT must be defined by the build system"
#endif

namespace {

bool has_cuda_device() {
  try {
    return !tdmd::gpu::probe_devices().empty();
  } catch (...) {
    return false;
  }
}

tdmd::io::YamlConfig nial_eam_config(tdmd::io::RuntimeBackendKind backend) {
  tdmd::io::YamlConfig cfg{};
  cfg.simulation.units = tdmd::io::UnitsKind::Metal;
  cfg.atoms.source = tdmd::io::AtomsSource::LammpsData;
  cfg.atoms.path = std::string(TDMD_REPO_ROOT) + "/verify/benchmarks/t4_nial_alloy/setup.data";
  cfg.potential.style = tdmd::io::PotentialStyle::EamAlloy;
  cfg.potential.eam_alloy.file =
      std::string(TDMD_REPO_ROOT) + "/verify/third_party/potentials/NiAl_Mishin_2004.eam.alloy";
  cfg.integrator.style = tdmd::io::IntegratorStyle::VelocityVerlet;
  cfg.integrator.dt = 0.001;
  cfg.neighbor.skin = 0.3;
  cfg.thermo.every = 1;
  cfg.run.n_steps = 10;
  cfg.runtime.backend = backend;
  return cfg;
}

std::string run_single_rank_gpu(std::uint64_t n_steps) {
  auto cfg = nial_eam_config(tdmd::io::RuntimeBackendKind::Gpu);
  cfg.run.n_steps = n_steps;
  tdmd::SimulationEngine engine;
  engine.init(cfg);
  std::ostringstream os;
  (void) engine.run(n_steps, &os);
  engine.finalize();
  return os.str();
}

std::string run_gpu_with_backend(tdmd::comm::CommBackend& backend, std::uint64_t n_steps) {
  auto cfg = nial_eam_config(tdmd::io::RuntimeBackendKind::Gpu);
  cfg.run.n_steps = n_steps;
  tdmd::SimulationEngine engine;
  engine.set_comm_backend(&backend);
  engine.init(cfg);
  std::ostringstream os;
  (void) engine.run(n_steps, &os);
  engine.finalize();
  return os.str();
}

}  // namespace

TEST_CASE("T6.7 — GPU K=1 P=2 thermo byte-exact to GPU K=1 P=1 (Ni-Al EAM)",
          "[runtime][engine][gpu][mpi][mpi2rank][byte-exact]") {
  if (!has_cuda_device()) {
    SKIP("no CUDA device visible");
  }
  int rank = -1;
  int nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  REQUIRE(nranks == 2);

  tdmd::comm::MpiHostStagingBackend backend;
  backend.initialize(tdmd::comm::CommConfig{});
  REQUIRE(backend.nranks() == 2);

  constexpr std::uint64_t kSteps = 10;
  const std::string mr = run_gpu_with_backend(backend, kSteps);
  const std::string sr = run_single_rank_gpu(kSteps);

  REQUIRE_FALSE(mr.empty());
  REQUIRE_FALSE(sr.empty());
  REQUIRE(mr == sr);

  backend.barrier();
  backend.shutdown();
}
