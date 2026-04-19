// SPEC: docs/specs/gpu/SPEC.md §9 (engine wire-up); docs/specs/runtime/SPEC.md §2.3
// Exec pack: docs/development/m6_execution_pack.md T6.7
// Decisions: D-M6-7 (Reference CPU ≡ GPU bit-exact gate extended to engine)
//
// T6.7 acceptance: SimulationEngine with `runtime.backend: gpu` drives the
// 864-atom Ni-Al EAM fixture through the GPU force + VV path and produces a
// thermo stream that matches the CPU Reference path byte-for-byte over 100
// steps. This extends the CPU-only D-M5-12 chain (K=1 P=1 byte-exact) to the
// GPU era — CPU ≡ GPU ≡ historical M4/M5 golden.
//
// CPU-only builds: the `runtime.backend: gpu` path throws at GpuContext
// construction (CUDA not built in). The test auto-skips via a runtime
// device probe, so this TU compiles cleanly under -DTDMD_BUILD_CUDA=OFF.

#include "tdmd/gpu/factories.hpp"
#include "tdmd/io/yaml_config.hpp"
#include "tdmd/runtime/simulation_engine.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <sstream>
#include <string>

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

tdmd::io::YamlConfig base_config(tdmd::io::RuntimeBackendKind backend) {
  tdmd::io::YamlConfig cfg{};
  cfg.simulation.units = tdmd::io::UnitsKind::Metal;
  cfg.atoms.source = tdmd::io::AtomsSource::LammpsData;
  cfg.atoms.path = std::string(TDMD_REPO_ROOT) + "/verify/benchmarks/t4_nial_alloy/setup.data";
  cfg.potential.style = tdmd::io::PotentialStyle::EamAlloy;
  cfg.potential.eam_alloy.file =
      std::string(TDMD_REPO_ROOT) + "/verify/third_party/potentials/NiAl_Mishin_2004.eam.alloy";
  cfg.integrator.style = tdmd::io::IntegratorStyle::VelocityVerlet;
  cfg.integrator.dt = 0.001;  // ps — matches M5 smoke
  cfg.neighbor.skin = 0.3;    // Å
  cfg.thermo.every = 1;       // full trace for byte-compare
  cfg.run.n_steps = 100;
  cfg.runtime.backend = backend;
  return cfg;
}

std::string run_and_capture_thermo(tdmd::io::RuntimeBackendKind backend, std::uint64_t n_steps) {
  auto cfg = base_config(backend);
  cfg.run.n_steps = n_steps;
  tdmd::SimulationEngine engine;
  engine.init(cfg);
  std::ostringstream os;
  (void) engine.run(n_steps, &os);
  engine.finalize();
  return os.str();
}

}  // namespace

TEST_CASE("T6.7 — CPU ≡ GPU thermo byte-exact on Ni-Al EAM (Reference)",
          "[runtime][engine][gpu][byte-exact]") {
  if (!has_cuda_device()) {
    SKIP("no CUDA device visible");
  }
  constexpr std::uint64_t kSteps = 100;
  const std::string cpu = run_and_capture_thermo(tdmd::io::RuntimeBackendKind::Cpu, kSteps);
  const std::string gpu = run_and_capture_thermo(tdmd::io::RuntimeBackendKind::Gpu, kSteps);
  REQUIRE(cpu == gpu);
}

TEST_CASE("T6.7 — GPU path produces non-empty thermo stream", "[runtime][engine][gpu]") {
  if (!has_cuda_device()) {
    SKIP("no CUDA device visible");
  }
  const std::string gpu = run_and_capture_thermo(tdmd::io::RuntimeBackendKind::Gpu, 10);
  REQUIRE(!gpu.empty());
  // Header + 11 rows (step 0 + steps 1..10) minimum.
  std::size_t newline_count = 0;
  for (char c : gpu) {
    if (c == '\n') {
      ++newline_count;
    }
  }
  REQUIRE(newline_count >= 12);
}
