// Exec pack: docs/development/m7_execution_pack.md T7.0
// SPEC: docs/specs/gpu/SPEC.md §8.3 (D-M6-8 dense-cutoff canonical, v1.0.12)
// SPEC: docs/specs/runtime/SPEC.md §2.3 (runtime.backend dispatch)
// Master spec: §D.1 Philosophy B + Приложение C T7.0 addendum
//
// T7.0 acceptance gate: 100-step NVE drift on Ni-Al EAM/alloy 864 atoms with
// `runtime.backend: gpu` + MixedFastBuild. Asserts
// `|E_total(100) - E_total(0)| / |E_total(0)| <= 1e-6`, which is the per-100-step
// budget giving 10x margin under the 1000-step D-M6-8 drift cap of 1e-5. The
// drift gate is integrator-level and complements the per-step force/PE
// thresholds exercised by test_eam_mixed_fast_within_threshold.cpp.
//
// Scope discipline:
//  * Reference build — SKIP: D-M6-7 is byte-exact CPU<->GPU, so the drift
//    against the CPU reference is a separate (tighter) gate. D-M6-8 drift
//    is MixedFast-specific.
//  * CPU-only build — SKIP: no CUDA runtime.
//  * No CUDA device visible — SKIP: CI path (Option A).

#include "tdmd/gpu/factories.hpp"
#include "tdmd/io/yaml_config.hpp"
#include "tdmd/runtime/simulation_engine.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

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

tdmd::io::YamlConfig base_config() {
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
  cfg.run.n_steps = 100;
  cfg.runtime.backend = tdmd::io::RuntimeBackendKind::Gpu;
  return cfg;
}

// Parse thermo rows. Columns: "# step temp pe ke etotal press".
// Returns etotal values in insertion order. Skips the header line.
std::vector<double> parse_etotal(const std::string& thermo) {
  std::vector<double> out;
  std::istringstream is(thermo);
  std::string line;
  while (std::getline(is, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream row(line);
    std::uint64_t step = 0;
    double temp = 0.0;
    double pe = 0.0;
    double ke = 0.0;
    double etot = 0.0;
    double press = 0.0;
    if (row >> step >> temp >> pe >> ke >> etot >> press) {
      out.push_back(etot);
    }
  }
  return out;
}

}  // namespace

TEST_CASE("T7.0 — MixedFast NVE drift within D-M6-8 dense-cutoff budget",
          "[gpu][mixed][drift][t7_0]") {
  if (!has_cuda_device()) {
    SKIP("no CUDA device visible");
  }
#ifndef TDMD_FLAVOR_MIXED_FAST
  SKIP(
      "D-M6-8 drift gate is MixedFastBuild-only; Reference is byte-exact per "
      "D-M6-7 (no drift test required). Build with -DTDMD_BUILD_FLAVOR=MixedFastBuild "
      "to exercise this gate.");
#endif

  constexpr std::uint64_t kSteps = 100;
  // D-M6-8 canonical: 1e-5 per 1000 steps; 100-step per-capita budget with
  // 10x margin = 1e-6. See gpu/SPEC §8.3 + verify/thresholds/thresholds.yaml
  // `gpu_mixed_fast.dense_cutoff.nve_drift_per_1000_steps`.
  constexpr double kDriftBudget = 1.0e-6;

  auto cfg = base_config();
  cfg.run.n_steps = kSteps;

  tdmd::SimulationEngine engine;
  engine.init(cfg);
  std::ostringstream thermo_stream;
  (void) engine.run(kSteps, &thermo_stream);
  engine.finalize();

  const std::vector<double> etotal = parse_etotal(thermo_stream.str());
  // Thermo emits step 0 + steps 1..100 = 101 rows.
  REQUIRE(etotal.size() >= kSteps + 1);

  const double e0 = etotal.front();
  const double e_final = etotal.back();
  const double denom = std::max(1.0, std::abs(e0));
  const double drift = std::abs(e_final - e0) / denom;

  INFO("E_total(step=0)   = " << e0);
  INFO("E_total(step=100) = " << e_final);
  INFO("|ΔE| / |E₀|       = " << drift);
  INFO("budget            = " << kDriftBudget << " (D-M6-8 dense, 100-step)");
  REQUIRE(drift <= kDriftBudget);
}
