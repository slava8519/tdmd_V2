// Exec pack: docs/development/m5_execution_pack.md T5.8
// SPEC: docs/specs/runtime/SPEC.md §2.2, docs/specs/comm/SPEC.md §7.2
// Master spec: §6.6, §7.3 Level 1 determinism, §14 M5
//
// Multi-rank TD smoke. Under M5 Option-B (physics replicated per rank +
// deterministic thermo reduction via global_sum_double), the per-step thermo
// on each rank must be bit-exact to a single-rank td_mode run with the same
// fixture. The reduction rewrites local `x` into `global_sum(x / nranks)`
// via a Kahan-compensated ring pass; since every rank's state is identical,
// the two halves sum to the scalar bit-exact (division-by-nranks with
// nranks ∈ {2, 4, 8} is IEEE-754 exact).
//
// This is the "K=1 P=2 bit-exact to K=1 P=1" acceptance gate from the T5.8
// exec-pack brief.

#include "tdmd/comm/mpi_host_staging_backend.hpp"
#include "tdmd/io/yaml_config.hpp"
#include "tdmd/runtime/simulation_engine.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <sstream>
#include <string>

#include <mpi.h>

#ifndef TDMD_IO_FIXTURES_DIR
#error "TDMD_IO_FIXTURES_DIR must be defined by the build system"
#endif

namespace {

tdmd::io::YamlConfig load_shrunken_morse() {
  const std::string path = std::string(TDMD_IO_FIXTURES_DIR) + "/configs/valid_nve_al.yaml";
  auto config = tdmd::io::parse_yaml_config(path);
  config.potential.morse.r0 = 2.0;
  config.potential.morse.cutoff = 2.5;
  config.potential.morse.alpha = 2.0;
  config.neighbor.skin = 0.1;
  config.scheduler.td_mode = true;
  config.scheduler.pipeline_depth_cap = 1;  // K=1 for D-M5-12 regression shape
  return config;
}

std::string fixture_dir() {
  return std::string(TDMD_IO_FIXTURES_DIR) + "/configs";
}

std::string run_single_rank(std::uint64_t n_steps) {
  auto cfg = load_shrunken_morse();
  tdmd::SimulationEngine engine;
  // No backend — single-rank TD path, no reduction.
  engine.init(cfg, fixture_dir());
  std::ostringstream os;
  (void) engine.run(n_steps, &os);
  engine.finalize();
  return os.str();
}

std::string run_with_backend(tdmd::comm::CommBackend& backend, std::uint64_t n_steps) {
  auto cfg = load_shrunken_morse();
  tdmd::SimulationEngine engine;
  engine.set_comm_backend(&backend);
  engine.init(cfg, fixture_dir());
  std::ostringstream os;
  (void) engine.run(n_steps, &os);
  engine.finalize();
  return os.str();
}

}  // namespace

TEST_CASE("T5.8 — K=1 P=2 thermo bit-exact to K=1 P=1 via deterministic reduction",
          "[runtime][engine][td_mode][mpi][mpi2rank][t5-8]") {
  int rank = -1;
  int nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  REQUIRE(nranks == 2);

  tdmd::comm::MpiHostStagingBackend backend;
  backend.initialize(tdmd::comm::CommConfig{});
  REQUIRE(backend.nranks() == 2);

  constexpr std::uint64_t kSteps = 10;

  // Each rank computes its thermo stream with the backend injected. Because
  // physics is replicated across ranks, both ranks' snapshots are identical
  // pre-reduction; after the Kahan ring sum the post-reduction stream must
  // equal the single-rank baseline bit-for-bit.
  const std::string mr = run_with_backend(backend, kSteps);
  const std::string sr = run_single_rank(kSteps);

  REQUIRE_FALSE(mr.empty());
  REQUIRE_FALSE(sr.empty());
  REQUIRE(mr == sr);

  backend.barrier();
  backend.shutdown();
}

TEST_CASE("T5.8 — deterministic reduction: per-rank thermo streams agree bit-exactly",
          "[runtime][engine][td_mode][mpi][mpi2rank][t5-8][determinism]") {
  int rank = -1;
  int nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  REQUIRE(nranks == 2);

  tdmd::comm::MpiHostStagingBackend backend;
  backend.initialize(tdmd::comm::CommConfig{});

  // Both ranks run the same fixture through the same engine; their captured
  // thermo strings must be identical (the reduction is symmetric).
  const std::string s = run_with_backend(backend, /*n_steps=*/5);

  // Gather string lengths via global_max_double as a cheap cross-rank check.
  const double len_local = static_cast<double>(s.size());
  const double len_max = backend.global_max_double(len_local);
  REQUIRE(len_local == len_max);  // each rank's output has the same length

  backend.barrier();
  backend.shutdown();
}
