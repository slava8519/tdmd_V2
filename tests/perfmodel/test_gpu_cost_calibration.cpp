// SPEC: docs/specs/perfmodel/SPEC.md §4.2 (calibration), §5.2 (Pattern 1 ±20%
//       gate), §11.6 (T7.13 fixture contract).
// Exec pack: docs/development/m7_execution_pack.md T7.13.
// Decision: D-M6-8 (Pattern 1 tolerance), D-M6-6 (Option A CI policy),
//           D-M7-9 (Pattern 2 ±25% orthogonal — not exercised here).
//
// T7.13 closes M6 carry-forward T6.11b — the Pattern 1 ±20% calibration
// gate vs measured per-step wall-time. This test:
//
//   (1) loads `verify/measurements/gpu_cost_calibration.yaml`;
//   (2) for each (hardware_id × build_flavor) row, looks up the matching
//       committed factory table (fp64_reference vs mixed_fast);
//   (3) iterates the row's measurements and asserts
//           |predict_step_gpu_sec(N) - measured_step_sec(N)| / measured < 0.20
//       with HardwareProfile::modern_x86_64 scheduler overhead folded in via
//       `predict_step_gpu_sec` (it adds scheduler_overhead_sec to the table
//       sum — the fixture's measured_step_sec is specified to include that
//       contribution).
//
// Missing fixture → SKIP (not FAIL). Option A CI runs without GPU hardware
// and may not carry the fixture; the gate only fires when a measurements
// file is present, so the public CI stays green on Linux x86_64 while local
// pre-push (with fixture + real GPU) still enforces the threshold.
//
// Pattern 2 (±25%) gate is T7.13b future — orthogonal and deliberately not
// wired here per D-M7-9.

#include "tdmd/perfmodel/gpu_cost_tables.hpp"
#include "tdmd/perfmodel/hardware_profile.hpp"
#include "tdmd/perfmodel/perfmodel.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#ifndef TDMD_REPO_ROOT
#error "TDMD_REPO_ROOT must be defined by CMake to locate the calibration fixture"
#endif

namespace {

constexpr const char* kFixtureRelativePath = "verify/measurements/gpu_cost_calibration.yaml";
constexpr double kPattern1GateTolerance = 0.20;  // D-M6-8

[[nodiscard]] std::string fixture_absolute_path() {
  std::filesystem::path p(TDMD_REPO_ROOT);
  p /= kFixtureRelativePath;
  return p.string();
}

[[nodiscard]] tdmd::GpuCostTables table_for_flavor(const std::string& flavor) {
  if (flavor == "fp64_reference") {
    return tdmd::gpu_cost_tables_fp64_reference();
  }
  // load_gpu_calibration_fixture validates the enum — "mixed_fast" is the
  // only alternative reaching this point.
  return tdmd::gpu_cost_tables_mixed_fast();
}

}  // namespace

TEST_CASE("Calibration fixture — graceful missing file returns nullopt",
          "[perfmodel][calibration][T7.13]") {
  const auto missing =
      tdmd::load_gpu_calibration_fixture("/does/not/exist/gpu_cost_calibration.yaml");
  REQUIRE(!missing.has_value());
}

TEST_CASE("Calibration fixture — committed file loads and is well-formed",
          "[perfmodel][calibration][T7.13]") {
  const auto path = fixture_absolute_path();
  const auto maybe_fx = tdmd::load_gpu_calibration_fixture(path);

  if (!maybe_fx.has_value()) {
    // Option A CI may not carry the fixture — emit a diagnostic and skip.
    WARN("Calibration fixture missing at " << path << " — SKIPping ±20% gate per T7.13 contract.");
    SUCCEED("Calibration fixture missing — gate skipped (Option A CI graceful path).");
    return;
  }

  const auto& fx = *maybe_fx;
  REQUIRE(fx.schema_version == 1);
  REQUIRE_FALSE(fx.rows.empty());

  for (const auto& row : fx.rows) {
    REQUIRE_FALSE(row.hardware_id.empty());
    REQUIRE_FALSE(row.cuda_version.empty());
    REQUIRE_FALSE(row.measurement_date.empty());
    REQUIRE_FALSE(row.build_flavor.empty());
    REQUIRE_FALSE(row.provenance.empty());
    REQUIRE_FALSE(row.measurements.empty());
    for (const auto& m : row.measurements) {
      REQUIRE(m.n_atoms > 0U);
      REQUIRE(std::isfinite(m.measured_step_sec));
      REQUIRE(m.measured_step_sec > 0.0);
    }
  }
}

TEST_CASE("Calibration gate — predict_step_gpu_sec within ±20% of measured (Pattern 1)",
          "[perfmodel][calibration][T7.13]") {
  const auto path = fixture_absolute_path();
  const auto maybe_fx = tdmd::load_gpu_calibration_fixture(path);

  if (!maybe_fx.has_value()) {
    WARN("Calibration fixture missing at " << path << " — SKIPping ±20% gate per T7.13 contract.");
    SUCCEED("Calibration fixture missing — gate skipped (Option A CI graceful path).");
    return;
  }

  auto hw = tdmd::HardwareProfile::modern_x86_64();
  hw.n_ranks = 1U;  // Pattern 1 anchor: single-rank so N_per_rank == N_total.

  tdmd::PerfModel pm(hw, tdmd::PotentialCost::eam_alloy());

  const auto& fx = *maybe_fx;
  for (const auto& row : fx.rows) {
    const tdmd::GpuCostTables table = table_for_flavor(row.build_flavor);
    for (const auto& m : row.measurements) {
      const double predicted = pm.predict_step_gpu_sec(m.n_atoms, table);
      const double rel_err = std::abs(predicted - m.measured_step_sec) / m.measured_step_sec;
      INFO("hardware=" << row.hardware_id << " flavor=" << row.build_flavor
                       << " n_atoms=" << m.n_atoms << " predicted=" << predicted
                       << " measured=" << m.measured_step_sec << " rel_err=" << rel_err);
      REQUIRE(rel_err < kPattern1GateTolerance);
    }
  }
}

TEST_CASE("Calibration fixture — schema rejects unknown build_flavor",
          "[perfmodel][calibration][T7.13]") {
  // Round-trip a minimal hand-rolled YAML via a temp file so the test doesn't
  // depend on keeping a fixture with a deliberately-broken build_flavor in
  // the tree. The intent is to exercise the loader's error path, not the
  // fixture content.
  const auto tmp = std::filesystem::temp_directory_path() / "tdmd_cal_bad_flavor.yaml";
  {
    std::ofstream out(tmp);
    out << "schema_version: 1\n"
           "rows:\n"
           "  - hardware_id: 'X'\n"
           "    cuda_version: '13.1'\n"
           "    measurement_date: '2026-04-19'\n"
           "    build_flavor: 'unknown_flavor'\n"
           "    provenance: 'test'\n"
           "    measurements:\n"
           "      - { n_atoms: 10000, measured_step_sec: 0.001 }\n";
  }
  REQUIRE_THROWS_AS(tdmd::load_gpu_calibration_fixture(tmp.string()), std::runtime_error);
  std::filesystem::remove(tmp);
}

TEST_CASE("Calibration fixture — schema rejects empty measurements list",
          "[perfmodel][calibration][T7.13]") {
  const auto tmp = std::filesystem::temp_directory_path() / "tdmd_cal_empty_meas.yaml";
  {
    std::ofstream out(tmp);
    out << "schema_version: 1\n"
           "rows:\n"
           "  - hardware_id: 'X'\n"
           "    cuda_version: '13.1'\n"
           "    measurement_date: '2026-04-19'\n"
           "    build_flavor: 'fp64_reference'\n"
           "    provenance: 'test'\n"
           "    measurements: []\n";
  }
  REQUIRE_THROWS_AS(tdmd::load_gpu_calibration_fixture(tmp.string()), std::runtime_error);
  std::filesystem::remove(tmp);
}

TEST_CASE("Calibration fixture — schema rejects wrong schema_version",
          "[perfmodel][calibration][T7.13]") {
  const auto tmp = std::filesystem::temp_directory_path() / "tdmd_cal_bad_schema.yaml";
  {
    std::ofstream out(tmp);
    out << "schema_version: 99\n"
           "rows: []\n";
  }
  REQUIRE_THROWS_AS(tdmd::load_gpu_calibration_fixture(tmp.string()), std::runtime_error);
  std::filesystem::remove(tmp);
}
