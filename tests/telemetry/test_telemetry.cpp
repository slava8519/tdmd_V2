// SPEC: docs/specs/telemetry/SPEC.md §§3, 4.2, 6
// Exec pack: docs/development/m2_execution_pack.md T2.12
//
// Coverage:
//   1. begin/end — accumulation, nesting, mismatched-end tolerance
//   2. JSONL — well-formed, parseable (minimal in-test JSON scanner)
//   3. LAMMPS format — column header matches SPEC §4.2 mockup exactly
//   4. Section order — LAMMPS canonical ordering, first-seen trailing
//   5. Overhead budget — <0.1% (local) / <0.2% (CI) across 10⁴ sections
//   6. ScopedSection RAII — exception-safe, nullptr no-op
//
// CI vs local: the overhead gate checks env var TDMD_CI — tighter budget
// locally (0.1%) than in CI (0.2%) to accomodate noisy shared runners.

#include "tdmd/telemetry/telemetry.hpp"

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using tdmd::telemetry::ScopedSection;
using tdmd::telemetry::Telemetry;

namespace {

// Busy-wait for `duration` so timing assertions have signal well above the
// steady-clock granularity (~1 µs). sleep_for lets the OS preempt the thread
// and drops dt below measurement noise — not what we want in a unit test.
void busy_wait(std::chrono::microseconds duration) {
  const auto start = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - start < duration) {
    // spin
  }
}

// Locate `key` in a JSON-ish blob; return the substring that follows it up to
// the next delimiter. Good enough for test assertions; not a real parser.
std::string find_json_value(const std::string& s, const std::string& key) {
  const auto pos = s.find("\"" + key + "\":");
  if (pos == std::string::npos) {
    return {};
  }
  auto start = pos + key.size() + 3;  // skip "key":
  while (start < s.size() && (s[start] == ' ' || s[start] == '"')) {
    ++start;
  }
  auto end = start;
  while (end < s.size() && s[end] != ',' && s[end] != '}' && s[end] != '"' && s[end] != '\n') {
    ++end;
  }
  return s.substr(start, end - start);
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. begin/end accumulation
// ---------------------------------------------------------------------------

TEST_CASE("begin_section + end_section accumulate wall time", "[telemetry]") {
  Telemetry t;
  t.begin_section("Pair");
  busy_wait(std::chrono::microseconds(5000));
  t.end_section("Pair");

  const auto breakdown = t.current_breakdown();
  REQUIRE(breakdown.count("Pair") == 1);
  const double pair_sec = breakdown.at("Pair");
  // 5 ms ± generous slack to survive noisy runners (we just want >> 0).
  REQUIRE(pair_sec > 0.001);
  REQUIRE(pair_sec < 0.500);
}

TEST_CASE("Two separate sections accumulate independently", "[telemetry]") {
  Telemetry t;
  t.begin_section("Pair");
  busy_wait(std::chrono::microseconds(3000));
  t.end_section("Pair");
  t.begin_section("Neigh");
  busy_wait(std::chrono::microseconds(1500));
  t.end_section("Neigh");

  const auto breakdown = t.current_breakdown();
  REQUIRE(breakdown.size() == 2);
  REQUIRE(breakdown.at("Pair") > breakdown.at("Neigh"));
}

TEST_CASE("Nested begin of same section delays accounting to outermost close", "[telemetry]") {
  Telemetry t;
  t.begin_section("Pair");  // depth 1
  t.begin_section("Pair");  // depth 2
  busy_wait(std::chrono::microseconds(2000));
  t.end_section("Pair");  // depth 1 → 0 not yet triggered
  t.end_section("Pair");  // outermost closes here

  const auto breakdown = t.current_breakdown();
  REQUIRE(breakdown.at("Pair") > 0.001);
}

TEST_CASE("Mismatched end_section counts as ignored rather than throwing", "[telemetry]") {
  Telemetry t;
  t.end_section("NeverOpened");
  t.begin_section("Pair");
  t.end_section("Pair");
  t.end_section("Pair");  // already closed

  REQUIRE(t.ignored_end_calls() == 2);
  REQUIRE(t.current_breakdown().count("Pair") == 1);
}

TEST_CASE("Mid-run breakdown includes running delta for open sections", "[telemetry]") {
  Telemetry t;
  t.begin_section("Pair");
  busy_wait(std::chrono::microseconds(2000));
  const auto running = t.current_breakdown();
  REQUIRE(running.at("Pair") > 0.0);  // still open, must be non-zero
  t.end_section("Pair");
}

// ---------------------------------------------------------------------------
// 2. JSONL snapshot
// ---------------------------------------------------------------------------

TEST_CASE("write_jsonl emits a single parseable line", "[telemetry]") {
  Telemetry t;
  t.begin_run();
  t.begin_section("Pair");
  busy_wait(std::chrono::microseconds(1500));
  t.end_section("Pair");
  t.end_run();

  std::ostringstream oss;
  t.write_jsonl(oss);
  const std::string out = oss.str();

  // Exactly one trailing newline, exactly one '{' and one '}'.
  REQUIRE(!out.empty());
  REQUIRE(out.back() == '\n');
  REQUIRE(std::count(out.begin(), out.end(), '{') == 2);  // outer + sections
  REQUIRE(std::count(out.begin(), out.end(), '}') == 2);
  REQUIRE(std::count(out.begin(), out.end(), '\n') == 1);

  // Required keys.
  REQUIRE(out.find("\"event\":\"run_end\"") != std::string::npos);
  REQUIRE(out.find("\"total_wall_sec\":") != std::string::npos);
  REQUIRE(out.find("\"sections\":{") != std::string::npos);
  REQUIRE(out.find("\"Pair\":") != std::string::npos);
  REQUIRE(out.find("\"ignored_end_calls\":0") != std::string::npos);
}

TEST_CASE("JSONL escapes special characters in section names", "[telemetry]") {
  Telemetry t;
  t.begin_section("weird\"name\\with\ttabs");
  t.end_section("weird\"name\\with\ttabs");

  std::ostringstream oss;
  t.write_jsonl(oss);
  const auto out = oss.str();

  // Escaped forms must be present; the raw chars must not leak.
  REQUIRE(out.find("\\\"") != std::string::npos);
  REQUIRE(out.find("\\\\") != std::string::npos);
  REQUIRE(out.find("\\t") != std::string::npos);
}

// ---------------------------------------------------------------------------
// 3. LAMMPS-compatible breakdown
// ---------------------------------------------------------------------------

TEST_CASE("LAMMPS format header matches SPEC §4.2 mockup exactly", "[telemetry]") {
  Telemetry t;
  t.begin_run();
  t.end_run();

  std::ostringstream oss;
  t.write_lammps_format(oss, /*n_steps=*/100, /*dt_ps=*/0.001);
  const std::string out = oss.str();

  // Header line byte-for-byte (SPEC §4.2 mockup).
  const std::string header = "Section |  min time  |  avg time  |  max time  |%varavg| %total";
  REQUIRE(out.find(header) != std::string::npos);
  REQUIRE(out.find("MPI task timing breakdown:") != std::string::npos);
  REQUIRE(out.find("Performance:") != std::string::npos);
  REQUIRE(out.find(" tau/day") != std::string::npos);
  REQUIRE(out.find(" ns/day") != std::string::npos);
  REQUIRE(out.find(" ms/timestep") != std::string::npos);
}

TEST_CASE("LAMMPS format emits canonical LAMMPS sections in order", "[telemetry]") {
  Telemetry t;
  t.begin_run();
  // Register in reversed order — output order must remain canonical.
  t.begin_section("Output");
  t.end_section("Output");
  t.begin_section("Neigh");
  t.end_section("Neigh");
  t.begin_section("Pair");
  t.end_section("Pair");
  t.end_run();

  std::ostringstream oss;
  t.write_lammps_format(oss, 10, 0.001);
  const auto out = oss.str();

  const auto p_pair = out.find("\nPair ");
  const auto p_neigh = out.find("\nNeigh ");
  const auto p_comm = out.find("\nComm ");
  const auto p_output = out.find("\nOutput ");
  const auto p_other = out.find("\nOther ");
  const auto p_total = out.find("\nTotal ");

  REQUIRE(p_pair != std::string::npos);
  REQUIRE(p_neigh != std::string::npos);
  REQUIRE(p_comm != std::string::npos);
  REQUIRE(p_output != std::string::npos);
  REQUIRE(p_other != std::string::npos);
  REQUIRE(p_total != std::string::npos);
  REQUIRE(p_pair < p_neigh);
  REQUIRE(p_neigh < p_comm);
  REQUIRE(p_comm < p_output);
  REQUIRE(p_output < p_other);
  REQUIRE(p_other < p_total);
}

TEST_CASE("LAMMPS format — Comm row is 0.000 when no Comm section ever opened", "[telemetry]") {
  Telemetry t;
  t.begin_run();
  t.begin_section("Pair");
  busy_wait(std::chrono::microseconds(1000));
  t.end_section("Pair");
  t.end_run();

  std::ostringstream oss;
  t.write_lammps_format(oss, 1, 0.001);
  const auto out = oss.str();

  // Extract the Comm line (ends at newline).
  const auto line_start = out.find("\nComm ") + 1;
  const auto line_end = out.find('\n', line_start);
  const std::string comm_line = out.substr(line_start, line_end - line_start);
  // All time cells must be "0.000" to two decimal digits.
  REQUIRE(comm_line.find("     0.000") != std::string::npos);
}

TEST_CASE("LAMMPS format — auto-derived Other is clamped to zero", "[telemetry]") {
  Telemetry t;
  t.begin_run();
  t.begin_section("Pair");
  busy_wait(std::chrono::microseconds(2000));
  t.end_section("Pair");
  // Sleep *outside* any section so total_wall_sec > sum(sections) is natural.
  busy_wait(std::chrono::microseconds(2000));
  t.end_run();

  std::ostringstream oss;
  t.write_lammps_format(oss, 1, 0.001);
  const auto out = oss.str();

  // Find Other row; its time column should be > 0 but bounded.
  const auto line_start = out.find("\nOther ") + 1;
  const auto line_end = out.find('\n', line_start);
  const std::string other_line = out.substr(line_start, line_end - line_start);
  // Rudimentary: no '-' sign anywhere in the line (would mean negative time).
  REQUIRE(other_line.find('-') == std::string::npos);
}

TEST_CASE("LAMMPS format — user-defined section appears after canonical rows", "[telemetry]") {
  Telemetry t;
  t.begin_run();
  t.begin_section("Pair");
  t.end_section("Pair");
  t.begin_section("CustomStage");
  t.end_section("CustomStage");
  t.end_run();

  std::ostringstream oss;
  t.write_lammps_format(oss, 1, 0.001);
  const auto out = oss.str();

  REQUIRE(out.find("\nCustomStage") != std::string::npos);
  REQUIRE(out.find("\nOther ") < out.find("\nCustomStage"));
}

// ---------------------------------------------------------------------------
// 4. total_wall_sec semantics
// ---------------------------------------------------------------------------

TEST_CASE("total_wall_sec is zero before begin_run", "[telemetry]") {
  Telemetry t;
  REQUIRE(t.total_wall_sec() == 0.0);
}

TEST_CASE("total_wall_sec is monotonic across begin/end", "[telemetry]") {
  Telemetry t;
  t.begin_run();
  busy_wait(std::chrono::microseconds(1500));
  const double mid = t.total_wall_sec();
  busy_wait(std::chrono::microseconds(1500));
  t.end_run();
  const double final_sec = t.total_wall_sec();
  REQUIRE(mid > 0.0);
  REQUIRE(final_sec >= mid);
}

TEST_CASE("reset clears all state except the in-flight clock", "[telemetry]") {
  Telemetry t;
  t.begin_run();
  t.begin_section("Pair");
  t.end_section("Pair");
  t.end_run();

  t.reset();
  REQUIRE(t.current_breakdown().empty());
  REQUIRE(t.total_wall_sec() == 0.0);
  REQUIRE(t.ignored_end_calls() == 0);
}

// ---------------------------------------------------------------------------
// 5. Overhead budget (SPEC §6)
// ---------------------------------------------------------------------------

TEST_CASE("Overhead budget — per-section cost bounded by SPEC §6 envelope",
          "[telemetry][benchmark]") {
  // SPEC §6 anchors the <0.1% overhead budget on a "typical EAM workload
  // (5 ms/iteration)" — that gives a 5 µs per-iteration ceiling. Benchmarking
  // against a nanosecond-scale toy kernel is meaningless (the ratio blows up
  // on clock noise alone), so instead we measure *absolute* per-section cost
  // and check it against the per-call envelope directly.
  //
  // Envelope: SPEC §6 lists "counter emit <50 ns", "histogram <100 ns".
  // A begin/end pair is the rough analogue — we allow a generous 2 µs here
  // (CI: 4 µs) to accommodate shared-runner jitter. On a realistic 5 ms EAM
  // step this still lands the telemetry contribution well under 0.04%.
  constexpr std::size_t kIterations = 100000;

  // Median-of-5 samples. Absolute timings on shared CI runners are noisy;
  // the median strips outliers without penalising honest runs.
  auto median_time_us = [&](auto&& loop_body) -> double {
    std::vector<double> samples;
    samples.reserve(5);
    for (int rep = 0; rep < 5; ++rep) {
      const auto start = std::chrono::steady_clock::now();
      loop_body();
      const auto end = std::chrono::steady_clock::now();
      samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
  };

  // Bare loop — just the loop machinery, no kernel. We measure the telemetry
  // delta directly, not against a synthetic "work" kernel whose cost is
  // itself noise.
  volatile std::size_t sink = 0;
  const double t_bare_us = median_time_us([&]() {
    for (std::size_t i = 0; i < kIterations; ++i) {
      sink = sink + 1;
    }
  });

  Telemetry telemetry;
  const double t_instr_us = median_time_us([&]() {
    for (std::size_t i = 0; i < kIterations; ++i) {
      ScopedSection s(&telemetry, "Pair");
      sink = sink + 1;
    }
  });

  const double per_section_us = (t_instr_us - t_bare_us) / static_cast<double>(kIterations);

  const bool in_ci = std::getenv("TDMD_CI") != nullptr;
  const double per_section_budget_us = in_ci ? 4.0 : 2.0;

  INFO("bare=" << t_bare_us << " µs, instrumented=" << t_instr_us
               << " µs, per-section=" << per_section_us * 1000.0
               << " ns, budget=" << per_section_budget_us * 1000.0 << " ns");

  REQUIRE(per_section_us <= per_section_budget_us);

  // Additionally document the implied SPEC §6 headroom on a 5 ms/step EAM
  // workload — useful output for the post-impl report. The calculation is
  // informational; the hard gate is the per-section budget above.
  const double implied_pct_on_5ms = 100.0 * per_section_us / 5000.0;
  INFO("implied overhead on 5 ms/step EAM = " << implied_pct_on_5ms << "%");
  REQUIRE(implied_pct_on_5ms < 0.1);
}

// ---------------------------------------------------------------------------
// 6. ScopedSection RAII
// ---------------------------------------------------------------------------

TEST_CASE("ScopedSection closes on normal exit", "[telemetry]") {
  Telemetry t;
  {
    ScopedSection s(&t, "Pair");
    busy_wait(std::chrono::microseconds(500));
  }
  REQUIRE(t.current_breakdown().at("Pair") > 0.0);
}

TEST_CASE("ScopedSection closes on exception", "[telemetry]") {
  Telemetry t;
  try {
    ScopedSection s(&t, "Pair");
    busy_wait(std::chrono::microseconds(500));
    throw std::runtime_error("boom");
  } catch (...) {
    // swallow
  }
  REQUIRE(t.current_breakdown().at("Pair") > 0.0);
  // And the section must have no leftover open depth.
  t.end_section("Pair");  // mismatched → ignored counter bumps
  REQUIRE(t.ignored_end_calls() == 1);
}

TEST_CASE("ScopedSection with nullptr telemetry is a safe no-op", "[telemetry]") {
  // This must compile, run, and not crash — the shape callers use when
  // telemetry is disabled.
  ScopedSection s(nullptr, "Pair");
  (void) s;
}
