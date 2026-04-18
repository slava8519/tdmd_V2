#pragma once

// SPEC: docs/specs/telemetry/SPEC.md §§3, 4.2, 6
// Exec pack: docs/development/m2_execution_pack.md T2.12
//
// M2 telemetry skeleton — a minimal, single-threaded timing aggregator. Scope
// is deliberately narrow so later milestones can layer on without rewriting:
//
//   * per-section wall time via `begin_section` / `end_section`;
//   * JSONL snapshot at end-of-run (SPEC §4.1 shape, single line for M2 —
//     per-step streaming is M3+);
//   * LAMMPS-compatible breakdown table (SPEC §4.2) for `tdmd run --timing`.
//
// The *canonical* section names mirror LAMMPS (Pair, Neigh, Comm, Output,
// Other) so the table can be diffed line-by-line against an oracle run. M2
// has no MPI, so `Comm` is present as a column and reports 0.0 — keeping the
// column there from day one avoids a format bump later.
//
// API freeze notice
// -----------------
// `begin_section` / `end_section` are frozen from M2: M6 will introduce NVTX
// ranges through these same entry points (see SPEC §5), so the signatures
// must not drift. `ScopedSection` is the preferred call site because it is
// exception-safe and is the shape NVTX wrappers will extend.
//
// Overhead budget: SPEC §6 caps telemetry at <0.1% of iteration wall time.
// A benchmark test enforces this in CI.
//
// Threading: M2 is single-thread only. Multi-thread safety is M3+ (requires
// ring-buffer + async writer — see SPEC §6.2). Do not share an instance
// across threads.

#include <chrono>
#include <cstdint>
#include <iosfwd>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace tdmd::telemetry {

// Accumulated wall time aggregator. One instance per run.
//
// M2: single-threaded; do not share across threads. Multi-thread safety is
// deferred to M3+ (ring-buffer + async writer per SPEC §6.2).
class Telemetry {
public:
  using Clock = std::chrono::steady_clock;

  Telemetry();

  Telemetry(const Telemetry&) = delete;
  Telemetry& operator=(const Telemetry&) = delete;
  Telemetry(Telemetry&&) = delete;
  Telemetry& operator=(Telemetry&&) = delete;

  // Run lifecycle. `begin_run` captures a start timestamp used to compute
  // `total_wall_sec()`; `end_run` freezes it. Both are idempotent in the
  // sense that a second call rebases the frozen values.
  void begin_run();
  void end_run();

  // Section instrumentation. **API frozen from M2** — M6 NVTX hooks will
  // thread through these exact signatures, so do not change them.
  //
  // `name` is hashed by value into an internal map. Sections nest: a
  // `begin_section("Pair")` while "Neigh" is still open is legal, but each
  // `end_section` must match the most recent unmatched `begin_section` of
  // the same name (LIFO). Mismatched end_section is ignored (logged as
  // `ignored_end_calls` counter) rather than throwing, matching LAMMPS's
  // permissive behaviour under release builds.
  void begin_section(std::string_view name);
  void end_section(std::string_view name);

  // Snapshot accessors — always safe to call, even mid-run. Mid-run calls
  // return whatever has been accumulated so far.
  [[nodiscard]] std::map<std::string, double> current_breakdown() const;
  [[nodiscard]] double total_wall_sec() const;
  [[nodiscard]] std::uint64_t ignored_end_calls() const noexcept { return ignored_end_calls_; }

  // Formatted output. JSONL emits a single-line JSON object (one snapshot
  // per run in M2; per-step streaming is M3+). `write_lammps_format` writes
  // the §4.2 table — pass `n_steps` and `dt_ps` so it can compute the
  // Performance: tau/day, ns/day, ms/timestep preamble.
  void write_jsonl(std::ostream& out) const;
  void write_lammps_format(std::ostream& out, std::uint64_t n_steps, double dt_ps) const;

  // Reset aggregation state. Does not touch the run-start timestamp; callers
  // wanting a clean slate must call begin_run() afterwards.
  void reset();

private:
  struct SectionState {
    double accumulated_sec = 0.0;    // total time closed into this section
    Clock::time_point open_start{};  // set when open_depth_ > 0 && outermost open
    int open_depth = 0;              // nesting count for this specific section
  };

  std::map<std::string, SectionState, std::less<>> sections_;

  // Preserve call order for output stability — LAMMPS emits sections in a
  // fixed order (Pair, Neigh, Comm, Output, Other), and we do the same by
  // tracking first-seen order here.
  std::vector<std::string> section_order_;

  Clock::time_point run_begin_{};
  Clock::time_point run_end_{};
  bool run_begun_ = false;
  bool run_ended_ = false;

  std::uint64_t ignored_end_calls_ = 0;
};

// RAII scope guard for `Telemetry::begin_section` / `end_section`. Prefer
// this to manual begin/end pairs — it is exception-safe and is the entry
// point that will grow NVTX integration at M6.
//
// A nullptr Telemetry* is explicitly allowed: the guard becomes a no-op,
// which lets call sites avoid conditionals when telemetry is disabled.
class ScopedSection {
public:
  ScopedSection(Telemetry* t, std::string_view name) noexcept;
  ~ScopedSection();

  ScopedSection(const ScopedSection&) = delete;
  ScopedSection& operator=(const ScopedSection&) = delete;
  ScopedSection(ScopedSection&&) = delete;
  ScopedSection& operator=(ScopedSection&&) = delete;

private:
  Telemetry* telemetry_;
  std::string name_;  // stored by value so the string_view's backing survives
};

}  // namespace tdmd::telemetry
