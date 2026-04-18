#include "tdmd/telemetry/telemetry.hpp"

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

namespace tdmd::telemetry {

namespace {

// ---------------------------------------------------------------------------
// LAMMPS-format column widths (SPEC §4.2)
// ---------------------------------------------------------------------------
// Header the oracle emits (exact bytes):
//
//   Section |  min time  |  avg time  |  max time  |%varavg| %total
//   ---------------------------------------------------------------
//   Pair    |     1.200  |     1.250  |     1.310  |   2.1 |  54.3
//
// Decomposed left-to-right:
//
//   name  (left-aligned, trailing space)     → 7 chars before the first `|`
//   time  (right-aligned 10.3f, trailing 2)  → 12 chars between successive `|`
//   var   (right-aligned  5.1f, trailing 1)  → 7 chars for the %varavg column
//   total (right-aligned  5.1f, leading 1)   → 6 chars for the %total column
//
// The separators are literal `|`. The dashed rule is 63 chars — the width of
// the header line.
constexpr const char* kBreakdownHeader =
    "Section |  min time  |  avg time  |  max time  |%varavg| %total";
constexpr const char* kBreakdownRule =
    "---------------------------------------------------------------";

void write_time_cell(std::ostream& out, double seconds) {
  // `%10.3f  ` — 10 chars of value + 2 trailing spaces ⇒ 12-char cell body
  out << std::setw(10) << std::fixed << std::setprecision(3) << seconds << "  ";
}

void write_row(std::ostream& out,
               const std::string& name,
               double min_sec,
               double avg_sec,
               double max_sec,
               double var_pct,
               double total_pct) {
  // Name column: left-aligned in 7 chars, then " |".
  out << std::left << std::setw(7) << name << std::right << " |";
  write_time_cell(out, min_sec);
  out << "|";
  write_time_cell(out, avg_sec);
  out << "|";
  write_time_cell(out, max_sec);
  out << "|";
  out << std::setw(6) << std::fixed << std::setprecision(1) << var_pct << " |";
  out << std::setw(6) << std::fixed << std::setprecision(1) << total_pct;
  out << '\n';
}

// Escape a string for JSON. The only characters we emit are section names
// (ASCII-ish) and literal doubles, so a narrow escaper is sufficient for the
// control characters that cannot appear unescaped in a JSON string.
std::string json_escape(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          std::ostringstream oss;
          oss << "\\u" << std::setw(4) << std::setfill('0') << std::hex << static_cast<int>(c);
          out += oss.str();
        } else {
          out += c;
        }
        break;
    }
  }
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Telemetry
// ---------------------------------------------------------------------------

Telemetry::Telemetry() = default;

void Telemetry::begin_run() {
  run_begin_ = Clock::now();
  run_begun_ = true;
  run_ended_ = false;
}

void Telemetry::end_run() {
  run_end_ = Clock::now();
  run_ended_ = true;
}

void Telemetry::begin_section(std::string_view name) {
  // First-seen order tracking — we want stable output regardless of how
  // std::map iterates.
  auto it = sections_.find(name);
  if (it == sections_.end()) {
    auto [inserted_it, inserted] = sections_.emplace(std::string(name), SectionState{});
    it = inserted_it;
    section_order_.emplace_back(std::string(name));
  }
  // Only the outermost open anchors the start timestamp — nested begins of
  // the same name inherit the existing open window.
  if (it->second.open_depth == 0) {
    it->second.open_start = Clock::now();
  }
  ++it->second.open_depth;
}

void Telemetry::end_section(std::string_view name) {
  auto it = sections_.find(name);
  if (it == sections_.end() || it->second.open_depth == 0) {
    // Mismatched end — do not mutate accumulators, just count the
    // misuse. LAMMPS similarly ignores rather than throwing.
    ++ignored_end_calls_;
    return;
  }
  --it->second.open_depth;
  if (it->second.open_depth == 0) {
    const auto dur = Clock::now() - it->second.open_start;
    it->second.accumulated_sec += std::chrono::duration<double>(dur).count();
  }
}

std::map<std::string, double> Telemetry::current_breakdown() const {
  std::map<std::string, double> out;
  for (const auto& [name, state] : sections_) {
    double t = state.accumulated_sec;
    // If still open mid-run, include the running delta so snapshots are
    // meaningful even before end_section is called.
    if (state.open_depth > 0) {
      const auto now = Clock::now();
      t += std::chrono::duration<double>(now - state.open_start).count();
    }
    out[name] = t;
  }
  return out;
}

double Telemetry::total_wall_sec() const {
  if (!run_begun_) {
    return 0.0;
  }
  const auto end = run_ended_ ? run_end_ : Clock::now();
  return std::chrono::duration<double>(end - run_begin_).count();
}

void Telemetry::write_jsonl(std::ostream& out) const {
  const double total = total_wall_sec();
  const auto breakdown = current_breakdown();

  // Single-object JSONL line — `{"k":v, ...}`. Per-step streaming lands at
  // M3+; for M2 one snapshot is enough.
  out << "{\"event\":\"run_end\"";
  out << ",\"total_wall_sec\":" << std::fixed << std::setprecision(9) << total;
  out << ",\"sections\":{";
  // Use first-seen order so diffs are stable.
  bool first = true;
  for (const auto& name : section_order_) {
    auto it = breakdown.find(name);
    if (it == breakdown.end()) {
      continue;
    }
    if (!first) {
      out << ",";
    }
    first = false;
    out << "\"" << json_escape(name) << "\":" << std::fixed << std::setprecision(9) << it->second;
  }
  out << "}";
  out << ",\"ignored_end_calls\":" << ignored_end_calls_;
  out << "}\n";
}

void Telemetry::write_lammps_format(std::ostream& out, std::uint64_t n_steps, double dt_ps) const {
  const double total = total_wall_sec();
  const auto breakdown = current_breakdown();

  // ---- Performance: tau/day, ns/day, ms/timestep ------------------------
  // tau/day: LAMMPS convention — unitless integration time per wall day.
  //   tau/day = steps_per_wall_second * seconds_per_day
  //           = (n_steps / total_wall_sec) * 86400
  //
  // ns/day:  (n_steps * dt_ps) / 1000 ps_per_ns → ns of simulated time; per
  //          second then * 86400.
  //
  // ms/timestep:  total_wall_sec / n_steps * 1000.
  //
  // Guard against total == 0 (benchmarks sometimes aggregate zero runs) by
  // falling back to nan-free zeros.
  std::ios::fmtflags saved_flags = out.flags();
  const auto saved_prec = out.precision();

  const double steps = static_cast<double>(n_steps);
  const double tau_per_day = total > 0 ? (steps / total) * 86400.0 : 0.0;
  const double ns_per_day = total > 0 ? (steps * dt_ps / 1000.0) / total * 86400.0 : 0.0;
  const double ms_per_step = n_steps > 0 ? (total / steps) * 1000.0 : 0.0;

  out << "Performance: " << std::fixed << std::setprecision(1) << tau_per_day << " tau/day, "
      << std::setprecision(4) << ns_per_day << " ns/day, " << std::setprecision(3) << ms_per_step
      << " ms/timestep\n";
  out << '\n';

  // ---- MPI task timing breakdown ----------------------------------------
  out << "MPI task timing breakdown:\n";
  out << kBreakdownHeader << '\n';
  out << kBreakdownRule << '\n';

  // Emit canonical LAMMPS order first so every run produces a consistent
  // prefix — then any extra user-defined sections trailing in first-seen
  // order. Sections absent from the run still print as zero rows so the
  // oracle diff is clean.
  const std::vector<std::string> canonical = {"Pair", "Neigh", "Comm", "Output", "Other"};

  double sum_known = 0.0;
  auto sec_time = [&](const std::string& n) {
    auto it = breakdown.find(n);
    return it == breakdown.end() ? 0.0 : it->second;
  };

  // Auto-derive Other as (total - sum_of_other_canonicals) if the caller
  // did not report it explicitly. Clamp to zero so tiny-step edge cases
  // (where section overhead slightly exceeds measured total) do not emit
  // a misleading negative.
  const bool caller_wrote_other = sections_.find(std::string("Other")) != sections_.end();

  for (const auto& n : canonical) {
    if (n == "Other" && !caller_wrote_other) {
      const double t_other = std::max(0.0, total - sum_known);
      const double pct = total > 0 ? 100.0 * t_other / total : 0.0;
      write_row(out, n, t_other, t_other, t_other, 0.0, pct);
    } else {
      const double t = sec_time(n);
      sum_known += t;
      const double pct = total > 0 ? 100.0 * t / total : 0.0;
      // Single-rank M2 → min = avg = max; %varavg = 0.
      write_row(out, n, t, t, t, 0.0, pct);
    }
  }

  // Trailing user-defined sections (not part of the canonical set).
  for (const auto& name : section_order_) {
    if (std::find(canonical.begin(), canonical.end(), name) != canonical.end()) {
      continue;
    }
    const double t = sec_time(name);
    const double pct = total > 0 ? 100.0 * t / total : 0.0;
    write_row(out, name, t, t, t, 0.0, pct);
  }

  out << kBreakdownRule << '\n';
  write_row(out, "Total", total, total, total, 0.0, total > 0 ? 100.0 : 0.0);

  out.flags(saved_flags);
  out.precision(saved_prec);
}

void Telemetry::reset() {
  sections_.clear();
  section_order_.clear();
  run_begun_ = false;
  run_ended_ = false;
  ignored_end_calls_ = 0;
}

// ---------------------------------------------------------------------------
// ScopedSection
// ---------------------------------------------------------------------------

ScopedSection::ScopedSection(Telemetry* t, std::string_view name) noexcept
    : telemetry_(t), name_(name) {
  if (telemetry_ != nullptr) {
    telemetry_->begin_section(name_);
  }
}

ScopedSection::~ScopedSection() {
  if (telemetry_ != nullptr) {
    telemetry_->end_section(name_);
  }
}

}  // namespace tdmd::telemetry
