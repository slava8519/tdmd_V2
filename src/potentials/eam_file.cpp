#include "tdmd/potentials/eam_file.hpp"

#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Reference: LAMMPS `pair_eam_alloy.cpp::read_file()` and
// `pair_eam_fs.cpp::read_file()` describe the same on-disk format this
// parser consumes. The TDMD implementation is a fresh rewrite — no source
// is copied; LAMMPS is only a format reference.

namespace tdmd::potentials {

namespace {

struct Token {
  std::string text;
  std::size_t line;
};

// Minimum floating-point precision required to round-trip a double through
// std::to_chars / std::strtod (used by std::stod). Setfl writers use plain
// decimal notation; `stod` handles this losslessly for any IEEE 754 double
// whose decimal representation is ≥ 17 significant digits — matching the
// output precision LAMMPS itself uses.

class TokenView {
public:
  TokenView(std::string path, std::vector<Token> toks)
      : path_(std::move(path)), toks_(std::move(toks)) {}

  [[nodiscard]] bool empty() const noexcept { return pos_ >= toks_.size(); }
  [[nodiscard]] std::size_t current_line() const noexcept {
    return empty() ? last_line() : toks_[pos_].line;
  }

  // Consume tokens from the current line, requiring at least `min_count`
  // tokens on it. After return, position is past the last token on that
  // line (any additional tokens on it are dropped). Mirrors LAMMPS's
  // `next_values(N)` semantics — N is a lower bound, not an exact count,
  // so optional trailing fields (e.g. lattice type on the per-species
  // metadata line) stay compatible.
  std::vector<std::string> consume_line(std::size_t min_count) {
    if (empty())
      throw_here("unexpected end of file");
    const std::size_t line = toks_[pos_].line;
    std::vector<std::string> out;
    while (pos_ < toks_.size() && toks_[pos_].line == line) {
      out.push_back(std::move(toks_[pos_].text));
      ++pos_;
    }
    if (out.size() < min_count) {
      std::ostringstream msg;
      msg << "expected at least " << min_count << " tokens, got " << out.size();
      throw_at(line, msg.str());
    }
    return out;
  }

  // Consume the next token as a double. Spans line boundaries naturally —
  // used for the bulk F(ρ), ρ(r), and φ(r) arrays.
  double next_double() {
    if (empty())
      throw_here("unexpected end of file in numeric array");
    const Token& t = toks_[pos_];
    try {
      std::size_t consumed = 0;
      const double v = std::stod(t.text, &consumed);
      if (consumed != t.text.size()) {
        throw_at(t.line, "expected number, got '" + t.text + "'");
      }
      ++pos_;
      return v;
    } catch (const std::invalid_argument&) {
      throw_at(t.line, "expected number, got '" + t.text + "'");
    } catch (const std::out_of_range&) {
      throw_at(t.line, "number out of double range: '" + t.text + "'");
    }
  }

  [[noreturn]] void throw_here(const std::string& msg) const {
    throw std::runtime_error(path_ + ": " + msg);
  }

  [[noreturn]] void throw_at(std::size_t line, const std::string& msg) const {
    std::ostringstream out;
    out << path_ << ":" << line << ": " << msg;
    throw std::runtime_error(out.str());
  }

private:
  [[nodiscard]] std::size_t last_line() const noexcept {
    return toks_.empty() ? 0 : toks_.back().line;
  }

  std::string path_;
  std::vector<Token> toks_;
  std::size_t pos_ = 0;
};

// Slurp the file, skip the first `skip` lines as comments, tokenize the
// remainder into a flat (text, line) stream. Line numbers start at 1 and
// include the skipped header lines so diagnostics point to the real line.
TokenView tokenize_file(const std::string& path, std::size_t skip) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error(path + ": cannot open for reading");
  }
  std::vector<Token> tokens;
  std::string raw;
  std::size_t line = 0;
  while (std::getline(in, raw)) {
    ++line;
    if (line <= skip)
      continue;
    std::istringstream iss(raw);
    std::string tok;
    while (iss >> tok) {
      tokens.push_back({tok, line});
    }
  }
  return TokenView(path, std::move(tokens));
}

int parse_int_field(const std::string& text,
                    TokenView& view,
                    std::size_t line,
                    const std::string& field_name) {
  try {
    std::size_t consumed = 0;
    const long long v = std::stoll(text, &consumed);
    if (consumed != text.size()) {
      view.throw_at(line, "expected integer " + field_name + ", got '" + text + "'");
    }
    if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
      view.throw_at(line, field_name + " out of int range");
    }
    return static_cast<int>(v);
  } catch (const std::invalid_argument&) {
    view.throw_at(line, "expected integer " + field_name + ", got '" + text + "'");
  } catch (const std::out_of_range&) {
    view.throw_at(line, field_name + " out of range: '" + text + "'");
  }
}

double parse_double_field(const std::string& text,
                          TokenView& view,
                          std::size_t line,
                          const std::string& field_name) {
  try {
    std::size_t consumed = 0;
    const double v = std::stod(text, &consumed);
    if (consumed != text.size()) {
      view.throw_at(line, "expected number " + field_name + ", got '" + text + "'");
    }
    return v;
  } catch (const std::invalid_argument&) {
    view.throw_at(line, "expected number " + field_name + ", got '" + text + "'");
  } catch (const std::out_of_range&) {
    view.throw_at(line, field_name + " out of double range: '" + text + "'");
  }
}

// Shared validation of the "<N_rho> <d_rho> <N_r> <d_r> <r_cutoff>" line.
struct GridParams {
  int nrho;
  double drho;
  int nr;
  double dr;
  double cutoff;
};

GridParams parse_grid_line(TokenView& view) {
  const std::size_t line = view.current_line();
  auto toks = view.consume_line(5);
  GridParams g;
  g.nrho = parse_int_field(toks[0], view, line, "N_rho");
  g.drho = parse_double_field(toks[1], view, line, "d_rho");
  g.nr = parse_int_field(toks[2], view, line, "N_r");
  g.dr = parse_double_field(toks[3], view, line, "d_r");
  g.cutoff = parse_double_field(toks[4], view, line, "r_cutoff");
  if (g.nrho < 5 || g.nr < 5) {
    view.throw_at(line, "N_rho and N_r must be ≥ 5 (spline minimum)");
  }
  if (!(g.drho > 0.0) || !(g.dr > 0.0) || !(g.cutoff > 0.0)) {
    view.throw_at(line, "d_rho, d_r, and r_cutoff must be strictly positive");
  }
  return g;
}

int parse_species_header(TokenView& view, std::vector<std::string>& names) {
  const std::size_t line = view.current_line();
  auto toks = view.consume_line(1);
  const int n_species = parse_int_field(toks[0], view, line, "N_species");
  if (n_species <= 0) {
    view.throw_at(line, "N_species must be positive");
  }
  if (toks.size() != static_cast<std::size_t>(n_species) + 1) {
    std::ostringstream msg;
    msg << "expected N_species + species names on one line (N_species=" << n_species << ", got "
        << toks.size() << " tokens)";
    view.throw_at(line, msg.str());
  }
  names.assign(toks.begin() + 1, toks.end());
  return n_species;
}

double parse_species_meta_mass(TokenView& view) {
  const std::size_t line = view.current_line();
  // "Z mass [a lattice_type ...]" — only mass is consumed; any trailing
  // tokens (atomic number, lattice constant, lattice type string) are
  // discarded by consume_line. LAMMPS's reader does the same.
  auto toks = view.consume_line(2);
  parse_int_field(toks[0], view, line, "atomic_number");  // validated, dropped
  return parse_double_field(toks[1], view, line, "mass");
}

std::vector<double> read_doubles(TokenView& view, std::size_t n) {
  std::vector<double> out;
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    out.push_back(view.next_double());
  }
  return out;
}

}  // namespace

std::size_t EamAlloyData::pair_index(std::size_t alpha, std::size_t beta) noexcept {
  const std::size_t hi = (alpha >= beta) ? alpha : beta;
  const std::size_t lo = (alpha >= beta) ? beta : alpha;
  return hi * (hi + 1) / 2 + lo;
}

std::size_t EamFsData::pair_index(std::size_t alpha, std::size_t beta) noexcept {
  const std::size_t hi = (alpha >= beta) ? alpha : beta;
  const std::size_t lo = (alpha >= beta) ? beta : alpha;
  return hi * (hi + 1) / 2 + lo;
}

EamAlloyData parse_eam_alloy(const std::string& path) {
  TokenView view = tokenize_file(path, 3);
  EamAlloyData data;

  const int n_species = parse_species_header(view, data.species_names);
  const auto n = static_cast<std::size_t>(n_species);

  const GridParams g = parse_grid_line(view);
  data.nrho = g.nrho;
  data.drho = g.drho;
  data.nr = g.nr;
  data.dr = g.dr;
  data.cutoff = g.cutoff;

  data.masses.reserve(n);
  data.F_rho.reserve(n);
  data.rho_r.reserve(n);

  for (std::size_t i = 0; i < n; ++i) {
    data.masses.push_back(parse_species_meta_mass(view));

    auto f_vals = read_doubles(view, static_cast<std::size_t>(g.nrho));
    data.F_rho.emplace_back(0.0, g.drho, std::move(f_vals));

    auto rho_vals = read_doubles(view, static_cast<std::size_t>(g.nr));
    data.rho_r.emplace_back(0.0, g.dr, std::move(rho_vals));
  }

  const std::size_t n_pairs = n * (n + 1) / 2;
  data.z2r.reserve(n_pairs);
  for (std::size_t p = 0; p < n_pairs; ++p) {
    auto vals = read_doubles(view, static_cast<std::size_t>(g.nr));
    data.z2r.emplace_back(0.0, g.dr, std::move(vals));
  }

  return data;
}

EamFsData parse_eam_fs(const std::string& path) {
  TokenView view = tokenize_file(path, 3);
  EamFsData data;

  const int n_species = parse_species_header(view, data.species_names);
  const auto n = static_cast<std::size_t>(n_species);

  const GridParams g = parse_grid_line(view);
  data.nrho = g.nrho;
  data.drho = g.drho;
  data.nr = g.nr;
  data.dr = g.dr;
  data.cutoff = g.cutoff;

  data.masses.reserve(n);
  data.F_rho.reserve(n);
  data.rho_ij.reserve(n * n);

  for (std::size_t i = 0; i < n; ++i) {
    data.masses.push_back(parse_species_meta_mass(view));

    auto f_vals = read_doubles(view, static_cast<std::size_t>(g.nrho));
    data.F_rho.emplace_back(0.0, g.drho, std::move(f_vals));

    // N per-neighbour ρ_{αβ}(r) tables for this α — one per species β.
    for (std::size_t j = 0; j < n; ++j) {
      auto rho_vals = read_doubles(view, static_cast<std::size_t>(g.nr));
      data.rho_ij.emplace_back(0.0, g.dr, std::move(rho_vals));
    }
  }

  const std::size_t n_pairs = n * (n + 1) / 2;
  data.z2r.reserve(n_pairs);
  for (std::size_t p = 0; p < n_pairs; ++p) {
    auto vals = read_doubles(view, static_cast<std::size_t>(g.nr));
    data.z2r.emplace_back(0.0, g.dr, std::move(vals));
  }

  return data;
}

}  // namespace tdmd::potentials
