// SPEC: docs/specs/potentials/SPEC.md §6.6 (SNAP parameter file format,
// LAMMPS-compatible). Exec pack: docs/development/m8_execution_pack.md T8.4.
//
// Parsers for LAMMPS `.snapcoeff` + `.snapparam` files. Matches the tokeniser
// в `verify/third_party/lammps/src/ML-SNAP/pair_snap.cpp` (`settings`,
// `coeff`, `read_file` methods): whitespace-separated tokens, `#` starts a
// comment line, blank lines skipped. No positional constraints beyond key
// ordering — `.snapparam` is key-value unordered; `.snapcoeff` header then
// per-species blocks. No unit_convert applied (metal units assumed per §6.6).

#include "tdmd/potentials/snap_file.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tdmd::potentials {

namespace {

// ---------------- line iterator ------------------------------------------

// Stateful line/token reader that tracks the source path and current line
// number so that diagnostics can point at the exact offending input. Blank
// lines and `#`-comment lines are skipped automatically; the rest is split
// on ASCII whitespace.
class TokenStream {
public:
  TokenStream(const std::string& path, std::ifstream& in) : path_(path), in_(in) {}

  // Advance to the next non-blank non-comment line. Returns false on EOF.
  bool next_line() {
    std::string raw;
    while (std::getline(in_, raw)) {
      ++line_no_;
      // Strip inline `#` comment.
      const auto hash = raw.find('#');
      if (hash != std::string::npos) {
        raw.resize(hash);
      }
      std::stringstream ss(raw);
      std::vector<std::string> toks;
      std::string t;
      while (ss >> t) {
        toks.push_back(std::move(t));
      }
      if (!toks.empty()) {
        tokens_ = std::move(toks);
        cursor_ = 0;
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool has_more_tokens() const noexcept { return cursor_ < tokens_.size(); }

  [[nodiscard]] const std::string& peek() const { return tokens_.at(cursor_); }

  std::string take() {
    if (cursor_ >= tokens_.size()) {
      throw_error("unexpected end of line");
    }
    return tokens_[cursor_++];
  }

  template <typename T>
  T take_as() {
    const auto tok = take();
    T out{};
    std::stringstream ss(tok);
    ss >> out;
    if (ss.fail() || !ss.eof()) {
      throw_error("failed to parse numeric token '" + tok + "'");
    }
    return out;
  }

  // Remaining tokens on the current line, joined with spaces — used for
  // diagnostic context.
  [[nodiscard]] std::string line_tail() const {
    std::string out;
    for (std::size_t i = cursor_; i < tokens_.size(); ++i) {
      if (!out.empty()) {
        out.push_back(' ');
      }
      out += tokens_[i];
    }
    return out;
  }

  [[noreturn]] void throw_error(const std::string& msg) const {
    std::ostringstream oss;
    oss << path_ << ':' << line_no_ << ": " << msg;
    throw std::runtime_error(oss.str());
  }

  [[nodiscard]] int line_no() const noexcept { return line_no_; }

private:
  const std::string& path_;
  std::ifstream& in_;
  std::vector<std::string> tokens_;
  std::size_t cursor_ = 0;
  int line_no_ = 0;
};

// ---------------- FNV-1a checksum over parsed fields ----------------------

// Stable 64-bit parameter identity key for `parameter_checksum()`. Order-
// sensitive по design — same values in different fields must hash differently.
constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
constexpr uint64_t kFnvPrime = 0x100000001b3ULL;

uint64_t fnv1a_update(uint64_t h, const void* data, std::size_t bytes) noexcept {
  const auto* p = static_cast<const uint8_t*>(data);
  for (std::size_t i = 0; i < bytes; ++i) {
    h ^= static_cast<uint64_t>(p[i]);
    h *= kFnvPrime;
  }
  return h;
}

template <typename T>
uint64_t fnv1a_update_scalar(uint64_t h, T value) noexcept {
  return fnv1a_update(h, &value, sizeof(T));
}

uint64_t snap_data_checksum(const SnapData& data) noexcept {
  uint64_t h = kFnvOffset;
  h = fnv1a_update_scalar(h, data.params.twojmax);
  h = fnv1a_update_scalar(h, data.params.rcutfac);
  h = fnv1a_update_scalar(h, data.params.rfac0);
  h = fnv1a_update_scalar(h, data.params.rmin0);
  h = fnv1a_update_scalar(h, data.params.switchflag);
  h = fnv1a_update_scalar(h, data.params.bzeroflag);
  h = fnv1a_update_scalar(h, data.params.quadraticflag);
  h = fnv1a_update_scalar(h, data.params.chemflag);
  h = fnv1a_update_scalar(h, data.params.bnormflag);
  h = fnv1a_update_scalar(h, data.params.wselfallflag);
  h = fnv1a_update_scalar(h, data.params.switchinnerflag);
  h = fnv1a_update_scalar(h, data.k_max);
  for (const auto& sp : data.species) {
    h = fnv1a_update(h, sp.name.data(), sp.name.size());
    h = fnv1a_update_scalar(h, sp.radius_elem);
    h = fnv1a_update_scalar(h, sp.weight_elem);
    for (double b : sp.beta) {
      h = fnv1a_update_scalar(h, b);
    }
  }
  return h;
}

// ---------------- param-file key handlers --------------------------------

void assign_bool(TokenStream& ts, bool& out) {
  const auto tok = ts.take();
  if (tok == "0" || tok == "false" || tok == "False" || tok == "FALSE") {
    out = false;
  } else if (tok == "1" || tok == "true" || tok == "True" || tok == "TRUE") {
    out = true;
  } else {
    ts.throw_error("expected boolean-like integer/string, got '" + tok + "'");
  }
}

void apply_param_key(TokenStream& ts, const std::string& key, SnapParams& out) {
  // LAMMPS pair_snap.cpp parses these keys в PairSNAP::read_files(); keep
  // the set in sync whenever LAMMPS SNAP adds новый optional field.
  if (key == "rcutfac") {
    out.rcutfac = ts.take_as<double>();
  } else if (key == "twojmax") {
    out.twojmax = ts.take_as<int>();
    if (out.twojmax < 0 || (out.twojmax % 2) != 0) {
      ts.throw_error("twojmax must be non-negative and even");
    }
  } else if (key == "rfac0") {
    out.rfac0 = ts.take_as<double>();
  } else if (key == "rmin0") {
    out.rmin0 = ts.take_as<double>();
  } else if (key == "switchflag") {
    assign_bool(ts, out.switchflag);
  } else if (key == "bzeroflag") {
    assign_bool(ts, out.bzeroflag);
  } else if (key == "quadraticflag") {
    assign_bool(ts, out.quadraticflag);
  } else if (key == "chemflag") {
    assign_bool(ts, out.chemflag);
    if (out.chemflag) {
      ts.throw_error(
          "chemflag=1 (multi-species chem SNAP) is deferred to M9+; "
          "TDMD v1 accepts only chemflag=0");
    }
  } else if (key == "bnormflag") {
    assign_bool(ts, out.bnormflag);
  } else if (key == "wselfallflag") {
    assign_bool(ts, out.wselfallflag);
  } else if (key == "switchinnerflag") {
    assign_bool(ts, out.switchinnerflag);
    if (out.switchinnerflag) {
      ts.throw_error("switchinnerflag=1 (inner-cutoff switching) is deferred to M9+");
    }
  } else {
    ts.throw_error("unknown .snapparam key '" + key + "'");
  }
}

}  // namespace

// ---------------- public API ---------------------------------------------

int snap_k_max(int twojmax) noexcept {
  // Matches LAMMPS `SNA::compute_ncoeff` (single-species, chem_flag=0)
  // per sna.cpp ~L1511. Count (j1, j2, j) triples with j1 ≥ j2, j1 ≤ j ≤
  // min(twojmax, j1+j2), step 2, and j ≥ j1.
  if (twojmax < 0) {
    return 0;
  }
  int ncount = 0;
  for (int j1 = 0; j1 <= twojmax; ++j1) {
    for (int j2 = 0; j2 <= j1; ++j2) {
      const int j_high = std::min(twojmax, j1 + j2);
      for (int j = j1 - j2; j <= j_high; j += 2) {
        if (j >= j1) {
          ++ncount;
        }
      }
    }
  }
  return ncount;
}

double SnapData::max_pairwise_cutoff() const noexcept {
  double rcut_max = 0.0;
  for (const auto& a : species) {
    for (const auto& b : species) {
      const double rcut = params.rcutfac * (a.radius_elem + b.radius_elem);
      if (rcut > rcut_max) {
        rcut_max = rcut;
      }
    }
  }
  return rcut_max;
}

SnapParams parse_snap_param(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error(path + ": cannot open for reading");
  }
  TokenStream ts(path, in);
  SnapParams out;
  bool saw_twojmax = false;
  bool saw_rcutfac = false;
  while (ts.next_line()) {
    while (ts.has_more_tokens()) {
      const auto key = ts.take();
      apply_param_key(ts, key, out);
      if (key == "twojmax") {
        saw_twojmax = true;
      }
      if (key == "rcutfac") {
        saw_rcutfac = true;
      }
    }
  }
  if (!saw_twojmax) {
    throw std::runtime_error(path + ": missing required key 'twojmax'");
  }
  if (!saw_rcutfac) {
    throw std::runtime_error(path + ": missing required key 'rcutfac'");
  }
  return out;
}

std::vector<SnapSpecies> parse_snap_coeff(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error(path + ": cannot open for reading");
  }
  TokenStream ts(path, in);

  if (!ts.next_line()) {
    throw std::runtime_error(path + ": empty file (expected <n_species> <n_coeffs> header)");
  }
  const int n_species = ts.take_as<int>();
  const int n_coeffs = ts.take_as<int>();
  if (n_species <= 0) {
    ts.throw_error("n_species must be positive, got " + std::to_string(n_species));
  }
  if (n_coeffs <= 0) {
    ts.throw_error("n_coeffs must be positive, got " + std::to_string(n_coeffs));
  }
  if (ts.has_more_tokens()) {
    ts.throw_error("unexpected trailing tokens on header line: '" + ts.line_tail() + "'");
  }

  std::vector<SnapSpecies> out;
  out.reserve(static_cast<std::size_t>(n_species));

  for (int s = 0; s < n_species; ++s) {
    if (!ts.next_line()) {
      throw std::runtime_error(path + ": EOF while reading species " + std::to_string(s));
    }
    SnapSpecies sp;
    sp.name = ts.take();
    sp.radius_elem = ts.take_as<double>();
    sp.weight_elem = ts.take_as<double>();
    if (ts.has_more_tokens()) {
      ts.throw_error("unexpected trailing tokens on species header: '" + ts.line_tail() + "'");
    }
    sp.beta.reserve(static_cast<std::size_t>(n_coeffs));
    while (static_cast<int>(sp.beta.size()) < n_coeffs) {
      if (!ts.next_line()) {
        throw std::runtime_error(path + ": EOF while reading β for species " + sp.name);
      }
      while (ts.has_more_tokens() && static_cast<int>(sp.beta.size()) < n_coeffs) {
        sp.beta.push_back(ts.take_as<double>());
      }
      if (ts.has_more_tokens() && static_cast<int>(sp.beta.size()) == n_coeffs) {
        ts.throw_error("extra β values beyond declared n_coeffs=" + std::to_string(n_coeffs));
      }
    }
    out.push_back(std::move(sp));
  }

  if (ts.next_line()) {
    ts.throw_error("unexpected trailing content after last species block");
  }
  return out;
}

SnapData parse_snap_files(const std::string& coeff_path, const std::string& param_path) {
  SnapData out;
  out.params = parse_snap_param(param_path);
  out.species = parse_snap_coeff(coeff_path);

  // Derive k_max + cross-check against declared n_coeffs (coefficient count
  // in the `.snapcoeff` header = `1 + k_max` for linear SNAP, or
  // `1 + k_max + k_max·(k_max+1)/2` for quadratic).
  out.k_max = snap_k_max(out.params.twojmax);
  const std::size_t expected_linear = static_cast<std::size_t>(out.k_max) + 1;
  const std::size_t expected_quad =
      expected_linear +
      static_cast<std::size_t>(out.k_max) * (static_cast<std::size_t>(out.k_max) + 1) / 2;
  const std::size_t expected = out.params.quadraticflag ? expected_quad : expected_linear;

  for (const auto& sp : out.species) {
    if (sp.beta.size() != expected) {
      std::ostringstream oss;
      oss << coeff_path << ": species '" << sp.name << "' has " << sp.beta.size()
          << " β coefficients but " << expected << " expected (twojmax=" << out.params.twojmax
          << ", k_max=" << out.k_max << ", quadraticflag=" << (out.params.quadraticflag ? 1 : 0)
          << ")";
      throw std::runtime_error(oss.str());
    }
  }

  // Build rcut_sq_ab (symmetric n×n, row-major).
  const std::size_t n = out.species.size();
  out.rcut_sq_ab.assign(n * n, 0.0);
  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t b = 0; b < n; ++b) {
      const double rcut =
          out.params.rcutfac * (out.species[a].radius_elem + out.species[b].radius_elem);
      out.rcut_sq_ab[SnapData::pair_index(a, b, n)] = rcut * rcut;
    }
  }

  out.checksum = snap_data_checksum(out);
  return out;
}

}  // namespace tdmd::potentials
