#include "tdmd/io/lammps_data_reader.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <istream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace tdmd::io {

namespace {

// ---- Error helpers ---------------------------------------------------------

std::string build_parse_error(std::size_t line_number, std::string_view message) {
  std::ostringstream oss;
  oss << "LAMMPS data parse error at line " << line_number << ": " << message;
  return oss.str();
}

[[noreturn]] void throw_parse(std::size_t line, std::string_view message) {
  throw LammpsDataParseError(line, message);
}

// ---- Small string utilities ------------------------------------------------

std::string_view strip_comment(std::string_view s) {
  const auto p = s.find('#');
  return p == std::string_view::npos ? s : s.substr(0, p);
}

std::string_view trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) {
    s.remove_prefix(1);
  }
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) {
    s.remove_suffix(1);
  }
  return s;
}

std::vector<std::string_view> tokenize(std::string_view s) {
  std::vector<std::string_view> out;
  std::size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) {
      ++i;
    }
    if (i >= s.size()) {
      break;
    }
    const auto start = i;
    while (i < s.size() && s[i] != ' ' && s[i] != '\t') {
      ++i;
    }
    out.emplace_back(s.substr(start, i - start));
  }
  return out;
}

// ---- Number parsing (strict: whole token must match) ----------------------

bool parse_int64(std::string_view s, std::int64_t& out) {
  const char* begin = s.data();
  const char* end = s.data() + s.size();
  auto res = std::from_chars(begin, end, out);
  return res.ec == std::errc() && res.ptr == end;
}

bool parse_uint64(std::string_view s, std::uint64_t& out) {
  const char* begin = s.data();
  const char* end = s.data() + s.size();
  auto res = std::from_chars(begin, end, out);
  return res.ec == std::errc() && res.ptr == end;
}

bool parse_double(std::string_view s, double& out) {
  // libstdc++ 11+ and libc++ 17+ support `std::from_chars` for double.
  const char* begin = s.data();
  const char* end = s.data() + s.size();
  auto res = std::from_chars(begin, end, out);
  return res.ec == std::errc() && res.ptr == end;
}

// ---- Line stream wrapper ---------------------------------------------------

struct RawLine {
  std::string text;        // CR-stripped original (no comment strip yet)
  std::size_t number = 0;  // 1-based physical line number
  bool eof = false;
};

class LineReader {
public:
  explicit LineReader(std::istream& in) : in_(in) {}

  // Reads the next physical line regardless of content. Sets `eof=true` on EOF.
  RawLine next() {
    RawLine out;
    std::string s;
    if (!std::getline(in_, s)) {
      out.eof = true;
      out.number = line_number_;
      return out;
    }
    line_number_ += 1;
    out.number = line_number_;
    if (!s.empty() && s.back() == '\r') {
      s.pop_back();
    }
    out.text = std::move(s);
    return out;
  }

  [[nodiscard]] std::size_t current_line() const noexcept { return line_number_; }

private:
  std::istream& in_;
  std::size_t line_number_ = 0;
};

// Advances past blank / comment-only lines, returns the next meaningful line.
// If EOF is hit, returns an `eof`-flagged RawLine.
RawLine next_meaningful(LineReader& reader) {
  while (true) {
    auto line = reader.next();
    if (line.eof) {
      return line;
    }
    const auto stripped = trim(strip_comment(std::string_view(line.text)));
    if (!stripped.empty()) {
      return line;
    }
  }
}

// ---- Header parsing --------------------------------------------------------

struct HeaderInfo {
  std::size_t atom_count = 0;
  std::size_t atom_type_count = 0;
  Box box;
  std::optional<std::uint64_t> timestep;
};

// Generic "N <keyword>" line: counts we accept non-zero only for `atoms` and
// `atom types`. Anything else (bonds, angles, ...) must be zero in M1.
//
// Returns true iff the line was recognized as a header-count line (and
// consumed). Populates `info` accordingly.
bool try_parse_count_line(std::string_view content,
                          const std::vector<std::string_view>& tokens,
                          std::size_t line_number,
                          HeaderInfo& info,
                          bool& matched) {
  matched = false;
  if (tokens.empty()) {
    return false;
  }

  // "xlo xhi" / "ylo yhi" / "zlo zhi" have the keywords at positions [n-2, n-1].
  if (tokens.size() == 4) {
    const auto kw0 = tokens[2];
    const auto kw1 = tokens[3];
    if ((kw0 == "xlo" && kw1 == "xhi") || (kw0 == "ylo" && kw1 == "yhi") ||
        (kw0 == "zlo" && kw1 == "zhi")) {
      double lo = 0.0;
      double hi = 0.0;
      if (!parse_double(tokens[0], lo) || !parse_double(tokens[1], hi)) {
        throw_parse(line_number, std::string("malformed box bound: ") + std::string(content));
      }
      if (!(hi > lo)) {
        throw_parse(line_number, "box hi must be greater than lo");
      }
      if (kw0 == "xlo") {
        info.box.xlo = lo;
        info.box.xhi = hi;
      } else if (kw0 == "ylo") {
        info.box.ylo = lo;
        info.box.yhi = hi;
      } else {
        info.box.zlo = lo;
        info.box.zhi = hi;
      }
      matched = true;
      return true;
    }
  }

  // Triclinic tilt: "xy xz yz xy xz yz" — reject in M1.
  if (tokens.size() == 6 && tokens[3] == "xy" && tokens[4] == "xz" && tokens[5] == "yz") {
    throw_parse(line_number, "triclinic tilt factors (xy xz yz) are not supported in M1");
  }

  // "<N> atoms"
  if (tokens.size() == 2 && tokens[1] == "atoms") {
    std::int64_t n = 0;
    if (!parse_int64(tokens[0], n) || n < 0) {
      throw_parse(line_number, "atom count must be a non-negative integer");
    }
    info.atom_count = static_cast<std::size_t>(n);
    matched = true;
    return true;
  }

  // "<N> atom types"
  if (tokens.size() == 3 && tokens[1] == "atom" && tokens[2] == "types") {
    std::int64_t n = 0;
    if (!parse_int64(tokens[0], n) || n <= 0) {
      throw_parse(line_number, "atom type count must be a positive integer");
    }
    info.atom_type_count = static_cast<std::size_t>(n);
    matched = true;
    return true;
  }

  // Counts that must be zero in atom_style atomic: bonds, angles, dihedrals,
  // impropers, and their *-types counterparts, plus ellipsoids/lines/triangles/bodies.
  static const char* const kZeroOnlyKeywords2[] = {
      "bonds",
      "angles",
      "dihedrals",
      "impropers",
      "ellipsoids",
      "lines",
      "triangles",
      "bodies",
  };
  for (const char* kw : kZeroOnlyKeywords2) {
    if (tokens.size() == 2 && tokens[1] == kw) {
      std::int64_t n = 0;
      if (!parse_int64(tokens[0], n)) {
        throw_parse(line_number, std::string("malformed count for ") + kw);
      }
      if (n != 0) {
        throw_parse(
            line_number,
            std::string("non-zero '") + kw + "' count is not supported in M1 (atom_style atomic)");
      }
      matched = true;
      return true;
    }
  }

  // "<N> bond types", "<N> angle types", etc. — must be zero.
  static const char* const kZeroOnlyKeywords3[] = {"bond", "angle", "dihedral", "improper"};
  for (const char* kw : kZeroOnlyKeywords3) {
    if (tokens.size() == 3 && tokens[1] == kw && tokens[2] == "types") {
      std::int64_t n = 0;
      if (!parse_int64(tokens[0], n)) {
        throw_parse(line_number, std::string("malformed count for ") + kw + " types");
      }
      if (n != 0) {
        throw_parse(line_number,
                    std::string("non-zero '") + kw + " types' count not supported in M1");
      }
      matched = true;
      return true;
    }
  }

  return false;  // not a header-count line
}

// Known M1 section keywords. The `Atoms` keyword may appear with a style hint,
// e.g. "Atoms # atomic" — any style other than "atomic" is an error in M1.
bool is_section_header(std::string_view raw_line,
                       std::string& section_kw,
                       std::string& style_hint) {
  section_kw.clear();
  style_hint.clear();
  // Split on first '#' to separate keyword part from style hint.
  const auto hash = raw_line.find('#');
  const auto kw_part = trim(hash == std::string_view::npos ? raw_line : raw_line.substr(0, hash));
  if (kw_part.empty()) {
    return false;
  }
  // Section keyword must be one of: Masses, Atoms, Velocities, etc.
  // We recognize them by exact match; anything else is (for M1) an error.
  const auto tokens = tokenize(kw_part);
  if (tokens.size() != 1) {
    return false;  // data rows always have multiple tokens
  }
  const auto tok = std::string(tokens[0]);
  if (!(std::isupper(static_cast<unsigned char>(tok.front())))) {
    return false;  // section headers are capitalized
  }
  section_kw = tok;
  if (hash != std::string_view::npos) {
    style_hint = std::string(trim(raw_line.substr(hash + 1)));
  }
  return true;
}

// ---- Section body parsers --------------------------------------------------

struct MassEntry {
  std::uint32_t type_id = 0;  // 1-based per LAMMPS convention
  double mass = 0.0;
};

void parse_masses_section(LineReader& reader, std::size_t expected, std::vector<MassEntry>& out) {
  out.clear();
  out.reserve(expected);
  for (std::size_t i = 0; i < expected; ++i) {
    const auto line = next_meaningful(reader);
    if (line.eof) {
      throw_parse(reader.current_line(), "unexpected EOF inside Masses section (missing rows)");
    }
    const auto content = trim(strip_comment(std::string_view(line.text)));
    const auto tokens = tokenize(content);
    if (tokens.size() < 2) {
      throw_parse(line.number, "Masses row must have at least 2 tokens: <type> <mass>");
    }
    std::uint64_t type_id = 0;
    double mass = 0.0;
    if (!parse_uint64(tokens[0], type_id) || type_id == 0) {
      throw_parse(line.number, "Masses row: type id must be a positive integer");
    }
    if (!parse_double(tokens[1], mass) || !(mass > 0.0)) {
      throw_parse(line.number, "Masses row: mass must be a positive finite number");
    }
    MassEntry entry;
    entry.type_id = static_cast<std::uint32_t>(type_id);
    entry.mass = mass;
    out.push_back(entry);
  }
}

struct AtomEntry {
  std::uint64_t id = 0;  // LAMMPS 1-based ID (not preserved — we mint fresh AtomIds)
  std::uint32_t type_id = 0;
  double x = 0.0, y = 0.0, z = 0.0;
};

struct VelocityEntry {
  std::uint64_t id = 0;
  double vx = 0.0, vy = 0.0, vz = 0.0;
};

void parse_atoms_section(LineReader& reader,
                         std::size_t expected,
                         std::size_t max_type_id,
                         std::vector<AtomEntry>& out) {
  out.clear();
  out.reserve(expected);
  for (std::size_t i = 0; i < expected; ++i) {
    const auto line = next_meaningful(reader);
    if (line.eof) {
      throw_parse(reader.current_line(), "unexpected EOF inside Atoms section (missing rows)");
    }
    const auto content = trim(strip_comment(std::string_view(line.text)));
    const auto tokens = tokenize(content);
    // atom_style atomic: <atom_id> <type> <x> <y> <z> [<ix> <iy> <iz>]
    if (tokens.size() != 5 && tokens.size() != 8) {
      throw_parse(line.number,
                  "Atoms row (atom_style atomic) must have 5 or 8 tokens: "
                  "<id> <type> <x> <y> <z> [<ix> <iy> <iz>]");
    }
    AtomEntry entry;
    std::uint64_t type_id = 0;
    if (!parse_uint64(tokens[0], entry.id) || entry.id == 0) {
      throw_parse(line.number, "Atoms row: atom id must be a positive integer");
    }
    if (!parse_uint64(tokens[1], type_id) || type_id == 0 || type_id > max_type_id) {
      throw_parse(line.number, "Atoms row: type id out of range (must be 1 .. atom_types)");
    }
    entry.type_id = static_cast<std::uint32_t>(type_id);
    if (!parse_double(tokens[2], entry.x) || !parse_double(tokens[3], entry.y) ||
        !parse_double(tokens[4], entry.z)) {
      throw_parse(line.number, "Atoms row: coordinates must be finite numbers");
    }
    // Image flags (tokens 5..7) ignored in M1 per exec pack scope.
    out.push_back(entry);
  }
}

void parse_velocities_section(LineReader& reader,
                              std::size_t expected,
                              std::vector<VelocityEntry>& out) {
  out.clear();
  out.reserve(expected);
  for (std::size_t i = 0; i < expected; ++i) {
    const auto line = next_meaningful(reader);
    if (line.eof) {
      throw_parse(reader.current_line(), "unexpected EOF inside Velocities section (missing rows)");
    }
    const auto content = trim(strip_comment(std::string_view(line.text)));
    const auto tokens = tokenize(content);
    if (tokens.size() != 4) {
      throw_parse(line.number, "Velocities row must have 4 tokens: <id> <vx> <vy> <vz>");
    }
    VelocityEntry entry;
    if (!parse_uint64(tokens[0], entry.id) || entry.id == 0) {
      throw_parse(line.number, "Velocities row: atom id must be a positive integer");
    }
    if (!parse_double(tokens[1], entry.vx) || !parse_double(tokens[2], entry.vy) ||
        !parse_double(tokens[3], entry.vz)) {
      throw_parse(line.number, "Velocities row: vx/vy/vz must be finite numbers");
    }
    out.push_back(entry);
  }
}

}  // namespace

// ---- Public constructors ---------------------------------------------------

LammpsDataParseError::LammpsDataParseError(std::size_t line_number, std::string_view message)
    : std::runtime_error(build_parse_error(line_number, message)), line_number_(line_number) {}

// ---- Public entry points ---------------------------------------------------

LammpsDataImportResult read_lammps_data(std::istream& in,
                                        const LammpsDataImportOptions& options,
                                        AtomSoA& out_atoms,
                                        Box& out_box,
                                        SpeciesRegistry& out_species) {
  if (options.units != UnitSystem::Metal) {
    throw_parse(0,
                "only UnitSystem::Metal is supported in M1 (T1.3); "
                "lj/real/cgs/si are reserved for later milestones");
  }
  if (!out_atoms.empty()) {
    throw_parse(0,
                "read_lammps_data expects an empty AtomSoA on entry "
                "(importer appends rather than resizes)");
  }
  if (!out_species.empty()) {
    throw_parse(0, "read_lammps_data expects an empty SpeciesRegistry on entry");
  }

  LineReader reader(in);

  // First physical line is the title comment. Per `write_data` LAMMPS always
  // emits one; accept anything (including blank) and move on.
  {
    const auto title = reader.next();
    if (title.eof) {
      throw_parse(0, "file is empty (no title line)");
    }
  }

  // Parse header until we encounter a recognized section keyword.
  HeaderInfo header;
  std::string first_section_kw;
  std::string first_section_style;
  std::size_t first_section_line = 0;
  while (true) {
    const auto line = next_meaningful(reader);
    if (line.eof) {
      throw_parse(reader.current_line(), "unexpected EOF in header (no sections found)");
    }
    // Is this a section header (capitalized keyword, optional style hint)?
    if (is_section_header(std::string_view(line.text), first_section_kw, first_section_style)) {
      first_section_line = line.number;
      break;
    }
    // Otherwise it must be a recognized header count / box line.
    const auto content = trim(strip_comment(std::string_view(line.text)));
    const auto tokens = tokenize(content);
    bool matched = false;
    try_parse_count_line(content, tokens, line.number, header, matched);
    if (!matched) {
      std::ostringstream oss;
      oss << "unrecognized header line: '" << content << "'";
      throw_parse(line.number, oss.str());
    }
  }

  // Validate header before consuming sections.
  if (header.atom_type_count == 0) {
    throw_parse(first_section_line,
                "header did not declare 'atom types' (atom_style atomic needs ≥1)");
  }
  const double lx = header.box.xhi - header.box.xlo;
  const double ly = header.box.yhi - header.box.ylo;
  const double lz = header.box.zhi - header.box.zlo;
  if (!(lx > 0.0) || !(ly > 0.0) || !(lz > 0.0)) {
    throw_parse(first_section_line, "header did not declare a valid box (xlo/xhi, etc.)");
  }
  // PBC defaults: write_data files do not carry the `boundary` flag. LAMMPS
  // convention for `atom_style atomic` T1 benchmarks is fully periodic; caller
  // can override afterwards if needed.
  header.box.periodic_x = true;
  header.box.periodic_y = true;
  header.box.periodic_z = true;

  // Parse sections in order. We allow sections to arrive in any order except
  // that `Masses` must come before `Atoms` (so that we can populate the
  // registry first). Practically, LAMMPS `write_data` emits Masses → Atoms →
  // Velocities, which matches this order.
  std::vector<MassEntry> masses;
  std::vector<AtomEntry> atoms_data;
  std::vector<VelocityEntry> velocities;
  bool seen_masses = false;
  bool seen_atoms = false;
  bool seen_velocities = false;

  std::string section_kw = first_section_kw;
  std::string style_hint = first_section_style;
  std::size_t section_line = first_section_line;

  while (!section_kw.empty()) {
    if (section_kw == "Masses") {
      if (seen_masses) {
        throw_parse(section_line, "duplicate 'Masses' section");
      }
      parse_masses_section(reader, header.atom_type_count, masses);
      seen_masses = true;
    } else if (section_kw == "Atoms") {
      if (!seen_masses) {
        throw_parse(section_line,
                    "'Atoms' section appeared before 'Masses' (Masses must precede Atoms)");
      }
      if (!style_hint.empty() && style_hint != "atomic") {
        throw_parse(section_line,
                    std::string("unsupported atom_style '") + style_hint +
                        "' in M1 (only 'atomic' is accepted)");
      }
      parse_atoms_section(reader, header.atom_count, header.atom_type_count, atoms_data);
      seen_atoms = true;
    } else if (section_kw == "Velocities") {
      if (!seen_atoms) {
        throw_parse(section_line, "'Velocities' section appeared before 'Atoms' (out of order)");
      }
      parse_velocities_section(reader, header.atom_count, velocities);
      seen_velocities = true;
    } else {
      throw_parse(section_line,
                  std::string("unrecognized or unsupported section '") + section_kw +
                      "' in M1 (only Masses / Atoms / Velocities are supported)");
    }

    // Look for the next section header (or EOF).
    section_kw.clear();
    style_hint.clear();
    const auto next = next_meaningful(reader);
    if (next.eof) {
      break;
    }
    if (!is_section_header(std::string_view(next.text), section_kw, style_hint)) {
      throw_parse(next.number,
                  std::string("expected section header or EOF, got: '") + next.text + "'");
    }
    section_line = next.number;
  }

  if (!seen_masses) {
    throw_parse(reader.current_line(), "file did not contain a 'Masses' section");
  }
  if (!seen_atoms) {
    throw_parse(reader.current_line(), "file did not contain an 'Atoms' section");
  }

  // ---- Build SpeciesRegistry -------------------------------------------------
  for (std::size_t type_idx = 0; type_idx < header.atom_type_count; ++type_idx) {
    const std::uint32_t expected_type_id = static_cast<std::uint32_t>(type_idx + 1);
    const MassEntry* match = nullptr;
    for (const auto& m : masses) {
      if (m.type_id == expected_type_id) {
        match = &m;
        break;
      }
    }
    if (match == nullptr) {
      std::ostringstream oss;
      oss << "Masses section missing entry for type " << expected_type_id;
      throw_parse(reader.current_line(), oss.str());
    }
    SpeciesInfo info;
    if (type_idx < options.species_names.size() && !options.species_names[type_idx].empty()) {
      info.name = options.species_names[type_idx];
    } else {
      info.name = "type_" + std::to_string(expected_type_id);
    }
    info.mass = match->mass;
    info.charge = 0.0;
    if (type_idx < options.atomic_numbers.size()) {
      info.atomic_number = options.atomic_numbers[type_idx];
    }
    out_species.register_species(info);
  }

  // ---- Populate AtomSoA ------------------------------------------------------
  // LAMMPS atom IDs may appear in any order; preserve that order (as-read) so
  // `AtomSoA` indices match the file. This makes round-trip trivially stable
  // when the dumper writes atoms in SoA index order.
  out_atoms.reserve(header.atom_count);
  for (const auto& a : atoms_data) {
    // SpeciesId is the 0-based dense id; LAMMPS type is 1-based.
    const SpeciesId sid = static_cast<SpeciesId>(a.type_id - 1);
    out_atoms.add_atom(sid, a.x, a.y, a.z);
  }

  // ---- Apply velocities ------------------------------------------------------
  if (seen_velocities) {
    // Build lookup: LAMMPS atom_id → SoA index. Atom IDs may be sparse or
    // unsorted in the file; we mirror the Atoms order by constructing an
    // inverse map from the `atoms_data` vector.
    // Guard: Velocities count == atom count (already enforced by the parser),
    // but also reject missing IDs.
    std::vector<std::size_t> atom_index_by_id;  // 0-based, sized to max_id+1
    std::uint64_t max_id = 0;
    for (const auto& a : atoms_data) {
      max_id = std::max(max_id, a.id);
    }
    atom_index_by_id.assign(max_id + 1, std::numeric_limits<std::size_t>::max());
    for (std::size_t i = 0; i < atoms_data.size(); ++i) {
      atom_index_by_id[atoms_data[i].id] = i;
    }
    for (const auto& v : velocities) {
      if (v.id >= atom_index_by_id.size() ||
          atom_index_by_id[v.id] == std::numeric_limits<std::size_t>::max()) {
        std::ostringstream oss;
        oss << "Velocities entry refers to unknown atom id " << v.id;
        throw_parse(reader.current_line(), oss.str());
      }
      const auto idx = atom_index_by_id[v.id];
      out_atoms.vx[idx] = v.vx;
      out_atoms.vy[idx] = v.vy;
      out_atoms.vz[idx] = v.vz;
    }
  }

  out_box = header.box;

  LammpsDataImportResult result;
  result.timestep = header.timestep;  // populated from title-line scan in a future pass
  result.atom_count = atoms_data.size();
  result.atom_types = header.atom_type_count;
  result.has_velocities = seen_velocities;
  return result;
}

LammpsDataImportResult read_lammps_data_file(const std::string& path,
                                             const LammpsDataImportOptions& options,
                                             AtomSoA& out_atoms,
                                             Box& out_box,
                                             SpeciesRegistry& out_species) {
  std::ifstream stream(path);
  if (!stream) {
    std::ostringstream oss;
    oss << "unable to open LAMMPS data file for reading: '" << path << "'";
    throw LammpsDataParseError(0, oss.str());
  }
  return read_lammps_data(stream, options, out_atoms, out_box, out_species);
}

}  // namespace tdmd::io
