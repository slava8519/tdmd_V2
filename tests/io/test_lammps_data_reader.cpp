// SPEC: docs/specs/io/SPEC.md §2.2 (LAMMPS data import)
// Exec pack: docs/development/m1_execution_pack.md T1.3
//
// Happy-path parsing on hermetic fixtures + failure-mode checks + a ≥10³
// round-trip property test. Fixtures live at tests/io/fixtures/ and are
// LAMMPS-compatible but hand-generated — we do not require a running LAMMPS
// binary to run these tests (that integration check is T1.11).

#include "tdmd/io/lammps_data_reader.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"
#include "tdmd/state/species.hpp"
#include "tdmd/state/unit_system.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>

namespace {

// Absolute path to tests/io/fixtures, injected at compile time by CMake so the
// tests work from whatever CWD ctest picks. Fallback to a relative path lets
// a developer run the binary directly from the repo root.
#ifndef TDMD_TEST_FIXTURES_DIR
#define TDMD_TEST_FIXTURES_DIR "tests/io/fixtures"
#endif

std::filesystem::path fixture_path(const std::string& name) {
  return std::filesystem::path(TDMD_TEST_FIXTURES_DIR) / name;
}

// Minimal test-only `write_data` emitter. Produces the subset of the format
// that our parser accepts so we can do parse → write → parse round-trips.
// This is NOT the real exporter — that is T2+ scope.
std::string dump_lammps_data(const tdmd::AtomSoA& atoms,
                             const tdmd::Box& box,
                             const tdmd::SpeciesRegistry& species,
                             bool include_velocities) {
  std::ostringstream oss;
  oss << std::setprecision(std::numeric_limits<double>::max_digits10);
  oss << "LAMMPS data file via tdmd-test-writer\n\n";
  oss << atoms.size() << " atoms\n";
  oss << species.count() << " atom types\n\n";
  oss << box.xlo << ' ' << box.xhi << " xlo xhi\n";
  oss << box.ylo << ' ' << box.yhi << " ylo yhi\n";
  oss << box.zlo << ' ' << box.zhi << " zlo zhi\n\n";
  oss << "Masses\n\n";
  for (std::size_t t = 0; t < species.count(); ++t) {
    const auto& info = species.get_info(static_cast<tdmd::SpeciesId>(t));
    oss << (t + 1) << ' ' << info.mass << '\n';
  }
  oss << "\nAtoms # atomic\n\n";
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    oss << (i + 1) << ' ' << (atoms.type[i] + 1) << ' ' << atoms.x[i] << ' ' << atoms.y[i] << ' '
        << atoms.z[i] << '\n';
  }
  if (include_velocities) {
    oss << "\nVelocities\n\n";
    for (std::size_t i = 0; i < atoms.size(); ++i) {
      oss << (i + 1) << ' ' << atoms.vx[i] << ' ' << atoms.vy[i] << ' ' << atoms.vz[i] << '\n';
    }
  }
  return oss.str();
}

struct ParseSink {
  tdmd::AtomSoA atoms;
  tdmd::Box box;
  tdmd::SpeciesRegistry species;
};

tdmd::io::LammpsDataImportResult parse_from_string(
    const std::string& text,
    ParseSink& sink,
    const tdmd::io::LammpsDataImportOptions& opts = {}) {
  std::istringstream iss(text);
  return tdmd::io::read_lammps_data(iss, opts, sink.atoms, sink.box, sink.species);
}

}  // namespace

TEST_CASE("LammpsDataReader parses the hermetic Al FCC fixture", "[io][lammps][happy]") {
  ParseSink sink;
  tdmd::io::LammpsDataImportOptions opts;
  opts.species_names = {"Al"};
  opts.atomic_numbers = {13};

  const auto result = tdmd::io::read_lammps_data_file(fixture_path("al_fcc_small.data").string(),
                                                      opts,
                                                      sink.atoms,
                                                      sink.box,
                                                      sink.species);

  REQUIRE(result.atom_count == 32);
  REQUIRE(result.atom_types == 1);
  REQUIRE(result.has_velocities == true);

  REQUIRE(sink.species.count() == 1);
  REQUIRE(sink.species.get_info(0).name == "Al");
  REQUIRE(sink.species.get_info(0).mass == Catch::Approx(26.98));
  REQUIRE(sink.species.get_info(0).atomic_number == 13);

  REQUIRE(sink.box.xlo == 0.0);
  REQUIRE(sink.box.xhi == Catch::Approx(8.1));
  REQUIRE(sink.box.periodic_x == true);
  REQUIRE(sink.box.periodic_y == true);
  REQUIRE(sink.box.periodic_z == true);
  REQUIRE(sink.box.tilt_xy == 0.0);

  REQUIRE(sink.atoms.size() == 32);
  REQUIRE(sink.atoms.x[0] == Catch::Approx(0.0));
  REQUIRE(sink.atoms.y[0] == Catch::Approx(0.0));
  REQUIRE(sink.atoms.z[0] == Catch::Approx(0.0));
  REQUIRE(sink.atoms.x[1] == Catch::Approx(2.025));
  REQUIRE(sink.atoms.vx[0] == Catch::Approx(0.10));
  REQUIRE(sink.atoms.vy[0] == Catch::Approx(-0.05));
  REQUIRE(sink.atoms.vz[0] == Catch::Approx(0.02));
  REQUIRE(sink.atoms.type[31] == 0);
  REQUIRE(sink.atoms.x[31] == Catch::Approx(4.050));
}

TEST_CASE("LammpsDataReader: auto-generated species name when none provided",
          "[io][lammps][species]") {
  ParseSink sink;
  tdmd::io::LammpsDataImportOptions opts;  // no species_names
  const auto result = tdmd::io::read_lammps_data_file(fixture_path("al_fcc_small.data").string(),
                                                      opts,
                                                      sink.atoms,
                                                      sink.box,
                                                      sink.species);
  REQUIRE(result.atom_count == 32);
  REQUIRE(sink.species.get_info(0).name == "type_1");
  REQUIRE(sink.species.get_info(0).atomic_number == 0);
}

TEST_CASE("LammpsDataReader rejects empty files", "[io][lammps][error]") {
  ParseSink sink;
  REQUIRE_THROWS_AS(tdmd::io::read_lammps_data_file(fixture_path("empty.data").string(),
                                                    {},
                                                    sink.atoms,
                                                    sink.box,
                                                    sink.species),
                    tdmd::io::LammpsDataParseError);
}

TEST_CASE("LammpsDataReader rejects missing Atoms section with line context",
          "[io][lammps][error]") {
  ParseSink sink;
  try {
    tdmd::io::read_lammps_data_file(fixture_path("corrupt_missing_atoms.data").string(),
                                    {},
                                    sink.atoms,
                                    sink.box,
                                    sink.species);
    FAIL("expected LammpsDataParseError");
  } catch (const tdmd::io::LammpsDataParseError& e) {
    const std::string msg = e.what();
    REQUIRE(msg.find("Atoms") != std::string::npos);
    // The error must cite a specific line so users can locate the problem.
    REQUIRE(msg.find("line ") != std::string::npos);
  }
}

TEST_CASE("LammpsDataReader rejects triclinic box (M1 orthogonal-only)",
          "[io][lammps][error][triclinic]") {
  ParseSink sink;
  try {
    tdmd::io::read_lammps_data_file(fixture_path("triclinic_rejected.data").string(),
                                    {},
                                    sink.atoms,
                                    sink.box,
                                    sink.species);
    FAIL("expected LammpsDataParseError for triclinic input");
  } catch (const tdmd::io::LammpsDataParseError& e) {
    const std::string msg = e.what();
    REQUIRE(msg.find("triclinic") != std::string::npos);
  }
}

TEST_CASE("LammpsDataReader: unsupported atom_style hint is rejected",
          "[io][lammps][error][atom_style]") {
  const std::string input =
      "LAMMPS data file via tdmd-test-writer\n"
      "\n"
      "1 atoms\n"
      "1 atom types\n"
      "\n"
      "0.0 10.0 xlo xhi\n"
      "0.0 10.0 ylo yhi\n"
      "0.0 10.0 zlo zhi\n"
      "\n"
      "Masses\n"
      "\n"
      "1 1.0\n"
      "\n"
      "Atoms # full\n"
      "\n"
      "1 1 0.0 0.0 0.0\n";
  ParseSink sink;
  try {
    parse_from_string(input, sink);
    FAIL("expected LammpsDataParseError for atom_style full");
  } catch (const tdmd::io::LammpsDataParseError& e) {
    REQUIRE(std::string(e.what()).find("full") != std::string::npos);
  }
}

TEST_CASE("LammpsDataReader: non-zero bonds/angles are rejected", "[io][lammps][error]") {
  const std::string input =
      "LAMMPS data file via tdmd-test-writer\n"
      "\n"
      "1 atoms\n"
      "1 atom types\n"
      "2 bonds\n"
      "\n"
      "0.0 10.0 xlo xhi\n"
      "0.0 10.0 ylo yhi\n"
      "0.0 10.0 zlo zhi\n"
      "\n"
      "Masses\n"
      "\n"
      "1 1.0\n"
      "\n"
      "Atoms # atomic\n"
      "\n"
      "1 1 0.0 0.0 0.0\n";
  ParseSink sink;
  REQUIRE_THROWS_AS(parse_from_string(input, sink), tdmd::io::LammpsDataParseError);
}

TEST_CASE("LammpsDataReader: type id out of range is rejected", "[io][lammps][error]") {
  const std::string input =
      "LAMMPS data file via tdmd-test-writer\n"
      "\n"
      "1 atoms\n"
      "1 atom types\n"
      "\n"
      "0.0 10.0 xlo xhi\n"
      "0.0 10.0 ylo yhi\n"
      "0.0 10.0 zlo zhi\n"
      "\n"
      "Masses\n"
      "\n"
      "1 1.0\n"
      "\n"
      "Atoms # atomic\n"
      "\n"
      "1 5 0.0 0.0 0.0\n";  // type 5 > atom_types (1)
  ParseSink sink;
  REQUIRE_THROWS_AS(parse_from_string(input, sink), tdmd::io::LammpsDataParseError);
}

TEST_CASE("LammpsDataReader: missing atom rows is rejected", "[io][lammps][error]") {
  const std::string input =
      "LAMMPS data file via tdmd-test-writer\n"
      "\n"
      "3 atoms\n"
      "1 atom types\n"
      "\n"
      "0.0 10.0 xlo xhi\n"
      "0.0 10.0 ylo yhi\n"
      "0.0 10.0 zlo zhi\n"
      "\n"
      "Masses\n"
      "\n"
      "1 1.0\n"
      "\n"
      "Atoms # atomic\n"
      "\n"
      "1 1 0.0 0.0 0.0\n"
      "2 1 1.0 1.0 1.0\n";  // only 2 of 3
  ParseSink sink;
  REQUIRE_THROWS_AS(parse_from_string(input, sink), tdmd::io::LammpsDataParseError);
}

TEST_CASE("LammpsDataReader: Velocities before Atoms is rejected", "[io][lammps][error][order]") {
  const std::string input =
      "LAMMPS data file via tdmd-test-writer\n"
      "\n"
      "1 atoms\n"
      "1 atom types\n"
      "\n"
      "0.0 10.0 xlo xhi\n"
      "0.0 10.0 ylo yhi\n"
      "0.0 10.0 zlo zhi\n"
      "\n"
      "Masses\n"
      "\n"
      "1 1.0\n"
      "\n"
      "Velocities\n"
      "\n"
      "1 0.0 0.0 0.0\n";
  ParseSink sink;
  REQUIRE_THROWS_AS(parse_from_string(input, sink), tdmd::io::LammpsDataParseError);
}

TEST_CASE("LammpsDataReader: Metal and Lj are accepted, Real/Cgs/Si rejected",
          "[io][lammps][units]") {
  // Minimal 1-atom fixture reused across unit-system probes.
  const std::string input =
      "LAMMPS data file via tdmd-test-writer\n"
      "\n"
      "1 atoms\n"
      "1 atom types\n"
      "\n"
      "0.0 10.0 xlo xhi\n"
      "0.0 10.0 ylo yhi\n"
      "0.0 10.0 zlo zhi\n"
      "\n"
      "Masses\n"
      "\n"
      "1 1.0\n"
      "\n"
      "Atoms # atomic\n"
      "\n"
      "1 1 0.0 0.0 0.0\n";

  // Metal — accepted (M1 baseline).
  {
    ParseSink sink;
    tdmd::io::LammpsDataImportOptions opts;
    opts.units = tdmd::UnitSystem::Metal;
    REQUIRE_NOTHROW(parse_from_string(input, sink, opts));
  }

  // Lj — accepted as of M2/T2.2 (reader is unit-agnostic; conversion happens
  // in runtime/SimulationEngine at the ingest boundary).
  {
    ParseSink sink;
    tdmd::io::LammpsDataImportOptions opts;
    opts.units = tdmd::UnitSystem::Lj;
    REQUIRE_NOTHROW(parse_from_string(input, sink, opts));
  }

  // Real / Cgs / Si — still rejected (TDMD has no converter for them).
  for (auto bad : {tdmd::UnitSystem::Real, tdmd::UnitSystem::Cgs, tdmd::UnitSystem::Si}) {
    ParseSink sink;
    tdmd::io::LammpsDataImportOptions opts;
    opts.units = bad;
    REQUIRE_THROWS_AS(parse_from_string(input, sink, opts), tdmd::io::LammpsDataParseError);
  }
}

TEST_CASE("LammpsDataReader: round-trip bit-exact on random inputs (≥10³ cases)",
          "[io][lammps][roundtrip][property]") {
  // Generate 1024 independent small systems (4-atom Al clusters) with random
  // coordinates and velocities; serialize with the test writer (max_digits10
  // precision) → parse → compare fields bit-exact.
  constexpr int kCases = 1024;
  std::mt19937_64 rng(0xd00dfaceULL);
  std::uniform_real_distribution<double> coord_dist(0.01, 9.99);
  std::uniform_real_distribution<double> vel_dist(-1.5, 1.5);

  int diffs = 0;
  for (int c = 0; c < kCases; ++c) {
    tdmd::AtomSoA src_atoms;
    tdmd::Box src_box;
    src_box.xhi = 10.0;
    src_box.yhi = 10.0;
    src_box.zhi = 10.0;
    src_box.periodic_x = true;
    src_box.periodic_y = true;
    src_box.periodic_z = true;
    tdmd::SpeciesRegistry src_species;
    src_species.register_species({"Al", 26.98, 0.0, 13});

    constexpr int kAtomsPerCase = 4;
    for (int a = 0; a < kAtomsPerCase; ++a) {
      src_atoms.add_atom(0,
                         coord_dist(rng),
                         coord_dist(rng),
                         coord_dist(rng),
                         vel_dist(rng),
                         vel_dist(rng),
                         vel_dist(rng));
    }

    const std::string serialized = dump_lammps_data(src_atoms, src_box, src_species, true);

    ParseSink sink;
    tdmd::io::LammpsDataImportOptions opts;
    opts.species_names = {"Al"};
    opts.atomic_numbers = {13};
    parse_from_string(serialized, sink, opts);

    REQUIRE(sink.atoms.size() == src_atoms.size());
    for (std::size_t i = 0; i < src_atoms.size(); ++i) {
      if (sink.atoms.x[i] != src_atoms.x[i] || sink.atoms.y[i] != src_atoms.y[i] ||
          sink.atoms.z[i] != src_atoms.z[i] || sink.atoms.vx[i] != src_atoms.vx[i] ||
          sink.atoms.vy[i] != src_atoms.vy[i] || sink.atoms.vz[i] != src_atoms.vz[i]) {
        ++diffs;
      }
    }
    REQUIRE(sink.box.xhi == src_box.xhi);
    REQUIRE(sink.species.get_info(0).mass == src_species.get_info(0).mass);
  }
  REQUIRE(diffs == 0);
}

TEST_CASE("LammpsDataReader: reject non-empty AtomSoA on entry", "[io][lammps][precondition]") {
  const std::string input =
      "title\n\n1 atoms\n1 atom types\n\n0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n"
      "\nMasses\n\n1 1.0\n\nAtoms # atomic\n\n1 1 0.0 0.0 0.0\n";
  ParseSink sink;
  sink.atoms.add_atom(0, 0.0, 0.0, 0.0);
  REQUIRE_THROWS_AS(parse_from_string(input, sink), tdmd::io::LammpsDataParseError);
}
