#include "tdmd/state/species.hpp"

#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <string>

TEST_CASE("SpeciesRegistry starts empty", "[state][species]") {
  tdmd::SpeciesRegistry reg;
  REQUIRE(reg.empty());
  REQUIRE(reg.count() == 0);
}

TEST_CASE("SpeciesRegistry assigns dense ascending ids", "[state][species]") {
  tdmd::SpeciesRegistry reg;
  const auto id_al = reg.register_species({"Al", 26.9815, 0.0, 13});
  const auto id_ni = reg.register_species({"Ni", 58.6934, 0.0, 28});
  const auto id_cu = reg.register_species({"Cu", 63.546, 0.0, 29});

  REQUIRE(id_al == 0);
  REQUIRE(id_ni == 1);
  REQUIRE(id_cu == 2);
  REQUIRE(reg.count() == 3);
}

TEST_CASE("SpeciesRegistry round-trip name↔id↔info", "[state][species]") {
  tdmd::SpeciesRegistry reg;
  reg.register_species({"Al", 26.9815, 0.0, 13});
  reg.register_species({"Ni", 58.6934, 0.0, 28});

  REQUIRE(reg.id_by_name("Al") == 0);
  REQUIRE(reg.id_by_name("Ni") == 1);

  const auto& info_al = reg.get_info(reg.id_by_name("Al"));
  REQUIRE(info_al.name == "Al");
  REQUIRE(info_al.mass == 26.9815);
  REQUIRE(info_al.atomic_number == 13u);
}

TEST_CASE("SpeciesRegistry rejects duplicate name", "[state][species]") {
  tdmd::SpeciesRegistry reg;
  reg.register_species({"Al", 26.9815, 0.0, 13});
  REQUIRE_THROWS_AS(reg.register_species({"Al", 27.0, 0.0, 13}), std::invalid_argument);
}

TEST_CASE("SpeciesRegistry rejects empty name or non-positive mass", "[state][species]") {
  tdmd::SpeciesRegistry reg;
  REQUIRE_THROWS_AS(reg.register_species({"", 1.0, 0.0, 1}), std::invalid_argument);
  REQUIRE_THROWS_AS(reg.register_species({"X", 0.0, 0.0, 1}), std::invalid_argument);
  REQUIRE_THROWS_AS(reg.register_species({"X", -1.0, 0.0, 1}), std::invalid_argument);
}

TEST_CASE("SpeciesRegistry throws on unknown name / id", "[state][species]") {
  tdmd::SpeciesRegistry reg;
  reg.register_species({"Al", 26.9815, 0.0, 13});

  REQUIRE_THROWS_AS(reg.id_by_name("Unobtanium"), std::out_of_range);
  REQUIRE_FALSE(reg.find_id_by_name("Unobtanium").has_value());
  REQUIRE(reg.contains("Al"));
  REQUIRE_FALSE(reg.contains("Unobtanium"));

  REQUIRE_THROWS_AS(reg.get_info(99), std::out_of_range);
  REQUIRE_FALSE(reg.try_get_info(99).has_value());
}

TEST_CASE("SpeciesRegistry::checksum is deterministic across identical builds",
          "[state][species][reproducibility]") {
  tdmd::SpeciesRegistry a;
  tdmd::SpeciesRegistry b;
  a.register_species({"Al", 26.9815, 0.0, 13});
  a.register_species({"Ni", 58.6934, 0.0, 28});
  b.register_species({"Al", 26.9815, 0.0, 13});
  b.register_species({"Ni", 58.6934, 0.0, 28});

  REQUIRE(a.checksum() == b.checksum());
}

TEST_CASE("SpeciesRegistry::checksum changes when any field changes",
          "[state][species][reproducibility]") {
  tdmd::SpeciesRegistry base;
  base.register_species({"Al", 26.9815, 0.0, 13});

  SECTION("adding another species") {
    auto r = base;
    r.register_species({"Ni", 58.6934, 0.0, 28});
    REQUIRE(r.checksum() != base.checksum());
  }

  SECTION("different mass") {
    tdmd::SpeciesRegistry r;
    r.register_species({"Al", 27.0, 0.0, 13});
    REQUIRE(r.checksum() != base.checksum());
  }

  SECTION("different name") {
    tdmd::SpeciesRegistry r;
    r.register_species({"Aluminum", 26.9815, 0.0, 13});
    REQUIRE(r.checksum() != base.checksum());
  }

  SECTION("different atomic_number") {
    tdmd::SpeciesRegistry r;
    r.register_species({"Al", 26.9815, 0.0, 14});
    REQUIRE(r.checksum() != base.checksum());
  }
}

TEST_CASE(
    "SpeciesRegistry::checksum is insensitive to registration order only "
    "if ids match",
    "[state][species][reproducibility]") {
  // SpeciesId is dense and assigned in registration order (SPEC §5.2). The
  // checksum hashes the canonical ascending-id sequence, so reordering
  // registration DOES change the checksum — this test pins that behaviour.
  tdmd::SpeciesRegistry ab;
  ab.register_species({"Al", 26.9815, 0.0, 13});
  ab.register_species({"Ni", 58.6934, 0.0, 28});

  tdmd::SpeciesRegistry ba;
  ba.register_species({"Ni", 58.6934, 0.0, 28});
  ba.register_species({"Al", 26.9815, 0.0, 13});

  REQUIRE(ab.checksum() != ba.checksum());
}

TEST_CASE("SpeciesInfo equality", "[state][species]") {
  const tdmd::SpeciesInfo a{"Al", 26.9815, 0.0, 13};
  const tdmd::SpeciesInfo b{"Al", 26.9815, 0.0, 13};
  const tdmd::SpeciesInfo c{"Ni", 58.6934, 0.0, 28};
  REQUIRE(a == b);
  REQUIRE_FALSE(a == c);
  REQUIRE(a != c);
}
