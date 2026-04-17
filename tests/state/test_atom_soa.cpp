#include "tdmd/state/atom_soa.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("AtomSoA default-constructs and reports empty", "[state][smoke]") {
  tdmd::AtomSoA atoms;
  REQUIRE(atoms.empty());
  REQUIRE(atoms.size() == 0);
}

TEST_CASE("AtomSoA is nothrow default-constructible", "[state][smoke]") {
  STATIC_REQUIRE(std::is_nothrow_default_constructible_v<tdmd::AtomSoA>);
}
