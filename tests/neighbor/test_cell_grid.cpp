#include "tdmd/neighbor/cell_grid.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("CellGrid default-constructs as empty", "[neighbor][smoke]") {
  tdmd::CellGrid grid;
  REQUIRE(grid.empty());
  REQUIRE(grid.cell_count() == 0);
}

TEST_CASE("CellGrid is nothrow default-constructible", "[neighbor][smoke]") {
  STATIC_REQUIRE(std::is_nothrow_default_constructible_v<tdmd::CellGrid>);
}
