#include "tdmd/integrator/integrator.hpp"

#include <catch2/catch_test_macros.hpp>
#include <memory>

namespace {

// Minimal concrete derived type, just to verify the interface is usable
// and the virtual dtor works correctly. Real integrators live in M1.
class DummyIntegrator final : public tdmd::Integrator {};

}  // namespace

TEST_CASE("Integrator interface is instantiable via concrete derived", "[integrator][smoke]") {
  std::unique_ptr<tdmd::Integrator> integrator = std::make_unique<DummyIntegrator>();
  REQUIRE(integrator != nullptr);
}

TEST_CASE("Integrator is non-copyable and non-movable", "[integrator][smoke]") {
  STATIC_REQUIRE_FALSE(std::is_copy_constructible_v<tdmd::Integrator>);
  STATIC_REQUIRE_FALSE(std::is_move_constructible_v<tdmd::Integrator>);
}
