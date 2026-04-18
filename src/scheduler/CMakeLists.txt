# tdmd::scheduler — M4 deterministic TD scheduler.
# SPEC: docs/specs/scheduler/SPEC.md (v1.0, 2026-04-16)
# Exec pack: docs/development/m4_execution_pack.md T4.2–T4.11
#
# T4.2 only lands types + abstract interface + PolicyFactory::for_reference.
# Concrete CausalWavefrontScheduler and state machine arrive in T4.3+.

add_library(tdmd_scheduler STATIC scheduler.cpp policy.cpp)
add_library(tdmd::scheduler ALIAS tdmd_scheduler)

target_include_directories(
  tdmd_scheduler PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                        $<INSTALL_INTERFACE:include>)

# No tdmd::state / tdmd::zoning link dep at T4.2: scheduler types are standalone (ZoningPlan is
# forward-declared in td_scheduler.hpp and only bound in T4.5 where the concrete scheduler pulls it
# in via .cpp).
target_compile_features(tdmd_scheduler PUBLIC cxx_std_20)

tdmd_apply_build_flavor(tdmd_scheduler)
