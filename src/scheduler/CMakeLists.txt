# tdmd::scheduler — M4 deterministic TD scheduler.
# SPEC: docs/specs/scheduler/SPEC.md (v1.0, 2026-04-16)
# Exec pack: docs/development/m4_execution_pack.md T4.2–T4.11

add_library(
  tdmd_scheduler STATIC
  scheduler.cpp
  policy.cpp
  safety_certificate.cpp
  certificate_store.cpp
  zone_state_machine.cpp
  zone_dag.cpp
  causal_wavefront_scheduler.cpp
  diagnostic_dump.cpp
  event_log.cpp)
add_library(tdmd::scheduler ALIAS tdmd_scheduler)

target_include_directories(
  tdmd_scheduler PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                        $<INSTALL_INTERFACE:include>)

# T4.5 adds tdmd::zoning — the concrete scheduler consumes ZoningPlan in its .cpp. Kept PRIVATE so
# downstream users of tdmd::scheduler that already pull in tdmd::zoning for their own reasons don't
# accumulate a double include.
target_link_libraries(tdmd_scheduler PRIVATE tdmd::zoning)

target_compile_features(tdmd_scheduler PUBLIC cxx_std_20)

tdmd_apply_build_flavor(tdmd_scheduler)
