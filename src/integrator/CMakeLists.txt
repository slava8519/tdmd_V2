# tdmd::integrator — time integration (VelocityVerlet, NoseHoover*, ...).
# SPEC: docs/specs/integrator/SPEC.md

add_library(tdmd_integrator STATIC integrator.cpp velocity_verlet.cpp gpu_velocity_verlet.cpp)
add_library(tdmd::integrator ALIAS tdmd_integrator)

target_include_directories(
  tdmd_integrator PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include>)

target_link_libraries(tdmd_integrator PUBLIC tdmd::state tdmd::gpu)

target_compile_features(tdmd_integrator PUBLIC cxx_std_20)

tdmd_apply_build_flavor(tdmd_integrator)
