# TDMD external dependencies.
# Kept narrow on purpose — dependency growth is a technical-debt signal.
# Adding a new dep requires Architect review (playbook §5.3).

include(FetchContent)

# Prefer system packages when available, falling back to FetchContent.
# FETCHCONTENT_TRY_FIND_PACKAGE_MODE=OPT_IN lets us control this per-dep.
set(FETCHCONTENT_QUIET OFF)

# ------------------------------------------------------------------------------
# Catch2 v3 — unit test framework. Pinned tag to avoid surprise changes.
# ------------------------------------------------------------------------------
if(TDMD_BUILD_TESTS)
  fetchcontent_declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.3
    GIT_SHALLOW TRUE
    SYSTEM # treat headers as system (suppresses warnings from Catch2)
  )
  fetchcontent_makeavailable(Catch2)

  # Register Catch2's CMake helpers for auto-discovery from test binaries.
  if(TARGET Catch2::Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(Catch)
  endif()
endif()

# ------------------------------------------------------------------------------
# yaml-cpp — tdmd.yaml parsing (M1/T1.4). Pinned tag; static-only to match the rest of TDMD's build
# shape; SYSTEM so its own headers do not trip our warning budget. Tests / install / contrib are all
# off — we only need the library.
# ------------------------------------------------------------------------------
set(YAML_BUILD_SHARED_LIBS
    OFF
    CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TESTS
    OFF
    CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB
    OFF
    CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS
    OFF
    CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL
    OFF
    CACHE BOOL "" FORCE)
set(YAML_CPP_FORMAT_SOURCE
    OFF
    CACHE BOOL "" FORCE)

fetchcontent_declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0
  GIT_SHALLOW TRUE
  SYSTEM)
fetchcontent_makeavailable(yaml-cpp)

# ------------------------------------------------------------------------------
# cxxopts — header-only CLI argument parser (M1/T1.9). Pinned tag; SYSTEM so its headers do not
# warn. Tests / examples / install are off.
# ------------------------------------------------------------------------------
set(CXXOPTS_BUILD_EXAMPLES
    OFF
    CACHE BOOL "" FORCE)
set(CXXOPTS_BUILD_TESTS
    OFF
    CACHE BOOL "" FORCE)
set(CXXOPTS_ENABLE_INSTALL
    OFF
    CACHE BOOL "" FORCE)

fetchcontent_declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG v3.2.0
  GIT_SHALLOW TRUE
  SYSTEM)
fetchcontent_makeavailable(cxxopts)

# ------------------------------------------------------------------------------
# Future deps (uncomment when milestones require): - HDF5 (M3 — trajectory output) - nlohmann_json
# (M3 — threshold registry, manifest) - fmt (M1 — structured logging; or rely on C++23 std::print
# once available) - MPI (M5 — multi-rank comm layer)
# ------------------------------------------------------------------------------
