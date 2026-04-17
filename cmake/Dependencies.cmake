# TDMD external dependencies.
# Kept narrow on purpose — dependency growth is a technical-debt signal.
# Adding a new dep requires Architect review (playbook §5.3).

include(FetchContent)

# Prefer system packages when available, falling back to FetchContent.
# FETCHCONTENT_TRY_FIND_PACKAGE_MODE=OPT_IN lets us control this per-dep.
set(FETCHCONTENT_QUIET OFF)

# ------------------------------------------------------------------------------
# Catch2 v3 — unit test framework.
# Pinned tag to avoid surprise changes.
# ------------------------------------------------------------------------------
if(TDMD_BUILD_TESTS)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.3
    GIT_SHALLOW TRUE
    SYSTEM # treat headers as system (suppresses warnings from Catch2)
  )
  FetchContent_MakeAvailable(Catch2)

  # Register Catch2's CMake helpers for auto-discovery from test binaries.
  if(TARGET Catch2::Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(Catch)
  endif()
endif()

# ------------------------------------------------------------------------------
# Future deps (uncomment when milestones require):
#   - yaml-cpp (M1 — CLI config parsing)
#   - HDF5 (M3 — trajectory output)
#   - nlohmann_json (M3 — threshold registry, manifest)
#   - fmt (M1 — structured logging; or rely on C++23 std::print once available)
#   - MPI (M5 — multi-rank comm layer)
# ------------------------------------------------------------------------------
