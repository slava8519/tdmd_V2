# VerifyLab root CMake — only the Catch2-wrapped differential tests live
# inside the build graph. The Python harness (verify/t1/run_differential.py,
# verify/compare.py) and the LAMMPS oracle submodule are standalone — the
# build never touches them.
#
# Gated on TDMD_BUILD_TESTS from the root CMakeLists.

add_subdirectory(t0)
add_subdirectory(t1)
add_subdirectory(t4)
