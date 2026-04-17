# FindLammps — locate the TDMD-built LAMMPS oracle.
#
# The build script tools/build_lammps.sh installs LAMMPS under
#   verify/third_party/lammps/install_tdmd/
# with the standard UNIX layout:
#   bin/lmp
#   include/liblammps/library.h (and headers)
#   lib/liblammps.so
#
# Populated variables:
#   Lammps_FOUND           — true iff all required bits are found
#   Lammps_EXECUTABLE      — path to the lmp binary (for diff tests)
#   Lammps_INCLUDE_DIRS    — headers for embedding (programmatic use)
#   Lammps_LIBRARIES       — shared library
#   Lammps_VERSION_TAG     — pinned tag string from .gitmodules (if present)

set(_tdmd_lammps_install "${CMAKE_SOURCE_DIR}/verify/third_party/lammps/install_tdmd")

find_program(
  Lammps_EXECUTABLE
  NAMES lmp lmp_serial lmp_mpi
  PATHS "${_tdmd_lammps_install}/bin"
  NO_DEFAULT_PATH)

find_path(
  Lammps_INCLUDE_DIR
  NAMES library.h
  PATHS "${_tdmd_lammps_install}/include" "${_tdmd_lammps_install}/include/lammps"
  NO_DEFAULT_PATH)

find_library(
  Lammps_LIBRARY
  NAMES lammps
  PATHS "${_tdmd_lammps_install}/lib" "${_tdmd_lammps_install}/lib64"
  NO_DEFAULT_PATH)

# Read pinned tag out of .gitmodules if available — informational only.
set(Lammps_VERSION_TAG "unknown")
if(EXISTS "${CMAKE_SOURCE_DIR}/verify/third_party/lammps/.git")
  execute_process(
    COMMAND git -C "${CMAKE_SOURCE_DIR}/verify/third_party/lammps" describe --tags --exact-match
    OUTPUT_VARIABLE _tag
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _tag_rc)
  if(_tag_rc EQUAL 0 AND _tag)
    set(Lammps_VERSION_TAG "${_tag}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Lammps
  REQUIRED_VARS Lammps_EXECUTABLE Lammps_INCLUDE_DIR Lammps_LIBRARY
  VERSION_VAR Lammps_VERSION_TAG)

if(Lammps_FOUND)
  set(Lammps_INCLUDE_DIRS "${Lammps_INCLUDE_DIR}")
  set(Lammps_LIBRARIES "${Lammps_LIBRARY}")

  if(NOT TARGET Lammps::lammps)
    add_library(Lammps::lammps SHARED IMPORTED)
    set_target_properties(
      Lammps::lammps
      PROPERTIES IMPORTED_LOCATION "${Lammps_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${Lammps_INCLUDE_DIR}")
  endif()
endif()

mark_as_advanced(Lammps_EXECUTABLE Lammps_INCLUDE_DIR Lammps_LIBRARY)
