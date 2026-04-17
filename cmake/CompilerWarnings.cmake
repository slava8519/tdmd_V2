# TDMD compiler-warning policy.
# Applied per-target via `tdmd_apply_warnings(<target>)`.
# Master spec §15 (engineering methodology).

function(tdmd_apply_warnings target)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "tdmd_apply_warnings: target '${target}' does not exist")
  endif()

  # ----------------------------------------------------------------------------
  # GCC / Clang host C++ warnings.
  # ----------------------------------------------------------------------------
  set(TDMD_GCC_CLANG_WARNINGS
      -Wall
      -Wextra
      -Wpedantic
      -Wshadow
      -Wnon-virtual-dtor
      -Wcast-align
      -Wunused
      -Woverloaded-virtual
      -Wconversion
      -Wsign-conversion
      -Wnull-dereference
      -Wdouble-promotion
      -Wformat=2
      -Wmisleading-indentation
      -Wduplicated-cond
      -Wlogical-op
      -Wimplicit-fallthrough)

  # MSVC isn't supported (TDMD is Linux/GPU first), but if it ever is...
  set(TDMD_MSVC_WARNINGS /W4 /permissive-)

  target_compile_options(
    ${target}
    PRIVATE
      $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:${TDMD_GCC_CLANG_WARNINGS}>
      $<$<CXX_COMPILER_ID:MSVC>:${TDMD_MSVC_WARNINGS}>)

  # ----------------------------------------------------------------------------
  # CUDA — nvcc passes host flags through -Xcompiler.
  # ----------------------------------------------------------------------------
  # Fp64ReferenceBuild treats CUDA warnings as errors (oracle must be clean). Other flavors:
  # warn-only.
  if(TDMD_BUILD_FLAVOR STREQUAL "Fp64ReferenceBuild")
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)
  endif()

  # ----------------------------------------------------------------------------
  # Warnings-as-errors (opt-in; CI sets this ON).
  # ----------------------------------------------------------------------------
  if(TDMD_WARNINGS_AS_ERRORS)
    target_compile_options(
      ${target}
      PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:-Werror>
        $<$<CXX_COMPILER_ID:MSVC>:/WX>)
  endif()
endfunction()
