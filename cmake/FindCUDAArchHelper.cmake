# TDMD CUDA arch resolver.
# Resolves TDMD_CUDA_ARCHS into a CMAKE_CUDA_ARCHITECTURES-compatible list.
#
# Accepted inputs:
#   "native"             — auto-detect via nvidia-smi (not reproducible; warn)
#   "120"                — single arch
#   "89;120"             — multi-arch (fat binary)
#
# Usage:
#   include(cmake/FindCUDAArchHelper.cmake)
#   tdmd_resolve_cuda_archs(OUT_VAR "${TDMD_CUDA_ARCHS}")

function(tdmd_resolve_cuda_archs out_var input)
  if(input STREQUAL "native")
    # Query nvidia-smi for the first visible device's compute cap.
    find_program(NVIDIA_SMI nvidia-smi)
    if(NOT NVIDIA_SMI)
      message(FATAL_ERROR "TDMD_CUDA_ARCHS=native requested but nvidia-smi not found. "
                          "Set TDMD_CUDA_ARCHS to an explicit value (e.g. 120).")
    endif()

    execute_process(
      COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
      OUTPUT_VARIABLE _compute_cap_raw
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _smi_rc)

    if(NOT _smi_rc EQUAL 0)
      message(FATAL_ERROR "nvidia-smi query failed (rc=${_smi_rc})")
    endif()

    # Take first line (first GPU), strip the dot ("12.0" -> "120").
    string(REGEX MATCH "^[0-9]+\\.[0-9]+" _first_cap "${_compute_cap_raw}")
    if(NOT _first_cap)
      message(FATAL_ERROR "Could not parse compute_cap from nvidia-smi output:\n"
                          "${_compute_cap_raw}")
    endif()
    string(REPLACE "." "" _arch "${_first_cap}")

    message(WARNING "TDMD_CUDA_ARCHS=native resolved to ${_arch} from nvidia-smi — "
                    "this build is NOT reproducible on machines with different GPUs. "
                    "Use an explicit arch list in CI.")

    set(${out_var}
        "${_arch}"
        PARENT_SCOPE)
  else()
    # Validate each entry is a 2-3 digit number.
    foreach(_arch IN LISTS input)
      if(NOT _arch MATCHES "^[0-9]+$")
        message(FATAL_ERROR "TDMD_CUDA_ARCHS entry '${_arch}' is not numeric")
      endif()
    endforeach()
    set(${out_var}
        "${input}"
        PARENT_SCOPE)
  endif()
endfunction()
