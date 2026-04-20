# TDMD BuildFlavor application.
# Master spec §7.1 — five flavors, compile-time numeric semantics.
# Playbook §5.1 — `Fp64ReferenceBuild + Reference profile` is the bitwise
# oracle; never weaken flags "for performance".
#
# Usage in module CMakeLists.txt:
#   tdmd_apply_build_flavor(tdmd_<module>)

# ------------------------------------------------------------------------------
# Shared helpers.
# ------------------------------------------------------------------------------
function(_tdmd_define_flavor target flavor_define)
  target_compile_definitions(${target} PUBLIC ${flavor_define})
endfunction()

# Reference flavor — strict numerics, no FMA contraction, no fast-math.
function(_tdmd_apply_fp64_reference target)
  _tdmd_define_flavor(${target} TDMD_FLAVOR_FP64_REFERENCE)
  target_compile_options(
    ${target}
    PRIVATE # Host
            $<$<OR:$<COMPILE_LANGUAGE:CXX>>:
            -fno-fast-math
            -ffp-contract=off
            >
            # CUDA — disable FMA merging (nvcc default would contract a*b+c).
            $<$<COMPILE_LANGUAGE:CUDA>:
            --fmad=false
            -Xcompiler=-fno-fast-math
            -Xcompiler=-ffp-contract=off
            >)
endfunction()

# Production Fp64 — same semantics but allow FMA contraction for perf (results are numerically
# "equivalent" but not bitwise).
function(_tdmd_apply_fp64_production target)
  _tdmd_define_flavor(${target} TDMD_FLAVOR_FP64_PRODUCTION)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
                                           $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>)
endfunction()

# MixedFastBuild — Philosophy B: FP32 math pipeline + FP64 accumulators. Same host/CUDA math policy
# as Fp64ProductionBuild (FMA allowed, no fast-math); the FP32 narrowing lives inside the kernels
# themselves. Adapters dispatch to the `*Mixed` class variants at compile time via
# `TDMD_FLAVOR_MIXED_FAST` (T6.8). Acceptance: D-M6-8 thresholds (rel force ≤ 1e-6, rel PE ≤ 1e-8 vs
# Fp64Reference GPU).
function(_tdmd_apply_mixed_fast target)
  _tdmd_define_flavor(${target} TDMD_FLAVOR_MIXED_FAST)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
                                           $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>)
endfunction()

# MixedFastSnapOnlyBuild — heterogeneous precision per §D.11 formal exception (M8 T8.8 §D.17
# procedure). StateReal=double, ForceReal=float for SNAP kernels ONLY, ForceReal=double for EAM
# kernels, AccumReal=double. Motivation: SNAP ML-fit RMSE (~1e-3 eV/atom) is orders of magnitude
# above FP32 ULP (~6e-8 rel), so FP32 SNAP force is scientifically indistinguishable from FP64 at MD
# scales; EAM tabulated Horner splines fail monotonicity in FP32 per D-M6-8 empirical data so they
# stay FP64. Acceptance thresholds: D-M8-8 dense-cutoff analog (SNAP force ≤ 1e-5 rel / energy ≤
# 1e-7 rel; EAM inherits MixedFastBuild 1e-5/1e-7/5e-6 ceiling; NVE drift ≤ 1e-5 per 1000 steps).
# Implementation kernel dispatch lands at T8.9 — this flavor configures cleanly at T8.8 but does not
# yet emit heterogeneous code paths.
function(_tdmd_apply_mixed_fast_snap_only target)
  _tdmd_define_flavor(${target} TDMD_FLAVOR_MIXED_FAST_SNAP_ONLY)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
                                           $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>)
  message(
    STATUS "  [flavor] MixedFastSnapOnlyBuild on target ${target} — T8.9 kernel split pending")
endfunction()

function(_tdmd_apply_mixed_fast_aggressive target)
  _tdmd_define_flavor(${target} TDMD_FLAVOR_MIXED_FAST_AGGRESSIVE)
  message(STATUS "  [flavor stub] MixedFastAggressiveBuild on target ${target} — M2 TODO")
endfunction()

function(_tdmd_apply_fp32_experimental target)
  _tdmd_define_flavor(${target} TDMD_FLAVOR_FP32_EXPERIMENTAL)
  message(STATUS "  [flavor stub] Fp32ExperimentalBuild on target ${target} — M2 TODO")
endfunction()

# ------------------------------------------------------------------------------
# Public entry point.
# ------------------------------------------------------------------------------
function(tdmd_apply_build_flavor target)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "tdmd_apply_build_flavor: target '${target}' does not exist")
  endif()

  if(TDMD_BUILD_FLAVOR STREQUAL "Fp64ReferenceBuild")
    _tdmd_apply_fp64_reference(${target})
  elseif(TDMD_BUILD_FLAVOR STREQUAL "Fp64ProductionBuild")
    _tdmd_apply_fp64_production(${target})
  elseif(TDMD_BUILD_FLAVOR STREQUAL "MixedFastBuild")
    _tdmd_apply_mixed_fast(${target})
  elseif(TDMD_BUILD_FLAVOR STREQUAL "MixedFastSnapOnlyBuild")
    _tdmd_apply_mixed_fast_snap_only(${target})
  elseif(TDMD_BUILD_FLAVOR STREQUAL "MixedFastAggressiveBuild")
    _tdmd_apply_mixed_fast_aggressive(${target})
  elseif(TDMD_BUILD_FLAVOR STREQUAL "Fp32ExperimentalBuild")
    _tdmd_apply_fp32_experimental(${target})
  else()
    message(
      FATAL_ERROR
        "Unknown TDMD_BUILD_FLAVOR='${TDMD_BUILD_FLAVOR}'. "
        "Valid: Fp64ReferenceBuild | Fp64ProductionBuild | "
        "MixedFastBuild | MixedFastSnapOnlyBuild | MixedFastAggressiveBuild | Fp32ExperimentalBuild"
    )
  endif()

  # Every target also gets the warnings policy and the flavor name as a compile definition (useful
  # for runtime identification in telemetry).
  tdmd_apply_warnings(${target})
  target_compile_definitions(${target} PUBLIC "TDMD_BUILD_FLAVOR_NAME=\"${TDMD_BUILD_FLAVOR}\"")
endfunction()
