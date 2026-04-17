#!/usr/bin/env bash
# Build LAMMPS (the TDMD scientific oracle) with GPU + required packages.
#
# The submodule lives in verify/third_party/lammps/ pinned at a specific
# stable tag. See verify/third_party/lammps_README.md for tag rationale.
#
# Usage:
#   tools/build_lammps.sh                              # default sm_120 + full packages
#   TDMD_LAMMPS_CUDA_ARCH=sm_89 tools/build_lammps.sh  # CUDA 12.6 fallback
#   tools/build_lammps.sh -D PKG_EXTRA_FIX=on          # extra CMake flags pass through
#
# Outputs:
#   verify/third_party/lammps/build_tdmd/   (build tree; gitignored)
#   verify/third_party/lammps/install_tdmd/ (install prefix; gitignored)
#     bin/lmp — the oracle binary used by differential tests.
#
# Build time: ~15-30 min on a 16-thread machine (first build).
# Incremental rebuilds (after package flag tweaks): 2-5 min.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAMMPS_DIR="$REPO_ROOT/verify/third_party/lammps"
BUILD_DIR="$LAMMPS_DIR/build_tdmd"
INSTALL_DIR="$LAMMPS_DIR/install_tdmd"

CUDA_ARCH="${TDMD_LAMMPS_CUDA_ARCH:-sm_120}"
JOBS="${TDMD_LAMMPS_JOBS:-$(nproc)}"

if [[ ! -d "$LAMMPS_DIR/cmake" ]]; then
  echo "ERROR: LAMMPS submodule not initialized at $LAMMPS_DIR" >&2
  echo "Run: git submodule update --init --depth 1 verify/third_party/lammps" >&2
  exit 2
fi

if [[ "$CUDA_ARCH" == "sm_120" ]]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found in PATH. Install CUDA 12.8+ or set TDMD_LAMMPS_CUDA_ARCH=sm_89." >&2
    exit 2
  fi
  NVCC_VER="$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
  if [[ "$(printf '%s\n12.8' "$NVCC_VER" | sort -V | head -1)" != "12.8" ]]; then
    echo "ERROR: sm_120 needs CUDA 12.8+; detected $NVCC_VER." >&2
    echo "Either upgrade CUDA or run: TDMD_LAMMPS_CUDA_ARCH=sm_89 $0" >&2
    exit 2
  fi
fi

echo "=== Building LAMMPS oracle ==="
echo "  source:  $LAMMPS_DIR"
echo "  build:   $BUILD_DIR"
echo "  install: $INSTALL_DIR"
echo "  GPU_ARCH: $CUDA_ARCH"
echo "  jobs:    $JOBS"
echo

cmake -B "$BUILD_DIR" -S "$LAMMPS_DIR/cmake" \
    -G Ninja \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -D BUILD_MPI=on \
    -D BUILD_OMP=on \
    -D BUILD_SHARED_LIBS=on \
    -D LAMMPS_EXCEPTIONS=on \
    -D PKG_GPU=on \
    -D GPU_API=cuda \
    -D GPU_ARCH="$CUDA_ARCH" \
    -D GPU_PREC=double \
    -D PKG_MANYBODY=on \
    -D PKG_MEAM=on \
    -D PKG_ML-SNAP=on \
    -D PKG_MOLECULE=on \
    -D PKG_KSPACE=on \
    -D PKG_EXTRA-PAIR=on \
    -D PKG_EXTRA-DUMP=on \
    "$@"

cmake --build "$BUILD_DIR" --parallel "$JOBS"
cmake --install "$BUILD_DIR"

echo
echo "=== LAMMPS build complete ==="
echo "Binary:   $INSTALL_DIR/bin/lmp"
echo "Verify:   $INSTALL_DIR/bin/lmp -h | grep -E 'GPU|MANYBODY|MEAM|ML-SNAP'"
echo
echo "Next: tools/lammps_smoke_test.sh"
