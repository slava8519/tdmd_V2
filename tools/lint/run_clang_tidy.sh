#!/usr/bin/env bash
# Batch-lint all TDMD C++ source files using .clang-tidy.
# Requires compile_commands.json from a prior CMake configure step.
#
# Usage:
#   cmake -B build --preset default      # produces build/compile_commands.json
#   tools/lint/run_clang_tidy.sh         # lint all src/
#   tools/lint/run_clang_tidy.sh --fix   # apply suggested fixes
#
# Honors CLANG_TIDY env var; BUILD_DIR defaults to build/.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

CLANG_TIDY="${CLANG_TIDY:-clang-tidy}"
BUILD_DIR="${BUILD_DIR:-build}"

if ! command -v "$CLANG_TIDY" >/dev/null 2>&1; then
  echo "ERROR: $CLANG_TIDY not found. Install with:" >&2
  echo "  sudo apt install clang-tidy-18    # then set CLANG_TIDY=clang-tidy-18" >&2
  exit 2
fi

if [[ ! -f "$BUILD_DIR/compile_commands.json" ]]; then
  echo "ERROR: $BUILD_DIR/compile_commands.json missing." >&2
  echo "Run: cmake -B $BUILD_DIR --preset default" >&2
  exit 2
fi

FIX_FLAG=""
if [[ "${1:-}" == "--fix" ]]; then
  FIX_FLAG="--fix"
fi

# Only lint our own source (src/, tests/). Never third-party.
mapfile -t FILES < <(
  git ls-files \
    'src/**/*.cpp' 'src/**/*.cc' 'src/**/*.h' 'src/**/*.hpp' \
    'tests/**/*.cpp' 'tests/**/*.cc' 'tests/**/*.h' 'tests/**/*.hpp'
)
# NOTE: .cu files are excluded — clang-tidy's CUDA support requires a clang
# CUDA install (not nvcc). Revisit in M3 when we have real CUDA code to lint.

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No C++ files found (expected during M0 skeleton phase)."
  exit 0
fi

echo "Running $CLANG_TIDY on ${#FILES[@]} files using $BUILD_DIR/compile_commands.json..."
FAIL=0
for f in "${FILES[@]}"; do
  if ! "$CLANG_TIDY" $FIX_FLAG -p "$BUILD_DIR" "$f"; then
    FAIL=1
  fi
done

exit "$FAIL"
