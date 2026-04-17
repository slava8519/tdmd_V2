#!/usr/bin/env bash
# Batch-format all TDMD C++/CUDA source files using .clang-format.
#
# Usage:
#   tools/lint/run_clang_format.sh            # format in place
#   tools/lint/run_clang_format.sh --check    # CI mode: diff-only, non-zero on deltas
#
# Honors CLANG_FORMAT env var; falls back to `clang-format` on PATH.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

CLANG_FORMAT="${CLANG_FORMAT:-clang-format}"
if ! command -v "$CLANG_FORMAT" >/dev/null 2>&1; then
  echo "ERROR: $CLANG_FORMAT not found. Install with:" >&2
  echo "  sudo apt install clang-format-18    # then set CLANG_FORMAT=clang-format-18" >&2
  exit 2
fi

MODE="fix"
if [[ "${1:-}" == "--check" ]]; then
  MODE="check"
fi

# Include: src/, tests/, benchmarks/. Exclude: build, third-party, deps.
mapfile -t FILES < <(
  git ls-files \
    'src/**/*.cpp' 'src/**/*.cc' 'src/**/*.cu' \
    'src/**/*.h'   'src/**/*.hpp' 'src/**/*.cuh' \
    'tests/**/*.cpp' 'tests/**/*.cc' 'tests/**/*.cu' \
    'tests/**/*.h'   'tests/**/*.hpp' 'tests/**/*.cuh' \
    'benchmarks/**/*.cpp' 'benchmarks/**/*.cu' \
    'benchmarks/**/*.h'   'benchmarks/**/*.hpp' 'benchmarks/**/*.cuh'
)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No C++/CUDA files found (expected during M0 skeleton phase)."
  exit 0
fi

echo "Running $CLANG_FORMAT on ${#FILES[@]} files..."
if [[ "$MODE" == "check" ]]; then
  FAIL=0
  for f in "${FILES[@]}"; do
    if ! diff -u "$f" <("$CLANG_FORMAT" --style=file "$f") >/dev/null; then
      echo "FORMAT MISMATCH: $f"
      FAIL=1
    fi
  done
  exit "$FAIL"
else
  "$CLANG_FORMAT" -i --style=file "${FILES[@]}"
  echo "Done."
fi
