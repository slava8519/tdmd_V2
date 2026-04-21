#!/usr/bin/env bash
# 1000-step 3-run median TDMD-only scout for T8.6c-v4 comparison against
# v3 baseline in RESULTS.md (292.5 / 208.6 / 209.2 ms/step).
# Fewer runs than run_scout.sh (no warmup discard, no LAMMPS re-measure)
# because LAMMPS GPU baseline is stable.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

declare -A BINS=(
  [fp64ref]=/home/slava8519/tdmd_V2/build/src/cli/tdmd
  [mixed_fast]=/home/slava8519/tdmd_V2/build-mixed/src/cli/tdmd
  [mixed_snap_only]=/home/slava8519/tdmd_V2/build-mixed-snap-only/src/cli/tdmd
)

measure() {
  local bin=$1
  { time -p "$bin" run --timing --quiet tdmd_gpu.yaml >/dev/null 2>&1; } 2>&1 \
    | awk '/^real/ {printf("%.3f", $2)}'
}

median3() {
  printf "%s\n%s\n%s\n" "$1" "$2" "$3" | sort -g | sed -n '2p'
}

echo "== T8.6c-v4 1000-step scout (3 runs, median) — 2026-04-20 =="
for flavor in fp64ref mixed_fast mixed_snap_only; do
  echo -n "$flavor: "
  t1=$(measure "${BINS[$flavor]}")
  sleep 20
  t2=$(measure "${BINS[$flavor]}")
  sleep 20
  t3=$(measure "${BINS[$flavor]}")
  med=$(median3 "$t1" "$t2" "$t3")
  per_step=$(awk -v w="$med" 'BEGIN { printf("%.1f", w) }')
  echo "runs=[${t1}s, ${t2}s, ${t3}s] median=${med}s per_step=${per_step}ms"
  sleep 30
done
