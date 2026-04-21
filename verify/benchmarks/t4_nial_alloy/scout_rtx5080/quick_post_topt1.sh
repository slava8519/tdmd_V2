#!/usr/bin/env bash
# T4 Ni-Al EAM/alloy quick post-T-opt-1 scout — TDMD only (3 flavors × 3 runs).
# LAMMPS KOKKOS baseline unchanged, reference from pre-opt run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REPO_ROOT=/home/slava8519/tdmd_V2

declare -A TDMD_BINS=(
  [fp64ref]="$REPO_ROOT/build/src/cli/tdmd"
  [mixed_fast]="$REPO_ROOT/build-mixed/src/cli/tdmd"
)

measure_tdmd() {
  local bin=$1
  local cfg=$2
  { time -p "$bin" run --timing --quiet "$cfg" >/dev/null 2>&1; } 2>&1 \
    | awk '/^real/ {printf("%.3f", $2)}'
}

median3() { printf "%s\n%s\n%s\n" "$1" "$2" "$3" | sort -g | sed -n '2p'; }

echo "== T4 post-T-opt-1 quick scout, 864 atoms, median of 3 =="

for cfg in tdmd_gpu_100step.yaml tdmd_gpu_1000step.yaml; do
  n_steps=$(awk '/n_steps:/ {print $2}' "$cfg")
  echo "-- $cfg (n_steps=$n_steps) --"
  for flavor in fp64ref mixed_fast; do
    echo -n "TDMD $flavor: "
    t1=$(measure_tdmd "${TDMD_BINS[$flavor]}" "$cfg"); sleep 6
    t2=$(measure_tdmd "${TDMD_BINS[$flavor]}" "$cfg"); sleep 6
    t3=$(measure_tdmd "${TDMD_BINS[$flavor]}" "$cfg")
    med=$(median3 "$t1" "$t2" "$t3")
    echo "runs=[${t1}s, ${t2}s, ${t3}s] median=${med}s"
    sleep 8
  done
done
