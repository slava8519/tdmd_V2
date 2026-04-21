#!/usr/bin/env bash
# Multi-rank TD SNAP scout — 2-rank K=1 (correctness baseline) + 2-rank K=4
# (K-batching probe), all three flavors. Single RTX 5080 oversubscribed —
# absolute ms/step is NOT representative of real multi-GPU; the point is to
# (a) verify the TD machinery end-to-end on SNAP fixture and (b) establish a
# reference wall-time baseline against which future ≥2-physical-GPU runs
# (T8.11 cloud-burst) can be compared.
#
# D-M7-10 analogue for SNAP: Fp64Reference K=1 P_space=2 thermo MUST byte-match
# single-rank K=1 P=1 thermo. This script diffs thermo logs; mismatch = blocker.
#
# Usage: ./quick_2rank_scout.sh
# Prereqs: mpirun on PATH; three builds at build/, build-mixed/,
#          build-mixed-snap-only/; single-rank thermo reference generated in
#          a /tmp workdir for the correctness diff.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REPO_ROOT="$(cd ../../../.. && pwd)"

declare -A BINS=(
  [fp64ref]="${REPO_ROOT}/build/src/cli/tdmd"
  [mixed_fast]="${REPO_ROOT}/build-mixed/src/cli/tdmd"
  [mixed_snap_only]="${REPO_ROOT}/build-mixed-snap-only/src/cli/tdmd"
)

for flavor in "${!BINS[@]}"; do
  if [[ ! -x "${BINS[$flavor]}" ]]; then
    echo "FATAL: binary missing for flavor=${flavor}: ${BINS[$flavor]}" >&2
    exit 2
  fi
done

if ! command -v mpirun >/dev/null 2>&1; then
  echo "FATAL: mpirun not on PATH" >&2
  exit 2
fi

WORKDIR="$(mktemp -d -t tdmd_2rank_scout.XXXXXX)"
trap 'rm -rf "${WORKDIR}"' EXIT

measure_1rank() {
  local bin=$1
  local config=$2
  { time -p "$bin" run --timing --quiet "$config" >/dev/null 2>&1; } 2>&1 \
    | awk '/^real/ {printf("%.3f", $2)}'
}

measure_2rank() {
  local bin=$1
  local config=$2
  { time -p mpirun -np 2 --oversubscribe "$bin" run --timing --quiet "$config" >/dev/null 2>&1; } 2>&1 \
    | awk '/^real/ {printf("%.3f", $2)}'
}

median3() {
  printf "%s\n%s\n%s\n" "$1" "$2" "$3" | sort -g | sed -n '2p'
}

to_ms_per_step() {
  awk -v w="$1" 'BEGIN { printf("%.1f", w * 10) }'
}

# --- Step 1: validate configs
echo "== Validate 2-rank K=1/K=4 configs =="
for config in tdmd_gpu_2rank_k1.yaml tdmd_gpu_2rank_k4.yaml; do
  if ! "${BINS[fp64ref]}" validate "$config" >/dev/null 2>&1; then
    echo "FATAL: validate failed on $config" >&2
    exit 2
  fi
  if ! mpirun -np 2 --oversubscribe "${BINS[fp64ref]}" validate "$config" >/dev/null 2>&1; then
    echo "FATAL: mpirun -np 2 validate failed on $config" >&2
    exit 2
  fi
  echo "  ${config}: validated (single + mpirun -np 2)"
done

# --- Step 2: correctness gate — Fp64Reference thermo 1-rank ≡ 2-rank K=1 P=2
echo ""
echo "== Correctness: D-M7-10 analogue for SNAP (Fp64Ref thermo diff) =="
THERMO_1R="${WORKDIR}/thermo_1rank.txt"
THERMO_2R_K1="${WORKDIR}/thermo_2rank_k1.txt"

"${BINS[fp64ref]}" run --quiet --thermo "${THERMO_1R}" tdmd_gpu_100step.yaml
mpirun -np 2 --oversubscribe "${BINS[fp64ref]}" run --quiet --thermo "${THERMO_2R_K1}" tdmd_gpu_2rank_k1.yaml

if diff -q "${THERMO_1R}" "${THERMO_2R_K1}" >/dev/null; then
  echo "  PASS: 1-rank K=1 P=1 ≡ 2-rank K=1 P_space=2 thermo (byte-exact)"
else
  echo "  FAIL: thermo differs — D-M7-10 regression on SNAP path" >&2
  echo "  diff:" >&2
  diff "${THERMO_1R}" "${THERMO_2R_K1}" | head -20 >&2
  echo "" >&2
  echo "  NOT continuing with perf measurement — investigate first." >&2
  exit 1
fi

# --- Step 3: perf baseline (2-rank K=1 — overhead of MPI+TD wrapper)
echo ""
echo "== 2-rank K=1 perf (3 runs × 100 steps, median) — 2026-04-21 =="
echo "   single-GPU oversubscribed; absolute ms/step NOT representative of multi-GPU"
for flavor in fp64ref mixed_fast mixed_snap_only; do
  echo -n "  ${flavor}: "
  t1=$(measure_2rank "${BINS[$flavor]}" tdmd_gpu_2rank_k1.yaml)
  sleep 5
  t2=$(measure_2rank "${BINS[$flavor]}" tdmd_gpu_2rank_k1.yaml)
  sleep 5
  t3=$(measure_2rank "${BINS[$flavor]}" tdmd_gpu_2rank_k1.yaml)
  med=$(median3 "$t1" "$t2" "$t3")
  per_step=$(to_ms_per_step "$med")
  echo "runs=[${t1}s, ${t2}s, ${t3}s] median=${med}s per_step=${per_step}ms"
  sleep 10
done

# --- Step 4: K-batching probe (2-rank K=4)
echo ""
echo "== 2-rank K=4 perf (3 runs × 100 steps, median) =="
echo "   K-batching amortization may be invisible on shared GPU — expected limitation"
for flavor in fp64ref mixed_fast mixed_snap_only; do
  echo -n "  ${flavor}: "
  t1=$(measure_2rank "${BINS[$flavor]}" tdmd_gpu_2rank_k4.yaml)
  sleep 5
  t2=$(measure_2rank "${BINS[$flavor]}" tdmd_gpu_2rank_k4.yaml)
  sleep 5
  t3=$(measure_2rank "${BINS[$flavor]}" tdmd_gpu_2rank_k4.yaml)
  med=$(median3 "$t1" "$t2" "$t3")
  per_step=$(to_ms_per_step "$med")
  echo "runs=[${t1}s, ${t2}s, ${t3}s] median=${med}s per_step=${per_step}ms"
  sleep 10
done

# --- Step 5: single-rank reference recap (already measured in prior scouts,
# here for side-by-side comparison in the final table)
echo ""
echo "== 1-rank K=1 P=1 perf (for reference — 3 runs × 100 steps, median) =="
for flavor in fp64ref mixed_fast mixed_snap_only; do
  echo -n "  ${flavor}: "
  t1=$(measure_1rank "${BINS[$flavor]}" tdmd_gpu_100step.yaml)
  sleep 5
  t2=$(measure_1rank "${BINS[$flavor]}" tdmd_gpu_100step.yaml)
  sleep 5
  t3=$(measure_1rank "${BINS[$flavor]}" tdmd_gpu_100step.yaml)
  med=$(median3 "$t1" "$t2" "$t3")
  per_step=$(to_ms_per_step "$med")
  echo "runs=[${t1}s, ${t2}s, ${t3}s] median=${med}s per_step=${per_step}ms"
  sleep 5
done

echo ""
echo "== Multi-rank SNAP scout complete =="
