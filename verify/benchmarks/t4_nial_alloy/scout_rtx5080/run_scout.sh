#!/usr/bin/env bash
# EAM Ni-Al scout — TDMD vs LAMMPS KOKKOS on RTX 5080.
# Runs 3× 100-step per config, reports median. Fixture: committed 864-atom
# Ni-Al FCC (6×6×6), bit-identical to the differential harness input.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REPO_ROOT=/home/slava8519/tdmd_V2
EAM_FILE="$REPO_ROOT/verify/third_party/potentials/NiAl_Mishin_2004.eam.alloy"
SETUP_DATA="$(realpath ../setup.data)"
LMP_KK="$REPO_ROOT/verify/third_party/lammps/build_kokkos_cuda/lmp"
LMP_CPU="$REPO_ROOT/verify/third_party/lammps/build_tdmd/lmp"

declare -A TDMD_BINS=(
  [fp64ref]="$REPO_ROOT/build/src/cli/tdmd"
  [mixed_fast]="$REPO_ROOT/build-mixed/src/cli/tdmd"
)

measure_tdmd() {
  local bin=$1
  { time -p "$bin" run --timing --quiet tdmd_gpu_100step.yaml >/dev/null 2>&1; } 2>&1 \
    | awk '/^real/ {printf("%.3f", $2)}'
}

measure_lammps_kokkos() {
  { time -p "$LMP_KK" -k on g 1 -sf kk -pk kokkos newton on neigh half \
      -var setup_data "$SETUP_DATA" -var eam_file "$EAM_FILE" -var nsteps 100 \
      -in lammps_script_eam.in >/dev/null 2>&1; } 2>&1 \
    | awk '/^real/ {printf("%.3f", $2)}'
}

measure_lammps_cpu() {
  { time -p "$LMP_CPU" -var setup_data "$SETUP_DATA" -var eam_file "$EAM_FILE" \
      -var nsteps 100 -in lammps_script_eam.in >/dev/null 2>&1; } 2>&1 \
    | awk '/^real/ {printf("%.3f", $2)}'
}

median3() { printf "%s\n%s\n%s\n" "$1" "$2" "$3" | sort -g | sed -n '2p'; }

echo "== T4 Ni-Al EAM/alloy scout, 864 atoms, 100 steps, median of 3 =="

for flavor in fp64ref mixed_fast; do
  echo -n "TDMD $flavor: "
  t1=$(measure_tdmd "${TDMD_BINS[$flavor]}"); sleep 8
  t2=$(measure_tdmd "${TDMD_BINS[$flavor]}"); sleep 8
  t3=$(measure_tdmd "${TDMD_BINS[$flavor]}")
  med=$(median3 "$t1" "$t2" "$t3")
  echo "runs=[${t1}s, ${t2}s, ${t3}s] median=${med}s"
  sleep 10
done

echo -n "LAMMPS KOKKOS eam/alloy/kk GPU: "
t1=$(measure_lammps_kokkos); sleep 8
t2=$(measure_lammps_kokkos); sleep 8
t3=$(measure_lammps_kokkos)
med=$(median3 "$t1" "$t2" "$t3")
echo "runs=[${t1}s, ${t2}s, ${t3}s] median=${med}s"
sleep 10

echo -n "LAMMPS eam/alloy CPU 1-rank: "
t1=$(measure_lammps_cpu); sleep 8
t2=$(measure_lammps_cpu); sleep 8
t3=$(measure_lammps_cpu)
med=$(median3 "$t1" "$t2" "$t3")
echo "runs=[${t1}s, ${t2}s, ${t3}s] median=${med}s"
