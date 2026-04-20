#!/usr/bin/env bash
# T8.10 scout — TDMD vs LAMMPS GPU SNAP perf on RTX 5080 (dev-only baseline).
#
# Produces a JSON artifact with wall-clock timings for:
#   * TDMD Fp64Reference            + runtime.backend=gpu (oracle path)
#   * TDMD MixedFastBuild           + runtime.backend=gpu (SNAP FP64, same as above)
#   * TDMD MixedFastSnapOnlyBuild   + runtime.backend=gpu (SNAP narrow-FP32, T8.9)
#   * LAMMPS SNAP GPU               (ML-SNAP package, FP64)
#   * LAMMPS SNAP CPU 1-rank        (reference — memory already notes 30% gap)
#
# Methodology: 4 runs per config, first discarded (warmup/JIT), median of 3.
# Sleep 30s between runs for thermal settle. Requires RTX 5080 idle at start.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(git -C . rev-parse --show-toplevel)"
SNAP_DIR="$REPO_ROOT/verify/third_party/lammps/examples/snap"
LMP_BIN="$REPO_ROOT/verify/third_party/lammps/build_tdmd/lmp"

# TDMD binaries per flavor (from existing build dirs; build-mixed-snap-only was
# the test-green build in T8.9 so we know the adapter dispatch is correct).
declare -A TDMD_BINS=(
  [fp64ref]="$REPO_ROOT/build/src/cli/tdmd"
  [mixed_fast]="$REPO_ROOT/build-mixed/src/cli/tdmd"
  [mixed_snap_only]="$REPO_ROOT/build-mixed-snap-only/src/cli/tdmd"
)

N_REPEATS=${N_REPEATS:-4}              # 1 warmup + 3 timed
COOLDOWN_S=${COOLDOWN_S:-30}
OUT_DIR="$SCRIPT_DIR/runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

log() { echo "[scout $(date +%H:%M:%S)] $*"; }

time_ms() {
  # Report user+sys wall-clock of the supplied command in milliseconds
  # via /usr/bin/time -f; falls back to $SECONDS wrap if -f unsupported.
  local start end
  start=$(date +%s.%N)
  "$@" >/dev/null 2>&1
  end=$(date +%s.%N)
  awk -v a="$start" -v b="$end" 'BEGIN { printf("%.3f", (b - a) * 1000) }'
}

run_tdmd_once() {
  local flavor=$1
  local bin="${TDMD_BINS[$flavor]}"
  local idx=$2
  local out="$OUT_DIR/tdmd_${flavor}_run${idx}"
  local t_ms
  t_ms=$( { time -p "$bin" run --timing --quiet tdmd_gpu.yaml >"$out.stdout" 2>"$out.stderr" ; } 2>&1 | awk '/^real/ {printf("%.3f", $2 * 1000)}')
  echo "$t_ms"
}

run_lammps_gpu_once() {
  local idx=$1
  local out="$OUT_DIR/lammps_gpu_run${idx}"
  local t_ms
  t_ms=$( { time -p "$LMP_BIN" -sf gpu -pk gpu 1 -var snap_dir "$SNAP_DIR" -var nsteps 1000 -in lammps_script_gpu.in >"$out.stdout" 2>"$out.stderr" ; } 2>&1 | awk '/^real/ {printf("%.3f", $2 * 1000)}')
  echo "$t_ms"
}

run_lammps_cpu_once() {
  local idx=$1
  local out="$OUT_DIR/lammps_cpu_run${idx}"
  local t_ms
  t_ms=$( { time -p "$LMP_BIN" -var snap_dir "$SNAP_DIR" -var nsteps 1000 -in lammps_script_gpu.in >"$out.stdout" 2>"$out.stderr" ; } 2>&1 | awk '/^real/ {printf("%.3f", $2 * 1000)}')
  echo "$t_ms"
}

# Pre-flight
for flavor in "${!TDMD_BINS[@]}"; do
  [[ -x "${TDMD_BINS[$flavor]}" ]] || { log "missing TDMD binary ${TDMD_BINS[$flavor]}"; exit 1; }
done
[[ -x "$LMP_BIN" ]] || { log "missing LAMMPS binary $LMP_BIN"; exit 1; }
[[ -f setup_2000.data ]] || { log "missing setup_2000.data — regen via generate_setup.py --nrep 10"; exit 1; }

log "RTX 5080 scout starting — $N_REPEATS runs per config, cooldown ${COOLDOWN_S}s"
nvidia-smi --query-gpu=name,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader | tee "$OUT_DIR/gpu_state_start.txt"

declare -A RESULTS_JSON

# TDMD (three flavors)
for flavor in fp64ref mixed_fast mixed_snap_only; do
  log "TDMD $flavor — $N_REPEATS runs"
  times=()
  for i in $(seq 1 "$N_REPEATS"); do
    t=$(run_tdmd_once "$flavor" "$i")
    log "  run $i: ${t} ms"
    times+=("$t")
    [[ $i -lt $N_REPEATS ]] && sleep "$COOLDOWN_S"
  done
  RESULTS_JSON[tdmd_${flavor}]="[${times[0]}, ${times[1]}, ${times[2]}, ${times[3]}]"
  sleep "$COOLDOWN_S"
done

# LAMMPS GPU
log "LAMMPS GPU — $N_REPEATS runs"
times=()
for i in $(seq 1 "$N_REPEATS"); do
  t=$(run_lammps_gpu_once "$i")
  log "  run $i: ${t} ms"
  times+=("$t")
  [[ $i -lt $N_REPEATS ]] && sleep "$COOLDOWN_S"
done
RESULTS_JSON[lammps_gpu]="[${times[0]}, ${times[1]}, ${times[2]}, ${times[3]}]"
sleep "$COOLDOWN_S"

# LAMMPS CPU 1-rank
log "LAMMPS CPU 1-rank — $N_REPEATS runs"
times=()
for i in $(seq 1 "$N_REPEATS"); do
  t=$(run_lammps_cpu_once "$i")
  log "  run $i: ${t} ms"
  times+=("$t")
  [[ $i -lt $N_REPEATS ]] && sleep "$COOLDOWN_S"
done
RESULTS_JSON[lammps_cpu]="[${times[0]}, ${times[1]}, ${times[2]}, ${times[3]}]"

# JSON emit
{
  echo "{"
  echo "  \"methodology\": \"$N_REPEATS runs each, first discarded (warmup), median of remaining ${COOLDOWN_S}s cooldown between; 2000-atom BCC W SNAP 1000-step NVE\","
  echo "  \"hardware\": \"$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)\","
  echo "  \"date\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  first=1
  for k in "${!RESULTS_JSON[@]}"; do
    [[ $first -eq 0 ]] && echo ","
    echo -n "  \"$k\": ${RESULTS_JSON[$k]}"
    first=0
  done
  echo ""
  echo "}"
} > "$OUT_DIR/results.json"

nvidia-smi --query-gpu=name,temperature.gpu --format=csv,noheader | tee "$OUT_DIR/gpu_state_end.txt"
log "Done. Results: $OUT_DIR/results.json"
