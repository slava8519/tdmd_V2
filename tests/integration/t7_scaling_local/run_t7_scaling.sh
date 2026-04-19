#!/usr/bin/env bash
# T7 mixed-scaling local probe — single-node strong scaling 1..N GPU.
#
# Per Option A (no public-runner GPU CI), this script lives outside CI and
# is invoked by developers pre-push. It auto-skips when no CUDA device is
# visible (matches the M6/M7 smoke pattern).
#
# Usage:
#   ./tests/integration/t7_scaling_local/run_t7_scaling.sh
#   TDMD_BIN=/path/to/tdmd ./tests/integration/t7_scaling_local/run_t7_scaling.sh
#   T7_RANKS=1,2 ./tests/integration/t7_scaling_local/run_t7_scaling.sh
#
# Exit codes:
#   0  all probes within efficiency gate
#   2  one or more probes below gate (RED)
#   3  runtime/setup error
#   77 SKIP — no CUDA device visible (matches autotools convention)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
TDMD_BIN="${TDMD_BIN:-$REPO_ROOT/build/tdmd}"
T7_RANKS="${T7_RANKS:-1,2}"  # conservative default for dev hardware (single GPU)

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[t7-scaling] SKIP — nvidia-smi not in PATH (no CUDA device visible)"
  exit 77
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "[t7-scaling] SKIP — nvidia-smi -L returned non-zero (no GPU enumerated)"
  exit 77
fi

if [[ ! -x "$TDMD_BIN" ]]; then
  echo "[t7-scaling] ERROR — tdmd binary not found at $TDMD_BIN" >&2
  echo "             set TDMD_BIN=/path/to/tdmd or build first" >&2
  exit 3
fi

cd "$REPO_ROOT"

echo "[t7-scaling] tdmd_bin=$TDMD_BIN  ranks=$T7_RANKS"
echo "[t7-scaling] gpu_visible=$(nvidia-smi -L | head -1)"

# Lazy-regen setup.data if missing — generate_setup.py handles this idempotently.
SETUP_DATA="$REPO_ROOT/verify/data/t7_mixed_scaling/setup.data"
if [[ ! -f "$SETUP_DATA" ]]; then
  echo "[t7-scaling] setup.data missing — regenerating via generate_setup.py"
  python3 verify/benchmarks/t7_mixed_scaling/generate_setup.py --out "$SETUP_DATA"
fi

python3 -m verify.harness.scaling_runner \
  --benchmark-dir verify/benchmarks/t7_mixed_scaling \
  --tdmd-bin "$TDMD_BIN" \
  --ranks "$T7_RANKS"

echo "[t7-scaling] probe complete"
