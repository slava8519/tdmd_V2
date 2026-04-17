#!/usr/bin/env bash
# LAMMPS smoke test — verifies the oracle binary runs a short Al FCC EAM
# trajectory on both CPU and GPU and that the GPU result matches CPU within
# a tight tolerance.
#
# Usage:
#   tools/lammps_smoke_test.sh          # CPU + GPU
#   TDMD_LAMMPS_GPU=off tools/lammps_smoke_test.sh   # CPU only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LMP="${LMP:-$REPO_ROOT/verify/third_party/lammps/install_tdmd/bin/lmp}"
INPUT="$SCRIPT_DIR/lammps_smoke_test.in"
POTENTIALS_DIR="${LAMMPS_POTENTIALS:-$REPO_ROOT/verify/third_party/lammps/potentials}"
RUN_GPU="${TDMD_LAMMPS_GPU:-on}"

WORK_DIR="$(mktemp -d -t tdmd-lammps-smoke-XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

if [[ ! -x "$LMP" ]]; then
  echo "ERROR: LAMMPS binary not found at $LMP" >&2
  echo "Run: tools/build_lammps.sh first." >&2
  exit 2
fi

if [[ ! -f "$INPUT" ]]; then
  echo "ERROR: smoke test input $INPUT missing" >&2
  exit 2
fi

if [[ ! -f "$POTENTIALS_DIR/Al99.eam.alloy" ]]; then
  echo "ERROR: Al EAM potential not found at $POTENTIALS_DIR/Al99.eam.alloy" >&2
  echo "Expected under LAMMPS potentials/ dir. Check submodule integrity." >&2
  exit 2
fi

echo "=== LAMMPS smoke test ==="
echo "  binary:    $LMP"
echo "  workdir:   $WORK_DIR"
echo "  potentials: $POTENTIALS_DIR"
echo

cd "$WORK_DIR"
ln -s "$POTENTIALS_DIR/Al99.eam.alloy" Al99.eam.alloy

# --------- CPU run ---------
echo "[CPU] Running 100-step Al FCC EAM NVE..."
"$LMP" -log cpu.log -in "$INPUT" -var gpu off
CPU_E=$(grep -E '^\s+100\s+' cpu.log | tail -1 | awk '{print $3}')
echo "[CPU] TotEng at step 100 = $CPU_E"

# --------- GPU run ---------
if [[ "$RUN_GPU" == "on" ]]; then
  echo
  echo "[GPU] Running same trajectory with package gpu 1..."
  "$LMP" -log gpu.log -in "$INPUT" -var gpu on
  GPU_E=$(grep -E '^\s+100\s+' gpu.log | tail -1 | awk '{print $3}')
  echo "[GPU] TotEng at step 100 = $GPU_E"

  # Compare: expect agreement within 1e-8 relative (GPU is fp64 per GPU_PREC=double).
  python3 - <<PY
cpu = float("$CPU_E")
gpu = float("$GPU_E")
rel = abs(cpu - gpu) / max(abs(cpu), 1e-30)
tol = 1e-8
print(f"Relative |CPU - GPU| = {rel:.3e} (tol {tol:.0e})")
if rel > tol:
    import sys
    print(f"FAIL: GPU differs from CPU beyond {tol}", file=sys.stderr)
    sys.exit(1)
print("PASS: GPU matches CPU within tolerance")
PY

else
  echo "[GPU] Skipped (TDMD_LAMMPS_GPU=off)"
fi

echo
echo "=== Smoke test complete ==="
