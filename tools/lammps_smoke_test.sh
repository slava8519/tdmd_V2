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
LAMMPS_INSTALL="$REPO_ROOT/verify/third_party/lammps/install_tdmd"
LMP="${LMP:-$LAMMPS_INSTALL/bin/lmp}"
INPUT="$SCRIPT_DIR/lammps_smoke_test.in"
POTENTIALS_DIR="${LAMMPS_POTENTIALS:-$REPO_ROOT/verify/third_party/lammps/potentials}"
RUN_GPU="${TDMD_LAMMPS_GPU:-on}"

# Shared-libs build: lmp links to liblammps.so.0 under $LAMMPS_INSTALL/lib.
# LAMMPS itself ships etc/profile.d/lammps.sh for this, but baking it into the
# script avoids the "source first" footgun.
export LD_LIBRARY_PATH="$LAMMPS_INSTALL/lib:${LD_LIBRARY_PATH:-}"

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

if [[ ! -f "$POTENTIALS_DIR/Al_zhou.eam.alloy" ]]; then
  echo "ERROR: Al EAM potential not found at $POTENTIALS_DIR/Al_zhou.eam.alloy" >&2
  echo "Expected under LAMMPS potentials/ dir. Check submodule integrity." >&2
  exit 2
fi

echo "=== LAMMPS smoke test ==="
echo "  binary:    $LMP"
echo "  workdir:   $WORK_DIR"
echo "  potentials: $POTENTIALS_DIR"
echo

cd "$WORK_DIR"
ln -s "$POTENTIALS_DIR/Al_zhou.eam.alloy" Al_zhou.eam.alloy

# --------- CPU run ---------
echo "[CPU] Running 100-step Al FCC EAM NVE..."
"$LMP" -log cpu.log -in "$INPUT" -var gpu off
CPU_E=$(grep -E '^\s+100\s+' cpu.log | tail -1 | awk '{print $3}')
echo "[CPU] TotEng at step 100 = $CPU_E"

# --------- GPU run ---------
if [[ "$RUN_GPU" == "on" ]]; then
  echo
  echo "[GPU] Running same trajectory with package gpu 1..."
  # Don't abort on arch mismatch — we catch it and downgrade to a skip with a
  # clear message (see below). `set -e` would otherwise kill us on non-zero exit.
  set +e
  "$LMP" -log gpu.log -in "$INPUT" -var gpu on > gpu.stdout 2>&1
  GPU_RC=$?
  set -e

  if [[ $GPU_RC -ne 0 ]]; then
    # Distinguish "binary not compatible with this GPU" (expected on
    # CUDA 12.6 + sm_120 hw build for sm_89) from a real bug.
    if grep -q "GPU library not compiled for this accelerator" gpu.stdout gpu.log 2>/dev/null; then
      echo "[GPU] SKIPPED: LAMMPS GPU binary built for an arch that is not"
      echo "              runnable on this hardware. This is expected if CUDA"
      echo "              < 12.8 forced a sm_89 fallback but the GPU is sm_120."
      echo "              Upgrade CUDA to 12.8+ and rebuild LAMMPS to enable GPU."
      echo
      echo "=== Smoke test complete (CPU-only, GPU skipped with known reason) ==="
      exit 0
    else
      echo "[GPU] FAIL (exit $GPU_RC). Last 30 lines of gpu.stdout:" >&2
      tail -30 gpu.stdout >&2
      exit $GPU_RC
    fi
  fi

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
