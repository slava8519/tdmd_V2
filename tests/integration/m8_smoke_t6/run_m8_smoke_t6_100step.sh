#!/usr/bin/env bash
#
# M8 T6 integration smoke — 100-step variant (authoritative D-M8-8 probe).
#
# Companion to `run_m8_smoke_t6.sh` (10-step short-run smoke): this script
# reads the AUTHORITATIVE NVE-drift threshold registered in
# `verify/thresholds/thresholds.yaml` directly —
# `gpu_mixed_fast_snap_only.nve_drift_per_100_steps = 3e-6` (v1.0.1,
# 2026-04-21 T8.13 Option A rescope). No scaling law applied; the gate is
# the gate the threshold registry publishes.
#
# Why this variant exists.
#   The 10-step smoke uses a √-scaled derivative of this gate (1e-6) that
#   sacrifices regression sensitivity on short runs inside the FP32 round-
#   off cloud. The 100-step probe measures the primary gate and therefore:
#   (a) catches regressions the 10-step variant would miss;
#   (b) demonstrates the T6 fixture's stability envelope empirically
#       (matches upstream Wood & Thompson 2017 reference-run length);
#   (c) closes T8.12 follow-up #3 ("add a 100-step variant of m8_smoke_t6").
#
# Flow:
#   1. Submodule probe — self-skip (exit 77) if LAMMPS submodule not
#      initialized (W_2940_2017_2.snap* fixture files absent).
#   2. LFS probe — fail (exit 2) if setup_1024.data is an unresolved LFS
#      pointer.
#   3. GPU probe — self-skip (exit 0) if nvidia-smi sees no CUDA device.
#   4. Instantiate config from benchmark template; rewrite n_steps to 100.
#   5. `tdmd validate <config>` — preflight.
#   6. `tdmd run --thermo <config>` — 100-step NVE.
#   7. Parse thermo `etotal` column; gate |E(step=100) − E(step=0)| / |E(0)|
#      against the authoritative 3e-6 threshold.
#
# Exec pack: docs/development/m8_execution_pack.md §T8.12 follow-up #3.
# Spec:      master spec §14 M8; verify/SPEC §4.7 T6 canonical fixture.
# Decision:  D-M8-8 (NVE drift envelope, v1.0.1 rescope 2026-04-21).
#
# Gate derivation.
#   The threshold is registered verbatim at
#   `verify/thresholds/thresholds.yaml`:
#     benchmarks.gpu_mixed_fast_snap_only.nve_drift_per_100_steps: 3.0e-6
#   Horizon chosen to stay inside the pure-SNAP stability envelope of the
#   T6 canonical fixture (LAMMPS itself diverges over 1000 steps under
#   Langevin NVT at the same config — see rationale doc §4.1 +
#   thresholds.yaml v1.0.1 header note). Bring-up (2026-04-21, RTX 5080):
#   Fp64Reference 5.80e-7 (~5× headroom), MixedFastSnapOnly 7.55e-7
#   (~4× headroom). Strong regression sensitivity without flapping.
#
# Flags / env:
#   --tdmd <path>                       Path to tdmd binary. Requires
#                                       -DTDMD_BUILD_CUDA=ON.
#                                       $TDMD_CLI_BIN also honored.
#   --keep-workdir                      Don't rm the tmp workdir on success.
#   TDMD_M8_SMOKE_100STEP_BUDGET_SEC    Wall-time budget (default 600s —
#                                       100 × 10-step smoke on RTX 5080
#                                       measures ~8-12 s; wide margin for
#                                       slower dev hardware).
#
# Exit codes:
#   0   green, or SKIPPED (missing submodule / missing GPU).
#   1   physics regression — energy drift exceeds 3e-6 gate.
#   2   infra — missing binary, LFS pointer unresolved, malformed output.
#   3   perf — smoke exceeded wall-time budget.
#  77   standardised skip — submodule not initialized.

set -euo pipefail

SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SMOKE_DIR}/../../.." && pwd)"

BENCH_DIR="${REPO_ROOT}/verify/benchmarks/t6_snap_tungsten"
TEMPLATE="${BENCH_DIR}/config.yaml.template"
ATOMS="${BENCH_DIR}/setup_1024.data"

SNAP_DIR="${REPO_ROOT}/verify/third_party/lammps/examples/snap"
SNAP_COEFF="${SNAP_DIR}/W_2940_2017_2.snapcoeff"
SNAP_PARAM="${SNAP_DIR}/W_2940_2017_2.snapparam"

BUDGET_SEC="${TDMD_M8_SMOKE_100STEP_BUDGET_SEC:-600}"
# Authoritative D-M8-8 NVE-drift threshold (v1.0.1 rescope, 2026-04-21).
# Reads directly from the threshold registry — no scaling law applied.
ENERGY_REL_GATE="3.0e-6"
N_STEPS=100

TDMD_BIN="${TDMD_CLI_BIN:-}"
KEEP_WORKDIR=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tdmd)
      TDMD_BIN="$2"
      shift 2
      ;;
    --keep-workdir)
      KEEP_WORKDIR=1
      shift
      ;;
    -h|--help)
      sed -n '3,70p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "[m8-smoke-100] error: unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

for f in "${SNAP_COEFF}" "${SNAP_PARAM}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[m8-smoke-100] SKIPPED — LAMMPS submodule not initialized (fixture missing)" >&2
    echo "[m8-smoke-100]   missing: ${f}" >&2
    echo "[m8-smoke-100]   run 'git submodule update --init --recursive' to fetch." >&2
    exit 77
  fi
done

if [[ ! -f "${ATOMS}" ]]; then
  echo "[m8-smoke-100] error: fixture missing: ${ATOMS}" >&2
  echo "[m8-smoke-100]   run 'cd verify/benchmarks/t6_snap_tungsten &&" >&2
  echo "[m8-smoke-100]       python3 generate_setup.py --nrep 8 --out setup_1024.data'" >&2
  exit 2
fi
if [[ "$(head -c 10 "${ATOMS}")" == "version ht" ]]; then
  echo "[m8-smoke-100] error: ${ATOMS} is an unresolved LFS pointer." >&2
  echo "[m8-smoke-100]   run 'git lfs pull'." >&2
  exit 2
fi

if [[ ! -f "${TEMPLATE}" ]]; then
  echo "[m8-smoke-100] error: config template missing: ${TEMPLATE}" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[m8-smoke-100] SKIPPED — nvidia-smi not on \$PATH (no GPU visible)"
  exit 0
fi
if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
  echo "[m8-smoke-100] SKIPPED — nvidia-smi reports no CUDA devices"
  exit 0
fi

if [[ -z "${TDMD_BIN}" ]]; then
  if command -v tdmd >/dev/null 2>&1; then
    TDMD_BIN="$(command -v tdmd)"
  else
    echo "[m8-smoke-100] error: tdmd binary not found" >&2
    echo "[m8-smoke-100]   pass --tdmd <path> or set TDMD_CLI_BIN=<path>." >&2
    exit 2
  fi
fi
if [[ ! -x "${TDMD_BIN}" ]]; then
  echo "[m8-smoke-100] error: tdmd binary not executable: ${TDMD_BIN}" >&2
  exit 2
fi

WORKDIR="$(mktemp -d -t tdmd_m8_smoke_100.XXXXXX)"
cleanup() {
  if [[ "${KEEP_WORKDIR}" -eq 0 ]]; then
    rm -rf "${WORKDIR}"
  else
    echo "[m8-smoke-100] workdir preserved: ${WORKDIR}"
  fi
}
trap cleanup EXIT

CONFIG="${WORKDIR}/config.yaml"
THERMO="${WORKDIR}/thermo.log"
RUN_ERR="${WORKDIR}/run.stderr"

# Substitute fixture paths, then override n_steps from the default 10 → 100.
# The template pins `n_steps: 10` at the short-run smoke's scale; this script
# rewrites that single line so the physical fixture + potential are untouched.
sed -e "s|{{ATOMS_PATH}}|${ATOMS}|g" \
    -e "s|{{SNAP_COEFF}}|${SNAP_COEFF}|g" \
    -e "s|{{SNAP_PARAM}}|${SNAP_PARAM}|g" \
    -e "s|^  n_steps: 10$|  n_steps: ${N_STEPS}|" \
    "${TEMPLATE}" > "${CONFIG}"

# Sanity — confirm the n_steps rewrite landed.
if ! grep -q "^  n_steps: ${N_STEPS}$" "${CONFIG}"; then
  echo "[m8-smoke-100] error: n_steps rewrite failed — config left unmodified" >&2
  grep -n n_steps "${CONFIG}" >&2 || true
  exit 2
fi

echo "[m8-smoke-100] tdmd binary: ${TDMD_BIN}"
echo "[m8-smoke-100] workdir:     ${WORKDIR}"
echo "[m8-smoke-100] budget:      ${BUDGET_SEC}s"
echo "[m8-smoke-100] n_steps:     ${N_STEPS}"
nvidia-smi -L 2>/dev/null | head -1 | sed 's/^/[m8-smoke-100] gpu:         /'

SECONDS=0

echo "[m8-smoke-100] step 1/3: tdmd validate"
if ! "${TDMD_BIN}" validate "${CONFIG}" >/dev/null 2>&1; then
  echo "[m8-smoke-100] FAIL (infra): tdmd validate exited non-zero." >&2
  echo "[m8-smoke-100]   re-run: ${TDMD_BIN} validate ${CONFIG}" >&2
  exit 2
fi

echo "[m8-smoke-100] step 2/3: tdmd run (${N_STEPS}-step NVE, 1024 atoms)"
if ! "${TDMD_BIN}" run --quiet --thermo "${THERMO}" "${CONFIG}" 2> "${RUN_ERR}"; then
  echo "[m8-smoke-100] FAIL (infra): tdmd run exited non-zero." >&2
  echo "[m8-smoke-100]   stderr excerpt:" >&2
  tail -30 "${RUN_ERR}" >&2 || true
  exit 2
fi

echo "[m8-smoke-100] step 3/3: energy-conservation gate (|ΔE|/|E₀| ≤ ${ENERGY_REL_GATE})"

if ! grep -q '^# step temp pe ke etotal press' "${THERMO}"; then
  echo "[m8-smoke-100] FAIL (infra): thermo header missing / mismatched." >&2
  head -3 "${THERMO}" >&2 || true
  exit 2
fi

expected_rows=$((N_STEPS + 1))
data_rows="$(grep -cv '^#' "${THERMO}" || true)"
if [[ "${data_rows}" -ne "${expected_rows}" ]]; then
  echo "[m8-smoke-100] FAIL (infra): expected ${expected_rows} thermo data rows (steps 0..${N_STEPS}), got ${data_rows}" >&2
  head -20 "${THERMO}" >&2 || true
  exit 2
fi

etotal_0="$(awk '!/^#/ {print $5; exit}' "${THERMO}")"
etotal_n="$(awk '!/^#/ {val=$5} END {print val}' "${THERMO}")"

if [[ -z "${etotal_0}" || -z "${etotal_n}" ]]; then
  echo "[m8-smoke-100] FAIL (infra): could not extract etotal from thermo." >&2
  exit 2
fi

rel_drift="$(awk -v e0="${etotal_0}" -v en="${etotal_n}" \
  'BEGIN { d = (en - e0); if (d < 0) d = -d; denom = e0; if (denom < 0) denom = -denom; printf("%.6e", d/denom) }')"

echo "[m8-smoke-100]   etotal(t=0)   = ${etotal_0}"
echo "[m8-smoke-100]   etotal(t=${N_STEPS}) = ${etotal_n}"
echo "[m8-smoke-100]   |ΔE|/|E₀|     = ${rel_drift}"

gate_ok=$(awk -v r="${rel_drift}" -v g="${ENERGY_REL_GATE}" 'BEGIN { print (r+0 <= g+0) ? 1 : 0 }')
if [[ "${gate_ok}" != "1" ]]; then
  echo "[m8-smoke-100] FAIL (physics): energy drift ${rel_drift} exceeds gate ${ENERGY_REL_GATE}" >&2
  echo "[m8-smoke-100]   Gate is the authoritative D-M8-8 NVE-drift envelope" >&2
  echo "[m8-smoke-100]   (gpu_mixed_fast_snap_only.nve_drift_per_100_steps = 3e-6," >&2
  echo "[m8-smoke-100]   v1.0.1 rescope 2026-04-21 — see thresholds.yaml header)." >&2
  echo "[m8-smoke-100]   A drift above this on T6 1024-atom NVE points to:" >&2
  echo "[m8-smoke-100]   integrator determinism regression, SNAP bispectrum" >&2
  echo "[m8-smoke-100]   reduction-order change, or neighbor-list ordering." >&2
  echo "[m8-smoke-100]   Bisect before relaxing the gate (master spec §D.15 red-flag)." >&2
  exit 1
fi

elapsed=$SECONDS
echo "[m8-smoke-100] elapsed:     ${elapsed}s"
if [[ "${elapsed}" -gt "${BUDGET_SEC}" ]]; then
  echo "[m8-smoke-100] FAIL (performance): ${elapsed}s > budget ${BUDGET_SEC}s." >&2
  exit 3
fi

echo "[m8-smoke-100] PASS — T6 1024-atom W BCC SNAP NVE: |ΔE|/|E₀| = ${rel_drift} ≤ ${ENERGY_REL_GATE}."
echo "              Authoritative D-M8-8 NVE-drift gate green (${N_STEPS}-step horizon)."
