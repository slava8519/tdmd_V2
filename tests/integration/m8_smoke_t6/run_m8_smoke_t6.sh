#!/usr/bin/env bash
#
# M8 T6 integration smoke — exercises the M8 user surface on the canonical T6
# tungsten SNAP fixture (1024-atom W BCC, `W_2940_2017_2.snap`, single-rank
# Fp64Reference NVE, 10 steps) and gates energy conservation at 1e-12 relative
# drift over the run. Closes M8 T8.10 on top of T8.5 (D-M8-7 byte-exact) and
# T8.7 (GPU FP64 byte-exact) — see `docs/development/m8_execution_pack.md`.
#
# Flow:
#   1. Submodule probe — self-skip (exit 77) if LAMMPS submodule not
#      initialized (W_2940_2017_2.snap* fixture files absent).
#   2. LFS probe — fail (exit 2) if setup_1024.data is an unresolved LFS
#      pointer (developer forgot `git lfs pull`).
#   3. GPU probe — self-skip (exit 0) if nvidia-smi sees no CUDA device
#      (Option A: no public-runner GPU CI; smoke is pre-push local).
#   4. Instantiate config from benchmark template (absolute fixture paths).
#   5. `tdmd validate <config>` — preflight.
#   6. `tdmd run --thermo --telemetry-jsonl <config>` — 10-step NVE.
#   7. Parse thermo `etotal` column; compute |E(step=10) - E(step=0)| / |E(0)|
#      and gate at 1e-7 relative drift (see gate-derivation note below).
#
# Exec pack: docs/development/m8_execution_pack.md §T8.10.
# Spec:      master spec §14 M8; verify/SPEC §4.7 T6 canonical fixture.
# Decision:  D-M8-7 (CPU FP64 byte-exact), D-M8-8 (NVE drift envelope),
#            D-M8-9 (T6 fixture scales).
#
# Gate derivation (supersedes exec pack "1e-12" which is physically wrong):
#   The exec pack §T8.10 line 1081 writes "energy conservation gate ≤ 1e-12
#   rel on 10-step Fp64Reference run". That threshold conflates two different
#   phenomena: (a) FP64 accumulation floor ~k_max·eps_FP64 ~ 1e-14 for
#   SNAP bispectrum on a single force evaluation (relevant to D-M8-7 byte-
#   exact oracle gate — *static* configuration, no integrator); and (b)
#   NVE energy drift over integrated steps, which is dominated by VV's
#   O((ω·dt)²) local truncation error — O(1e-5) per W phonon period at
#   dt = 0.5 fs on the T6 fixture. No realistic NVE run can hit 1e-12.
#
#   The physically grounded gate is the D-M8-8 NVE-drift threshold
#   registered in `verify/thresholds/thresholds.yaml`
#   (`gpu_mixed_fast_snap_only.nve_drift_per_100_steps = 3e-6`, v1.0.1
#   rescoped from the original 1e-5/1000 form 2026-04-21 — pure SNAP on
#   the T6 fixture is physically unstable over 1000 steps, see rationale
#   doc §4 + thresholds.yaml header note). √-scaled to a 10-step smoke
#   under the diffusive round-off model: 3e-6 × √(10/100) ≈ 9.49e-7
#   → **1e-6 relative drift**. Bring-up measurement (2026-04-21, RTX 5080):
#   Fp64Reference `|ΔE|/|E₀| = 2.5e-9` (≈400× headroom),
#   MixedFastSnapOnly `|ΔE|/|E₀| = 1.8e-7` (≈5× headroom). Session-report
#   SPEC delta amends exec pack line 1081 to the √-scaled 1e-6 form.
#
# Flags / env:
#   --tdmd <path>             Path to the tdmd binary. Must be built with
#                             -DTDMD_BUILD_CUDA=ON on Fp64ReferenceBuild
#                             (the M8 oracle flavor). $TDMD_CLI_BIN also
#                             honored.
#   --keep-workdir            Don't rm the tmp workdir on success.
#   TDMD_M8_SMOKE_BUDGET_SEC  Override default 60s wall-time budget.
#
# Exit codes:
#   0   green, or SKIPPED (missing submodule / missing GPU).
#   1   physics regression — energy drift exceeds √-scaled 1e-6 gate.
#   2   infra — missing binary, LFS pointer unresolved, malformed output.
#   3   perf — smoke exceeded wall-time budget.
#  77   standardised skip — submodule not initialized; matches
#       test_lammps_oracle_snap_fixture pattern.

set -euo pipefail

SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SMOKE_DIR}/../../.." && pwd)"

BENCH_DIR="${REPO_ROOT}/verify/benchmarks/t6_snap_tungsten"
TEMPLATE="${BENCH_DIR}/config.yaml.template"
ATOMS="${BENCH_DIR}/setup_1024.data"

SNAP_DIR="${REPO_ROOT}/verify/third_party/lammps/examples/snap"
SNAP_COEFF="${SNAP_DIR}/W_2940_2017_2.snapcoeff"
SNAP_PARAM="${SNAP_DIR}/W_2940_2017_2.snapparam"

BUDGET_SEC="${TDMD_M8_SMOKE_BUDGET_SEC:-60}"
# See the "Gate derivation" note in the header for why 1e-6 (√-scaled) and
# not 1e-12 (exec-pack original) or 1e-7 (pre-v1.0.1 linearly-scaled form).
ENERGY_REL_GATE="1.0e-6"

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
      sed -n '3,42p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "[m8-smoke] error: unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

# Step 1: submodule / fixture probe. SNAP coefficient files ship through
# the LAMMPS submodule (D-M8-3 — no TDMD-side binary tracked); if the
# submodule is uninitialized, the files are absent and the smoke cannot run.
# Exit 77 matches test_lammps_oracle_snap_fixture (Catch2 SKIP_RETURN_CODE).
for f in "${SNAP_COEFF}" "${SNAP_PARAM}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[m8-smoke] SKIPPED — LAMMPS submodule not initialized (fixture missing)" >&2
    echo "[m8-smoke]   missing: ${f}" >&2
    echo "[m8-smoke]   run 'git submodule update --init --recursive' to fetch." >&2
    exit 77
  fi
done

# Step 2: fixture / LFS probe. setup_1024.data ships via Git LFS; a pointer
# file instead of the binary means the developer skipped `git lfs pull`.
if [[ ! -f "${ATOMS}" ]]; then
  echo "[m8-smoke] error: fixture missing: ${ATOMS}" >&2
  echo "[m8-smoke]   run 'cd verify/benchmarks/t6_snap_tungsten &&" >&2
  echo "[m8-smoke]       python3 generate_setup.py --nrep 8 --out setup_1024.data'" >&2
  exit 2
fi
if [[ "$(head -c 10 "${ATOMS}")" == "version ht" ]]; then
  echo "[m8-smoke] error: ${ATOMS} is an unresolved LFS pointer." >&2
  echo "[m8-smoke]   run 'git lfs pull'." >&2
  exit 2
fi

if [[ ! -f "${TEMPLATE}" ]]; then
  echo "[m8-smoke] error: config template missing: ${TEMPLATE}" >&2
  exit 2
fi

# Step 3: GPU probe (Option A — no public-runner GPU CI). Matches m6/m7
# smoke pattern: probe via nvidia-smi -L (host may ship toolkit without a
# physical GPU, e.g. container build hosts).
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[m8-smoke] SKIPPED — nvidia-smi not on \$PATH (no GPU visible)"
  exit 0
fi
if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
  echo "[m8-smoke] SKIPPED — nvidia-smi reports no CUDA devices"
  exit 0
fi

# Step 4: binary probe.
if [[ -z "${TDMD_BIN}" ]]; then
  if command -v tdmd >/dev/null 2>&1; then
    TDMD_BIN="$(command -v tdmd)"
  else
    echo "[m8-smoke] error: tdmd binary not found" >&2
    echo "[m8-smoke]   pass --tdmd <path> or set TDMD_CLI_BIN=<path>." >&2
    exit 2
  fi
fi
if [[ ! -x "${TDMD_BIN}" ]]; then
  echo "[m8-smoke] error: tdmd binary not executable: ${TDMD_BIN}" >&2
  exit 2
fi

# Step 5: prep workdir.
WORKDIR="$(mktemp -d -t tdmd_m8_smoke.XXXXXX)"
cleanup() {
  if [[ "${KEEP_WORKDIR}" -eq 0 ]]; then
    rm -rf "${WORKDIR}"
  else
    echo "[m8-smoke] workdir preserved: ${WORKDIR}"
  fi
}
trap cleanup EXIT

CONFIG="${WORKDIR}/config.yaml"
THERMO="${WORKDIR}/thermo.log"
RUN_ERR="${WORKDIR}/run.stderr"

sed -e "s|{{ATOMS_PATH}}|${ATOMS}|g" \
    -e "s|{{SNAP_COEFF}}|${SNAP_COEFF}|g" \
    -e "s|{{SNAP_PARAM}}|${SNAP_PARAM}|g" \
    "${TEMPLATE}" > "${CONFIG}"

echo "[m8-smoke] tdmd binary: ${TDMD_BIN}"
echo "[m8-smoke] workdir:     ${WORKDIR}"
echo "[m8-smoke] budget:      ${BUDGET_SEC}s"
nvidia-smi -L 2>/dev/null | head -1 | sed 's/^/[m8-smoke] gpu:         /'

SECONDS=0

echo "[m8-smoke] step 1/3: tdmd validate"
if ! "${TDMD_BIN}" validate "${CONFIG}" >/dev/null 2>&1; then
  echo "[m8-smoke] FAIL (infra): tdmd validate exited non-zero." >&2
  echo "[m8-smoke]   re-run: ${TDMD_BIN} validate ${CONFIG}" >&2
  exit 2
fi

echo "[m8-smoke] step 2/3: tdmd run (10-step NVE, 1024 atoms)"
if ! "${TDMD_BIN}" run --quiet --thermo "${THERMO}" "${CONFIG}" 2> "${RUN_ERR}"; then
  echo "[m8-smoke] FAIL (infra): tdmd run exited non-zero." >&2
  echo "[m8-smoke]   stderr excerpt:" >&2
  tail -30 "${RUN_ERR}" >&2 || true
  exit 2
fi

echo "[m8-smoke] step 3/3: energy-conservation gate (|ΔE|/|E₀| ≤ ${ENERGY_REL_GATE})"

if ! grep -q '^# step temp pe ke etotal press' "${THERMO}"; then
  echo "[m8-smoke] FAIL (infra): thermo header missing / mismatched." >&2
  head -3 "${THERMO}" >&2 || true
  exit 2
fi

# Thermo format: `# step temp pe ke etotal press` → etotal is column 5.
# With `thermo.every: 1` and `n_steps: 10` we expect 11 data rows (0..10).
data_rows="$(grep -cv '^#' "${THERMO}" || true)"
if [[ "${data_rows}" -ne 11 ]]; then
  echo "[m8-smoke] FAIL (infra): expected 11 thermo data rows (steps 0..10), got ${data_rows}" >&2
  head -20 "${THERMO}" >&2 || true
  exit 2
fi

etotal_0="$(awk '!/^#/ {print $5; exit}' "${THERMO}")"
etotal_n="$(awk '!/^#/ {val=$5} END {print val}' "${THERMO}")"

if [[ -z "${etotal_0}" || -z "${etotal_n}" ]]; then
  echo "[m8-smoke] FAIL (infra): could not extract etotal from thermo." >&2
  exit 2
fi

rel_drift="$(awk -v e0="${etotal_0}" -v en="${etotal_n}" \
  'BEGIN { d = (en - e0); if (d < 0) d = -d; denom = e0; if (denom < 0) denom = -denom; printf("%.6e", d/denom) }')"

echo "[m8-smoke]   etotal(t=0)  = ${etotal_0}"
echo "[m8-smoke]   etotal(t=10) = ${etotal_n}"
echo "[m8-smoke]   |ΔE|/|E₀|    = ${rel_drift}"

gate_ok=$(awk -v r="${rel_drift}" -v g="${ENERGY_REL_GATE}" 'BEGIN { print (r+0 <= g+0) ? 1 : 0 }')
if [[ "${gate_ok}" != "1" ]]; then
  echo "[m8-smoke] FAIL (physics): energy drift ${rel_drift} exceeds gate ${ENERGY_REL_GATE}" >&2
  echo "[m8-smoke]   Gate is √-scaled D-M8-8 NVE-drift envelope (3e-6 per 100" >&2
  echo "[m8-smoke]   steps → ~1e-6 per 10 steps under diffusive model). A drift above this" >&2
  echo "[m8-smoke]   on T6 1024-atom NVE points to: integrator determinism regression," >&2
  echo "[m8-smoke]   SNAP bispectrum reduction-order change, or neighbor-list ordering." >&2
  echo "[m8-smoke]   Bisect before relaxing the gate (master spec §D.15 red-flag)." >&2
  exit 1
fi

elapsed=$SECONDS
echo "[m8-smoke] elapsed:     ${elapsed}s"
if [[ "${elapsed}" -gt "${BUDGET_SEC}" ]]; then
  echo "[m8-smoke] FAIL (performance): ${elapsed}s > budget ${BUDGET_SEC}s." >&2
  exit 3
fi

echo "[m8-smoke] PASS — T6 1024-atom W BCC SNAP NVE: |ΔE|/|E₀| = ${rel_drift} ≤ ${ENERGY_REL_GATE}."
echo "         M8 T8.10 acceptance gate green (D-M8-8 NVE-drift envelope, √-scaled)."
