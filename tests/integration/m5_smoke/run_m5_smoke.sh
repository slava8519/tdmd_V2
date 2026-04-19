#!/usr/bin/env bash
#
# M5 integration smoke — closes the M5 milestone (master spec §14 M5 gate).
# End-to-end exercise of the M5 user surface:
#
#   1. Pre-flight: M5 thermo golden byte-matches M4's (D-M5-12 chain).
#   2. mpirun -np 2 tdmd validate     → 2-rank comm config accepted.
#   3. mpirun -np 2 tdmd run --telemetry-jsonl
#                                     → 10-step NVE with TD scheduler +
#                                        MpiHostStaging backend.
#   4. Thermo log matches M5 golden byte-for-byte (D-M5-12 / D-M4-9 /
#                                        D-M3-6 chain).
#   5. Telemetry invariants: run_end event emitted, wall-time ≤ budget,
#                            ignored_end_calls == 0 (no RAII leaks).
#
# Pipeline inputs are identical to the M4 smoke (D-M5-12 byte-exact chain):
#   verify/benchmarks/t4_nial_alloy/setup.data  (864 atoms, LFS)
#   verify/third_party/potentials/NiAl_Mishin_2004.eam.alloy (1.9 MiB)
#
# Exec pack: docs/development/m5_execution_pack.md §T5.12.
# Spec: master spec §14 M5 gate; scheduler/SPEC §12.3 (determinism);
#       comm/SPEC §3 (MpiHostStaging); verify/SPEC §7 (oracle-free smoke).
#
# Flags / env:
#   --tdmd <path>             Path to the `tdmd` binary (required if not
#                             on $PATH; $TDMD_CLI_BIN is also honoured).
#                             Must be built with -DTDMD_ENABLE_MPI=ON.
#   --mpirun <path>           MPI launcher (default: `mpirun` on $PATH).
#                             mpiexec works on MPICH installs.
#   --keep-workdir            Don't rm the tmp workdir on success.
#   TDMD_UPDATE_GOLDENS=1     Overwrite golden instead of comparing.
#   TDMD_SMOKE_BUDGET_SEC=N   Override the default 30s wall-time budget.
#
# Exit codes:
#   0   smoke green — all checks passed.
#   1   physics / determinism regression — thermo diverged OR telemetry
#       invariant broken.
#   2   infrastructure — missing binary, mpirun absent, malformed output,
#       bad invocation.
#   3   performance — wall-time exceeded the budget.

set -euo pipefail

SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SMOKE_DIR}/../../.." && pwd)"

TEMPLATE="${SMOKE_DIR}/smoke_config.yaml.template"
THERMO_GOLDEN="${SMOKE_DIR}/thermo_golden.txt"
M4_GOLDEN="${REPO_ROOT}/tests/integration/m4_smoke/thermo_golden.txt"
TELEMETRY_EXPECTED="${SMOKE_DIR}/telemetry_expected.txt"
ATOMS="${REPO_ROOT}/verify/benchmarks/t4_nial_alloy/setup.data"
EAM="${REPO_ROOT}/verify/third_party/potentials/NiAl_Mishin_2004.eam.alloy"

BUDGET_SEC="${TDMD_SMOKE_BUDGET_SEC:-30}"

TDMD_BIN="${TDMD_CLI_BIN:-}"
MPIRUN_BIN=""
KEEP_WORKDIR=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tdmd)
      TDMD_BIN="$2"
      shift 2
      ;;
    --mpirun)
      MPIRUN_BIN="$2"
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
      echo "[m5-smoke] error: unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${TDMD_BIN}" ]]; then
  if command -v tdmd >/dev/null 2>&1; then
    TDMD_BIN="$(command -v tdmd)"
  else
    echo "[m5-smoke] error: tdmd binary not found" >&2
    echo "[m5-smoke]   pass --tdmd <path> or set TDMD_CLI_BIN=<path>." >&2
    exit 2
  fi
fi

if [[ ! -x "${TDMD_BIN}" ]]; then
  echo "[m5-smoke] error: tdmd binary not executable: ${TDMD_BIN}" >&2
  exit 2
fi

if [[ -z "${MPIRUN_BIN}" ]]; then
  if command -v mpirun >/dev/null 2>&1; then
    MPIRUN_BIN="$(command -v mpirun)"
  elif command -v mpiexec >/dev/null 2>&1; then
    MPIRUN_BIN="$(command -v mpiexec)"
  else
    echo "[m5-smoke] SKIPPED — no mpirun/mpiexec on \$PATH" >&2
    echo "[m5-smoke]   install openmpi-bin (or mpich-bin) to enable." >&2
    exit 0
  fi
fi

for p in "${TEMPLATE}" "${ATOMS}" "${EAM}" "${TELEMETRY_EXPECTED}"; do
  if [[ ! -f "${p}" ]]; then
    echo "[m5-smoke] error: required file missing: ${p}" >&2
    if [[ "${p}" == "${ATOMS}" ]]; then
      echo "[m5-smoke]   setup.data is LFS-tracked; did you run 'git lfs pull'?" >&2
    fi
    exit 2
  fi
done

if [[ "$(head -c 10 "${ATOMS}")" == "version ht" ]]; then
  echo "[m5-smoke] error: ${ATOMS} is an unresolved LFS pointer." >&2
  echo "[m5-smoke]   run 'git lfs pull' (or set lfs:true on the CI checkout)." >&2
  exit 2
fi

# D-M5-12 pre-flight: the M5 golden MUST be byte-identical to M4's. Catches
# the classic "someone edited one without syncing the other" mistake before
# we spend MPI rank time reproducing it.
if [[ -f "${M4_GOLDEN}" ]]; then
  if ! diff -q "${M4_GOLDEN}" "${THERMO_GOLDEN}" >/dev/null; then
    echo "[m5-smoke] FAIL (infra): M5 thermo golden diverges from M4 golden." >&2
    echo "[m5-smoke]   D-M5-12 requires byte-for-byte identity: the M5 golden IS the M4 golden." >&2
    echo "[m5-smoke]   diff ${M4_GOLDEN} ${THERMO_GOLDEN}" >&2
    exit 2
  fi
fi

WORKDIR="$(mktemp -d -t tdmd_m5_smoke.XXXXXX)"
cleanup() {
  if [[ "${KEEP_WORKDIR}" -eq 0 ]]; then
    rm -rf "${WORKDIR}"
  else
    echo "[m5-smoke] workdir preserved: ${WORKDIR}"
  fi
}
trap cleanup EXIT

CONFIG="${WORKDIR}/smoke_config.yaml"
THERMO="${WORKDIR}/thermo.log"
TELEMETRY="${WORKDIR}/telemetry.jsonl"
RUN_ERR="${WORKDIR}/run.stderr"

sed -e "s|{{ATOMS_PATH}}|${ATOMS}|g" \
    -e "s|{{EAM_PATH}}|${EAM}|g" \
    "${TEMPLATE}" > "${CONFIG}"

echo "[m5-smoke] tdmd binary: ${TDMD_BIN}"
echo "[m5-smoke] mpirun:      ${MPIRUN_BIN}"
echo "[m5-smoke] workdir:     ${WORKDIR}"
echo "[m5-smoke] budget:      ${BUDGET_SEC}s"

SECONDS=0

echo "[m5-smoke] step 1/5: golden parity pre-flight (D-M5-12)"
# Already checked above; this line is just for the progress log.
echo "[m5-smoke]   M5 golden ≡ M4 golden."

echo "[m5-smoke] step 2/5: mpirun -np 2 tdmd validate"
# `mpirun -np 2 tdmd validate` runs validate on 2 ranks; each emits its
# own exit status. We accept the join: non-zero iff ANY rank disagreed.
if ! "${MPIRUN_BIN}" -np 2 --oversubscribe "${TDMD_BIN}" validate "${CONFIG}" >/dev/null 2>&1; then
  # Some mpirun builds reject --oversubscribe — retry without it.
  if ! "${MPIRUN_BIN}" -np 2 "${TDMD_BIN}" validate "${CONFIG}" >/dev/null 2>&1; then
    echo "[m5-smoke] FAIL (infra): mpirun -np 2 tdmd validate exited non-zero." >&2
    echo "[m5-smoke]   re-run: ${MPIRUN_BIN} -np 2 ${TDMD_BIN} validate ${CONFIG}" >&2
    exit 2
  fi
fi

echo "[m5-smoke] step 3/5: mpirun -np 2 tdmd run --thermo --telemetry-jsonl"
# Oversubscribe is needed on CI runners that report 1 logical core.
MPI_ARGS=(-np 2 --oversubscribe)
if ! "${MPIRUN_BIN}" "${MPI_ARGS[@]}" "${TDMD_BIN}" run \
     --quiet \
     --thermo "${THERMO}" \
     --telemetry-jsonl "${TELEMETRY}" \
     "${CONFIG}" 2> "${RUN_ERR}"; then
  # Fallback without oversubscribe for MPICH / LAM.
  MPI_ARGS=(-np 2)
  if ! "${MPIRUN_BIN}" "${MPI_ARGS[@]}" "${TDMD_BIN}" run \
       --quiet \
       --thermo "${THERMO}" \
       --telemetry-jsonl "${TELEMETRY}" \
       "${CONFIG}" 2> "${RUN_ERR}"; then
    echo "[m5-smoke] FAIL (infra): mpirun -np 2 tdmd run exited non-zero." >&2
    echo "[m5-smoke]   stderr excerpt:" >&2
    tail -30 "${RUN_ERR}" >&2 || true
    exit 2
  fi
fi

echo "[m5-smoke] step 4/5: thermo D-M5-12 byte-exact diff vs M4 golden"

if ! grep -q '^# step temp pe ke etotal press' "${THERMO}"; then
  echo "[m5-smoke] FAIL (infra): thermo header missing or mismatched in ${THERMO}" >&2
  head -3 "${THERMO}" >&2 || true
  exit 2
fi

actual_rows="$(wc -l < "${THERMO}")"
if [[ "${actual_rows}" -ne 12 ]]; then
  echo "[m5-smoke] FAIL (infra): expected 12 thermo lines (1 header + 11 data), got ${actual_rows}" >&2
  exit 2
fi

if [[ "${TDMD_UPDATE_GOLDENS:-0}" == "1" ]]; then
  cp "${THERMO}" "${THERMO_GOLDEN}"
  echo "[m5-smoke] thermo golden updated: ${THERMO_GOLDEN}"
  echo "[m5-smoke] WARNING: D-M5-12 requires this to equal M4's golden — sync both or revert." >&2
elif [[ -f "${THERMO_GOLDEN}" ]]; then
  if ! diff -u "${THERMO_GOLDEN}" "${THERMO}"; then
    echo "[m5-smoke] FAIL (D-M5-12): K=1 P=2 MpiHostStaging thermo diverges from golden." >&2
    echo "[m5-smoke]   K=1 P=2 MUST produce thermo byte-identical to K=1 P=1 = M4 golden." >&2
    echo "[m5-smoke]   if the backend reduction is perturbing bits, check that thermo goes" >&2
    echo "[m5-smoke]   through deterministic_sum_double and not raw MPI_Allreduce." >&2
    exit 1
  fi
else
  echo "[m5-smoke] FAIL (infra): thermo golden missing at ${THERMO_GOLDEN}." >&2
  echo "[m5-smoke]   generate it on first bring-up with TDMD_UPDATE_GOLDENS=1 $0" >&2
  exit 2
fi

echo "[m5-smoke] step 5/5: telemetry invariants"

if [[ ! -s "${TELEMETRY}" ]]; then
  echo "[m5-smoke] FAIL (infra): telemetry file empty or missing: ${TELEMETRY}" >&2
  exit 2
fi

# Read first line (Telemetry writes exactly one JSON object).
telemetry_line="$(head -n 1 "${TELEMETRY}")"

fail=0
while IFS= read -r line; do
  # Strip comments + blanks.
  line="${line%%#*}"
  # shellcheck disable=SC2001
  line="$(echo "${line}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  [[ -z "${line}" ]] && continue

  key="$(echo "${line}" | awk '{print $1}')"
  op="$(echo "${line}" | awk '{print $2}')"
  want="$(echo "${line}" | awk '{print $3}')"

  # Extract value from JSON line. Strings come back quoted; strip outer quotes.
  if [[ "${key}" == "event" ]]; then
    got="$(echo "${telemetry_line}" | grep -oE '"event":"[^"]+"' | sed -e 's/"event":"//' -e 's/"$//')"
  else
    got="$(echo "${telemetry_line}" | grep -oE "\"${key}\":[^,}]+" | sed -e "s/\"${key}\"://" -e 's/[}" ]//g')"
  fi

  if [[ -z "${got}" ]]; then
    echo "[m5-smoke] FAIL (telemetry): key '${key}' missing from JSONL output" >&2
    fail=1
    continue
  fi

  ok=0
  case "${op}" in
    "==")
      if [[ "${got}" == "${want}" ]]; then ok=1; fi
      ;;
    ">=")
      ok=$(awk -v a="${got}" -v b="${want}" 'BEGIN { print (a+0 >= b+0) ? 1 : 0 }')
      ;;
    "<=")
      ok=$(awk -v a="${got}" -v b="${want}" 'BEGIN { print (a+0 <= b+0) ? 1 : 0 }')
      ;;
    ">")
      ok=$(awk -v a="${got}" -v b="${want}" 'BEGIN { print (a+0 > b+0) ? 1 : 0 }')
      ;;
    "<")
      ok=$(awk -v a="${got}" -v b="${want}" 'BEGIN { print (a+0 < b+0) ? 1 : 0 }')
      ;;
    *)
      echo "[m5-smoke] FAIL (telemetry): unknown operator '${op}' for key '${key}'" >&2
      fail=1
      continue
      ;;
  esac

  if [[ "${ok}" != "1" ]]; then
    echo "[m5-smoke] FAIL (telemetry): ${key} ${op} ${want} — got ${got}" >&2
    fail=1
  fi
done < "${TELEMETRY_EXPECTED}"

if [[ "${fail}" -ne 0 ]]; then
  echo "[m5-smoke]   telemetry JSONL:" >&2
  echo "[m5-smoke]   ${telemetry_line}" >&2
  exit 1
fi

elapsed=$SECONDS
echo "[m5-smoke] elapsed:     ${elapsed}s"
if [[ "${elapsed}" -gt "${BUDGET_SEC}" ]]; then
  echo "[m5-smoke] FAIL (performance): smoke took ${elapsed}s > budget ${BUDGET_SEC}s." >&2
  exit 3
fi

echo "[m5-smoke] PASS — K=1 P=2 MpiHostStaging 10-step EAM thermo matches M4 golden"
echo "         byte-for-byte; telemetry clean; D-M5-12 acceptance gate green."
