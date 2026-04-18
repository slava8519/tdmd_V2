#!/usr/bin/env bash
#
# M4 integration smoke — closes the M4 milestone (master spec §14 M4 gate).
# End-to-end exercise of the M4 user surface:
#
#   1. tdmd validate                      → config with `scheduler.td_mode: true` accepted
#   2. tdmd run --timing                  → 10-step NVE with TD scheduler active
#   3. Thermo log matches the M3 golden byte-for-byte (D-M4-9)
#   4. Neigh-time fraction > 5 % (directional — rebuild still fires under TD)
#
# Pipeline inputs are the existing T4 assets (D-M3-6 / D-M4 reuse):
#   verify/benchmarks/t4_nial_alloy/setup.data  (864 atoms, LFS)
#   verify/third_party/potentials/NiAl_Mishin_2004.eam.alloy (1.9 MiB)
#
# Exec pack: docs/development/m4_execution_pack.md §T4.11.
# Spec: master spec §14 M4 gate;
#       scheduler/SPEC §2.3 (Reference canonical realisation);
#       verify/SPEC §7 (oracle-free smoke philosophy).
#
# Byte-exactness note (D-M4-9): the M4 golden IS the M3 golden, copied
# verbatim. K=1 single-rank (D-M4-1, D-M4-6) leaves force/integrator
# reduction order identical to the legacy path — a diff here means either
# (a) the scheduler touched simulation state it shouldn't have, or
# (b) M3 regressed underneath us. `diff -u m3/thermo_golden.txt
# m4/thermo_golden.txt` at smoke-edit time is how we guard (b).
#
# Flags / env:
#   --tdmd <path>             Path to the `tdmd` binary (required if not
#                             on $PATH; $TDMD_CLI_BIN is also honoured).
#   --keep-workdir            Don't rm the tmp workdir on success.
#   TDMD_UPDATE_GOLDENS=1     Overwrite golden instead of comparing.
#                             Commit only after a Validation Engineer review.
#                             NOTE: updating the M4 golden without also
#                             updating M3's golden breaks D-M4-9.
#   TDMD_SMOKE_BUDGET_SEC=N   Override the default 10s wall-time budget.
#
# Exit codes:
#   0   smoke green — all checks passed.
#   1   physics / determinism regression — thermo diverged from golden.
#   2   infrastructure — missing binary, malformed output, bad invocation.
#   3   performance — wall-time exceeded the budget.

set -euo pipefail

SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SMOKE_DIR}/../../.." && pwd)"

TEMPLATE="${SMOKE_DIR}/smoke_config.yaml.template"
THERMO_GOLDEN="${SMOKE_DIR}/thermo_golden.txt"
M3_GOLDEN="${REPO_ROOT}/tests/integration/m3_smoke/thermo_golden.txt"
ATOMS="${REPO_ROOT}/verify/benchmarks/t4_nial_alloy/setup.data"
EAM="${REPO_ROOT}/verify/third_party/potentials/NiAl_Mishin_2004.eam.alloy"

BUDGET_SEC="${TDMD_SMOKE_BUDGET_SEC:-10}"

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
      sed -n '3,45p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "[m4-smoke] error: unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${TDMD_BIN}" ]]; then
  if command -v tdmd >/dev/null 2>&1; then
    TDMD_BIN="$(command -v tdmd)"
  else
    echo "[m4-smoke] error: tdmd binary not found" >&2
    echo "[m4-smoke]   pass --tdmd <path> or set TDMD_CLI_BIN=<path>." >&2
    exit 2
  fi
fi

if [[ ! -x "${TDMD_BIN}" ]]; then
  echo "[m4-smoke] error: tdmd binary not executable: ${TDMD_BIN}" >&2
  exit 2
fi

for p in "${TEMPLATE}" "${ATOMS}" "${EAM}"; do
  if [[ ! -f "${p}" ]]; then
    echo "[m4-smoke] error: required file missing: ${p}" >&2
    if [[ "${p}" == "${ATOMS}" ]]; then
      echo "[m4-smoke]   setup.data is LFS-tracked; did you run 'git lfs pull'?" >&2
    fi
    exit 2
  fi
done

if [[ "$(head -c 10 "${ATOMS}")" == "version ht" ]]; then
  echo "[m4-smoke] error: ${ATOMS} is an unresolved LFS pointer." >&2
  echo "[m4-smoke]   run 'git lfs pull' (or set lfs:true on the CI checkout)." >&2
  exit 2
fi

# D-M4-9 meta-check: the M4 golden MUST be byte-identical to M3's. If this
# file drifts, someone has broken the acceptance gate contract at commit
# time. Catch it before running the binary.
if [[ -f "${M3_GOLDEN}" ]]; then
  if ! diff -q "${M3_GOLDEN}" "${THERMO_GOLDEN}" >/dev/null; then
    echo "[m4-smoke] FAIL (infra): M4 thermo golden diverges from M3 golden." >&2
    echo "[m4-smoke]   D-M4-9 requires byte-for-byte identity: the M4 golden IS the M3 golden." >&2
    echo "[m4-smoke]   diff ${M3_GOLDEN} ${THERMO_GOLDEN}" >&2
    exit 2
  fi
fi

WORKDIR="$(mktemp -d -t tdmd_m4_smoke.XXXXXX)"
cleanup() {
  if [[ "${KEEP_WORKDIR}" -eq 0 ]]; then
    rm -rf "${WORKDIR}"
  else
    echo "[m4-smoke] workdir preserved: ${WORKDIR}"
  fi
}
trap cleanup EXIT

CONFIG="${WORKDIR}/smoke_config.yaml"
THERMO="${WORKDIR}/thermo.log"
TIMING_ERR="${WORKDIR}/timing.stderr"

sed -e "s|{{ATOMS_PATH}}|${ATOMS}|g" \
    -e "s|{{EAM_PATH}}|${EAM}|g" \
    "${TEMPLATE}" > "${CONFIG}"

echo "[m4-smoke] tdmd binary: ${TDMD_BIN}"
echo "[m4-smoke] workdir:     ${WORKDIR}"
echo "[m4-smoke] budget:      ${BUDGET_SEC}s"

SECONDS=0

echo "[m4-smoke] step 1/4: tdmd validate (td_mode=true)"
if ! "${TDMD_BIN}" validate "${CONFIG}" >/dev/null 2>&1; then
  echo "[m4-smoke] FAIL (infra): tdmd validate exited non-zero on TD-mode smoke config." >&2
  echo "[m4-smoke]   re-run manually: ${TDMD_BIN} validate ${CONFIG}" >&2
  exit 2
fi

echo "[m4-smoke] step 2/4: tdmd run --quiet --thermo --timing (td_mode=true)"
if ! "${TDMD_BIN}" run \
     --quiet \
     --thermo "${THERMO}" \
     --timing \
     "${CONFIG}" 2> "${TIMING_ERR}"; then
  echo "[m4-smoke] FAIL (infra): tdmd run exited non-zero under td_mode=true." >&2
  echo "[m4-smoke]   stderr excerpt:" >&2
  tail -20 "${TIMING_ERR}" >&2 || true
  exit 2
fi

echo "[m4-smoke] step 3/4: thermo D-M4-9 byte-exact diff vs M3 golden"

if ! grep -q '^# step temp pe ke etotal press' "${THERMO}"; then
  echo "[m4-smoke] FAIL (infra): thermo header missing or mismatched in ${THERMO}" >&2
  head -3 "${THERMO}" >&2 || true
  exit 2
fi

actual_rows="$(wc -l < "${THERMO}")"
if [[ "${actual_rows}" -ne 12 ]]; then
  echo "[m4-smoke] FAIL (infra): expected 12 thermo lines (1 header + 11 data), got ${actual_rows}" >&2
  exit 2
fi

if [[ "${TDMD_UPDATE_GOLDENS:-0}" == "1" ]]; then
  cp "${THERMO}" "${THERMO_GOLDEN}"
  echo "[m4-smoke] thermo golden updated: ${THERMO_GOLDEN}"
  echo "[m4-smoke] WARNING: D-M4-9 requires this to equal M3's golden — sync both or revert." >&2
elif [[ -f "${THERMO_GOLDEN}" ]]; then
  if ! diff -u "${THERMO_GOLDEN}" "${THERMO}"; then
    echo "[m4-smoke] FAIL (D-M4-9): td_mode=true thermo diverges from golden." >&2
    echo "[m4-smoke]   K=1 single-rank MUST produce thermo byte-identical to legacy." >&2
    echo "[m4-smoke]   if scheduler wiring changed legal bits, the byte-exact contract" >&2
    echo "[m4-smoke]   is broken — not a tolerance discussion." >&2
    exit 1
  fi
else
  echo "[m4-smoke] FAIL (infra): thermo golden missing at ${THERMO_GOLDEN}." >&2
  echo "[m4-smoke]   generate it on first bring-up with TDMD_UPDATE_GOLDENS=1 $0" >&2
  exit 2
fi

echo "[m4-smoke] step 4/4: neighbor-rebuild directional check"

BREAKDOWN_HEADER='Section |  min time  |  avg time  |  max time  |%varavg| %total'
if ! grep -Fq "${BREAKDOWN_HEADER}" "${TIMING_ERR}"; then
  echo "[m4-smoke] FAIL (infra): LAMMPS breakdown header missing from --timing stderr." >&2
  tail -20 "${TIMING_ERR}" >&2
  exit 2
fi

neigh_pct="$(grep -E '^Neigh +\|' "${TIMING_ERR}" | awk -F'|' '{gsub(/ /, "", $6); print $6}')"
if [[ -z "${neigh_pct}" ]]; then
  echo "[m4-smoke] FAIL (infra): could not extract Neigh %total from timing breakdown." >&2
  tail -20 "${TIMING_ERR}" >&2
  exit 2
fi

if ! awk -v v="${neigh_pct}" 'BEGIN { exit !(v > 5.0) }'; then
  echo "[m4-smoke] FAIL (neighbor): Neigh %total = ${neigh_pct} ≤ 5 — rebuild did not fire." >&2
  echo "[m4-smoke]   with skin=0.05 Å on the T4 300 K starting config, at least one" >&2
  echo "[m4-smoke]   mid-run rebuild must occur even under td_mode=true." >&2
  tail -20 "${TIMING_ERR}" >&2
  exit 1
fi

elapsed=$SECONDS
echo "[m4-smoke] elapsed:    ${elapsed}s"
echo "[m4-smoke] Neigh %:    ${neigh_pct}"
if [[ "${elapsed}" -gt "${BUDGET_SEC}" ]]; then
  echo "[m4-smoke] FAIL (performance): smoke took ${elapsed}s > budget ${BUDGET_SEC}s." >&2
  exit 3
fi

echo "[m4-smoke] PASS — td_mode=true 10-step EAM thermo matches M3 golden byte-for-byte,"
echo "         neighbor rebuild fired mid-run (Neigh ${neigh_pct}% of total)."
echo "         D-M4-9 acceptance gate green."
