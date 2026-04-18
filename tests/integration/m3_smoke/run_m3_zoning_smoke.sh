#!/usr/bin/env bash
#
# M3 integration smoke — closes the M3 milestone (master spec §14 M3 gate).
# End-to-end exercise of the M3 user surface:
#
#   1. tdmd validate                      → config accepted
#   2. tdmd explain --zoning              → ZoningPlan rationale (T3.6)
#   3. tdmd run --timing                  → mid-run neighbor rebuild (T3.8)
#   4. Thermo log matches committed golden
#   5. Neigh-time fraction > 5 % (directional — confirms rebuild fired)
#
# Pipeline inputs are the existing T4 assets (D-M3-6):
#   verify/benchmarks/t4_nial_alloy/setup.data  (864 atoms, LFS)
#   verify/third_party/potentials/NiAl_Mishin_2004.eam.alloy (1.9 MiB)
#
# Exec pack: docs/development/m3_execution_pack.md §T3.9.
# Spec: master spec §14 M3 gate;
#       zoning/SPEC §3.4 (scheme selection);
#       neighbor/SPEC §5 (displacement tracker rebuild hygiene);
#       verify/SPEC §7 (oracle-free smoke philosophy).
#
# Flags / env:
#   --tdmd <path>             Path to the `tdmd` binary (required if not
#                             on $PATH; $TDMD_CLI_BIN is also honoured).
#   --keep-workdir            Don't rm the tmp workdir on success.
#   TDMD_UPDATE_GOLDENS=1     Overwrite goldens instead of comparing.
#                             Commit only after a Validation Engineer review.
#   TDMD_SMOKE_BUDGET_SEC=N   Override the default 10s wall-time budget.
#
# Exit codes:
#   0   smoke green — all checks passed.
#   1   physics / plan regression — thermo or zoning golden diverged.
#   2   infrastructure — missing binary, malformed output, bad invocation.
#   3   performance — wall-time exceeded the budget.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SMOKE_DIR}/../../.." && pwd)"

TEMPLATE="${SMOKE_DIR}/smoke_config.yaml.template"
THERMO_GOLDEN="${SMOKE_DIR}/thermo_golden.txt"
ZONING_GOLDEN="${SMOKE_DIR}/zoning_rationale_golden.txt"
ATOMS="${REPO_ROOT}/verify/benchmarks/t4_nial_alloy/setup.data"
EAM="${REPO_ROOT}/verify/third_party/potentials/NiAl_Mishin_2004.eam.alloy"

BUDGET_SEC="${TDMD_SMOKE_BUDGET_SEC:-10}"

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
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
      sed -n '3,34p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "[smoke] error: unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${TDMD_BIN}" ]]; then
  if command -v tdmd >/dev/null 2>&1; then
    TDMD_BIN="$(command -v tdmd)"
  else
    echo "[smoke] error: tdmd binary not found" >&2
    echo "[smoke]   pass --tdmd <path> or set TDMD_CLI_BIN=<path>." >&2
    exit 2
  fi
fi

if [[ ! -x "${TDMD_BIN}" ]]; then
  echo "[smoke] error: tdmd binary not executable: ${TDMD_BIN}" >&2
  exit 2
fi

for p in "${TEMPLATE}" "${ATOMS}" "${EAM}"; do
  if [[ ! -f "${p}" ]]; then
    echo "[smoke] error: required file missing: ${p}" >&2
    if [[ "${p}" == "${ATOMS}" ]]; then
      echo "[smoke]   setup.data is LFS-tracked; did you run 'git lfs pull'?" >&2
    fi
    exit 2
  fi
done

if [[ "$(head -c 10 "${ATOMS}")" == "version ht" ]]; then
  echo "[smoke] error: ${ATOMS} is an unresolved LFS pointer." >&2
  echo "[smoke]   run 'git lfs pull' (or set lfs:true on the CI checkout)." >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Workdir + config materialisation
# ---------------------------------------------------------------------------
WORKDIR="$(mktemp -d -t tdmd_m3_smoke.XXXXXX)"
cleanup() {
  if [[ "${KEEP_WORKDIR}" -eq 0 ]]; then
    rm -rf "${WORKDIR}"
  else
    echo "[smoke] workdir preserved: ${WORKDIR}"
  fi
}
trap cleanup EXIT

CONFIG="${WORKDIR}/smoke_config.yaml"
ZONING_OUT="${WORKDIR}/zoning.txt"
THERMO="${WORKDIR}/thermo.log"
TIMING_ERR="${WORKDIR}/timing.stderr"

sed -e "s|{{ATOMS_PATH}}|${ATOMS}|g" \
    -e "s|{{EAM_PATH}}|${EAM}|g" \
    "${TEMPLATE}" > "${CONFIG}"

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
echo "[smoke] tdmd binary: ${TDMD_BIN}"
echo "[smoke] workdir:     ${WORKDIR}"
echo "[smoke] budget:      ${BUDGET_SEC}s"

SECONDS=0

echo "[smoke] step 1/5: tdmd validate"
if ! "${TDMD_BIN}" validate "${CONFIG}" >/dev/null 2>&1; then
  echo "[smoke] FAIL (infra): tdmd validate exited non-zero on the smoke config." >&2
  echo "[smoke]   re-run manually: ${TDMD_BIN} validate ${CONFIG}" >&2
  exit 2
fi

echo "[smoke] step 2/5: tdmd explain --zoning"
if ! "${TDMD_BIN}" explain --zoning "${CONFIG}" > "${ZONING_OUT}" 2>&1; then
  echo "[smoke] FAIL (infra): tdmd explain --zoning exited non-zero." >&2
  echo "[smoke]   see ${ZONING_OUT}" >&2
  tail -20 "${ZONING_OUT}" >&2 || true
  exit 2
fi

# Structural check on the explain --zoning output. Keeps the harness
# insensitive to whitespace reflow while still catching the hardest
# regressions (wrong scheme, missing N_min, missing canonical order).
for needle in 'TDMD Zoning Plan' 'Scheme:' 'N_min per rank:' 'Canonical order:' 'Advisories:'; do
  if ! grep -Fq "${needle}" "${ZONING_OUT}"; then
    echo "[smoke] FAIL (infra): explain --zoning missing '${needle}' line." >&2
    cat "${ZONING_OUT}" >&2
    exit 2
  fi
done

# ---------------------------------------------------------------------------
# Zoning golden — only the load-bearing fields (scheme, zones, N_min, n_opt,
# canonical order length, advisories) are compared. Box / cutoff / skin
# lines drift with the committed T4 assets and would make the golden
# brittle; the §3.4 decision-tree output is what we actually want to pin.
# The awk range stops at "Caveats:" so the static preamble isn't compared.
# ---------------------------------------------------------------------------
ZONING_KEYS="${WORKDIR}/zoning_keys.txt"
awk '/^Scheme:/,/^Caveats:/ { if ($0 !~ /^Caveats:/) print }' \
  "${ZONING_OUT}" > "${ZONING_KEYS}"

if [[ "${TDMD_UPDATE_GOLDENS:-0}" == "1" ]]; then
  cp "${ZONING_KEYS}" "${ZONING_GOLDEN}"
  echo "[smoke] zoning golden updated: ${ZONING_GOLDEN}"
elif [[ -f "${ZONING_GOLDEN}" ]]; then
  if ! diff -u "${ZONING_GOLDEN}" "${ZONING_KEYS}"; then
    echo "[smoke] FAIL (plan regression): explain --zoning diverges from golden." >&2
    echo "[smoke]   if the change is intentional, regenerate with:" >&2
    echo "[smoke]     TDMD_UPDATE_GOLDENS=1 $0 --tdmd ${TDMD_BIN}" >&2
    exit 1
  fi
else
  echo "[smoke] FAIL (infra): zoning golden missing at ${ZONING_GOLDEN}." >&2
  echo "[smoke]   generate it on first bring-up with TDMD_UPDATE_GOLDENS=1 $0" >&2
  exit 2
fi

echo "[smoke] step 3/5: tdmd run --quiet --thermo --timing"
if ! "${TDMD_BIN}" run \
     --quiet \
     --thermo "${THERMO}" \
     --timing \
     "${CONFIG}" 2> "${TIMING_ERR}"; then
  echo "[smoke] FAIL (infra): tdmd run exited non-zero." >&2
  echo "[smoke]   stderr excerpt:" >&2
  tail -20 "${TIMING_ERR}" >&2 || true
  exit 2
fi

# ---------------------------------------------------------------------------
# Thermo golden diff (step 4/5)
# ---------------------------------------------------------------------------
echo "[smoke] step 4/5: thermo golden diff"

if ! grep -q '^# step temp pe ke etotal press' "${THERMO}"; then
  echo "[smoke] FAIL (infra): thermo header missing or mismatched in ${THERMO}" >&2
  head -3 "${THERMO}" >&2 || true
  exit 2
fi

actual_rows="$(wc -l < "${THERMO}")"
if [[ "${actual_rows}" -ne 12 ]]; then
  echo "[smoke] FAIL (infra): expected 12 thermo lines (1 header + 11 data), got ${actual_rows}" >&2
  exit 2
fi

if [[ "${TDMD_UPDATE_GOLDENS:-0}" == "1" ]]; then
  cp "${THERMO}" "${THERMO_GOLDEN}"
  echo "[smoke] thermo golden updated: ${THERMO_GOLDEN}"
elif [[ -f "${THERMO_GOLDEN}" ]]; then
  if ! diff -u "${THERMO_GOLDEN}" "${THERMO}"; then
    echo "[smoke] FAIL (physics): thermo diverges from golden." >&2
    echo "[smoke]   if the change is intentional, regenerate with:" >&2
    echo "[smoke]     TDMD_UPDATE_GOLDENS=1 $0 --tdmd ${TDMD_BIN}" >&2
    exit 1
  fi
else
  echo "[smoke] FAIL (infra): thermo golden missing at ${THERMO_GOLDEN}." >&2
  echo "[smoke]   generate it on first bring-up with TDMD_UPDATE_GOLDENS=1 $0" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Neigh time directional (step 5/5) — with skin=0.05 Å, the displacement
# tracker MUST fire at least one rebuild in 10 steps. If Neigh's %total is
# ≤ 5 %, either the rebuild path bypassed the tracker or the skin drifted.
# ---------------------------------------------------------------------------
echo "[smoke] step 5/5: neighbor-rebuild directional check"

BREAKDOWN_HEADER='Section |  min time  |  avg time  |  max time  |%varavg| %total'
if ! grep -Fq "${BREAKDOWN_HEADER}" "${TIMING_ERR}"; then
  echo "[smoke] FAIL (infra): LAMMPS breakdown header missing from --timing stderr." >&2
  tail -20 "${TIMING_ERR}" >&2
  exit 2
fi

neigh_pct="$(grep -E '^Neigh +\|' "${TIMING_ERR}" | awk -F'|' '{gsub(/ /, "", $6); print $6}')"
if [[ -z "${neigh_pct}" ]]; then
  echo "[smoke] FAIL (infra): could not extract Neigh %total from timing breakdown." >&2
  tail -20 "${TIMING_ERR}" >&2
  exit 2
fi

# awk floating-point comparison — bash doesn't do reals natively.
if ! awk -v v="${neigh_pct}" 'BEGIN { exit !(v > 5.0) }'; then
  echo "[smoke] FAIL (neighbor): Neigh %total = ${neigh_pct} ≤ 5 — rebuild did not fire." >&2
  echo "[smoke]   with skin=0.05 Å on the T4 300 K starting config, at least one" >&2
  echo "[smoke]   mid-run rebuild must occur. Check T3.8 displacement tracker wiring." >&2
  tail -20 "${TIMING_ERR}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Wall-time budget (enforced last so diagnostics above still run)
# ---------------------------------------------------------------------------
elapsed=$SECONDS
echo "[smoke] elapsed:    ${elapsed}s"
echo "[smoke] Neigh %:    ${neigh_pct}"
if [[ "${elapsed}" -gt "${BUDGET_SEC}" ]]; then
  echo "[smoke] FAIL (performance): smoke took ${elapsed}s > budget ${BUDGET_SEC}s." >&2
  exit 3
fi

echo "[smoke] PASS — zoning rationale matches, 10-step EAM thermo matches golden,"
echo "       neighbor rebuild fired mid-run (Neigh ${neigh_pct}% of total)."
