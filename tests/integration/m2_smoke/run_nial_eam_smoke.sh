#!/usr/bin/env bash
#
# M2 integration smoke — end-to-end exercise of the M2 user surface:
#
#   1. tdmd validate                      → config accepted
#   2. tdmd explain --perf                → Pattern 1 + Pattern 3 prediction
#   3. tdmd run --timing --telemetry-jsonl → thermo + telemetry emitted
#
# On a committed 864-atom Ni-Al EAM/alloy (Mishin 2004) NVE scenario clipped
# to 10 steps. Pipeline inputs are the existing T4 assets
# (verify/benchmarks/t4_nial_alloy/setup.data + NiAl_Mishin_2004.eam.alloy);
# see tests/integration/m2_smoke/README.md for the reuse rationale.
#
# Exec pack: docs/development/m2_execution_pack.md §T2.13.
# Spec: master spec §14 M2 gate; telemetry/SPEC §4.2 (LAMMPS format);
#       verify/SPEC §7 (harness philosophy — no oracle dependency).
#
# Flags / env:
#   --tdmd <path>             Path to the `tdmd` binary (required if not
#                             on $PATH; $TDMD_CLI_BIN is also honoured).
#   --keep-workdir            Don't rm the tmp workdir on success.
#   TDMD_UPDATE_GOLDENS=1     Overwrite the golden instead of comparing.
#                             Commit only after a Validation Engineer review.
#   TDMD_SMOKE_BUDGET_SEC=N   Override the default 10s wall-time budget.
#
# Exit codes (mirror the convention the user set for this harness):
#   0   smoke green — all checks passed.
#   1   physics regression — thermo log diverges from golden.
#   2   infrastructure — missing binary, malformed output, bad invocation.
#   3   performance — wall-time exceeded the budget.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SMOKE_DIR}/../../.." && pwd)"

TEMPLATE="${SMOKE_DIR}/smoke_config.yaml.template"
GOLDEN="${SMOKE_DIR}/nial_eam_10steps_thermo_golden.txt"
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

# Real file content, not an LFS pointer. If setup.data shows up as ~130 bytes,
# the checkout never pulled the actual blob.
if [[ "$(head -c 10 "${ATOMS}")" == "version ht" ]]; then
  echo "[smoke] error: ${ATOMS} is an unresolved LFS pointer." >&2
  echo "[smoke]   run 'git lfs pull' (or set lfs:true on the CI checkout)." >&2
  exit 2
fi

# Python is used for the JSONL schema check — cheap dependency already
# present on every CI runner and in the dev profile.
if ! command -v python3 >/dev/null 2>&1; then
  echo "[smoke] error: python3 not available — needed for JSONL schema check" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Workdir + config materialisation
# ---------------------------------------------------------------------------
WORKDIR="$(mktemp -d -t tdmd_m2_smoke.XXXXXX)"
cleanup() {
  if [[ "${KEEP_WORKDIR}" -eq 0 ]]; then
    rm -rf "${WORKDIR}"
  else
    echo "[smoke] workdir preserved: ${WORKDIR}"
  fi
}
trap cleanup EXIT

CONFIG="${WORKDIR}/smoke_config.yaml"
THERMO="${WORKDIR}/thermo.log"
EXPLAIN_OUT="${WORKDIR}/explain.txt"
TELEMETRY_ERR="${WORKDIR}/timing.stderr"
TELEMETRY_JSONL="${WORKDIR}/telemetry.jsonl"

# Substitute {{ATOMS_PATH}} / {{EAM_PATH}} with absolute paths. Pipe-delimit
# sed so '/' in paths isn't ambiguous.
sed -e "s|{{ATOMS_PATH}}|${ATOMS}|g" \
    -e "s|{{EAM_PATH}}|${EAM}|g" \
    "${TEMPLATE}" > "${CONFIG}"

# ---------------------------------------------------------------------------
# Pipeline — with wall-time budget enforcement
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

echo "[smoke] step 2/5: tdmd explain --perf"
if ! "${TDMD_BIN}" explain --perf "${CONFIG}" > "${EXPLAIN_OUT}" 2>&1; then
  echo "[smoke] FAIL (infra): tdmd explain --perf exited non-zero." >&2
  echo "[smoke]   see ${EXPLAIN_OUT}" >&2
  tail -20 "${EXPLAIN_OUT}" >&2 || true
  exit 2
fi

# Structural checks on the explain output — Pattern1_TD and Pattern3_SD rows
# must be present, plus the EAM M5 roadmap caveat (see explain_command.cpp
# td_available_milestone()). We do NOT check numerical predictions — the
# model is uncalibrated on CI hardware (SPEC §6, carry-forward to M4+).
if ! grep -q 'Pattern1_TD' "${EXPLAIN_OUT}"; then
  echo "[smoke] FAIL (infra): explain --perf missing 'Pattern1_TD' section." >&2
  cat "${EXPLAIN_OUT}" >&2
  exit 2
fi
if ! grep -q 'Pattern3_SD' "${EXPLAIN_OUT}"; then
  echo "[smoke] FAIL (infra): explain --perf missing 'Pattern3_SD' section." >&2
  cat "${EXPLAIN_OUT}" >&2
  exit 2
fi
if ! grep -q 'M5' "${EXPLAIN_OUT}"; then
  echo "[smoke] FAIL (infra): explain --perf missing the M5 EAM-roadmap caveat." >&2
  cat "${EXPLAIN_OUT}" >&2
  exit 2
fi

echo "[smoke] step 3/5: tdmd run --quiet --thermo --timing --telemetry-jsonl"
if ! "${TDMD_BIN}" run \
     --quiet \
     --thermo "${THERMO}" \
     --timing \
     --telemetry-jsonl "${TELEMETRY_JSONL}" \
     "${CONFIG}" 2> "${TELEMETRY_ERR}"; then
  echo "[smoke] FAIL (infra): tdmd run exited non-zero." >&2
  echo "[smoke]   stderr excerpt:" >&2
  tail -20 "${TELEMETRY_ERR}" >&2 || true
  exit 2
fi

# ---------------------------------------------------------------------------
# Thermo sanity + golden diff  (step 4/5)
# ---------------------------------------------------------------------------
echo "[smoke] step 4/5: thermo golden diff"

if ! grep -q '^# step temp pe ke etotal press' "${THERMO}"; then
  echo "[smoke] FAIL (infra): thermo header missing or mismatched in ${THERMO}" >&2
  head -3 "${THERMO}" >&2 || true
  exit 2
fi

# thermo.every=1 + n_steps=10 → step 0 emitted as initial row + 10 post-step
# rows = 11 data lines, plus 1 header = 12 total.
actual_rows="$(wc -l < "${THERMO}")"
if [[ "${actual_rows}" -ne 12 ]]; then
  echo "[smoke] FAIL (infra): expected 12 thermo lines (1 header + 11 data), got ${actual_rows}" >&2
  exit 2
fi

if [[ "${TDMD_UPDATE_GOLDENS:-0}" == "1" ]]; then
  cp "${THERMO}" "${GOLDEN}"
  echo "[smoke] golden updated: ${GOLDEN}"
  echo "[smoke]   review with 'git diff' and commit only after Validation Engineer sign-off."
  # Still verify telemetry below so an update run catches telemetry regressions too.
elif [[ -f "${GOLDEN}" ]]; then
  if ! diff -u "${GOLDEN}" "${THERMO}"; then
    echo "[smoke] FAIL (physics): thermo diverges from golden." >&2
    echo "[smoke]   if the change is intentional, regenerate with:" >&2
    echo "[smoke]     TDMD_UPDATE_GOLDENS=1 $0 --tdmd ${TDMD_BIN}" >&2
    exit 1
  fi
else
  echo "[smoke] FAIL (infra): golden missing at ${GOLDEN}." >&2
  echo "[smoke]   generate it on first bring-up with TDMD_UPDATE_GOLDENS=1 $0" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Telemetry checks  (step 5/5)
# ---------------------------------------------------------------------------
echo "[smoke] step 5/5: telemetry (JSONL schema + LAMMPS breakdown + directional)"

# Loose JSONL schema — required keys present; extra keys allowed (forward-
# compatible with M3+ per-step streaming).
if ! python3 - "$TELEMETRY_JSONL" <<'PY' ; then
import json
import sys

path = sys.argv[1]
with open(path) as f:
    first = f.readline()
doc = json.loads(first)
required = {"event", "total_wall_sec", "sections", "ignored_end_calls"}
missing = required - set(doc.keys())
if missing:
    print(f"missing required JSONL keys: {sorted(missing)}", file=sys.stderr)
    sys.exit(1)
if doc["event"] != "run_end":
    print(f"unexpected event: {doc['event']!r}", file=sys.stderr)
    sys.exit(1)
if not isinstance(doc["sections"], dict):
    print("sections must be an object", file=sys.stderr)
    sys.exit(1)
if "Pair" not in doc["sections"] or doc["sections"]["Pair"] <= 0:
    print(f"Pair section missing or non-positive: {doc['sections']!r}", file=sys.stderr)
    sys.exit(1)
if doc["total_wall_sec"] <= 0:
    print(f"total_wall_sec non-positive: {doc['total_wall_sec']!r}", file=sys.stderr)
    sys.exit(1)
PY
  echo "[smoke] FAIL (infra): telemetry JSONL failed schema check (see stderr above)." >&2
  echo "[smoke]   JSONL was: $(cat "${TELEMETRY_JSONL}")" >&2
  exit 2
fi

# LAMMPS breakdown header — exact byte match (guards SPEC §4.2 mockup).
BREAKDOWN_HEADER='Section |  min time  |  avg time  |  max time  |%varavg| %total'
if ! grep -Fq "${BREAKDOWN_HEADER}" "${TELEMETRY_ERR}"; then
  echo "[smoke] FAIL (infra): LAMMPS breakdown header missing from --timing stderr." >&2
  echo "[smoke]   stderr excerpt:" >&2
  tail -20 "${TELEMETRY_ERR}" >&2
  exit 2
fi

# All five canonical rows + Total must appear.
for row in 'Pair ' 'Neigh ' 'Comm ' 'Output ' 'Other ' 'Total '; do
  if ! grep -Fq "${row}" "${TELEMETRY_ERR}"; then
    echo "[smoke] FAIL (infra): LAMMPS breakdown missing row '${row}'." >&2
    tail -20 "${TELEMETRY_ERR}" >&2
    exit 2
  fi
done

# Directional — EAM force compute dominates on 864 atoms: Pair / total > 50%.
# Pair can legitimately exceed total_wall_sec by a few % (measurement jitter
# at sub-ms scale) so we clamp the ratio for the assertion.
if ! python3 - "$TELEMETRY_JSONL" <<'PY' ; then
import json
import sys

with open(sys.argv[1]) as f:
    doc = json.loads(f.readline())
pair = doc["sections"].get("Pair", 0.0)
total = max(doc["total_wall_sec"], pair)
if total <= 0:
    print("directional: total<=0", file=sys.stderr); sys.exit(1)
ratio = pair / total
if ratio <= 0.50:
    print(f"directional: Pair/Total = {ratio:.3f} (expected >0.50 for EAM 864 atoms)",
          file=sys.stderr)
    sys.exit(1)
PY
  echo "[smoke] FAIL (infra): Pair/Total directional check failed." >&2
  echo "[smoke]   EAM force compute should dominate on 864 atoms." >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Wall-time budget  (enforced last so we still get the diagnostics above)
# ---------------------------------------------------------------------------
elapsed=$SECONDS
echo "[smoke] elapsed:    ${elapsed}s"
if [[ "${elapsed}" -gt "${BUDGET_SEC}" ]]; then
  echo "[smoke] FAIL (performance): smoke took ${elapsed}s > budget ${BUDGET_SEC}s." >&2
  echo "[smoke]   EAM physics is correct but something got slower — investigate" >&2
  echo "[smoke]   before merging (likely culprits: neighbor-list tuning, cache flip)." >&2
  exit 3
fi

echo "[smoke] PASS — 10-step Ni-Al EAM NVE matches golden, telemetry schema clean."
