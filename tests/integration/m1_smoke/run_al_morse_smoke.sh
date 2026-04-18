#!/usr/bin/env bash
#
# M1 integration smoke — runs the full TDMD M1 pipeline end-to-end against a
# committed 500-atom Al FCC Morse NVE scenario, compares the 10-step thermo
# output to a golden trace (tests/integration/m1_smoke/al_fcc_500_10steps_golden.txt),
# and exits non-zero if anything diverges.
#
# Exec pack: docs/development/m1_execution_pack.md §T1.12.
# Spec: master spec §14 M1 gate; verify/SPEC.md §7 (harness philosophy).
#
# Flags / env:
#   --tdmd <path>                Path to the `tdmd` binary (required if not
#                                on $PATH; $TDMD_CLI_BIN is also honoured).
#   --keep-workdir               Don't rm the tmp workdir on success.
#   TDMD_UPDATE_GOLDENS=1        Overwrite the golden instead of comparing.
#                                Commit only after a Validation Engineer review.
#
# Exit codes:
#   0   smoke green (thermo matches golden byte-for-byte).
#   1   golden diff or pipeline failure.
#   2   harness setup error (missing binary, bad invocation).

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SMOKE_DIR}/smoke_config.yaml.template"
GOLDEN="${SMOKE_DIR}/al_fcc_500_10steps_golden.txt"
ATOMS="${SMOKE_DIR}/setup.data"

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
      sed -n '3,26p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${TDMD_BIN}" ]]; then
  if command -v tdmd >/dev/null 2>&1; then
    TDMD_BIN="$(command -v tdmd)"
  else
    echo "error: tdmd binary not found (pass --tdmd or set TDMD_CLI_BIN)" >&2
    exit 2
  fi
fi

if [[ ! -x "${TDMD_BIN}" ]]; then
  echo "error: tdmd binary not executable: ${TDMD_BIN}" >&2
  exit 2
fi

for p in "${TEMPLATE}" "${ATOMS}"; do
  if [[ ! -f "${p}" ]]; then
    echo "error: required file missing: ${p}" >&2
    exit 2
  fi
done

# ---------------------------------------------------------------------------
# Workdir + config materialisation
# ---------------------------------------------------------------------------
WORKDIR="$(mktemp -d -t tdmd_m1_smoke.XXXXXX)"
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

# Substitute {{ATOMS_PATH}} with the absolute path to the committed setup.data.
# Use a pipe-delimiter to avoid collision with '/' in paths.
sed "s|{{ATOMS_PATH}}|${ATOMS}|g" "${TEMPLATE}" > "${CONFIG}"

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
echo "[smoke] tdmd binary: ${TDMD_BIN}"
echo "[smoke] workdir:     ${WORKDIR}"

echo "[smoke] step 1/3: tdmd validate"
"${TDMD_BIN}" validate "${CONFIG}" >/dev/null

echo "[smoke] step 2/3: tdmd run (10 steps)"
"${TDMD_BIN}" run --quiet --thermo "${THERMO}" "${CONFIG}"

# ---------------------------------------------------------------------------
# Sanity: thermo header present and expected columns emitted.
# ---------------------------------------------------------------------------
if ! grep -q '^# step temp pe ke etotal press' "${THERMO}"; then
  echo "[smoke] FAIL: thermo header missing or mismatched in ${THERMO}" >&2
  head -3 "${THERMO}" >&2 || true
  exit 1
fi

# 1 header + 11 data rows (steps 0..10).
actual_rows="$(wc -l < "${THERMO}")"
if [[ "${actual_rows}" -ne 12 ]]; then
  echo "[smoke] FAIL: expected 12 thermo lines (1 header + 11 data), got ${actual_rows}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Golden comparison (or regeneration under TDMD_UPDATE_GOLDENS=1).
# ---------------------------------------------------------------------------
echo "[smoke] step 3/3: compare vs golden"
if [[ "${TDMD_UPDATE_GOLDENS:-0}" == "1" ]]; then
  cp "${THERMO}" "${GOLDEN}"
  echo "[smoke] golden updated: ${GOLDEN}"
  echo "[smoke] (review the diff in 'git diff' and commit only after a Validation Engineer sign-off)"
  exit 0
fi

if ! diff -u "${GOLDEN}" "${THERMO}"; then
  echo "[smoke] FAIL: thermo output diverges from golden." >&2
  echo "[smoke] If the divergence is intentional (e.g. unit conversion fix)," >&2
  echo "[smoke] regenerate with: TDMD_UPDATE_GOLDENS=1 $0" >&2
  exit 1
fi

echo "[smoke] PASS — 10-step Al FCC Morse NVE matches golden byte-for-byte."
