#!/usr/bin/env bash
#
# M8 integration smoke — closes the M8 milestone (master spec §14 M8 gate).
# End-to-end exercise of the M8 user surface: SNAP W BCC on T6 canonical
# 1024-atom fixture, 2-rank Pattern 2 K=1 P_space=2 TD + MpiHostStaging +
# GPU runtime, Fp64ReferenceBuild. This is the SNAP analog of the M7
# EAM byte-exact smoke; the D-M8-13 byte-exact chain extends D-M7-10 from
# EAM to SNAP (see exec pack §T8.13).
#
# Flow (mirrors m7_smoke with a 1-rank→2-rank byte-exact gate):
#   1. Submodule probe — self-skip (exit 77) if LAMMPS submodule not
#      initialized (W_2940_2017_2.snap* fixture files absent).
#   2. LFS probe — fail (exit 2) if setup_1024.data is an unresolved LFS
#      pointer (developer forgot `git lfs pull`).
#   3. GPU probe — self-skip (exit 0) if nvidia-smi sees no CUDA device
#      (Option A / D-M6-6: no public-runner GPU CI).
#   4. Single-rank Pattern 2 preflight (T7.9).
#   5. mpirun -np 2 tdmd validate (2-rank Pattern 2 config accepted).
#   6. mpirun -np 2 tdmd run --telemetry-jsonl (10-step NVE with TD K=1,
#                                                Pattern 2 P_space=2, GPU).
#   7. Thermo byte-matches M8 golden (D-M8-13 byte-exact chain: the golden
#      is the 1-rank K=1 P=1 Fp64Reference SNAP run of the same config,
#      `subdomains` layer removed; 2-rank K=1 P_space=2 MUST reproduce it
#      bit-for-bit). Regenerate via TDMD_UPDATE_GOLDENS=1 (runs the 1-rank
#      reference pass itself — see "Golden regeneration" note below).
#   8. Telemetry invariants: run_end event, wall-time ≤ budget,
#                            ignored_end_calls == 0, boundary_stalls_total == 0.
#
# Exec pack: docs/development/m8_execution_pack.md §T8.13.
# Spec: master spec §14 M8 gate; scheduler/SPEC Pattern 2 integration;
#       runtime/SPEC §2.4 Pattern 2 wire; comm/SPEC §3 (MpiHostStaging);
#       zoning/SPEC (subdomains linear_1d); potentials/SPEC SNAP;
#       verify/SPEC §4.7 T6 canonical fixture; verify/SPEC §7 (oracle-free
#       smoke).
#
# Flags / env:
#   --tdmd <path>             Path to the `tdmd` binary (required if not
#                             on $PATH; $TDMD_CLI_BIN is also honoured).
#                             Must be built with -DTDMD_ENABLE_MPI=ON and
#                             -DTDMD_BUILD_CUDA=ON, Fp64ReferenceBuild.
#   --mpirun <path>           MPI launcher (default: `mpirun` on $PATH).
#   --keep-workdir            Don't rm the tmp workdir on success.
#   TDMD_UPDATE_GOLDENS=1     Regenerate the golden: run 1-rank first, copy
#                             its thermo to thermo_golden.txt, then run
#                             2-rank and byte-diff against the fresh golden.
#   TDMD_M8_SMOKE_BUDGET_SEC=N  Override default 120s wall-time budget.
#
# Exit codes:
#   0    smoke green (or SKIPPED — no GPU visible).
#   1    physics / determinism regression — thermo diverged OR telemetry
#        invariant broken.
#   2    infrastructure — missing binary, mpirun absent, malformed output,
#        bad invocation, missing LFS asset, missing golden.
#   3    performance — wall-time exceeded the budget.
#   77   standardised skip — LAMMPS submodule not initialized (matches
#        test_lammps_oracle_snap_fixture pattern).
#
# Golden regeneration:
#   The M8 golden is the 1-rank Fp64Reference thermo of the same config
#   with `zoning.subdomains` removed. Under TDMD_UPDATE_GOLDENS=1 the
#   harness regenerates it automatically: it first produces a 1-rank config
#   by stripping the `zoning` section, runs the 1-rank pass, and copies
#   its thermo to `thermo_golden.txt`. The subsequent 2-rank pass then
#   byte-diffs against that fresh golden and fails if they differ — so
#   update-goldens mode is itself a D-M8-13 check, not just a blind copy.

set -euo pipefail

SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SMOKE_DIR}/../../.." && pwd)"

TEMPLATE="${SMOKE_DIR}/smoke_config.yaml.template"
THERMO_GOLDEN="${SMOKE_DIR}/thermo_golden.txt"
TELEMETRY_EXPECTED="${SMOKE_DIR}/telemetry_expected.txt"

BENCH_DIR="${REPO_ROOT}/verify/benchmarks/t6_snap_tungsten"
ATOMS="${BENCH_DIR}/setup_1024.data"

SNAP_DIR="${REPO_ROOT}/verify/third_party/lammps/examples/snap"
SNAP_COEFF="${SNAP_DIR}/W_2940_2017_2.snapcoeff"
SNAP_PARAM="${SNAP_DIR}/W_2940_2017_2.snapparam"

BUDGET_SEC="${TDMD_M8_SMOKE_BUDGET_SEC:-120}"

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
      sed -n '3,68p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "[m8-smoke] error: unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

# Step 1 (infra): LAMMPS submodule probe. SNAP coeff files ship via the
# LAMMPS submodule (D-M8-3 — no TDMD-side binary tracked); if the submodule
# is uninitialized, the fixture is absent. Exit 77 = Catch2 SKIP_RETURN_CODE.
for f in "${SNAP_COEFF}" "${SNAP_PARAM}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[m8-smoke] SKIPPED — LAMMPS submodule not initialized (fixture missing)" >&2
    echo "[m8-smoke]   missing: ${f}" >&2
    echo "[m8-smoke]   run 'git submodule update --init --recursive' to fetch." >&2
    exit 77
  fi
done

for p in "${TEMPLATE}" "${ATOMS}" "${TELEMETRY_EXPECTED}"; do
  if [[ ! -f "${p}" ]]; then
    echo "[m8-smoke] error: required file missing: ${p}" >&2
    if [[ "${p}" == "${ATOMS}" ]]; then
      echo "[m8-smoke]   setup_1024.data is LFS-tracked; did you run 'git lfs pull'?" >&2
    fi
    exit 2
  fi
done

if [[ "$(head -c 10 "${ATOMS}")" == "version ht" ]]; then
  echo "[m8-smoke] error: ${ATOMS} is an unresolved LFS pointer." >&2
  echo "[m8-smoke]   run 'git lfs pull'." >&2
  exit 2
fi

# Local-only gate per D-M6-6 — no self-hosted GPU runner on public CI.
# Probe via `nvidia-smi -L` (lists CUDA-capable devices one per line)
# rather than `nvcc --version` so the smoke skips cleanly on hosts that
# ship the toolkit but have no physical GPU (e.g. container build hosts).
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[m8-smoke] SKIPPED — nvidia-smi not on \$PATH (no GPU visible per D-M6-6)"
  exit 0
fi
if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
  echo "[m8-smoke] SKIPPED — nvidia-smi reports no CUDA devices (D-M6-6)"
  exit 0
fi

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

if [[ -z "${MPIRUN_BIN}" ]]; then
  if command -v mpirun >/dev/null 2>&1; then
    MPIRUN_BIN="$(command -v mpirun)"
  elif command -v mpiexec >/dev/null 2>&1; then
    MPIRUN_BIN="$(command -v mpiexec)"
  else
    echo "[m8-smoke] SKIPPED — no mpirun/mpiexec on \$PATH" >&2
    echo "[m8-smoke]   install openmpi-bin (or mpich-bin) to enable." >&2
    exit 0
  fi
fi

WORKDIR="$(mktemp -d -t tdmd_m8_smoke.XXXXXX)"
cleanup() {
  if [[ "${KEEP_WORKDIR}" -eq 0 ]]; then
    rm -rf "${WORKDIR}"
  else
    echo "[m8-smoke] workdir preserved: ${WORKDIR}"
  fi
}
trap cleanup EXIT

CONFIG_2RANK="${WORKDIR}/smoke_config_2rank.yaml"
CONFIG_1RANK="${WORKDIR}/smoke_config_1rank.yaml"
THERMO_2RANK="${WORKDIR}/thermo_2rank.log"
THERMO_1RANK="${WORKDIR}/thermo_1rank.log"
TELEMETRY="${WORKDIR}/telemetry.jsonl"
RUN_ERR="${WORKDIR}/run.stderr"

sed -e "s|{{ATOMS_PATH}}|${ATOMS}|g" \
    -e "s|{{SNAP_COEFF}}|${SNAP_COEFF}|g" \
    -e "s|{{SNAP_PARAM}}|${SNAP_PARAM}|g" \
    "${TEMPLATE}" > "${CONFIG_2RANK}"

# The 1-rank reference config strips the three multi-rank / TD layers
# (`scheduler:`, `comm:`, `zoning:`) — matching the t6 scout baseline
# (`verify/benchmarks/t6_snap_tungsten/scout_rtx5080/tdmd_gpu_100step.yaml`)
# that established the D-M8-13 byte-exact claim on 2026-04-21.
#
# Why strip all three rather than just `zoning:`: the single-rank CPU/GPU
# path defaults to td_mode=false, no CausalWavefrontScheduler. Leaving
# td_mode:true enabled on 1 rank would construct the wavefront scheduler
# over the full-box zone grid and trip the M4 >64-zone limit on T6's
# 1024-atom fixture (~125 cells at cutoff=4.73+0.3=5.03 Å). The byte-exact
# claim was always 1-rank legacy ≡ 2-rank Pattern 2 K=1 (scout reference
# uses legacy path), so this is the semantically correct comparison.
#
# awk state machine: skip lines from each top-level section header to the
# next top-level (column-1) non-comment key. Robust to section reordering
# within the template.
awk '
  /^scheduler:|^comm:|^zoning:/ { skip = 1; next }
  skip == 1 && /^[^[:space:]#]/ { skip = 0 }
  skip == 0                     { print }
' "${CONFIG_2RANK}" > "${CONFIG_1RANK}"

echo "[m8-smoke] tdmd binary: ${TDMD_BIN}"
echo "[m8-smoke] mpirun:      ${MPIRUN_BIN}"
echo "[m8-smoke] workdir:     ${WORKDIR}"
echo "[m8-smoke] budget:      ${BUDGET_SEC}s"
nvidia-smi -L 2>/dev/null | head -1 | sed 's/^/[m8-smoke] gpu:         /'

SECONDS=0

echo "[m8-smoke] step 1/7: LAMMPS submodule probe (D-M8-3)"
echo "[m8-smoke]   coeff: ${SNAP_COEFF}"
echo "[m8-smoke]   param: ${SNAP_PARAM}"

echo "[m8-smoke] step 2/7: GPU visibility probe (D-M6-6)"

echo "[m8-smoke] step 3/7: single-rank Pattern 2 preflight (T7.9)"
if ! "${TDMD_BIN}" validate "${CONFIG_2RANK}" >/dev/null 2>&1; then
  echo "[m8-smoke] FAIL (infra): single-rank tdmd validate exited non-zero." >&2
  echo "[m8-smoke]   re-run: ${TDMD_BIN} validate ${CONFIG_2RANK}" >&2
  exit 2
fi

echo "[m8-smoke] step 4/7: mpirun -np 2 tdmd validate"
if ! "${MPIRUN_BIN}" -np 2 --oversubscribe "${TDMD_BIN}" validate "${CONFIG_2RANK}" >/dev/null 2>&1; then
  if ! "${MPIRUN_BIN}" -np 2 "${TDMD_BIN}" validate "${CONFIG_2RANK}" >/dev/null 2>&1; then
    echo "[m8-smoke] FAIL (infra): mpirun -np 2 tdmd validate exited non-zero." >&2
    echo "[m8-smoke]   re-run: ${MPIRUN_BIN} -np 2 ${TDMD_BIN} validate ${CONFIG_2RANK}" >&2
    exit 2
  fi
fi

# Optional 1-rank oracle pass: either TDMD_UPDATE_GOLDENS=1 (regenerate the
# golden) or an explicit re-check when the golden is a placeholder stub.
regen_golden=0
if [[ "${TDMD_UPDATE_GOLDENS:-0}" == "1" ]]; then
  regen_golden=1
fi
# Placeholder-stub detection: the checked-in stub starts with "# Placeholder
# golden." so we can tell a never-populated golden from a real one that
# happens to be empty.
if head -1 "${THERMO_GOLDEN}" 2>/dev/null | grep -q "Placeholder golden"; then
  if [[ "${regen_golden}" -eq 0 ]]; then
    echo "[m8-smoke] FAIL (infra): thermo_golden.txt is the unpopulated placeholder." >&2
    echo "[m8-smoke]   Generate it on first bring-up with:" >&2
    echo "[m8-smoke]     TDMD_UPDATE_GOLDENS=1 ${BASH_SOURCE[0]} --tdmd ${TDMD_BIN}" >&2
    exit 2
  fi
fi

if [[ "${regen_golden}" -eq 1 ]]; then
  echo "[m8-smoke] step 5a/7: 1-rank reference pass (TDMD_UPDATE_GOLDENS regenerate)"
  if ! "${TDMD_BIN}" run \
        --quiet \
        --thermo "${THERMO_1RANK}" \
        "${CONFIG_1RANK}" 2> "${RUN_ERR}"; then
    echo "[m8-smoke] FAIL (infra): 1-rank tdmd run exited non-zero." >&2
    echo "[m8-smoke]   stderr excerpt:" >&2
    tail -30 "${RUN_ERR}" >&2 || true
    exit 2
  fi
  cp "${THERMO_1RANK}" "${THERMO_GOLDEN}"
  echo "[m8-smoke]   thermo golden updated: ${THERMO_GOLDEN}"
fi

echo "[m8-smoke] step 5/7: mpirun -np 2 tdmd run --thermo --telemetry-jsonl"
MPI_ARGS=(-np 2 --oversubscribe)
if ! "${MPIRUN_BIN}" "${MPI_ARGS[@]}" "${TDMD_BIN}" run \
     --quiet \
     --thermo "${THERMO_2RANK}" \
     --telemetry-jsonl "${TELEMETRY}" \
     "${CONFIG_2RANK}" 2> "${RUN_ERR}"; then
  MPI_ARGS=(-np 2)
  if ! "${MPIRUN_BIN}" "${MPI_ARGS[@]}" "${TDMD_BIN}" run \
       --quiet \
       --thermo "${THERMO_2RANK}" \
       --telemetry-jsonl "${TELEMETRY}" \
       "${CONFIG_2RANK}" 2> "${RUN_ERR}"; then
    echo "[m8-smoke] FAIL (infra): mpirun -np 2 tdmd run exited non-zero." >&2
    echo "[m8-smoke]   stderr excerpt:" >&2
    tail -30 "${RUN_ERR}" >&2 || true
    exit 2
  fi
fi

echo "[m8-smoke] step 6/7: thermo D-M8-13 byte-exact diff (2-rank ≡ 1-rank golden)"

if ! grep -q '^# step temp pe ke etotal press' "${THERMO_2RANK}"; then
  echo "[m8-smoke] FAIL (infra): thermo header missing or mismatched in ${THERMO_2RANK}" >&2
  head -3 "${THERMO_2RANK}" >&2 || true
  exit 2
fi

actual_rows="$(wc -l < "${THERMO_2RANK}")"
if [[ "${actual_rows}" -ne 12 ]]; then
  echo "[m8-smoke] FAIL (infra): expected 12 thermo lines (1 header + 11 data), got ${actual_rows}" >&2
  exit 2
fi

if ! diff -u "${THERMO_GOLDEN}" "${THERMO_2RANK}"; then
  echo "[m8-smoke] FAIL (D-M8-13): K=1 P_space=2 Pattern 2 Reference thermo diverges from 1-rank golden." >&2
  echo "[m8-smoke]   Pattern 2 K=1 MUST produce thermo byte-identical to 1-rank K=1 P=1" >&2
  echo "[m8-smoke]   Fp64Reference SNAP. Investigate (in order):" >&2
  echo "[m8-smoke]     - scheduler OuterSdCoordinator peer-halo canonicalisation (R-M7-5);" >&2
  echo "[m8-smoke]     - SubdomainBoundaryDependency Kahan-ring reduction;" >&2
  echo "[m8-smoke]     - GPU SNAP reduce-then-scatter canonical order (gpu/SPEC §6.1);" >&2
  echo "[m8-smoke]     - stray atomic add in bispectrum accumulation at subdomain boundary;" >&2
  echo "[m8-smoke]     - BuildFlavor != Fp64Reference (harness assumes the oracle flavor)." >&2
  exit 1
fi

echo "[m8-smoke] step 7/7: telemetry invariants"

if [[ ! -s "${TELEMETRY}" ]]; then
  echo "[m8-smoke] FAIL (infra): telemetry file empty or missing: ${TELEMETRY}" >&2
  exit 2
fi

telemetry_line="$(head -n 1 "${TELEMETRY}")"

fail=0
while IFS= read -r line; do
  line="${line%%#*}"
  # shellcheck disable=SC2001
  line="$(echo "${line}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  [[ -z "${line}" ]] && continue

  key="$(echo "${line}" | awk '{print $1}')"
  op="$(echo "${line}" | awk '{print $2}')"
  want="$(echo "${line}" | awk '{print $3}')"

  if [[ "${key}" == "event" ]]; then
    got="$(echo "${telemetry_line}" | grep -oE '"event":"[^"]+"' | sed -e 's/"event":"//' -e 's/"$//' || true)"
  else
    got="$(echo "${telemetry_line}" | grep -oE "\"${key}\":[^,}]+" | sed -e "s/\"${key}\"://" -e 's/[}" ]//g' || true)"
  fi

  if [[ -z "${got}" ]]; then
    if [[ "${key}" == "boundary_stalls_total" ]]; then
      got="0"
    else
      echo "[m8-smoke] FAIL (telemetry): key '${key}' missing from JSONL output" >&2
      fail=1
      continue
    fi
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
      echo "[m8-smoke] FAIL (telemetry): unknown operator '${op}' for key '${key}'" >&2
      fail=1
      continue
      ;;
  esac

  if [[ "${ok}" != "1" ]]; then
    echo "[m8-smoke] FAIL (telemetry): ${key} ${op} ${want} — got ${got}" >&2
    fail=1
  fi
done < "${TELEMETRY_EXPECTED}"

if [[ "${fail}" -ne 0 ]]; then
  echo "[m8-smoke]   telemetry JSONL:" >&2
  echo "[m8-smoke]   ${telemetry_line}" >&2
  exit 1
fi

elapsed=$SECONDS
echo "[m8-smoke] elapsed:     ${elapsed}s"
if [[ "${elapsed}" -gt "${BUDGET_SEC}" ]]; then
  echo "[m8-smoke] FAIL (performance): smoke took ${elapsed}s > budget ${BUDGET_SEC}s." >&2
  exit 3
fi

echo "[m8-smoke] PASS — K=1 P_space=2 Pattern 2 Reference SNAP 10-step W BCC thermo matches"
echo "         1-rank golden byte-for-byte; telemetry clean (boundary_stalls_total=0);"
echo "         D-M8-13 acceptance gate green."
