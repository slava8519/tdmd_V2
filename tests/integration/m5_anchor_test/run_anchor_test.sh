#!/usr/bin/env bash
# M5 anchor-test driver — local-only slow tier (not wired into CI).
#
# Wraps `python3 -m verify.harness.anchor_test_runner` so contributors
# don't need to remember the flag set. Honours:
#
#   TDMD_BIN        path to the tdmd binary (default: build/tdmd)
#   LAMMPS_BIN      path to LAMMPS (only consulted on a fresh workspace
#                   where setup.data needs regeneration)
#   T3_RANKS        space-separated list overriding checks.yaml
#                   (e.g. "T3_RANKS='1 2 4'" for a faster local smoke)
#   T3_FORCE_PROBE  non-empty ⇒ bypass 24h hardware probe cache
#
# Exit codes match the runner itself (0=GREEN, 1=YELLOW, 2=RED,
# 3=infrastructure error). See README.md for the full contract.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." &>/dev/null && pwd)"

tdmd_bin="${TDMD_BIN:-${repo_root}/build/tdmd}"
report_path="${T3_REPORT_PATH:-${repo_root}/build/t3_anchor_report.json}"
workdir="${T3_WORKDIR:-${repo_root}/build/t3_anchor_workdir}"

if [[ ! -x "${tdmd_bin}" ]]; then
  echo "error: tdmd binary not executable at '${tdmd_bin}'" >&2
  echo "       build with: cmake -S . -B build -DTDMD_ENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release" >&2
  echo "                   cmake --build build --target tdmd -j" >&2
  exit 3
fi

cmd=(python3 -m verify.harness.anchor_test_runner
     --tdmd-bin "${tdmd_bin}"
     --output "${report_path}"
     --workdir "${workdir}")

if [[ -n "${LAMMPS_BIN:-}" ]]; then
  cmd+=(--lammps-bin "${LAMMPS_BIN}")
fi
if [[ -n "${T3_RANKS:-}" ]]; then
  # T3_RANKS is a space-separated list — word-split into an array so the
  # resulting argv is a sequence of individual integers rather than one
  # quoted blob.
  read -r -a t3_ranks_array <<<"${T3_RANKS}"
  cmd+=(--ranks "${t3_ranks_array[@]}")
fi
if [[ -n "${T3_FORCE_PROBE:-}" ]]; then
  cmd+=(--force-probe)
fi

echo "[m5_anchor_test] cwd=${repo_root}"
echo "[m5_anchor_test] ${cmd[*]}"
cd "${repo_root}"
exec "${cmd[@]}"
