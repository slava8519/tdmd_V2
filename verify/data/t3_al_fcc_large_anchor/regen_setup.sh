#!/usr/bin/env bash
# T3 anchor-test — regenerate the 10^6-atom Al FCC setup.data via LAMMPS.
#
# Invoked by the T5.11 harness (anchor_test_runner) before the first TDMD
# run in a fresh workspace. Idempotent — a second invocation recognises an
# existing `setup.data` and exits 0; pass --force to overwrite.
#
# Why this script instead of an LFS blob:
#   T5.10 deferred committing setup.data.xz (~24 MiB compressed, ~95 MiB
#   uncompressed) in favour of the T1-precedent "regenerate at harness
#   run-time". Keeps the repo git-LFS-free; costs ~30 seconds of LAMMPS
#   CPU per fresh CI workspace. See ../../benchmarks/t3_al_fcc_large_anchor/README.md
#   §"Files in this directory".
#
# Usage:
#   ./regen_setup.sh --lammps <path-to-lmp> [--force]
#   ./regen_setup.sh --help

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
setup_data="${script_dir}/setup.data"
lammps_script="${script_dir}/../../benchmarks/t3_al_fcc_large_anchor/lammps_script.in"
lammps_bin=""
force=0

usage() {
  cat <<EOF
Usage: $0 --lammps <path-to-lmp> [--force]

Regenerates verify/data/t3_al_fcc_large_anchor/setup.data by running the
T3 LAMMPS script once. Produces a 10^6-atom Al FCC lattice at 300 K with
initial velocities, ready for TDMD ingest.

Options:
  --lammps <path>   Path to the LAMMPS binary (required unless LAMMPS_BIN
                    is set in the environment).
  --force           Overwrite an existing setup.data.
  -h, --help        Show this help and exit.

Exit codes:
  0  success (file produced or already-present when --force omitted)
  1  argument / environment error
  2  LAMMPS run failed
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lammps)
      lammps_bin="$2"
      shift 2
      ;;
    --force)
      force=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${lammps_bin}" ]]; then
  lammps_bin="${LAMMPS_BIN:-}"
fi
if [[ -z "${lammps_bin}" ]]; then
  echo "error: --lammps <path> (or LAMMPS_BIN env var) is required" >&2
  exit 1
fi
if [[ ! -x "${lammps_bin}" ]]; then
  echo "error: '${lammps_bin}' is not an executable LAMMPS binary" >&2
  exit 1
fi
if [[ ! -f "${lammps_script}" ]]; then
  echo "error: lammps_script.in not found at '${lammps_script}'" >&2
  exit 1
fi

if [[ -f "${setup_data}" && "${force}" -eq 0 ]]; then
  echo "setup.data already exists at ${setup_data} — pass --force to overwrite"
  exit 0
fi

echo "running LAMMPS to produce setup.data ..."
echo "  binary: ${lammps_bin}"
echo "  script: ${lammps_script}"
echo "  output: ${setup_data}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

# The LAMMPS script writes setup.data (via write_data) and then proceeds
# to run 1000 MD steps. For regen purposes we only need the first
# write_data; intercept by pointing workdir at tmp_dir and moving only
# the post-write_data artefact out.
if ! "${lammps_bin}" \
    -in "${lammps_script}" \
    -var workdir "${tmp_dir}" \
    -log "${tmp_dir}/regen.log" \
    >"${tmp_dir}/regen.stdout" 2>"${tmp_dir}/regen.stderr"; then
  echo "error: LAMMPS run failed — see ${tmp_dir}/regen.{stdout,stderr,log}" >&2
  exit 2
fi

if [[ ! -f "${tmp_dir}/setup.data" ]]; then
  echo "error: LAMMPS completed but setup.data was not produced in ${tmp_dir}" >&2
  exit 2
fi

mv -f "${tmp_dir}/setup.data" "${setup_data}"
echo "ok: wrote $(stat -c%s "${setup_data}") bytes to ${setup_data}"
