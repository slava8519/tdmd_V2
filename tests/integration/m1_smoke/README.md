# M1 integration smoke (`m1_smoke`)

A single-command end-to-end regression test that exercises the complete M1
pipeline (config → data ingest → neighbor build → Morse pair loop → velocity-Verlet
integration → thermo snapshot) and compares the 10-step thermo trace to a
committed golden. It runs in well under a second and is wired into every CI
build, so any change that silently moves M1 physics is caught on the PR.

See also: `docs/development/m1_execution_pack.md` §T1.12,
`docs/specs/verify/SPEC.md` §7 (harness philosophy).

## What it checks

The 500-atom Al FCC system from T1 (`verify/benchmarks/t1_al_morse_500`), clipped
to 10 steps. The smoke is intentionally short and strict:

1. `tdmd validate` accepts the config.
2. `tdmd run --quiet --thermo <log>` runs to completion.
3. The thermo output has the expected header (`# step temp pe ke etotal press`)
   and the expected number of rows (1 header + 11 data rows, steps 0..10).
4. The thermo file matches `al_fcc_500_10steps_golden.txt` **byte-for-byte**.

Step 4 is the meat: a byte-level `diff` against the golden catches any change
to force magnitude, integrator coefficient, unit conversion, output formatter,
neighbor-rebuild policy, or reduction order. There is no tolerance — FP64 on a
single compiler + arch is deterministic, and any drift is either a real
regression or an intentional change that warrants a golden update (review-gated,
see below).

## What it does *not* check

Cross-validation against LAMMPS is the job of the T1 differential harness
(`verify/t1/run_differential.py`, wired into CI as `differential-t1`). That
harness is oracle-gated and skips on public CI because the LAMMPS submodule is
not fetched there. The smoke complements it: no oracle, always runs.

Longer-horizon physics (NVE drift over 100+ ps, ensemble statistics) lands at
M2 with the full `DifferentialRunner` and `ConservationChecker`.

## Running locally

From repo root, after a CPU build:

```bash
cmake -B build_cpu --preset cpu-only
cmake --build build_cpu --parallel
tests/integration/m1_smoke/run_al_morse_smoke.sh --tdmd build_cpu/src/cli/tdmd
```

Expected output ends with `[smoke] PASS`. Wall-time: ≈0.01 s on a modern
workstation.

You can also set `TDMD_CLI_BIN` and omit `--tdmd`:

```bash
TDMD_CLI_BIN=$PWD/build_cpu/src/cli/tdmd tests/integration/m1_smoke/run_al_morse_smoke.sh
```

## Files in this directory

| File | Role |
|---|---|
| `run_al_morse_smoke.sh` | Driver — generates the config, runs the pipeline, diffs against golden |
| `smoke_config.yaml.template` | TDMD YAML config template (`{{ATOMS_PATH}}` gets substituted at run-time) |
| `setup.data` | Committed LAMMPS `.data` file — initial state (500-atom Al FCC + Gaussian velocities @ 300 K, seed 12345). Produced once by LAMMPS (see Regenerating below) |
| `al_fcc_500_10steps_golden.txt` | Committed reference thermo trace. Byte-matched against the live run |
| `README.md` | This file |

The `setup.data` file is committed rather than regenerated per-run: TDMD CI
does not have LAMMPS available (Option A — `docs/development/m0_execution_pack.md`),
and baking the initial state into the repo removes any run-time dependency on
the oracle. The file is ~52 KB of ASCII and stable — it changes only when the
T1 scenario itself changes.

## Regenerating the golden

The golden is **review-gated**: it is never silently overwritten. When an
intentional change (say, a new unit-conversion factor, a numeric fix in the
integrator) shifts the thermo trace, regenerate it explicitly and commit the
diff under Validation Engineer review:

```bash
TDMD_UPDATE_GOLDENS=1 tests/integration/m1_smoke/run_al_morse_smoke.sh \
  --tdmd build_cpu/src/cli/tdmd
git diff tests/integration/m1_smoke/al_fcc_500_10steps_golden.txt
```

The PR description must explain **why** the trace shifted. A golden update
without rationale is an auto-reject (playbook §5).

## Regenerating `setup.data`

Only needed if the T1 scenario itself (atom count, lattice, seed, initial
temperature) changes. The procedure matches T1:

```bash
WORKDIR=$(mktemp -d)
LD_LIBRARY_PATH=verify/third_party/lammps/install_tdmd/lib \
  verify/third_party/lammps/install_tdmd/bin/lmp \
    -in verify/benchmarks/t1_al_morse_500/lammps_script.in \
    -var workdir "$WORKDIR" \
    -log "$WORKDIR/lammps.log" -screen none
cp "$WORKDIR/setup.data" tests/integration/m1_smoke/setup.data
# Then regenerate the golden (see above) and commit both.
```

## Cross-compiler bit-exactness

The smoke runs in the `build-cpu` matrix (gcc-13 + clang-17, ubuntu-latest). The
golden is byte-compared in both jobs. FP64 determinism across these two
toolchains is expected on the same runner because the `cpu-only` preset disables
floating-point contractions and both compilers link the same `libm`. If an
upstream compiler release breaks this assumption, the fix is per-compiler
goldens (not a tolerance relaxation — we want to catch real regressions, not
paper over toolchain drift).
