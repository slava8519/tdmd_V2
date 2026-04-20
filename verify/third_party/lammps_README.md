# LAMMPS oracle (submodule wrapper)

This directory hosts the LAMMPS submodule used as TDMD's **external scientific oracle**. TDMD does not link against LAMMPS at runtime — it is used only for differential validation (master spec §3 and §13).

## Pinned version

| Field | Value |
|---|---|
| Upstream | https://github.com/lammps/lammps |
| Tag | `stable_22Jul2025_update4` |
| Date | 2025-07-22 (latest update July 2025) |
| Why this tag | Latest `stable_*` release series as of 2026-04-17 that has been in the wild long enough to accumulate minor-release fixes. Avoids `develop` branch — we need a frozen reference. |

Re-pinning policy: only the **Architect role** can re-pin, in a dedicated SPEC-delta PR that also updates `docs/specs/verify/SPEC.md §3` and regenerates all affected threshold baselines. Do not bump opportunistically.

## Layout

```
verify/third_party/
├── lammps/               # the submodule itself (upstream LAMMPS, read-only)
│   ├── cmake/            # upstream CMake build config
│   ├── src/              # upstream source
│   ├── potentials/       # shipped empirical potentials (EAM, SNAP fixtures, ...)
│   ├── build_tdmd/       # our build tree (gitignored)
│   └── install_tdmd/     # our install prefix (gitignored)
│       └── bin/lmp       # the binary consumed by differential tests
└── lammps_README.md      # this file (sibling — not inside the submodule)
```

Wrapper scripts live in `tools/` (not inside the submodule, to keep its git tree pristine):

- `tools/build_lammps.sh` — configure + build + install with TDMD's required packages
- `tools/lammps_smoke_test.sh` — runs a 100-step Al FCC EAM trajectory on CPU and GPU, asserts agreement
- `tools/lammps_smoke_test.in` — the LAMMPS input script the smoke test drives

## Required packages

TDMD's build wrapper enables these LAMMPS packages (rationale → milestone that needs them):

| Package | For what | Milestone |
|---|---|---|
| `GPU` (CUDA, fp64) | Apples-to-apples GPU comparison against TDMD's CUDA kernels | M6+ |
| `MANYBODY` | EAM, Tersoff, Stillinger-Weber | M2+ |
| `MEAM` | Modified EAM | M10 |
| `ML-SNAP` | SNAP machine-learned potential | M8 |
| `MOLECULE` | Basic molecular topology (needed by many data files) | M1+ |
| `KSPACE` | Long-range Coulomb | M13 |
| `EXTRA-PAIR` | Auxiliary pair styles (Buckingham, Coul, etc.) | ongoing |
| `EXTRA-DUMP` | Extra dump output formats, helpful for diff harness | M1+ |

Deliberately **not** enabled (add via SPEC delta when milestones require):

- `KOKKOS` — alternative GPU path; GPU package is sufficient for M6–M11.
- `ML-PACE`, `ML-IAP` — needed only at M12.
- Python bindings — standalone `lmp` binary + subprocess is enough for the diff harness.

## How to build

Prerequisites: CUDA 12.8+ (for `sm_120` / RTX 5080), GCC 13+, CMake 3.25+, Ninja, MPI (OpenMPI or MPICH), ~4 GB free disk, ~20 min build time on a 16-thread machine.

```bash
# From repo root:
git submodule update --init --depth 1 verify/third_party/lammps
tools/build_lammps.sh
```

If you're on CUDA 12.6 (sm_120 not supported):

```bash
TDMD_LAMMPS_CUDA_ARCH=sm_89 tools/build_lammps.sh
```

Verify packages and GPU support:

```bash
verify/third_party/lammps/install_tdmd/bin/lmp -h | grep -E 'GPU|MANYBODY|MEAM|ML-SNAP'
```

Run the smoke test:

```bash
tools/lammps_smoke_test.sh
```

## SNAP fixture (T6 benchmark — M8)

Canonical T6 tungsten SNAP fixture consumed by TDMD's SNAP differential harness
(M8 T8.5+):

| Artifact | Path (inside submodule) | Purpose |
|---|---|---|
| `W_2940_2017_2.snap` | `verify/third_party/lammps/examples/snap/` | LAMMPS include-style fixture: `pair_style hybrid/overlay zbl ... snap` + `pair_coeff` wiring |
| `W_2940_2017_2.snapcoeff` | same dir | 30-param bispectrum coefficient file |
| `W_2940_2017_2.snapparam` | same dir | SNAP hyperparameters (twojmax, rcutfac, ...) |
| `in.snap.W.2940` | same dir | driver example (100-step NVE, 128-atom BCC W) |
| `log.15Jun20.snap.W.2940.g++.1` | same dir | upstream reference log — sanity check, not TDMD gate |

Reference: Wood, M. A. & Thompson, A. P. "Quantum-Accurate Molecular Dynamics
Potential for Tungsten" arXiv:1702.07042. Pure W single-species BCC, 2940 DFT
training configurations, dated 2017-02-20. Preferred over `WBe_Wood_PRB2019`
(binary alloy — deferred to M9+ SNAP alloy gate).

**No binary `.snap` tracked by the tdmd repo** per m8 exec pack D-M8-3 —
submodule owns the file; dev setup cost is one-time `git submodule update --init`.

Sanity run (expected to match upstream `log.15Jun20.snap.W.2940.g++.1` byte-
exactly to LAMMPS float precision, on any CPU):

```bash
cd verify/third_party/lammps/examples/snap
LD_LIBRARY_PATH=../../install_tdmd/lib ../../install_tdmd/bin/lmp -in in.snap.W.2940
# Step 0:   TotEng = -10.98985    Temp = 300
# Step 100: TotEng = -10.989847   Press = 11987.181
```

## Updating the submodule

Don't. If a bump is genuinely needed:

1. Open an Architect-role SPEC delta PR modifying `docs/specs/verify/SPEC.md §3` with rationale.
2. Update the tag in `tools/build_lammps.sh` comments and in this README.
3. `git -C verify/third_party/lammps fetch origin stable_<new_tag>:refs/tags/stable_<new_tag>`
4. `git -C verify/third_party/lammps checkout stable_<new_tag>`
5. `git add verify/third_party/lammps` (updates the submodule pointer)
6. Re-baseline affected differential thresholds (Validation Engineer).
7. Explain in the PR what changed upstream that required the bump.

## CI

The weekly `scheduled-cuda-rebuild` workflow does **not** rebuild LAMMPS — that would burn ~20 min every Monday. Instead, a separate `build-lammps` job (added in M1+) runs on-demand via the `lammps-rebuild` PR label or weekly cron. For M0, LAMMPS is built once by the developer and committed to the self-hosted runner disk; CI only checks that the submodule pointer is correct.

## See also

- [`docs/specs/verify/SPEC.md`](../../docs/specs/verify/SPEC.md) — full verify-layer contract, including threshold registry
- [`docs/development/build_instructions.md`](../../docs/development/build_instructions.md) — TDMD build
- Master spec §3 (LAMMPS as oracle), §13 (test pyramid)
