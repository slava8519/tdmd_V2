# CI setup

TDMD CI is a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs on every PR and push to `main`. Pipeline A (lint + build) was live from M0; Pipelines B–C (unit/property + CPU differential) landed in M1–M2; M6 added compile-only CUDA coverage (T6.12).

## Jobs

| Job | Runner | Purpose |
|---|---|---|
| `lint` | GitHub-hosted `ubuntu-latest` | `pre-commit run --all-files` — every hook from `.pre-commit-config.yaml` |
| `docs-lint` | GitHub-hosted `ubuntu-latest` | `markdownlint-cli2` on every tracked `.md` |
| `build-cpu` (matrix: gcc-13, clang-17) | GitHub-hosted `ubuntu-latest` | `cmake --preset cpu-only && cmake --build && ctest` + M1..M5 smokes |
| `differential-t1` | GitHub-hosted `ubuntu-latest` | LAMMPS diff harness (SKIP on public CI — no LAMMPS submodule) |
| `differential-t4` | GitHub-hosted `ubuntu-latest` | LAMMPS diff harness (SKIP on public CI — no LAMMPS submodule) |
| `build-gpu` (matrix: Fp64ReferenceBuild, MixedFastBuild) | GitHub-hosted `ubuntu-latest` | **Compile + link only**; NVTX audit + PerfModel GPU cost-table tests run (pure C++); CUDA runtime tests self-skip |

### Option A CI policy (no self-hosted GPU runner)

TDMD is a public repo. Per user decision + memory `project_option_a_ci.md` + D-M6-6 in `m6_execution_pack.md`: **no self-hosted runner**. Rationale: a public-repo self-hosted runner would execute arbitrary PR code on the dev workstation. Trade-off accepted: **CUDA runtime gates (kernel bit-exactness, MixedFast thresholds, T3-gpu anchor, M6 smoke) run locally pre-push** instead of in CI. The `build-gpu` job catches compile/link regressions + NVTX + cost-table wiring; the rest is protected by the local pre-push protocol.

### Local pre-push GPU gate

Developer workstation with CUDA 12.8+ + an sm_80/86/89/90/120 GPU must run before `git push` on any GPU-touching commit:

```bash
# Reference flavor (bit-exact CPU↔GPU, D-M6-7 invariant)
cmake --preset default && cmake --build build --parallel
ctest --test-dir build --output-on-failure

# MixedFast flavor (D-M6-8 thresholds)
cmake -B build-mixed -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTDMD_BUILD_CUDA=ON \
  -DTDMD_BUILD_TESTS=ON \
  -DTDMD_BUILD_FLAVOR=MixedFastBuild \
  -DTDMD_CUDA_ARCHS="120"
cmake --build build-mixed --parallel
ctest --test-dir build-mixed --output-on-failure

# CPU-only-strict (catches -Werror regressions invisible to non-strict preset)
cmake -B build-cpu-strict -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTDMD_BUILD_CUDA=OFF \
  -DTDMD_ENABLE_MPI=OFF \
  -DTDMD_BUILD_TESTS=ON \
  -DTDMD_WARNINGS_AS_ERRORS=ON
cmake --build build-cpu-strict --parallel
ctest --test-dir build-cpu-strict --output-on-failure
```

All three must be green. LAMMPS oracle checks (`tools/build_lammps.sh && tools/lammps_smoke_test.sh`) required for T1/T4 differential diffs when changing `potentials/` or `integrator/`.

### `build-gpu` (compile-only) details

The GitHub-hosted `ubuntu-latest` runner installs stock `nvidia-cuda-toolkit` via apt (CUDA 12.x). Matrix covers Fp64ReferenceBuild + MixedFastBuild so both the flavor-dispatched adapter in `src/potentials/eam_alloy_gpu_adapter.cpp` and the MixedFast EAM kernel in `src/gpu/eam_alloy_gpu_mixed.cu` stay compile-clean. CUDA archs `80;86;89;90` (Ampere/Ada/Hopper); sm_100 + sm_120 (Blackwell + RTX 5080 dev) require CUDA 12.8+ which apt doesn't ship — local dev covers those via `--preset default`.

The job runs a narrow `ctest` filter after build:

- `test_gpu_types` — pure C++ PIMPL compile-firewall check;
- `test_nvtx_audit` — grep-based audit over `src/gpu/*.cu` (no GPU needed);
- `test_gpu_cost_tables` — structural checks on PerfModel linear-model math;
- `test_perfmodel` — existing CPU PerfModel tests.

Runtime-CUDA tests (`test_neighbor_list_gpu`, `test_eam_alloy_gpu`, `test_integrator_vv_gpu`, `test_eam_mixed_fast_within_threshold`, `test_device_pool`, `test_gpu_backend_smoke`) self-skip via `cudaGetDeviceCount() != cudaSuccess` — compile + link coverage only.

## Local pre-commit mirror of CI

Developer machines should run the same gates before pushing:

```bash
pip install --user pre-commit
pre-commit install            # git hook
pre-commit run --all-files    # dry run
```

If you cannot run a hook locally (missing `clang-format-18` etc.), CI will catch it. Do not push with `--no-verify` and hope.

## Branch protection

Set in GitHub: **Settings → Branches → Branch protection rule** for `main`:

- **Require a pull request before merging** ✓
- **Require status checks to pass before merging** ✓
  - Required checks: `Lint (pre-commit)`, `Docs lint (markdownlint-cli2)`, `Build CPU (gcc-13)`, `Build CPU (clang-17)`, `Build GPU compile-only (Fp64ReferenceBuild)`, `Build GPU compile-only (MixedFastBuild)`
- **Require branches to be up to date before merging** ✓
- **Include administrators** ✓ (solo-maintainer phase; disable only if forced by emergency)
- **Restrict deletions** ✓
- **Require linear history** (optional — forbids merge commits; keep off unless team prefers rebase)

## Caching strategy

- `lint` job: `~/.cache/pre-commit` keyed on `.pre-commit-config.yaml` hash (built-in via `actions/cache`).
- `build-cpu`: currently uncached. Can add `ccache` in M1+ if rebuilds become slow.
- `build-gpu`: uncached (compile-only; small working set). CUDA toolkit apt install dominates (~45s); nvcc compile of 4 `.cu` TUs + link ~3 min.

## Troubleshooting

### `build-gpu` nvcc fails with "unsupported GNU version"

Ubuntu stock `nvidia-cuda-toolkit` nvcc may not track the very latest GCC. If the matrix job fails at the nvcc compile step with a host-compiler version check, either:

1. Pin `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-12` in the workflow step; or
2. Switch the apt install to NVIDIA's own deb repo for CUDA 12.6+ (deferred; stock ubuntu is usually fine for compile-only).

### Pre-commit hook fails in CI but passes locally

Most common cause: you have an older hook version cached locally. Update:

```bash
pre-commit autoupdate
pre-commit clean
pre-commit run --all-files
```

### "sm_120 requires CUDA 12.8+" during local build

Your local toolkit is older than expected. Either install CUDA 12.8+ on your workstation or fall back to `cmake --preset default-sm89` (targets sm_89 Ada — works with CUDA 12.6).

### `docs-lint` complains on a SPEC file you just edited

Check `.markdownlint.yaml` — TDMD is permissive (line-length and duplicate-heading checks disabled). If a rule genuinely doesn't fit, exempt per-file with `<!-- markdownlint-disable-file MD034 -->` at the top.

## Adding a new pipeline (M1+)

1. Add the job to `.github/workflows/ci.yml` (or a new workflow file for long-running jobs).
2. Add the test source tree under `tests/<module>/<layer>/`.
3. Update `docs/development/ci_setup.md` job table.
4. Add the job to required status checks in branch protection.
5. Reference it from `docs/specs/<module>/TESTPLAN.md` §10 (CI pipeline mapping).

## See also

- [`build_instructions.md`](build_instructions.md) — local build setup
- [`code_style.md`](code_style.md) — lint rules enforced by pre-commit
- Master spec §11 (CI integration), §15.1 (Spec-Driven TDD)
- Playbook §8 (CI integration; merge gates A–F)
