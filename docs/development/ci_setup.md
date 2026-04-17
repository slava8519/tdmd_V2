# CI setup

TDMD CI is a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs on every PR and push to `main`. Pipeline A (lint + build) is the only one live at M0; Pipelines B–F (unit, property, differential, performance, reproducibility) activate in M1+ as the corresponding test layers appear.

## Jobs at M0

| Job | Runner | Purpose |
|---|---|---|
| `lint` | GitHub-hosted `ubuntu-latest` | `pre-commit run --all-files` — every hook from `.pre-commit-config.yaml` |
| `docs-lint` | GitHub-hosted `ubuntu-latest` | `markdownlint-cli2` on every tracked `.md` |
| `build-cpu` (matrix: gcc-13, clang-17) | GitHub-hosted `ubuntu-latest` | `cmake --preset cpu-only && cmake --build && ctest` |
| `build-cuda` | **Self-hosted `gpu-rtx5080`** | `cmake --preset default && cmake --build && ctest` on sm_120 |
| `cuda-rebuild` (weekly, separate workflow) | Self-hosted `gpu-rtx5080` | From-scratch rebuild, Monday 06:00 UTC |

All four PR jobs must be green to merge into `main` (enforced by branch protection — see [below](#branch-protection)).

## Self-hosted runner registration (one-time)

The `build-cuda` job requires a self-hosted runner on the dev machine (RTX 5080 + CUDA 12.8+). GitHub-hosted runners do not have GPUs.

### 1. Register the runner

In GitHub: **Settings → Actions → Runners → New self-hosted runner**. Pick Linux x64. GitHub shows commands like:

```bash
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -o actions-runner-linux-x64.tar.gz -L https://github.com/actions/runner/releases/download/vX.Y.Z/actions-runner-linux-x64-X.Y.Z.tar.gz
tar xzf actions-runner-linux-x64.tar.gz
./config.sh --url https://github.com/slava8519/tdmd_V2 --token <token-from-github>
```

During `./config.sh` assign labels: `self-hosted,linux,gpu-rtx5080`. The workflows target these labels.

### 2. Install as a systemd service

Single-user interactive mode (`./run.sh`) is fine for testing but dies on logout. For persistence:

```bash
sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status
```

### 3. System prerequisites on the runner host

```bash
# Build tooling
sudo apt install -y cmake ninja-build git git-lfs gcc-13 g++-13

# CUDA 12.8+ (exact package name may vary)
sudo apt install -y cuda-toolkit-12-8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc  # or add to /etc/environment

# Python (for pre-commit on this runner if reused for lint)
sudo apt install -y python3 python3-pip
```

Confirm:

```bash
nvidia-smi                   # shows RTX 5080, compute cap 12.0
nvcc --version               # shows 12.8 or later
cmake --version              # ≥ 3.25
```

### 4. Runner hygiene

The workflow sets `actions/checkout@v4` without `clean: true` on PR jobs. Between jobs it deletes `build*/` directories. If the runner disk fills up:

```bash
cd ~/actions-runner/_work/tdmd_V2/tdmd_V2
rm -rf build build_cpu build_debug build_sm89 build_release
```

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
  - Required checks: `Lint (pre-commit)`, `Docs lint (markdownlint-cli2)`, `Build CPU (gcc-13)`, `Build CPU (clang-17)`, `Build CUDA (sm_120, self-hosted RTX 5080)`
- **Require branches to be up to date before merging** ✓
- **Include administrators** ✓ (solo-maintainer phase; disable only if forced by emergency)
- **Restrict deletions** ✓
- **Require linear history** (optional — forbids merge commits; keep off unless team prefers rebase)

Do not add `Scheduled CUDA rebuild` as a required status — it runs once a week and would block all merges.

## Caching strategy

- `lint` job: `~/.cache/pre-commit` keyed on `.pre-commit-config.yaml` hash (built-in via `actions/cache`).
- `build-cpu`: currently uncached. Can add `ccache` in M1+ if rebuilds become slow.
- `build-cuda`: uncached by design — self-hosted runner already has warm disk; CI time is dominated by Catch2 fetch (~30s) and build (~2 min for M0 skeleton).
- `cuda-rebuild` (weekly): explicitly uncached — the whole point is to catch cache-hidden regressions.

## Troubleshooting

### `build-cuda` job stuck in "queued"

The self-hosted runner is not online. Check:

```bash
sudo systemctl status actions.runner.slava8519-tdmd_V2.*.service
```

Restart if needed. GitHub **Settings → Actions → Runners** shows runner online/offline status.

### Pre-commit hook fails in CI but passes locally

Most common cause: you have an older hook version cached locally. Update:

```bash
pre-commit autoupdate
pre-commit clean
pre-commit run --all-files
```

### "sm_120 requires CUDA 12.8+" in `build-cuda`

The runner's CUDA toolkit is older than expected. Upgrade on the runner host, not in CI.

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
