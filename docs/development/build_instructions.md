# Build instructions

## Prerequisites

| Component   | Required | Notes |
|---|---|---|
| CMake       | ≥ 3.25 | `cmake --version` |
| Ninja       | any recent | `apt install ninja-build` |
| C++ compiler | GCC ≥ 13 or Clang ≥ 17 | C++20 concepts, ranges |
| CUDA toolkit | ≥ 12.8 for sm_120 (RTX 5080) <br> ≥ 12.4 for sm_89 (RTX 4090) | See CUDA fallback below |
| git         | any | LFS assets are fetched on `git lfs install` |

Optional:

- `clang-format-18`, `clang-tidy-18` — for lint (`tools/lint/run_clang_format.sh`)
- `pre-commit` (via `pip install pre-commit`) — pre-commit hooks

## Quickstart (dev machine with RTX 5080 + CUDA 12.8+)

```bash
cmake -B build --preset default
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Preset reference

| Preset          | Purpose | Build dir | CUDA |
|---|---|---|---|
| `default`       | Canonical dev config, sm_120 | `build/`         | ON (sm_120) |
| `default-sm89`  | Fallback for CUDA 12.6 or RTX 40xx | `build_sm89/` | ON (sm_89) |
| `debug`         | ASan + UBSan on host, CUDA debug symbols | `build_debug/` | ON (sm_120) |
| `cpu-only`      | No CUDA — for lint/CI/CPU-only machines | `build_cpu/` | OFF |
| `release`       | Production binary, Fp64Production, warnings-as-errors | `build_release/` | ON (sm_120) |

List all presets: `cmake --list-presets=all`

## CUDA architecture selection

The default preset targets **sm_120 (Blackwell, RTX 5080)**. This requires **CUDA 12.8+**. On machines with CUDA 12.6 (sm_120 not yet supported), either:

1. **Upgrade CUDA** (recommended once 12.8+ is available):

   ```bash
   sudo apt install cuda-toolkit-12-8
   export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
   ```

2. **Use the sm_89 fallback preset** (Ada, RTX 40xx):

   ```bash
   cmake -B build_sm89 --preset default-sm89
   cmake --build build_sm89 --parallel
   ```

3. **Build a fat binary** covering both:

   ```bash
   cmake -B build --preset default -DTDMD_CUDA_ARCHS="89;120"
   ```

4. **Auto-detect** (not recommended for reproducible builds):

   ```bash
   cmake -B build --preset default -DTDMD_CUDA_ARCHS=native
   ```

If CMake detects `sm_120` with CUDA < 12.8 it emits a `FATAL_ERROR` with upgrade instructions rather than letting nvcc fail cryptically.

## BuildFlavor

Five flavors per master spec §7.1; all are compile-time-fixed (no runtime switching). Select via cache variable:

```bash
cmake -B build --preset default -DTDMD_BUILD_FLAVOR=Fp64ProductionBuild
```

| Flavor | Status (M0) | Semantics |
|---|---|---|
| `Fp64ReferenceBuild` | active | Bitwise oracle. `-fno-fast-math`, `--fmad=false`. |
| `Fp64ProductionBuild` | active | Same precision, FMA allowed. |
| `MixedFastBuild` | stub | Philosophy B, mixed fp. Real impl in M2+. |
| `MixedFastAggressiveBuild` | stub | Philosophy A, aggressive. M2+. |
| `Fp32ExperimentalBuild` | stub | fp32-only, research. M2+. |

The active flavor is reported on configure and baked into every target as `TDMD_BUILD_FLAVOR_NAME` (useful for telemetry).

## Common problems

### "CUDA compiler not found"

Either install CUDA or disable CUDA targets:

```bash
cmake -B build --preset cpu-only
```

### "sm_120 requires CUDA 12.8+"

See [CUDA architecture selection](#cuda-architecture-selection).

### "In-source builds are disallowed"

You ran `cmake .` from the repo root. Use `-B build` to place artifacts outside the source tree.

### Clang-tidy "no compile database"

Clang-tidy needs `compile_commands.json`. Configure first, then run:

```bash
cmake -B build --preset default   # emits build/compile_commands.json
tools/lint/run_clang_tidy.sh
```

### ASan false positives with CUDA

ASan and nvcc are not fully compatible. The `debug` preset applies ASan to host C++ only; CUDA targets are debug-symbol'd but un-sanitized. If you need pure sanitizer coverage, use `cpu-only` + `debug`-style flags.

## Verifying the build

After `cmake --build build`:

```bash
ls build/compile_commands.json    # exists
ctest --test-dir build            # all tests pass (0 tests until T0.5)
```

## See also

- [`code_style.md`](code_style.md) — lint + pre-commit setup
- Master spec §7 (BuildFlavor × ExecProfile), §D.14 (build system integration)
- [`m0_execution_pack.md`](m0_execution_pack.md) T0.4 — origin task spec
