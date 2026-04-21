#!/usr/bin/env python3
"""T3 anchor-test — hardware normalisation helper.

The dissertation's absolute performance numbers (figures 29-30) were
measured on a 2007-era Intel Xeon Harpertown node (single-core peak ~9
GFLOPS double-precision). Current hardware runs 3-30x faster per core.
Comparing TDMD's raw steps/second against the dissertation without
normalisation would trivially "pass" the 10% threshold on modern silicon.

This script produces a single scalar ``ghz_flops_ratio`` that the harness
(T5.11) multiplies into the expected-performance column before diffing
against TDMD's measured performance. Offline-only (no network, no package
downloads) and relies only on a system C compiler (``cc``, universally
present on Linux build hosts) for the native micro-benchmark.

Algorithm:
  1. Compile a tight, L2-resident DAXPY-MAC kernel in
     ``hardware_flops_probe.c`` using ``cc -O3 -march=native`` (auto-
     vectorises to AVX2 / AVX-512 / NEON as appropriate). Binary cached
     at ``~/.cache/tdmd/hw_probe_<arch>_<cc_hash>``.
  2. Runs the binary, parses the single-scalar GFLOPS number it prints.
  3. Divides by the fixed 2007 Harpertown peak baseline (9 GFLOPS, per
     Andreev 2007 §3.5.1 + archival Intel spec sheet).
  4. Emits the ratio as a float; the harness passes this into its
     threshold comparison alongside the CSV entries.

Usage:
    python3 hardware_normalization.py           # prints scalar to stdout
    python3 hardware_normalization.py --json    # JSON object on stdout

Exit codes:
    0 — success
    1 — argument parse / runtime / compiler failure
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys

# 2007-era Intel Xeon Harpertown single-core peak FP64 throughput.
# Source: Andreev 2007 §3.5.1 + archival Intel spec sheet.
HARPERTOWN_GFLOPS_BASELINE = 9.0

_PROBE_C_FILENAME = "hardware_flops_probe.c"
_PROBE_BIN_PREFIX = "hw_flops_probe"


def _probe_c_path() -> pathlib.Path:
    """Absolute path to the native probe source (sibling to this script)."""
    return pathlib.Path(__file__).resolve().parent / _PROBE_C_FILENAME


def _cache_root() -> pathlib.Path:
    """Honours XDG_CACHE_HOME; default ~/.cache/tdmd."""
    xdg = os.environ.get("XDG_CACHE_HOME", "")
    root = pathlib.Path(xdg) if xdg else pathlib.Path.home() / ".cache"
    return root / "tdmd"


def _compiler() -> str:
    """Resolve the C compiler from $CC, falling back to ``cc``."""
    return os.environ.get("CC", "cc")


def _binary_cache_path() -> pathlib.Path:
    """Binary is namespaced by arch + compiler-version hash to avoid
    reusing a binary compiled for a different ISA / toolchain."""
    cc = _compiler()
    try:
        ver = subprocess.run(
            [cc, "--version"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except (OSError, FileNotFoundError):
        ver = "unknown"
    tag = hashlib.sha1(f"{ver}|{platform.machine()}".encode()).hexdigest()[:12]
    return _cache_root() / f"{_PROBE_BIN_PREFIX}_{platform.machine()}_{tag}"


def _compile_native_probe(binary_path: pathlib.Path) -> None:
    """Compile ``hardware_flops_probe.c`` to ``binary_path``.

    Raises ``RuntimeError`` with a detailed message on any failure so the
    caller (runner or stand-alone invocation) can surface it through the
    normalisation log rather than silently falling back to a broken
    Python kernel.
    """
    cc = _compiler()
    if shutil.which(cc) is None:
        raise RuntimeError(
            f"C compiler '{cc}' not found on PATH. The T3 anchor-test hardware "
            "probe requires a system C compiler (set $CC or install gcc/clang)."
        )
    src = _probe_c_path()
    if not src.is_file():
        raise RuntimeError(f"probe source missing: {src}")

    binary_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        cc,
        "-O3",
        "-march=native",
        "-o",
        str(binary_path),
        str(src),
        "-lm",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"native probe compile failed: {' '.join(cmd)}\n"
            f"stderr: {result.stderr.strip()}"
        )


def _run_native_probe(binary_path: pathlib.Path) -> float:
    """Execute the compiled probe and return the GFLOPS it printed."""
    result = subprocess.run(
        [str(binary_path)], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"native probe exited with code {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    try:
        return float(line)
    except ValueError as exc:
        raise RuntimeError(
            f"native probe did not emit a scalar GFLOPS on its last stdout "
            f"line: '{result.stdout.strip()}'"
        ) from exc


def measure_local_gflops() -> float:
    """Return the locally-measured peak single-core FP64 throughput (GFLOPS).

    Compiles the native probe on first use, caches the binary under
    ``~/.cache/tdmd`` (arch- and compiler-tagged), and runs it. Subsequent
    calls re-use the cached binary — the 24h cache TTL in
    ``hardware_probe.py`` already amortises the probe run itself across
    anchor-test invocations.
    """
    binary = _binary_cache_path()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        _compile_native_probe(binary)
    return _run_native_probe(binary)


def ghz_flops_ratio(
    baseline_gflops: float = HARPERTOWN_GFLOPS_BASELINE,
    local_gflops: float | None = None,
) -> float:
    """Return current-hw / 2007-Harpertown single-core FP64 throughput ratio."""
    if local_gflops is None:
        local_gflops = measure_local_gflops()
    if baseline_gflops <= 0.0 or local_gflops <= 0.0:
        return 1.0  # safest fallback — degenerates to un-normalised compare
    return local_gflops / baseline_gflops


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Emit the current-hw / 2007-Harpertown FP64 throughput ratio."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a JSON object instead of a plain scalar",
    )
    parser.add_argument(
        "--baseline-gflops",
        type=float,
        default=HARPERTOWN_GFLOPS_BASELINE,
        help=(
            "2007-era per-core baseline in GFLOPS "
            f"(default: {HARPERTOWN_GFLOPS_BASELINE})"
        ),
    )
    args = parser.parse_args(argv)

    try:
        local = measure_local_gflops()
    except RuntimeError as exc:
        print(f"hardware_normalization: {exc}", file=sys.stderr)
        return 1
    ratio = ghz_flops_ratio(args.baseline_gflops, local_gflops=local)
    if args.json:
        print(
            json.dumps(
                {
                    "ghz_flops_ratio": ratio,
                    "baseline_gflops": args.baseline_gflops,
                    "local_gflops": local,
                }
            )
        )
    else:
        print(f"{ratio:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
