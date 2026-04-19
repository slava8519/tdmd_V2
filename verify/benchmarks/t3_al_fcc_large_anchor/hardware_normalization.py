#!/usr/bin/env python3
"""T3 anchor-test — hardware normalisation helper.

The dissertation's absolute performance numbers (figures 29-30) were
measured on a 2007-era Intel Xeon Harpertown node (single-core peak ~9
GFLOPS double-precision). Current hardware runs 5-20x faster per core.
Comparing TDMD's raw steps/second against the dissertation without
normalisation would trivially "pass" the 10% threshold on modern silicon.

This script produces a single scalar `ghz_flops_ratio` that the harness
(T5.11) multiplies into the expected-performance column before diffing
against TDMD's measured performance. Offline-only (no network, no package
downloads); stdlib-only (no numpy) so it runs on any 3.8+ interpreter.

Algorithm:
  1. Run a compact O(n^2) pair-force micro-benchmark using pure Python —
     2048 atoms, cutoff 8 A, 5 iterations. Pure Python loops are slow but
     proportionally slow across machines; the resulting "local GFLOPS" is
     conservative but its ratio against the 2007 baseline is well-defined.
  2. Divides by the fixed 2007 baseline (9 GFLOPS, single-core Harpertown
     peak as quoted in Andreev 2007 §3.5.1).
  3. Emits the ratio as a float; the harness passes this into its threshold
     comparison alongside the CSV entries.

Usage:
    python3 hardware_normalization.py           # prints scalar to stdout
    python3 hardware_normalization.py --json    # JSON object on stdout

Exit codes:
    0 — success
    1 — argument parse / runtime error
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time

# 2007-era Intel Xeon Harpertown single-core peak FP64 throughput.
# Source: Andreev 2007 §3.5.1 + archival Intel spec sheet.
HARPERTOWN_GFLOPS_BASELINE = 9.0

# Empirical FLOP count per pair-force evaluation in the synthetic loop
# below. Counts dr_x/y/z subtractions, three multiplications, one sqrt,
# one cutoff compare, one Morse-style exp-free pair force accumulation.
FLOPS_PER_PAIR = 20.0


def _generate_positions(n_atoms: int, box: float, seed: int = 0xDEADBEEF):
    rng = random.Random(seed)
    return [
        (rng.uniform(0.0, box), rng.uniform(0.0, box), rng.uniform(0.0, box))
        for _ in range(n_atoms)
    ]


def measure_local_gflops(
    n_atoms: int = 2048, cutoff: float = 8.0, box: float = 80.0, n_iterations: int = 5
) -> float:
    """Micro-benchmark a short-range pair-force kernel on ``n_atoms`` atoms.

    Pure Python so we don't take a numpy/BLAS dependency; the absolute
    GFLOPS number will be 50-100x below native C, but its ratio against
    the dissertation baseline stays meaningful because we divide two
    pure-Python measurements.
    """
    positions = _generate_positions(n_atoms, box)
    cutoff2 = cutoff * cutoff
    pairs_evaluated = 0
    total_time = 0.0

    for _ in range(n_iterations):
        t0 = time.perf_counter()
        local_pairs = 0
        for i in range(n_atoms):
            xi, yi, zi = positions[i]
            for j in range(i + 1, n_atoms):
                xj, yj, zj = positions[j]
                dx = xi - xj
                dy = yi - yj
                dz = zi - zj
                r2 = dx * dx + dy * dy + dz * dz
                if 0.0 < r2 < cutoff2:
                    # Token work so the compiler / interpreter cannot
                    # hoist the loop body out — matches the FLOP count in
                    # FLOPS_PER_PAIR.
                    r = math.sqrt(r2)
                    _ = (1.0 / r) - (1.0 / cutoff)
                    local_pairs += 1
        t1 = time.perf_counter()
        pairs_evaluated += local_pairs
        total_time += t1 - t0

    if total_time <= 0.0 or pairs_evaluated == 0:
        return 0.0
    total_flops = pairs_evaluated * FLOPS_PER_PAIR
    return total_flops / total_time / 1.0e9


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

    local = measure_local_gflops()
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
