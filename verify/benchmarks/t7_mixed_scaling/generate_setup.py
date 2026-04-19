#!/usr/bin/env python3
"""Generate the T7 mixed-scaling initial state (``setup.data``).

Delegates to the T4 generator (``../t4_nial_alloy/generate_setup.py``)
with a larger lattice — same algorithmic chain (FCC basis, 50:50
type assignment with seed 12345, Maxwell-Boltzmann at 300 K, COM-zeroed,
KE-rescaled to exactly 300 K). Default 32×32×32 unit cells → 131,072
atoms (~1.3×10⁵), divisible by 1, 2, 4, 8, 16 for clean Pattern 2
strong-scaling probes.

This script is the **single source** of T7's atom state. The harness
(``scaling_runner``) calls it lazily when ``setup.data`` is missing
(same pattern as ``t3_al_fcc_large_anchor/regen_setup.sh``); the file
is intentionally not committed to keep the repo git-LFS-free
(~7.5 MB at 131k atoms).

Usage:
    python3 generate_setup.py                      # defaults: 32x32x32
    python3 generate_setup.py --nx 16 --ny 16 --nz 16   # smaller probe
    python3 generate_setup.py --out /tmp/setup.data
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def _import_t4_generator():
    """Locate and import the T4 ``generate_setup.generate`` symbol.

    Done as a function-local import so the script can be invoked from
    any cwd without polluting sys.modules at import time.
    """
    here = pathlib.Path(__file__).resolve().parent
    t4_dir = here.parent / "t4_nial_alloy"
    if str(t4_dir) not in sys.path:
        sys.path.insert(0, str(t4_dir))
    from generate_setup import generate  # noqa: E402

    return generate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help=(
            "Output path (default: ../../data/t7_mixed_scaling/setup.data "
            "next to the verify/data/ tree; auto-created)"
        ),
    )
    parser.add_argument("--nx", type=int, default=32, help="unit cells along x")
    parser.add_argument("--ny", type=int, default=32, help="unit cells along y")
    parser.add_argument("--nz", type=int, default=32, help="unit cells along z")
    parser.add_argument("--a0", type=float, default=3.52, help="lattice constant (Å)")
    parser.add_argument("--temp", type=float, default=300.0, help="temperature (K)")
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed")
    args = parser.parse_args(argv)

    out = args.out or (
        pathlib.Path(__file__).resolve().parents[2]
        / "data"
        / "t7_mixed_scaling"
        / "setup.data"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    generate = _import_t4_generator()
    generate(
        out_path=out,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        a0=args.a0,
        temp_K=args.temp,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
