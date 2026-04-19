#!/usr/bin/env python3
"""T7 mixed-scaling — PerfModel-based hardware normalisation (M7 stub).

Future role (T7.13 + M8): emit a per-(GPU, n_atoms_per_subdomain) ratio
that lets the harness translate the dissertation's Harpertown 2007 efficiency
numbers (or any reference cluster's) onto the local hardware. The math is
PerfModel `t_step_hybrid_seconds(N_per_sd, n_face_neighbors)` predicted vs
measured ratio — once T7.13 lands the calibration JSON, this script becomes
a thin wrapper that loads the JSON and emits the scalar.

T7.11 (this file): stub that emits ``perfmodel_calibration_ratio: 1.0`` so
the scaling_runner has a stable schema + callable entry. The 1.0 ratio
means "no normalisation applied" — the harness reports raw measured
efficiency, which is what the M7 acceptance gate (D-M7-8) actually checks.

Usage:
    python3 hardware_normalization.py            # scalar to stdout
    python3 hardware_normalization.py --json     # JSON object to stdout
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


STUB_RATIO = 1.0
STUB_NOTE = (
    "T7.11 stub — PerfModel calibration JSON loader lands with T7.13. "
    "Until then, raw measured efficiency is reported (no hardware "
    "normalisation applied)."
)


def perfmodel_calibration_ratio() -> float:
    return STUB_RATIO


def emit_payload() -> dict[str, Any]:
    return {
        "perfmodel_calibration_ratio": STUB_RATIO,
        "calibration_source": "stub",
        "note": STUB_NOTE,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a JSON object instead of a plain scalar",
    )
    args = parser.parse_args(argv)
    if args.json:
        print(json.dumps(emit_payload()))
    else:
        print(f"{STUB_RATIO:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
