#!/usr/bin/env python3
"""Generate the T6 BCC tungsten initial state (``setup.data``).

Committed output: ``verify/benchmarks/t6_snap_tungsten/setup.data``. Runs
once at authoring time; the committed file is what LAMMPS and TDMD actually
consume in the differential harness, so the two engines see bit-identical
atoms — the foundation of the D-M8-7 byte-exact gate (T8.5).

Configuration (frozen at T8.5 authoring, 2026-04-20):
    * BCC lattice, a0 = 3.1803 Å (Wood & Thompson 2017 fixture lattice
      constant — `verify/benchmarks/t6_snap_tungsten/lammps_script.in`).
    * 5 × 5 × 5 conventional cells × 2 atoms/cell → 250 atoms.
      The README "small" variant nominally specified 4×4×4 (128 atoms,
      L = 12.72 Å), but TDMD's CellGrid requires L_axis >= 3·(cutoff+skin)
      = 3·(4.73442 + 0.3) = 15.103 Å, so 5×5×5 (L = 15.90 Å) is the
      smallest BCC variant the SNAP path can run on. The medium variant
      (8×8×8 = 1024) and large (16×16×16 = 8192) clear the constraint
      with margin.
    * Single species W (mass 183.84 g/mol — `lammps_script.in:48`).
    * Initial velocities drawn from Maxwell-Boltzmann at 300 K, fixed
      seed Python RNG, COM momentum subtracted, KE rescaled so reported
      temperature is exactly 300 K. Mirrors verify/benchmarks/t4_nial_alloy/
      generate_setup.py — see that file for the rationale on why each step
      cannot be reordered without breaking byte-exactness.

This generator uses only the Python standard library (no NumPy / no SciPy)
so it runs in the CI `python3` image without extra deps.

Re-running is idempotent: given the same seed + lattice parameters, output
is byte-identical.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import random
import sys

# LAMMPS `units metal` mvv2e — couples ½mv² (g/mol · Å²/ps²) to energy (eV).
# Mirrored from src/runtime/include/tdmd/runtime/physical_constants.hpp;
# if that constant changes, this generator's output drifts.
METAL_MVV2E = 1.0364269e-4

# CODATA 2018 Boltzmann constant, eV/K — matches kBoltzmann_eV_per_K in
# physical_constants.hpp. LAMMPS uses 8.617343e-5 (older truncation); the
# 1.13e-6 relative residual is documented in thresholds.yaml §t1_al_morse_500
# and does NOT perturb this file because the velocity rescale below pins
# the kinetic energy, not k_B.
KB_EV_PER_K = 8.617333262e-5

# W mass — matches the LAMMPS reference script `mass 1 183.84` line.
W_MASS = 183.84


def generate(
    out_path: pathlib.Path,
    nrep: int = 5,
    a0: float = 3.1803,
    temp_K: float = 300.0,
    seed: int = 12345,
) -> None:
    rng = random.Random(seed)

    # BCC basis (2 atoms per conventional cell).
    basis = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
    ]

    # Lattice points — ordered in a reproducible scan so setup.data bytes are
    # deterministic given (nrep, a0).
    positions: list[tuple[float, float, float]] = []
    for iz in range(nrep):
        for iy in range(nrep):
            for ix in range(nrep):
                for bx, by, bz in basis:
                    positions.append(
                        (
                            (ix + bx) * a0,
                            (iy + by) * a0,
                            (iz + bz) * a0,
                        )
                    )
    n_atoms = len(positions)
    assert n_atoms == nrep * nrep * nrep * 2

    # Maxwell-Boltzmann initial velocities. Same formula as LAMMPS
    # `velocity create` (`velocity.cpp::create_gaussian`), modulo the COM
    # subtraction + KE rescale below that pin the temperature exactly.
    sigma = math.sqrt(KB_EV_PER_K * temp_K / (W_MASS * METAL_MVV2E))
    velocities: list[tuple[float, float, float]] = [
        (rng.gauss(0.0, sigma), rng.gauss(0.0, sigma), rng.gauss(0.0, sigma))
        for _ in range(n_atoms)
    ]

    # Subtract COM velocity so the system has zero net momentum. Single
    # species, so total mass = N · W_MASS.
    com_vx = sum(v[0] for v in velocities) / n_atoms
    com_vy = sum(v[1] for v in velocities) / n_atoms
    com_vz = sum(v[2] for v in velocities) / n_atoms
    velocities = [
        (vx - com_vx, vy - com_vy, vz - com_vz) for (vx, vy, vz) in velocities
    ]

    # Rescale to hit exactly temp_K using DOF = 3N − 3 (COM subtraction
    # removes 3 DOF), matching SimulationEngine::snapshot_thermo on the
    # TDMD side and `compute temp` on the LAMMPS side (default DOF treatment).
    ke_sum = sum(W_MASS * (vx * vx + vy * vy + vz * vz) for (vx, vy, vz) in velocities)
    ke_eV = 0.5 * ke_sum * METAL_MVV2E
    dof = 3.0 * n_atoms - 3.0
    current_T = 2.0 * ke_eV / (dof * KB_EV_PER_K)
    scale = math.sqrt(temp_K / current_T)
    velocities = [(vx * scale, vy * scale, vz * scale) for (vx, vy, vz) in velocities]

    # Lattice bounds — periodic box exactly spans nrep · a0 in each axis.
    hi = nrep * a0

    with out_path.open("w") as fh:
        fh.write(
            "LAMMPS data file for T6 BCC tungsten SNAP benchmark "
            f"(seed={seed}, T={temp_K:g} K, a0={a0:g} A, nrep={nrep})\n\n"
        )
        fh.write(f"{n_atoms} atoms\n")
        fh.write("1 atom types\n\n")
        fh.write(f"0.0 {hi:.10f} xlo xhi\n")
        fh.write(f"0.0 {hi:.10f} ylo yhi\n")
        fh.write(f"0.0 {hi:.10f} zlo zhi\n\n")
        fh.write("Masses\n\n")
        fh.write(f"1 {W_MASS:.10f}\n\n")
        fh.write("Atoms  # atomic\n\n")
        for i, (x, y, z) in enumerate(positions, start=1):
            fh.write(f"{i} 1 {x:.17g} {y:.17g} {z:.17g}\n")
        fh.write("\nVelocities\n\n")
        for i, (vx, vy, vz) in enumerate(velocities, start=1):
            fh.write(f"{i} {vx:.17g} {vy:.17g} {vz:.17g}\n")

    sys.stderr.write(
        f"[t6.generate_setup] wrote {out_path} — "
        f"{n_atoms} BCC W atoms, box = [{hi:g}]^3 A, T ~ {temp_K:g} K\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("setup.data"),
        help="Output path (default: setup.data next to this script)",
    )
    parser.add_argument(
        "--nrep",
        type=int,
        default=5,
        help=(
            "BCC conventional-cell replication per axis (default 5 → 250 atoms,"
            " canonical T8.5 byte-exact fixture). Common alternates:"
            " 8 → 1024 atoms (T8.10 fixture); 10 → 2000 atoms (scout);"
            " 16 → 8192 atoms (T8.11 strong-scaling base)."
        ),
    )
    args = parser.parse_args(argv)
    generate(args.out, nrep=args.nrep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
