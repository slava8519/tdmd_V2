#!/usr/bin/env python3
"""Generate the T4 Ni-Al FCC initial state (``setup.data``).

Committed output: ``verify/benchmarks/t4_nial_alloy/setup.data``. Runs once at
authoring time; the committed file is what LAMMPS and TDMD actually consume
in the differential harness, so the two engines see bit-identical atoms.

Configuration (frozen at T2.9 authoring):
    * FCC lattice, a0 = 3.52 Å (Ni native), 6 × 6 × 6 conventional cells.
    * 4 atoms / cell × 216 cells → 864 atoms.
    * 50:50 Ni / Al, uniformly shuffled with Python's ``random.Random(12345)``.
      Type 1 = Ni (mass 58.71), type 2 = Al (mass 26.982) — masses taken
      verbatim from the Mishin 2004 setfl header (species_names = [Ni, Al]).
      LAMMPS ``pair_eam_alloy::coeff()`` overrides atom masses with the setfl
      values on ``pair_coeff``; writing a different mass in ``Masses`` here
      would create a TDMD/LAMMPS divergence of ~1.7e-5 in KE at step 0.
    * Initial velocities drawn from Maxwell-Boltzmann at 300 K, same PRNG,
      COM momentum subtracted, kinetic energy rescaled so the reported
      temperature is exactly 300 K. Any of these steps reordered would move
      the committed bytes — do not refactor without a SPEC delta.

This generator intentionally uses only the Python standard library (no NumPy
/ no SciPy) so it runs in the CI `python3` image without extra deps.

Re-running is idempotent: given the same random seed + lattice parameters,
the output is byte-identical.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import random
import sys

# LAMMPS `units metal` mvv2e — couples ½mv² (g/mol · Å²/ps²) to energy (eV).
# Mirrored from ``src/runtime/include/tdmd/runtime/physical_constants.hpp``;
# if that constant changes, this generator's output drifts.
METAL_MVV2E = 1.0364269e-4

# CODATA 2018 Boltzmann constant, eV/K — matches ``kBoltzmann_eV_per_K`` in
# ``physical_constants.hpp``.  LAMMPS uses 8.617343e-5 (older truncation);
# the 1.13e-6 relative mismatch is a known residual (see thresholds.yaml
# §t1_al_morse_500 rationale) and does **not** perturb this file because
# the velocity rescale below pins the kinetic energy, not k_B.
KB_EV_PER_K = 8.617333262e-5


def generate(
    out_path: pathlib.Path,
    nx: int = 6,
    ny: int = 6,
    nz: int = 6,
    a0: float = 3.52,
    temp_K: float = 300.0,
    seed: int = 12345,
) -> None:
    rng = random.Random(seed)

    # FCC basis (4 atoms per conventional cell).
    basis = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    ]

    # Lattice points — ordered in a reproducible scan so setup.data bytes are
    # deterministic given (nx, ny, nz, a0).
    positions: list[tuple[float, float, float]] = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                for bx, by, bz in basis:
                    positions.append(
                        (
                            (ix + bx) * a0,
                            (iy + by) * a0,
                            (iz + bz) * a0,
                        )
                    )
    n_atoms = len(positions)
    assert n_atoms == nx * ny * nz * 4

    # 50:50 Ni/Al type assignment. For 864 atoms → 432 each exactly.
    n_ni = n_atoms // 2
    n_al = n_atoms - n_ni
    types = [1] * n_ni + [2] * n_al
    rng.shuffle(types)

    # Maxwell-Boltzmann init velocities at ``temp_K``.
    masses = {1: 58.71, 2: 26.982}
    # v_sigma² = kT / (m · mvv2e) — same formula as LAMMPS `velocity create`
    # (see `velocity.cpp::create_gaussian`), modulo the velocity rescale pass
    # below that pins kinetic energy exactly.
    sigmas = {
        t: math.sqrt(KB_EV_PER_K * temp_K / (masses[t] * METAL_MVV2E)) for t in masses
    }
    velocities: list[tuple[float, float, float]] = []
    for t in types:
        s = sigmas[t]
        velocities.append((rng.gauss(0.0, s), rng.gauss(0.0, s), rng.gauss(0.0, s)))

    # Subtract COM velocity so the system has zero net momentum. This is
    # what LAMMPS does internally when the default `loop geom` produces a
    # non-zero COM; we do it unconditionally because we're emitting the
    # final file (there is no LAMMPS rebalance pass after).
    total_mass = sum(masses[t] for t in types)
    com_px = sum(masses[t] * v[0] for t, v in zip(types, velocities))
    com_py = sum(masses[t] * v[1] for t, v in zip(types, velocities))
    com_pz = sum(masses[t] * v[2] for t, v in zip(types, velocities))
    com_vx = com_px / total_mass
    com_vy = com_py / total_mass
    com_vz = com_pz / total_mass
    velocities = [
        (vx - com_vx, vy - com_vy, vz - com_vz) for (vx, vy, vz) in velocities
    ]

    # Rescale to hit exactly ``temp_K`` using DOF = 3N − 3 (COM subtraction
    # removes three DOF), which is what ``SimulationEngine::snapshot_thermo``
    # uses on the TDMD side.
    ke_sum = 0.0  # in g/mol · Å²/ps²; convert to eV via · mvv2e.
    for t, (vx, vy, vz) in zip(types, velocities):
        ke_sum += masses[t] * (vx * vx + vy * vy + vz * vz)
    ke_eV = 0.5 * ke_sum * METAL_MVV2E
    dof = 3.0 * n_atoms - 3.0
    current_T = 2.0 * ke_eV / (dof * KB_EV_PER_K)
    scale = math.sqrt(temp_K / current_T)
    velocities = [(vx * scale, vy * scale, vz * scale) for (vx, vy, vz) in velocities]

    # Lattice bounds — periodic box exactly spans nx × a0 etc.
    xhi = nx * a0
    yhi = ny * a0
    zhi = nz * a0

    with out_path.open("w") as fh:
        fh.write(
            "LAMMPS data file for T4 Ni-Al FCC benchmark "
            f"(seed={seed}, T={temp_K:g} K, a0={a0:g} Å)\n\n"
        )
        fh.write(f"{n_atoms} atoms\n")
        fh.write("2 atom types\n\n")
        fh.write(f"0.0 {xhi:.10f} xlo xhi\n")
        fh.write(f"0.0 {yhi:.10f} ylo yhi\n")
        fh.write(f"0.0 {zhi:.10f} zlo zhi\n\n")
        fh.write("Masses\n\n")
        fh.write(f"1 {masses[1]:.10f}\n")
        fh.write(f"2 {masses[2]:.10f}\n\n")
        fh.write("Atoms  # atomic\n\n")
        for i, (t, (x, y, z)) in enumerate(zip(types, positions), start=1):
            fh.write(f"{i} {t} {x:.17g} {y:.17g} {z:.17g}\n")
        fh.write("\nVelocities\n\n")
        for i, (vx, vy, vz) in enumerate(velocities, start=1):
            fh.write(f"{i} {vx:.17g} {vy:.17g} {vz:.17g}\n")

    sys.stderr.write(
        f"[t4.generate_setup] wrote {out_path} — "
        f"{n_atoms} atoms ({n_ni} Ni + {n_al} Al), "
        f"box = [{xhi:g}]³ Å, T ≈ {temp_K:g} K\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("setup.data"),
        help="Output path (default: setup.data next to this script)",
    )
    args = parser.parse_args(argv)
    generate(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
