# Third-party EAM potentials

Potentials that **do not ship** with the pinned LAMMPS submodule (`verify/third_party/lammps/`) but are required by TDMD differential benchmarks live here. Each file is committed verbatim alongside a provenance note. Do **not** regenerate or re-tabulate these files — the whole point of using a reference EAM in the differential harness is that LAMMPS and TDMD consume the exact same bytes.

## Files

### `NiAl_Mishin_2004.eam.alloy`

- **Source**: NIST Interatomic Potentials Repository, entry `2004--Mishin-Y--Ni-Al`, version 4.
- **URL**: https://www.ctcms.nist.gov/potentials/Download/2004--Mishin-Y--Ni-Al/4/NiAl_Mishin_2004.eam.alloy
- **Primary citation**: Y. Mishin, *Acta Materialia* **52**, 1451–1467 (2004). "Atomistic modeling of the γ and γ′-phases of the Ni-Al system."
- **LAMMPS-format reimplementation**: Lucas Hale, 12 Dec 2020. Cubic-spline interpolation of Mishin's original `.plt` tables; `F(ρ=0)` pinned to 0 so isolated-atom energies vanish.
- **Tabulation**: `Nrho = 10001`, `drho = 6.9951e-04`; `Nr = 10001`, `dr = 6.7249e-04`; `cutoff = 6.7249 Å`. Species `Ni` (Z=28, mass=58.71, lattice=3.52 Å, fcc), `Al` (Z=13, mass=26.982, lattice=4.05 Å, fcc).
- **License**: NIST data — public domain (U.S. government work, 17 U.S.C. §105). Redistribution unrestricted.
- **Consumer**: `verify/benchmarks/t4_nial_alloy/` (M2 T2.9 gate).

## Policy

- **No edits in place.** If a tabulation needs to be changed (e.g., higher-resolution resample), ship a new file alongside the original with a distinct name and cite the transformation in this README. Editing silently breaks byte-for-byte reproducibility of recorded differential runs.
- **No git-lfs.** These files are small (< 5 MB each) and part of the reproducibility contract — committing them verbatim keeps `git clone` self-sufficient.
- **Licensing note.** Only redistributable potentials may live here. Potentials under restrictive licenses (e.g., research-use-only) must be fetched at test time via a script, not committed.
