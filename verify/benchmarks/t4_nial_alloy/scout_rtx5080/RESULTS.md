# T4 EAM Ni-Al scout — TDMD vs LAMMPS KOKKOS on RTX 5080

**Date:** 2026-04-20
**Hardware:** NVIDIA GeForce RTX 5080 (sm_120, 16 GB GDDR7, driver 590.48.01)
**Fixture:** 864-atom Ni-Al FCC (6×6×6), NiAl_Mishin_2004.eam.alloy
**Scope:** NVE, dt=1 fs, 100-step + 1000-step configs
**LAMMPS KOKKOS GPU:** `build_kokkos_cuda/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half`

## Headline numbers

1000-step median-of-3 (per `run_scout.sh` full protocol):

| Config                               | Wall (s) | ms/step | vs LAMMPS KOKKOS |
|--------------------------------------|---------:|--------:|-----------------:|
| TDMD `Fp64ReferenceBuild` (GPU)      |   5.11   | 5.1     | 7.3×             |
| TDMD `MixedFastBuild` (GPU)          |   4.22   | 4.2     | **6.0×**         |
| LAMMPS EAM KOKKOS `eam/alloy/kk`     |   0.697  | 0.70    | 1.00×            |
| LAMMPS EAM CPU 1-rank                |   2.12   | 2.1     | 3.0×             |

100-step quick scout (`quick_post_topt1.sh`, setup overhead inflated per-step):

| Config                               | Wall (s) | per-step ms (incl. setup) |
|--------------------------------------|---------:|--------------------------:|
| TDMD `Fp64ReferenceBuild` (GPU)      |   1.09   | 10.9                      |
| TDMD `MixedFastBuild` (GPU)          |   0.99   | 9.9                       |

## Key findings

1. **TDMD EAM MixedFast is ~6× slower than LAMMPS KOKKOS `eam/alloy/kk`** on
   864-atom Ni-Al FCC. Substantially smaller gap than the 21-48× observed
   on SNAP (T6 tungsten), because EAM's per-pair math is lighter and LAMMPS
   KOKKOS's per-bond team dispatch delivers less of a relative advantage.

2. **MixedFast 4.22 → Fp64Ref 5.11 ms/step = 21 % FP64→mixed speedup** on
   EAM. Consistent with the 30 % observed on SNAP. EAM's narrow-FP32 scope
   (`--fmad=true` + FP32 interior; all splines stay FP64 per D-M6-8) is
   already tight; no further narrow-FP32 lever available without hitting
   EAM monotonicity failures documented in D-M6-8 empirical data.

3. **T-opt-1 spline fusion REJECTED (2026-04-20).** Attempted to fuse the
   four per-bond spline calls in EAM force kernel (`spline_eval_dev` +
   `spline_deriv_dev` on ρ, F(ρ), pair φ) around a shared `locate_dev`
   computation. Implementation was clean and byte-exact (removed helpers
   that became unreferenced; density_kernel only uses spline_eval_dev and
   the fused force kernel does both value + deriv). **Perf result was
   flat** (1000-step MixedFast stayed at 4.22 s, no measurable
   speedup). Root cause: NVCC auto-CSE already folds the identical
   `__forceinline__` locate_dev calls at Ptx level; my manual fusion was
   invisible to the final SASS. Reverted per CLAUDE.md "Three similar
   lines is better than a premature abstraction" + "Don't add features
   beyond what the task requires."

4. **Real EAM lever is warp-parallel per-bond dispatch**, mirroring the
   LAMMPS KOKKOS `pair_eam_alloy_kokkos` team policy over bonds rather
   than atoms. This is architecturally similar to T8.6c-v5 for SNAP and
   is deferred to M9 (post-M8 SNAP work per project_m1_complete
   roadmap).

## Files

- `run_scout.sh` — full 3-flavor × 3-run median protocol with LAMMPS KOKKOS + CPU
- `quick_post_topt1.sh` — TDMD-only 2-flavor × 3-run quick scout (100+1000 step)
- `tdmd_gpu_100step.yaml` / `tdmd_gpu_1000step.yaml` — TDMD configs
- `lammps_script_eam.in` — matching LAMMPS EAM script
- `log.lammps` — latest LAMMPS run log (reference baseline)
- `post_topt1_scout.log` — captured output from quick_post_topt1.sh reference run
