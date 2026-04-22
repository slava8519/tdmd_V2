# T-opt-4 Item 1 — SNAP GPU bond-list on-demand rebuild — Pre-implementation report

**Author:** Claude (GPU / Performance Engineer role)
**Date:** 2026-04-21
**Status:** Draft — requires user review per CLAUDE.md §1.3 (no code until review)
**Budget:** Part of T-opt-4 scout 2–3 day envelope

---

## 1. Task as directed

> "Bond-list должен reuse существующий DisplacementTracker из M3 (T3.8 landed). Если DisplacementTracker показывает max_displacement > skin/2 — rebuild, иначе skip. Это LAMMPS check yes semantics. Byte-exact preserved (тот же bond-list, вычисляется реже). Expected 7–10% wall."

Three claims in that directive:

1. Skip bond-list rebuild while `max_displacement ≤ skin/2`.
2. Mirrors LAMMPS `neigh_modify check yes` semantics.
3. Byte-exact T8.7 gate (≤ 1e-12 rel) preserved.

**All three hold in principle for a pure neighbour-list refactor. This pre-impl documents why the third one does *not* hold for the current TDMD bond-list structure without a further change, and proposes three paths forward.**

---

## 2. Research findings

### 2.1 Bond-list structure

Per `src/gpu/include/tdmd/gpu/snap_bond_list_gpu.hpp` lines 52–76:

```cpp
struct SnapBondListGpuView {
  std::size_t atom_count, bond_count;
  const std::uint32_t* d_atom_bond_start;   // CSR offsets
  const std::uint32_t* d_bond_i, *d_bond_j; // pair identity
  const std::uint32_t* d_bond_type_i, *d_bond_type_j;
  const double* d_bond_dx, *d_bond_dy, *d_bond_dz;  // <-- per-bond Δr
  const double* d_bond_rsq;                          // <-- per-bond |Δr|²
  const std::uint32_t* d_reverse_bond_index;         // T-opt-3b
};
```

The bond list stores **both topology (i,j,type_i,type_j, reverse_index) and geometry (Δr.x, Δr.y, Δr.z, r²) at build time.**

### 2.2 Downstream consumption is purely from cache

`snap_gpu.cu:2013–2027` — `snap_ui_bond_kernel` is invoked with `bond_view.d_bond_dx/dy/dz/rsq` as its position input. It **never reads `d_x/d_y/d_z`** (the current atom position arrays). Same pattern at lines 2110–2135 (deidrj_bond) and 2138–2155 (force_gather). Mirror in `snap_gpu_mixed.cu:1891–1909` and downstream.

This means: skipping the bond-list rebuild on step N does not just reuse the *set* of (i,j) pairs — it reuses the *geometric values* from the last rebuild. Downstream SNAP kernels then compute the bispectrum from stale Δr, producing forces that disagree with a fresh rebuild at bit level (and at physical level, since positions have moved).

### 2.3 LAMMPS `check yes` is not analogous

LAMMPS KOKKOS SNAP path (`pair_snap_kokkos_impl.h`, confirmed via nsys):

- `TagPairSNAPComputeNeigh` — **runs every step** (one call per `compute()`), walks the NL, filters by rcutfac, writes a per-atom neighbour offset + per-neighbour (dx,dy,dz,rsq) structure **fresh each step**.
- `NPair…BuildFunctor` — the underlying NL rebuild — runs `check yes`, triggered by displacement (nsys: 2 calls / 100 steps on our fixture).

In other words, LAMMPS has two layers:
- *Neighbour list* (outer cutoff = rcut_snap + skin): on-demand, `check yes`.
- *Per-step SNAP-filtered neighbour geometry* (inner cutoff = rcut_snap): every step, recomputed from current positions.

TDMD has **one** layer today (`snap_bond_list_gpu`) that is doing the job of LAMMPS's inner layer, but which was assumed in the directive to be the outer layer. The two layers cannot be collapsed without losing correctness.

### 2.4 Why rebuild-when-displacement-exceeds-skin/2 is not byte-exact

Even under the standard skin invariant (`max_displacement ≤ skin/2`, so `|Δr_ij|` changes by ≤ skin across the interval), two failure modes arise:

**Failure mode A — stale stored geometry.** Downstream SNAP kernels compute ui/yi/deidrj from `d_bond_dx/dy/dz/rsq` stored at build time. These are not refreshed when positions change. Forces drift by an amount proportional to `|Δposition since last build|`. At step k after rebuild with dt·velocity ~ 1e-3 Å/step, after 100 skipped rebuilds the stored Δr is off by ~0.1 Å — far past 1e-12 rel force tolerance.

**Failure mode B — bond set transitions at rcutfac.** A pair starting at `|r_ij| = rcutfac − ε` enters the bond list. If it moves outward past rcutfac within the NL interval (possible under `max_displacement ≤ skin/2` whenever `ε < skin`), the force kernel still iterates the stored bond, applies the smooth cutoff `fc(r_stored) > 0`, and contributes a spurious force. Conversely, a pair that moves *into* rcutfac is missed entirely. Both directions break byte-exact; the smooth cutoff at rcutfac bounds the error but does not zero it because the kernel reads stored r², not current r².

### 2.5 The cell grid is already rebuilt per step

`src/potentials/snap_gpu_adapter.cpp:93–109` — BoxParams is populated per invocation; `grid.bin(atoms)` runs per scheduler zone task. So the per-step cell bin refresh is already free to us and is not the source of the 1.7 ms/step `snap.bond_list.build` cost.

---

## 3. Where the 1.7 ms/step actually goes

Nsys profile (100 steps, 2000-atom W BCC, MixedFastSnapOnly):

- `snap.bond_list.build` NVTX range: 101 calls × 872 µs ≈ 88 ms total ≈ **1.75 ms/step of per-step cost** (post-T-opt-2).
- Breakdown inside the range (kernel-level):
  - `count_bonds_kernel` 1.8 %
  - `reverse_index_kernel` (implicit via scan)
  - `emit_bonds_kernel` 1.9 %
  - Host scan + alloc + launch overhead (remainder)
- KOKKOS analog (`TagPairSNAPComputeNeigh` inferred from total kernel time): ≤ 0.05 ms/step on same fixture.

Gap ≈ 30× on the bond-list build alone. This is the real measurable inefficiency.

---

## 4. Invariants that must be preserved

| Invariant | Source | Why |
|---|---|---|
| T8.7 byte-exact ≤ 1e-12 rel force / PE / virial | master spec §14 M8, test_snap_gpu_bit_exact | Reference path is the oracle |
| Determinism across ranks (D-M7-10) | neighbor SPEC §6, m8_smoke_t6 test | Multi-rank TD requires byte-exact single-rank |
| No `atomicAdd(double*)` | gpu SPEC §6.1 | Non-determinism |
| T3 anchor-test (M5) still passes | CLAUDE.md "Architectural invariants" | Foundational |
| `displacement_tracker` ownership remains with NeighborManager | neighbor SPEC §2.2 | Scheduler-driven rebuild cadence |

No SNAP-specific bond-list cadence policy exists in any SPEC today — this task will likely add one.

---

## 5. Three design options, ranked honestly

### Option 1 — **Cache reuse (as directed)**: skip build when `max_displacement ≤ skin/2`

**Implementation:** pass `const DisplacementTracker*` into `SnapGpu::compute()`; at the top, read `max_displacement()` and only rebuild if above threshold; also always rebuild on NL build_version change.

**Byte-exact status:** **FAILS** for the reasons in §2.4. Force drift grows with skipped-step count and crosses 1e-12 rel within the first skipped step.

**Correctness impact:** Physical simulation is wrong, not just numerically imprecise — stored Δr is the *old* Δr, not a slightly perturbed one.

**Do not pursue as-is.** This is where my agreement with the directive breaks.

### Option 2 — **Split into topology + per-step geometry**

Introduce two objects:

1. `SnapBondTopologyGpu` — stores `(i, j, type_i, type_j, reverse_bond_index, atom_bond_start)`; rebuilt on-demand when `max_displacement > skin/2` (LAMMPS `check yes` for the topology). Requires **NL skin applied to SNAP cutoff** (topology filtered at `rcutfac + skin`, not `rcutfac`) — this is a spec change.
2. `SnapBondGeometryGpu` — stores `(dx, dy, dz, rsq)`; rebuilt **every step** from current positions walking the cached topology.

**Byte-exact status:** **preserved** if the topology contains every pair within `rcutfac + skin` (strict superset of the true rcutfac pairs). Bonds with `r > rcutfac` get `fc(r) = 0` and contribute zero — fine.

**Cost savings:** ≈ the cell-stencil walk (count + emit + scan ≈ ~0.5–1.0 ms/step moved to rebuild-cadence-only, amortized over ~30–100 steps per NL rebuild on typical fixtures). Net: **expected 50–80 % reduction in bond-list overhead = ~0.8–1.4 ms/step wall = 3–5 % of 29.5 ms/step**.

**Effort:** 1.5–2 days. Largest lift is adding skin-aware rcutsq to `SnapBondListGpu::build_from_device`, plus a new per-step geometry kernel, plus SPEC delta to `gpu/SPEC.md §7.5` and `potentials/SPEC.md §6`.

**Tests:** T8.7 byte-exact, T6 differential, T3 anchor-test, m8_smoke_t6 100-step. Plus a new unit test for `SnapBondTopologyGpu` superset invariant.

**Risks:**
- Topology grows roughly `(1 + skin/rcutfac)³ ≈ 1.3×` larger than current bond list for skin = 1.0 Å, rcutfac = 4.8 Å on T6 fixture → per-step geometry kernel has 1.3× more bonds to touch; some of the savings is given back.
- If skin is much smaller than rcutfac (e.g. 0.3 Å on a different fixture), topology grows only ~1.06× — better.

### Option 3 — **Lean the existing per-step build; don't change cadence**

Accept that rebuild must be per-step, and just make the existing `snap.bond_list.build` faster by closing the 30× gap to KOKKOS. Candidates:

- Fuse `count_bonds_kernel` + `emit_bonds_kernel` (currently a 2-pass CUB scan with a host-side sync for the bond count). LAMMPS does a single-pass walk with pre-sized views.
- Replace the D2H-for-total-count + H2D-for-alloc round-trip with a device-side allocation (pre-alloc to `n_atoms * max_neighbours_upper_bound`, track actual count on device).
- Merge bond-list walk into `snap_ui_bond_kernel` (fused walk + ui).

**Byte-exact status:** preserved (still rebuilding every step from current positions).

**Cost savings:** up to 100 % of 1.75 ms/step (5.9 % of total wall). Upper bound only if we match KOKKOS's 0.05 ms/step.

**Effort:** 2–3 days. Higher-risk: single-pass CUB or device-side arena allocators. Kernel fusion touches T-opt-3b paired-bond invariants.

**Tests:** T8.7 byte-exact, T6 diff, anchor — all preserved.

**Risks:** the 30× gap may be structural (data layout, shared-memory sizing). Without ncu profile we don't know occupancy / register pressure on `count_bonds_kernel`. Matches the wider T-opt-4 ncu requirement.

---

## 6. My recommendation

**Do Option 3 first** (per-step build leaner). Rationale:

1. It is the path that is guaranteed byte-exact with no spec delta, no new tests beyond the existing T8.7 gate.
2. It closes the concrete measurable 30× gap to KOKKOS on the same bookkeeping step.
3. The ncu data we're about to gather (T-opt-4 umbrella) will directly inform the bottleneck — register pressure, shared-memory sizing, or launch overhead.
4. Option 2 is a ~spec-level structural change that is better landed as a dedicated M9+ follow-up once ncu data says whether the savings are worth the complexity.

**Skip Option 1.** The byte-exact promise in the directive is not achievable with the current bond-list structure. Implementing it as stated would fail T8.7 at first skipped step.

**Option 2 as a secondary path** — if Option 3 hits a structural wall (e.g. the gap is dominated by allocator/host-sync not by kernel compute), then Option 2 becomes the next lever.

---

## 7. Files that would change (Option 3, if authorized)

- `src/gpu/snap_bond_list_gpu.cu` — fuse count+emit, remove host scan round-trip (lines ~520–730)
- `src/gpu/include/tdmd/gpu/snap_bond_list_gpu.hpp` — possibly extend `Impl` with device-side counter
- `src/gpu/snap_gpu.cu:1973–1988` and `src/gpu/snap_gpu_mixed.cu:1891–1909` — unchanged if we keep the same API
- `tests/gpu/test_bond_list_matches_cpu_stencil_order.cpp` — exists? if so keeps guarding emission order
- `docs/specs/gpu/SPEC.md §7.5` — addendum on per-step invariant (no policy change, just clarify)

No SPEC delta required for Option 3 if we don't change the bond-list interface.

---

## 8. Tests planned

| Test | Command | Gate |
|---|---|---|
| T8.7 byte-exact SNAP GPU | `ctest --test-dir build-mixed-snap-only -R test_snap_gpu_bit_exact` | ≤ 1e-12 rel |
| T6 differential | `ctest -R test_t6_differential` | byte-exact vs CPU FP64 |
| T6 NVE drift 100-step | `./tests/integration/m8_smoke_t6/run_m8_smoke_t6_100step.sh` | ≤ 3e-6 |
| Multi-rank D-M7-10 | `ctest -R test_multirank_td_smoke_2rank` | byte-exact cross-rank |
| T3 anchor-test | `./tests/integration/m5_anchor_test/run_anchor_test.sh` | multi-rank correctness (not wall) |
| Scout re-measure | T6 2000-atom 100-step wall | record ms/step, gap vs KOKKOS |

---

## 9. Risks & open questions

1. **User directive vs. byte-exact reality.** The directive says "byte-exact preserved"; this report says Option 1 cannot deliver that. User should adjudicate: pursue Option 3 (my recommendation), Option 2 (spec-level), or decline to land item 1 and proceed to item 2 (LaunchBounds) directly.
2. **Savings upper bound is modest.** 1.75 ms/step is 5.9 % of 29.5 ms/step. Even perfect closure only moves the MixedFast wall to ~27.75 ms/step → gap vs KOKKOS drops from 6.86× to 6.45×. This alone does not unlock the ≥20 % gate for M8 on 1-GPU, as the memory already notes. T-opt-4 item 3 (fusion) has a larger headroom.
3. **Anchor-test regression check.** The anchor-test fixture is Al FCC Morse — does not exercise SNAP bond list at all. Regression check is *sufficient* (guards scheduler/NL changes that Option 3 does not make) but *not informative* for SNAP-specific bugs. The T6 differential + byte-exact tests are the real SNAP guardrails.
4. **Device-side allocator complexity.** The current 2-pass build guarantees exact sizing; a single-pass build either over-allocates or tracks dynamic counters. Either can break at high-T runs where neighbour counts spike (linked to issue #167).

---

## 10. Decision requested

Reply with:
- **"Option 3, proceed"** → I implement the lean per-step build (byte-exact preserved, ~4–6 % wall win on T6).
- **"Option 2, proceed"** → I implement topology-vs-geometry split (needs SPEC delta, ~3–5 % wall win).
- **"Skip item 1, go to item 2"** → I move to LaunchBounds tuning on snap_yi_kernel; we revisit bond-list in M9+.
- **"Option 1 anyway, I accept non-byte-exact"** → I do not recommend this; would require rewriting T8.7 gate acceptance and is likely to break T6 drift. Please push back if you want to go this way.
