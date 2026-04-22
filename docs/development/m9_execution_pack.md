# M9 Execution Pack

**Document:** `docs/development/m9_execution_pack.md`
**Status:** draft, awaiting human review
**Parent:** `TDMD_Engineering_Spec.md` §14 (M9), `docs/specs/integrator/SPEC.md` §4 (NVT), §5 (NPT), §7.3 (thermostat-in-TD policy), `docs/specs/verify/SPEC.md` (new T8 NVT + T9 NPT canonical benchmarks), `docs/development/m8_execution_pack.md` (template), `docs/development/claude_code_playbook.md` §3
**Milestone:** M9 — NVT baseline (Nosé-Hoover NVT + NPT isotropic, CPU + GPU, Variant A policy) — 8 недель target per master spec §14
**Created:** 2026-04-22 (M8 closed same day, `v1.0.0-alpha1` pushed)
**Author:** Architect / Spec Steward role (Claude Opus 4.7)

---

## 0. Purpose

Этот документ декомпозирует milestone **M9** master spec'а §14 на **14 PR-size
задач** (T9.0..T9.13). Документ — **process artifact**, не SPEC delta; сами
изменения в `integrator/SPEC.md` и `verify/SPEC.md` приходят отдельными
SPEC-delta PR'ами (playbook §9.1) в рамках T9.1 + T9.10 + T9.11.

M9 — **first encounter TDMD с thermostatted ensembles**. После M8 на main
работают: (a) весь reference-path (M3 ≡ M4 ≡ M5 ≡ M6 ≡ M7 Pattern 2 K=1
byte-exact chain, D-M7-10 + D-M8-13); (b) SNAP CPU + GPU + MixedFast (D-M8-7 /
D-M8-13 chain); (c) 6 BuildFlavors шипают including MixedFastSnapOnlyBuild;
(d) `v1.0.0-alpha1` tag pushed 2026-04-22 with honest Case B artifact gate
outcome (D-M8-6). M9 добавляет:

- **`NoseHooverNvtIntegrator` CPU + GPU** — canonical ensemble с chains of M=3
  (default), Trotter decomposition + Yoshida-Suzuki 3rd-order chain update per
  `integrator/SPEC.md` §4.2; thermostat coupling interval default 50 steps
  (§4.5);
- **`NoseHooverNptIntegrator` CPU + GPU** — isotropic volume flex (Parrinello-
  Rahman-like), barostat coupling interval default 100 steps; anisotropic
  (stress tensor) explicitly v2+ per `integrator/SPEC.md` §5.3;
- **PolicyValidator K=1 enforcement** — `integrator.style ∈ {nvt, npt}` ⇒
  `scheduler.pipeline_depth_cap == 1` else reject config с clear error
  (Variant A из `integrator/SPEC.md` §7.3.1; Variant B rejected; Variant C
  deferred to M11 research window);
- **T8 NVT Al FCC canonical benchmark** — 512 atoms, 300 K, 10⁵ steps;
  equipartition ⟨KE⟩ = (3/2)Nk_BT within ±2σ; Maxwell-Boltzmann velocity
  distribution within χ² p > 0.05; `pair_style morse` (M1 baseline — CPU only
  for M9 since Morse GPU is M9 scope only if bandwidth permits, otherwise
  deferred — see §6 R-M9-6); `verify/benchmarks/t8_al_fcc_nvt/`;
- **T9 NPT Al FCC canonical benchmark** — 512 atoms, 300 K, 1 bar, 10⁵ steps;
  equilibrium volume match LAMMPS NPT within 2% relative; isotropic volume
  fluctuations; `verify/benchmarks/t9_al_fcc_npt/`;
- **Differential harness vs LAMMPS NVT/NPT** — 100-step deterministic NVT diff
  (pseudo-canonical: identical velocity seed) + 10⁴-step statistical NPT diff
  (volume mean + fluctuation match within statistical noise);
- **CI gate: NVT/NPT with K>1 rejected** — preflight unit test + integration
  test that asserts clean rejection messages;
- **M9 integration smoke + v1.0.0-beta1 tag** — финальный regression gate + git
  tag fixing NVT baseline.

**Conceptual leap от M8 к M9:**

- M8 = "proof-of-value on ML kernel" (SnapPotential, 6 flavors, Case B artifact
  gate outcome per D-M8-6; `v1.0.0-alpha1` tag).
- **M9 = "thermostatted ensembles baseline"** (NVT + NPT Variant A; K=1 policy
  enforced; production NVT works but без TD speedup — known trade-off per
  `integrator/SPEC.md` §7.3.1; `v1.0.0-beta1` tag).
- M10 = "MEAM integration" (port LAMMPS `pair_style meam/c`; T5 silicon
  benchmark; first potential где TD architecture's advantage на angular-moment
  halo pressure is pronounced — primary hope для proof-of-value если M8 Case B
  outcome persists).
- M11 = "NVT-in-TD research window" (Variant C prototype on
  `research/nvt-in-td` branch; go/no-go decision based on speedup >10% при
  equipartition match).

Критически — **M8 SNAP path + M7 Pattern 2 + M6 EAM GPU path preserved**. Любой
M9 PR проходит: (a) M1..M8 integration smokes (includes m7_smoke, m8_smoke,
m8_smoke_t6, m8_smoke_t6_100step); (b) T1/T4/T6 differentials; (c) T3-gpu
anchor; (d) byte-exact chains D-M7-10 (EAM) + D-M8-13 (SNAP); (e) all 6
BuildFlavors build clean. Zero-regression mandate (master spec §14 M9 inherited
from M8).

**M8 carry-forward — намеренно выделено в orthogonal track, не в M9 critical
path:**

- **#165 EAM per-bond refactor (M9 scout).** Structural refactor mirroring SNAP
  T8.6c-v5 per-bond kernel dispatch для EAM/alloy force kernel. NOT blocking
  M9 NVT baseline; lives on separate branch; lands opportunistically if bandwidth
  permits. Не part of M9 acceptance gate.
- **#167 SNAP GPU robust-failure-mode guard** (defense-in-depth for
  `max_neighbours` overflow at high T per `verify/slow_tier/m8_mixed_fast_snap_only_REPORT.md`
  §9 resolution notes). Orthogonal to M9; no runtime impact на current SNAP
  correctness; standalone PR.
- **T7.8b 30% runtime measurement** — still cloud-burst-gated (≥ 2 physical GPUs
  required). Deferred to M11 v1 beta window along with multi-node scaling
  campaign. Not M9 scope.

После успешного закрытия всех 14 задач и acceptance gate (§5) — milestone M9
завершён, git tag `v1.0.0-beta1` pushed; execution pack для M10 создаётся как
новый аналогичный документ.

---

## 1. Decisions log (зафиксировано до старта T9.1 — this pack)

| # | Решение | Значение | Rationale / источник |
|---|---|---|---|
| **D-M9-1** | NVT chain length default | **M = 3** (Nosé-Hoover chains standard). User may override via `integrator.chain_length: <n>` (range 1..10). | `integrator/SPEC.md` §4.1 + §4.3 — Tuckerman 2010 standard; 1-chain insufficient для ergodicity, >5 chains overkill без statistical benefit. |
| **D-M9-2** | Thermostat coupling interval default | **50 steps** (NVT). User may override via `integrator.thermostat_update_interval: <n>` (constraint: `damping_time / (dt * interval) > 2.0`). | `integrator/SPEC.md` §4.5.1 — GROMACS documentation precedent; 50 steps negligible accuracy cost (~1% overhead) but avoids MPI Allreduce dominating каждый шаг. |
| **D-M9-3** | Barostat coupling interval default | **100 steps** (NPT). User may override via `integrator.barostat_update_interval: <n>`. | `integrator/SPEC.md` §4.5.4 — pressure response intrinsically slower than temperature; longer interval safe. |
| **D-M9-4** | NPT ensemble scope | **Isotropic only в v1.5** (single scalar `isoenthalpic scaling factor`). Anisotropic (stress tensor, non-cubic box) — **explicit v2+**; preflight rejects anisotropic config с clear error pointing к Variant A limitation. | `integrator/SPEC.md` §5.3 — "anisotropic (stress tensor) — v2+"; box flex requires `StateManager::set_box(new_box)` + zoning plan invalidation (infrequent but expensive). |
| **D-M9-5** | Policy validator enforcement | **`integrator.style ∈ {nvt, npt}` ⇒ `scheduler.pipeline_depth_cap == 1`** else preflight reject with verbose rationale (suggests `style: nve` or `pipeline_depth_cap: 1`). Enforced в `PolicyValidator::check` per `integrator/SPEC.md` §7.3.1 Variant A. | `integrator/SPEC.md` §7.3.1 — Nose-Hoover chains require global sync каждый step; K>1 batched pipeline would use inconsistent thermostat state; Variant B (per-zone) rejected; Variant C deferred M11. |
| **D-M9-6** | K=1 TD for NVT/NPT trade-off | **Accepted and documented**. NVT/NPT production работает correctly in multi-rank Pattern 2 but без TD speedup (effectively SD — все zones на одном time_level). NVE retains full K>1 TD speedup. Scientist-facing docs (`docs/user/ensembles.md` new at T9.9) spell out this trade-off explicitly. | `integrator/SPEC.md` §7.3.1 Variant A — scientific correctness первична; known v1.5 limitation; Variant C (lazy thermostat, v2+ research) documented pathway. |
| **D-M9-7** | LAMMPS NVT/NPT oracle | **Reuse existing submodule infrastructure** (M1 T1.11 landed; same path `verify/third_party/lammps/install_tdmd/lmp`). LAMMPS `fix nvt temp` + `fix npt temp iso` are default-enabled in MANYBODY+MISC packages (already built); no new CMake option. | D-M8-2 precedent; memory `project_m8_t813_blocker.md` confirms LAMMPS oracle already exercised на SNAP path; NVT/NPT is MANYBODY-adjacent, no new packages required. |
| **D-M9-8** | T8/T9 canonical fixture | **Al FCC 512 atoms (8×8×8 conventional = 4 atoms/cell × 8³ = 2048 atoms)** WAIT — 512 atoms = 2×4³=128 conventional cells × 4 atoms = 512 atoms = 5×5×5 FCC conventional. Verified via `generate_setup.py --structure fcc_al --nrep 5` (master spec §14 M9 + integrator/SPEC §4.4/§9.3). Pair style = **`morse`** (M1 baseline; CPU only — Morse GPU is M9+ optional scope) OR **`eam/alloy`** single-species Al (reuse T3 fixture). **Decision locked: eam/alloy Al** — leverages existing GPU path per M6 T6.5 precedent; Morse GPU deferred (see R-M9-6). Canonical path: `verify/benchmarks/t8_al_fcc_nvt/setup.data` (new, 512-atom FCC Al) + `config.yaml` (nvt, 300 K, 10⁵ steps) + `setup_npt.data` shared or new for T9. | Integrator/SPEC §9.3 — "NVT Al FCC 512 atoms"; master spec §14 M9 — "NVT Al FCC 512 atoms, 10⁵ steps"; EAM/alloy path preferred over Morse для GPU coverage, reuses M6 kernel. |
| **D-M9-9** | Differential gates for thermostatted ensembles | **Statistical, not byte-exact.** NVT/NPT с stochastic thermostat variables (even deterministic Nosé-Hoover propagates numerical ε differently than LAMMPS due to different Trotter operator order) diverges byte-from-byte within ~100 steps. Gates: (a) 100-step **velocity-Verlet-core diff** — same initial state, same velocity seed, identical thermostat params: positions + velocities match LAMMPS ≤ 1e-10 rel at step 100 (deterministic stochastic-free gate); (b) 10⁴-step **statistical diff** — ⟨T⟩ match within ±3 K (1% rel at 300 K), ⟨KE⟩ within ±2σ of analytic equipartition, velocity distribution Maxwell-Boltzmann χ² p > 0.05; (c) NPT: ⟨V⟩ match LAMMPS NPT within 2% rel, volume fluctuation σ_V match within 5% rel. | `integrator/SPEC.md` §4.4 (NVT validation) + §9.3 (NVT tests) + §9.4 (differential vs LAMMPS); byte-exact не applicable to thermostatted ensembles. |
| **D-M9-10** | GPU extension strategy | **Reuse M6 T6.6 VelocityVerletGpu precedent** — thread-per-atom, no atomics, `__restrict__` on all pointer params (master spec §D.16). NoseHooverNvtGpu adds two host→device reductions per thermostat_update_interval (total KE + total virial for NPT); both через existing `DevicePool` pinned-host buffers + canonical Kahan host-side reduction per D-M6-15. No new device allocation pattern. | `integrator/SPEC.md` §3.5 GPU-resident mandate; §8.3 `__restrict__` canonical signature; D-M6-7 byte-exact chain extended to NVT via Reference-flavor `--fmad=false` build. |
| **D-M9-11** | Trotter decomposition order | **Fixed per `integrator/SPEC.md` §4.2** (7-step Trotter): (1) chain dt/2 → (2) `v *= exp(-ξ_1·dt/2)` → (3) VV pre_force → (4) compute forces → (5) VV post_force → (6) `v *= exp(-ξ_1·dt/2)` → (7) chain dt/2. Yoshida-Suzuki 3rd-order used inside chain update per Tuckerman 2010. | `integrator/SPEC.md` §4.2 — canonical Trotter for Nosé-Hoover chains; order matters для bit-exact gate at byte level (LAMMPS uses same order in `fix_nh.cpp`). |
| **D-M9-12** | Active BuildFlavors (M9 closure) | **No new flavor.** Carry forward 6 шипающих flavors from M8: `Fp64ReferenceBuild` (oracle) + `Fp64ProductionBuild` + `MixedFastBuild` + `MixedFastSnapOnlyBuild` + `MixedFastAggressiveBuild` (opt-in) + `Fp32ExperimentalBuild` (opt-in). NVT/NPT integrators template on `NumericConfig`; work identically across all 6. | §D.11 matrix stable at 6 flavors; §D.17 procedure not re-invoked in M9; CI matrix size preserved. |
| **D-M9-13** | Active ExecProfiles | **Unchanged** — `Reference` (byte-exact gate) + `Production` (performance tuning). `Fast` остаётся deferred v2+ (no change from M8). | D-M8-11 carry-forward. |
| **D-M9-14** | CI strategy (M9 addition) | **Option A continues (D-M6-6 / D-M8-12 carry-forward).** New `m9_smoke` harness follows m6/m7/m8 convention: self-skips on no-CUDA (exit 0), exit 77 on missing LAMMPS submodule, exit 1 on physics regression. No new matrix cell required — NVT/NPT integrators build under existing flavors × existing MPI/non-MPI matrix. Cloud-burst out of scope в M9 (no scaling gate). | Memory `project_option_a_ci.md`; M9 is correctness milestone, not performance milestone. |
| **D-M9-15** | Byte-exact chain extension (M9) | **NVE byte-exact chain preserved** (D-M7-10 EAM + D-M8-13 SNAP unchanged). **NVT/NPT: separate regression chain** — bit-exact CPU↔GPU under Reference+`--fmad=false` on single-rank only (D-M6-7 extension); no multi-rank byte-exact claim для NVT (global thermostat reduction breaks bitwise determinism under Kahan ring if zones commit in non-canonical order — mitigated via D-M5-9 Kahan-ring but not gated bit-exactly at multi-rank level для M9). Statistical gate covers multi-rank NVT/NPT correctness. | `integrator/SPEC.md` §7.3.1 + master spec §13.5 — Reference profile byte-exact oracle is per-flavor per-integrator; NVT chain rooted at T9.5 (GPU bit-exact vs CPU) on single-rank; multi-rank statistical validation via T9.10/T9.11 canonical benchmarks. |
| **D-M9-16** | Timeline | **8 недель target, 10 acceptable, flag at 12**. Most expensive: T9.2 NoseHoover CPU (~5 days — Trotter plumbing + Yoshida-Suzuki chain), T9.4 NoseHoover GPU (~4 days — mirrors T6.6 pattern), T9.6 NPT CPU (~4 days — box flex + zoning invalidation), T9.10/T9.11 benchmarks (~3 days each — 10⁵-step runs + statistical harness). Остальные 2-4 дня. | Master spec §14 M9 = 8 недель. M8 shipped в ~42 дня (budget 42 = 6 нед × 7); M9 has comparable surface area so 8 нед realistic. |
| **D-M9-17** | v1 beta tag format | **`v1.0.0-beta1`** annotated git tag pushed at T9.13 closure. Release notes extend M8 `v1.0.0-alpha1` format: NVT/NPT gates met; T8/T9 canonical benchmarks landed; K=1 policy enforced; known trade-off documented. | Semver convention: alpha = M8 (SNAP proof-of-value); beta = M11 (NVT-in-TD research + v1 feature-complete sans long-range); 1.0 = M13 (long-range + v2.0 readiness). M9 ships **mid-alpha** tag `v1.0.0-beta1` to mark NVT baseline — terminology reconciliation with master spec §14 post-v1 roadmap: *beta1 reserves "feature-complete v1" label for M11; M9 tag is a beta milestone step not a full v1 beta release*. |

---

## 2. Глобальные параметры окружения

| Параметр | Значение | Примечание |
|---|---|---|
| OS | Linux (Ubuntu 24.04 LTS) | Dev-машина пользователя; ubuntu-latest в CI |
| C++ compiler | GCC 13+ / Clang 17+ | C++20; CI проверяет оба (M6..M8 matrix) |
| CMake | 3.25+ | Master spec §15.2 |
| CUDA | **13.1** installed (system `/usr/local/cuda`) | Memory `env_cuda_13_path.md`; CI compile-only |
| GPU archs | sm_80, sm_86, sm_89, sm_90, sm_100, sm_120 | D-M6-1 carry-forward |
| MPI | **CUDA-aware OpenMPI ≥4.1** preferred; non-CUDA-aware → fallback | D-M7-3 carry-forward |
| NCCL | **≥2.18** (bundled с CUDA 13.x) | D-M7-4 carry-forward; intra-node only |
| LAMMPS oracle | **Shipped M1 T1.11** — submodule `verify/third_party/lammps/` pinned `stable_22Jul2025_update4`; `tools/build_lammps.sh` builds with MANYBODY+ML-SNAP+MISC packages enabled by default; install prefix `verify/third_party/lammps/install_tdmd/`. NVT/NPT via `fix nvt` / `fix npt iso` (default-enabled). | D-M9-7; SKIP on public CI per Option A |
| Python | 3.10+ | pre-commit + anchor-test + T8/T9 statistical harness + differential runner |
| Test framework | Catch2 v3 (FetchContent) + MPI wrapper | GPU+MPI tests local-only per D-M7-11 |
| Active BuildFlavors | `Fp64ReferenceBuild`, `Fp64ProductionBuild`, `MixedFastBuild`, `MixedFastSnapOnlyBuild`, `MixedFastAggressiveBuild`, `Fp32ExperimentalBuild` (6 шипают из M8) | D-M9-12 |
| Active ExecProfiles | `Reference`, `Production` (GPU) | D-M9-13 |
| Run mode | multi-rank MPI × GPU-per-rank × 1:1 subdomain binding (M7 carry) | D-M7-2 carry-forward |
| Pipeline depth K | `{1, 2, 4, 8}` для NVE; **`{1}` only для NVT/NPT** (enforced D-M9-5) | D-M9-5 new; NVE unchanged |
| Subdomain topology | Cartesian 1D/2D/3D (M7 shipped 2D; 3D config allowed but cloud-burst-tested only) | D-M7-1 carry-forward |
| Streams per rank | 2 (default) — compute + mem | D-M6-13 carry; T8.0 2-rank infra shipped |
| CI CUDA | compile-only matrix: unchanged от M8 | D-M9-14 |
| Local pre-push gates | Full GPU suite + T3-gpu + M1..M8 smokes + **M9 smoke** (added T9.13) | D-M7-17 carry + T9.13 addition |
| Cloud burst trigger | **None в M9** — scaling measurements не part of M9 scope | D-M9-14 |
| Branch policy | `m9/T9.X-<topic>` per PR → `main` | CI required: lint + build-cpu + build-gpu + M1..M8 smokes; M9 smoke добавляется в T9.13 |

---

## 3. Suggested PR order

**Dependency graph:**

```
T9.0 (this pack, meta-task) ──┐
                              │
T9.1 (SPEC delta: integrator  │
      + verify SPEC finalize) │
                              ▼
                         T9.2 (NoseHooverNvt CPU — Trotter + chains)
                              │
                              ▼
                         T9.3 (NVT CPU diff vs LAMMPS — velocity-Verlet-core)
                              │
                              ▼
                         T9.4 (NoseHooverNvt GPU — thread-per-atom kernel)
                              │
                              ▼
                         T9.5 (NVT GPU bit-exact gate vs CPU — D-M6-7 extension)
                              │
                              ▼
                         T9.6 (NoseHooverNpt CPU — box flex + zoning invalidate)
                              │
                              ▼
                         T9.7 (NPT CPU diff vs LAMMPS — volume statistics)
                              │
                              ▼
                         T9.8 (NoseHooverNpt GPU)
                              │
                              ▼
                         T9.9 (PolicyValidator K=1 enforcement + scientist docs)
                              │
                              ▼
                         T9.10 (T8 NVT Al FCC canonical benchmark + harness)
                              │
                              ▼
                         T9.11 (T9 NPT Al FCC canonical benchmark + harness)
                              │
                              ▼
                         T9.12 (Equipartition + Maxwell-Boltzmann statistical
                                harness shared T8 + T9)
                              │
                              ▼
                         T9.13 (M9 integration smoke + v1.0.0-beta1 tag)
```

**Линейная последовательность (single agent):**
T9.0 → T9.1 → T9.2 → T9.3 → T9.4 → T9.5 → T9.6 → T9.7 → T9.8 → T9.9 → T9.10 →
T9.11 → T9.12 → T9.13.

**Параллельный режим (multi-agent):** M9 has a **moderately linear dep chain**
— NVT path (T9.2..T9.5) + NPT path (T9.6..T9.8) could in principle parallelize
after T9.2 lands, но NPT builds on NVT's Trotter + chain infrastructure so
serial saves duplicate work. T9.9 PolicyValidator + T9.10/T9.11 benchmarks
could land parallel to T9.8 GPU port (different files, different reviewers).
T9.12 statistical harness depends на T9.10/T9.11 fixtures — must be serial.

**Estimated effort:** 8 недель target (single agent, per D-M9-16).
Breakdown: T9.1 ~2 дня; T9.2 ~5 дней; T9.3 ~2 дня; T9.4 ~4 дня; T9.5 ~1 день;
T9.6 ~4 дня; T9.7 ~2 дня; T9.8 ~3 дня; T9.9 ~2 дня; T9.10 ~3 дня; T9.11 ~3 дня;
T9.12 ~2 дня; T9.13 ~1 день. Sum ~34 дня = 5 нед серьёзной работы + 3 нед
buffer для SPEC review cycles, incident debugging, and M8 carry-forward merge
windows.

---

## 4. Tasks

### T9.0 — M9 execution pack authored (this document)

```
# TDMD Task: Author M9 execution pack — NVT baseline milestone planning

## Context
- Master spec: §14 M9 (NVT baseline, 8 weeks)
- Module SPECs: `docs/specs/integrator/SPEC.md` §4 (NVT), §5 (NPT), §7.3 (thermostat-in-TD)
- Predecessor: m8_execution_pack.md (shipped 2026-04-20..22, 1770 lines)

## Objective
Author `docs/development/m9_execution_pack.md` with 14 PR-size tasks (T9.0..T9.13)
spanning integrator/SPEC §4/§5 implementation + T8/T9 canonical benchmarks +
PolicyValidator + integration smoke + v1.0.0-beta1 tag.

## Scope
- Docs-only PR; no code changes.
- Structure matches M8 pack: Purpose / Decisions log / Environment / PR order
  / Tasks / Acceptance Gate / Risks & OQ / Roadmap Alignment.
- Integrator/SPEC + verify/SPEC deltas tracked as T9.1 / T9.10 / T9.11
  sub-PRs (playbook §9.1 spec-delta procedure).

## Acceptance
- Pack reviewed by human + committed on main.
- v1.0.0-beta1 tag format + semver convention explicitly reconciled with
  master spec §14 (D-M9-17).
- Parallel-track notes for orthogonal carry-forwards (#165, #167,
  T7.8b runtime) spelled out in §6.

## Status
CLOSED on commit landing this pack.
```

### T9.1 — SPEC delta: integrator/SPEC.md §4/§5 finalize + verify/SPEC T8/T9 register

```
# TDMD Task: SPEC delta for NVT/NPT implementation readiness

## Context
- `integrator/SPEC.md` §4 (NVT) + §5 (NPT) authored в v2.0 but tagged
  "v1.5 — M9+" — need implementation-ready body (YAML surface final,
  Trotter order locked, parameter ranges explicit).
- `verify/SPEC.md` needs T8 NVT + T9 NPT canonical benchmark registration
  with threshold-registry anchors.
- Playbook §9.1: interface/architectural changes → separate SPEC delta PR
  BEFORE implementation PRs.

## Objective
Ship two SPEC deltas on branch `spec-delta-m9-integrator`:
1. `integrator/SPEC.md` §4/§5 promoted from "v1.5 — M9+" placeholder → full
   implementation spec. Lock YAML schema, Trotter step order, thermostat/
   barostat interval constraint validation, Variant A policy text.
2. `verify/SPEC.md` new §4.8 (T8 NVT) + §4.9 (T9 NPT) benchmark descriptions.
3. `verify/thresholds/thresholds.yaml` new `benchmarks.t8_al_fcc_nvt.*` +
   `benchmarks.t9_al_fcc_npt.*` threshold entries (reserved until T9.10/T9.11
   measurements land; promoted to ACTIVE at benchmark close).
4. Master spec Приложение C T9.1 addendum.

## Files changed
- `docs/specs/integrator/SPEC.md` (§4/§5 promote + §9.3/§9.4 updated)
- `docs/specs/verify/SPEC.md` (new §4.8 + §4.9)
- `verify/thresholds/thresholds.yaml` (new benchmarks.t8_al_fcc_nvt + t9_al_fcc_npt sections, reserved status)
- `TDMD_Engineering_Spec.md` Приложение C v2.5 block (T9.1 addendum prepended)

## Acceptance
- Markdownlint green.
- Integrator/SPEC YAML schema block is implementation-ready (no placeholders).
- Verify/SPEC T8/T9 descriptions sufficient for T9.10/T9.11 to proceed
  without further spec clarification.
- Threshold values derived from integrator/SPEC §4.4 + §5 validation
  criteria (equipartition ±2σ, χ² p>0.05, volume 2% rel, fluctuation 5% rel).

## Status
pending
```

### T9.2 — NoseHooverNvtIntegrator CPU (Trotter + Yoshida-Suzuki chains)

```
# TDMD Task: NoseHooverNvtIntegrator CPU implementation

## Context
- integrator/SPEC.md §4.1..§4.3 (chain math + Trotter 7-step)
- §4.5.1 (thermostat_update_interval policy)
- §4.5.3 (validation constraint: damping_time / (dt * interval) > 2.0)
- Precedent: `src/integrator/velocity_verlet_integrator.cpp` (M1)
- Output of this task: CPU FP64 NVT ready for T9.3 diff vs LAMMPS.

## Objective
Ship `src/integrator/nose_hoover_nvt_integrator.{hpp,cpp}` implementing
integrator/SPEC §4.2 Trotter decomposition (7 steps). Yoshida-Suzuki 3rd-order
chain update per Tuckerman 2010. FP64 only в T9.2; mixed precision deferred
to T9.4 (GPU variant).

## Files changed
- `src/integrator/include/tdmd/integrator/nose_hoover_nvt_integrator.hpp` (new — interface)
- `src/integrator/nose_hoover_nvt_integrator.cpp` (new — implementation)
- `src/integrator/CMakeLists.txt` (+1 source)
- `src/io/preflight.cpp` (accept `integrator.style: nvt`)
- `src/io/yaml_config.cpp` (parse `integrator.{temperature, damping_time, chain_length, thermostat_update_interval}`)
- `src/runtime/simulation_engine.cpp` (dispatch on integrator.style; route to Nvt when requested)
- `tests/integrator/test_nose_hoover_nvt.cpp` (new — unit tests per integrator/SPEC §9.1/§9.3)

## Tests (Catch2)
- Single-atom harmonic: equipartition ⟨KE⟩ = (1/2)k_BT within 3σ at 10⁴ steps.
- Chain length ∈ {1, 3, 5}: equipartition held; chain=1 rejected if ergodicity
  metric poor (documented caveat).
- Trotter symmetry: forward 100 steps + reverse velocity + 100 steps back
  returns to (x, -v) within FP64 round-off (integrator/SPEC §9.1).
- thermostat_update_interval ∈ {1, 50, 500}: equipartition held; interval=500
  with dt=1 fs + damping=0.1 ps triggers preflight rejection.

## Acceptance
- All 6 BuildFlavors build clean (NumericConfig-templated).
- Full ctest bank green — no regressions on M1..M8 suites.
- Equipartition ±2σ met on 512-atom Al FCC canonical fixture at 10⁵ steps
  (exercised via harness, not unit test — T9.10 formalizes benchmark).

## Status
pending
```

### T9.3 — NVT CPU differential vs LAMMPS

```
# TDMD Task: NVT CPU differential gate vs LAMMPS fix nvt

## Context
- integrator/SPEC.md §9.4 — "NVT: long run; observables match LAMMPS within
  statistical noise"
- D-M9-9 differential gates — 100-step velocity-Verlet-core diff + 10⁴-step
  statistical diff
- LAMMPS oracle: `verify/third_party/lammps/install_tdmd/lmp` with
  `fix 1 all nvt temp 300.0 300.0 0.1`

## Objective
Ship `verify/benchmarks/t8_al_fcc_nvt_preview/` (new, preliminary fixture —
formal canonical benchmark at T9.10) + `verify/t8/run_differential.py` (new,
clones T4 runner pattern) + `tests/integrator/test_t8_nvt_differential.cpp`
(Catch2 wrapper) exercising:
- 100-step identical-seed velocity-Verlet-core comparison (positions +
  velocities ≤ 1e-10 rel at step 100).
- 10⁴-step statistical comparison (⟨T⟩ ±3 K, equipartition ±2σ, M-B χ² p>0.05).

## Files changed
- `verify/benchmarks/t8_al_fcc_nvt_preview/` (new — 512-atom Al FCC, setup.data,
  config.yaml, lammps_script.in, README.md)
- `verify/t8/run_differential.py` (new — NVT-aware extension)
- `tests/integrator/test_t8_nvt_differential.cpp` (new — Catch2 harness)

## Acceptance
- 100-step rel ≤ 1e-10 force/position/velocity match LAMMPS.
- 10⁴-step statistical gate passes (human review on the emitted histograms).
- SKIP_RETURN_CODE 77 on uninitialized LAMMPS submodule.
- No regressions in M1..M8 differential suite.

## Status
pending; depends T9.2
```

### T9.4 — NoseHooverNvtIntegrator GPU

```
# TDMD Task: NoseHooverNvtIntegrator GPU path

## Context
- integrator/SPEC.md §3.5 — GPU-resident mandate (no CPU-GPU ping-pong per-step)
- D-M9-10 — reuse M6 T6.6 VV GPU pattern (thread-per-atom, no atomics)
- gpu/SPEC §9 — adapter facade + PIMPL compile firewall

## Objective
Ship `src/gpu/nose_hoover_nvt_gpu.{hpp,cu}` + `src/gpu/include/tdmd/gpu/nose_hoover_nvt_gpu.hpp`
(PIMPL) + `src/integrator/nose_hoover_nvt_gpu_adapter.{hpp,cpp}` (domain facade).
Two kernels on existing compute stream:
1. `nose_hoover_nvt_pre_force_kernel` — chain update (host-side chain math;
   device kernel does velocity rescale + VV pre_force half-kick + drift).
2. `nose_hoover_nvt_post_force_kernel` — VV post_force half-kick + velocity
   rescale + chain update (host-side).

Per thermostat_update_interval N: host performs chain dt·N/2 update using
accumulated KE from prior N steps; device kernels use frozen ξ_1 value between
updates.

## Files changed
- `src/gpu/nose_hoover_nvt_gpu.cu` (new — ~500 lines)
- `src/gpu/include/tdmd/gpu/nose_hoover_nvt_gpu.hpp` (new — PIMPL)
- `src/gpu/CMakeLists.txt` (+1 source)
- `src/integrator/nose_hoover_nvt_gpu_adapter.hpp` (new)
- `src/integrator/nose_hoover_nvt_gpu_adapter.cpp` (new)
- `src/runtime/simulation_engine.cpp` (wire GPU NVT dispatch)
- `tests/gpu/test_nose_hoover_nvt_gpu.cpp` (new — functional tests)

## Tests (Catch2)
- CPU stub throws on CPU-only build.
- Functional: 10-step NVT GPU vs NVT CPU ≤ 1e-10 rel on 512-atom Al FCC
  (bit-exact gate at T9.5; T9.4 is functional).
- NVTX audit green (`test_nvtx_audit` walks `src/gpu/*.cu`).

## Acceptance
- All 6 flavors compile (Fp64Ref, Fp64Prod, Mixed, MixedSnapOnly, MixedAggr, Fp32Exp).
- CUDA build 3 flavors (Ref+CUDA, Mixed+CUDA, MixedSnapOnly+CUDA) all green.
- CPU-only-strict green.

## Status
pending; depends T9.2
```

### T9.5 — NVT GPU bit-exact gate (D-M6-7 extension)

```
# TDMD Task: NVT GPU FP64 bit-exact vs CPU FP64 gate

## Context
- D-M6-7 chain: CPU FP64 ≡ GPU FP64 на Reference+--fmad=false build for
  all (potential × integrator) combinations
- D-M9-15 — NVT chain rooted at T9.5 on single-rank; multi-rank statistical
  (not bit-exact) gate at T9.10

## Objective
Ship `tests/gpu/test_nose_hoover_nvt_bit_exact.cpp` asserting:
- 100-step NVT на 512-atom Al FCC: step-wise positions + velocities +
  thermostat ξ chain + pe + ke + virial match CPU FP64 ≤ 1e-12 rel.
- Uses Fp64ReferenceBuild + `--fmad=false` + NumericConfig=Reference.
- Chain length ∈ {1, 3, 5} exercised.

## Files changed
- `tests/gpu/test_nose_hoover_nvt_bit_exact.cpp` (new)
- `tests/gpu/CMakeLists.txt` (+1 target + SKIP_RETURN_CODE 77)
- `verify/thresholds/thresholds.yaml` — `benchmarks.t8_al_fcc_nvt.gpu_fp64_vs_cpu_fp64.*` promoted reserved → ACTIVE with 1e-12 rel gate

## Acceptance
- 1e-12 rel held (expected ≤ 1e-13 on commodity hardware per T6.6 VV precedent).
- D-M9-15 chain extended.

## Status
pending; depends T9.4
```

### T9.6 — NoseHooverNptIntegrator CPU (isotropic)

```
# TDMD Task: NoseHooverNptIntegrator CPU implementation (isotropic only)

## Context
- integrator/SPEC.md §5.1..§5.3 — Parrinello-Rahman-like isotropic
- D-M9-4 — anisotropic rejected в v1.5
- Box flex requires `StateManager::set_box(new_box)` + neighbor rebuild

## Objective
Ship `src/integrator/nose_hoover_npt_integrator.{hpp,cpp}` extending NVT
with isotropic volume flex. Barostat variable η coupled к pressure residual
P - P_target. Box scales as V ← V · exp(η · dt). Neighbor rebuild on
box change per D-M3-* infrastructure (already exists в CellGrid).

Anisotropic YAML (integrator.pressure_matrix or style: npt/aniso) → preflight
reject с message pointing to v2+ roadmap.

## Files changed
- `src/integrator/include/tdmd/integrator/nose_hoover_npt_integrator.hpp` (new)
- `src/integrator/nose_hoover_npt_integrator.cpp` (new)
- `src/integrator/CMakeLists.txt` (+1 source)
- `src/io/preflight.cpp` (accept `integrator.style: npt`; reject aniso)
- `src/io/yaml_config.cpp` (parse pressure, pressure_damping, barostat_update_interval)
- `src/runtime/simulation_engine.cpp` (NPT dispatch)
- `src/state/state_manager.cpp` (if needed — box mutation API audit)
- `tests/integrator/test_nose_hoover_npt.cpp` (new)

## Tests
- 512-atom Al FCC at 1 bar: equilibrium volume matches NVE+relax result within 0.5%.
- Anisotropic config → clean preflight reject.
- Volume fluctuation σ_V > 0 (non-degenerate) at 10⁴ steps.

## Status
pending; depends T9.2
```

### T9.7 — NPT CPU differential vs LAMMPS

```
# TDMD Task: NPT CPU differential gate vs LAMMPS fix npt iso

## Context
- D-M9-9 — NPT gates: ⟨V⟩ match LAMMPS 2% rel, σ_V match 5% rel
- LAMMPS: `fix 1 all npt temp 300.0 300.0 0.1 iso 1.0 1.0 1.0`

## Objective
Mirror T9.3 pattern for NPT — 10⁴-step statistical diff (no byte-level
comparison due to barostat stochasticity + FP-order sensitivity). Volume
trajectory compared statistically.

## Files changed
- `verify/benchmarks/t9_al_fcc_npt_preview/` (new)
- `verify/t9/run_differential.py` (new)
- `tests/integrator/test_t9_npt_differential.cpp` (new)

## Acceptance
- ⟨V⟩ match LAMMPS 2% rel.
- σ_V match LAMMPS 5% rel.
- 100-step deterministic-core diff (positions/velocities ≤ 1e-10 rel) —
  NPT box scaling is deterministic given identical η₀ seed.
- SKIP_RETURN_CODE 77 on uninitialized LAMMPS submodule.

## Status
pending; depends T9.6
```

### T9.8 — NoseHooverNptIntegrator GPU

```
# TDMD Task: NoseHooverNptIntegrator GPU path

## Context
- Mirrors T9.4 pattern but with box flex kernel (scale all positions by
  exp(η · dt / 3) per cartesian axis — isotropic)
- Box mutation requires host-side coordination; kernel does per-atom scale
  + VV half-kick; host updates box dimensions + triggers neighbor rebuild

## Objective
Ship `src/gpu/nose_hoover_npt_gpu.{hpp,cu}` + adapter + engine wiring.
Neighbor rebuild on box change handled by existing neighbor/ module
(triggered by `CellGrid::rebuild_on_box_change()`).

## Files changed
- `src/gpu/nose_hoover_npt_gpu.cu` (new)
- `src/gpu/include/tdmd/gpu/nose_hoover_npt_gpu.hpp` (new)
- `src/gpu/CMakeLists.txt` (+1 source)
- `src/integrator/nose_hoover_npt_gpu_adapter.{hpp,cpp}` (new)
- `src/runtime/simulation_engine.cpp` (NPT GPU dispatch)
- `tests/gpu/test_nose_hoover_npt_gpu.cpp` (new — functional + bit-exact vs CPU)

## Acceptance
- Functional + bit-exact (1e-12 rel vs CPU FP64, Reference+--fmad=false).
- NVTX audit green.
- All 3 CUDA flavors green.

## Status
pending; depends T9.6
```

### T9.9 — PolicyValidator K=1 enforcement + scientist docs

```
# TDMD Task: PolicyValidator K=1 enforcement for NVT/NPT + scientist-facing docs

## Context
- D-M9-5 — Variant A policy: integrator.style ∈ {nvt, npt} ⇒ pipeline_depth_cap=1
- integrator/SPEC.md §7.3.1 — clear error message format documented
- D-M9-6 — scientist docs spell out trade-off

## Objective
Enforce D-M9-5 in `src/runtime/policy_validator.cpp` (or equivalent existing
module). Preflight reject with verbose rationale. New user guide
`docs/user/ensembles.md` explains NVE/NVT/NPT trade-offs and K=1 limitation
explicitly.

## Files changed
- `src/runtime/policy_validator.cpp` (or equivalent) — new check + error message
- `src/io/preflight.cpp` (invoke new check)
- `tests/io/test_preflight.cpp` — new test cases: (nvt, K>1) → reject;
  (npt, K>1) → reject; (nve, K>1) → accept; (nvt, K=1) → accept
- `docs/user/ensembles.md` (new, ~200 lines)
- `docs/user/build_flavors.md` (cross-ref to new ensembles doc)

## Acceptance
- All rejection paths produce error messages matching integrator/SPEC §7.3.1
  format.
- Scientist doc approved by Architect.

## Status
pending; depends T9.2 + T9.6
```

### T9.10 — T8 NVT Al FCC canonical benchmark

```
# TDMD Task: T8 NVT Al FCC 512-atom canonical benchmark

## Context
- Master spec §14 M9 artifact gate: NVT Al FCC 512 atoms, 10⁵ steps,
  equipartition ±2σ, M-B χ² p>0.05
- verify/SPEC §4.8 (registered at T9.1)
- integrator/SPEC §9.3 (validation criteria)

## Objective
Ship `verify/benchmarks/t8_al_fcc_nvt/` canonical fixture + harness hook.
Generator + setup.data + config.yaml + lammps_script.in + README.md + checks.yaml.
Integrate into VerifyLab threshold registry and T8 differential harness.

## Files changed
- `verify/benchmarks/t8_al_fcc_nvt/generate_setup.py` (new)
- `verify/benchmarks/t8_al_fcc_nvt/setup.data` (new — 512-atom Al FCC, LFS? — fixture small enough to inline)
- `verify/benchmarks/t8_al_fcc_nvt/config.yaml` (new — EAM/alloy Al, 10⁵ steps, thermostat interval 50)
- `verify/benchmarks/t8_al_fcc_nvt/lammps_script.in` (new — LAMMPS fix nvt oracle)
- `verify/benchmarks/t8_al_fcc_nvt/checks.yaml` (new)
- `verify/benchmarks/t8_al_fcc_nvt/README.md` (new)
- `verify/thresholds/thresholds.yaml` — `benchmarks.t8_al_fcc_nvt.*` promoted reserved → ACTIVE

## Acceptance
- 10⁵-step run completes < 30 min on RTX 5080.
- Equipartition ±2σ, M-B χ² p>0.05 reproducibly on 3 different random seeds.
- Statistical harness (T9.12) reports both metrics in summary JSON.

## Status
pending; depends T9.2 + T9.4 + T9.9
```

### T9.11 — T9 NPT Al FCC canonical benchmark

```
# TDMD Task: T9 NPT Al FCC 512-atom canonical benchmark

## Context
- Master spec §14 M9 artifact gate: "Same для NPT isotropic" as T8
- verify/SPEC §4.9 (registered at T9.1)

## Objective
Mirror T9.10 for NPT. Equilibrium volume + volume fluctuation are the
canonical observables (vs temperature distribution for NVT).

## Files changed
- `verify/benchmarks/t9_al_fcc_npt/` (new — mirror structure of t8_al_fcc_nvt)
- `verify/thresholds/thresholds.yaml` — `benchmarks.t9_al_fcc_npt.*` promoted reserved → ACTIVE

## Acceptance
- 10⁵-step run completes < 45 min on RTX 5080.
- ⟨V⟩ matches LAMMPS NPT iso within 2% rel.
- σ_V matches LAMMPS NPT iso within 5% rel.
- Equipartition ±2σ held (NPT preserves equipartition).

## Status
pending; depends T9.6 + T9.8 + T9.9
```

### T9.12 — Statistical harness (equipartition + Maxwell-Boltzmann)

```
# TDMD Task: Statistical validation harness for thermostatted ensembles

## Context
- integrator/SPEC §9.3 — equipartition + M-B chi² test
- T8 + T9 both need the same statistical primitives

## Objective
Ship `verify/harness/thermostat_statistics.py` (new) implementing:
- ⟨KE⟩ = (3/2) N k_B T equipartition with ±σ bounds from 3σ/√N_samples formula
- Maxwell-Boltzmann chi² goodness-of-fit on 3D velocity distribution
  (binning, expected density f(v) = 4πv² (m/2πk_BT)^(3/2) exp(-mv²/2k_BT))
- Volume histogram analysis for NPT (normality + fluctuation σ_V)

Hook into T8 + T9 runners so `run_differential.py` emits a summary JSON with
equipartition_passed, m_b_chi2_p_value, volume_match_rel, fluctuation_match_rel.

## Files changed
- `verify/harness/thermostat_statistics.py` (new)
- `verify/t8/run_differential.py` — call thermostat_statistics at end
- `verify/t9/run_differential.py` — call thermostat_statistics at end
- `tests/harness/test_thermostat_statistics.py` (new — pytest unit tests
  against analytical M-B samples + known equipartition references)

## Acceptance
- Harness passes 3 synthetic tests: perfect M-B sample (χ² p > 0.3), biased
  sample (χ² p < 0.001 — detect bias), small-sample (noise floor).
- Integrated into T8 + T9 benchmarks.

## Status
pending; depends T9.10 + T9.11
```

### T9.13 — M9 integration smoke + v1.0.0-beta1 tag

```
# TDMD Task: M9 integration smoke + v1.0.0-beta1 tag ship

## Context
- Sibling smokes: m7_smoke (EAM Pattern 2), m8_smoke (SNAP Pattern 2),
  m8_smoke_t6 (SNAP NVE drift)
- M9 scope: NVT single-rank (Variant A K=1 only — no TD speedup)

## Objective
Ship `tests/integration/m9_smoke/` — single-rank 10-step NVT Al FCC
smoke. Byte-for-byte equal to T9.5 GPU bit-exact output на Reference+--fmad=false
build. Self-skip on no-CUDA (exit 0).

Ship `CHANGELOG.md` update: `[v1.0.0-beta1] — YYYY-MM-DD` block with M9
acceptance gates table + what's new + known limitations + quality gates +
roadmap link to M10.

Create annotated tag `v1.0.0-beta1`.

## Files changed
- `tests/integration/m9_smoke/run_m9_smoke.sh` (new, executable)
- `tests/integration/m9_smoke/smoke_config.yaml.template` (new)
- `tests/integration/m9_smoke/thermo_golden.txt` (generated at bring-up)
- `tests/integration/m9_smoke/telemetry_expected.txt` (new)
- `tests/integration/m9_smoke/README.md` (new)
- `.github/workflows/ci.yml` (add m9_smoke step after m8_smoke)
- `CHANGELOG.md` — new `[v1.0.0-beta1]` section prepended
- `TDMD_Engineering_Spec.md` Приложение C v2.5 — T9.13 addendum prepended
- `docs/development/m9_execution_pack.md` §5 — all T9.*[x] + M9 CLOSED marker

## Acceptance
- All 7 smoke steps pass locally.
- M1..M8 smokes still green (no regression).
- CI green on main.
- Tag annotation reviewed by user before push (mirrors v1.0.0-alpha1 gate).

## Status
pending; terminal task для M9
```

---

## 5. M9 Acceptance Gate

**Master spec §14 M9 — verbatim artifact gate:**

> NVT Al FCC 512 atoms, 10⁵ steps, equipartition within ±2σ, temperature
> distribution Maxwell-Boltzmann within chi² p=0.05. Same для NPT isotropic.

**Checklist (all must be `[x]` before M9 closure):**

- [ ] **T9.0** — M9 execution pack authored (this document). Shipped commit
  `<hash>`.
- [ ] **T9.1** — SPEC delta landed: integrator/SPEC §4/§5 implementation-ready;
  verify/SPEC §4.8/§4.9 authored; threshold registry reserved entries.
- [ ] **T9.2** — NoseHooverNvtIntegrator CPU shipped; unit tests green;
  Trotter symmetry + equipartition canonical fixture passes.
- [ ] **T9.3** — NVT CPU differential vs LAMMPS: 100-step ≤ 1e-10 rel +
  10⁴-step statistical gate passes.
- [ ] **T9.4** — NoseHooverNvtIntegrator GPU shipped; functional test green;
  NVTX audit green; all 3 CUDA flavors green.
- [ ] **T9.5** — NVT GPU bit-exact gate vs CPU ≤ 1e-12 rel on 100-step
  512-atom fixture; D-M9-15 chain rooted.
- [ ] **T9.6** — NoseHooverNptIntegrator CPU shipped; isotropic only;
  anisotropic config clean reject; volume fluctuation non-degenerate.
- [ ] **T9.7** — NPT CPU differential vs LAMMPS: ⟨V⟩ 2% rel, σ_V 5% rel,
  100-step deterministic-core 1e-10 rel.
- [ ] **T9.8** — NoseHooverNptIntegrator GPU shipped; bit-exact vs CPU
  (≤ 1e-12 rel); neighbor rebuild on box change verified.
- [ ] **T9.9** — PolicyValidator K=1 enforcement shipped: (nvt,K>1) + (npt,K>1)
  clean reject с correct error messages per integrator/SPEC §7.3.1;
  `docs/user/ensembles.md` published.
- [ ] **T9.10** — T8 NVT Al FCC canonical benchmark landed; threshold registry
  promoted; 10⁵-step run green on 3 seeds; equipartition ±2σ, M-B χ² p>0.05.
- [ ] **T9.11** — T9 NPT Al FCC canonical benchmark landed; threshold registry
  promoted; 10⁵-step run green; ⟨V⟩ + σ_V match LAMMPS within gates.
- [ ] **T9.12** — Statistical harness shipped + integrated in T8 + T9 runners;
  pytest 3-case synthetic suite green.
- [ ] **T9.13** — M9 integration smoke landed; CHANGELOG.md v1.0.0-beta1 notes;
  `v1.0.0-beta1` annotated tag created (push pending review gate).
- [ ] No regressions: M1..M8 smokes + T1/T4/T6 differentials + T3-gpu anchor +
  m8_smoke_t6 + m8_smoke_t6_100step all green.
- [ ] D-M7-10 byte-exact chain preserved: M3 ≡ M4 ≡ M5 ≡ M6 ≡ M7 Pattern 2 K=1
  for EAM path (unchanged).
- [ ] D-M8-13 byte-exact chain preserved: SNAP Pattern 2 K=1 ≡ 1-rank legacy
  (unchanged).
- [ ] D-M9-15 chain established: NVT GPU FP64 ≡ NVT CPU FP64 single-rank
  Reference+--fmad=false (≤ 1e-12 rel).
- [ ] D-M9-5 policy enforced: (nvt ∨ npt) ∧ K>1 rejected at preflight.
- [ ] CI Pipelines A (lint+build+smokes), B (unit/property), C (differentials),
  D (build-gpu compile-only), E (build-gpu-snap compile-only) — all green.
- [ ] Pre-implementation + session reports attached в каждом PR.
- [ ] Human review approval для каждого PR.

**M9 milestone closure criteria** (master spec §14 M9):

- NoseHooverNVT + NoseHooverNPT (isotropic) CPU + GPU shipped and bit-exact
  chain established on single-rank.
- T8 NVT + T9 NPT canonical benchmarks landed; differential gates green.
- PolicyValidator K=1 enforcement in place для thermostatted ensembles.
- Equipartition + Maxwell-Boltzmann statistical artifact gate passed.
- v1.0.0-beta1 tag pushed.

**M9 status:** PLANNED (2026-04-22). Window opens when M8 final-review gates
on `v1.0.0-alpha1` close. Next window after M9 closure is M10 (MEAM integration
— port LAMMPS `pair_style meam/c`; T5 silicon benchmark; first candidate для
demonstrable TD-native value на angular-moment halo pressure если M8 Case B
persists through cloud burst).

---

## 6. Risks & Open Questions

**Risks:**

- **R-M9-1 — Trotter order drift vs LAMMPS.** LAMMPS `fix_nh.cpp` uses a
  specific Trotter factorization that differs subtly from the 7-step form in
  `integrator/SPEC.md` §4.2 (Yoshida-Suzuki inside chain vs around chain).
  Byte-exact match is impossible; statistical match is required. Mitigation:
  T9.3/T9.7 gates are statistical; 100-step deterministic-core diff exercises
  velocity-Verlet backbone (identical between LAMMPS + TDMD by construction —
  NVE baseline established M1).
- **R-M9-2 — Thermostat MPI Allreduce latency at K=1 multi-rank.** NVT with
  pipeline_depth_cap=1 still exercises multi-rank Pattern 2; thermostat chain
  update requires MPI Allreduce на total KE каждые `thermostat_update_interval`
  steps. On 8-rank commodity network ~10-100 μs × (10⁵ / 50) = 20-200 ms
  overhead per 10⁵-step run. Mitigation: default interval=50 (D-M9-2); scientist
  doc spells out trade-off.
- **R-M9-3 — NPT box flex + neighbor rebuild cost.** Every barostat update
  (interval=100 default) triggers `CellGrid::rebuild`; rebuild cost ~10 ms на
  512 atoms GPU, ~100 ms на 10⁴ atoms — small proportion of 10⁵-step runs but
  can dominate short benchmarks. Mitigation: benchmark wall-time budgeted with
  rebuild included (D-M9-16); default interval=100 balances accuracy + cost.
- **R-M9-4 — 8-week timeline slippage.** M9 is CPU+GPU × NVT+NPT = 4-way
  implementation + benchmarks + policy work. Contingency: T9.7/T9.8 NPT GPU
  can slip to M10 if NPT CPU reveals unexpected box-flex coordination issues
  (NPT integrator/SPEC §5.3 warns of zoning-plan invalidation); M9 still
  closes если NPT GPU deferred (ship CPU-only NPT, mark GPU path "shipped M10").
  Mitigation: explicit 10-week extension budget (D-M9-16).
- **R-M9-5 — Variant A multi-rank statistical validation complexity.** Under
  K=1 Variant A, multi-rank NVT loses TD speedup but must still be validated
  statistically. Chain reductions use Kahan-ring (D-M5-9) для bitwise
  determinism on commit order — but NVT state is global (ξ_k values), not
  per-zone. Risk: non-deterministic ξ drift across 2-rank runs. Mitigation:
  T9.10 runs 3 independent seeds на 1-rank + 2-rank; accept ~1-3 K drift in
  ⟨T⟩ across multi-rank configs as fundamental к canonical reduction semantic;
  byte-exact multi-rank NVT не gated (D-M9-15).
- **R-M9-6 — Morse GPU kernel NOT shipped в M9.** Original M8 memory suggested
  M9 includes "Morse GPU kernel". Master spec §14 M9 does NOT mandate it;
  scope is NVT baseline only. Morse GPU remains pending (blocker для T3-gpu
  full dissertation replication per memory `project_m1_complete.md` T6.10b
  notes). Mitigation: Morse GPU explicitly deferred to M9 stretch goal or M10;
  T8/T9 benchmarks use EAM/alloy Al (single-species) where GPU path already
  exists (D-M9-8).
- **R-M9-7 — 512-atom system size тонок для statistical gates.** 512 atoms =
  1536 degrees of freedom; Maxwell-Boltzmann χ² binning requires ≥30 counts
  per bin for chi² validity, so histogram limited to ~50 bins — reduced resolution
  vs LAMMPS NVT tutorials using 4000+-atom systems. Mitigation: T9.12 harness
  documents expected chi² p-value range on 512-atom system; gate sets p>0.05
  not p>0.3. If resolution insufficient, promote T8 fixture to 2048 atoms (8×
  cost — still < 1 hour per run).
- **R-M9-8 — "beta" tag collision with master spec §14.** Master spec §14
  post-v1 roadmap labels M11 as "v1 beta" (feature-complete sans long-range).
  M9 ships `v1.0.0-beta1` tag which could confuse. Mitigation: explicit
  reconciliation в D-M9-17: beta1 denotes NVT baseline (M9), beta2 denotes
  v1 feature-complete (M11), beta3 reserved (TBD if needed). Communicated in
  CHANGELOG.md intro block.

**Open questions (resolved before or at task time):**

- **OQ-M9-1 — Chain-length upper bound enforcement.** integrator/SPEC §4.1
  says "M = 3 (default)" but no hard max. **To decide at T9.1:** M ∈ [1, 10]
  validated range; M > 10 preflight reject as "no known stability benefit".
- **OQ-M9-2 — Yoshida-Suzuki order.** §4.2 says "3rd-order". Alternative
  5th-order or 7th-order (Tuckerman-Berne-Martyna) is more accurate but costs
  more. **To decide at T9.2:** start 3rd-order (matches LAMMPS default); revisit
  if equipartition gate fails.
- **OQ-M9-3 — NPT volume relaxation on multi-rank under Pattern 2.** K=1
  Pattern 2 с NPT requires all ranks see same volume at all times; barostat η
  is global. Standard Allreduce на pressure — but zoning plan invalidation on
  box change must propagate to ALL ranks atomically. **To decide at T9.6:**
  barostat update happens в scheduler top-level loop (not inside zone commit),
  all ranks re-plan zoning synchronously after Allreduce; single-rank path is
  trivially correct.
- **OQ-M9-4 — Thermostat seed for deterministic 100-step diff gate.** D-M9-9
  assumes "identical velocity seed" — but Nosé-Hoover chain also carries ξ_k
  state. **To decide at T9.3:** force ξ_k=0, p_ξk=0 as initial condition for
  diff gate; production runs may use thermalized IC.
- **OQ-M9-5 — EAM Al single-species vs Morse for T8/T9.** D-M9-8 locks
  EAM/alloy Al. **Confirmed** в pack authoring; revisit only if EAM Al
  fixture reveals issues (unlikely — T3 used same Al EAM in M5).
- **OQ-M9-6 — Statistical test multiple-comparison correction.** T9.12 runs
  equipartition + M-B χ² + (NPT) volume normality + fluctuation. p=0.05
  threshold on 4 parallel tests = ~19% Type-I error per run. **To decide at
  T9.12:** Bonferroni correction to p=0.0125 per test, OR single-metric gate
  (М-B χ² is the primary canonical-ensemble test — equipartition is corollary).
  Preferred: primary = M-B, secondary = equipartition (report but don't fail).
- **OQ-M9-7 — NVT/NPT benchmark run time budget on CI.** T9.10/T9.11 10⁵-step
  runs are 30-45 min. Too slow for CI Pipeline A. **To decide at T9.10:** CI
  runs 10³-step smoke version with relaxed statistical gates; full 10⁵-step
  run is local-only pre-push + released as `verify/slow_tier/m9_nvt_npt_report.md`.
- **OQ-M9-8 — M9 smoke golden file format.** m9_smoke analog of m8_smoke
  needs a "thermo_golden.txt" for byte-exact comparison. NVT thermo includes
  temperature column + pressure (NPT). **To decide at T9.13:** 10-step golden
  using `thermostat_update_interval=1` (bypass interval skip so every step's
  thermo is predictable); same awk stripper pattern as m8_smoke.
- **OQ-M9-9 — Integrator/SPEC §4.5.3 constraint enforcement point.** Should
  `damping_time / (dt * interval) > 2.0` check run at preflight (hard reject)
  or at runtime with warning? **To decide at T9.1:** preflight hard reject
  (consistent with §4.5.3 example error message); users correct config once.
- **OQ-M9-10 — Regression coverage of M8 SNAP path under NVT/NPT.** SNAP +
  NVT combination не part of M9 scope but may be user-expected. **To decide
  at T9.1 SPEC delta:** explicitly out-of-scope M9; integrator/SPEC §7.3.1
  examples use morse/eam; SNAP + NVT works mechanically (SNAP is Potential,
  NVT integrator doesn't care about potential family) but not gated via
  canonical benchmark in M9.
- **OQ-M9-11 — v1.0.0-beta1 tag push timing.** Do we push v1.0.0-beta1
  immediately upon T9.13 closure, or batch with M10 if M10 follows closely?
  **To decide at T9.13:** push immediately (mirror v1.0.0-alpha1 precedent
  2026-04-22); users may want to pin NVT baseline independent of M10 MEAM.

**Parallel-track notes (orthogonal carry-forwards during M9 window):**

- **#165 EAM per-bond refactor (M9 scout).** Structural refactor of EAM/alloy
  GPU force kernel mirroring SNAP T8.6c-v5 per-bond dispatch pattern. Memory
  `project_m1_complete.md` notes EAM T-opt-1 was REJECTED per profile;
  per-bond refactor may change that verdict. Lands on branch `m9/eam-per-bond`
  opportunistically; not part of M9 acceptance gate.
- **#167 SNAP GPU robust-failure-mode guard.** Defense-in-depth for high-T
  neighbour overflow per `verify/slow_tier/m8_mixed_fast_snap_only_REPORT.md`
  §9. Orthogonal to M9; standalone PR.
- **T7.8b 30% runtime overlap measurement.** Still cloud-burst-gated (≥2 GPU
  hardware prerequisite). Deferred to M11 along with multi-node scaling
  campaign. Not M9 scope.

---

## 7. Roadmap Alignment

| Deliverable | Consumer milestone | Why it matters |
|---|---|---|
| T9.1 integrator/SPEC §4/§5 finalize | M10 MEAM integration; M11 NVT-in-TD research; M13 long-range + NVT coupling | Thermostatted ensemble contract locked; Variant A policy stable for M10+ |
| T9.2 NoseHooverNvtIntegrator CPU | M10 MEAM NVT differential; M11 Variant C research baseline | Canonical NVT reference for thermostatted potential families |
| T9.3 NVT CPU diff vs LAMMPS | M10 MEAM diff template; M11 Variant C comparison | Oracle infrastructure (landed M1) exercised on thermostatted path |
| T9.4 NoseHooverNvtIntegrator GPU | M10 MEAM GPU NVT; M13 long-range + NVT hybrid integration | GPU NVT baseline production-ready |
| T9.5 NVT GPU bit-exact gate | Continuous regression guard M9-M13; M10 MEAM bit-exact | D-M6-7 chain extended to thermostatted ensembles |
| T9.6 NoseHooverNptIntegrator CPU (iso) | M10 MEAM NPT; M13 full-physics runs | First thermodynamic ensemble beyond NVT для TDMD |
| T9.7 NPT CPU diff vs LAMMPS | M10 MEAM NPT diff; validation of box-flex mechanics | Oracle coverage extended to NPT |
| T9.8 NoseHooverNpt GPU | M10 MEAM GPU NPT; M13 long-range + NPT | Full thermostatted/barostatted GPU stack |
| T9.9 PolicyValidator K=1 enforce + docs | All M9+ production deployments — prevents silent incorrectness | Scientist UX critical; "fast but wrong" auto-reject |
| T9.10 T8 NVT benchmark | M10 MEAM vs LAMMPS; M11 Variant C performance comparison | Canonical NVT benchmark установлен |
| T9.11 T9 NPT benchmark | M10 MEAM NPT; M13 long-range NPT | Canonical NPT benchmark установлен |
| T9.12 Statistical harness | All M9+ ensemble validation; M11 Variant C statistical comparison | Reusable validation primitives |
| T9.13 M9 smoke + v1.0.0-beta1 tag | M10-M13 stability floor | Pre-push gate extended to include thermostatted path |

**Downstream milestone impact:**

- **M10 (MEAM integration):** T9.2/T9.4 NVT + T9.6/T9.8 NPT infrastructure
  reused directly — MEAM NVT/NPT tests add fixtures only, no new integrator
  work. T9.3/T9.7 diff patterns cloned для MEAM. M10 primary hope: MEAM на
  angular-moment halo pressure може exhibit TD architecture's value more
  clearly than SNAP did (Case B outcome в M8 shifts proof-of-value burden to
  M10).
- **M11 (NVT-in-TD research window):** T9.2 + T9.4 + T9.10 form the baseline
  against which Variant C prototype is compared. M9 establishes "Variant A
  canonical performance"; M11 asks "does Variant C beat Variant A by ≥10%
  while preserving equipartition?"
- **M12 (PACE + MLIAP):** NVT/NPT для ML potentials — same integrator surface
  applies; PACE paper validation suites include NVT runs which M9 will reproduce.
- **M13 (long-range + NVT-in-TD if M11 go):** Ewald/PPPM splits force
  computation into short-range + long-range; thermostat couples to both.
  T9.4 GPU NVT establishes the GPU-resident thermostat pattern that M13
  long-range plugs into. If M11 go-decision accepted, Variant C lazy
  thermostat replaces Variant A on main; T9.9 PolicyValidator relaxes K=1
  mandate под Variant C flavor guard.

---

## 8. Links

- Master spec: `TDMD_Engineering_Spec.md` §14 M9, §D.11 (BuildFlavor matrix — unchanged in M9), §D.17 (no new flavor in M9)
- Module SPECs: `docs/specs/integrator/SPEC.md` §4 / §5 / §7.3 / §8 / §9
- Verify SPEC: `docs/specs/verify/SPEC.md` (T8 + T9 registered in T9.1)
- Threshold registry: `verify/thresholds/thresholds.yaml`
- Predecessor packs: `docs/development/m7_execution_pack.md`,
  `docs/development/m8_execution_pack.md`
- Playbook: `docs/development/claude_code_playbook.md` §1.3 (pre-impl reports),
  §3 (session rhythm), §9.1 (spec-delta procedure)
- CHANGELOG.md (v1.0.0-alpha1 published; v1.0.0-beta1 target at T9.13)
- Memory context: `project_m1_complete.md` (M1..M8 closure state),
  `project_option_a_ci.md` (Option A CI policy), `env_cuda_13_path.md`

---
