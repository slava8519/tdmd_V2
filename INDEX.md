# TDMD Documentation Index

Quick-access index to all documents и ключевых секций.

---

## Core documents

| Document | Lines | Purpose |
|---|---|---|
| [README.md](README.md) | - | Entry point, navigation |
| [TDMD_Engineering_Spec.md](TDMD_Engineering_Spec.md) | ~3000 | Master spec v2.5 (Приложения D + E, v2.5 production lessons) |
| [docs/development/claude_code_playbook.md](docs/development/claude_code_playbook.md) | 806 | AI agent playbook |

## Module SPECs (12 documents, ~7300 lines total)

### Critical path (прочитать в первую очередь)

| Module | Lines | Covers |
|---|---|---|
| [scheduler](docs/specs/scheduler/SPEC.md) | 626 | TD DAG, certificates, commit protocol, invariants |
| [zoning](docs/specs/zoning/SPEC.md) | 547 | N_min math, Hilbert 3D, canonical ordering |
| [perfmodel](docs/specs/perfmodel/SPEC.md) | 550 | Performance prediction, validation gates |

### Compute path

| Module | Lines | Covers |
|---|---|---|
| [potentials](docs/specs/potentials/SPEC.md) | 666 | Morse, EAM, MEAM, SNAP; precision policy |
| [integrator](docs/specs/integrator/SPEC.md) | 445 | Velocity-Verlet, NVT, NPT, timestep policy |
| [state](docs/specs/state/SPEC.md) | 552 | AtomSoA, Box, species, version, identity |
| [neighbor](docs/specs/neighbor/SPEC.md) | 633 | Cell grid, lists, skin, migration, reorder |

### Infrastructure

| Module | Lines | Covers |
|---|---|---|
| [comm](docs/specs/comm/SPEC.md) | 664 | MPI/NCCL/NVSHMEM, protocols, topologies |
| [runtime](docs/specs/runtime/SPEC.md) | 714 | SimulationEngine, lifecycle, policy bundles |

### User-facing

| Module | Lines | Covers |
|---|---|---|
| [io](docs/specs/io/SPEC.md) | 675 | YAML, LAMMPS data, dumps, restart, repro |
| [telemetry](docs/specs/telemetry/SPEC.md) | 563 | Metrics, NVTX, breakdown, logging |
| [cli](docs/specs/cli/SPEC.md) | 678 | run/validate/explain/compare/resume |

### Scientific validation

| Module | Lines | Covers |
|---|---|---|
| [verify](docs/specs/verify/SPEC.md) | ~900 | ⭐ VerifyLab: cross-module validation, thresholds, LAMMPS submodule, anchor-test, tiers |

---

## Key sections in master spec

### Physics / Method (Часть I)

- [§1 Что такое TDMD](TDMD_Engineering_Spec.md) — canonical one-sentence definition
- [§2 Суть метода](TDMD_Engineering_Spec.md) — якорь из диссертации Андреева
- [§3 Целевая ниша](TDMD_Engineering_Spec.md) — почему EAM/MEAM/SNAP
- [§4 Performance model](TDMD_Engineering_Spec.md) — когда TD выигрывает
- [§4a Two-level decomposition](TDMD_Engineering_Spec.md) — TD×SD hybrid

### Architecture (Часть II)

- §5 Требования (functional + non-functional + unit systems)
- §6 Теория алгоритма TD: zones, state machine, certificates
- §7 Режимы исполнения: BuildFlavor × ExecProfile
- §8 Архитектура: модули, ownership, data flow
- §9 Zoning planner
- §10 Parallel model и коммуникации
- §11 Long-range (v2+)

### Interfaces (Часть III)

- §12 Все C++/CUDA interfaces

### Testing (Часть IV)

- §13 Test pyramid: 6 layers, canonical benchmarks T0-T7, anchor-test

### Roadmap (Часть V)

- §14 Milestones M0-M8
- §15 Engineering methodology

### Appendices (Часть VI)

- Приложение A — traceability с диссертацией
- Приложение B — assumptions и open questions
- Приложение C — change log

---

## Key topics — прямой поиск

### Хочу понять TD метод
→ master spec §2 + Приложение A.2 (formulas map)
→ диссертация: `docs/_sources/dissertation_andreev_2007.docx`

### Хочу понять когда TD выигрывает
→ master spec §4 (базовая модель) + §4a (two-level)
→ детали: `docs/specs/perfmodel/SPEC.md` §3

### Хочу начать реализацию
→ master spec §14 (roadmap)
→ `docs/development/claude_code_playbook.md` §3 (task template)
→ playbook §10 (onboarding нового агента)

### Хочу понять scheduler (сердце TD)
→ `docs/specs/scheduler/SPEC.md`
→ master spec §6 (theory) + §12.4 (interfaces)

### Хочу понять deployment patterns
→ master spec §10.1 (три pattern'а)
→ `docs/specs/comm/SPEC.md` §6 (backends)
→ `docs/specs/runtime/SPEC.md` §7 (pattern awareness)

### Хочу понять unit systems
→ master spec §5.3 (policy)
→ `docs/specs/io/SPEC.md` §3 (yaml format)
→ `docs/specs/state/SPEC.md` §5 (internal representation)

### Хочу понять как работает с LAMMPS
→ master spec §3 (positioning, LAMMPS как oracle)
→ `docs/specs/io/SPEC.md` §4 (LAMMPS data import)
→ `docs/specs/cli/SPEC.md` §6 (`tdmd compare`)

### Хочу понять reproducibility
→ master spec §7.3 (три уровня гарантий)
→ `docs/specs/runtime/SPEC.md` §5 (restart) + §8 (repro bundle)
→ `docs/specs/io/SPEC.md` §6-7

### Хочу понять правила разработки
→ `docs/development/claude_code_playbook.md` §1 (7 обязательных правил)
→ playbook §5 (auto-reject conditions)
→ master spec §15 (Spec-Driven TDD)

### Хочу понять как TDMD валидируется
→ `docs/specs/verify/SPEC.md` (VerifyLab — cross-module validation layer)
→ master spec §13 (test pyramid)
→ verify SPEC §3 (threshold registry) + §4 (canonical benchmarks)

### Хочу понять precision policy (float / double / mixed)
→ master spec Приложение D (full 18-subsection policy)
→ master spec §7 (quick overview)
→ 5 BuildFlavors: Fp64Ref, Fp64Prod, MixedFast (B), MixedFastAggressive (A), Fp32Exp
→ `docs/specs/potentials/SPEC.md` §8 (module-specific applications)

### Хочу понять bitwise reproducibility и toolchain binding
→ master spec §D.10 (FMA policy + cross-compiler binding)
→ master spec §7.3 (три уровня воспроизводимости)
→ `docs/specs/verify/SPEC.md` §3 (toolchain_binding thresholds)

### Хочу понять NVT/NPT в TD
→ `docs/specs/integrator/SPEC.md` §7.3 (thermostat global state analysis)
→ master spec §14 M9 (NVT baseline delivery) + M11 (research window)
→ Rationale: почему K=1 required для NVT в v1.5, что research в v2+

### Хочу понять pipeline depth K — как выбирать
→ master spec §6.5a (Auto-K policy — три режима: manual / measurement-based / perfmodel-assisted)
→ master spec §4a.6 (K как единственный параметр баланса в Pattern 2)
→ `docs/specs/perfmodel/SPEC.md` §3.3 (K_opt formula)

### Хочу понять cutoff и smoothing
→ `docs/specs/potentials/SPEC.md` §2.4 (four strategies: hard/shifted-energy/shifted-force/smoothed)
→ `docs/specs/verify/SPEC.md` §3 (cutoff_treatment thresholds)
→ Matrix: Morse/LJ/EAM → C, MEAM/SNAP/PACE → D

### Хочу понять Python integration
→ master spec Приложение E (full strategy, 11 sections)
→ Roadmap: Layer 1 pybind11 (M9), Layer 2 ASE (M11), Layer 3 workflows (v2.0)

### Хочу понять load balancing (dynamic distribution of work)
→ `docs/specs/scheduler/SPEC.md` §11a (three-phase DLB maturity)
→ Phase 1 M5-M6 measurement-only, Phase 2 M7-M8 advisory, Phase 3 v2+ active
→ Different philosophy от LAMMPS/GROMACS: scientific defaults vs performance defaults

### Хочу понять GPU-resident mode
→ `docs/specs/integrator/SPEC.md` §3.5 (mandatory policy М6+)
→ Data lives on GPU; CPU orchestrates но не касается hot data per-step
→ Learned from NAMD 3.0 (traditional offload → GPU idle 30-50%)

### Хочу понять когда TDMD НЕ подходит (small systems)
→ `docs/specs/perfmodel/SPEC.md` §3.7 (workload saturation)
→ Minimum atoms per rank: LJ 10k / EAM 5k / SNAP 1k (на A100)
→ `tdmd explain --perf` даёт recommendations когда GPU under-saturated

### Хочу понять почему coupling интервал важен
→ master spec §6.5b (expensive compute intervals)
→ `docs/specs/integrator/SPEC.md` §4.5 (thermostat update interval)
→ Default: potential_energy=100, kinetic=50, virial=100 (не каждый шаг)

### Хочу понять anchor-test (воспроизведение диссертации)
→ `docs/specs/verify/SPEC.md` §4.4 (T3 benchmark)
→ master spec §13.3
→ verify SPEC §7.4 (AnchorTestRunner)

---

## Источники (read-only, контекст)

- `docs/_sources/dissertation_andreev_2007.docx` — оригинальная диссертация В.В. Андреева (2007)
- `docs/_sources/deep_research_report.md` — research report с LAMMPS integration analysis
- `docs/_sources/spec_v0_1_draft.md` — первая черновая версия ТЗ
- `docs/_sources/spec_v1_0_reframed.md` — промежуточная версия (classical-many-body first reframe)

---

*TDMD Documentation Index, 2026-04-16.*
