# M2 Execution Pack

**Document:** `docs/development/m2_execution_pack.md`
**Status:** draft, awaiting human review
**Parent:** `TDMD_Engineering_Spec.md` §14 (M2), `docs/development/claude_code_playbook.md` §3
**Milestone:** M2 — EAM CPU + perf model skeleton + lj support (6 нед.)
**Created:** 2026-04-18
**Author:** Architect / Spec Steward role (Claude Opus 4.7)

---

## 0. Purpose

Этот документ декомпозирует milestone **M2** master spec'а на **13 PR-size задач**, каждая сформулирована по каноническому шаблону `claude_code_playbook.md` §3.1.

Документ — **process artifact**, не SPEC delta. Если в процессе реализации M2 обнаружится gap в spec'е, оформляется отдельная SPEC delta по playbook §9.1, не в этом pack'е.

M2 строит три независимые линии поверх M1 baseline:

1. **EAM потенциалы на CPU** — первый многочастичный (many-body local) потенциал. Это главный скальпель milestones: differential test T4 (Ni-Al alloy) vs LAMMPS обязан проходить до `1e-10 rel` на forces.
2. **`UnitConverter` полная поддержка `lj`** + T0/T1 cross-checks в двух unit systems. Доказательство того, что абстракция units корректна: T1 в metal и в lj дают идентичные результаты после преобразования.
3. **PerfModel + telemetry skeleton** — первая версия analytic performance model (Pattern 1 / Pattern 3) и `tdmd explain --perf`. Прогнозы ещё без TD (это M4), но каркас готов для auto-K в M5 и валидации в M7.

После M2 TDMD: (a) численно корректен для Al-Ni EAM alloy, (b) поддерживает оба канонических unit system'а, (c) прогнозирует собственную производительность на тривиальных случаях. Zoning / scheduler — M3+. GPU — M6.

После успешного закрытия всех 13 задач и acceptance gate (§5) — milestone M2 завершён; execution pack для M3 создаётся как новый аналогичный документ.

---

## 1. Decisions log (зафиксировано до старта T2.1)

| # | Решение | Значение | Rationale / источник |
|---|---|---|---|
| **D-M2-1** | EAM spline interpolation — source of truth | Bit-for-bit копия LAMMPS `pair_eam.cpp` interpolation code с credit + GPL-2.0 header | Потенциалы/SPEC §4.4 requires "идентичную" формулу для differential bit-match. Любая отступающая реализация ломает T4 gate. |
| **D-M2-2** | EAM module layout | `src/potentials/eam_alloy.{hpp,cpp}` + `eam_fs.{hpp,cpp}`; tabulated-function primitives — в `src/potentials/tabulated.{hpp,cpp}` (общий с MEAM/SNAP в будущем) | Потенциалы/SPEC §4.3 — математика для alloy/fs одинаковая; только parameter file format различается. Держим splines reusable. |
| **D-M2-3** | EAM parameter file parsers — ownership | `src/potentials/eam_file_parser.{hpp,cpp}` (не в `io/`) | io/SPEC §1.2 — io владеет only low-level file I/O; per-potential формат принадлежит потенциалу. |
| **D-M2-4** | T4 reference system | Ni-Al alloy, FCC 6×6×6 = 864 атомов, половина Ni / половина Al random distribution seed 12345, Mishin 2004 EAM parameters (published, LGPL-compatible) | Master spec §13.2; Mishin набор — standard in the community, LAMMPS ships it in examples. |
| **D-M2-5** | `LjReference` значения в lj unit tests | σ=1.0 Å, ε=1.0 eV, m=1.0 g/mol (identity canonical) + второй набор (σ=3.405 Å, ε=0.0104 eV, m=39.948) для round-trip robustness | io/SPEC канонический LJ — dimensionless; identity reference делает round-trip тесты читаемыми. Ar-like набор — физический sanity. |
| **D-M2-6** | PerfModel M2 scope | Только Pattern 1 (чистый TD) + Pattern 3 (чистый SD); Pattern 2 — M7. Calibration table — hardcoded (single HW class: "modern_x86_64"); auto-calibration — M4. | perfmodel/SPEC §3.2, §3.3; master spec §14 M2 — "первая версия, analytic TD vs SD". |
| **D-M2-7** | `tdmd explain --perf` output shape | Default: 3-line recommendation + alternatives block (per perfmodel/SPEC §6.1). `--format json` — M3+. | Keep M2 вывод human-first; JSON — когда появится telemetry consumer в M4. |
| **D-M2-8** | Telemetry sink в M2 | JSONL file sink (`runs/<timestamp>.jsonl`), один пишуший поток, synchronous I/O; LAMMPS-compatible текстовый breakdown в `tdmd run --timing` (stderr). | telemetry/SPEC §4.2 — LAMMPS-compatible format обязателен; JSONL — простой и CI-grep-friendly. Async — M5+. |
| **D-M2-9** | DifferentialRunner MVP (T2.8) | Python + stdlib + PyYAML (как T1.11); расширение verify/compare.py до сравнения dump-файлов (forces per-atom), плюс thermo (уже есть) | verify/SPEC §7.1 — forces comparison для EAM обязателен; ThDMD и LAMMPS dump в `atom_style atomic id type x y z fx fy fz` — совместимый формат. |
| **D-M2-10** | T1 lj-variant — **один** benchmark директория, два config'а | `verify/benchmarks/t1_al_morse_500/config_metal.yaml` + `config_lj.yaml`; harness запускает оба и cross-checks identical post-conversion | verify/SPEC §4.6: two variants of same benchmark для unit-system cross-check. Избегаем дублирования reference data. |
| **D-M2-11** | M2 merge freeze boundary | T2.8 (DifferentialRunner MVP) — architectural gate; T2.8 merged до того как T2.9 / T2.10 начинаются | T2.9 (EAM classes) и T2.10 (T4 benchmark) оба depend на forces-сравнение из MVP. Параллельная работа возможна только после T2.8. |
| **D-M2-12** | BuildFlavor в M2 | Только `Fp64ReferenceBuild` (продолжение M1). `Fp64ProductionBuild` — M3 где появится zoning. | Master spec §D: reference path — canonical oracle; remaining flavors wait until optimization work starts. |

---

## 2. Глобальные параметры окружения

| Параметр | Значение | Примечание |
|---|---|---|
| OS | Linux (Ubuntu 24.04 LTS) | Dev-машина пользователя; ubuntu-latest в CI |
| C++ compiler | GCC 13+ / Clang 17+ | C++20; CI уже проверяет оба |
| CMake | 3.25+ | Master spec §15.2 |
| CUDA | 13.1 installed, **не используется в M2** | GPU path — M6 |
| Python | 3.10+ | pre-commit, verify harness, perfmodel validation |
| Test framework | Catch2 v3 (FetchContent) | Унаследовано из M0 |
| LAMMPS oracle | `verify/third_party/lammps/install_tdmd/bin/lmp` | M0 T0.7 artifact; EAM pair_style уже включён (default build) |
| Active BuildFlavor | `Fp64ReferenceBuild` | D-M2-12 |
| Run mode | single-rank, single-thread, CPU only | Master spec §14 M2 |
| EAM reference parameters | `Ni.eam.alloy` + `AlNi.eam.alloy` (Mishin 2004) — скачиваются в `verify/reference/eam/` при harness setup | D-M2-4 |
| Branch policy | `m2/T2.X-<topic>` per PR → `main` | CI required: 5 hosted checks (lint + docs-lint + build-cpu×2 + differential-t1) |

---

## 3. Suggested PR order

Dependency graph:

```
                   ┌─► T2.1 (UnitConverter lj) ──► T2.2 (data+yaml lj wiring) ──► T2.4 (T1 lj cross) ─┐
                   │                                                                                  │
                   │                                                                                  │
M1 baseline ───────┼─► T2.3 (T0 morse-analytic both units) ───────────────────────────────────────────┤
                   │                                                                                  │
                   │                                                                                  │
                   ├─► T2.5 (TabulatedFunction) ──► T2.6 (EAM file parsers) ──► T2.7 (EAM classes) ─┐ │
                   │                                                                                │ │
                   │                                                                                ▼ ▼
                   │                                                    T2.8 (DifferentialRunner MVP) ──► T2.9 (T4 benchmark, GATE)
                   │                                                                                         │
                   ├─► T2.10 (PerfModel::predict) ─┬─► T2.11 (tdmd explain --perf) ──────────────────────────┤
                   │                                                                                         │
                   └─► T2.12 (telemetry JSONL + LAMMPS-format breakdown) ────────────────────────────────────┤
                                                                                                             │
                                                                                                             ▼
                                                                                                       T2.13 (M2 smoke, CI Pipeline B)
```

**Линейная последовательность (single agent):**
T2.1 → T2.2 → T2.3 → T2.4 → T2.5 → T2.6 → T2.7 → T2.8 → T2.9 → T2.10 → T2.11 → T2.12 → T2.13.

**Параллельный режим (multi-agent):** после M1 закрыт — `{T2.1, T2.3, T2.5, T2.10, T2.12}` все параллельно (они независимы). После T2.1 — T2.2; после T2.2 — T2.4. После T2.5 — T2.6 → T2.7 последовательно (каждый требует предыдущего). После T2.7 — T2.8. После T2.7 + T2.8 — T2.9. После T2.10 — T2.11. T2.13 — последняя.

**Estimated effort:** 6–8 недель (single agent). Самые длинные: T2.7 (EAM implementation, ~1.5 нед.), T2.8 (DifferentialRunner MVP, ~1 нед.), T2.9 (T4 benchmark gate, ~1 нед.).

---

## 4. Tasks

### T2.1 — `UnitConverter` full `lj` implementation

```
# TDMD Task: Complete UnitConverter for LAMMPS 'lj' unit system

## Context
- Master spec: §5.3 (unit system support), §14 M2
- Module SPEC: docs/specs/io/SPEC.md §3; docs/specs/runtime/SPEC.md (UnitConverter lives in runtime)
- Role: Physics Engineer
- Milestone: M2

## Goal
Реализовать все 16 `_from_lj / _to_lj` методов в `UnitConverter` (сейчас throwing `NotImplementedInM1Error`) по каноническим LAMMPS формулам. Идея: `UnitConverter` становится transparent abstraction — downstream modules работают в internal metal representation, а lj ↔ metal conversion существует только на I/O boundary.

## Scope
- [included] Все 8 dimensions × 2 directions = 16 functions:
  length, energy, time, mass, force, pressure, velocity, temperature
- [included] Формулы из LAMMPS docs (https://docs.lammps.org/units.html):
  - length_metal  = value_lj · σ
  - energy_metal  = value_lj · ε
  - time_metal    = value_lj · sqrt(m·σ²/ε)
  - mass_metal    = value_lj · m
  - force_metal   = value_lj · ε/σ
  - pressure_metal= value_lj · ε/σ³
  - velocity_metal= value_lj · sqrt(ε/m)
  - temperature_metal = value_lj · ε/k_B
- [included] Input validation: reject σ ≤ 0 или ε ≤ 0 или m ≤ 0 с `std::invalid_argument`
- [included] Round-trip тесты: `to_lj(from_lj(x)) == x` bitwise для каждой dimension (FP64)
- [included] Property tests (≥10⁴ cases) — random σ, ε, m в physical ranges; round-trip holds

## Out of scope
- [excluded] Кастомные unit системы (`units real`, `units electron`) — post-v1
- [excluded] Изменение internal representation — остаётся metal (см. unit_converter.hpp line 98)
- [excluded] YAML config parsing для `reference:` block — это T2.2

## Mandatory invariants
- Round-trip bitwise на identity reference (σ=1, ε=1, m=1): `to_lj(from_lj(x)) == x` для любого finite `x`
- Round-trip на Ar-like (σ=3.405, ε=0.0104, m=39.948): residual ≤ 2 ulp
- Formulas bitwise matching LAMMPS `src/update.cpp` + `src/domain.cpp` (no divergence in fused sqrt/div)
- No hidden mutable state в `UnitConverter` (все методы `static` + `constexpr`-able где возможно)

## Required files
- `src/runtime/unit_converter.cpp` — добавить реализации 16 методов (stub throw удаляется)
- `tests/runtime/test_unit_converter_lj.cpp` — новый файл
- `tests/runtime/CMakeLists.txt` — добавить новый test

## Required tests
- [unit] 16 round-trips на identity reference: bitwise-exact
- [unit] 16 round-trips на Ar reference: ≤ 2 ulp
- [unit] invalid reference (σ=0, ε=-1, m=NaN) — каждый метод бросает `std::invalid_argument`
- [property ≥10⁴] random x ∈ [-1e6, 1e6] + random reference в physical ranges — round-trip within 4 ulp
- [regression] T1.2 existing metal tests продолжают passing (no accidental breakage)

## Expected artifacts
- Единый PR
- Pre-impl report + session report в PR description

## Acceptance criteria
- [ ] Все 16 методов implemented, no `NotImplementedInM1Error` throws
- [ ] Unit tests green на gcc-13 и clang-17
- [ ] Property tests green (10⁴+ cases)
- [ ] No regressions в existing UnitConverter metal tests
- [ ] Pre-commit hooks green (clang-format, ruff, yamllint, markdownlint)
- [ ] Human review approved

## Hints
- k_B (для temperature conversion) — используй существующий constexpr в `simulation_engine.cpp`; перенеси в header-owned constant (не дублируй)
- `sqrt` в `time_from_lj` — use `std::sqrt`, не `std::sqrtf`; FP64 throughout
- Identity-reference round-trip тест полезен как reader documentation: "when σ=ε=m=1, lj === metal"
```

---

### T2.2 — LAMMPS data importer + YAML config: `units: lj` integration

```
# TDMD Task: Wire `units: lj` through the ingestion path

## Context
- Master spec: §5.3, §14 M2
- Module SPEC: docs/specs/io/SPEC.md (LAMMPS reader + YAML config)
- Role: Core Runtime Engineer
- Depends on: T2.1
- Milestone: M2

## Goal
Существующий `io::read_lammps_data_file` и `io::parse_yaml_config` отказывают на `units: lj` в M1 (preflight reject). M2 открывает lj path: data reader распознаёт заголовок, YAML config принимает `reference:` блок, engine внутренне всё равно работает в metal (UnitConverter конвертит на boundary).

## Scope
- [included] `io::LammpsDataImportOptions`: добавить поле `lj_reference: std::optional<LjReference>` — required when `units=lj`
- [included] Data reader: парсит `# LAMMPS data file, units = lj` header, converts values через `UnitConverter::*_from_lj` до заполнения `AtomSoA` / `Box`
- [included] YAML `simulation.units: lj` accepted; `simulation.reference: { sigma, epsilon, mass }` блок — required если `units=lj`, ignored если `units=metal`
- [included] Preflight validator:
  - units=lj без reference block → error "lj requires reference (sigma, epsilon, mass)"
  - units=lj с invalid reference (σ≤0) → error "lj reference must have sigma > 0..."
  - units=metal с reference block → warning "reference block ignored for units=metal"
- [included] Round-trip integration test: (a) metal config, (b) same system в lj config (σ, ε chosen для identity), (c) оба ingest → identical AtomSoA bitwise

## Out of scope
- [excluded] Кастомная unit system в YAML (`units: real`) — post-v1
- [excluded] `write_data` для lj (output остаётся в metal) — M3+
- [excluded] CLI `tdmd run --units-override lj` — N/A; units идут из config

## Mandatory invariants
- После ingest'а AtomSoA.x[i], box.xhi, species.mass[0] хранятся в metal internal representation (Å, eV, g/mol) независимо от input units
- `UnitConverter` вызывается exactly once per value на ingest (no double conversion)
- Preflight rejects ill-formed config до engine init (M1 invariant preserved)

## Required files
- `src/io/lammps_data_reader.cpp` — добавить lj branch
- `src/io/yaml_config.hpp/.cpp` — расширить schema (`reference:` block, `LjReference`)
- `src/io/preflight.cpp` — новые validation rules
- `tests/io/test_lammps_data_lj.cpp` — новый
- `tests/io/test_yaml_config_lj.cpp` — новый или extension
- `tests/io/test_preflight.cpp` — добавить lj cases

## Required tests
- [unit] lj data file parsed → internal metal values match pre-computed expected
- [unit] YAML с units=lj и reference parses; exposed via `YamlConfig::lj_reference` accessor
- [unit] preflight rejects units=lj без reference; rejects invalid sigma
- [integration] metal ↔ lj round-trip: same system written in two ways, ingest → identical state
- [regression] T1.3/T1.4 existing metal tests pass

## Expected artifacts
- Единый PR
- Pre-impl + session reports

## Acceptance criteria
- [ ] Both units systems accepted by data reader + YAML + preflight
- [ ] metal/lj round-trip integration test bitwise
- [ ] Existing metal tests green
- [ ] Pre-commit green
- [ ] Human review approved

## Hints
- LAMMPS `.data` header: `# LAMMPS data file, ... units = lj` — regex-parse, не full header reinterpretation
- Для YAML schema: `reference` block optional на top level, но required when units=lj — reflect this в preflight, не в schema parser (которому должен быть permissive)
```

---

### T2.3 — T0 `morse-analytic` benchmark (metal + lj cross-check)

```
# TDMD Task: T0 analytic dimer benchmark, both unit systems

## Context
- Master spec: §13.2 (T0), §14 M2
- Module SPEC: docs/specs/verify/SPEC.md §4
- Role: Validation / Reference Engineer
- Depends on: T2.1 (lj), T2.2 (lj ingestion)
- Milestone: M2

## Goal
T0 — двухатомный dimer с Morse потенциалом, ground-truth — closed-form аналитическая формула (не LAMMPS). Это unit-sanity test: гарантирует что force/energy pipeline даёт аналитически правильные ответы до того как мы доверяем differential против LAMMPS. В M2 T0 публикуется в двух unit systems как cross-check для UnitConverter.

## Scope
- [included] `verify/benchmarks/t0_morse_analytic/`:
  - README.md — формулы, rationale
  - config_metal.yaml — 2 атома at r=r0+0.5 Å, Morse params (D=1 eV, α=1 Å⁻¹, r0=3 Å, cutoff=6 Å)
  - config_lj.yaml — identical system, expressed в lj units (σ=1 Å, ε=1 eV, m=1 g/mol → identity reference для readable numbers)
  - setup_metal.data / setup_lj.data — LAMMPS-style data files (committed, generated once)
  - checks.yaml — analytic expectations (F, E по closed-form) + cross-check (metal == lj post-conversion)
- [included] `verify/t0/run_analytic.py` driver: для каждого units варианта:
  1. Запускает TDMD `run` на 0 шагов (just compute forces)
  2. Извлекает forces + PE из dump / thermo
  3. Сравнивает с аналитической формулой до `1e-12 rel`
  4. Cross-check: после UnitConverter-конверсии, metal и lj выдают bitwise-identical internal state
- [included] Catch2 wrapper `verify/t0/test_t0_analytic.cpp` (аналогично T1.11)
- [included] CI wiring в `.github/workflows/ci.yml` — работает без LAMMPS (чистый TDMD + analytic)

## Out of scope
- [excluded] N > 2 atoms — T0 намеренно тривиален
- [excluded] Integration across dt — T0 только force check
- [excluded] Multi-species — T1+

## Mandatory invariants
- Residual на analytic formula ≤ 1e-12 rel для каждого F component + PE
- metal-vs-lj cross-check — bitwise identical AtomSoA + forces post-conversion
- Benchmark runs в public CI (no LAMMPS dependency)

## Required files
- `verify/benchmarks/t0_morse_analytic/{README.md, config_metal.yaml, config_lj.yaml, setup_metal.data, setup_lj.data, checks.yaml}`
- `verify/t0/{CMakeLists.txt, run_analytic.py, test_t0_analytic.cpp}`
- `verify/thresholds/thresholds.yaml` — добавить `benchmarks.t0_morse_analytic.forces_analytic_relative: 1.0e-12`
- `verify/CMakeLists.txt` — `add_subdirectory(t0)`
- `.github/workflows/ci.yml` — добавить T0 step к build-cpu (runs под обоими compilers)

## Required tests
- [smoke/analytic] F и PE соответствуют closed-form на обоих units — green
- [cross-check] metal path и lj path producе identical AtomSoA post-conversion
- [regression] T1 benchmark continues passing (no unintended changes)

## Expected artifacts
- Единый PR
- Pre-impl + session reports

## Acceptance criteria
- [ ] T0 analytic driver runs in < 2s, exit 0
- [ ] Both units variants pass
- [ ] Cross-check green
- [ ] CI integration visible (new step passes on ubuntu-latest)
- [ ] Human review approved

## Hints
- Closed-form Morse: F(r) = 2·D·α·(1 − exp(−α·(r−r0)))·exp(−α·(r−r0))·r̂
- r0+0.5 — положение близко к equilibrium, чтобы exp не overflow
- Для identity-lj setup (σ=ε=m=1): все числа в config_lj.yaml === config_metal.yaml; post-conversion check становится тривиальным readability-test
```

---

### T2.4 — T1 `lj`-variant cross-check

```
# TDMD Task: Publish T1 benchmark in the lj unit system

## Context
- Master spec: §13.2 (T1), §14 M2 (artifact gate: "T1 differential green одновременно в metal и lj")
- Module SPEC: docs/specs/verify/SPEC.md §4.6 T1 landed
- Role: Validation / Reference Engineer
- Depends on: T2.1, T2.2
- Milestone: M2

## Goal
T1 (Al FCC Morse 500) landed в M1 только для metal. M2 artifact gate требует identical differential results когда тот же T1 конфигурирован в lj units. Это архитектурное доказательство того, что `UnitConverter` действительно transparent — downstream force/integrator не видит разницу.

## Scope
- [included] `verify/benchmarks/t1_al_morse_500/config_lj.yaml` — idem что config_metal.yaml но в lj (identity reference для bit-exact post-conversion)
- [included] `verify/benchmarks/t1_al_morse_500/lammps_script_lj.in` — LAMMPS-side lj variant (`units lj`)
- [included] `verify/t1/run_differential.py` — расширить CLI: `--variant metal|lj|both`; `both` запускает obе and cross-checks identical thermo after unit conversion
- [included] `verify/benchmarks/t1_al_morse_500/checks.yaml` — новый блок `cross_check` (metal vs lj) с threshold `1e-10 rel`

## Out of scope
- [excluded] Переписывание config_metal.yaml (он остаётся unchanged)
- [excluded] Новый benchmark T1b — re-use existing T1 directory per D-M2-10

## Mandatory invariants
- T1 differential green в metal (no regression from M1)
- T1 differential green в lj (same thresholds, independent run)
- metal-vs-lj cross-check: post-conversion thermo residual ≤ 1e-10 rel на каждой column
- CI `differential-t1` job runs обе variants sequentially (still skips on public CI per Option A)

## Required files
- `verify/benchmarks/t1_al_morse_500/config_lj.yaml`
- `verify/benchmarks/t1_al_morse_500/lammps_script_lj.in`
- `verify/t1/run_differential.py` — CLI extension
- `verify/benchmarks/t1_al_morse_500/checks.yaml` — cross-check block
- `docs/specs/verify/SPEC.md` — дополнение §4.5 (T1 landed) про lj cross-check
- `verify/thresholds/thresholds.yaml` — `benchmarks.t1_al_morse_500.cross_unit_relative: 1.0e-10`

## Required tests
- [smoke locally] `run_differential.py --variant metal` passes
- [smoke locally] `run_differential.py --variant lj` passes
- [smoke locally] `run_differential.py --variant both` passes + cross-check green
- [CI] differential-t1 job SKIPs в public CI (unchanged)

## Expected artifacts
- Единый PR
- Pre-impl + session reports

## Acceptance criteria
- [ ] Both variants green locally vs LAMMPS (+the kB floor documented in M1)
- [ ] Cross-check green
- [ ] SPEC §4.5 updated
- [ ] Pre-commit green
- [ ] Human review approved

## Hints
- Identity reference (σ=ε=m=1) sufficient — no physical meaning needed для cross-check-only goal
- LAMMPS `units lj` + Morse — supported; params scaled trivially
- Cross-check implementation: parse both TDMD thermo files, diff column-by-column post-conversion (use existing compare.py)
```

---

### T2.5 — `TabulatedFunction` primitive (cubic spline + derivative)

```
# TDMD Task: Natural cubic spline tabulated function

## Context
- Master spec: §14 M2 (EAM requires tabulated functions); potentials/SPEC §4.4
- Module SPEC: docs/specs/potentials/SPEC.md §4.4 (LAMMPS-bit-match requirement)
- Role: Physics Engineer
- Depends on: M1 baseline
- Milestone: M2

## Goal
Общий primitive для EAM, будущего MEAM, SNAP: хранит tabulated `y(x_grid)` и выдаёт `eval(x)` + `derivative(x)` через cubic spline с pre-computed coefficients. Bit-for-bit match с LAMMPS interpolation — тут рождается T4 differential gate.

## Scope
- [included] `src/potentials/tabulated.hpp` + `.cpp`:
  - `class TabulatedFunction`
  - Ctor: `TabulatedFunction(vector<double> x_grid, vector<double> y_values)`; spline coeffs precomputed
  - Eval: `double eval(double x) const` — returns y(x); uniform-grid fast path + non-uniform-grid general path
  - Derivative: `double derivative(double x) const` — returns y'(x) analytically from spline coeffs
  - Extrapolation policy: below `x_grid[0]` → return `y_values[0]` + linear extension; above `x_grid.back()` → same. Matches LAMMPS `pair_eam.cpp:PairEAM::array2spline()`
- [included] Bit-for-bit match с LAMMPS: copy `array2spline` + `splint`/`splint_xderiv` алгоритмов из `verify/third_party/lammps/src/MANYBODY/pair_eam.cpp` (GPL-2.0) с license header + credit comment
- [included] Tests:
  - [unit] polynomial of degree ≤3 reproduced exactly до `1e-15 rel`
  - [unit] sin/cos table (1000 points, grid step 0.01) — eval residual ≤ 1e-6; derivative residual ≤ 1e-4
  - [unit] extrapolation returns boundary values
  - [bit-exact vs LAMMPS] identical input grid + values → same eval output byte-for-byte

## Out of scope
- [excluded] Monotonic cubic / PCHIP — natural cubic matches LAMMPS
- [excluded] 2D tabulation (для многокомпонентных MEAM) — M3+
- [excluded] SIMD / GPU paths — M6

## Mandatory invariants
- Eval / derivative exact для polynomial input ≤ deg 3 до FP64 epsilon
- Interpolation formula bit-identical LAMMPS (verified by copying + one-liner diff test)
- Const-correct: `eval` и `derivative` — `const` methods, no side effects
- `__restrict__` on inner-loop pointer args (master spec §D.16)

## Required files
- `src/potentials/tabulated.hpp`, `src/potentials/tabulated.cpp`
- `src/potentials/CMakeLists.txt` — add sources
- `tests/potentials/test_tabulated.cpp`

## Required tests
- [unit] 5+ cases per bullets в scope
- [property ≥10⁴] random cubic `a·x³ + b·x² + c·x + d` reproduced до 4 ulp
- [bit-exact] fixture matching LAMMPS fixture output (prepare reference bytes once)

## Expected artifacts
- Единый PR
- Pre-impl + session reports
- LAMMPS credit + license header в tabulated.cpp

## Acceptance criteria
- [ ] Polynomial-degree-3 reproduction exact
- [ ] Bit-exact LAMMPS match verified
- [ ] All tests green gcc + clang
- [ ] No tdmd-missing-restrict lints
- [ ] Human review approved

## Hints
- LAMMPS `array2spline`: pre-computes 7 coefficients per grid cell; у нас такая же схема
- Не изобретай свою форму — bit-match — hard requirement D-M2-1
- Test fixture для bit-match: можно сделать небольшой binary file (< 1 KB) с known-good LAMMPS output
```

---

### T2.6 — EAM parameter file parsers (`.eam.alloy` + `.eam.fs`)

```
# TDMD Task: Parse LAMMPS-compatible EAM parameter files

## Context
- Master spec: §14 M2
- Module SPEC: docs/specs/potentials/SPEC.md §4.5
- Role: Physics Engineer
- Depends on: T2.5 (TabulatedFunction)
- Milestone: M2

## Goal
Читать `.eam.alloy` и `.eam.fs` файлы (LAMMPS-compatible text format) → набор `TabulatedFunction` instances + metadata (species names, masses, grid params). Parser живёт в `potentials/` per D-M2-3.

## Scope
- [included] `src/potentials/eam_file_parser.hpp` + `.cpp`:
  - `struct EamAlloyData { N_species, species_names, masses, F_funcs, rho_funcs, phi_funcs, cutoff }`
  - `struct EamFsData { ... phi_funcs, rho_funcs_pair[species][species], ... }`
  - `EamAlloyData parse_eam_alloy(path)` — throws `std::runtime_error` on malformed
  - `EamFsData parse_eam_fs(path)` — same pattern
- [included] Support both file variants:
  - `.eam.alloy`: 3 comment lines; 1 line with N_species + names; 1 line with N_rho d_rho N_r d_r r_cutoff; per-species `Z mass a lattice` + F values + rho values; phi values (N_species·(N_species+1)/2 × N_r)
  - `.eam.fs`: similar but rho depends on species pair — N_species² × N_r rho values
- [included] Parse tests:
  - Canonical Al99.eam.alloy (Mendelev 2008) — commit truncated version (first 200 values) в `tests/potentials/fixtures/`
  - Multi-species AlNi.eam.alloy — similar fixture
- [included] Robust против whitespace, Windows line endings, comment line variance

## Out of scope
- [excluded] Binary EAM files (не поддерживаются LAMMPS либо)
- [excluded] Setfl/Funcfl distinction — обе LAMMPS-style files в одном parser (format inferred by file extension)
- [excluded] Multi-cutoff (pair vs density separate) — M3+

## Mandatory invariants
- Parser reject mal-formed файл с clear error message (file:line:message format)
- Numeric values parse to FP64 без loss of precision
- Ouput — `TabulatedFunction` objects с grid consistency invariant: `|x_grid| == |y_values|`

## Required files
- `src/potentials/eam_file_parser.hpp/.cpp`
- `tests/potentials/test_eam_file_parser.cpp`
- `tests/potentials/fixtures/{Al99_truncated.eam.alloy, AlNi_truncated.eam.fs}`

## Required tests
- [unit] canonical alloy file — all fields populated как expected
- [unit] canonical fs file — все `N²` rho-pairs
- [unit] malformed (missing N_r, truncated) — throw with diagnostic
- [bit-exact] parsed output идентично LAMMPS parsed output (cross-check via one-shot LAMMPS debug dump)

## Expected artifacts
- Единый PR
- Pre-impl + session reports

## Acceptance criteria
- [ ] Both formats parse
- [ ] Diagnostic on failure
- [ ] Bit-exact LAMMPS parity на canonical files
- [ ] Tests green gcc+clang
- [ ] Human review approved

## Hints
- LAMMPS reference: `src/MANYBODY/pair_eam.cpp:PairEAM::read_alloy()` и `read_fs()` — идеальный template
- Do NOT copy/paste — parser не под GPL-2.0 (unlike T2.5 math); rewrite with fresh code, reference LAMMPS source in comment
- Commit only truncated fixture files (~ 10 KB), full reference files stay в verify/reference/eam/ (gitignored, downloaded by harness)
```

---

### T2.7 — `EamAlloyPotential` + `EamFsPotential` CPU FP64

```
# TDMD Task: EAM force kernel (both variants)

## Context
- Master spec: §14 M2 (M2 gate artifact); potentials/SPEC §4.3
- Role: Physics Engineer
- Depends on: T2.5 (TabulatedFunction), T2.6 (parsers)
- Milestone: M2

## Goal
Central M2 deliverable. Реализовать two-pass EAM force evaluation (density pass + force pass) для обоих вариантов параметризации (alloy, FS). Forces должны bit-match LAMMPS до `1e-10 rel` на canonical test system (T4).

## Scope
- [included] `src/potentials/eam_alloy.hpp/.cpp`:
  - `class EamAlloyPotential : public Potential` (inherits from whatever abstract потенциал interface был landed в M1 для Morse; если не было формального base class — вводится в этой task'е, `Potential` header)
  - Ctor: `EamAlloyPotential(EamAlloyData data, CutoffStrategy = HardCutoff)`
  - `compute(AtomSoA&, NeighborList&, Box&)` — two-pass:
    - Pass 1: per-atom ρ accumulation via neighbor half-list (newton on)
    - Pass 2: per-atom dF/dρ eval; force contribution from both embedding term and pair term
- [included] `src/potentials/eam_fs.hpp/.cpp` — identical force math, different parameter source (per-pair ρ_ij instead of per-species ρ_α)
- [included] Newton's 3rd law preserved via half-list accumulation; force symmetric pairs: f_i += Δf, f_j -= Δf
- [included] Virial tensor accumulation (for pressure thermo) — Clausius convention, matching Morse
- [included] Cutoff handling: HardCutoff only в M2 (потенциалы/SPEC §2.4, Strategy A); Smoothing deferred
- [included] If abstract `Potential` base does not exist yet — introduce it in this task (`compute`, `cutoff`, `name`); refactor MorsePotential to inherit; no behavior change

## Out of scope
- [excluded] GPU kernels — M6
- [excluded] Shifted-force (Strategy C) for EAM — M3+
- [excluded] MEAM / SNAP — M4+ milestone work
- [excluded] Multi-cutoff support — M3+

## Mandatory invariants
- Newton's 3rd law: `Σ forces = 0` до 1e-12 rel (property test, ≥10⁴ random configurations)
- Virial symmetric (W[xy] == W[yx] modulo rounding)
- Two-pass: density pass completes before force pass starts (no interleaving)
- `__restrict__` on all hot-path pointers
- No hidden allocations in compute() — per-atom ρ buffer allocated once, reused

## Required files
- `src/potentials/eam_alloy.hpp/.cpp`
- `src/potentials/eam_fs.hpp/.cpp`
- `src/potentials/potential.hpp` — abstract base (if new)
- `src/potentials/morse.hpp/.cpp` — refactor to inherit (minor; no behavior change)
- `src/potentials/CMakeLists.txt` — add sources
- `tests/potentials/test_eam_alloy.cpp`, `test_eam_fs.cpp`

## Required tests
- [unit] single-atom density: manual ρ calculation matches
- [unit] 2-atom system: F and E match closed-form (for specific toy EAM)
- [property ≥10⁴] Newton 3rd law residual ≤ 1e-12 rel
- [property ≥10⁴] virial symmetry residual ≤ 1e-14
- [regression] MorsePotential tests continue passing (interface refactor safe)

## Expected artifacts
- Большой PR (~800-1200 LOC); allowed to split если > 1500 LOC
- Pre-impl + session reports
- Если `Potential` base introduced: explicit note в PR description

## Acceptance criteria
- [ ] Both classes compile, tests green
- [ ] Newton 3rd law + virial invariants
- [ ] No regressions in Morse
- [ ] Pre-commit green
- [ ] Ready to be plugged into T4 differential (T2.9)
- [ ] Human review approved

## Hints
- Two-pass density: храни ρ[i] и dF/dρ[i] в EamAlloyPotential как member vectors; resize на n_atoms at compute() start; no per-step alloc
- Neighbor list: reuse T1.6 half-list; density pass и force pass оба итерируются по тем же парам
- Virial: `W_αβ = Σ F_iα · r_ijβ` — ровно как в Morse
- Forces + E bit-match с LAMMPS — потребуется T4 (T2.9); здесь достаточно invariants + analytic toy tests
```

---

### T2.8 — DifferentialRunner MVP (forces dump comparison)

```
# TDMD Task: Extend differential harness to compare per-atom forces

## Context
- Master spec: §14 M2; verify/SPEC §7.1 (differential runner)
- Role: Validation / Reference Engineer
- Depends on: T2.7 (EAM ready для T4 consumer)
- Milestone: M2

## Goal
M1 T1.11 harness сравнивает only thermo (PE/KE/T/P). Для T4 EAM gate нужно forces-уровень сравнение (per-atom F vectors). Эта task расширяет existing `verify/compare.py` + `verify/t1/run_differential.py` до полного DifferentialRunner MVP: парсинг dump files (LAMMPS `dump custom id type x y z fx fy fz`), alignment по id, column-wise comparison.

## Scope
- [included] `verify/compare.py`: добавить
  - `parse_lammps_dump(path)` — parses `dump` files со "ITEM: ATOMS" header
  - `parse_tdmd_dump(path)` — TDMD emits compatible format (новый CLI flag `tdmd run --dump <path>`)
  - `compare_forces(tdmd_rows, lmp_rows, threshold_rel)` — align by atom id, per-component residual, returns max + at-atom-id
- [included] `verify/harness/differential_runner.py` — генеральный driver (не T1-specific); принимает benchmark dir + thresholds + variant flags; запускает LAMMPS → TDMD → сравнивает thermo + forces (если both emit dumps) → report
- [included] CLI: `tdmd run --dump <path>` — emits per-atom dump после финального шага в LAMMPS-compatible format
- [included] Refactor: T1.11 driver (`verify/t1/run_differential.py`) now thin wrapper around общий `differential_runner.py`

## Out of scope
- [excluded] Full trajectory dump (каждый thermo step) — M3+
- [excluded] Velocity comparison — forces + positions sufficient для M2; velocities validated indirectly via KE thermo
- [excluded] Binary dump format — text only

## Mandatory invariants
- `tdmd run --dump` output parseable by LAMMPS `rerun` command (format compliance)
- Force comparison aligned by `id` (integer), not row order (LAMMPS may reorder)
- Residual reported per-component + scalar max

## Required files
- `verify/compare.py` — extend
- `verify/harness/differential_runner.py` — new (или `verify/t1/run_differential.py` generalized)
- `src/runtime/simulation_engine.cpp` — add dump emission
- `src/cli/run_command.cpp` — add `--dump` flag
- `tests/cli/test_dump_flag.cpp`

## Required tests
- [unit] parse_lammps_dump correctness on fixture
- [unit] parse_tdmd_dump correctness on fixture
- [unit] compare_forces detection of injected differences
- [integration] T1 differential continues passing (no regression — still thermo-only for T1)

## Expected artifacts
- Единый PR
- Pre-impl + session reports

## Acceptance criteria
- [ ] Forces comparison library works
- [ ] `tdmd run --dump` emits LAMMPS-compatible output
- [ ] T1.11 harness refactored, still green
- [ ] No regressions
- [ ] Human review approved

## Hints
- LAMMPS dump format: `ITEM: ATOMS id type x y z fx fy fz` with line per atom; parse is stdlib-friendly
- TDMD dump emit: after final step, iterate atoms, print `id type x y z fx fy fz` в 10-digit scientific
- Id alignment: critical — LAMMPS reorders atoms during migration; sort by id before compare
```

---

### T2.9 — T4 `nial-alloy` benchmark (M2 ACCEPTANCE GATE)

```
# TDMD Task: Ni-Al EAM differential benchmark vs LAMMPS

## Context
- Master spec: §13.2 (T4), §14 M2 — **mandatory gate**
- Role: Validation / Reference Engineer
- Depends on: T2.7, T2.8
- Milestone: M2

## Goal
Главный artifact gate milestone'а. Ni-Al FCC alloy, EAM (Mishin 2004), forces + thermo diff vs LAMMPS до `1e-10 rel`. Без T4 green — M2 не закрыт.

## Scope
- [included] `verify/benchmarks/t4_nial_alloy/`:
  - README.md — полное описание system + rationale + acceptance criteria
  - config.yaml — TDMD input (metal units initially; lj variant — M3)
  - setup.data — committed Ni-Al FCC 6×6×6 (864 atoms); Ni:Al ratio 50:50 random seed 12345; LAMMPS-generated
  - lammps_script.in — oracle script
  - checks.yaml — forces comparison (via T2.8 DifferentialRunner MVP) + thermo comparison (existing)
- [included] Reference EAM parameters `AlNi.eam.alloy` (Mishin 2004) — downloaded at harness setup (not committed, LAMMPS examples dir has it, ~50 KB)
- [included] Threshold entries `benchmarks.t4_nial_alloy.forces_relative: 1.0e-10` + thermo entries mirroring T1
- [included] CI integration — differential-t4 job (SKIPs on public CI same as T1)
- [included] Local pre-push: green в differential_runner

## Out of scope
- [excluded] T4 lj variant — M3+
- [excluded] Medium-tier observables (MSD, RDF) — M3+
- [excluded] Trajectory comparison — use final-frame forces + thermo

## Mandatory invariants
- Forces rel ≤ 1e-10 per-atom-per-component (per EAM acceptance in potentials/SPEC §4.6)
- Thermo rel ≤ 1e-10 на PE/KE/Etotal; temperature ≤ 2e-6 (same kB floor as T1)
- NVE dt=0.001, 100 steps (same profile as T1 для consistency)
- Both Ni and Al species populated, non-trivial force interactions

## Required files
- `verify/benchmarks/t4_nial_alloy/{README.md, config.yaml, setup.data, lammps_script.in, checks.yaml}`
- `verify/thresholds/thresholds.yaml` — new `benchmarks.t4_nial_alloy.*` block
- `verify/t4/{CMakeLists.txt, run_differential.py (or wrapper), test_t4_differential.cpp}`
- `verify/CMakeLists.txt` — `add_subdirectory(t4)`
- `.github/workflows/ci.yml` — differential-t4 job (mirrors differential-t1 structure)
- `docs/specs/verify/SPEC.md` — new §4.7 (T4 landed, ниже existing T1 section)

## Required tests
- [integration local] differential_runner green с LAMMPS oracle
- [CI] differential-t4 job SKIPs cleanly on public CI (LAMMPS absent)
- [regression] T1 differential continues green

## Expected artifacts
- Единый PR
- Pre-impl + session reports
- SPEC update committed along with implementation

## Acceptance criteria
- [ ] Forces ≤ 1e-10 rel per-component
- [ ] Thermo ≤ existing thresholds
- [ ] SPEC §4.7 published
- [ ] CI green (SKIP on public, PASS on local)
- [ ] Human review approved

## Hints
- Mishin 2004 file: LAMMPS ships `potentials/AlNi.eam.alloy`; check pre-existing copy в submodule or download via `tools/download_eam_ref.sh` (new script)
- Ni-Al random configuration — use Python one-shot generator committed в benchmark dir (not LAMMPS script) для reproducibility без LAMMPS
- Expected forces magnitude: ~1-10 eV/Å — residual 1e-10 rel → abs 1e-10..1e-9 eV/Å
```

---

### T2.10 — `PerfModel::predict` (Pattern 1 + Pattern 3)

```
# TDMD Task: Analytic performance prediction (SD vs TD)

## Context
- Master spec: §14 M2; perfmodel/SPEC §§2–3, §6
- Role: GPU / Performance Engineer (despite CPU-only scope — model formulas are HW-agnostic)
- Depends on: M1 baseline
- Milestone: M2

## Goal
Первая версия `PerfModel::predict()` — analytic time-to-step для Pattern 1 (чистый TD) и Pattern 3 (чистый SD). Formulas из perfmodel/SPEC §3.2 + §3.3. Calibration — hardcoded hardware profile (modern x86_64 + default CUDA; autodetect — M4).

## Scope
- [included] `src/perfmodel/perfmodel.hpp` + `.cpp`:
  - `struct HardwareProfile { cpu_flops_per_sec, mem_bw_bytes_per_sec, pci_bw, ... }`
  - `struct PotentialCost { flops_per_pair, bytes_per_pair, ... }`
  - `struct PerfPrediction { pattern_name, t_step_sec, recommended_K, speedup_vs_baseline }`
  - `class PerfModel`:
    - `PerfModel(HardwareProfile, PotentialCost)`
    - `PerfPrediction predict_pattern1(n_atoms, K)` — Pattern 1 formula §3.3
    - `PerfPrediction predict_pattern3(n_atoms)` — Pattern 3 baseline §3.2
    - `std::vector<PerfPrediction> rank(n_atoms)` — sorted by t_step_sec ascending
- [included] Hardcoded `HardwareProfile::modern_x86_64()` factory с numbers из perfmodel/SPEC §4.1
- [included] Potential cost table — Morse + EAM (Morse: low, EAM: ~3× Morse) per §4.2
- [included] `K_opt` calculation: `K_opt ≈ sqrt(T_p / T_c_startup)` rounded up to power of 2; clamped [1, 16]
- [included] Unit tests: на тривиальных cases (малое N, SD > TD из-за TD overhead; большое N, TD > SD)
- [included] Validation harness stub (для M7 gates) — пустой placeholder-тест, no assertions yet

## Out of scope
- [excluded] Pattern 2 (two-level) — M7
- [excluded] Auto-calibration — M4
- [excluded] GPU profile — M6
- [excluded] Real-world validation gates — M7

## Mandatory invariants
- Prediction positive, finite для any sane n_atoms > 10
- `predict_pattern1` с K=1 ≤ 1.5× `predict_pattern3` (TD overhead at K=1 — bounded)
- `predict_pattern1` с K=K_opt <  `predict_pattern3` для large n_atoms
- `rank()` вывод stable (deterministic sort on ties)

## Required files
- `src/perfmodel/perfmodel.hpp/.cpp`
- `src/perfmodel/hardware_profile.hpp`
- `src/perfmodel/CMakeLists.txt` — new if skeleton missing
- `tests/perfmodel/test_perfmodel.cpp`
- `CMakeLists.txt` — add `add_subdirectory(src/perfmodel)` if needed

## Required tests
- [unit] SD trivial monotone increase in n_atoms
- [unit] TD Pattern 1 с K=1 close to SD (overhead-only regime)
- [unit] K_opt scales ~sqrt(n_atoms) in asymptotic range
- [unit] rank() output ordering correct

## Expected artifacts
- Единый PR
- Pre-impl + session reports

## Acceptance criteria
- [ ] Compile, tests green
- [ ] Formulas ровно как в SPEC §3.2-3.3 (cite line numbers in comments)
- [ ] Human review approved

## Hints
- Не изобретай: формулы — copy из perfmodel/SPEC §3 и cite line number в comment
- `HardwareProfile::modern_x86_64` — hardcoded dict; values из §4.1
- Pattern naming consistent: internal Pattern1 = "TD", Pattern3 = "SD", Pattern2 reserved (do not implement)
```

---

### T2.11 — `tdmd explain --perf` subcommand

```
# TDMD Task: CLI integration of performance predictions

## Context
- Master spec: §14 M2 artifact: "explain --perf корректно ранжирует SD > TD на тривиальных случаях"
- Module SPEC: docs/specs/cli/SPEC.md §5.3; docs/specs/perfmodel/SPEC.md §6
- Role: Scientist UX Engineer
- Depends on: T2.10
- Milestone: M2

## Goal
T1.10 landed `tdmd explain --field <key>` — readable field-level schema help. Теперь расширяем до `tdmd explain --perf <config.yaml>` — принимает config, строит `PerfModel::predict`, выводит rec ranking.

## Scope
- [included] `src/cli/explain_command.cpp` — новый `--perf` flag:
  - Parses config (re-uses T1.4 YAML loader)
  - Extracts n_atoms (from committed setup.data или preflight-derived) + potential type
  - Invokes `PerfModel::rank(n_atoms)` (T2.10)
  - Prints ranking in format per perfmodel/SPEC §6.1
- [included] Output format — human-readable (не JSON в M2, per D-M2-7):
  ```
  Recommended deployment: Pattern 3 (SD-only)  [M2: TD не implemented, SD is only option]
    Expected T_step:       2.3 ms
    Expected throughput:   N per day
  Alternatives:
    Pattern 1 (TD, K=4):  T_step = 4.7 ms (NOT AVAILABLE until M4)
  ```
- [included] Exit code 0 on success; 2 on config parse failure (preflight error)

## Out of scope
- [excluded] `--format json` — D-M2-7
- [excluded] TD recommendation as primary — TD doesn't exist yet until M4; model always recommends SD in M2

## Mandatory invariants
- Invocation не запускает MD — только analytic prediction
- Output ≤ 40 lines; stable ordering (deterministic)
- Deprecation-safe: explain --field still works (T1.10 regression)

## Required files
- `src/cli/explain_command.cpp` — extend
- `src/cli/CMakeLists.txt` — no new file, existing wiring
- `tests/cli/test_explain_perf.cpp` — new

## Required tests
- [integration CLI] `tdmd explain --perf <small-config>` exits 0, stdout contains "Pattern 3"
- [integration CLI] `tdmd explain --perf <malformed-config>` exits 2
- [regression] `tdmd explain --field <key>` (T1.10) continues green

## Expected artifacts
- Единый PR
- Pre-impl + session reports

## Acceptance criteria
- [ ] SD > TD ranking в trivial case (TD shown as "not available")
- [ ] Config integration works
- [ ] T1.10 --field behaviour preserved
- [ ] Human review approved

## Hints
- Re-use `ConfigLoader` from T1.4; no new parser
- N_atoms extraction: either parse setup.data header (first 20 bytes yield `N atoms`) or delegate to preflight
```

---

### T2.12 — Telemetry skeleton (JSONL sink + LAMMPS-compatible breakdown)

```
# TDMD Task: Basic telemetry sink with timing breakdown

## Context
- Master spec: §14 M2 artifact: "telemetry skeleton (timing breakdown в LAMMPS-compatible формате)"
- Module SPEC: docs/specs/telemetry/SPEC.md §§3, 4.2
- Role: Core Runtime Engineer
- Depends on: M1 baseline
- Milestone: M2

## Goal
Первая версия telemetry: per-step timing breakdown записан в JSONL sink + LAMMPS-compatible текстовый dump на `tdmd run --timing`. Async / NVTX / distributed telemetry — M5+; здесь только локальная synchronous запись.

## Scope
- [included] `src/telemetry/telemetry.hpp` + `.cpp`:
  - `class Telemetry`:
    - `void begin_section(std::string_view name)`
    - `void end_section(std::string_view name)`
    - `std::map<std::string, double> current_breakdown()` — returns per-section accumulated time
    - `void write_jsonl(std::ostream&)` — writes current snapshot
    - `void write_lammps_format(std::ostream&)` — writes table per telemetry/SPEC §4.2
- [included] Instrumentation hooks в SimulationEngine: sections "Pair", "Neigh", "Comm" (stub at 0 since no MPI в M2), "Output", "Other"
- [included] CLI: `tdmd run --timing` flag; prints LAMMPS-format summary to stderr at end of run; `--telemetry-jsonl <path>` writes JSONL
- [included] Overhead budget: < 0.1% wall-time per step (measured in integration test)

## Out of scope
- [excluded] NVTX — M6 (GPU)
- [excluded] Async writer — M5
- [excluded] Structured fields beyond time (counters, rates) — M3+
- [excluded] Hot-reload / runtime config — never

## Mandatory invariants
- Timing overhead < 0.1% (assertion in a benchmark test)
- JSONL output valid JSON (one object per line)
- LAMMPS-format breakdown columns match oracle: `Section | min | avg | max | %varavg | %total`

## Required files
- `src/telemetry/telemetry.hpp/.cpp`
- `src/telemetry/CMakeLists.txt`
- `src/runtime/simulation_engine.cpp` — add begin/end hooks
- `src/cli/run_command.cpp` — add `--timing` + `--telemetry-jsonl`
- `tests/telemetry/test_telemetry.cpp`
- `CMakeLists.txt` — hook `add_subdirectory(src/telemetry)` if new

## Required tests
- [unit] begin/end with timing accumulation
- [unit] JSONL output parseable by Python `json.loads`
- [unit] LAMMPS-format output matches regex expected columns
- [integration] overhead < 0.1% (benchmark test, 10⁴ steps)

## Expected artifacts
- Единый PR
- Pre-impl + session reports

## Acceptance criteria
- [ ] Both outputs work from CLI
- [ ] Overhead budget held
- [ ] No regressions
- [ ] Human review approved

## Hints
- LAMMPS format reference — grep `verify/third_party/lammps/src/*.cpp` для "MPI task timing breakdown"
- Section names fixed (mirror LAMMPS) для easy diff against oracle
- Для timing — use `std::chrono::steady_clock`
```

---

### T2.13 — M2 integration smoke: end-to-end EAM + telemetry + explain

```
# TDMD Task: M2 integration smoke, closes the milestone

## Context
- Master spec: §14 M2 final artifact
- Role: Validation / Reference Engineer
- Depends on: T2.7, T2.10, T2.11, T2.12
- Milestone: M2 (final)

## Goal
End-to-end smoke (< 10s, CI-integrated) — прогоняет Ni-Al EAM 10 steps, emits telemetry, runs explain --perf, compares всё к committed goldens. Mirrors T1.12 в scope.

## Scope
- [included] `tests/integration/m2_smoke/`:
  - README.md
  - smoke_config.yaml.template — Ni-Al EAM config
  - setup.data — committed 64-atom Ni-Al (меньше T4's 864 для смок-speed)
  - eam_params.alloy — committed small EAM parameter file (~10 KB)
  - al_ni_eam_10steps_golden.txt — committed thermo golden
  - telemetry_golden.jsonl — committed telemetry schema golden (not values — schema check only)
  - run_m2_smoke.sh — driver (like T1.12)
- [included] `.github/workflows/ci.yml` — "M2 smoke" step в build-cpu matrix jobs
- [included] Smoke runs без LAMMPS dependency (ингредиенты committed)

## Out of scope
- [excluded] Full T4 (medium tier) — it's T2.9
- [excluded] TD — M4+
- [excluded] LJ cross-check — covered by T2.4

## Mandatory invariants
- Wall-time < 10s CI budget
- Bit-exact thermo match (same-compiler-same-arch)
- Cross-compiler (gcc-13 vs clang-17) match — tolerance backup if needed
- Smoke stable under shuffled run order (deterministic)

## Required files
- `tests/integration/m2_smoke/*` (6 files)
- `.github/workflows/ci.yml`

## Required tests
- [smoke local] < 10s, exit 0
- [CI] passes on gcc-13 и clang-17
- [regression] M1 smoke + T1 differential + T0 не ломаются

## Expected artifacts
- Единый PR
- Pre-impl + session reports

## Acceptance criteria
- [ ] All artifacts committed
- [ ] CI green
- [ ] Smoke < 10s
- [ ] M2 acceptance gate (§5) fully closed
- [ ] Human review approved

## Hints
- Reduce N_atoms от T4 (864) к ~64 для smoke speed — reference cell 2×2×2 Ni FCC + Al substitutions
- Small EAM parameter file — truncate grid (N_r=200, not 1000) — mathematically valid, smaller file
```

---

## 5. M2 Acceptance Gate

После закрытия всех 13 задач — проверить полный M2 artifact gate (master spec §14 M2):

- [ ] **UnitConverter lj full support** (T2.1), round-trip + property tests green
- [ ] **data+yaml lj ingestion** (T2.2), preflight rules updated
- [ ] **T0 morse-analytic** (T2.3) green в обоих unit systems
- [ ] **T1 lj cross-check** (T2.4) — metal + lj produce identical thermo post-conversion
- [ ] **TabulatedFunction** (T2.5), bit-match LAMMPS interpolation
- [ ] **EAM file parsers** (T2.6), bit-exact LAMMPS parse output
- [ ] **EamAlloyPotential + EamFsPotential** (T2.7), Newton 3rd law + virial invariants green
- [ ] **DifferentialRunner MVP** (T2.8), forces comparison library exercised
- [ ] **T4 nial-alloy** (T2.9) — **mandatory gate** — forces rel ≤ 1e-10 vs LAMMPS
- [ ] **PerfModel::predict** (T2.10), Pattern 1 + Pattern 3 analytic
- [ ] **tdmd explain --perf** (T2.11), SD > TD ranking в trivial cases
- [ ] **Telemetry skeleton** (T2.12), JSONL + LAMMPS-format breakdown
- [ ] **M2 smoke** (T2.13) < 10s, CI-integrated
- [ ] **T1 differential continues green** (no regression)
- [ ] CI Pipelines A (lint+build+smokes) + B (unit/integration) + C (differential T1 + T4) active
- [ ] Pre-implementation + session reports attached в каждом PR
- [ ] Human review approval для каждого PR

---

## 6. Risks & Open Questions

**Risks:**

- **R-M2-1 — EAM spline bit-match fragility.** LAMMPS interpolation formula use specific ordering of floating-point ops; a stealth diff (e.g. compiler reordering FMA) could break T4. Mitigation: explicit `volatile` or `#pragma STDC FP_CONTRACT OFF` in the copied LAMMPS code path; verify via fixture in T2.5.
- **R-M2-2 — EAM parameter file format drift.** LAMMPS sometimes emits slightly different file layouts across versions. Mitigation: pin Mishin 2004 file хранится в `verify/reference/eam/` with checksum; T2.6 parser tested against exact bytes.
- **R-M2-3 — PerfModel prediction drift from reality.** M2 predictions не gated (no validation thresholds); risk low в M2 but critical в M5/M7. Mitigation: `validate_prediction()` harness stub в T2.10 ready для M5 Pattern 1 validation.
- **R-M2-4 — Telemetry overhead > 0.1%.** If naive implementation (map lookup per begin/end) busts budget. Mitigation: integration test в T2.12 explicitly benchmarks overhead на 10⁴ step run.

**Open questions (deferred to task-time decisions):**

- **OQ-M2-1 — Abstract `Potential` base class.** М1 MorsePotential не унаследован от abstract base. T2.7 вводит его if absent. Question: does introducing the base require a SPEC delta? **Answer:** no, interface already specified in potentials/SPEC §2.1; T2.7 implements что там написано.
- **OQ-M2-2 — Should DifferentialRunner be C++ or Python?** verify/SPEC §7 показывает pseudo-Python. T2.8 — Python-first; if T2.9 T4 gate requires deeper intro-inspection of TDMD state, could be refactored to C++ в M3. **Decision:** Python until evidence of need.
- **OQ-M2-3 — Where does `K_opt` clamping policy live?** Master spec §6.5a — "auto-K policy lives in master spec §6.5a". T2.10 implements formula-based K_opt; clamping [1,16] — per master spec §6.5a or emergent? **Decision:** [1,16] в T2.10; если §6.5a говорит другое — fix в T2.10 PR review.

---

## 7. Roadmap Alignment

| Deliverable | Consumer milestone | Why it matters |
|---|---|---|
| `UnitConverter lj` full (T2.1-T2.4) | M3+ для scientist UX | `lj` — default в academic community; без full support TDMD невидим в academic workflows |
| `TabulatedFunction` (T2.5) | M3 (MEAM), M4 (SNAP) | Reusable — same spline library used by MEAM density, SNAP descriptors |
| `EamAlloyPotential` + `EamFsPotential` (T2.7) | M3 zoning (для realistic workload benchmarks), M4 TD scheduler | Many-body potential — canonical test case для TD эффективности (dissertation focus) |
| T4 benchmark (T2.9) | M3, M4, M5 — regression gate | Prevents EAM regression when zoning / scheduler work starts modifying neighbor + compute paths |
| DifferentialRunner MVP (T2.8) | M3-M5 для all many-body potentials | Core infrastructure для scientific credibility |
| PerfModel Pattern 1/3 (T2.10) | M4 auto-K calibration; M5 K-batching selection; M7 validation gates | Без analytic model TD benefits — guesswork; anchor-test (M5) needs predictions для comparison |
| `tdmd explain --perf` (T2.11) | M4+ scientist UX — "why SD instead of TD for my system?" | Preparation для TD rollout в M4 |
| Telemetry skeleton (T2.12) | M5 K-batching diagnostics, M6 GPU profiling, M7 Pattern-2 validation | Without telemetry — no visibility into where TD overhead goes; debugging blind |

---

*Конец m2_execution_pack.md v0.1, draft 2026-04-18.*
