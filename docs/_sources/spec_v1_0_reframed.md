# TDMD Engineering Spec

Версия: 1.0-reframed  
Статус: master engineering spec  
Назначение: главный живой документ проекта TDMD. Все дальнейшие архитектурные, методические и продуктовые изменения должны сначала отражаться здесь, затем — в модульных `SPEC.md`, коде, тестах и execution pack.

---

## 1. Новое стратегическое позиционирование проекта

### 1.1. Что именно мы строим

**TDMD** — это самостоятельная программа молекулярной динамики, построенная вокруг идеи **декомпозиции по времени (Time Decomposition, TD)**, но ориентированная не на «универсальный MD для всего подряд» и не на «ML-first движок», а прежде всего на класс задач, где TD даёт наиболее естественное и ощутимое архитектурное преимущество.

Главная целевая ниша TDMD:

> **дорогие локальные many-body потенциалы высокой вычислительной сложности**,  
> для которых стандартная spatial decomposition (SD) начинает заметно деградировать из-за роста halo-обмена, увеличения stencil/dependency radius и усложнения локального вычисления.

### 1.2. Почему это новая и правильная фокусировка

Исходная идея TD Андреева особенно сильна там, где:

- взаимодействие локально по пространству;
- но вычислительно тяжело по локальной структуре;
- и цена разделения шага по пространству между многими процессорами становится высокой.

В простых pairwise задачах TD интересен, но не обязательно даёт драматический выигрыш относительно хорошо отточенных SD-кодов.

В **high-order local many-body** задачах ситуация другая:

- растёт цена halo-обмена;
- растёт число зависимых локальных сущностей;
- растёт цена поддержания согласованного соседского контекста между процессами;
- возрастает выигрыш от схемы, где целый шаг для согласованной локальной области исполняется без пространственного разрыва вычисления.

Поэтому новая продуктовая рамка TDMD такая:

> **TDMD — это специализированный high-performance TD-oriented engine для локальных many-body взаимодействий, прежде всего классических, с расширением в сторону локальных ML и гибридов.**

### 1.3. Что больше не является главным смыслом проекта

Не считаются центральной идентичностью проекта:

- «ещё один универсальный MLIP engine»;
- «просто клон LAMMPS с TD»;
- «универсальный MD-код для любых физических задач».

LAMMPS остаётся:

- эталоном для верификации корректности физики;
- эталоном для части архитектурных решений;
- эталоном для UX-паттернов scientific software;
- baseline для performance-сравнений.

Но TDMD не позиционируется как его копия.

---

## 2. Приоритетная физическая область проекта

### 2.1. Primary target: local classical many-body

Первая и главная целевая область TDMD:

1. `EAM`
2. `EAM/FS`
3. `MEAM`
4. `MEAM-like` / angular local many-body
5. generic local 3-body / 4-body / angular models
6. local bond-order-like models без обязательного глобального solve

Это основной фокус проекта.

### 2.2. Secondary target: reactive many-body

Вторая, более сложная волна:

- `ReaxFF`-класс моделей;
- reactive force fields с dynamic bonding;
- модели с charge equilibration / related inner solves.

Это направление считается **стратегически важным, но не стартовым**.

Причина:

- оно действительно потенциально выгодно для TD;
- но архитектурно существенно сложнее из-за charge-equilibration и сопутствующих глобальных/квазиглобальных процедур.

### 2.3. Tertiary target: local ML potentials

Локальные ML-потенциалы остаются важным направлением, но не главным смыслом TDMD.

Поддерживаются как **вторичная и расширяемая capability**:

- `SNAP`
- `MLIAP`
- `PACE/ACE`
- локальные descriptor-based модели с конечным cutoff

### 2.4. Future target: hybrid classical + ML

Долгосрочно наиболее сильная траектория проекта:

- classical many-body base
- + local ML correction
- + task-specific hybrid workflows

То есть TDMD должен проектироваться как:

> **classical-many-body-first, local-ML-compatible, hybrid-ready**

---

## 3. Цели проекта

### 3.1. Научная цель

Создать MD-систему, которая:

- реализует декомпозицию по времени как первичный принцип исполнения;
- особенно эффективна для локальных many-body взаимодействий;
- обеспечивает проверяемую физическую корректность;
- остаётся пригодной для научной публикационной работы.

### 3.2. Инженерная цель

Создать программу, которая:

- standalone;
- GPU-oriented;
- детерминируемо валидируема;
- поддерживает scientist-friendly workflows;
- может разрабатываться в основном ИИ-агентами без расползания архитектуры.

### 3.3. Продуктовая цель

Получить не просто экспериментальный код, а **научно пригодный инструмент**, у которого есть:

- чёткое позиционирование;
- reproducibility story;
- compare path with LAMMPS;
- benchmark governance;
- recipe-based UX.

---

## 4. Методологическая опора: TD Андреева в современной трактовке

### 4.1. Что сохраняется из исходной идеи

Из диссертационной идеи TD сохраняется главное:

- область можно продвигать по времени, если причинная зависимость от удалённых областей гарантированно отсутствует на данном окне;
- зона — естественная единица работы;
- корректность держится на буфере, ограничении смещения и контроле зависимостей.

### 4.2. Что меняется в современной реализации

Современный TDMD **не** реализует примитивный pipeline «процессор считает h, следующий считает h+1». Вместо этого используется:

- scheduler причинных волн;
- граф зависимостей зон;
- safety certificates;
- hybrid `time × space` decomposition;
- GPU-resident data path.

### 4.3. Практический смысл для целевой ниши

TD выгоден прежде всего там, где локальное вычисление тяжёлое, а пространственное разрезание шага дорого. Именно поэтому TDMD должен быть заточен под local many-body.

---

## 5. Главные продуктовые решения

### 5.1. Standalone, а не плагин-зависимость от LAMMPS

TDMD — отдельная программа. Она не обязана жить как пакет внутри LAMMPS.

### 5.2. LAMMPS как внешний scientific oracle

LAMMPS используется для:

- `run 0` force/energy/virial compare;
- NVE drift compare;
- observables compare;
- benchmark baseline.

### 5.3. Scientist-first UX

TDMD изначально проектируется как продукт для учёного:

- validate/explain/compare/repro-bundle;
- recipes;
- понятные ошибки;
- воспроизводимость по умолчанию.

### 5.4. AI-agent-first development process

Проект изначально проектируется так, чтобы его можно было развивать через Codex/агентов:

- master-spec;
- module specs;
- execution pack;
- strict testing discipline;
- no hidden architectural drift.

---

## 6. Режимы исполнения и численная архитектура

### 6.1. Один core, две оси конфигурации

В TDMD теперь официально зафиксирована следующая модель:

1. **BuildFlavor** — compile-time вариант сборки, фиксирующий numerical semantics;
2. **ExecProfile** — runtime профиль исполнения, задающий policy-поведение движка.

### 6.2. Что задаёт BuildFlavor

BuildFlavor задаёт:

- тип хранения состояния атомов;
- тип force kernels;
- тип accumulation;
- тип reductions;
- допустимость atomics;
- deterministic numeric traits;
- compile-time specialization paths.

### 6.3. Что задаёт ExecProfile

ExecProfile задаёт:

- scheduler policy;
- comm policy;
- reorder policy;
- overlap policy;
- validation strictness;
- diagnostics behavior.

### 6.4. Канонические BuildFlavor v1

- `Fp64ReferenceBuild`
- `Fp64ProductionBuild`
- `MixedFastBuild`
- `Fp32ExperimentalBuild`

### 6.5. Канонические ExecProfile v1

- `Reference`
- `Production`
- `FastExperimental`

### 6.6. Матрица совместимости

| BuildFlavor | Allowed ExecProfile |
|---|---|
| `Fp64ReferenceBuild` | `Reference`, `Production` |
| `Fp64ProductionBuild` | `Production`, optionally `Reference` |
| `MixedFastBuild` | `FastExperimental`, optionally validated `Production` |
| `Fp32ExperimentalBuild` | `FastExperimental` only |

### 6.7. Ключевой вывод

Fast mode — это не отдельный движок, а другой runtime profile поверх общего core и конкретного build flavor.

### 6.8. Human-facing labels

В scientist-facing UX пользователь видит:

- `reference`
- `production`
- `fast-experimental`

Но в reproducibility bundle обязательно фиксируется реальный `BuildFlavor`.

---

## 7. Методология разработки

### 7.1. Обязательная методика

Проект ведётся по схеме:

> **Spec-Driven TDD + Differential Testing + Determinism Checks + Performance Gates**

### 7.2. Порядок зрелости любой фичи

1. Спецификация
2. CPU reference path
3. Deterministic validation
4. GPU path
5. Reproducible profile
6. Fast profile

### 7.3. Главный запрет

Нельзя жертвовать deterministic/reference path ради throughput.

---

## 8. Архитектура верхнего уровня

### 8.1. Модули

```text
io/
state/
neighbor/
potentials/
integrator/
scheduler/
comm/
telemetry/
interop/
cli/
analysis/
```

### 8.2. Дополнительные стратегические спецификации

```text
docs/specs/runtime/SPEC.md
docs/specs/policies/SPEC.md
docs/specs/testing/SPEC.md
```

### 8.3. Главная структурная идея

- `runtime` orchestrates;
- `scheduler` decides legality;
- `neighbor` manages locality;
- `potentials` compute physics;
- `integrator` advances state;
- `comm` moves data;
- `telemetry` observes;
- `cli` explains and runs workflows.

---

## 9. Архитектура под целевую нишу many-body

### 9.1. New first-class prioritization

Подсистема `potentials` теперь проектируется не как “pair first, ML later”, а как:

1. pair reference path (для простоты и верификации);
2. **local classical many-body first-class**;
3. reactive many-body later;
4. local ML later;
5. hybrid classical+ML ready.

### 9.2. First-class families

Первая проектная волна:

- `Morse` как reference pair path
- `EAM`
- `EAM/FS`
- `MEAM`
- `MEAM-like angular`

### 9.3. Why MEAM becomes strategic earlier

MEAM и подобные ему модели должны быть подняты раньше, чем ReaxFF и раньше, чем heavy ML integration, потому что именно они лучше всего выражают целевую нишу TDMD.

### 9.4. ReaxFF placement

`ReaxFF` входит в проект как **phase-2 strategic track**, а не как обязательная часть первой production-волны.

Причина:

- высокая практическая ценность;
- но заметно более сложная архитектурная интеграция.

### 9.5. Local ML placement

ML остаётся важным расширением, но не ядром идентичности проекта.

---

## 10. Runtime orchestration

### 10.1. SimulationEngine

`SimulationEngine` — единственная точка orchestration.

Он отвечает за:

- lifecycle;
- policy application;
- module coordination;
- run/finalize/shutdown;
- invariant enforcement.

### 10.2. Lifecycle

1. ParseConfig
2. ResolvePolicies
3. BootstrapState
4. InitializeExecution
5. RunLoop
6. FinalizeOutputs
7. Shutdown

### 10.3. Engine state machine

- `Created`
- `Configured`
- `PoliciesResolved`
- `StateBootstrapped`
- `ExecutionInitialized`
- `Running`
- `Finalized`
- `Shutdown`
- `Failed`

### 10.4. Canonical run loop

```text
while not finished:
    begin_iteration()
    refresh_state_tracking()
    refresh_certificates()
    select_ready_tasks()
    execute_ready_tasks()
    progress_comm()
    commit_completed_tasks()
    maybe_rebuild_neighbors()
    emit_outputs_if_needed()
    update_progress()
    watchdog_check()
```

---

## 11. Scheduler and safety certificates

### 11.1. Главная обязанность scheduler

Scheduler отвечает за **законное продвижение зон по времени**.

### 11.2. ZoneTask

Единица работы:

`(zone_id, time_level, local_state_version, dependency_mask, certificate_version)`

### 11.3. SafetyCertificate

Сертификат — формальная проверка, что зону можно продвинуть без нарушения причинной изоляции.

### 11.4. Особая важность для many-body

Для many-body моделей scheduler и certificate layer критичнее, чем в простых pairwise системах, потому что локальная вычислительная область и цена ошибки выше.

---

## 12. Neighbor, migration, stable reorder

### 12.1. Три разные операции

В TDMD нужно жёстко различать:

- neighbor rebuild
- migration
- stable reorder

### 12.2. Почему это особенно важно для many-body

Для many-body моделей неправильная работа этого слоя разрушает не только pair coverage, но и более сложные локальные структуры взаимодействия.

### 12.3. Deterministic importance

Stable reorder и versioning здесь особенно важны для `Reference` профиля.

---

## 13. Потенциалы: новая приоритетная архитектура

### 13.1. Классы потенциалов

1. `PairPotential`
2. `ManyBodyLocalPotential`
3. `ReactiveManyBodyPotential`
4. `LocalDescriptorMlPotential`
5. `HybridCorrectionPotential` (future)

### 13.2. Wave 1

- `Morse`
- `EAM`
- `EAM/FS`
- `MEAM`

### 13.3. Wave 2

- `MEAM-like angular variants`
- generic local high-order models
- early reactive abstractions

### 13.4. Wave 3

- `ReaxFF`-class
- charge-equilibration-aware reactive execution

### 13.5. Wave 4

- local ML potentials
- hybrid classical + ML correction

### 13.6. Strategic rule

Потенциальный слой TDMD должен быть **classical-many-body-first**, а ML — расширением.

---

## 14. Integrator, ensembles, timestep policy

### 14.1. v1 baseline

- `VelocityVerletIntegrator`
- `NVEEnsembleController`

### 14.2. v1.5 scientific usability

- `NVTEnsembleController`
- `NPTEnsembleController`

### 14.3. Future hooks

- long-range service
- `OuterMtsController`

### 14.4. Dt policy

На старте:

- один глобальный `dt`;
- safety-based validation;
- no hidden adaptive local dt.

---

## 15. CLI, UX, scientist workflow

### 15.1. Scientist mode first

TDMD обязан быть scientist-usable.

### 15.2. Минимальные команды

- `run`
- `validate`
- `explain`
- `compare --with lammps`
- `resume`
- `repro-bundle`

### 15.3. Why compare is first-class

Поскольку TDMD — специализированный движок с сильным архитектурным отличием, compare with reference должен быть встроенной научной функцией, а не внешним скриптом.

---

## 16. Reproducibility and validation

### 16.1. Три уровня гарантий

1. Bitwise determinism
2. Layout-invariant determinism
3. Scientific reproducibility

### 16.2. What matters most by profile

- `Reference` — максимум строгих гарантий
- `Production` — научная воспроизводимость
- `FastExperimental` — observables + performance

### 16.3. Per-build validation

Каждый `BuildFlavor` должен иметь собственные:

- thresholds
- drift expectations
- performance baselines

---

## 17. Benchmark governance: переписано под новую цель

### 17.1. Новый benchmark hierarchy

#### Tier 0 — analytic
- two-body Morse
- tiny deterministic checks

#### Tier 1 — pair/local correctness
- small crystal
- rebuild/reorder boundaries

#### Tier 2 — alloy EAM baseline
- multi-component alloy
- CPU/GPU run0 compare

#### Tier 3 — **MEAM / local many-body benchmark**
- angular effects
- many-body halo sensitivity
- TD advantage target zone

#### Tier 4 — reactive benchmark
- later phase
- ReaxFF-class or simplified reactive workload

#### Tier 5 — local ML benchmark
- descriptor-based local model

#### Tier 6 — hybrid workflow benchmark
- classical + correction

### 17.2. Benchmark strategy rule

Нельзя строить проектовую идентичность только вокруг EAM benchmark и ML benchmark.  
Обязателен **many-body classical benchmark tier** как центральный scientific proof-of-value.

---

## 18. Roadmap: переписанный порядок приоритетов

### 18.1. New milestone priorities

#### M0 — Bootstrap
repo, CI, policies, runtime skeleton

#### M1 — Config/validate/explain shell
scientist-facing shell without physics

#### M2 — State + neighbor + reorder
core locality layer

#### M3 — NVE + Morse CPU reference
first physical baseline

#### M4 — compare with LAMMPS / reference
scientific oracle path

#### M5 — EAM CPU reference
metal baseline

#### M6 — Deterministic TD scheduler
core TD legality

#### M7 — Full CPU deterministic TD pipeline
restart, repro bundle, compare path

#### M8 — Single GPU Morse + neighbor
GPU baseline

#### M9 — GPU EAM
production metal path

#### M10 — **MEAM / local many-body CPU reference**
new strategic milestone

#### M11 — **MEAM / local many-body GPU path**
proof of target niche advantage

#### M12 — Multi-GPU time×space
scaled execution

#### M13 — Reproducible Production profile
scientific production mode

#### M14 — FastExperimental profile
throughput mode

#### M15 — NVT/NPT geometry-changing workflows
scientific usability

#### M16 — Reactive track bootstrap
Reax-like infrastructure, not yet full production

#### M17 — Local ML track
descriptor/model local ML integration

#### M18 — Hybrid classical + ML correction
long-term differentiator

### 18.2. New strategic target

Ближайшая большая цель теперь формулируется так:

> **validated-alpha для local classical many-body TDMD**

То есть:
- standalone engine;
- deterministic TD;
- EAM baseline;
- MEAM/local-many-body baseline;
- compare with LAMMPS;
- single-GPU performance baseline;
- reproducibility bundle.

---

## 19. AI-agent development model

### 19.1. Главная дисциплина

Агенты не должны оптимизировать «что быстрее написать». Они должны оптимизировать **непротиворечивое продвижение по dependency-aware roadmap**.

### 19.2. Канонические роли

- Architect / Spec Steward
- Core Runtime Engineer
- Scheduler / Determinism Engineer
- Neighbor / Migration Engineer
- Physics Engineer
- GPU / Performance Engineer
- Validation / Reference Engineer
- Scientist UX Engineer

### 19.3. Главный запрет для агентов

Нельзя:
- делать hidden second engine for fast mode;
- делать ML-first fork архитектуры;
- встраивать runtime numerical switches вместо build flavors;
- начинать Reax/ML track до стабилизации classical-many-body core.

---

## 20. File-ready SPEC map

### 20.1. Module specs

```text
docs/specs/io/SPEC.md
docs/specs/state/SPEC.md
docs/specs/neighbor/SPEC.md
docs/specs/potentials/SPEC.md
docs/specs/integrator/SPEC.md
docs/specs/scheduler/SPEC.md
docs/specs/comm/SPEC.md
docs/specs/telemetry/SPEC.md
docs/specs/cli/SPEC.md
docs/specs/interop/SPEC.md
docs/specs/testing/SPEC.md
docs/specs/runtime/SPEC.md
docs/specs/policies/SPEC.md
```

### 20.2. Additional strategy docs to add/refresh

Теперь рекомендуется добавить или обновить ещё:

```text
docs/specs/positioning/SPEC.md
docs/specs/benchmarks/SPEC.md
```

Потому что product positioning и benchmark strategy стали частью differentiator-а проекта, а не только appendix.

---

## 21. Policies SPEC: новая норма

### 21.1. Policies now mean runtime semantics only

`policies/SPEC.md` должен отражать:

- `ExecProfile`
- scheduler/comm/reorder/overlap/validation policies

Но больше не должен трактовать numeric precision как свободный runtime switch.

### 21.2. Numeric semantics are build-level

Численная семантика должна ссылаться на `BuildFlavor` и `NumericConfig`.

---

## 22. Runtime SPEC: новая норма

### 22.1. Runtime sees BuildFlavorInfo

`runtime/SPEC.md` должен считать, что бинарник сообщает о себе через:

- `build_flavor`
- `numeric_config_id`

### 22.2. Runtime only resolves allowed profile

Runtime резолвит `ExecProfile`, проверяет compatibility matrix и применяет policy bundle.

---

## 23. Potentials SPEC: новая норма

### 23.1. Potentials must be reprioritized

`potentials/SPEC.md` должен быть переписан как:

- pair reference path;
- local classical many-body first-class path;
- reactive later;
- local ML later.

### 23.2. MEAM becomes strategic milestone

`MEAM` должен перейти из разряда optional future в обязательную стратегическую волну после EAM.

---

## 24. Testing SPEC: новая норма

### 24.1. Benchmark set must prove target niche

`testing/SPEC.md` и benchmark governance должны отражать, что проектовая ценность должна быть доказана на **local classical many-body workloads**, а не только на EAM и ML.

---

## 25. CLI/UX SPEC: новая норма

### 25.1. User messaging must reflect new identity

CLI, docs и recipes должны говорить не «ML-first», а:

- classical-many-body-first
- local-ML-compatible
- hybrid-ready

---

## 26. Immediate rewrite instructions for the project package

Следующие файлы теперь требуют синхронного обновления под новую стратегию:

1. `tdmd_engineering_spec.md`
2. `docs/specs/policies/SPEC.md`
3. `docs/specs/runtime/SPEC.md`
4. `docs/specs/potentials/SPEC.md`
5. `docs/specs/testing/SPEC.md`
6. `docs/specs/cli/SPEC.md`
7. `docs/specs/interop/SPEC.md`
8. `docs/specs/benchmarks/SPEC.md` (новый)
9. `docs/specs/positioning/SPEC.md` (новый)
10. `docs/execution/M0_M3_EXECUTION_PACK.md` — нужен update roadmap wording

---

## 27. Practical next actions

### 27.1. What should happen next

Правильный следующий шаг после этого redesign:

1. обновить zip-пакет проекта;
2. синхронно переработать модульные SPEC под новую стратегию;
3. обновить execution pack;
4. обновить Codex prompts, чтобы они отражали:
   - build flavor compile-time
   - exec profile runtime
   - classical-many-body-first positioning.

### 27.2. What not to do now

Не нужно сейчас:

- расширять master-spec ещё сильнее без синхронизации файлового пакета;
- начинать ML-first execution pack;
- проектировать ReaxFF production path до завершения local classical many-body baseline.

---

## 28. Canonical one-paragraph positioning

Для сайта, README, grant, presentation, Codex context и engineer onboarding нужно считать канонической следующую короткую формулировку:

> **TDMD is a standalone time-decomposition molecular dynamics engine focused primarily on expensive local many-body interactions — especially classical high-order models such as EAM/MEAM-class potentials — where time decomposition can outperform standard spatial decomposition by reducing communication and preserving local step consistency. Local ML potentials are supported as an extension, not as the sole identity of the system; the long-term direction is hybrid classical+ML workflows.**

---

## 29. Change log

### v1.0-reframed
- проект перепозиционирован как classical-many-body-first TD engine;
- compile-time `BuildFlavor` + runtime `ExecProfile` закреплены как базовая архитектурная модель;
- MEAM/local many-body moved into strategic core;
- ReaxFF moved to later specialized track;
- ML retained as important extension, not primary identity;
- benchmark hierarchy rewritten to prove value on local many-body workloads;
- roadmap reordered accordingly.

