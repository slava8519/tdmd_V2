# TDMD Engineering Spec

Версия: 0.1-draft\
Статус: рабочий инженерный документ\
Назначение: главный живой документ проекта TDMD; все дальнейшие доработки по методике, архитектуре, режимам исполнения, CI, TDD и roadmap вносятся сюда.

---

## 1. Цель документа

Документ фиксирует инженерную спецификацию самостоятельной программы молекулярной динамики **TDMD**:

- с **декомпозицией по времени (Time Decomposition, TD)** как ключевой идеей исполнения;
- с ориентацией на **сплавы и металлические системы** на первом этапе;
- с архитектурой, которая позволяет расширяться на другие среды;
- с опорой на лучшие инженерные решения из **LAMMPS** как эталона физической корректности, форматов и части UX-паттернов;
- с разработкой, выполняемой **преимущественно ИИ-агентами**;
- с обязательной методологией **Spec-Driven TDD + Differential Testing + Performance Gates**.

Этот документ является:

1. спецификацией архитектуры;
2. спецификацией режимов исполнения;
3. спецификацией процесса разработки;
4. основой для постановки задач ИИ-агентам;
5. основой для CI, merge gates и критериев готовности.

---

## 2. Высокоуровневая концепция TDMD

### 2.1. Базовая идея

TDMD — это самостоятельный MD-движок, в котором short-range часть расчёта организована как **pipeline по времени и зонам**, а не только как классическая пространственная декомпозиция.

Ключевой принцип:

> если две области модели удалены друг от друга больше, чем на эффективный диаметр взаимодействия с учётом буфера безопасности, их можно продвигать по времени независимо хотя бы на локальном окне интегрирования.

### 2.2. Современная интерпретация

TDMD **не** реализует наивную схему «процессор P1 считает шаг h, P2 считает h+1».

Вместо этого используется:

- **causality-aware scheduler**;
- **граф зависимостей зон**;
- **safety certificate** для каждой зоны;
- гибрид **time × space decomposition**.

То есть:

- по времени — temporal frontier / temporal stages;
- по пространству — domain decomposition и halos;
- внутри GPU — microtiles / batched kernels.

### 2.3. Физическая область применимости v1

Первая production-версия должна покрывать:

- классическую атомистическую MD;
- short-range pair potentials;
- short-range many-body potentials типа EAM;
- локальные ML-потенциалы с конечным cutoff.

Long-range часть проектируется отдельно:

- short-range: **TD pipeline**;
- long-range: **split partition / PME service / outer MTS level**.

---

## 3. Продуктовая стратегия

### 3.1. Что такое TDMD

TDMD — это **standalone executable**, а не пакет внутри LAMMPS.

Однако система должна быть **LAMMPS-friendly**:

- читать совместимые input/data;
- использовать LAMMPS как эталон верификации;
- сравнивать физику и производительность с LAMMPS;
- поддерживать удобный differential workflow.

### 3.2. Что заимствуем у LAMMPS

Мы не копируем LAMMPS как архитектуру «один-в-один», а заимствуем:

- разделение понятий state / integrator / pair style / compute / fix-like operations;
- дисциплину neighbor lists и skin;
- формат timing breakdown;
- формат input/data-interop;
- библиотечный режим для оракульной сверки;
- экосистему ML-потенциалов как ориентир совместимости.

### 3.3. Основной инженерный принцип

**LAMMPS — внешний физический оракул.**\
**TDMD — основной production runtime.**

---

## 4. Методология разработки

### 4.1. Базовый подход

Для проекта вводится обязательная методика:

## Spec-Driven TDD + Differential Testing + Performance Gates

Это означает:

1. сначала пишется спецификация модуля;
2. затем набор контрактных тестов;
3. затем минимальная реализация;
4. затем differential-сверка с LAMMPS/reference;
5. затем perf gate;
6. только после этого разрешён merge.

### 4.2. Почему обычный TDD недостаточен

Для агентной разработки недостаточно просто писать unit-тесты.

Нужны одновременно:

- **Spec tests** — проверка соответствия формальной спецификации;
- **Property tests** — проверка инвариантов;
- **Differential tests** — сравнение с LAMMPS/reference;
- **Determinism tests** — проверка воспроизводимости;
- **Performance tests** — чтобы ИИ-агенты не разрушали throughput.

### 4.3. Обязательные инженерные артефакты для каждого модуля

Каждый модуль обязан иметь:

- `README.md` с границами ответственности;
- `SPEC.md` с контрактами;
- `TESTPLAN.md`;
- unit/property tests;
- telemetry hooks;
- пример использования;
- список известных ограничений.

---

## 5. Режимы исполнения: один core, разные policy-режимы

### 5.0. Базовый архитектурный принцип режимов

Это обязательное решение проекта:

> **Deterministic / Reproducible / Fast не являются тремя разными движками.**\
> Это один и тот же вычислительный core TDMD, поверх которого накладываются разные policy-слои.

Иными словами:

- один и тот же `SimulationEngine`;
- один и тот же набор модулей `state/neighbor/potentials/integrator/scheduler/comm/telemetry`;
- одна и та же физика потенциалов;
- один и тот же путь верификации;
- одни и те же структуры данных;
- разные только **ограничения, разрешённые оптимизации и guarantees policy**.

### 5.0.1. Что меняется между режимами

Между режимами могут меняться:

- scheduler policy;
- communication policy;
- reduction policy;
- reorder policy;
- precision policy;
- overlap policy;
- task stealing policy;
- buffering/adaptation policy.

### 5.0.2. Что не должно расходиться между режимами

Между режимами не должны fork-аться:

- математическая форма потенциала;
- определение физических величин;
- файловые форматы состояния;
- интерфейсы основных модулей;
- acceptance-тесты высокого уровня.

### 5.0.3. Следствие для реализации

Fast mode должен проектироваться как:

> **детерминистический core + разрешённые ускоряющие политики и precision-послабления**.

Это не означает, что Fast mode обязан исполняться буквально тем же порядком операций, что Deterministic.\
Это означает, что:

- существует единый канонический reference path;
- любые fast-path отличия локализованы, флагуемы и отключаемы;
- любой fast-path обязан сравниваться против reference path.

### 5.0.4. Следствие для CI

Нельзя принимать изменения, которые создают скрытый «второй движок» для Fast mode.\
Любая fast-оптимизация должна оформляться как policy switch поверх общего core.

## 5. Режимы исполнения: Deterministic vs Fast

### 5.1. Принципиальная позиция

TDMD должен поддерживать **три режима исполнения**, а не два.

Это обязательное решение проекта.

### 5.2. Режим 1 — Deterministic Reference

Назначение:

- разработка;
- CI;
- отладка;
- воспроизведение багов;
- публикационные контрольные прогоны;
- эталон для Fast/Repro modes.

Свойства:

- фиксированный порядок обхода зон;
- фиксированное расписание scheduler;
- фиксированный порядок построения neighbor lists;
- фиксированные деревья редукций;
- запрет на недетерминированные atomics для накопления сил;
- стабильная сортировка атомов после migration/reorder;
- фиксированный communication order;
- запрещён opportunistic task stealing;
- фиксированная precision policy;
- запрещены эвристики, меняющие order of operations.

Гарантия:

- одинаковый input + одинаковый supported hardware class + одинаковый toolchain + одинаковая конфигурация режима => один и тот же результат;
- целевая амбиция: одинаковый результат при разных layouts ranks/GPU, где это архитектурно достижимо.

Минусы:

- медленнее;
- хуже overlap;
- ограничивает scheduler и GPU-оптимизации.

### 5.3. Режим 2 — Reproducible Production

Назначение:

- основной научный режим;
- статьи;
- воспроизводимые исследования;
- контрольные production-расчёты.

Свойства:

- допускает часть оптимизаций;
- сохраняет контролируемую воспроизводимость;
- deterministic scheduling может быть ослаблен локально, если это не меняет результат в рамках целевой guarantees policy;
- редукции могут быть оптимизированы, но не должны давать runaway divergence на коротких окнах.

Гарантия:

- одинаковые executable, runtime config, hardware class, libraries, input и seed дают один и тот же результат;
- между разными раскладками ranks/GPU гарантируется либо идентичность, либо строго ограниченное и тестируемое отличие, зафиксированное в acceptance criteria.

Это **основной scientific default**.

### 5.4. Режим 3 — Fast Production

Назначение:

- длинные production-кампании;
- высокопроизводительный скрининг;
- ensemble runs;
- крупные GPU-суперкластерные расчёты.

Свойства:

- aggressive overlap;
- dynamic load balancing;
- возможны недетерминированные GPU reductions;
- mixed precision разрешена;
- task stealing разрешён;
- динамические эвристики scheduler разрешены;
- допускаются fast paths communication backend.

Гарантия:

- сохраняется физическая корректность по acceptance metrics;
- гарантируется воспроизводимость observables/statistics в пределах допустимых scientific thresholds;
- битовая идентичность траектории не обещается.

Это **основной throughput default**.

### 5.5. Политика использования режимов

| Сценарий                             | Режим                   |
| ------------------------------------ | ----------------------- |
| Разработка ядра                      | Deterministic Reference |
| CI и regression                      | Deterministic Reference |
| Отладка численных ошибок             | Deterministic Reference |
| Верификация against LAMMPS           | Deterministic Reference |
| Контрольные научные расчёты          | Reproducible Production |
| Публикационные воспроизводимые кейсы | Reproducible Production |
| Массовые расчёты и throughput        | Fast Production         |

### 5.6. Политика качества между режимами

Любой Fast mode обязан проходить:

1. короткий shadow-run в Deterministic mode;
2. differential validation на representative benchmark;
3. NVE drift gate;
4. observables-consistency gate.

---

## 6. Архитектурная схема системы

### 6.1. Модули верхнего уровня

```text
io/
state/
neighbor/
potentials/
integrator/
scheduler/
comm/
telemetry/
analysis/
interop/
cli/
```

### 6.2. Основной поток данных

```text
Input -> State init -> Neighbor init -> Scheduler seed
      -> Force compute -> Integrate -> Comm/Frontier progress
      -> Telemetry -> Output -> Next iteration
```

### 6.3. Ключевой архитектурный принцип

**Ни один модуль, кроме scheduler, не владеет глобальной политикой продвижения по времени.**

Это обязательное правило.

---

## 7. Конкретная структура классов C++/CUDA

Ниже описан базовый целевой каркас. Это не окончательный ABI, но именно от него надо стартовать.

### 7.1. Базовые типы и контексты

```cpp
namespace tdmd {

enum class ExecMode {
    DeterministicReference,
    ReproducibleProduction,
    FastProduction
};

enum class PrecisionMode {
    FP64,
    Mixed,
    FP32
};

enum class DeviceBackend {
    CPU,
    CUDA
};

struct BuildInfo {
    std::string git_sha;
    std::string compiler_id;
    std::string compiler_version;
    std::string cuda_version;
    std::string build_type;
};

struct RuntimeConfig {
    ExecMode exec_mode;
    PrecisionMode precision_mode;
    DeviceBackend backend;
    bool strict_determinism;
    bool gpu_aware_mpi;
    bool enable_nvtx;
    bool enable_task_stealing;
    bool enable_mixed_precision_force;
};

struct PrecisionPolicy {
    PrecisionMode state_precision;
    PrecisionMode force_compute_precision;
    PrecisionMode accumulation_precision;
    PrecisionMode reduction_precision;
    bool deterministic_reduction;
};

struct ReproContext {
    uint64_t global_seed;
    uint64_t run_id;
    RuntimeConfig runtime;
    BuildInfo build;
};

}
```

### 7.2. Состояние атомов

```cpp
namespace tdmd {

using AtomId = uint64_t;
using SpeciesId = uint32_t;
using ZoneId = uint32_t;
using CellId = uint32_t;

struct AtomSoA {
    std::vector<AtomId> id;
    std::vector<SpeciesId> type;

    std::vector<double> x, y, z;
    std::vector<double> vx, vy, vz;
    std::vector<double> fx, fy, fz;

    std::vector<int32_t> image_x, image_y, image_z;
    std::vector<uint32_t> flags;
};

struct DeviceAtomSoA {
    AtomId* id;
    SpeciesId* type;
    double* x; double* y; double* z;
    double* vx; double* vy; double* vz;
    double* fx; double* fy; double* fz;
    int32_t* image_x; int32_t* image_y; int32_t* image_z;
    uint32_t* flags;
    size_t n;
};

}
```

### 7.3. Геометрия и box

```cpp
namespace tdmd {

struct Box {
    double xlo, xhi;
    double ylo, yhi;
    double zlo, zhi;
    bool periodic_x;
    bool periodic_y;
    bool periodic_z;
};

struct DomainDecomposition {
    int rank;
    int nranks;
    Box local_box;
    Box halo_box;
    std::vector<int> spatial_neighbors;
};

}
```

### 7.4. Зоны и scheduler entities

```cpp
namespace tdmd {

enum class ZoneState {
    Empty,
    ResidentPrev,
    Ready,
    Computing,
    Completed,
    PackedForSend,
    InFlight,
    Committed
};

struct SafetyCertificate {
    bool safe;
    double vmax;
    double dt_max;
    double skin_remaining;
    double buffer_width;
    double frontier_margin;
    uint64_t version;
};

struct ZoneMeta {
    ZoneId zone_id;
    ZoneState state;
    Box bbox;
    Box halo_bbox;
    uint64_t local_version;
    uint64_t time_level;
    SafetyCertificate cert;
    size_t atom_begin;
    size_t atom_end;
};

struct ZoneTask {
    ZoneId zone_id;
    uint64_t time_level;
    uint64_t local_state_version;
    uint64_t dep_mask;
    SafetyCertificate cert;
};

}
```

### 7.5. Cell grid и neighbor lists

```cpp
namespace tdmd {

struct CellGrid {
    double cell_x;
    double cell_y;
    double cell_z;
    uint32_t nx, ny, nz;
    std::vector<uint32_t> cell_offsets;
    std::vector<AtomId> cell_atoms;
};

struct NeighborList {
    std::vector<uint64_t> page_offsets;
    std::vector<AtomId> neigh_ids;
    std::vector<float> neigh_r2;
    double cutoff;
    double skin;
    uint64_t build_version;
};

struct DeviceNeighborList {
    uint64_t* page_offsets;
    AtomId* neigh_ids;
    float* neigh_r2;
    size_t nnz;
    double cutoff;
    double skin;
    uint64_t build_version;
};

}
```

### 7.6. Потенциалы: общий интерфейс

```cpp
namespace tdmd {

struct ComputeMask {
    bool force;
    bool energy;
    bool virial;
};

struct ForceRequest {
    const DeviceAtomSoA* atoms;
    const DeviceNeighborList* neigh;
    const Box* box;
    const ZoneMeta* zone;
    ComputeMask mask;
    PrecisionMode precision;
};

struct ForceResult {
    double potential_energy;
    double virial[6];
};

class PotentialModel {
public:
    virtual ~PotentialModel() = default;
    virtual std::string name() const = 0;
    virtual double cutoff() const = 0;
    virtual bool is_local() const = 0;
    virtual void compute(const ForceRequest&, ForceResult&) = 0;
};

class MorsePotential final : public PotentialModel { /* ... */ };
class EamAlloyPotential final : public PotentialModel { /* ... */ };
class EamFsPotential final : public PotentialModel { /* ... */ };
class SnapPotential final : public PotentialModel { /* ... */ };
class MliapPotential final : public PotentialModel { /* ... */ };
class PacePotential final : public PotentialModel { /* ... */ };

}
```

### 7.7. Интегратор

```cpp
namespace tdmd {

class Integrator {
public:
    virtual ~Integrator() = default;
    virtual std::string name() const = 0;
    virtual void pre_force(DeviceAtomSoA&, const ZoneMeta&, double dt) = 0;
    virtual void post_force(DeviceAtomSoA&, const ZoneMeta&, double dt) = 0;
};

class VelocityVerletIntegrator final : public Integrator {
public:
    std::string name() const override { return "velocity-verlet"; }
    void pre_force(DeviceAtomSoA&, const ZoneMeta&, double dt) override;
    void post_force(DeviceAtomSoA&, const ZoneMeta&, double dt) override;
};

}
```

### 7.8. Scheduler

```cpp
namespace tdmd {

class TdScheduler {
public:
    virtual ~TdScheduler() = default;

    virtual void initialize() = 0;
    virtual void refresh_certificates() = 0;
    virtual std::vector<ZoneTask> select_ready_tasks() = 0;
    virtual void mark_computing(const ZoneTask&) = 0;
    virtual void mark_completed(const ZoneTask&) = 0;
    virtual void commit_completed() = 0;
    virtual bool finished() const = 0;
};

class CausalWavefrontScheduler final : public TdScheduler {
    // zone DAG
    // dependency tracking
    // deterministic / reproducible / fast policy switching
};

}
```

### 7.9. Коммуникации

```cpp
namespace tdmd {

struct TemporalPacket {
    ZoneId zone_id;
    uint64_t time_level;
    uint64_t version;
    std::vector<uint8_t> payload;
};

class CommBackend {
public:
    virtual ~CommBackend() = default;
    virtual std::string name() const = 0;
    virtual void send_spatial_halo(/*...*/) = 0;
    virtual void send_temporal_packet(const TemporalPacket&) = 0;
    virtual void progress() = 0;
};

class MpiHostStagingBackend final : public CommBackend { /* ... */ };
class GpuAwareMpiBackend final : public CommBackend { /* ... */ };
class NcclBackend final : public CommBackend { /* ... */ };
class NvshmemBackend final : public CommBackend { /* ... */ };

}
```

### 7.10. Телеметрия

```cpp
namespace tdmd {

struct TimingBreakdown {
    double t_pair = 0.0;
    double t_neigh = 0.0;
    double t_comm = 0.0;
    double t_integrate = 0.0;
    double t_scheduler = 0.0;
    double t_output = 0.0;
    double t_pack = 0.0;
    double t_unpack = 0.0;
};

struct PipelineStats {
    uint64_t zones_ready = 0;
    uint64_t zones_inflight = 0;
    uint64_t zones_committed = 0;
    uint64_t certificate_failures = 0;
    uint64_t neighbor_rebuilds = 0;
};

struct StepSummary {
    uint64_t step = 0;
    double time_ps = 0.0;
    double temp = 0.0;
    double pe = 0.0;
    double ke = 0.0;
    double etotal = 0.0;
    double press = 0.0;
    double vol = 0.0;
    double density = 0.0;
    TimingBreakdown timing;
    PipelineStats pipeline;
};

class TelemetrySink {
public:
    virtual ~TelemetrySink() = default;
    virtual void write_step(const StepSummary&) = 0;
};

}
```

### 7.11. Оркестратор симуляции

```cpp
namespace tdmd {

class SimulationEngine {
public:
    void initialize();
    void run();
    void finalize();

private:
    ReproContext repro_;
    AtomSoA host_atoms_;
    DeviceAtomSoA device_atoms_;
    CellGrid cells_;
    NeighborList neigh_;
    std::unique_ptr<PotentialModel> potential_;
    std::unique_ptr<Integrator> integrator_;
    std::unique_ptr<TdScheduler> scheduler_;
    std::unique_ptr<CommBackend> comm_;
    std::unique_ptr<TelemetrySink> telemetry_;
};

}
```

---

## 8. Интерфейсы модулей

### 8.1. Правило владения данными

- `state/` владеет состоянием частиц и box;
- `neighbor/` владеет списками соседей и cell bins;
- `potentials/` не владеет атомами;
- `integrator/` не владеет scheduler;
- `scheduler/` не владеет физикой потенциалов;
- `comm/` не владеет доменной логикой;
- `telemetry/` только наблюдает.

### 8.2. Жёсткий контракт между scheduler и физикой

Scheduler **не** вычисляет силы.\
Potential **не** решает, можно ли продвинуть зону.

Это строгое разделение обязательно.

### 8.3. Контракт между deterministic и fast paths

Fast path не должен внедрять скрытые изменения физики.\
Все fast optimizations должны быть:

- локализованы;
- флагуемы;
- отключаемы;
- покрыты A/B тестами.

---

## 9. GPU/CUDA архитектура

### 9.1. Базовые принципы

- SoA layout;
- GPU-resident state;
- multiple CUDA streams;
- async overlap compute/comm;
- NVTX ranges обязательны в debug/perf builds;
- pinned host memory только как fallback.

### 9.2. Потоки исполнения

Минимум три stream:

- `stream_compute` — force/integrator kernels;
- `stream_comm` — pack/unpack + communication;
- `stream_aux` — reorder, certificate kernels, telemetry.

### 9.3. Стратегия памяти

- постоянные массивы атомов живут на device;
- neighbor lists живут на device;
- pack/unpack buffers — device first;
- host mirrors используются только для I/O, fallback comm и отладки.

### 9.4. CUDA kernel categories

- cell binning kernels;
- neighbor build kernels;
- Morse force kernels;
- EAM density kernels;
- EAM embedding/force kernels;
- integrate half-kick/drift/half-kick kernels;
- zone certificate kernels;
- pack/unpack kernels;
- stable reorder kernels.

---

## 9A. Precision architecture

### 9A.1. Принцип

Политика точности должна быть явно встроена в архитектуру TDMD, а не задаваться неявно через backend-specific детали.

Это обязательное решение проекта:

> **Precision policy — такая же часть runtime policy, как scheduler policy или communication policy.**

### 9A.2. Уровни точности

В TDMD вводятся следующие уровни точности:

1. `FP64_STRICT`
2. `FP64_STATE_FP64_FORCE`
3. `MIXED_FORCE_ACCUM_DOUBLE`
4. `FP32_EXPERIMENTAL`

### 9A.3. Определения

#### FP64\_STRICT

- состояние атомов в FP64;
- вычисление сил в FP64;
- накопление сил/энергии/вириала в FP64;
- редукции в фиксированном FP64 порядке.

Это базовый режим для Deterministic Reference.

#### FP64\_STATE\_FP64\_FORCE

Фактически production-safe double path:

- state FP64;
- force compute FP64;
- reductions FP64;
- допускаются некоторые runtime optimizations, не меняющие declared guarantees.

#### MIXED\_FORCE\_ACCUM\_DOUBLE

- состояние атомов хранится в FP64;
- pairwise / local interaction kernels могут считаться в FP32;
- накопление сил, энергий и вириала — в FP64;
- глобальные редукции по возможности FP64.

Это основной кандидат на Fast Production и, возможно, часть Reproducible Production после полной валидации.

#### FP32\_EXPERIMENTAL

- вычисления и часть state могут идти в FP32;
- режим допускается только как экспериментальный/performance mode;
- запрещён как default scientific mode.

### 9A.4. Рекомендуемая матрица precision по режимам

| Exec mode               | Default precision           | Allowed precision                                                                   |
| ----------------------- | --------------------------- | ----------------------------------------------------------------------------------- |
| Deterministic Reference | FP64\_STRICT                | FP64\_STRICT                                                                        |
| Reproducible Production | FP64\_STATE\_FP64\_FORCE    | FP64\_STRICT, FP64\_STATE\_FP64\_FORCE, validated MIXED\_FORCE\_ACCUM\_DOUBLE       |
| Fast Production         | MIXED\_FORCE\_ACCUM\_DOUBLE | FP64\_STATE\_FP64\_FORCE, MIXED\_FORCE\_ACCUM\_DOUBLE, validated FP32\_EXPERIMENTAL |

### 9A.5. Политика по умолчанию

На старте проекта фиксируется:

- **Deterministic Reference = только FP64\_STRICT**
- **Reproducible Production = по умолчанию FP64**
- **Fast Production = mixed only после прохождения полной валидации**

### 9A.6. Почему это нужно фиксировать явно

Актуальная документация LAMMPS подтверждает, что разные accelerated paths используют разные precision policies:

- GPU package умеет single, double и mixed precision, где парные силы могут считаться в single, но аккумулироваться в double. ([docs.lammps.org](https://docs.lammps.org/Speed_gpu.html))
- INTEL package по умолчанию использует mixed precision; расчёты между парами/триплетами выполняются в single, а накопление — в double. ([docs.lammps.org](https://docs.lammps.org/Speed_intel.html))
- KOKKOS в документации сравнивается в том числе в режиме double precision и ориентирован на другой execution model, но precision policy там тоже должна задаваться осмысленно. ([docs.lammps.org](https://docs.lammps.org/Speed_kokkos.html))

Следовательно, в TDMD precision policy должна быть first-class architectural concept.

### 9A.7. Правило для физики

Любая смена precision policy должна считаться изменением научного режима и попадать в reproducibility bundle.

### 9A.8. Правило для acceptance tests

Для каждой precision policy обязаны существовать:

- `run 0` thresholds;
- NVE drift thresholds;
- long-run observable thresholds;
- differential reports against FP64\_STRICT.

### 9A.9. Правило для реализации

Precision policy должна прокидываться через явный объект конфигурации:

```cpp
struct PrecisionPolicy {
    PrecisionMode state_precision;
    PrecisionMode force_compute_precision;
    PrecisionMode accumulation_precision;
    PrecisionMode reduction_precision;
    bool deterministic_reduction;
};
```

Это обязательнее, чем просто enum `PrecisionMode`, потому что одна только метка `mixed` недостаточно точно описывает scientific guarantees.

### 9A.10. Стратегическое решение проекта

На этапе разработки:

- весь reference path строится и валидируется в FP64\_STRICT;
- mixed precision вводится только после стабилизации physical correctness;
- fast path не имеет права становиться scientific default до полной differential validation.

## 10. Determinism policy: детальная спецификация

### 10.1. Что считается детерминированностью

Система должна различать:

1. **Bitwise determinism**;
2. **Layout-invariant determinism**;
3. **Scientific reproducibility**.

### 10.2. Политика режима Deterministic Reference

Обязательно:

- stable sort after migration;
- stable sort after neighbor rebuild;
- canonical ordering `(zone_id, cell_id, atom_id)`;
- fixed reduction trees;
- запрет unordered atomics accumulation;
- fixed packet ordering;
- fixed scheduler priority function;
- deterministic RNG only.

### 10.3. Политика режима Reproducible Production

Разрешено:

- ограниченный overlap;
- часть оптимизаций reduction, если их error envelope фиксирован;
- часть reorder optimizations, если они не меняют canonical scientific output.

### 10.4. Политика режима Fast Production

Разрешено:

- non-canonical scheduling;
- device atomics accumulation;
- dynamic balancing;
- mixed precision;
- opportunistic communication progress.

Но запрещено:

- любые изменения, нарушающие accepted physics thresholds;
- оптимизации без benchmark coverage.

### 10.5. Determinism test matrix

Обязательная матрица:

- same run, repeated twice;
- restart vs non-restart;
- 1 GPU vs 2 GPU;
- 1 rank/GPU vs 2 ranks/GPU;
- different spatial decomposition;
- different temporal layout;
- deterministic vs reproducible vs fast comparison.

---

## 11. CI схема

### 11.1. Ветки и merge policy

- `main` — только зелёные merge;
- `dev` — интеграционная ветка;
- feature branches — работа агентов;
- architectural RFC changes — через отдельный review.

### 11.2. CI pipelines

#### Pipeline A — Lint & Build

- clang-format
- clang-tidy
- include guards
- CMake configure
- CPU build
- CUDA build

#### Pipeline B — Unit

- parser tests
- zone state tests
- certificate math
- stable sort
- reductions
- Morse unit tests
- EAM unit tests

#### Pipeline C — Property

- scheduler invariants
- zone transition invariants
- neighbor rebuild invariants
- migration invariants

#### Pipeline D — Differential

- run 0 vs reference
- force/energy/virial comparison
- NVE drift vs reference
- deterministic checks

#### Pipeline E — Performance

- small benchmark
- medium benchmark
- GPU kernel timings
- regression against stored baseline

#### Pipeline F — Reproducibility

- repeated run hash
- restart equivalence
- layout equivalence

### 11.3. CI environments

Минимально нужны:

- CPU-only Linux;
- CUDA single GPU;
- CUDA multi-GPU;
- optional HPC runner later.

### 11.4. Хранение baseline

Baselines должны храниться в:

- `ci/perf_baselines/`
- `ci/determinism_baselines/`
- `benchmarks/reference_outputs/`

---

## 12. Merge gates: обязательный чек-лист

Ни один PR не может быть слит без выполнения всех обязательных пунктов.

### 12.1. Обязательные gates

-

### 12.2. Дополнительные gates для performance-sensitive изменений

-

### 12.3. Дополнительные gates для scheduler changes

-

---

## 13. Схема тестирования

### 13.1. Тестовые слои

1. Unit tests
2. Property tests
3. Differential tests
4. Determinism tests
5. Performance tests
6. UX tests

### 13.2. Обязательные эталонные кейсы

- simple Morse 2-body
- small FCC single-species
- small FCC alloy
- EAM single-element
- EAM multicomponent small
- benchmark small/medium/large

### 13.3. Main oracle suite

Для верификации against reference обязательно:

- `run 0`
- `NVE drift`
- `NVT stability`
- `NPT relax`
- `MSD/diffusion`
- `performance sweeps`

---

## 14. UX и scientist-first design

### 14.1. Принципы

- simple by default;
- explicit when needed;
- preflight before launch;
- explainability of TD runtime;
- built-in compare with reference;
- reproducibility capsule by default.

### 14.2. CLI

Минимальный интерфейс:

```bash
tdmd run case.yaml
tdmd validate case.yaml
tdmd explain case.yaml
tdmd compare --with lammps case.yaml
tdmd repro-bundle run_dir/
```

### 14.3. Input philosophy

Поддерживаются:

- `tdmd.yaml` как основной user-facing формат;
- импорт LAMMPS data для compatibility path.

### 14.4. Preflight checks

Перед стартом обязательны:

- units consistency;
- potential checksum and compatibility;
- species mapping;
- timestep safety estimate;
- memory estimate;
- precision/mode warnings;
- deterministic limitations warning.

### 14.5. Explainability

Команда `tdmd explain` должна выводить:

- zoning strategy;
- estimated pipeline depth;
- expected bottlenecks;
- selected mode policy;
- communication backend;
- precision policy.

---

## 15. Стартовый backlog первых 30 задач для ИИ-агентов

Ниже — начальный управляемый backlog. Каждая задача должна быть оформлена как отдельный issue/spec task.

### Block A — Repository & Process

1. Создать monorepo skeleton с каталогами `src/tests/docs/tools/benchmarks/ci`.
2. Настроить CMake project с CPU и CUDA targets.
3. Добавить clang-format, clang-tidy, pre-commit hooks.
4. Создать шаблоны `SPEC.md`, `TESTPLAN.md`, `README.md` для модулей.
5. Настроить CI pipeline A (lint + build).

### Block B — Core types

6. Реализовать базовые enums/config types (`ExecMode`, `PrecisionMode`, `RuntimeConfig`).
7. Реализовать `BuildInfo` и `ReproContext`.
8. Реализовать `AtomSoA` и базовые unit tests.
9. Реализовать `Box`, `DomainDecomposition`.
10. Реализовать сериализацию runtime config в JSON/YAML.

### Block C — IO & Input

11. Реализовать парсер `tdmd.yaml`.
12. Реализовать preflight validator.
13. Реализовать импорт простого LAMMPS data файла.
14. Реализовать запись reproducibility capsule.
15. Реализовать минимальный CLI `tdmd validate case.yaml`.

### Block D — Neighbor & Geometry

16. Реализовать `CellGrid` на CPU.
17. Реализовать deterministic binning и stable reorder.
18. Реализовать CPU neighbor list build.
19. Реализовать property tests на invariants neighbor rebuild.
20. Реализовать displacement tracking и skin trigger.

### Block E — Potentials & Integrator

21. Реализовать CPU `VelocityVerletIntegrator`.
22. Реализовать CPU reference `MorsePotential`.
23. Реализовать unit tests for Morse force/energy.
24. Реализовать CPU reference `EamAlloyPotential` skeleton.
25. Реализовать `run 0` execution path.

### Block F — Scheduler & TD core

26. Реализовать `ZoneState` machine и transition tests.
27. Реализовать `SafetyCertificate` math и tests.
28. Реализовать `ZoneTask` + minimal deterministic scheduler.
29. Реализовать `SimulationEngine` happy path для single-zone CPU.
30. Реализовать первый differential test vs reference on small case.

---

## 16. Roadmap milestones

### M0 — Process bootstrap

- repo
- CI skeleton
- templates
- specs

### M1 — CPU reference MD

- YAML input
- AtomSoA
- neighbor build
- velocity-Verlet
- Morse
- run 0

### M2 — CPU EAM + differential validation

- EAM
- benchmark small
- NVE drift
- reference compare

### M3 — Deterministic TD scheduler

- zones
- certificates
- wavefront scheduling
- determinism tests

### M4 — Single-GPU production path

- CUDA state
- GPU neighbor build
- GPU Morse/EAM
- NVTX telemetry

### M5 — Multi-GPU time × space

- halo exchange
- temporal packets
- GPU-aware MPI
- reproducible mode

### M6 — Fast mode

- overlap
- mixed precision
- dynamic balancing
- performance gates

### M7 — Long-range service hooks

- split partition interface
- outer MTS hooks

### M8 — ML local potentials

- SNAP
- MLIAP
- PACE
- KIM adapter

### M9 — Scientist-facing release

- polished CLI
- recipes
- compare command
- visualization support

---

## 17. Open design decisions

Эти вопросы пока считаются открытыми и должны отслеживаться в документе:

1. Политика triclinic boxes в v1: поддерживать сразу или отложить.
2. Строгая layout-invariant bitwise determinism: обязательная цель или stretch goal.
3. Граница между Reproducible и Fast mode: какие exactly optimizations включаются.
4. Политика mixed precision для EAM и ML kernels.
5. Формат пользовательского input: YAML only или YAML + script mode.
6. Нужно ли Python API в v1.
7. Когда включать NPT в strict deterministic path.
8. Когда включать long-range split service в основной backlog.

---

## 18. Правила сопровождения этого документа

1. Любое архитектурное решение сначала отражается здесь.
2. Любой новый режим или существенная оптимизация сначала оформляется как изменение этого документа.
3. Любая новая категория тестов сначала добавляется в этот документ.
4. Любой ИИ-агент должен ссылаться на соответствующий раздел этого документа при постановке задачи.
5. Если код и документ расходятся, источником истины до resolution считается этот документ.

---

## 18A. Явная архитектурная оговорка по режимам

Формулировка, которую нужно считать канонической для проекта:

> **Fast mode — не отдельный движок и не отдельная физика.**\
> Это запуск общего TDMD core с другим policy-набором: scheduler/comm/reduction/precision/overlap.

Более точно:

- Deterministic mode задаёт reference semantics;
- Reproducible mode задаёт scientific production semantics;
- Fast mode задаёт throughput-oriented semantics;
- все они обязаны опираться на один и тот же physics core и один и тот же oracle/test framework.

## 19. Следующие рекомендуемые шаги

Немедленно после принятия этого документа рекомендуется:

1. выделить разделы в отдельные `SPEC.md` по модулям;
2. сформировать issue board из первых 30 задач;
3. утвердить политику трёх режимов как обязательную;
4. зафиксировать acceptance thresholds для differential/determinism/performance tests;
5. начать с M0 и M1 без параллельного расползания в ML и long-range.

---

## 20. Change log

### v0.1-draft

- зафиксирована standalone стратегия TDMD;
- зафиксирована методика Spec-Driven TDD;
- добавлена спецификация трёх режимов исполнения;
- добавлена начальная архитектура классов C++/CUDA;
- добавлены CI/gates/backlog/roadmap;
- документ объявлен главным живым инженерным документом проекта.

---

## 21. Детализация: TD scheduler

### 21.1. Назначение scheduler

`td_scheduler` — центральный модуль исполнения TDMD. Он отвечает не за физику и не за коммуникации как таковые, а за **законное продвижение зон по времени**.

Scheduler обязан обеспечивать:

1. корректность причинных зависимостей;
2. отсутствие illegal transitions между состояниями зон;
3. отсутствие deadlock в поддерживаемых режимах;
4. bounded progress — если существуют ready tasks, scheduler обязан их выдать;
5. соблюдение policy выбранного режима исполнения.

### 21.2. Единица работы

Единица работы — `ZoneTask`.

Минимальный состав:

```text
ZoneTask:
  zone_id
  time_level
  local_state_version
  dependency_mask
  safety_certificate_version
  priority
  mode_policy_tag
```

Дополнительные runtime-поля:

```text
ZoneRuntimeMeta:
  assigned_device
  assigned_rank
  retry_count
  last_failure_reason
  task_epoch
  launch_order_id
```

### 21.3. Граф зависимостей

Scheduler должен рассматривать исполнение как DAG:

- вершина: `(zone_id, time_level, version)`;
- ребро: причинная зависимость short-range, spatial halo dependency, temporal frontier dependency или long-range barrier dependency.

Типы зависимостей:

1. `SpatialHaloDependency`
2. `TemporalFrontierDependency`
3. `CertificateDependency`
4. `LongRangeWindowDependency`
5. `NeighborValidityDependency`

### 21.4. Приоритеты задач

В deterministic режиме приоритет обязан быть фиксированным и чистой функцией от:

```text
priority = f(time_level, zone_order, local_state_version)
```

Рекомендуемый базовый порядок:

1. меньший `time_level` имеет больший приоритет;
2. при равном `time_level` используется фиксированный `zone_order`;
3. при равных значениях — меньший `local_state_version`.

В reproducible режиме допустим ограниченный adaptive priority, но только если он сохраняет доказуемую инвариантность результата в пределах заявленной guarantees policy.

В fast режиме priority может включать telemetry:

- queue length;
- device pressure;
- halo readiness;
- estimated compute cost.

### 21.5. Очереди scheduler

Минимальный набор очередей:

- `ready_queue`
- `blocked_queue`
- `inflight_queue`
- `completed_queue`
- `retry_queue`

В deterministic режиме очереди должны быть реализованы так, чтобы порядок извлечения был каноническим.

### 21.6. События scheduler

Scheduler обрабатывает следующие события:

1. `ZoneDataArrived`
2. `HaloArrived`
3. `CertificateRefreshed`
4. `NeighborRebuildCompleted`
5. `ForceComputeCompleted`
6. `IntegrateCompleted`
7. `TemporalPacketCommitted`
8. `LongRangeWindowReleased`

Любое событие должно приводить либо к:

- обновлению графа зависимостей;
- смене состояния зоны;
- выдаче новых ready tasks;
- записи telemetry.

### 21.7. Инварианты scheduler

Обязательные инварианты:

1. `Committed` зона не может вернуться в `Ready` без смены `time_level`.
2. Зона не может быть выдана в `Computing`, если сертификат невалиден.
3. Зона не может быть одновременно в `ready_queue` и `inflight_queue`.
4. Для одной и той же `(zone_id, time_level, version)` не может существовать более одной активной compute-задачи.
5. `Completed` не означает `Committed`; commit — отдельный шаг после подтверждения всех postconditions.

### 21.8. Commit protocol

Коммит зоны обязан выполняться в две фазы:

**Phase A — local completion**

- force/integrator завершены;
- local buffers обновлены;
- zone marked `Completed`.

**Phase B — global legality**

- packet/halo dependencies выполнены;
- post-step certificate сохранён;
- telemetry snapshot сделан;
- зона переводится в `Committed`.

### 21.9. Fail/retry policy

Если задача не может быть законно завершена:

- причина фиксируется в `last_failure_reason`;
- задача переводится в `retry_queue`;
- для deterministic режима число и порядок retry должны быть каноническими.

### 21.10. Deadlock policy

Scheduler обязан иметь watchdog-инвариант:

```text
If unfinished() and no ready tasks and no inflight progress for T_watchdog
=> hard diagnostic failure
```

В deterministic и reproducible режимах это должно считаться ошибкой исполнения, а не silently recovered condition.

---

## 22. Детализация: Safety Certificate

### 22.1. Назначение

`SafetyCertificate` — формальное доказательство того, что конкретную зону можно продвинуть на следующий локальный шаг без нарушения причинной изоляции.

Без валидного сертификата зона не может быть выдана в compute.

### 22.2. Компоненты сертификата

Минимальный состав сертификата:

```text
SafetyCertificate:
  safe: bool
  certificate_id
  zone_id
  time_level
  version
  vmax_zone
  amax_zone
  dt_candidate
  displacement_bound
  buffer_width
  skin_remaining
  frontier_margin
  neighbor_valid_until
  halo_valid_until
  mode_policy_tag
```

### 22.3. Основная идея оценки

Базовая оценка должна быть консервативной:

```text
displacement_bound = vmax_zone * dt + 0.5 * amax_zone * dt^2
```

Минимальное условие безопасности:

```text
safe if displacement_bound < min(buffer_width, skin_remaining, frontier_margin)
```

Для v1 разрешается использовать более простую оценку с `vmax_zone`, если это отражено в policy и test coverage.

### 22.4. Источники данных для сертификата

Сертификат использует:

- скорости атомов зоны;
- оценки ускорений зоны;
- текущий `skin_remaining`;
- расстояние до temporal frontier;
- ширину буфера зоны;
- статус spatial halo;
- версию neighbor list.

### 22.5. Режимы строгости сертификата

#### Deterministic Reference

- максимально консервативный;
- одинаковый алгоритм оценки для всех платформ;
- без runtime-адаптивных эвристик;
- одно и то же состояние => один и тот же сертификат.

#### Reproducible Production

- допускает предсказуемую адаптацию буфера;
- допускает mode-stable heuristics;
- не допускает nondeterministic thresholding.

#### Fast Production

- допускает более агрессивные оценки;
- допускает cost-aware widening/narrowing zones;
- все fast эвристики должны иметь safety fallback.

### 22.6. Certificate lifecycle

1. построение;
2. валидация;
3. использование для launch decision;
4. инвалидация при изменении neighbor/halo/frontier;
5. пересчёт;
6. архивирование telemetry.

### 22.7. Certificate invalidation triggers

Сертификат обязан инвалидироваться при:

- изменении версии zone state;
- rebuild neighbor list;
- migration атомов;
- изменении halo;
- изменении `dt`;
- изменении режима исполнения;
- изменении long-range window state.

### 22.8. Certificate tests

Обязательные тесты:

- safe case with static atoms;
- unsafe case with high `vmax`;
- invalidation after neighbor rebuild;
- invalidation after migration;
- deterministic equality for same state;
- monotonicity: увеличение `dt` не должно делать unsafe case safe.

---

## 23. Детализация: Determinism Policy

### 23.1. Уровни guarantees

В TDMD фиксируются три уровня гарантий:

1. **Bitwise determinism**
2. **Layout-invariant determinism**
3. **Scientific reproducibility**

Не все уровни обязаны выполняться во всех режимах.

### 23.2. Определения

#### Bitwise determinism

Одинаковый binary + одинаковый runtime + одинаковая hardware class + одинаковый запуск дают идентичный бинарный результат:

- координаты;
- скорости;
- силы;
- энергии;
- dumps;
- restarts.

#### Layout-invariant determinism

Изменение числа ranks/GPU и схемы раскладки не меняет результат в пределах bitwise equality или заранее определённого stricter-than-scientific threshold.

#### Scientific reproducibility

Идентичны наблюдаемые и статистики в accepted tolerance, даже если отдельная траектория не совпадает бит-в-бит.

### 23.3. Что обязательно для Deterministic Reference

- bitwise determinism: да;
- layout-invariant determinism: целевая цель для supported layouts;
- scientific reproducibility: да.

### 23.4. Что обязательно для Reproducible Production

- bitwise determinism: желательно в пределах одного runtime layout;
- layout invariance: опционально, но с тестируемым envelope;
- scientific reproducibility: обязательно.

### 23.5. Что обязательно для Fast Production

- bitwise determinism: нет;
- layout invariance: нет;
- scientific reproducibility: обязательно.

### 23.6. Источники потери детерминизма

Нужно явно отслеживать:

1. порядок суммирования floating-point величин;
2. unordered atomics;
3. nondeterministic communication arrival order;
4. dynamic load balancing;
5. unstable sorting;
6. varying neighbor build order; 7

---

## 29. Универсальные роли и промты для Codex

Этот раздел предназначен для переноса дальнейшей реализации в Codex.  
Цель — дать **универсальный набор ролей, системных инструкций и рабочих промтов**, чтобы проект TDMD можно было дальше вести уже через Codex-поток без постоянной ручной перепостановки контекста.

### 29.1. Базовый принцип работы с Codex

Codex не должен работать как «свободный автогенератор кода».  
Он должен работать как **исполнитель в рамках master-spec**.

Обязательное правило:

> Любая задача для Codex должна ссылаться на текущий `TDMD Engineering Spec` и, если применимо, на соответствующий модульный `SPEC.md`.

То есть Codex всегда получает:

1. роль;
2. цель задачи;
3. scope / out-of-scope;
4. ссылки на релевантные секции spec;
5. required tests;
6. expected artifacts;
7. merge/acceptance gates.

### 29.2. Глобальный системный промт для Codex

Ниже — **универсальный системный промт**, который рекомендуется использовать как основной для рабочих сессий Codex по TDMD.

```text
Ты работаешь над проектом TDMD — самостоятельной программой молекулярной динамики с декомпозицией по времени (Time Decomposition, TD), ориентированной на сплавы, EAM, локальные ML-потенциалы и GPU.

Главный источник истины по проекту — TDMD Engineering Spec и модульные SPEC.md. Нельзя придумывать архитектуру, интерфейсы, режимы исполнения или физические допущения в обход этих документов.

Обязательные правила:
1. Сначала прочитай релевантные секции spec, затем предложи план, затем меняй код.
2. Не создавай второй скрытый движок для fast mode: все режимы должны быть policy-слоями над общим core.
3. Не ослабляй deterministic/reference path ради производительности.
4. Любая fast-оптимизация должна быть флагуемой, отключаемой и покрытой validation.
5. Не меняй public interface без явного отражения в spec или без предложения SPEC delta.
6. Любая задача должна завершаться тестами, документацией и кратким отчётом о том, что было сделано.
7. Если задача затрагивает физическую корректность, обязательно предложи differential tests / run0 / NVE drift / reproducibility checks.
8. Если задача performance-sensitive, добавь telemetry hooks и perf-note.
9. Если задача неоднозначна, не придумывай молча: сначала перечисли допущения.
10. Если код и spec расходятся, сначала исправь расхождение через явное замечание, а не скрытой правкой.

Формат ответа по умолчанию:
- Что понял
- План
- Какие файлы изменишь
- Что именно реализуешь
- Какие тесты добавишь
- Риски / допущения
```

### 29.3. Универсальный developer prompt для Codex-сессий

Этот prompt рекомендуется использовать как рабочий «developer message» поверх системного промта.

```text
Работай строго в рамках TDMD Engineering Spec.

Перед началом:
- найди релевантные разделы spec;
- выпиши ограничения;
- выпиши invariants;
- перечисли required tests.

Во время реализации:
- не делай скрытых архитектурных изменений;
- не меняй режимы исполнения без отражения в policy semantics;
- не добавляй ad-hoc flags, если они должны быть частью RuntimePolicyBundle;
- не внедряй физику в orchestration-слой и наоборот.

После реализации:
- перечисли изменённые файлы;
- перечисли добавленные тесты;
- перечисли, какие acceptance criteria теперь закрыты;
- перечисли, остались ли открытые риски.
```

### 29.4. Универсальный user prompt-шаблон для задач Codex

Ниже — универсальный шаблон постановки задачи в Codex.

```text
Задача: <краткое название>

Контекст:
- Проект: TDMD
- Главный spec: TDMD Engineering Spec
- Модульный spec: <указать файл или раздел>
- Milestone: <M0/M1/...>

Цель:
<что нужно получить>

Scope:
- <что входит>
- <что входит>

Out of scope:
- <что не делать>
- <что не делать>

Обязательные инварианты:
- <инвариант 1>
- <инвариант 2>

Нужные файлы:
- <файл 1>
- <файл 2>

Требуемые тесты:
- <unit>
- <property>
- <differential>
- <determinism>

Ожидаемые артефакты:
- код
- тесты
- краткий отчёт
- при необходимости обновление docs/spec

Формат ответа:
1. Понимание задачи
2. План
3. Изменяемые файлы
4. Реализация
5. Тесты
6. Риски
```

---

### 29.5. Канонические роли для Codex

Ниже — рекомендуемый набор ролей. Эти роли можно использовать как переключаемые «persona prompts» для разных типов задач.

#### Роль 1 — Architect / Spec Steward

Когда использовать:
- проектирование архитектуры;
- уточнение интерфейсов;
- reconciliation между модулями;
- SPEC delta.

Промт:

```text
Ты выступаешь как Architect / Spec Steward проекта TDMD.

Твоя задача:
- сохранять архитектурную целостность;
- следить, чтобы код соответствовал TDMD Engineering Spec;
- не позволять скрытым fork-ам runtime или physics semantics;
- выявлять конфликты между интерфейсами модулей.

При ответе всегда:
1. ссылайся на relevant sections spec;
2. перечисляй invariants;
3. отделяй обязательное от желательного;
4. если нужен interface change — сначала предложи SPEC delta.
```

#### Роль 2 — Core Runtime Engineer

Когда использовать:
- `SimulationEngine`;
- lifecycle;
- state;
- orchestration;
- config/policy plumbing.

Промт:

```text
Ты выступаешь как Core Runtime Engineer проекта TDMD.

Твоя задача:
- реализовывать жизненный цикл TDMD;
- соблюдать orchestrator contracts;
- не встраивать физику в runtime-слой;
- не ломать mode/policy semantics.

Всегда проверяй:
- ownership границ модулей;
- state machine transitions;
- policy enforcement points;
- restart/resume compatibility.
```

#### Роль 3 — Scheduler / Determinism Engineer

Когда использовать:
- `td_scheduler`;
- `SafetyCertificate`;
- queues;
- determinism policy;
- commit ordering.

Промт:

```text
Ты выступаешь как Scheduler / Determinism Engineer проекта TDMD.

Твоя задача:
- защищать legality зонного продвижения по времени;
- сохранять deterministic/reference semantics;
- не допускать illegal transitions, deadlock и hidden nondeterminism.

Особое внимание:
- queue ordering;
- certificate invalidation;
- commit protocol;
- layout-invariant behavior where declared.
```

#### Роль 4 — Neighbor / Migration Engineer

Когда использовать:
- `CellGrid`;
- `NeighborList`;
- rebuild policy;
- migration;
- stable reorder.

Промт:

```text
Ты выступаешь как Neighbor / Migration Engineer проекта TDMD.

Твоя задача:
- реализовывать neighbor, migration и reorder как отдельные, прозрачные слои;
- не смешивать rebuild, migration и reorder в скрытые side effects;
- сохранять atom identity и canonical ordering там, где это требуется.

Обязательно контролируй:
- versioning;
- rebuild triggers;
- migration records;
- reorder maps;
- deterministic ordering guarantees.
```

#### Роль 5 — Physics Engineer (Morse/EAM)

Когда использовать:
- `Morse`;
- `EAM`;
- energy/force/virial;
- interpolation;
- CPU/GPU differential path.

Промт:

```text
Ты выступаешь как Physics Engineer проекта TDMD.

Твоя задача:
- реализовывать физически корректные short-range потенциалы;
- поддерживать CPU reference path как канонический оракул;
- делать GPU path только как validated variant reference path.

Нельзя:
- внедрять hidden migration/rebuild logic в potential layer;
- менять физическую форму модели ради производительности;
- включать mixed precision без явной policy semantics и validation.
```

#### Роль 6 — GPU / Performance Engineer

Когда использовать:
- CUDA kernels;
- streams;
- overlap;
- GPU memory layout;
- NVTX/profiling.

Промт:

```text
Ты выступаешь как GPU / Performance Engineer проекта TDMD.

Твоя задача:
- ускорять validated core,
- не создавая второй скрытый движок;
- все fast paths должны быть policy-controlled, measurable и reversible.

Обязательно:
- добавляй telemetry/NVTX;
- сравнивай до/после;
- не ломай deterministic/reference path;
- если оптимизация меняет numerical behavior, это надо явно классифицировать.
```

#### Роль 7 — Validation / Reference Engineer

Когда использовать:
- compare with LAMMPS;
- thresholds;
- differential testing;
- benchmark manifests;
- reproducibility checks.

Промт:

```text
Ты выступаешь как Validation / Reference Engineer проекта TDMD.

Твоя задача:
- превращать LAMMPS и reference cases в рабочий scientific oracle;
- следить, чтобы каждая новая возможность имела путь верификации;
- оформлять thresholds, compare reports и acceptance verdicts.

Особое внимание:
- run 0;
- NVE drift;
- observables;
- restart equivalence;
- mode-specific acceptance.
```

#### Роль 8 — Scientist UX Engineer

Когда использовать:
- CLI;
- validate/explain/compare;
- recipes;
- docs;
- reproducibility bundle UX.

Промт:

```text
Ты выступаешь как Scientist UX Engineer проекта TDMD.

Твоя задача:
- делать TDMD удобным для исследователя;
- упрощать вход в продукт без потери научной строгости;
- превращать runtime complexity в понятный scientist-facing workflow.

Приоритеты:
- actionable errors;
- понятные warnings;
- reproducibility by default;
- explainability of TD runtime;
- compare with reference as first-class workflow.
```

---

### 29.6. Универсальные промты по типам работ

#### A. Промт на реализацию модуля

```text
Роль: <выбери роль>

Нужно реализовать модуль/часть модуля TDMD.

Сначала:
1. выпиши, какие разделы TDMD Engineering Spec относятся к задаче;
2. перечисли invariants;
3. перечисли public interfaces, которые нельзя сломать;
4. перечисли тесты, которые обязан добавить.

Потом:
- предложи короткий план;
- перечисли изменяемые файлы;
- реализуй код;
- добавь тесты;
- в конце дай отчёт: что реализовано, что не реализовано, какие риски остались.
```

#### B. Промт на рефакторинг

```text
Роль: Architect / Spec Steward + <нужная техническая роль>

Нужно выполнить рефакторинг в TDMD без изменения scientific semantics.

Обязательно:
- сначала перечисли текущие invariants;
- явно укажи, какие public interfaces сохраняются;
- не меняй mode/policy semantics;
- добавь regression tests или обнови существующие.

В конце:
- перечисли, какие инварианты сохранились;
- изменилась ли performance / determinism / reproducibility semantics.
```

#### C. Промт на fast-оптимизацию

```text
Роль: GPU / Performance Engineer

Нужно предложить и реализовать fast-оптимизацию для TDMD.

Ограничения:
- нельзя создавать второй движок;
- оптимизация должна быть policy-controlled;
- deterministic/reference path должен остаться доступным;
- нужно явно указать, влияет ли оптимизация на numerical behavior.

Обязательно:
1. опиши baseline;
2. опиши proposed optimization;
3. укажи, какие guarantees сохраняются и какие ослабляются;
4. добавь perf + validation tests;
5. добавь telemetry/profiling hooks.
```

#### D. Промт на верификацию

```text
Роль: Validation / Reference Engineer

Нужно спроектировать или реализовать validation для новой возможности TDMD.

Обязательно:
- определи scientific risk;
- определи compare strategy against reference;
- предложи thresholds;
- перечисли required artifacts:
  - run0 diff
  - NVE drift
  - observables summary
  - determinism / reproducibility checks

В конце дай verdict template для CI.
```

#### E. Промт на SPEC delta

```text
Роль: Architect / Spec Steward

Нужно предложить изменение архитектуры TDMD.

Сначала:
- перечисли, какие разделы текущего spec затрагиваются;
- объясни, зачем change нужен;
- перечисли риски;
- отдели обязательное изменение от опционального.

Потом:
- предложи SPEC delta в формате:
  1. Что меняется
  2. Что остаётся прежним
  3. Какие интерфейсы затрагиваются
  4. Какие тесты и acceptance criteria надо изменить

Не меняй код, пока SPEC delta не сформулирован.
```

---

### 29.7. Универсальные правила для Codex при работе с TDMD

Codex должен всегда соблюдать следующие правила.

#### Rule 1 — Сначала spec
Перед кодом всегда сначала определить, на какие секции spec опирается задача.

#### Rule 2 — Никаких скрытых side effects
Нельзя тайно вставлять rebuild/migration/reorder/precision switches в несоответствующие слои.

#### Rule 3 — Один core
Нельзя делать отдельный fast-engine или alternative physics path.

#### Rule 4 — CPU reference sacred
CPU reference path нельзя деградировать или размывать ради performance.

#### Rule 5 — Validation mandatory
Любая задача, влияющая на физику, determinism или precision, обязана завершаться test plan и validation hooks.

#### Rule 6 — Explain changes
Если меняется что-то нетривиальное, в конце ответа должен быть раздел:
- что изменилось в semantics;
- что изменилось в tests;
- что изменилось в guarantees.

#### Rule 7 — Не угадывать silently
Если в задаче не хватает данных, сначала перечислить допущения.

### 29.8. Минимальный стартовый комплект для Codex

Если нужно быстро перенести работу в Codex, минимально достаточно всегда передавать ему:

1. глобальный системный промт из `29.2`;
2. одну из ролей из `29.5`;
3. универсальный шаблон задачи из `29.4`;
4. ссылку на соответствующие секции master-spec / module spec.

### 29.9. Рекомендуемая рабочая схема в Codex

Для каждой новой задачи:

1. выбрать роль;
2. дать системный промт;
3. дать task prompt по шаблону;
4. заставить Codex сначала выписать invariants и plan;
5. только потом принимать код.

### 29.10. Правило завершения Codex-сессии

Любая сессия Codex по TDMD должна завершаться структурированным отчётом:

```text
1. Что реализовано
2. Какие файлы изменены
3. Какие тесты добавлены/обновлены
4. Какие acceptance criteria теперь покрыты
5. Какие риски/долги остались
6. Требуется ли SPEC delta
```

### 29.11. Что стоит сразу передать в Codex вместе с этим разделом

Рекомендуемый базовый пакет контекста:

- `TDMD Engineering Spec`
- `docs/specs/policies/SPEC.md`
- `docs/specs/runtime/SPEC.md`
- `docs/specs/testing/SPEC.md`
- `docs/specs/<целевой модуль>/SPEC.md`
- `docs/execution/M0_M3_EXECUTION_PACK.md`

### 29.12. Final recommendation

Для дальнейшей работы в Codex использовать такую иерархию истины:

1. `TDMD Engineering Spec`
2. модульный `SPEC.md`
3. execution pack / issue task
4. код

Если код противоречит spec, Codex должен сначала предложить reconciliation, а не молча продолжать реализацию.


---

## 30. Version 0.3 redesign: compile-time numeric builds + runtime execution profiles

Этот раздел является **новой версией спецификации** по вопросам режимов исполнения, точности и policy architecture.

Если этот раздел противоречит более ранним формулировкам про `runtime PrecisionPolicy`, **истиной считается именно этот раздел**.

### 30.1. Причина пересмотра

После дополнительного анализа зафиксировано новое архитектурное решение:

> **численные типы, арифметика и низкоуровневые численные пути должны задаваться compile-time, а не как свободный runtime-переключатель внутри одного бинарника.**

Это решение принято потому что оно:

- безопаснее;
- проще для CUDA specialization;
- проще для тестирования;
- легче для воспроизводимости;
- уменьшает риск скрытых численных режимов;
- уменьшает ветвление в коде;
- снижает риск случайных численных расхождений в одном и том же бинарнике.

### 30.2. Новая модель: две оси вместо одной

Теперь архитектура делится на две независимые оси.

#### Ось A — BuildFlavor

Это **compile-time вариант сборки**, который фиксирует numerical semantics.

BuildFlavor определяет:

- тип хранения состояния атомов;
- тип вычисления сил;
- тип накопления сил/энергии/вириала;
- тип редукций;
- deterministic numeric traits;
- допустимые CPU/CUDA specialization paths.

#### Ось B — ExecProfile

Это **runtime-профиль исполнения**, который задаёт поведение runtime layer.

ExecProfile определяет:

- scheduler policy;
- communication policy;
- reorder policy;
- overlap policy;
- validation strictness;
- telemetry verbosity;
- guardrails.

### 30.3. Что теперь запрещено

В TDMD теперь запрещено:

- иметь один бинарник с произвольным runtime переключением `float/double/mixed` на уровне реальных типов состояния и арифметики;
- скрыто менять тип накопления или редукций через runtime flags;
- маскировать разные numerical implementations под один и тот же runtime mode.

### 30.4. Канонические BuildFlavor v1

На старте проекта фиксируются следующие compile-time варианты:

1. `Fp64ReferenceBuild`
2. `Fp64ProductionBuild`
3. `MixedFastBuild`
4. `Fp32ExperimentalBuild` (не scientific default)

### 30.5. Канонические ExecProfile v1

На runtime доступны следующие профили:

1. `Reference`
2. `Production`
3. `FastExperimental`

### 30.6. Новый главный принцип

> **Fast mode — это не отдельный движок и не отдельная физика.**
> Это общий TDMD core + другой runtime policy profile, исполняемый поверх конкретного compile-time build flavor.

То есть:

- один core;
- одна физика;
- одна модульная архитектура;
- одна система тестов;
- разные runtime policies;
- но численная семантика фиксируется build flavor’ом.

### 30.7. Канонические соответствия BuildFlavor × ExecProfile

Нужно считать действующей следующую матрицу совместимости.

| BuildFlavor | Allowed ExecProfile |
|---|---|
| `Fp64ReferenceBuild` | `Reference`, `Production` |
| `Fp64ProductionBuild` | `Production`, optionally `Reference` |
| `MixedFastBuild` | `FastExperimental`, optionally validated `Production` |
| `Fp32ExperimentalBuild` | `FastExperimental` only |

### 30.8. Человеческие runtime labels

Для scientist-facing UX пользователь видит только понятные execution labels:

- `reference`
- `production`
- `fast-experimental`

Но движок обязан явно логировать, какой именно `BuildFlavor` реально используется.

### 30.9. Compile-time NumericConfig

Численная семантика должна быть оформлена через compile-time конфиги.

Рекомендуемая модель:

```cpp
namespace tdmd {

struct NumericConfigFp64Reference {
    using StateReal = double;
    using ForceReal = double;
    using AccumReal = double;
    using ReductionReal = double;
    static constexpr bool deterministic_reduction = true;
    static constexpr bool allow_device_atomics = false;
};

struct NumericConfigFp64Production {
    using StateReal = double;
    using ForceReal = double;
    using AccumReal = double;
    using ReductionReal = double;
    static constexpr bool deterministic_reduction = true;
    static constexpr bool allow_device_atomics = false;
};

struct NumericConfigMixedFast {
    using StateReal = double;
    using ForceReal = float;
    using AccumReal = double;
    using ReductionReal = double;
    static constexpr bool deterministic_reduction = false;
    static constexpr bool allow_device_atomics = true;
};

}
```

### 30.10. Что становится runtime policy после redesign

После redesign runtime policy bundle отвечает за:

- task ordering;
- canonical/non-canonical commit order;
- communication semantics;
- overlap;
- task stealing;
- validation gates;
- diagnostics.

Но больше **не отвечает** за выбор реальных численных типов внутри одного бинарника.

### 30.11. Новый RuntimeConfig

Пользовательская runtime-конфигурация должна задавать execution profile, но не numerical types.

Рекомендуемая форма:

```cpp
namespace tdmd {

enum class ExecProfile {
    Reference,
    Production,
    FastExperimental
};

struct RuntimeConfig {
    ExecProfile exec_profile;
    DeviceBackend backend;
    bool gpu_aware_mpi;
    bool enable_nvtx;
    bool enable_task_stealing;
};

struct BuildFlavorInfo {
    std::string build_flavor;
    std::string numeric_config_id;
};

}
```

### 30.12. Новый RuntimePolicyBundle

Runtime policy bundle теперь должен содержать runtime semantics и view на numeric build, но не управлять арифметикой как свободной осью.

Рекомендуемая форма:

```cpp
namespace tdmd {

struct ExecProfilePolicy {
    ExecProfile profile;
    bool require_bitwise_determinism;
    bool require_layout_invariant_execution;
    bool require_scientific_reproducibility;
};

struct NumericPolicyView {
    std::string build_flavor;
    std::string numeric_config_id;
    bool deterministic_reduction;
    bool allow_device_atomics;
};

struct RuntimePolicyBundle {
    ExecProfilePolicy exec;
    NumericPolicyView numeric;
    SchedulerPolicy scheduler;
    ReductionPolicy reduction;
    CommPolicy comm;
    ReorderPolicy reorder;
    OverlapPolicy overlap;
    ValidationPolicy validation;
};

}
```

### 30.13. Новый PolicyFactory

PolicyFactory теперь должен строить runtime policy profile **с учётом build flavor**.

```cpp
namespace tdmd {

class PolicyFactory {
public:
    static RuntimePolicyBundle make_reference_profile(const BuildFlavorInfo&);
    static RuntimePolicyBundle make_production_profile(const BuildFlavorInfo&);
    static RuntimePolicyBundle make_fast_profile(const BuildFlavorInfo&);
};

}
```

### 30.14. Новый PolicyValidator

PolicyValidator теперь обязан проверять compatibility matrix:

- `Reference` profile на `MixedFastBuild` — reject;
- `Production` profile на `Fp32ExperimentalBuild` — reject;
- unsupported combinations — reject с понятной диагностикой.

### 30.15. Scientist-facing config after redesign

Пользовательский YAML должен выбирать `exec_profile`, а не “живые” numerical types.

```yaml
runtime:
  exec_profile: production     # reference | production | fast-experimental
  backend: cuda
  device_count: 1

scheduler:
  preset: auto

comm:
  preset: auto

policy_overrides:
  allow_task_stealing: false
  canonical_packet_order: true
```

`BuildFlavor` должен определяться самим бинарником и быть доступным через introspection и в логах.

### 30.16. CI after redesign

CI должен тестировать разные build flavors как отдельные binary targets, например:

- `tdmd_ref_fp64`
- `tdmd_prod_fp64`
- `tdmd_fast_mixed`

Это предпочтительнее, чем один супербинарник с множеством runtime numerical switches.

### 30.17. Acceptance policy after redesign

Для каждого build flavor должны быть отдельно определены:

- differential thresholds;
- NVE drift thresholds;
- reproducibility expectations;
- performance baselines.

### 30.18. Impact on module specs

После принятия этого redesign все модульные SPEC должны считать действующим следующее правило:

> runtime-режимы влияют на policy semantics,  
> а численная арифметика и реальные типы — часть compile-time build flavor.

Это особенно важно для:

- `policies/SPEC.md`
- `runtime/SPEC.md`
- `potentials/SPEC.md`
- `integrator/SPEC.md`
- `testing/SPEC.md`
- `cli/SPEC.md`

### 30.19. Immediate migration rule

До физической переработки всех модульных SPEC нужно считать, что:

- старые упоминания `runtime PrecisionPolicy` как свободного выбора арифметики **устарели**;
- новая норма — `BuildFlavor + ExecProfile`.

### 30.20. New strategic wording for Codex and agents

Во всех дальнейших задачах для Codex и ИИ-агентов нужно использовать следующую формулировку:

> не добавлять runtime-переключение фундаментальных численных типов;  
> numerical semantics задаётся compile-time build flavor;  
> runtime выбирает только execution profile и policy behavior.

### 30.21. What changes immediately in practice

С этого момента в проекте считается правильным следующее:

1. reference path строится вокруг `Fp64ReferenceBuild`;
2. production baseline — `Fp64ProductionBuild`;
3. mixed precision появляется позже как отдельный `MixedFastBuild`;
4. scientist-facing CLI говорит про `reference/production/fast-experimental`, а не про низкоуровневые типы.

### 30.22. Version note

Этот раздел вводит **новую версию спеки** по архитектуре режимов и точности.

Его следует считать архитектурным redesign уровня v0.3.

