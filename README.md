# TDMD Project — Documentation Package

**Version:** 2.1 (master spec) + 1.0 (all module SPECs)
**Status:** documentation-complete, ready for M0 implementation
**Date:** 2026-04-16

---

## Что это

Это полный документационный пакет проекта **TDMD** — standalone программы молекулярной динамики на основе метода декомпозиции по времени (Time Decomposition, TD), описанного в диссертации В.В. Андреева (2007), с модернизацией под современные GPU-кластеры.

Пакет содержит всё необходимое для начала реализации: мастер-специю, 12 модульных SPEC-документов, playbook для AI-агентной разработки, и исходные материалы (диссертация + предыдущие версии ТЗ) как контекст.

**Что это НЕ:** ещё не код. Это фаза «спецификация завершена, реализация начинается».

---

## Кому читать что

### Первое знакомство (30 минут)

1. `TDMD_Engineering_Spec.md` — Часть I (§1–§4): что такое TDMD, зачем он нужен, когда TD выигрывает над SD;
2. `TDMD_Engineering_Spec.md` — Приложение A: соответствие с диссертацией.

Этого достаточно, чтобы понять суть проекта и его нишу.

### Техлид / архитектор (2-3 часа)

1. `TDMD_Engineering_Spec.md` полностью — все 6 частей;
2. Три ключевых модульных SPEC'а:
   - `docs/specs/scheduler/SPEC.md` — сердце TD;
   - `docs/specs/zoning/SPEC.md` — математика N_min, Hilbert 3D;
   - `docs/specs/perfmodel/SPEC.md` — когда TD реально выигрывает.

### Инженер реализации (по мере работы)

- `TDMD_Engineering_Spec.md` §5–§12 как reference;
- Соответствующий `docs/specs/<my_module>/SPEC.md`;
- `docs/development/claude_code_playbook.md` — правила работы (особенно если работаешь с AI-агентами).

### Пользователь (научный user)

- `TDMD_Engineering_Spec.md` Часть I и §14 (UX);
- `docs/specs/cli/SPEC.md` — как использовать CLI;
- `docs/specs/io/SPEC.md` §3 — формат config YAML.

### Reviewer / историк идей

- `docs/_sources/dissertation_andreev_2007.docx` — оригинальная диссертация;
- `docs/_sources/spec_v0_1_draft.md` — первая версия ТЗ;
- `docs/_sources/spec_v1_0_reframed.md` — промежуточная версия;
- `TDMD_Engineering_Spec.md` Приложение C — change log с обоснованием изменений.

---

## Структура

```
tdmd_project/
│
├── README.md                          # этот файл
├── TDMD_Engineering_Spec.md           # ⭐ главный документ проекта (master spec v2.1)
│
└── docs/
    ├── development/
    │   └── claude_code_playbook.md    # правила работы с AI-агентами
    │
    ├── specs/                          # 13 модульных SPEC'ов
    │   ├── scheduler/SPEC.md          # TD scheduler, зонный DAG, certificates
    │   ├── zoning/SPEC.md             # разбиение на зоны, Hilbert 3D
    │   ├── perfmodel/SPEC.md          # модель производительности TD×SD
    │   ├── neighbor/SPEC.md           # cell grid, neighbor list, skin
    │   ├── potentials/SPEC.md         # Morse, EAM, MEAM, SNAP
    │   ├── comm/SPEC.md               # MPI/NCCL/NVSHMEM backends
    │   ├── state/SPEC.md              # AtomSoA, box, species, identity
    │   ├── integrator/SPEC.md         # Velocity-Verlet, NVT, NPT
    │   ├── runtime/SPEC.md            # SimulationEngine, lifecycle
    │   ├── io/SPEC.md                 # YAML, LAMMPS data, dumps
    │   ├── telemetry/SPEC.md          # metrics, NVTX, logs
    │   ├── cli/SPEC.md                # user-facing commands
    │   └── verify/SPEC.md             # ⭐ cross-module scientific validation
    │
    └── _sources/                       # исходные материалы (read-only context)
        ├── dissertation_andreev_2007.docx
        ├── deep_research_report.md
        ├── spec_v0_1_draft.md
        └── spec_v1_0_reframed.md
```

---

## Что делает TDMD

В одном абзаце (каноническая формулировка из §3.1 master spec):

> **TDMD — это GPU-first standalone MD-движок, ориентированный на дорогие локальные many-body и ML-потенциалы (EAM, MEAM, SNAP, MLIAP, PACE), использующий time decomposition для снижения требований к межпроцессорной коммуникации и повышения масштабируемости на системах со сравнительно медленными каналами.**

### Главная идея (из диссертации)

Если радиус потенциала взаимодействия значительно меньше размера модели (что всегда верно для MD), то **области, удалённые более чем на диаметр потенциала, не влияют друг на друга в пределах одного шага интегрирования**. Это значит, что разные области можно продвигать **на разных временных шагах** одновременно. Получается time decomposition (TD) — параллелизм не по пространству, а по времени.

### Современное развитие

TDMD реализует TD как first-class принцип + добавляет:

- **two-level decomposition** (TD внутри subdomain через NVLink + SD между subdomain'ами через IB) для multi-node HPC;
- **K-batched pipeline** — K последовательных шагов перед передачей, снижает bandwidth в K раз (формула (51) диссертации);
- **Hilbert 3D zoning** вместо линейной нумерации — снижает `N_min` с O(N²) до O(N^(2/3));
- **Safety certificates** — формализация условий корректности для каждой зоны;
- **BuildFlavor × ExecProfile** — compile-time numeric semantics + runtime policy profile (вместо свободного runtime-переключения типов).

---

## Ключевые документы

### `TDMD_Engineering_Spec.md` (master spec, 1618 строк)

Главный источник истины. Структура:

- **Часть I** — метод и почему он стоит реализации (§1–§4a);
- **Часть II** — алгоритм и архитектура (§5–§11);
- **Часть III** — интерфейсы и реализация (§12);
- **Часть IV** — верификация и тест-план (§13);
- **Часть V** — план реализации (§14–§15) — 8 milestones (M0–M8);
- **Часть VI** — приложения (traceability с диссертацией, assumptions, change log).

### `docs/development/claude_code_playbook.md` (806 строк)

Правила работы с AI-агентами (Claude Code). 8 канонических ролей (Architect, Core Runtime Engineer, Scheduler Engineer, ...), task prompt templates, structured report format, auto-reject conditions.

### 12 модульных SPEC'ов

Каждый покрывает один модуль с единой структурой:
1. Purpose / scope / границы;
2. Public interface (C++/CUDA);
3. Алгоритмы и формулы;
4. Определение policies;
5. Tests (unit / property / differential / determinism);
6. Telemetry hooks;
7. Roadmap alignment;
8. Open questions.

---

## Roadmap (из master spec §14)

| Milestone | Duration | Deliverable |
|---|---|---|
| M0 | 4 нед. | Process & skeleton: repo, CI, CMake, templates |
| M1 | 6 нед. | CPU Reference MD without TD: AtomSoA, Verlet, Morse, YAML |
| M2 | 6 нед. | EAM CPU + perfmodel skeleton + lj unit support |
| M3 | 4 нед. | Zoning planner (Linear1D / Decomp2D / Hilbert3D) |
| **M4** | 8 нед. | **Deterministic TD scheduler** — первый работающий TD |
| **M5** | 6 нед. | **Multi-rank TD + anchor-test** (воспроизведение диссертации) |
| M6 | 8 нед. | GPU path, CUDA kernels, single-GPU |
| **M7** | 10 нед. | **Two-level Pattern 2** (TD × SD hybrid) — production multi-node |
| **M8** | 6 нед. | **SNAP proof-of-value** — flagship ML benchmark |

Total v1 (M0–M8): ~58 недель ≈ 14 месяцев команды 3-5 инженеров.

### Критические gates

- **M5:** anchor-test — численное воспроизведение эксперимента Андреева Al FCC 10⁶ с погрешностью ≤ 10%. Без прохождения этого теста проект не имеет права называть себя "TD method by Andreev";
- **M8:** TDMD либо обгоняет LAMMPS SNAP ≥ 20% на ≥ 8 ranks, либо документирует честно почему не обгоняет.

---

## Стек реализации

- **Language:** C++17/20 + CUDA 12.x;
- **Build system:** CMake 3.25+;
- **GPU:** NVIDIA A100/H100 primary target в v1;
- **MPI:** OpenMPI 4.1+ / MVAPICH2-GDR / Spectrum MPI;
- **Dependencies:** NCCL, HDF5, yaml-cpp, Catch2 (tests);
- **LAMMPS:** external oracle для differential validation (не runtime dependency).

---

## Сборка (M0+ status: skeleton only)

M0 (Process & skeleton) находится в процессе. Декомпозиция на 7 PR-size задач и execution plan — в [`docs/development/m0_execution_pack.md`](docs/development/m0_execution_pack.md).

Формальные build instructions появятся после T0.4 (root CMake) в [`docs/development/build_instructions.md`](docs/development/build_instructions.md). Пока — quick reference по окружению:

| Компонент | Minimal version | Примечание |
|---|---|---|
| OS | Linux (Ubuntu 24.04 LTS) | Dev baseline |
| C++ compiler | GCC 13+ / Clang 17+ | C++20 required |
| CMake | 3.25+ | |
| CUDA toolkit | **12.8+** | Обязательно для `sm_120` (Blackwell consumer / RTX 50). `sm_80` A100 и `sm_90` H100 тоже supported |
| Python | 3.10+ | Для `pre-commit`, build helpers |
| Ninja | 1.11+ | Рекомендуемый CMake generator |
| MPI | OpenMPI 4.1+ | Нужен с M5+ |
| LAMMPS | `stable_22Jul2025_update4` | Git submodule в `verify/third_party/lammps/`; см. [`verify/third_party/lammps_README.md`](verify/third_party/lammps_README.md) |

### Quick start (TDMD build)

```bash
# Клонирование с submodule
git clone --recursive https://github.com/slava8519/tdmd_V2.git
cd tdmd_V2

# Сборка TDMD (sm_120 RTX 5080, CUDA 12.8+)
cmake -B build --preset default
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

На CUDA 12.6 (sm_120 ещё не поддерживается): используйте preset `default-sm89` (см. [`docs/development/build_instructions.md`](docs/development/build_instructions.md)).

### LAMMPS oracle (для differential-валидации, M1+)

```bash
# Однократно, ~15-30 мин:
git submodule update --init --depth 1 verify/third_party/lammps
tools/build_lammps.sh
tools/lammps_smoke_test.sh      # sanity check
```

Для CUDA 12.6: `TDMD_LAMMPS_CUDA_ARCH=sm_89 tools/build_lammps.sh`.

Статус задач M0: см. GitHub issues / PRs с тегом `milestone:M0`.

---

## Связь документов (dependency graph)

```
                    TDMD_Engineering_Spec.md  (master)
                              │
                              ├───► claude_code_playbook.md
                              │          (правила разработки)
                              │
                              └───► 12 × docs/specs/<module>/SPEC.md
                                          (контракты модулей)
                                     │
                                     └───► будущий исходный код:
                                              src/<module>/
                                              tests/<module>/
```

**Порядок истины** (из master spec §18):

1. `TDMD_Engineering_Spec.md` (этот документ — master);
2. `docs/specs/<module>/SPEC.md` (модульные контракты);
3. Execution pack конкретного milestone'а (будет создан на M0);
4. Код.

Противоречие между кодом и спекой — bug кода. Противоречие между модульным SPEC'ом и мастер-специей — bug документации, требует SPEC delta.

---

## Известные open questions

Зафиксированы в master spec §B.2 и в соответствующих модульных SPEC'ах. Критичные:

1. **Triclinic box в v1** — поддерживать или отложить post-v1?
2. **Python API** — pybind11 в v1 или только CLI?
3. **Long-range electrostatics** — M9 или позже?
4. **Anchor-test hardware equivalence** — как нормализовать результаты 2007 года на современное железо?
5. **KIM OpenKIM integration** — стратегический вопрос экосистемы.
6. **Auto-K policy** — алгоритм автоподбора pipeline depth.

Эти вопросы не блокируют старт реализации, но требуют решений в процессе.

---

## Контакт и эволюция документации

Эволюция документа — через explicit change log (Приложение C master spec). Любые изменения:

1. **Interface change** → сначала SPEC delta, потом код (playbook §9.1);
2. **Architectural change** → мастер-спец update + проверка consistency модульных SPEC'ов;
3. **Module internals** → только модульный SPEC update.

---

## Что делать дальше

**Если начинается реализация:**

1. Создать git repo с этой структурой как отправной точкой;
2. Начать M0 (bootstrap): CMake skeleton, CI pipelines, pre-commit hooks;
3. Параллельно — детальный issue tracker из backlog (master spec §14 milestones разбить на PR-size tasks);
4. Первый assignment для Claude Code агента — по шаблону из playbook §3.1.

**Если нужно что-то расширить в спеке:**

- Missing SPEC: `analysis/` (post-processing) или `interop/` как отдельный модуль, если понадобится;
- Расширение master spec: только через change log с обоснованием;
- Новый вопрос в Open Questions — добавить в соответствующий SPEC.

---

*Документационный пакет TDMD, версия 2.1, дата: 2026-04-16.*
