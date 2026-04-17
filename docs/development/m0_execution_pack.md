# M0 Execution Pack

**Document:** `docs/development/m0_execution_pack.md`
**Status:** draft, awaiting human review
**Parent:** `TDMD_Engineering_Spec.md` §14 (M0), `docs/development/claude_code_playbook.md` §3
**Milestone:** M0 — Process & skeleton (4 нед.)
**Created:** 2026-04-17
**Author:** Architect / Spec Steward role (Claude Opus 4.7)

---

## 0. Purpose

Этот документ декомпозирует milestone **M0** master spec'а на **7 PR-size задач**, каждая сформулирована по каноническому шаблону `claude_code_playbook.md` §3.1 и готова к прямому назначению Claude Code-агенту.

Документ — **process artifact**, не SPEC delta. Не модифицирует контракты, не меняет hierarchy of truth (master spec → module SPEC → код). Если в процессе реализации M0 будет обнаружен gap в спеке — оформляется отдельной SPEC delta по playbook §9.1, не в этом packs.

После успешного закрытия всех 7 задач и выполнения acceptance gate (§5 этого документа) — milestone M0 считается завершённым; execution pack для M1 создаётся как новый аналогичный документ.

---

## 1. Decisions log (зафиксировано до старта T0.1)

| # | Решение | Значение | Rationale / источник |
|---|---|---|---|
| **D-M0-1** | LICENSE | `BSD-3-Clause` | Научный стандарт; совместим с LAMMPS-as-submodule (LAMMPS остаётся под GPLv2 внутри submodule, основной TDMD source — BSD). Approved by human 2026-04-17. |
| **D-M0-2** | Стартовые 3 модуля | `state/`, `neighbor/`, `integrator/` | First M1 deliverables (master spec §14 M1); демонстрируют compute-path tier из INDEX.md. |
| **D-M0-3** | CUDA в CI | Полная компиляция + запуск | Self-hosted runner на dev-машине с RTX 5080. См. §2 environment. |
| **D-M0-4** | C++ standard | C++20 | Concepts/ranges нужны в scheduler (M4); единый стандарт без миграции в середине проекта. |
| **D-M0-5** | Pre-commit framework | `pre-commit.com` | Python-based mainstream; версионируется через `.pre-commit-config.yaml`. |
| **D-M0-6** | LAMMPS submodule | `verify/third_party/lammps/`, GPU build, packages: `MANYBODY`, `MEAM`, `ML-SNAP`, `MOLECULE`, `KSPACE`, `EXTRA-PAIR`, `EXTRA-DUMP` | Готов для M1+ differential gates без rebuild; user explicitly requested GPU build с full v1 packages. Pinning policy в T0.7. |

---

## 2. Глобальные параметры окружения

Эти параметры — common context для всех 7 задач. Зафиксированы здесь, чтобы не повторять в каждом task prompt.

| Параметр | Значение | Примечание |
|---|---|---|
| OS | Linux (Ubuntu 24.04 LTS assumed) | Dev-машина пользователя |
| C++ compiler | GCC 13+ или Clang 17+ | C++20 conformance required |
| CMake | 3.25+ | Master spec §15.2 minimum |
| CUDA toolkit | **12.8+** | Required для `sm_120` (Blackwell consumer) |
| Python | 3.10+ | Для `pre-commit`, build helper scripts |
| GPU primary target | `sm_120` (RTX 5080, Blackwell consumer) | Verify через `nvidia-smi --query-gpu=compute_cap --format=csv` |
| GPU future targets | `sm_80` (A100), `sm_90` (H100) | Compile-time multi-arch когда появится cluster доступ |
| MPI | OpenMPI 4.1+ | Не критично в M0; install для готовности к M5 |
| Test framework | Catch2 v3 | FetchContent, header-only |
| Build generator | Ninja | Faster чем Make для incremental builds |

---

## 3. Suggested PR order

Линейная последовательность для одного агента (последовательная отладка CI и pre-commit):

```
T0.1 (skeleton) ──► T0.2 (templates) ──┐
                ──► T0.3 (lint+pre-commit) ──┐
                ──► T0.4 (root CMake) ──┬──► T0.5 (3 modules) ──┐
                                         │                       ├──► T0.6 (CI Pipeline A)
                                         └──► T0.7 (LAMMPS) ─────┘
```

**Параллельный режим (если несколько агентов):** после T0.1 lands — `{T0.2, T0.3, T0.4}` параллельно; после T0.4 — `{T0.5, T0.7}` параллельно; T0.6 последней.

**Estimated effort (single agent, sequential):** ~3 недели. T0.7 (LAMMPS first build) — самый длинный (~3-5 дней включая отладку GPU build на sm_120).

---

## 4. Tasks

### T0.1 — Repository skeleton

```
# TDMD Task: Repository skeleton

## Context
- Project: TDMD
- Master spec: TDMD Engineering Spec v2.5
- Milestone: M0
- Role: Architect / Spec Steward

## Goal
Создать top-level каталоги monorepo, базовые инфраструктурные файлы (LICENSE, .gitignore, CITATION.cff, CONTRIBUTING.md), без CMake и без кода. Цель — все последующие задачи имеют корректные пути для размещения артефактов.

## Scope
- [included] Каталоги: `src/`, `tests/`, `benchmarks/`, `tools/`, `examples/`, `ci/`, `cmake/`, `docs/development/templates/`, `verify/third_party/`
- [included] `.gitignore` — покрывает: CMake build artifacts (`build/`, `build_*/`, `CMakeCache.txt`, `CMakeFiles/`), IDE (`.vscode/`, `.idea/`), Python (`__pycache__/`, `*.pyc`, `.venv/`, `venv/`), OS (`.DS_Store`, `Thumbs.db`), pre-commit (`.pre-commit-cache/`), CUDA (`*.cubin`, `*.fatbin`)
- [included] `LICENSE` — BSD-3-Clause, copyright "(c) 2026, TDMD Project Contributors"
- [included] `CITATION.cff` — citation в CFF v1.2.0 формате; cite Andreev 2007 dissertation как foundation, TDMD project как implementation
- [included] `CONTRIBUTING.md` — короткий, ссылается на playbook §1 (mandatory rules) и §3 (task template) как канонические процедуры
- [included] `.gitattributes` — Git LFS для `*.lammps`, `*.h5`, `*.npz`, `*.bin` в `verify/` и `benchmarks/` (для будущих reference data)
- [included] `.editorconfig` — base settings (UTF-8, LF, no trailing whitespace, indent_style=space, indent_size=4 для C++ / 2 для YAML/JSON)
- [included] `.gitkeep` в каждой пустой директории чтобы сохранить структуру в git
- [included] Update top-level `README.md` — добавить секцию "Building TDMD (M0+ status: skeleton only)" со ссылкой на этот pack

## Out of scope
- [excluded] CMake (это T0.4)
- [excluded] Templates content (это T0.2)
- [excluded] Code style configs (это T0.3)
- [excluded] Submodule registration (это T0.7)
- [excluded] Module subdirectories с CMake (это T0.5)

## Mandatory invariants
- LICENSE распознаётся GitHub linguist'ом (отображается на repo page как "BSD-3-Clause License")
- `.gitignore` покрывает все common build artifacts (тест: `cmake -B build` затем `git status` показывает clean)
- Структура соответствует master spec §14 M0 каталогам

## Required files
- `.gitignore` — comprehensive
- `LICENSE` — BSD-3-Clause text от opensource.org/license/bsd-3-clause
- `CITATION.cff` — CFF v1.2.0; references: Andreev V.V. 2007 dissertation (URL/citation TBD)
- `CONTRIBUTING.md` — ссылка на playbook
- `.gitattributes` — LFS rules
- `.editorconfig` — base
- `src/.gitkeep`, `tests/.gitkeep`, `benchmarks/.gitkeep`, `tools/.gitkeep`, `examples/.gitkeep`, `ci/.gitkeep`, `cmake/.gitkeep`, `docs/development/templates/.gitkeep`, `verify/third_party/.gitkeep`
- `README.md` — minor edit (add Building section stub)

## Required tests
- [structure] `find . -type d -maxdepth 2 | sort` — сравнить с expected list
- [license] `gh api repos/slava8519/tdmd_V2 --jq '.license.spdx_id'` returns `BSD-3-Clause` после push

## Expected artifacts
- Single PR с структурой
- Verified LICENSE recognition в GitHub UI
- Updated README.md

## Acceptance criteria
- [ ] Все 9 директорий существуют (см. Required files)
- [ ] `LICENSE` recognized как BSD-3-Clause GitHub'ом
- [ ] `.gitignore` тестируется: `mkdir build && touch build/test.o && git status` — не показывает `build/`
- [ ] `CITATION.cff` validates (online cff-validator или `pip install cffconvert && cffconvert --validate`)
- [ ] PR review approved by human

## Hints
- BSD-3-Clause text: https://opensource.org/license/bsd-3-clause (copy verbatim, заменить year/holder)
- CFF spec: https://citation-file-format.github.io
- Не забудь установить .gitattributes ДО первого LFS commit'а — иначе LFS не подхватит historical files
```

---

### T0.2 — Templates для SPEC.md / TESTPLAN.md / Module README

```
# TDMD Task: Canonical templates

## Context
- Project: TDMD
- Master spec: TDMD Engineering Spec v2.5
- Playbook: §3.1 (task template), §6.1 (new module session structure)
- Milestone: M0
- Role: Architect / Spec Steward

## Goal
Создать канонические шаблоны для трёх типов документов модуля: `SPEC.md` (контракт), `TESTPLAN.md` (тестовая стратегия), `README.md` (обзор и границы). Шаблоны используются Architect'ом при bootstrap'е любого нового модуля; гарантируют единый формат через все 13+ модулей.

## Scope
- [included] `docs/development/templates/SPEC_template.md` — структура из 8 секций по аналогии с существующими `docs/specs/*/SPEC.md` (purpose / public interface / algorithms / policies / tests / telemetry / roadmap / open questions)
- [included] `docs/development/templates/TESTPLAN_template.md` — секции: test pyramid layers, unit / property / differential / determinism / performance, threshold registry refs, fixture requirements, CI pipeline mapping
- [included] `docs/development/templates/MODULE_README_template.md` — секции: purpose, scope boundaries, public API summary, dependencies, build instructions, examples, known limitations, telemetry hooks summary
- [included] `docs/development/templates/README.md` — meta-document: какой шаблон когда использовать, чем отличаются, ссылки на existing examples
- [included] Каждый template содержит inline комментарии `<!-- TODO: ... -->` объясняющие что вписать
- [included] Каждый template ссылается на master spec и playbook соответствующими секциями

## Out of scope
- [excluded] Применение templates к существующим 13 module SPECs (они уже написаны в свободном формате; conformance — отдельная задача поздних milestones)
- [excluded] Pull request template (это T0.6, в `.github/`)

## Mandatory invariants
- Templates самосогласованы с playbook §6.1 (new module workflow)
- Каждый template — valid markdown (markdownlint pass)
- Templates на русском (matching project language; English code/identifiers сохраняются)

## Required files
- `docs/development/templates/SPEC_template.md` (~150 lines)
- `docs/development/templates/TESTPLAN_template.md` (~80 lines)
- `docs/development/templates/MODULE_README_template.md` (~60 lines)
- `docs/development/templates/README.md` (~40 lines, meta)

## Required tests
- [lint] `markdownlint docs/development/templates/*.md` — pass
- [structural] templates содержат все необходимые секции (manual review checklist в PR description)

## Expected artifacts
- 4 файла в `docs/development/templates/`
- PR description содержит manual checklist подтверждающий все секции на месте

## Acceptance criteria
- [ ] 4 templates созданы
- [ ] markdownlint pass
- [ ] Каждый template ссылается на playbook + master spec
- [ ] Inline TODO комментарии присутствуют для каждой секции
- [ ] PR review approved by human

## Hints
- За референс структуры SPEC взять `docs/specs/scheduler/SPEC.md` (best-developed) и `docs/specs/zoning/SPEC.md`
- TESTPLAN ещё не существует ни для одного модуля — структуру вывести из master spec §13 (test pyramid 6 layers) + verify/SPEC.md threshold registry
- Не делать templates слишком жёсткими — оставить место для module-specific секций под `## N+1. Module-specific`
```

---

### T0.3 — Code style + pre-commit hooks

```
# TDMD Task: clang-format, clang-tidy, pre-commit hooks

## Context
- Project: TDMD
- Master spec: §15 (engineering methodology), §D.16 (`__restrict__` policy)
- Playbook: §5.1 (auto-reject patterns), §5.4 (performance violations)
- Milestone: M0
- Role: Core Runtime Engineer

## Goal
Установить enforce'имый baseline code style (clang-format) и static analysis (clang-tidy) с автоматическим запуском через pre-commit hooks. Гарантирует, что любой commit проходит базовые гейты до того как попасть в CI.

## Scope
- [included] `.clang-format` — based on Google style; column_limit=100; AccessModifierOffset=-2; SortIncludes=CaseSensitive
- [included] `.clang-tidy` — checks: `bugprone-*`, `cppcoreguidelines-*` (с opt-out для performance-critical exceptions), `modernize-*`, `performance-*`, `readability-*`, `cert-*`. Exclude: `cppcoreguidelines-avoid-magic-numbers`, `readability-magic-numbers`, `modernize-use-trailing-return-type`
- [included] `.pre-commit-config.yaml` с hooks:
  - `pre-commit-hooks/end-of-file-fixer`
  - `pre-commit-hooks/trailing-whitespace`
  - `pre-commit-hooks/check-yaml`
  - `pre-commit-hooks/check-merge-conflict`
  - `pre-commit-hooks/check-added-large-files` (max 500KB; больше — через LFS)
  - `pre-commit/mirrors-clang-format` v17+ (для C++ и CUDA)
  - `markdownlint-cli2` для `.md` (config: `.markdownlint.yaml`)
  - `cmake-format` для `CMakeLists.txt` и `*.cmake`
  - Local hook `tdmd-check-restrict` — STUB (Python script, exit 0); реальная implementation в M0+ как clang-tidy custom check позже
- [included] `tools/lint/run_clang_format.sh` — wrapper для batch формата всего src/
- [included] `tools/lint/run_clang_tidy.sh` — wrapper для batch lint всего src/ (требует compile_commands.json)
- [included] `docs/development/code_style.md` — coding conventions (~100 lines): naming (`snake_case` для variables, `PascalCase` для классов, `UPPER_SNAKE` для constants), header guards (`#pragma once`), `__restrict__` policy ссылка на §D.16, error handling philosophy
- [included] `.markdownlint.yaml` — relaxed config (line-length disabled, MD013/MD024 disabled)
- [included] `.cmake-format.yaml` — base config

## Out of scope
- [excluded] Реализация custom clang-tidy check `tdmd-missing-restrict` — это требует C++ AST plugin (LLVM dev headers); deferred до M2-M3 когда будут реальные hot kernels. На M0 — только STUB Python hook
- [excluded] CI integration этих hooks (это T0.6 — pre-commit запускается в CI через `pre-commit run --all-files`)
- [excluded] Applying формата к existing markdown spec docs — separate cleanup PR

## Mandatory invariants
- `pre-commit run --all-files` после установки — pass на clean tree (никаких изменений)
- Intentional bad commit (trailing whitespace, mis-indented C++) — blocked locally
- Custom hook `tdmd-check-restrict` STUB не блокирует существующий код

## Required files
- `.clang-format`
- `.clang-tidy`
- `.pre-commit-config.yaml`
- `.markdownlint.yaml`
- `.cmake-format.yaml`
- `tools/lint/run_clang_format.sh`
- `tools/lint/run_clang_tidy.sh`
- `tools/lint/check_restrict_stub.py` — Python script, returns 0 always; comment объясняет план
- `docs/development/code_style.md`

## Required tests
- [unit] `pre-commit run --all-files` — pass на чистом дереве
- [unit] Создать temp файл с `void foo() {  return; }` (extra space), `git add`, `git commit` — clang-format hook автоматически форматирует
- [unit] Создать temp `.md` с heading нарушающим правила — markdownlint catches

## Expected artifacts
- 9 config files / scripts
- `code_style.md` doc
- PR с screenshot или log проверок

## Acceptance criteria
- [ ] `pip install pre-commit && pre-commit install` локально работает
- [ ] `pre-commit run --all-files` pass
- [ ] Demonstration: bad commit blocked (record terminal session)
- [ ] PR review approved by human

## Hints
- Clang-format Google style template: `clang-format -style=google -dump-config > .clang-format` затем edit
- pre-commit + LFS interaction: убедиться что check-added-large-files не блокирует LFS-tracked файлы
- `cmake-format` ставится через `pip install cmakelang`
- CUDA `.cu` файлы тоже должны проходить clang-format (включить в include patterns)
```

---

### T0.4 — Root CMake + BuildFlavors stubs

```
# TDMD Task: Root CMake configuration with build flavors

## Context
- Project: TDMD
- Master spec: §7 (BuildFlavor × ExecProfile), §D (precision policy), §D.14 (build system integration), §D.16 (__restrict__)
- Milestone: M0
- Role: Core Runtime Engineer

## Goal
Создать root `CMakeLists.txt` и поддерживающие cmake-модули, реализующие пять `BuildFlavor`'ов из master spec §7.1 (как stub-targets — только compile flags, без implementation). Build configures и собирает `Fp64ReferenceBuild` на CPU + CUDA для sm_120 без ошибок.

## Scope
- [included] `CMakeLists.txt` (root):
  - `cmake_minimum_required(VERSION 3.25)`
  - `project(tdmd VERSION 0.1.0 LANGUAGES CXX CUDA)`
  - `set(CMAKE_CXX_STANDARD 20)`, `set(CMAKE_CXX_STANDARD_REQUIRED ON)`, `set(CMAKE_CXX_EXTENSIONS OFF)`
  - `set(CMAKE_CUDA_STANDARD 17)` (CUDA C++17 — nvcc 12.x cap)
  - `set(CMAKE_CUDA_ARCHITECTURES 120)` (sm_120 RTX 5080; configurable через `TDMD_CUDA_ARCHS` cache var)
  - `set(CMAKE_EXPORT_COMPILE_COMMANDS ON)` (для clang-tidy)
  - `option(TDMD_BUILD_CUDA "Enable CUDA targets" ON)`
  - `option(TDMD_BUILD_TESTS "Build unit tests" ON)`
  - `option(TDMD_BUILD_BENCHMARKS "Build benchmarks" OFF)`
  - `set(TDMD_BUILD_FLAVOR "Fp64ReferenceBuild" CACHE STRING "Active BuildFlavor")`
  - `set_property(CACHE TDMD_BUILD_FLAVOR PROPERTY STRINGS "Fp64ReferenceBuild" "Fp64ProductionBuild" "MixedFastBuild" "MixedFastAggressiveBuild" "Fp32ExperimentalBuild")`
  - `include(cmake/CompilerWarnings.cmake)`
  - `include(cmake/BuildFlavors.cmake)`
  - `include(cmake/Dependencies.cmake)` (Catch2 via FetchContent)
  - `enable_testing()`
  - `add_subdirectory(src)`
  - `if(TDMD_BUILD_TESTS) add_subdirectory(tests) endif()`
- [included] `cmake/CompilerWarnings.cmake`:
  - GCC/Clang: `-Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wcast-align -Wunused -Woverloaded-virtual -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion -Wformat=2`
  - Treat warnings as errors в CI (controlled via `TDMD_WARNINGS_AS_ERRORS` option, default OFF локально, ON в CI)
  - CUDA: `--Werror=all-warnings` для `Fp64ReferenceBuild`
- [included] `cmake/BuildFlavors.cmake`:
  - Function `tdmd_apply_build_flavor(target)` устанавливает compile definitions и flags per flavor
  - `Fp64ReferenceBuild`: `-DTDMD_FLAVOR_FP64_REFERENCE`, `-fno-fast-math`, `-ffp-contract=off`, CUDA `--fmad=false`
  - `Fp64ProductionBuild`: `-DTDMD_FLAVOR_FP64_PRODUCTION`, allow FMA contraction
  - `MixedFastBuild`, `MixedFastAggressiveBuild`, `Fp32ExperimentalBuild`: STUBs (definitions present, message(STATUS "BuildFlavor X is stub in M0"))
- [included] `cmake/Dependencies.cmake`:
  - `include(FetchContent)`
  - Catch2 v3.5.0+ via FetchContent (pinned tag)
  - Future: yaml-cpp, HDF5 (deferred до M1+)
- [included] `cmake/FindCUDAArchHelper.cmake` — helper detecting `compute_cap` через `nvidia-smi` если `TDMD_CUDA_ARCHS=auto`
- [included] `CMakePresets.json`:
  - `default` (CPU + CUDA, Release, sm_120)
  - `debug` (Debug, ASan, UBSan)
  - `cpu-only` (no CUDA, для CI lint job)
  - `release` (Release, no debug info)
- [included] `src/CMakeLists.txt` — пустой (T0.5 заполнит); `# Empty placeholder; modules added in T0.5`
- [included] `tests/CMakeLists.txt` — пустой; `# Empty placeholder; tests added in T0.5`
- [included] `docs/development/build_instructions.md` — short guide: how to configure, build, test; preset usage; common errors

## Out of scope
- [excluded] Module sources (T0.5)
- [excluded] LAMMPS integration (T0.7)
- [excluded] Real BuildFlavor implementations (только stubs); полная реализация — M2+
- [excluded] cmake/FindLammps.cmake (это T0.7)

## Mandatory invariants
- `cmake -B build --preset default && cmake --build build` succeeds на dev-машине (RTX 5080, CUDA 12.8+)
- `cmake -B build_cpu --preset cpu-only && cmake --build build_cpu` succeeds (no CUDA required)
- `ctest --test-dir build` returns 0 (no tests yet, but не error)
- `compile_commands.json` генерируется в build dir (для T0.3 clang-tidy)

## Required files
- `CMakeLists.txt` (root, ~80 lines)
- `cmake/CompilerWarnings.cmake`
- `cmake/BuildFlavors.cmake`
- `cmake/Dependencies.cmake`
- `cmake/FindCUDAArchHelper.cmake`
- `CMakePresets.json`
- `src/CMakeLists.txt` (placeholder)
- `tests/CMakeLists.txt` (placeholder)
- `docs/development/build_instructions.md`

## Required tests
- [build] `cmake --preset default && cmake --build build --parallel` — exit 0
- [build] `cmake --preset cpu-only && cmake --build build_cpu --parallel` — exit 0 (без CUDA toolkit available — для CI verification)
- [build] `cmake --preset debug --build build_debug` — exit 0
- [test] `ctest --test-dir build --output-on-failure` — exit 0
- [smoke] `nvcc --version` совпадает с CMake-detected CUDA version

## Expected artifacts
- 8 cmake-related файлов
- `build_instructions.md`
- Verified build на dev-машине (terminal log в PR)

## Acceptance criteria
- [ ] Configure + build succeed для всех 4 presets на dev-машине
- [ ] CUDA arch detected как 120 для RTX 5080
- [ ] `compile_commands.json` exists в build dir
- [ ] Catch2 v3 fetched и доступен для linking
- [ ] PR review approved by human

## Hints
- CMake 3.25 `set(CMAKE_CUDA_ARCHITECTURES native)` — convenient но не reproducible; используй explicit `120`
- Catch2 v3 через FetchContent: `FetchContent_Declare(Catch2 GIT_REPOSITORY https://github.com/catchorg/Catch2.git GIT_TAG v3.5.3)`
- Если CMake не находит CUDA — `export CUDACXX=/usr/local/cuda-12.8/bin/nvcc` перед configure
- ASan + CUDA не дружат полностью — disable ASan для CUDA targets в `debug` preset
- CMakePresets.json schema: https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html
```

---

### T0.5 — Three module skeletons (state, neighbor, integrator)

```
# TDMD Task: Empty module skeletons — state, neighbor, integrator

## Context
- Project: TDMD
- Master spec: §8 (architecture, ownership boundaries), §14 (M0 artifact gate)
- Module specs: `docs/specs/state/SPEC.md`, `docs/specs/neighbor/SPEC.md`, `docs/specs/integrator/SPEC.md`
- Milestone: M0
- Role: Core Runtime Engineer

## Goal
Создать пустые скелеты трёх модулей (`state/`, `neighbor/`, `integrator/`) с per-module CMakeLists.txt, минимальными header'ами с forward declarations, stub source файлами, smoke unit тестом. Цель — закрыть M0 artifact gate ("3 empty modules с рабочим CMake") и подготовить почву для M1 наполнения.

## Scope

Для каждого из трёх модулей (`state`, `neighbor`, `integrator`):

- [included] `src/<module>/CMakeLists.txt`:
  - `add_library(tdmd_<module> STATIC <sources>)`
  - `target_include_directories(tdmd_<module> PUBLIC include)`
  - `target_compile_features(tdmd_<module> PUBLIC cxx_std_20)`
  - `tdmd_apply_build_flavor(tdmd_<module>)` (вызов из BuildFlavors.cmake)
  - `target_link_libraries(tdmd_<module> PUBLIC <deps>)` — пока без deps
  - `add_library(tdmd::<module> ALIAS tdmd_<module>)` — для consistent namespace

- [included] `src/<module>/include/tdmd/<module>/<primary_class>.hpp`:
  - state: `atom_soa.hpp` с `class AtomSoA { /* TODO M1 */ };`
  - neighbor: `cell_grid.hpp` с `class CellGrid { /* TODO M1 */ };`
  - integrator: `integrator.hpp` с `class IIntegrator { /* TODO M1 */ };` (interface)
  - Каждый header — `#pragma once`, namespace `tdmd::<module>`
  - Header содержит только class declaration с TODO comment, без members

- [included] `src/<module>/<primary_class>.cpp` — пустая реализация конструктора (если вообще нужно), `// TODO M1`

- [included] `src/<module>/README.md` — module overview based на MODULE_README_template.md (T0.2):
  - Покажет: purpose (1 параграф), scope boundaries из docs/specs/<module>/SPEC.md, public API summary (что будет в M1), dependencies, ссылка на полный SPEC

- [included] `tests/<module>/CMakeLists.txt`:
  - `add_executable(test_<module> test_<primary_class>.cpp)`
  - `target_link_libraries(test_<module> PRIVATE tdmd::<module> Catch2::Catch2WithMain)`
  - `add_test(NAME test_<module> COMMAND test_<module>)`

- [included] `tests/<module>/test_<primary_class>.cpp` — smoke test:
  - `TEST_CASE("<class> is constructible")` — instantiate class, assert no throw
  - 1-2 простых assertions для placeholder behavior

- [included] Update `src/CMakeLists.txt`:
  ```
  add_subdirectory(state)
  add_subdirectory(neighbor)
  add_subdirectory(integrator)
  ```

- [included] Update `tests/CMakeLists.txt`:
  ```
  add_subdirectory(state)
  add_subdirectory(neighbor)
  add_subdirectory(integrator)
  ```

## Out of scope
- [excluded] Реальная реализация AtomSoA / CellGrid / Integrator — M1 work
- [excluded] CUDA-side header'ы (`*.cuh`) — добавляются когда GPU path появится в M6
- [excluded] Other modules (scheduler, zoning, etc.) — добавляются по мере implementation milestones
- [excluded] `docs/specs/<module>/TESTPLAN.md` — отдельная задача в M1+

## Mandatory invariants
- Каждый модуль строится независимо: `cmake --build build --target tdmd_state` succeeds
- Stub класс не нарушает ownership boundaries из master spec §8.2 (никакой mutual coupling между state, neighbor, integrator на уровне header'ов в M0)
- Smoke test действительно запускается и passes (3/3 в `ctest`)
- Naming совпадает с module spec: `AtomSoA` (не `AtomData`), `CellGrid`, `IIntegrator`

## Required files
Per module (× 3 = 18 files плюс 2 updated):
- `src/<module>/CMakeLists.txt`
- `src/<module>/include/tdmd/<module>/<class>.hpp`
- `src/<module>/<class>.cpp`
- `src/<module>/README.md`
- `tests/<module>/CMakeLists.txt`
- `tests/<module>/test_<class>.cpp`

Updated:
- `src/CMakeLists.txt` (add 3 subdirectory)
- `tests/CMakeLists.txt` (add 3 subdirectory)

## Required tests
- [build] `cmake --build build --parallel` succeeds
- [unit] `ctest --test-dir build --output-on-failure` shows 3 passed
- [unit] Каждый smoke test instantiates primary class без throw
- [structure] `find src -name "CMakeLists.txt"` returns: `src/CMakeLists.txt`, `src/state/CMakeLists.txt`, `src/neighbor/CMakeLists.txt`, `src/integrator/CMakeLists.txt`

## Expected artifacts
- 18 created files + 2 updated
- ctest output showing 3/3 passed
- PR description с tree output

## Acceptance criteria
- [ ] 3 модуля строятся
- [ ] 3 smoke тесты passes
- [ ] README.md в каждом модуле ссылается на correctly на `docs/specs/<module>/SPEC.md`
- [ ] Public API headers в `include/tdmd/<module>/` (не в module root) — для clean install layout позже
- [ ] PR review approved by human

## Hints
- Catch2 v3 macros: `TEST_CASE("name")`, `SECTION("name")`, `REQUIRE(expr)`, `CHECK(expr)`
- Header layout `include/tdmd/<module>/<class>.hpp` — позже install будет `<prefix>/include/tdmd/<module>/<class>.hpp` без префикс-смешивания с другими libs
- IIntegrator с `I` prefix — interface convention; реальная Velocity-Verlet реализация будет `VelocityVerletIntegrator` в M1
- Не вкладывать smarts в stubs — playbook §1.1 п.7 (no scope creep)
```

---

### T0.6 — CI Pipeline A (GitHub Actions: lint + build)

```
# TDMD Task: CI Pipeline A — lint + build на каждом PR

## Context
- Project: TDMD
- Master spec: §11 (CI integration), §15.1 (Spec-Driven TDD)
- Playbook: §8 (CI integration; merge gates A-F)
- Milestone: M0
- Role: Validation / Reference Engineer

## Goal
Активировать CI Pipeline A (master spec §11): на каждом PR — lint pass + build pass на CPU и CUDA. Pipelines B-F (unit / property / differential / performance / reproducibility) появятся в M1+ — их job'ы оставляем как stubs или просто не добавляем в M0.

## Scope
- [included] `.github/workflows/ci.yml` — multi-job workflow:
  - **Job `lint`** (ubuntu-latest, GitHub-hosted):
    - checkout
    - setup Python 3.10
    - `pip install pre-commit`
    - `pre-commit run --all-files`
    - markdownlint via container or action
  - **Job `build-cpu`** (ubuntu-latest, GitHub-hosted):
    - checkout
    - install GCC 13, Clang 17, CMake 3.25+, Ninja
    - `cmake --preset cpu-only`
    - `cmake --build build_cpu --parallel`
    - `ctest --test-dir build_cpu --output-on-failure`
    - matrix: compiler ∈ {gcc-13, clang-17}
  - **Job `build-cuda`** (self-hosted, label `gpu-rtx5080`):
    - checkout (с submodules: false — LAMMPS не нужен в этом job'е)
    - `cmake --preset default` (CUDA on, sm_120)
    - `cmake --build build --parallel`
    - `ctest --test-dir build --output-on-failure`
  - **Job `docs-lint`** (ubuntu-latest):
    - markdownlint всех `.md` файлов
    - link check (опционально — markdown-link-check action)
  - All jobs required for merge

- [included] `.github/CODEOWNERS` — placeholder:
  ```
  *           @slava8519
  /docs/specs @slava8519
  /docs/development @slava8519
  ```

- [included] `.github/PULL_REQUEST_TEMPLATE.md` — checklist согласованный с playbook §1.3 (pre-implementation report) и §4 (session report):
  - Pre-implementation report attached (link or inline)
  - Session report attached
  - SPEC deltas listed (or "none")
  - Tests added: unit / property / differential / determinism counts
  - Pipelines impacted: A B C D E F (check applicable)
  - Acceptance criteria met (link to task prompt)

- [included] `.github/ISSUE_TEMPLATE/`:
  - `bug_report.md` — short template
  - `task.md` — based on playbook §3.1 task template

- [included] `docs/development/ci_setup.md` — how to:
  - Set up self-hosted runner на dev-машине (RTX 5080, label `gpu-rtx5080`)
  - Install `actions-runner` service
  - Required system packages (CUDA 12.8+, GCC 13, CMake 3.25+, Ninja)
  - Troubleshooting common failures
  - Cache strategy для CI builds

- [included] `.github/workflows/scheduled-cuda-rebuild.yml` — weekly job rebuilding from scratch (catches submodule drift, CUDA toolkit updates)

## Out of scope
- [excluded] Pipelines B-F (Unit / Property / Differential / Performance / Reproducibility) — добавляются по мере появления соответствующего test infrastructure в M1+
- [excluded] Release automation (semantic versioning, changelog generation, artifact publishing) — post-v1
- [excluded] LAMMPS build в CI (это T0.7 + отдельный CI job в M1+)

## Mandatory invariants
- Все jobs обязательны (required status checks); failed CI блокирует merge
- Self-hosted runner работает в isolated workspace (не уносит state между runs)
- CI использует те же CMake presets что и developer (no CI-specific build flags)

## Required files
- `.github/workflows/ci.yml`
- `.github/workflows/scheduled-cuda-rebuild.yml`
- `.github/CODEOWNERS`
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/task.md`
- `docs/development/ci_setup.md`

## Required tests
- [smoke] PR open → CI triggers → all jobs run → all pass
- [smoke] Intentional failure (e.g. broken header in src/state/) → build job fails → merge blocked
- [smoke] Self-hosted runner picks up `build-cuda` job (verify через GitHub Actions UI logs)

## Expected artifacts
- 7 файлов
- Successful CI run на dummy PR (link in PR description)
- Self-hosted runner registered и visible в GitHub repo settings (screenshot)

## Acceptance criteria
- [ ] CI green на test PR
- [ ] All 4 jobs (lint, build-cpu × 2 compilers, build-cuda, docs-lint) green
- [ ] Required status checks set в GitHub repo settings (`main` branch protection)
- [ ] Dummy bad PR demonstrates CI catches the failure
- [ ] PR review approved by human

## Hints
- Self-hosted runner setup: GitHub Settings → Actions → Runners → New self-hosted runner. Linux x64. Run as systemd service.
- Cache CMake build dir между runs через `actions/cache` keyed на CMake config files hash (~30% speedup на rebuilds)
- Если CUDA build job падает на нехватке VRAM — RTX 5080 имеет 16GB; для smoke tests это с запасом
- ccache можно intergrated через `CMAKE_CXX_COMPILER_LAUNCHER=ccache` (опционально, но ускоряет CI на 50%+)
- Branch protection rules: Settings → Branches → main → Add rule → "Require status checks", select all CI jobs
```

---

### T0.7 — LAMMPS oracle integration (submodule + GPU build)

```
# TDMD Task: LAMMPS as git submodule with GPU + required packages

## Context
- Project: TDMD
- Master spec: §3 (LAMMPS as oracle, не runtime dependency), §13 (test pyramid uses LAMMPS for differential)
- Module spec: `docs/specs/verify/SPEC.md` — LAMMPS как git submodule в `verify/third_party/lammps/`, agent-buildable
- Milestone: M0
- Role: Validation / Reference Engineer

## Goal
Интегрировать LAMMPS как git submodule в `verify/third_party/lammps/`, build с GPU support (sm_120) и required packages (`MANYBODY`, `MEAM`, `ML-SNAP`, `MOLECULE`, `KSPACE`, `EXTRA-PAIR`, `EXTRA-DUMP`). Цель — готовый oracle binary для M1+ differential gates без необходимости rebuild при каждой новой задаче.

## Scope
- [included] Pin LAMMPS на specific stable tag (recommend latest stable as of 2026-04-17 — verify latest на https://github.com/lammps/lammps/tags перед добавлением; рекомендация — последний `stable_<date>`, не `develop`)
- [included] `git submodule add https://github.com/lammps/lammps.git verify/third_party/lammps`
- [included] `git -C verify/third_party/lammps checkout <tag>`
- [included] `verify/third_party/lammps/build_tdmd.sh` — wrapper build script:
  ```bash
  #!/usr/bin/env bash
  set -euo pipefail
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  BUILD_DIR="$SCRIPT_DIR/build_tdmd"
  INSTALL_DIR="$SCRIPT_DIR/install_tdmd"
  CUDA_ARCH="${TDMD_LAMMPS_CUDA_ARCH:-sm_120}"
  cmake -B "$BUILD_DIR" -S "$SCRIPT_DIR/cmake" \
      -G Ninja \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -D BUILD_MPI=on -D BUILD_OMP=on -D BUILD_SHARED_LIBS=on \
      -D PKG_GPU=on -D GPU_API=cuda -D GPU_ARCH="$CUDA_ARCH" -D GPU_PREC=double \
      -D PKG_MANYBODY=on \
      -D PKG_MEAM=on \
      -D PKG_ML-SNAP=on \
      -D PKG_MOLECULE=on \
      -D PKG_KSPACE=on \
      -D PKG_EXTRA-PAIR=on \
      -D PKG_EXTRA-DUMP=on \
      -D LAMMPS_EXCEPTIONS=on \
      "$@"
  cmake --build "$BUILD_DIR" --parallel
  cmake --install "$BUILD_DIR"
  ```
- [included] `verify/third_party/lammps/README.md`:
  - Purpose: external scientific oracle для differential validation
  - Pinned version (tag, commit SHA, date)
  - How to build: `./build_tdmd.sh`
  - Required packages explained per use case:
    - `MANYBODY` — EAM (M2+), Tersoff, Stillinger-Weber
    - `MEAM` — MEAM (M10)
    - `ML-SNAP` — SNAP (M8)
    - `MOLECULE` — basic molecular topology (для general data files)
    - `KSPACE` — long-range Coulomb (M13)
    - `EXTRA-PAIR` — additional pair styles (Buckingham, Coul, etc.)
    - `EXTRA-DUMP` — additional dump formats (helpful для diff harness)
    - `GPU` — CUDA-accelerated kernels (essential для apples-to-apples GPU comparison в M6+)
  - Build time estimate: ~15-30 min на этой машине (16-thread compile)
  - Re-pinning policy: только Architect role + human approval; новая версия = SPEC delta to verify/SPEC §3

- [included] `cmake/FindLammps.cmake`:
  - Locate LAMMPS install в `verify/third_party/lammps/install_tdmd/`
  - Set `LAMMPS_FOUND`, `LAMMPS_INCLUDE_DIRS`, `LAMMPS_LIBRARIES`, `LAMMPS_EXECUTABLE`
  - Provide `LAMMPS_VERSION_TAG` cache var

- [included] `tools/lammps_smoke_test.sh` — runs short Al FCC EAM trajectory:
  ```bash
  # 100 atoms Al FCC, 100 steps, EAM Mishin Al, both CPU и GPU runs
  # Verify: completes без crash, produces log.lammps, energy в expected range
  ```
- [included] `tools/lammps_smoke_test.in` — LAMMPS input script для smoke test (10-15 lines)
- [included] `verify/third_party/lammps/.gitignore` — игнорировать `build_tdmd/`, `install_tdmd/`

- [included] Documentation update:
  - `docs/development/build_instructions.md` — добавить секцию "Building LAMMPS oracle"
  - Top-level `README.md` — упомянуть LAMMPS dependency и build wrapper

- [included] Update `.github/workflows/ci.yml` — добавить optional job `build-lammps`:
  - Job runs только on labels `lammps-rebuild` или `weekly` schedule (не на каждом PR — слишком долго)
  - Triggers daily через cron
  - Caches build artifacts

## Out of scope
- [excluded] Differential test harness (M1+, в `verify/` proper module)
- [excluded] LAMMPS Python bindings (отдельная задача если понадобится для M1+ test scripting)
- [excluded] LAMMPS upstream contributions / bug fixes — submodule стоит на pinned tag, не tracking branch
- [excluded] ML-PACE, ML-IAP packages — нужны только в M12, добавим pinning re-pin time
- [excluded] KOKKOS package — альтернатива GPU, не critical в v1
- [excluded] `pair_style snap` через GPU package — встроен в ML-SNAP, GPU acceleration через KOKKOS (deferred)

## Mandatory invariants
- LAMMPS submodule на pinned tag, не на moving branch
- `build_tdmd.sh` reproducible: same checkout → same binary (modulo timestamps)
- Smoke test passes на dev-машине (RTX 5080)
- LAMMPS version recorded в reproducibility bundle (когда тот появится в M5+)
- `verify/third_party/lammps/` — read-only от перспективы main TDMD code (никаких patches inline)

## Required files
- `.gitmodules` — обновлено
- `verify/third_party/lammps/` — submodule pointer
- `verify/third_party/lammps/build_tdmd.sh` — wrapper (NOT inside submodule, в parent dir? Или внутри? — См. Hints)
- `verify/third_party/lammps/README.md` — wrapper readme (рядом с submodule, не внутри)
- `verify/third_party/lammps/.gitignore` — wrapper ignores
- `cmake/FindLammps.cmake`
- `tools/lammps_smoke_test.sh`
- `tools/lammps_smoke_test.in`
- Updates: `.github/workflows/ci.yml`, `docs/development/build_instructions.md`, top-level `README.md`

## Required tests
- [submodule] `git submodule status` shows pinned commit; `cd verify/third_party/lammps && git describe --tags` returns expected stable tag
- [build] `./verify/third_party/lammps/build_tdmd.sh` succeeds на dev-машине от clean checkout
- [smoke] `tools/lammps_smoke_test.sh` runs 100-step Al FCC EAM, energy в expected range
- [smoke gpu] Same smoke test с `package gpu 1` directive — runs successfully на RTX 5080
- [findpackage] `find_package(Lammps)` в test CMake project locates installed binary

## Expected artifacts
- LAMMPS binary в `verify/third_party/lammps/install_tdmd/bin/lmp`
- Build log saved (one-time) для reference в PR description
- Smoke test log saved
- Documentation updates
- CI workflow updated

## Acceptance criteria
- [ ] LAMMPS builds от clean state с required packages (verify через `lmp -h | grep -E "GPU|MANYBODY|MEAM|ML-SNAP"`)
- [ ] GPU smoke test runs successfully (energy reproduces CPU result within `1e-10`)
- [ ] Build time documented в README
- [ ] FindLammps.cmake locates binary
- [ ] CI job `build-lammps` runs (manually-triggered first time)
- [ ] PR review approved by human

## Hints
- LAMMPS GPU package с CUDA sm_120: убедиться что используется CUDA 12.8+; LAMMPS CMake auto-detects compute capability но можно override через `-D GPU_ARCH=sm_120`
- LAMMPS stable tag формат: `stable_<DDMonYYYY>` (например `stable_29Aug2024`); list через `git ls-remote --tags https://github.com/lammps/lammps.git | grep stable_`
- `build_tdmd.sh` placement: я бы положил его **рядом** с submodule в `verify/third_party/lammps/build_tdmd.sh` — wait, это inside submodule; правильнее в `verify/third_party/build_lammps.sh` или `tools/build_lammps.sh`. Reconsider в реализации. Recommendation: **`tools/build_lammps.sh`** (рядом с другими tool wrappers); README placement — в `verify/third_party/lammps_README.md` (sibling, не внутри submodule).
- `LAMMPS_EXCEPTIONS=on` важно для programmatic error handling в C++ harness позже
- Smoke test input — minimal: `lattice fcc 4.05; region box block 0 5 0 5 0 5; create_box 1 box; create_atoms 1 box; mass 1 26.98; pair_style eam/alloy; pair_coeff * * Al99.eam.alloy Al; velocity all create 300.0 12345; fix 1 all nve; run 100`
- LAMMPS data files (`Al99.eam.alloy`) поставляются вместе с LAMMPS в `potentials/` — путь `verify/third_party/lammps/potentials/Al99.eam.alloy`
- Чтобы избежать накопления `build_tdmd/` в git history — `.gitignore` обязателен
```

---

## 5. M0 Acceptance Gate

После закрытия всех 7 задач — проверить полный M0 artifact gate (master spec §14):

- [ ] **Repository structure** соответствует master spec §14 M0 (T0.1)
- [ ] **3 empty modules** (`state`, `neighbor`, `integrator`) интегрированы в build (T0.5)
- [ ] **Repo собирается на CI green** — both CPU и CUDA jobs pass (T0.4 + T0.6)
- [ ] **Lint + pre-commit** активны и enforced (T0.3 + T0.6)
- [ ] **Templates** готовы для M1 onboarding (T0.2)
- [ ] **LAMMPS oracle** собран с GPU + required packages, smoke test green (T0.7)
- [ ] **CI Pipeline A** active как required status check (T0.6)
- [ ] **Documentation** updated (build instructions, CI setup, code style)
- [ ] **Pre-implementation reports** для каждой задачи attached в PR descriptions
- [ ] **Session reports** для каждой задачи attached в PR descriptions
- [ ] **Human review** approval для каждого PR

После прохождения gate — Architect создаёт `docs/development/m1_execution_pack.md` тем же шаблоном.

---

## 6. Risks и open questions для M0

### Risks

- **R1: sm_120 toolchain readiness.** RTX 5080 (Blackwell consumer) появился недавно; CUDA 12.8 — minimal версия с поддержкой. Если `nvcc` < 12.8 на dev-машине, T0.4 / T0.7 заблокируются. **Mitigation:** в T0.4 добавить explicit version check.
- **R2: LAMMPS stable tag drift.** `stable_<date>` может deprecate features используемые TDMD. **Mitigation:** pinning + re-pin policy в verify/SPEC §3 (already documented).
- **R3: Self-hosted CI runner reliability.** Single dev-машина = single point of failure. **Mitigation:** scheduled rebuild job (T0.6) catches drift; ручной rerun возможен.
- **R4: pre-commit и LFS interaction.** check-added-large-files может блокировать LFS-tracked файлы. **Mitigation:** test в T0.3 PR с реальным LFS файлом.

### Open questions (escalate to human если возникнут)

- **OQ1: LAMMPS version pin policy.** Latest stable tag, или specific tested version? — Default: latest stable as of T0.7 implementation; re-pin требует human approval.
- **OQ2: Self-hosted runner authentication.** Personal Access Token или GitHub App? — Default: PAT; миграция на GitHub App когда станет multi-developer project.
- **OQ3: `compile_commands.json` для clang-tidy на CUDA.** clang-tidy не всегда корректно парсит nvcc-generated commands. — Default: skip CUDA в clang-tidy, only C++ host code; добавить linting CUDA через `cuda-clang-tidy` если найдём reliable tool в M2+.
- **OQ4: ccache в CI.** Включить или нет? — Default: defer до M2 когда build time станет measurably painful.

---

## 7. Что НЕ входит в M0 (важно для предотвращения scope creep)

Эти items могут показаться "почти M0" но строго отнесены к M1+:

- ❌ Реальная имплементация AtomSoA, CellGrid, Velocity-Verlet — **M1**
- ❌ YAML config parsing — **M1**
- ❌ LAMMPS data file importer — **M1**
- ❌ UnitConverter — **M1**
- ❌ Differential test harness — **M1** (M0 только готовит LAMMPS binary)
- ❌ Pipelines B-F в CI — **появляются по мере появления tests**
- ❌ Performance benchmarks — **M2** (с perf model skeleton)
- ❌ scheduler/, zoning/, perfmodel/ skeletons — **M3+** (не нужны до M3 zoning planner)
- ❌ Реализация custom clang-tidy `tdmd-missing-restrict` check — **M2-M3** (сейчас только STUB)
- ❌ MixedFastBuild и др. flavor реализации — **M2+ для production, M6+ для GPU-specific**

При искушении агента "заодно сделать X" — `claude_code_playbook.md §1.1 п.7` (no scope creep) применяется автоматически.

---

## 8. Change log пакета

### v0.1 (2026-04-17)

- Initial draft. 7 задач, 6 решений, 4 risks, 4 open questions.
- Approved decisions: BSD-3, state/neighbor/integrator, full CUDA CI, C++20, pre-commit.com, LAMMPS GPU build с full v1 packages.

---

*Конец M0 Execution Pack v0.1.*
