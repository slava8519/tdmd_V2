# M1 Execution Pack

**Document:** `docs/development/m1_execution_pack.md`
**Status:** draft, awaiting human review
**Parent:** `TDMD_Engineering_Spec.md` §14 (M1), `docs/development/claude_code_playbook.md` §3
**Milestone:** M1 — CPU Reference MD без TD (6 нед.)
**Created:** 2026-04-17
**Author:** Architect / Spec Steward role (Claude Opus 4.7)

---

## 0. Purpose

Этот документ декомпозирует milestone **M1** master spec'а на **12 PR-size задач**, каждая сформулирована по каноническому шаблону `claude_code_playbook.md` §3.1 и готова к прямому назначению Claude Code-агенту.

Документ — **process artifact**, не SPEC delta. Не модифицирует контракты, не меняет hierarchy of truth (master spec → module SPEC → код). Если в процессе реализации M1 будет обнаружен gap в спеке — оформляется отдельной SPEC delta по playbook §9.1, не в этом pack'е.

M1 — первая реальная имплементация: после M0 (skeleton) TDMD получает работающую однопоточную CPU-only reference-версию, которая: (1) читает LAMMPS `.data` и минимальный YAML config; (2) строит neighbor list; (3) прогоняет velocity-Verlet NVE с Morse потенциалом; (4) выдаёт thermo dump; (5) **проходит T1 differential test vs LAMMPS oracle**. TD ещё нет — это M4. GPU ещё нет — это M6.

После успешного закрытия всех 12 задач и acceptance gate (§5) — milestone M1 считается завершённым; execution pack для M2 создаётся как новый аналогичный документ.

---

## 1. Decisions log (зафиксировано до старта T1.1)

| # | Решение | Значение | Rationale / источник |
|---|---|---|---|
| **D-M1-1** | YAML parser | `yaml-cpp` (via apt / vcpkg / FetchContent — выбор в T1.4) | Mainstream C++ YAML lib, maintained, Ubuntu-packaged. Альтернативы (`rapidyaml`, `ryml`) — performance-focused, излишни для config parsing в M1. |
| **D-M1-2** | Первый потенциал | `MorsePotential` (analytic, pair-only) | Master spec §14 M1. Простейший с closed-form force, идеален для T1 differential baseline. EAM — M2. |
| **D-M1-3** | T1 reference config | Al FCC, 5×5×5 = 500 атомов, NVE, 100 steps, Morse (fitted to Al) | Повторяет размер LAMMPS smoke test (M0/T0.7), но с Morse вместо EAM. Morse параметры для Al — published (Girifalco-Weizer 1959). |
| **D-M1-4** | CLI dispatcher lib | Header-only `cxxopts` (FetchContent) | Легче чем CLI11 для M1 scope (`tdmd run`, `tdmd validate` — 2 subcommands). |
| **D-M1-5** | Integrator в M1 | **Только** velocity-Verlet NVE | Master spec §14 M1. NVT/NPT — M9; Langevin — post-v1. |
| **D-M1-6** | UnitConverter в M1 | Full API, `metal` native реализован, `lj` stub возвращает `NotImplementedInM1Error` | Master spec §5.3 + §14 M1: "только metal, adapter-заглушка для lj". API stable — чтобы M2 добавил lj без ломки. |
| **D-M1-7** | Neighbor list тип | Half-list, `newton on`, `skin = 0.3 Å` (LAMMPS default) | Master spec §14 M1 + neighbor/SPEC. Упрощение: full-list, newton off — в M2+ по необходимости. |
| **D-M1-8** | Restart/resume | **Defer to M2** | Master spec §14 M1 не упоминает. `run 0` в M1 = всегда с чистого старта. |
| **D-M1-9** | BuildFlavor в M1 | Только `Fp64ReferenceBuild` | Master spec §D: reference path — canonical oracle. Остальные flavors — M2+. |
| **D-M1-10** | Determinism scope | Bitwise reproducibility на одном compile + hardware (без cross-compiler guarantee в M1) | Master spec §7.5: full cross-compiler determinism — в M4+ scheduler; в M1 same-binary reproducibility достаточна для T1. |

---

## 2. Глобальные параметры окружения

Эти параметры — common context для всех 12 задач.

| Параметр | Значение | Примечание |
|---|---|---|
| OS | Linux (Ubuntu 24.04 LTS) | Dev-машина пользователя |
| C++ compiler | GCC 13+ / Clang 17+ | C++20; CI уже проверяет оба |
| CMake | 3.25+ | Master spec §15.2 |
| CUDA | 13.1 installed, **not used в M1** | GPU path — M6 |
| Python | 3.10+ | Для pre-commit, toolchain scripts |
| Test framework | Catch2 v3 (FetchContent, header-only) | Уже в M0 T0.5 |
| YAML parser | yaml-cpp (FetchContent или find_package) | D-M1-1 |
| CLI parser | cxxopts (FetchContent) | D-M1-4 |
| LAMMPS oracle | `verify/third_party/lammps/install_tdmd/bin/lmp` | Собран в M0 T0.7; fp64, sm_120 |
| Active BuildFlavor | `Fp64ReferenceBuild` | D-M1-9 |
| Run mode | single-rank, single-thread, CPU only | Master spec §14 M1 |
| Branch | новая `m1/T1.X-<topic>` для каждой задачи, PR в `main` | CI required: 4 hosted checks |

---

## 3. Suggested PR order

Dependency graph:

```
T1.1 (state: AtomSoA/Box/Species) ─────┬──► T1.5 (CellGrid) ──► T1.6 (Neighbor list) ──┐
                                         │                                                │
T1.2 (UnitConverter) ──┬──► T1.3 (.data importer) ──┐                                    │
                        └──► T1.4 (YAML + preflight) ──┬──► T1.10 (tdmd validate CLI)    │
                                                        │                                 │
T1.7 (Velocity-Verlet) ─────────────────────────────────┼─────────────────────────────────┤
                                                        │                                 │
T1.8 (MorsePotential) ──────────────────────────────────┤                                 │
                                                        │                                 │
                                                        └─────► T1.9 (tdmd run + engine) ─┤
                                                                                          │
                                                                                          ├──► T1.11 (T1 differential harness)
                                                                                          │                            │
                                                                                          └────────────────────────────┴──► T1.12 (M1 integration smoke)
```

**Линейная последовательность (single agent):** T1.1 → T1.2 → T1.5 → T1.6 → T1.7 → T1.8 → T1.3 → T1.4 → T1.9 → T1.10 → T1.11 → T1.12.

**Параллельный режим (multi-agent):** после T1.1 — `{T1.2, T1.5, T1.7}` параллельно; после T1.5 — T1.6; после T1.2 — `{T1.3, T1.4}` параллельно; после T1.6 + T1.8 + T1.9-deps ready — T1.9 (большая); T1.10 параллельно с T1.9; T1.11 после T1.9; T1.12 последний.

**Estimated effort:** 5–7 недель (single agent). T1.9 (SimulationEngine wiring) и T1.11 (differential harness) — самые длинные (~1 нед. каждая).

---

## 4. Tasks

### T1.1 — `AtomSoA` + `Box` + `Species` real implementation

```
# TDMD Task: Core state data types

## Context
- Project: TDMD
- Master spec: §5, §8 (ownership), §D (precision), §14 M1
- Module SPEC: docs/specs/state/SPEC.md
- Playbook role: Core Runtime Engineer
- Milestone: M1

## Goal
Реализовать три первичных data type модуля `state/`: `AtomSoA` (SoA layout атомов), `Box` (periodic box + triclinic-ready API), `Species` (atom types registry). После T1.1 модуль `state/` перестаёт быть skeleton и предоставляет полный interface для всех последующих задач.

## Scope
- [included] `AtomSoA`: positions `x`, `y`, `z` (double), velocities `vx`, `vy`, `vz`, forces `fx`, `fy`, `fz`, `id`, `type`, `mass` — SoA layout, aligned allocations (64-byte для будущего SIMD/GPU)
- [included] Add/remove atoms API: `add_atom(id, type, pos, vel)`, `remove_atom(local_index)` (swap-and-pop)
- [included] Resize/reserve API с capacity tracking; growth strategy 1.5x
- [included] Bounds-checked accessors (debug-only asserts) + unchecked fast path
- [included] `Box`: orthogonal first, triclinic-ready поле `tilt_xy, tilt_xz, tilt_yz` (zeroed в M1); PBC flags per axis; `wrap(pos)`, `unwrap_minimum_image(dx)`, `volume()`, `length(axis)`
- [included] `Species`: registry of atom types, each with name (e.g. "Al"), mass, charge (unused в M1, stored); `by_name()`, `by_index()`, `count()`
- [included] Ownership invariant: только `AtomSoA` mutates atoms; потенциалы пишут в `forces[]` view (неперехватывают allocation)

## Out of scope
- [excluded] `CellGrid`-sorted view — это T1.5
- [excluded] Migration (halo/ghost atoms) — это M5
- [excluded] Serialization to restart file — M2+
- [excluded] Triclinic support тесты — M3+ (в M1 только orthogonal; triclinic поля zeroed и unused)

## Mandatory invariants
- SoA fields aligned на 64 bytes (static_assert via `alignof`)
- `AtomSoA::size()` консистентен с length каждого поля
- `Box::wrap(pos)` идемпотентен: `wrap(wrap(p)) == wrap(p)`
- `unwrap_minimum_image(dx)` возвращает `|Δ| ≤ L/2` по каждой оси (PBC invariant)
- Никаких hardcoded единиц — mass/length/energy приходят в SI-неутральных типах; конверсия — через `UnitConverter` (T1.2)

## Required files
- `src/state/atom_soa.h`, `src/state/atom_soa.cpp` — real impl (replace skeleton)
- `src/state/box.h`, `src/state/box.cpp`
- `src/state/species.h`, `src/state/species.cpp`
- `src/state/CMakeLists.txt` — add new sources
- `tests/state/test_atom_soa.cpp` — расширить (add/remove/resize, alignment)
- `tests/state/test_box.cpp` — wrap/unwrap/volume, edge cases
- `tests/state/test_species.cpp`

## Required tests
- [unit] atom_soa: add N, remove K, check size invariants, check alignment
- [unit] box: wrap каждой грани, corners, minimum-image по каждой оси
- [unit] species: round-trip by_name → by_index → by_name
- [property ≥10⁴ cases] box.wrap идемпотентен; unwrap_minimum_image даёт norm ≤ √3·L/2
- [alignment] `reinterpret_cast<uintptr_t>(&x[0]) % 64 == 0`

## Expected artifacts
- Единый PR
- Pre-impl report + session report в PR description
- 5+ tests (minimum per required tests list); 10⁴ property cases

## Acceptance criteria
- [ ] 3 types с полным API из Scope
- [ ] Все unit tests green на gcc-13 и clang-17
- [ ] Property tests green (10⁴+ cases)
- [ ] compile_commands.json обновлён (CI artifact)
- [ ] Code coverage ≥ 80% для src/state/
- [ ] No warnings (CI -Werror passes)
- [ ] Human review approved

## Hints
- Для aligned allocation: `std::vector` + custom allocator, или `std::aligned_alloc` + manual lifetime. Второй вариант проще для SoA.
- Triclinic API: смотри LAMMPS `domain.cpp` для inspiration; в M1 просто хранить tilt=0.
- Unit tests — Catch2 `SECTION` для isolation; property tests — `GENERATE(take(N, random(a,b)))`.
```

---

### T1.2 — `UnitConverter` (metal native + lj stub)

```
# TDMD Task: Unit system converter

## Context
- Master spec: §5.3 (unit system support), §14 M1
- Module SPEC: docs/specs/runtime/SPEC.md (UnitConverter живёт в runtime)
- Role: Physics Engineer
- Milestone: M1

## Goal
Реализовать `UnitConverter` — централизованный единый point of конверсии между unit systems (LAMMPS `metal`, `lj`) и внутренним SI-нейтральным представлением. В M1 полностью работает `metal`; `lj` имеет API, но функции возвращают `NotImplementedInM1Error`. API stable: M2 добавит `lj` без ломки сигнатур.

## Scope
- [included] `UnitSystem` enum: `Metal`, `Lj` (+ `Real`, `Cgs`, `Si` как forward-declared для future)
- [included] `UnitConverter::from_metal(...)`, `to_metal(...)` для: length (Å), energy (eV), time (ps), mass (g/mol), force (eV/Å), pressure (bar), velocity (Å/ps), temperature (K)
- [included] `UnitConverter::from_lj(...)`, `to_lj(...)` — returns `std::unexpected(NotImplementedInM1)` или throws `NotImplementedInM1Error` (consistent error strategy — документировать)
- [included] Internal representation = `metal`-units в M1 (D-M1-6). NOT hardcoded — через `internal_system()` accessor; M2 может переключить.
- [included] Validation: отказ сконвертировать между incompatible dimensions (e.g. length → mass)

## Out of scope
- [excluded] `lj` real impl — M2
- [excluded] `real`, `cgs`, `si` — post-v1
- [excluded] Temperature conversion для non-kB contexts — всегда в K

## Mandatory invariants
- Round-trip: `to_metal(from_metal(x)) == x` bitwise (no precision loss в metal↔metal identity)
- Dimensional check rejects invalid conversions at compile time where possible (`strong typedef`) или runtime
- NO hardcoded unit literals в других модулях — все converters проходят через `UnitConverter`

## Required files
- `src/runtime/unit_converter.h`, `src/runtime/unit_converter.cpp`
- `src/runtime/CMakeLists.txt` — add
- `tests/runtime/test_unit_converter.cpp`

## Required tests
- [unit] metal identity (all 8 dims): from_metal(to_metal(x)) == x
- [unit] dimensional check: length → mass fails с clear error
- [unit] lj stub: returns NotImplementedInM1 error
- [precision] identity round-trip bitwise equal (double)

## Expected artifacts
- Single PR
- API documented в header (each public method — docstring, формула/константа)

## Acceptance criteria
- [ ] Full metal support (8 dimensions)
- [ ] lj returns explicit NotImplementedInM1 error
- [ ] Tests green
- [ ] API stable checked через zero breakage в T1.3 (which uses it)
- [ ] Human review approved

## Hints
- Constants LAMMPS `metal` units: https://docs.lammps.org/units.html
- Paper Andreev §2 использует `metal` (Al FCC 10⁶ atoms) — наш anchor test.
- Strong typedef `Quantity<Length>` vs primitive double: выбор за implementer'ом; рекомендуется минимум `struct` wrapper для LengthQ/EnergyQ.
```

---

### T1.3 — LAMMPS `.data` file importer

```
# TDMD Task: LAMMPS data file importer

## Context
- Master spec: §14 M1 ("LAMMPS data importer")
- Module SPEC: docs/specs/io/SPEC.md
- Role: Scientist UX Engineer
- Depends on: T1.1 (AtomSoA), T1.2 (UnitConverter)
- Milestone: M1

## Goal
Реализовать парсер LAMMPS `write_data` output format — первый ingestion path. После T1.3 TDMD может читать LAMMPS-generated initial conditions и строить `AtomSoA` + `Box` + `Species`.

## Scope
- [included] Sections parsing: header (counts + box bounds), `Masses`, `Atoms` (atom_style `atomic` только в M1; `full` etc — M2+), `Velocities` (если есть)
- [included] Units detected from file header comment (если LAMMPS записал) или переданы caller'ом через `units` param; default = metal (document)
- [included] Error handling: corrupt file, missing sections, inconsistent counts — ясные messages с line number
- [included] Streaming parser (не загружает весь файл в память)
- [included] Output = `ImportResult { AtomSoA, Box, Species, std::optional<timestep>}`

## Out of scope
- [excluded] atom_style `full`, `charge`, `molecular` — M2+
- [excluded] LAMMPS `restart` binary format — M2+
- [excluded] Triclinic box (`xy xz yz` tilt) — M3+
- [excluded] Image flags (`ix iy iz`) — M3+ (molecular)

## Mandatory invariants
- Parser идемпотентен: parse → dump → parse даёт bit-exact тот же `AtomSoA`
- Все значения проходят через `UnitConverter`; никаких raw numeric literals
- Box reading — сохраняет orthogonal invariant в M1

## Required files
- `src/io/lammps_data_reader.h`, `src/io/lammps_data_reader.cpp`
- `src/io/CMakeLists.txt`
- `tests/io/test_lammps_data_reader.cpp`
- `tests/io/fixtures/` — минимум 3 test files:
  - `al_fcc_500.data` — generated by LAMMPS (matches T1 benchmark)
  - `corrupt_missing_atoms.data`
  - `empty.data`

## Required tests
- [unit] parse valid Al FCC 500 — check atom count, box bounds, first/last atom coords match expected
- [unit] parse corrupt file — get ParseError с line number
- [property ≥10³] dump-and-reparse round-trip: positions bit-exact, velocities bit-exact
- [integration] LAMMPS writes `.data` → TDMD parses → compare to direct LAMMPS read (via `read_data` in LAMMPS)

## Expected artifacts
- Single PR
- 3 fixture files committed (small — not LFS)
- README in `tests/io/fixtures/` explaining what each fixture tests

## Acceptance criteria
- [ ] Valid Al FCC 500 parses correctly
- [ ] Corrupt files reject with line numbers
- [ ] Round-trip property test green (10³ cases через randomized coords)
- [ ] Human review approved

## Hints
- LAMMPS data format spec: https://docs.lammps.org/read_data.html
- Reference implementation: LAMMPS `src/read_data.cpp` (БД — для понимания format quirks, не copy-paste)
- `al_fcc_500.data` можно сгенерировать через `tools/lammps_smoke_test.in` + `write_data output.data` line
```

---

### T1.4 — `tdmd.yaml` parser + preflight validator

```
# TDMD Task: YAML config parser + preflight

## Context
- Master spec: §5 (functional requirements), §14 M1 ("minimal tdmd.yaml")
- Module SPEC: docs/specs/io/SPEC.md, docs/specs/cli/SPEC.md
- Role: Scientist UX Engineer
- Depends on: T1.2 (UnitConverter — нужно для units field validation)
- Milestone: M1

## Goal
Реализовать чтение минимального `tdmd.yaml` config file + preflight validator, который отклоняет config без обязательных полей до запуска simulation. M1 scope: только базовые поля, достаточно для `tdmd run` на T1 benchmark.

## Scope
- [included] Required top-level fields: `units` (string: `metal`/`lj`), `atoms.source` (path to .data file), `run.steps` (int), `integrator` (string: `nve` only в M1), `timestep` (float — в units системы)
- [included] Optional: `seed` (int, default 12345), `thermo.every` (int, default 100)
- [included] Preflight rules:
  - `units` обязателен (reject — master spec §14 M1 gate)
  - `atoms.source` file must exist
  - `run.steps` ≥ 1
  - `integrator == "nve"` (warn если что-либо другое — M1 только NVE)
  - `timestep` > 0
- [included] YAML parsing через yaml-cpp (D-M1-1)
- [included] Error reporting: multi-error mode — collect все issues, report все сразу (не fail-on-first)

## Out of scope
- [excluded] TD-specific fields (`td.pipeline_depth`, `td.zoning`) — M3+
- [excluded] `dump`, `restart`, `fix` — M2+
- [excluded] Variable substitution / includes — post-v1
- [excluded] JSON alternative — post-v1

## Mandatory invariants
- Parser fails gracefully на invalid YAML (syntax error → clear message)
- Preflight deterministic: same config → same error list, same order
- Никаких hidden defaults для required fields — missing = hard error

## Required files
- `src/io/yaml_config.h`, `src/io/yaml_config.cpp`
- `src/io/preflight.h`, `src/io/preflight.cpp`
- `src/io/CMakeLists.txt` — add yaml-cpp dependency
- `tests/io/test_yaml_config.cpp`
- `tests/io/test_preflight.cpp`
- `tests/io/fixtures/configs/`:
  - `valid_nve_al.yaml` — minimal valid
  - `missing_units.yaml` — expect reject
  - `bad_timestep.yaml`, `bad_steps.yaml`, `missing_atoms_source.yaml`
- `cmake/Dependencies.cmake` — add yaml-cpp FetchContent or find_package

## Required tests
- [unit] valid config parses into expected struct
- [unit] missing `units` → preflight error "units is required"
- [unit] all 5 bad fixtures → expected errors
- [unit] multi-error: file с 3 problems → 3 errors returned
- [integration] preflight on T1 reference config from T1.11 passes

## Expected artifacts
- Single PR
- Fixtures documented
- README `tests/io/fixtures/configs/README.md` — per-fixture expected outcome

## Acceptance criteria
- [ ] `units` required — demonstrated via missing_units.yaml test
- [ ] yaml-cpp integrated (FetchContent pinned tag)
- [ ] All 5 invalid fixtures reject
- [ ] Valid fixture passes
- [ ] Human review approved

## Hints
- yaml-cpp FetchContent: `GIT_TAG 0.8.0` (latest stable as of 2026)
- Multi-error accumulation: accumulate в `std::vector<PreflightError>`, возвращать только если non-empty
- Preflight — pure function: `Preflight(config) -> std::vector<PreflightError>`
```

---

### T1.5 — `CellGrid` real impl + deterministic reorder

```
# TDMD Task: Cell grid with deterministic reorder

## Context
- Master spec: §6.1 (spatial structure), §14 M1
- Module SPEC: docs/specs/neighbor/SPEC.md
- Role: Neighbor / Migration Engineer
- Depends on: T1.1 (AtomSoA, Box)
- Milestone: M1

## Goal
Реализовать `CellGrid` — spatial hash, bin атомов по положению. Deterministic reorder — обязателен: "stable" sort (preserving insertion order внутри bin'а) чтобы Reference-path был bitwise-reproducible.

## Scope
- [included] `CellGrid::build(box, cutoff, skin)` — определяет `nx × ny × nz` cells исходя из `cutoff + skin`; minimum 3 cells per axis (для periodic stencil)
- [included] `CellGrid::bin(atom_soa)` — распределение атомов по cells
- [included] `CellGrid::reorder(atom_soa)` — stable in-place reordering by cell index; updates `id_to_local_index` map; preserves `id` field for tracing
- [included] `CellGrid::cell_of(pos)` → cell index (with PBC wrapping)
- [included] `CellGrid::neighbor_cells(cell_idx)` → iterator over 27 (3×3×3) candidate cells including self
- [included] Rebuild triggers: full `build()` when Box changes; `bin() + reorder()` иначе

## Out of scope
- [excluded] Skin/displacement certificate — это T1.6 (neighbor list)
- [excluded] Triclinic binning — M3+
- [excluded] GPU version — M6
- [excluded] Non-cubic cells — post-v1

## Mandatory invariants
- Reorder — **stable sort** (D-M1-10; master spec §D: Reference profile mandates stable)
- `cell_of(pos) ∈ [0, nx·ny·nz)` для любого `pos` after `box.wrap`
- Neighbors iterator visits каждую из 27 cells ровно один раз
- `reorder` preserves atom count и всех per-atom attributes (positions, velocities, types, ids) — только local indexing меняется

## Required files
- `src/neighbor/cell_grid.h`, `src/neighbor/cell_grid.cpp` (replace skeleton)
- `src/neighbor/CMakeLists.txt` — update
- `tests/neighbor/test_cell_grid.cpp` — расширить

## Required tests
- [unit] build: 500 atoms Al FCC box → expected 2×2×2 cells для cutoff=6Å
- [unit] bin + reorder: atoms в одной cell остаются в insertion order (stable)
- [property ≥10⁴] для random configs: all atoms in one of 27 neighbors of their cell
- [property ≥10⁴] reorder determinism: одинаковый input → одинаковая permutation (bit-exact)
- [edge] tiny box (3×3×3 cells min) — stencil не дублирует cells

## Expected artifacts
- Single PR
- Profile / perf note в PR description (wall-time bin+reorder на 500/5000/50000 atoms)

## Acceptance criteria
- [ ] Stable sort verified (property test)
- [ ] Neighbors stencil correctness (property test 10⁴ cases)
- [ ] No warnings
- [ ] Human review approved

## Hints
- Stable sort: `std::stable_sort` с сравнением по cell index
- Для deterministic reorder — избегай `std::partition` (unstable); используй `stable_partition` или index-sort approach
- Box wrapping — reuse `Box::wrap` из T1.1
```

---

### T1.6 — Neighbor list build (half, newton on)

```
# TDMD Task: Neighbor list

## Context
- Master spec: §6.1, §14 M1; neighbor/SPEC.md
- Role: Neighbor / Migration Engineer
- Depends on: T1.1 (AtomSoA), T1.5 (CellGrid)
- Milestone: M1

## Goal
Построить neighbor list поверх `CellGrid` — для каждой пары атомов в пределах `cutoff + skin`, сохранить pair только один раз (half-list, `newton on`). Это structure feeds pair force kernel.

## Scope
- [included] `NeighborList::build(atom_soa, box, cell_grid, cutoff, skin)`
- [included] Storage: per-atom `std::vector<int>` of neighbor local indices (SoA of vectors; CSR representation — M2+ optimization)
- [included] Half list rule: include `j` in `i`'s list **only if** `j > i` (newton on convention)
- [included] Skin-tracking: `displacement_cert(atom_soa)` — accumulates max |Δr| since last build; `needs_rebuild()` когда `2·max|Δr| > skin`
- [included] Rebuild triggers: explicit `needs_rebuild` check от integrator; also on `Box::resize()` (rare в NVE)

## Out of scope
- [excluded] Full list / newton off — M2+ (нужно для некоторых pair_styles)
- [excluded] Long-range (Coulomb) — M9+
- [excluded] Multi-cutoff (EAM cutoff > pair cutoff) — M2 (EAM)
- [excluded] GPU neighbor build — M6

## Mandatory invariants
- No duplicate pairs (strict `j > i`)
- No self-pairs
- Все pairs с `|r_ij| ≤ cutoff + skin` присутствуют (completeness)
- No pairs с `|r_ij| > cutoff + skin` (no false positives outside skin)
- `needs_rebuild` — conservative: может trigger чуть раньше нужного, но не позже

## Required files
- `src/neighbor/neighbor_list.h`, `src/neighbor/neighbor_list.cpp`
- `src/neighbor/displacement_cert.h`, `src/neighbor/displacement_cert.cpp`
- `src/neighbor/CMakeLists.txt`
- `tests/neighbor/test_neighbor_list.cpp`
- `tests/neighbor/test_displacement_cert.cpp`

## Required tests
- [unit] 500 atoms Al FCC: expected neighbor count per atom ~42 (12 nearest × 3 shells for cutoff=6Å)
- [unit] half-list invariant: for every (i, j) in list, j > i
- [property ≥10⁴ random configs] completeness: brute-force check — все pairs in cutoff+skin found
- [property ≥10⁴] displacement cert: move atoms by known Δ, check `needs_rebuild` triggers at threshold
- [determinism] same input → same neighbor list (bit-exact)

## Expected artifacts
- Single PR
- Perf note: build time для 500/5000 atoms

## Acceptance criteria
- [ ] Completeness property test green (10⁴+ cases)
- [ ] Half-list invariant property test green
- [ ] Determinism test green (same input → bit-exact output)
- [ ] Human review approved

## Hints
- Brute-force reference O(N²) — OK for testing, used inside property tests for correctness comparison
- `displacement_cert` — track `max_disp_since_build` per atom, reduce when rebuilding
- Master spec §6.1 диктует CSR layout для production; в M1 допускается vec-of-vec (M2+ переход на CSR — отдельная task)
```

---

### T1.7 — `VelocityVerletIntegrator` NVE

```
# TDMD Task: Velocity-Verlet NVE integrator

## Context
- Master spec: §14 M1, integrator/SPEC.md §7.3 (NVT в TD — M9+, в M1 только NVE)
- Role: Physics Engineer
- Depends on: T1.1 (AtomSoA)
- Milestone: M1

## Goal
Реализовать classical velocity-Verlet integration step (3 sub-steps: velocity half-kick, position full drift, force recompute [caller responsibility], velocity half-kick). NVE only — no thermostat/barostat.

## Scope
- [included] `VelocityVerletIntegrator::pre_force_step(atom_soa, dt)` — half-kick velocities + full drift positions
- [included] `VelocityVerletIntegrator::post_force_step(atom_soa, dt)` — half-kick velocities after new forces
- [included] `dt` validation: positive, finite (NaN-guard)
- [included] Mass access через `AtomSoA::mass(type_idx)` → `Species` lookup
- [included] Force zeroing — caller responsibility ДО potential evaluation (документировать)
- [included] Kinetic energy computation: `kinetic_energy(atom_soa) -> double` (используется в telemetry)

## Out of scope
- [excluded] NVT/NPT — M9
- [excluded] Langevin — post-v1
- [excluded] Per-atom timestep (TD-related) — M4
- [excluded] Symplectic higher-order — not planned

## Mandatory invariants
- Time-reversal symmetry: integrate N steps forward + N backward с `-dt` → positions within `1e-10 Å` bit-error
- Energy drift в NVE: на canonical T1 bench (500 Al, 100 steps, dt=0.001 ps) energy drift < `1e-5` relative
- Deterministic: same state + same dt → same output (bit-exact)

## Required files
- `src/integrator/velocity_verlet.h`, `src/integrator/velocity_verlet.cpp` (replace skeleton)
- `src/integrator/CMakeLists.txt`
- `tests/integrator/test_velocity_verlet.cpp` — расширить

## Required tests
- [unit] single-atom free flight: position linear в time, velocity constant (zero force)
- [unit] harmonic oscillator: 1 atom, spring force, period check within 1%
- [unit] two-body Morse (closed-form): 10⁴ steps, energy drift < 1e-5
- [property] time-reversal: N forward + N backward → bit-exact recovery within 1e-10
- [determinism] same inputs → bit-exact outputs

## Expected artifacts
- Single PR
- Energy-conservation plot в PR description (500 atoms Al, 1000 steps, drift curve)

## Acceptance criteria
- [ ] Time-reversal property test green
- [ ] Energy drift unit test green (< 1e-5 relative)
- [ ] Determinism test green
- [ ] Human review approved

## Hints
- Use kahan summation для forces accumulation если видишь drift — но в M1 с 500 atoms простой `sum +=` ОК
- `kinetic_energy = 0.5 · Σ m_i · (vx² + vy² + vz²)`
- Harmonic oscillator analytic: `T = 2π · sqrt(m/k)`, compare numeric period to analytic
```

---

### T1.8 — `MorsePotential` CPU analytic

```
# TDMD Task: Morse pair potential

## Context
- Master spec: §14 M1 ("MorsePotential"), potentials/SPEC.md
- Role: Physics Engineer
- Depends on: T1.1 (AtomSoA), T1.6 (NeighborList)
- Milestone: M1

## Goal
Первый потенциал проекта. Morse pair — классический analytic, nearest-neighbor style, идеальный baseline для T1 differential vs LAMMPS `pair_style morse`.

## Scope
- [included] `MorsePotential::compute(atom_soa, neighbor_list, box)` — computes forces + potential energy + virial
- [included] Formula: `U(r) = D·[exp(-2α(r-r₀)) - 2·exp(-α(r-r₀))]`, cutoff `r_c` — shifted-force или shifted-potential (выбор в task; LAMMPS uses shifted-potential default)
- [included] Force kernel: `F_ij = -dU/dr · r̂_ij` computed analytically (NO finite differences)
- [included] Parameters per species pair: `D, α, r₀, r_c` — supplied via config или from `potential.morse.yaml` chunk в T1.4
- [included] Virial tensor computation (Clausius): `W_αβ = Σ F_ij_α · r_ij_β` — нужен для pressure diagnostics
- [included] Energy partition: per-atom energy optional field (disabled default)

## Out of scope
- [excluded] EAM, MEAM, SNAP — M2, M10, M8
- [excluded] Multi-species with different pair params — M2 (в M1 только single-species Al)
- [excluded] Long-range tail corrections — post-v1
- [excluded] Hybrid potentials (pair + 3-body) — M10+

## Mandatory invariants
- Newton's 3rd law: for each pair, `F_ij = -F_ji` (via half-list accumulation — каждая pair вносит в obе atoms)
- Energy is minimum at `r = r₀` (test — numerical minimization recovers r₀)
- Force is zero at `r = r₀`
- NO allocations in hot path (inner loop) — all scratch buffers pre-allocated
- `__restrict__` on pointer params per master spec §D.16 (NOLINT acceptable with rationale)

## Required files
- `src/potentials/morse.h`, `src/potentials/morse.cpp`
- `src/potentials/CMakeLists.txt`
- `tests/potentials/test_morse.cpp`
- `tests/potentials/fixtures/morse_al.yaml` — canonical Al Morse params (Girifalco-Weizer 1959: D=0.2703 eV, α=1.1646 Å⁻¹, r₀=3.253 Å, r_c=8.0 Å)

## Required tests
- [unit] energy at r=r₀ is minimum (dU/dr=0 numerically)
- [unit] force at r=r₀ is zero (within 1e-12)
- [unit] Newton 3rd law: compute forces, sum = 0 (momentum conservation)
- [unit] virial trace + kinetic energy = 3·N·k_B·T·V relation (if at equilibrium)
- [property ≥10³] per pair: force = -analytical_derivative(energy) within 1e-8

## Expected artifacts
- Single PR
- Fixture file with Al Morse parameters cited

## Acceptance criteria
- [ ] All 5 test categories green
- [ ] Newton 3rd law test — sum of forces ~ 1e-10 relative (not 0 due to rounding)
- [ ] No allocations in hot path (verified via Catch2 BENCHMARK macro or manual profiling)
- [ ] No unnecessary warnings
- [ ] Human review approved

## Hints
- Shifted potential (LAMMPS default): `U_shifted(r) = U(r) - U(r_c)` for `r < r_c`, else 0
- For Al reference params см. Girifalco & Weizer, Phys. Rev. 114, 687 (1959) — open access
- `__restrict__` example: `void compute(const double* __restrict__ x, double* __restrict__ fx, ...)`
```

---

### T1.9 — `tdmd run` CLI + SimulationEngine skeleton (no TD)

```
# TDMD Task: Top-level run CLI + engine wiring

## Context
- Master spec: §14 M1 (CLI), runtime/SPEC.md
- Role: Core Runtime Engineer
- Depends on: T1.1, T1.2, T1.3, T1.4, T1.5, T1.6, T1.7, T1.8
- Milestone: M1

## Goal
Собрать всё вместе: `tdmd run config.yaml` — читает config, строит AtomSoA/Box/Species, инициализирует CellGrid + NeighborList, запускает velocity-Verlet loop с Morse forces, выдаёт thermo output. НЕТ TD, НЕТ MPI, НЕТ GPU — чистый single-rank reference.

## Scope
- [included] `SimulationEngine` (runtime/): lifecycle = `init(config)` → `run(nsteps)` → `finalize()`
- [included] Main loop structure:
  ```
  for step in 0..nsteps:
    if needs_neighbor_rebuild: cell_grid.bin+reorder; neighbor_list.build
    potentials.compute → forces
    thermo every N steps
    integrator.pre_force_step (half-kick + drift)
    potentials.compute → forces
    integrator.post_force_step (half-kick)
  ```
- [included] `tdmd run <config.yaml>` CLI через cxxopts (D-M1-4)
- [included] Thermo output: step, temp, pe, etotal, press — LAMMPS-compatible format (whitespace-separated, header row)
- [included] Dispatch of "config → engine" через `SimulationEngine::from_config(config, preflight_result)` factory
- [included] Exit codes: 0 success, 2 preflight failure, 1 runtime error

## Out of scope
- [excluded] Thermostat/barostat — M9
- [excluded] Dump файлы — M2
- [excluded] TD scheduler — M4
- [excluded] MPI — M5
- [excluded] GPU — M6
- [excluded] Restart/resume — M2

## Mandatory invariants
- Engine lifecycle strict: init → run → finalize; double-init rejected
- Ownership: engine owns AtomSoA, Box, Species; potentials/integrator ТОЛЬКО read or write forces via non-owning view
- Thermo output deterministic bit-exact для identical run (no timestamps, no nondeterministic order)

## Required files
- `src/runtime/simulation_engine.h`, `src/runtime/simulation_engine.cpp`
- `src/cli/main.cpp`, `src/cli/run_command.h`, `src/cli/run_command.cpp`
- `src/cli/CMakeLists.txt` — add executable target `tdmd`
- `CMakeLists.txt` root — ensure `tdmd` executable exported
- `tests/runtime/test_simulation_engine.cpp`
- `tests/cli/test_run_command.cpp` (end-to-end invoke `tdmd run` via temp config)

## Required tests
- [unit] engine lifecycle: init → run → finalize; bad sequences rejected
- [integration] run Al FCC 500 for 10 steps, verify thermo output has expected columns + 1 line per thermo step
- [determinism] same config → identical thermo output (bit-exact)

## Expected artifacts
- Single PR
- `tdmd --help` and `tdmd run --help` sample outputs в PR description
- Example config `examples/al_fcc_500_nve.yaml` — full working config

## Acceptance criteria
- [ ] `tdmd run examples/al_fcc_500_nve.yaml` completes without errors
- [ ] Thermo output шапка + 10 lines (for 100 steps, thermo every 10)
- [ ] Determinism test green
- [ ] Pre-impl + session reports attached
- [ ] Human review approved

## Hints
- cxxopts examples: https://github.com/jarro2783/cxxopts/blob/master/README.md
- Engine = composition of sub-objects; НЕ god-class. SimulationEngine holds pointers/refs to stage-specific objects.
- Main loop — simple linear; не сейчас разбивать на stages/policies (premature). Рефакторинг в M4 при добавлении TD.
```

---

### T1.10 — `tdmd validate` CLI

```
# TDMD Task: Validate subcommand

## Context
- Master spec: §14 M1 (CLI: tdmd validate)
- Role: Scientist UX Engineer
- Depends on: T1.4 (preflight)
- Milestone: M1

## Goal
`tdmd validate config.yaml` — standalone preflight run БЕЗ запуска simulation. Выводит все preflight errors (если есть) или "OK" + summary параметров. Используется пользователем перед долгим run'ом.

## Scope
- [included] Parse yaml → run preflight → print results
- [included] On success: print short summary (units, atoms source, steps, timestep)
- [included] On failure: print all errors with line numbers если доступно, exit 2
- [included] `--strict` flag: treat preflight warnings as errors
- [included] `--explain <field>` — human-readable explanation of what a field does (documentation hook; в M1 minimal set of fields)

## Out of scope
- [excluded] Interactive mode / config generator — post-v1
- [excluded] Config migration (old → new format) — post-v1

## Mandatory invariants
- `tdmd validate` не читает `.data` файлы или не запускает simulation — только preflight
- Exit code semantics: 0 OK, 2 preflight errors, 1 internal/IO error

## Required files
- `src/cli/validate_command.h`, `src/cli/validate_command.cpp`
- `src/cli/CMakeLists.txt` — add
- `tests/cli/test_validate_command.cpp`

## Required tests
- [integration] `tdmd validate` on valid config → exit 0
- [integration] `tdmd validate` on missing_units.yaml → exit 2, stderr contains "units is required"
- [integration] `--strict` promotes warnings to errors

## Expected artifacts
- Single PR
- `tdmd validate --help` sample output

## Acceptance criteria
- [ ] All 5 config fixtures (T1.4) validated с correct exit codes
- [ ] `--strict` works
- [ ] Human review approved

## Hints
- Re-use preflight module T1.4
- `--explain` в M1 — только для `units` field (text copied from master spec §5.3 summary)
```

---

### T1.11 — T1 differential harness (Al FCC Morse vs LAMMPS)

```
# TDMD Task: First differential test against LAMMPS oracle

## Context
- Master spec: §13.2 (canonical benchmarks), §13.3 (differential test strategy), verify/SPEC.md §5 (T1 benchmark)
- Role: Validation / Reference Engineer
- Depends on: T1.9 (tdmd run), T1.3 (data importer), LAMMPS oracle (M0 T0.7 — already built)
- Milestone: M1 (this is the M1 artifact gate enabler)

## Goal
Первый numerical validation gate проекта. TDMD прогоняет Al FCC 500 atoms + Morse + NVE + 100 steps; LAMMPS прогоняет то же самое; сравнение: positions, velocities, forces, energies, pressure — все в SI-units через UnitConverter; threshold registered в verify/SPEC.md.

## Scope
- [included] Reference benchmark spec: `verify/benchmarks/t1_al_morse_500.yaml`
  - Units: metal
  - System: Al FCC 5×5×5 lattice 4.05 Å = 500 atoms
  - Potential: Morse D=0.2703 eV, α=1.1646 Å⁻¹, r₀=3.253 Å, r_c=8.0 Å
  - Integrator: NVE, dt=0.001 ps, 100 steps
  - Initial: velocity distribution @ 300 K, seed=12345
- [included] LAMMPS input generator: `verify/t1/generate_lammps_input.py` → emits LAMMPS script matching TDMD config
- [included] Test harness `verify/t1/run_differential.py`:
  1. Generate LAMMPS + TDMD inputs from common YAML
  2. Run LAMMPS oracle → capture thermo + dump
  3. Run TDMD → capture thermo + dump
  4. Compare via `verify/compare.py`: max|Δpos| < threshold, max|Δvel| < threshold, max|Δenergy_rel| < threshold
- [included] Threshold registration in `verify/thresholds.yaml`:
  - `max_pos_diff: 1e-10 Å`
  - `max_vel_diff: 1e-10 Å/ps`
  - `energy_rel_diff: 1e-12`
  - (SI units explicit per master spec auto-reject §5)
- [included] CI Pipeline C (differential) — new job `differential-t1` в `.github/workflows/ci.yml`: builds TDMD (cpu-only preset) + uses pre-built LAMMPS submodule + runs harness

## Out of scope
- [excluded] T2, T3, T4 benchmarks — M2, M3, M5
- [excluded] GPU differential — M6 T4
- [excluded] Performance comparison (only correctness в M1) — M2

## Mandatory invariants
- Thresholds в SI units (no unitless `< 0.01`) — master spec auto-reject
- Harness deterministic: same inputs → same pass/fail
- LAMMPS oracle не rebuild'ится в CI — используется pre-built из M0 T0.7 (если нет — test skipped с clear message, не fail)
- Threshold tighter для Reference path — 1e-10 Å (master spec §D: Reference is bitwise oracle)

## Required files
- `verify/benchmarks/t1_al_morse_500.yaml`
- `verify/t1/generate_lammps_input.py`
- `verify/t1/run_differential.py`
- `verify/compare.py` (generic comparison lib)
- `verify/thresholds.yaml` — threshold registry (new file, master)
- `verify/t1/test_t1_differential.cpp` — Catch2 wrapper that invokes python harness + parses result
- `.github/workflows/ci.yml` — add `differential-t1` job (cpu-only, uses local LAMMPS if present, else skip with warning)
- `docs/specs/verify/SPEC.md` — update §5 threshold registry с T1 row (SPEC edit — отдельный commit per playbook §9.1; OR inline if no interface change)

## Required tests
- [integration] run harness locally → all thresholds pass
- [integration] intentional drift (e.g. wrong Morse D) → harness fails with clear diagnostic
- [ci] differential-t1 job runs on PR, passes на valid M1 impl, fails с diff output на broken one
- [threshold-docs] `verify/thresholds.yaml` parseable by compare.py

## Expected artifacts
- PR содержит:
  - Full differential harness
  - Thresholds file
  - Updated verify/SPEC.md threshold table
  - Passing CI run на test PR
- Pre-impl + session reports в PR description

## Acceptance criteria
- [ ] T1 harness runs locally и в CI (when oracle available)
- [ ] All thresholds in SI units
- [ ] Pass on correct M1 impl
- [ ] Fail loudly on planted bug (test harness self-check)
- [ ] verify/SPEC.md updated without changing threshold-registry contract (additive only)
- [ ] Human review approved (including Architect role для SPEC delta)

## Hints
- LAMMPS script template: base на `tools/lammps_smoke_test.in` (M0 T0.7)
- Compare in SI: convert both sides via UnitConverter before compare
- Catch2 wrapper через `std::system` или `subprocess` lib — keep simple; python harness does heavy lifting
- Если LAMMPS binary отсутствует — test SKIP, не FAIL; clear message pointing to tools/build_lammps.sh
```

---

### T1.12 — M1 integration smoke: end-to-end Al FCC run

```
# TDMD Task: End-to-end M1 smoke test

## Context
- Role: Validation / Reference Engineer
- Depends on: T1.9, T1.11
- Milestone: M1 (final)

## Goal
Тонкий smoke test, проверяющий что полный pipeline M1 работает: config → data ingest → build → run → thermo → differential. Запускается как CI Pipeline B (unit+integration fast tier).

## Scope
- [included] `tests/integration/m1_smoke/run_al_morse_smoke.sh`:
  1. Generates test config from template
  2. Runs `tdmd validate`
  3. Runs `tdmd run` for 10 steps (short для smoke, не full T1)
  4. Greps thermo output для expected keys
  5. Compares final `etotal` с cached reference (golden file)
- [included] Golden file: `tests/integration/m1_smoke/al_fcc_500_10steps_golden.txt`
- [included] CI integration: `.github/workflows/ci.yml` — добавить step "M1 smoke" в build-cpu jobs после tests
- [included] Documentation: `tests/integration/m1_smoke/README.md` — что тест проверяет, как regenerate golden, когда golden обновлять

## Out of scope
- [excluded] Full 100-step T1 run — это T1.11 harness
- [excluded] GPU — M6
- [excluded] Multi-rank — M5

## Mandatory invariants
- Wall-time smoke < 10 секунд (CI budget)
- Golden file regeneration — documented procedure, requires manual review
- Smoke runs in cpu-only preset (CI build)

## Required files
- `tests/integration/m1_smoke/run_al_morse_smoke.sh`
- `tests/integration/m1_smoke/smoke_config.yaml.template`
- `tests/integration/m1_smoke/al_fcc_500_10steps_golden.txt`
- `tests/integration/m1_smoke/README.md`
- `.github/workflows/ci.yml` — new step in build-cpu jobs

## Required tests
- [smoke] runs locally в < 10s, exit 0
- [smoke] CI integrates (new step runs on every PR)
- [regression] intentional bug (e.g. wrong Morse D) → smoke fails on golden comparison

## Expected artifacts
- Single PR
- Golden file committed (small text file, not LFS)
- CI green на test PR

## Acceptance criteria
- [ ] Smoke < 10s
- [ ] Golden matches within bit-exact tolerance для identical compile+arch
- [ ] CI integration verified (PR shows 5th check "M1 smoke" ... pass)
- [ ] Human review approved
- [ ] M1 Acceptance Gate (§5) полностью закрыт

## Hints
- Golden regeneration procedure: `./run_al_morse_smoke.sh --update-golden` gated by `TDMD_UPDATE_GOLDENS=1` env var
- Bit-exactness — возможен только на same compiler+arch; CI запускает на hosted runner (deterministic Ubuntu), golden generated там же
```

---

## 5. M1 Acceptance Gate

После закрытия всех 12 задач — проверить полный M1 artifact gate (master spec §14):

- [ ] **AtomSoA, Box, Species** real impl (T1.1), unit+property tests green
- [ ] **UnitConverter** с metal native + lj stub (T1.2)
- [ ] **LAMMPS .data importer** working (T1.3), round-trip verified
- [ ] **tdmd.yaml parser + preflight** (T1.4); preflight rejects missing `units:` (master spec §14 M1 gate)
- [ ] **CellGrid + stable reorder** (T1.5)
- [ ] **Neighbor list half + newton on** (T1.6), completeness/half-list invariants green
- [ ] **VelocityVerletIntegrator NVE** (T1.7), time-reversal + energy drift tests green
- [ ] **MorsePotential** with virial (T1.8), Newton 3rd law verified
- [ ] **tdmd run** CLI end-to-end (T1.9), deterministic thermo output
- [ ] **tdmd validate** CLI (T1.10), exit codes correct
- [ ] **T1 differential test** (T1.11) — **mandatory gate** — green against LAMMPS oracle; thresholds in SI units in verify/thresholds.yaml
- [ ] **M1 smoke** (T1.12), < 10s, CI-integrated
- [ ] CI Pipelines A (lint+build) + B (unit/integration) + C (differential, T1) active на required checks
- [ ] Pre-implementation + session reports attached в каждом PR description
- [ ] Human review approval для каждого PR

После прохождения gate — Architect создаёт `docs/development/m2_execution_pack.md`.

---

## 6. Risks и open questions для M1

### Risks

- **R-M1-1: T1 differential harness скоуп underestimate.** Парсинг LAMMPS dump + comparison в SI — несколько moving pieces. **Mitigation:** T1.11 — dedicated task с 1 wk budget; LAMMPS oracle уже built (M0); harness минимальный (Morse, 100 steps, 500 atoms).
- **R-M1-2: Energy conservation drift в NVE.** Может не достичь `< 1e-5` relative на 100 steps если virial/force consistency нарушена. **Mitigation:** T1.7 + T1.8 cross-test — compute `dU/dr` numerically из energy, compare to analytic force; also time-reversal test в T1.7.
- **R-M1-3: Determinism на Catch2 property tests.** `random()` в Catch2 использует global state; недетерминизм между runs. **Mitigation:** property test generators seeded explicitly; test сам self-checks `GENERATE(random(seed=...))`.
- **R-M1-4: yaml-cpp dependency.** FetchContent добавляет build time; version pinning риск. **Mitigation:** try `find_package(yaml-cpp)` first (system package), fall back to FetchContent 0.8.0 pinned tag.
- **R-M1-5: Morse parameters accuracy.** Girifalco-Weizer Al Morse ≠ true Al interaction (LAMMPS eam/alloy Al_zhou более accurate). M1 использует Morse для simplicity; NOT physically realistic, но differential test сравнивает ТОТ ЖЕ potential на обеих сторонах — correct для validation goal. **Mitigation:** documented в T1 benchmark description.

### Open questions

- **OQ-M1-1: Shifted-potential vs shifted-force Morse?** LAMMPS `pair_style morse` default shifted-potential; нужен same behavior для bitwise T1 match. **Default:** shifted-potential (matches LAMMPS). Decision в T1.8.
- **OQ-M1-2: Catch2 v3 vs Catch2 v2?** M0 T0.5 зафиксировал v3 via FetchContent. M1 не меняет. OK.
- **OQ-M1-3: `tdmd explain --perf` — delivers в M1 или M2?** Master spec §14 M2 says explain с perf — M2. **Default:** M2. M1 `tdmd explain` — out of scope (no task).
- **OQ-M1-4: Per-atom energy storage — always computed или opt-in?** Memory for 500 atoms — trivial; для 10⁶ atoms (future) — 8 MB. **Default:** opt-in (config field), disabled в M1 testbench; implemented stub в T1.8.
- **OQ-M1-5: LAMMPS dump format для diff — `custom` с explicit column order, или default?** Need deterministic column order для compare.py. **Default:** `dump 1 all custom 1 dump.txt id type x y z vx vy vz` explicit fields (escalate в T1.11 если окажется что precision ≠ sufficient).

---

## 7. Что НЕ входит в M1 (предотвращение scope creep)

Эти items могут показаться "почти M1" но строго отнесены к M2+:

- ❌ **EAM potential** — M2 (требует отдельной module + density/force two-pass)
- ❌ **LJ unit system real impl** — M2 (M1 — API stub)
- ❌ **PerfModel** — M2
- ❌ **Dump files** (beyond thermo stdout) — M2
- ❌ **Restart/resume** — M2
- ❌ **`tdmd explain --perf`** — M2
- ❌ **Telemetry skeleton** — M2
- ❌ **Zoning (TD spatial decomposition)** — M3
- ❌ **TD scheduler** — M4
- ❌ **Multi-rank MPI** — M5
- ❌ **GPU kernels** — M6
- ❌ **NVT/NPT integrators** — M9
- ❌ **Langevin, MC, reactive MD** — post-v1
- ❌ **Triclinic box support tests** — M3+ (API поля zeroed в M1)
- ❌ **Charges, full atom_style** — M2
- ❌ **Stable cross-compiler bit-exact determinism** — M4 (scheduler milestone)

При искушении агента "заодно сделать X" — `claude_code_playbook.md §1.1 п.7` (no scope creep) применяется автоматически.

---

## 8. Change log пакета

### v0.1 (2026-04-17)

- Initial draft. 12 задач, 10 решений, 5 risks, 5 open questions.
- Approved decisions: yaml-cpp, Morse first potential, Al FCC 500 T1 benchmark, cxxopts CLI, NVE-only, metal-only units с lj stub, half-list neighbor, no restart, Fp64Reference only, same-binary determinism.

---

*Конец M1 Execution Pack v0.1.*
