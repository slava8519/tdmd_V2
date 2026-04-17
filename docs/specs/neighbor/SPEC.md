# neighbor/SPEC.md

**Module:** `neighbor/`
**Status:** master module spec
**Parent:** `TDMD Engineering Spec v2.1` §5.1, §6.3, §12
**Last updated:** 2026-04-16

---

## 1. Purpose и scope

### 1.1. Что делает модуль

`neighbor/` — модуль **локальности**. Отвечает на единственный вопрос: «какие атомы находятся рядом с каждым атомом, с достаточной для валидного расчёта сил точностью, в пределах cutoff + skin?».

Формально — четыре подсистемы:

1. **Cell grid** — пространственная сетка с шагом ≥ `r_c + r_skin`, быстрый binning атомов по ячейкам.
2. **Neighbor list** — list neighbors per atom (или per zone в TDMD), с accompanying distances, построен из cell grid.
3. **Skin tracker** — отслеживание накопленного displacement атомов с момента последнего rebuild, trigger rebuild когда skin исчерпан.
4. **Migration** — перемещение атомов между зонами (и в Pattern 2 — между subdomain'ами) с preservation atom identity.

### 1.2. Scope: что НЕ делает

- **не владеет атомами** (это `state/`);
- **не вычисляет силы** (это `potentials/`);
- **не решает когда rebuild'ить** с т.з. scheduling (trigger emit, но scheduler решает когда обработать);
- **не строит zones** (это `zoning/`);
- **не упаковывает атомы для сети** (это `comm/`).

Neighbor — исполнитель с жёсткими контрактами: given state + zoning plan + skin, produces neighbor list. Pure transformation.

### 1.3. Три отдельных слоя — принципиально

Мастер-спец §8.2 требует: **neighbor rebuild / migration / stable reorder** — три разные операции, которые не должны смешиваться в скрытые side effects. Это частая ошибка MD-кодов. TDMD этого избегает жёстко:

- **rebuild** — пересчёт пар соседей без изменения atom index;
- **migration** — изменение `atom → zone` mapping, с возможным изменением индексов;
- **reorder** — физическое переупорядочивание массивов `AtomSoA` для locality, stable sort.

Каждая операция имеет свой API, свой telemetry event, свой set triggers. Composition возможна (rebuild может вызвать reorder), но explicit.

---

## 2. Public interface

### 2.1. Базовые типы

```cpp
namespace tdmd {

struct CellGrid {
    double                cell_x, cell_y, cell_z;   // размер ячейки ≥ r_c + r_skin
    uint32_t              nx, ny, nz;
    std::vector<uint32_t> cell_offsets;             // CSR-style: offsets[i]..offsets[i+1]
    std::vector<AtomId>   cell_atoms;               // atom indices per cell, sorted
    uint64_t              build_version;
};

struct NeighborList {
    std::vector<uint64_t>  page_offsets;            // CSR: offset in neigh_ids per atom
    std::vector<AtomId>    neigh_ids;
    std::vector<float>     neigh_r2;                // squared distance (for reuse in force)
    double                 cutoff;
    double                 skin;
    uint64_t               build_version;           // incremented on each rebuild
    uint64_t               associated_state_version; // state version at build time
};

struct DisplacementTracker {
    std::vector<double>    x_at_build, y_at_build, z_at_build;
    double                 max_displacement_since_build;   // tracked across all atoms
    double                 threshold;                       // usually r_skin / 2
    uint64_t               associated_build_version;
};

struct MigrationRecord {
    AtomId          atom_id;
    ZoneId          from_zone;
    ZoneId          to_zone;
    TimeLevel       time_level;
    uint64_t        version;
};

struct ReorderMap {
    std::vector<uint64_t>  old_to_new;    // old_to_new[i] = new position of atom i
    std::vector<uint64_t>  new_to_old;    // inverse
    uint64_t               applied_at_version;
};

} // namespace tdmd
```

### 2.2. Главный интерфейс

```cpp
namespace tdmd {

class NeighborManager {
public:
    // Cell grid:
    virtual void  build_cell_grid(
        const AtomSoA&, const Box&,
        double cutoff, double skin) = 0;
    virtual const CellGrid&  cell_grid() const = 0;

    // Neighbor list:
    virtual void  build_neighbor_list(
        const AtomSoA&, const CellGrid&,
        const ZoningPlan&) = 0;
    virtual const NeighborList&  neighbor_list() const = 0;

    // Displacement tracking:
    virtual void  init_displacement_tracker(const AtomSoA&) = 0;
    virtual void  update_displacement(const AtomSoA&) = 0;   // called per iteration
    virtual double  max_displacement() const = 0;
    virtual bool  skin_exceeded() const = 0;

    // Rebuild orchestration (emits trigger, does NOT decide when to rebuild):
    virtual void  request_rebuild(const std::string& reason) = 0;
    virtual bool  rebuild_pending() const = 0;
    virtual void  execute_rebuild(AtomSoA&, const ZoningPlan&) = 0;

    // Migration (separate operation!):
    virtual std::vector<MigrationRecord>  detect_migrations(
        const AtomSoA&, const ZoningPlan&) const = 0;
    virtual void  apply_migrations(
        AtomSoA&, const ZoningPlan&,
        const std::vector<MigrationRecord>&) = 0;

    // Stable reorder (separate operation!):
    virtual ReorderMap  compute_stable_reorder(
        const AtomSoA&, const ZoningPlan&) const = 0;
    virtual void  apply_reorder(AtomSoA&, const ReorderMap&) = 0;

    virtual ~NeighborManager() = default;
};

class DefaultNeighborManager final : public NeighborManager { /*...*/ };

} // namespace tdmd
```

### 2.3. Events

Neighbor emit'ит события в telemetry и в scheduler:

| Event | Когда | Payload |
|---|---|---|
| `NeighborRebuildStarted` | start of execute_rebuild | affected zones |
| `NeighborRebuildCompleted` | end of execute_rebuild | affected zones, new build_version |
| `MigrationDetected` | detect_migrations returned non-empty | migration records |
| `ReorderApplied` | apply_reorder | reorder map summary |
| `SkinExceeded` | displacement > threshold | current max displacement |

Scheduler (§10 scheduler/SPEC) реагирует на `NeighborRebuildCompleted`, `MigrationDetected` invalidate certificates в затронутых зонах.

---

## 3. Cell grid

### 3.1. Инвариант размера

```
cell_x ≥ r_c + r_skin
cell_y ≥ r_c + r_skin
cell_z ≥ r_c + r_skin
```

Если не соблюдается — preflight error. Это критический инвариант: без него stencil cell search (3×3×3 = 27 cells) не покрывает all possible neighbors, и force evaluation даёт wrong results.

### 3.2. Binning algorithm

```
function build_cell_grid(atoms, box, r_c, r_skin):
    w = r_c + r_skin
    nx = max(1, floor((box.x_hi - box.x_lo) / w))
    ny = max(1, floor((box.y_hi - box.y_lo) / w))
    nz = max(1, floor((box.z_hi - box.z_lo) / w))

    # Actual cell sizes (slightly larger than w to cover box exactly):
    cell_x = (box.x_hi - box.x_lo) / nx
    cell_y = (box.y_hi - box.y_lo) / ny
    cell_z = (box.z_hi - box.z_lo) / nz

    # Pass 1: count atoms per cell
    counts = [0] * (nx·ny·nz)
    for  atom in atoms:
        ix = floor((atom.x - box.x_lo) / cell_x)
        iy = floor((atom.y - box.y_lo) / cell_y)
        iz = floor((atom.z - box.z_lo) / cell_z)
        cell_id = ix + nx·iy + nx·ny·iz
        counts[cell_id]++

    # Pass 2: exclusive prefix sum → cell_offsets
    cell_offsets = exclusive_prefix_sum(counts)

    # Pass 3: fill cell_atoms (stable, deterministic)
    position = copy(cell_offsets)
    for  atom_idx in 0..N_atoms - 1:  # ordered iteration
        cell_id = compute_cell_id(atom_idx)
        cell_atoms[position[cell_id]] = atom_idx
        position[cell_id]++

    return CellGrid{cell_x, cell_y, cell_z, nx, ny, nz, cell_offsets, cell_atoms}
```

**Deterministic ordering** в `cell_atoms`: атомы внутри каждой ячейки упорядочены по исходному `atom_idx` (не по id), что гарантирует reproducibility независимо от atom id distribution.

### 3.3. GPU variant

На GPU binning делается двумя kernel'ами:
1. `compute_cell_ids_kernel` — per-atom, вычисляет cell_id;
2. `stable_radix_sort` — sort atoms by cell_id, stable (preserves input order within equal keys).

Critical: `stable_radix_sort` — именно stable. Нестабильная сортировка нарушает determinism в Reference profile.

### 3.4. Periodic wrap

Perioduc BC обрабатывается в `compute_cell_id`:
- if `periodic_x` и атом вышел за `[x_lo, x_hi]` — wrap до primary image ПЕРЕД cell id computation;
- wrap делается on-the-fly без modification `AtomSoA.x` (это делает integrator отдельно при необходимости).

Здесь важно: **не мутировать state в neighbor module**. Wrap — это query-time преобразование.

---

## 4. Neighbor list

### 4.1. Structure: per-zone vs per-atom

В TDMD neighbor list — **per-zone**, а не традиционный per-atom. Причина: zone — primary unit of work в scheduler. Для зоны `z`, list содержит атомы, попадающие в radius `r_c + r_skin` от любого атома зоны.

Но внутренне хранится как **per-atom** (как в LAMMPS), а группировка per-zone достигается через `ZoningPlan::zone_to_atoms` mapping из `state/`. Это упрощает force kernels (per-atom loop) и позволяет reuse LAMMPS-совместимых potential implementations.

### 4.2. Build algorithm

```
function build_neighbor_list(atoms, cell_grid, zoning):
    r_c = potential.cutoff()
    r_skin = user_config.skin
    r_cutoff_sq = (r_c + r_skin)^2

    page_offsets = []
    neigh_ids = []
    neigh_r2 = []

    for  i in 0..N_atoms - 1:  # ordered iteration for determinism
        page_offsets.append(len(neigh_ids))

        ix, iy, iz = atom_to_cell(i)
        for  dx, dy, dz in [-1, 0, 1]³:  # 27-neighbor stencil
            neighbor_cell = (ix+dx, iy+dy, iz+dz)
            if  is_periodic_wrap(neighbor_cell):
                wrap neighbor_cell to primary image

            for  j in cell_atoms[neighbor_cell]:
                if  j == i:  continue

                dr² = squared_distance(i, j, with periodic wrap)
                if  dr² <= r_cutoff_sq:
                    # Newton's third law optimization:
                    if  newton_enabled  and  j < i:
                        continue     # pair (i,j) will be covered when iterating j
                    neigh_ids.append(j)
                    neigh_r2.append(dr²)

    page_offsets.append(len(neigh_ids))   # sentinel

    return NeighborList{page_offsets, neigh_ids, neigh_r2, cutoff=r_c, skin=r_skin,
                         build_version=new, associated_state_version=state.version}
```

### 4.3. Newton's third law

По умолчанию `newton_enabled = true` — хранятся только пары `(i, j)` с `j > i`. Это **вдвое** сокращает list size и compute.

Но в Pattern 2 (two-level) на boundary атомах Newton может быть disabled для корректности halo: если atom `j` лежит в другом subdomain, мы всё равно должны учитывать pair `(i, j)` в нашем subdomain'е, потому что там atom `j` — ghost copy и не «owner» pair computation.

Policy: `newton_per_zone_flag[zone]` — boolean. По умолчанию true для internal zones, false для boundary zones.

### 4.4. Valid-until prediction

`neighbor_list.valid_until_step` — прогноз scheduler'ом момента, когда neighbor list станет stale:

```
function predict_valid_until_step(neigh_list, current_step, v_max_global, dt):
    r_skin_remaining = r_skin - tracker.max_displacement
    steps_to_exhaustion = floor(r_skin_remaining / (v_max_global · dt))
    return current_step + steps_to_exhaustion
```

Consumed by `SafetyCertificate::neighbor_valid_until_step` (§4 scheduler/SPEC).

---

## 5. Skin tracker

### 5.1. Algorithm

```
function update_displacement(atoms, tracker):
    max_d = 0
    for  i in 0..N_atoms - 1:
        dx = atoms.x[i] - tracker.x_at_build[i]
        dy = atoms.y[i] - tracker.y_at_build[i]
        dz = atoms.z[i] - tracker.z_at_build[i]
        d = sqrt(dx² + dy² + dz²)
        if  d > max_d:
            max_d = d
    tracker.max_displacement = max_d
```

### 5.2. Threshold

По умолчанию `threshold = r_skin / 2`. Rebuild запрашивается когда `max_displacement > threshold`, но actual rebuild может быть отложен scheduler'ом до удобного момента (между time steps, не in the middle of compute).

**Почему r_skin / 2, а не r_skin?** Потому что `max_displacement` — оценка single-atom движения, но в neighbor list учитывается displacement **обоих** атомов пары. `d_i + d_j ≤ 2 · max_d`, и для гарантированной корректности нужно `2 · max_d < r_skin`, откуда threshold = `r_skin / 2`.

### 5.3. Per-zone tracking (optimization)

Для очень больших систем (10⁸+ атомов) глобальный max displacement может быть дорогим. Optimization: per-zone tracker, `rebuild_zone_if_needed()` — rebuild только affected zones.

- **v1:** global tracker. Simple, correct.
- **v2+:** per-zone tracker как optimization. Требует careful boundary handling.

---

## 6. Rebuild policy

### 6.1. Triggers

Rebuild может быть запрошен по пяти причинам:

1. `SkinExceeded` — tracker detected displacement > threshold;
2. `MigrationOccurred` — после `apply_migrations`;
3. `PotentialChanged` — изменился `r_c` (редко, обычно между stages run'а);
4. `ManualRequest` — explicit call `request_rebuild("user reason")`;
5. `VersionSkew` — state_version больше чем на `MAX_SKEW` выше associated_state_version (safety).

### 6.2. Execute

```
function execute_rebuild(atoms, zoning):
    build_cell_grid(atoms, box, r_c, r_skin)
    build_neighbor_list(atoms, cell_grid, zoning)
    reset_displacement_tracker(atoms)
    neighbor_list.build_version++
    emit_event NeighborRebuildCompleted(affected_zones, new_version)
```

### 6.3. Scheduler coordination

Scheduler (§5 scheduler/SPEC) listens на `request_rebuild`:
- в Reference profile: rebuild выполняется в конце текущей iteration, после всех `commit_completed`;
- в Production/Fast: rebuild может быть piggy-backed на empty iteration ("no ready tasks this iteration → rebuild instead of idle").

Rebuild **никогда не выполняется** в середине compute phase. Это инвариант.

### 6.4. Incremental rebuild (optimization)

Полный rebuild всех атомов — O(N). Для больших систем — дорого. Optimization: rebuild только affected cells (те, где max displacement exceeded).

- **v1:** full rebuild only. Simple, correct.
- **v2+:** incremental rebuild. Требует careful bookkeeping какие cells dirty.

---

## 7. Migration

### 7.1. Detection

```
function detect_migrations(atoms, zoning):
    migrations = []
    for  atom_idx in 0..N_atoms - 1:
        current_zone = zoning.get_zone_of_atom(atom_idx)   # from cached mapping
        x, y, z = atoms.x[idx], atoms.y[idx], atoms.z[idx]

        # Periodic wrap first:
        if  periodic:
            wrap (x, y, z) to primary image

        new_zone = zoning.compute_zone_for_position(x, y, z)
        if  new_zone != current_zone:
            migrations.append(MigrationRecord(
                atom_id=atoms.id[idx],
                from_zone=current_zone,
                to_zone=new_zone,
                time_level=current_step,
                version=state.version))
    return migrations
```

### 7.2. Apply

```
function apply_migrations(atoms, zoning, migrations):
    if migrations.empty():
        return

    # Update zoning mapping (delegated to zoning/ or state/):
    for  m in migrations:
        zoning.reassign_atom(m.atom_id, m.from_zone, m.to_zone)

    # Signal that reorder may be needed (atoms no longer sorted by zone):
    if  migrations.size() > migration_reorder_threshold:
        emit_event ReorderNeeded(reason="migration")

    # Rebuild requested (current neighbor list stale):
    request_rebuild("migration")
```

### 7.3. Atom identity invariant

`MigrationRecord.atom_id` — persistent atom ID. **AtomId не меняется** при migration, reorder или rebuild. Меняется только индекс в `AtomSoA` (при reorder). Identity следит за атомом через всю симуляцию.

Это инвариант: `atoms.id[i]` для atom `A` может меняться (потому что index `i` меняется при reorder), но `atoms.id[index_of(A)]` всегда равен оригинальному ID `A`.

Для dump'а и restart'а identity критична — без неё trajectory нельзя правильно прочесть.

### 7.4. Cross-subdomain migration (Pattern 2)

Если атом пересекает subdomain boundary — migration становится **inter-rank event**, требует коммуникации:

1. Source subdomain detect'ит atom exiting its box → помечает для export.
2. Упаковывает atom в `MigrationPacket`, отправляет target subdomain'у через `comm/`.
3. Target subdomain принимает → добавляет atom в свой `AtomSoA`, рассчитывает new zone.
4. Source удаляет atom из своего `AtomSoA`.

Это coordinated operation между двумя subdomain'ами. В v1 (M7+) делается synchronous — barrier-подобный step между iterations. Async cross-subdomain migration — post-v1 optimization.

---

## 8. Stable reorder

### 8.1. Why stable

После migration или rebuild атомы могут быть "неупорядочены" по zones: атом из zone 5 может быть в памяти между атомами zone 2 и zone 3. Это плохо для cache locality в force kernels.

**Stable reorder** — переупорядочивание `AtomSoA`, так чтобы атомы одной зоны лежали последовательно в памяти. **Stable** = если два атома в одной zone, их relative order сохраняется (важно для determinism).

### 8.2. Algorithm

```
function compute_stable_reorder(atoms, zoning):
    zone_of_atom = [zoning.compute_zone_for_position(atoms.x[i], ...) for i]

    # Stable sort indices by zone_id:
    indices = [0, 1, ..., N_atoms - 1]
    stable_sort(indices by key=lambda i: zone_of_atom[i])

    old_to_new = [0] * N_atoms
    for  new_idx, old_idx in enumerate(indices):
        old_to_new[old_idx] = new_idx

    new_to_old = inverse_permutation(old_to_new)

    return ReorderMap{old_to_new, new_to_old, applied_at_version=state.version}


function apply_reorder(atoms, reorder_map):
    # Apply permutation to every field of AtomSoA:
    for  field in [id, type, x, y, z, vx, vy, vz, fx, fy, fz, flags]:
        atoms.field = permute(atoms.field, reorder_map.new_to_old)

    state.version++
    emit_event ReorderApplied(reorder_map summary)
```

### 8.3. When to reorder

- При первом build neighbor list (initial ordering may be arbitrary);
- После значительной migration (> threshold, default 5% atoms);
- По explicit request scheduler'а (telemetry shows cache misses spike).

**Не на каждой iteration** — reorder тяжёлый (O(N log N) или O(N) для radix), делать его часто contraproductive.

### 8.4. Integrity check

После reorder инвариант:
```
forall atom A:
    let new_idx = reorder_map.old_to_new[old_idx_of(A)]
    assert atoms.id[new_idx] == A.id
    assert atoms.x[new_idx] == A.x_before_reorder
    # ... для всех fields
```

Это — property-тест, обязательный в unit suite.

---

## 9. Determinism policy

### 9.1. Reference profile

- Cell grid binning: stable, deterministic;
- Neighbor list build: ordered iteration по atom_idx;
- Rebuild trigger: deterministic (same displacement state → same decision);
- Migration detection: ordered;
- Reorder: stable sort.

**Zero sources of nondeterminism** в Reference profile.

### 9.2. Production profile

- Allowed: GPU parallel binning with deterministic reduction at end;
- Allowed: incremental rebuild (since cells are deterministic);
- Allowed: async displacement tracking (с periodic consolidation).

### 9.3. Fast profile

- Allowed: GPU atomics в binning (stable ordering не гарантируется);
- Allowed: lazy rebuild threshold;
- Allowed: approximate displacement tracking.

Trade-off: slight non-determinism для throughput; scientific reproducibility (observables) сохраняется.

---

## 10. Tests

### 10.1. Unit tests

- **Cell grid:**
  - Binning 1000 atoms в known positions → known cell assignments;
  - Edge cases: atom на границе (floor bias к младшей зоне); box corners.
- **Neighbor list:**
  - Pair `(i, j)` с known distance → presence в list ⟺ `d² ≤ (r_c + r_skin)²`;
  - Newton optimization: pair count ≈ half of naive;
  - Periodic BC: pairs across box boundary detected.
- **Displacement tracker:**
  - After init, `max_displacement = 0`;
  - After moving atom by δ, `max_displacement = δ`;
  - `skin_exceeded` triggers at threshold, not before.
- **Reorder:**
  - `apply(apply(reorder)) = identity` для corresponding reverse permutation;
  - Identity preservation: `atoms.id[new_idx]` stable after reorder.

### 10.2. Property tests

```
forall (random_atoms, random_box, r_c, skin):
    build_cell_grid(atoms, box, r_c, skin)
    build_neighbor_list(atoms, grid, zoning)

    # Correctness: neighbor list recovers all pairs within cutoff:
    for  i in atoms:
        for  j in atoms  where  j != i:
            d² = squared_distance(i, j)
            if  d² ≤ (r_c + skin)²:
                if newton and j > i:
                    assert pair (i,j) в neighbor_list
                elif not newton:
                    assert pair (i,j) в neighbor_list

    # Performance property: list size ≤ O(N · ρ · V_cutoff)
    expected_max = N_atoms · (atom_density · (4/3)·π·(r_c+skin)^3)
    assert  neighbor_list.neigh_ids.size() ≤ 2 · expected_max
```

### 10.3. Determinism tests

- **Same atoms twice**: cell_grid.cell_atoms identical byte-for-byte;
- **Same neighbor build twice**: neighbor_list.neigh_ids identical sequence;
- **Reorder applied twice (identity)**: `apply_reorder(apply_reorder(atoms, R), R_inv)` = original.

### 10.4. Integration tests

- **M1 canonical T1** (Al FCC small, Morse): build list, differential vs LAMMPS — should match within forces tolerance.
- **Rebuild correctness**: after running 100 steps, force trebuild, compare neighbor list vs fresh build — should be identical.
- **Migration correctness**: move atom across zone boundary, detect migration, reassign, neighbor list updated correctly.

### 10.5. Merge gates

PR затрагивающий neighbor:
- All unit + property tests green;
- Differential vs LAMMPS stable (forces match to FP64 tolerance);
- Determinism: bitwise identical builds;
- If performance work: no regression in T3/T4 benchmark time_spent_in_neighbor.

---

## 11. Telemetry

Metrics:
```
neighbor.cell_grid_build_time_ms
neighbor.neighbor_list_build_time_ms
neighbor.neighbor_list_size_total
neighbor.neighbor_list_size_per_atom_avg
neighbor.displacement_max_current
neighbor.rebuilds_total
neighbor.rebuilds_skin_triggered
neighbor.rebuilds_migration_triggered
neighbor.migrations_total
neighbor.migrations_per_step_avg
neighbor.reorder_operations_total
```

NVTX ranges:
- `neighbor::build_cell_grid`
- `neighbor::build_neighbor_list`
- `neighbor::detect_migrations`
- `neighbor::apply_reorder`

---

## 12. Roadmap alignment

| Milestone | Neighbor deliverable |
|---|---|
| M1 | CPU cell grid + CPU neighbor list; basic skin tracker |
| M2 | Newton optimization; periodic BC fully; property tests |
| **M3** | Full rebuild/migration/reorder separation; displacement cert integration with scheduler |
| M4 | Pass scheduler integration tests (cert invalidation on rebuild) |
| **M6** | GPU binning (stable radix sort); GPU neighbor build |
| M7 | Cross-subdomain migration (Pattern 2) |
| v2+ | Incremental rebuild; per-zone tracking; GPU-resident list permanently |

---

## 13. Open questions

1. **Per-zone displacement tracking** — worth it в v1? Глобальный tracker проще, но может быть wasteful для heterogeneous systems (одна зона с фазовым переходом, остальные стабильные).
2. **Cross-subdomain migration async vs sync** — в Pattern 2 v1 — sync (barrier); в post-v1 — async? Async даёт больше overlap но добавляет consistency issues.
3. **Stable radix sort на GPU** — стандартные thrust/CUB implementations stable, но precision trade-offs? Проверить corner cases.
4. **Rebuild fraction limits** — если >10% времени уходит в neighbor rebuild, это сигнал что skin слишком маленький. Auto-adjust skin? Или warn-only?

---

*Конец neighbor/SPEC.md v1.0, дата: 2026-04-16.*
