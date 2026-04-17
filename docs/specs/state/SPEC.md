# state/SPEC.md

**Module:** `state/`
**Status:** master module spec
**Parent:** `TDMD Engineering Spec v2.1` §8.2, §12.2
**Last updated:** 2026-04-16

---

## 1. Purpose и scope

### 1.1. Что делает модуль

`state/` — **единственный владелец** физического состояния системы. Это узкий модуль с широкой ответственностью: хранит атомы и геометрию box'а, обеспечивает доступ к ним через explicit API, отслеживает версионирование.

Делает четыре вещи:

1. **Хранит `AtomSoA`** — все per-atom данные (id, type, x, y, z, v, f, image flags);
2. **Хранит `Box`** — геометрия расчётного объёма и periodic conditions;
3. **Хранит species registry** — mapping species name → SpeciesId, масса, references;
4. **Версионирует state** — monotonic `Version` counter, incremented на каждое изменение.

### 1.2. Scope: что НЕ делает

- **не строит neighbor list** (это `neighbor/`);
- **не вычисляет силы** (это `potentials/`);
- **не двигает атомы** — только записывает новые значения через явный API, двигает `integrator/`;
- **не владеет zoning** (это `zoning/`), но знает атом→zone mapping cached;
- **не общается с сетью** (это `comm/`);
- **не знает про time — это scheduler/**;
- **не делает I/O** (это `io/`).

### 1.3. Почему state критичен

Master spec §8.2 fires an architectural invariant: **«никто кроме state не владеет атомами»**. Нарушение → бессистемная мутация → невозможно найти баг. В Claude Code playbook есть отдельная auto-reject condition на это.

Этот модуль — тот контракт, который все остальные соблюдают.

---

## 2. Public interface

### 2.1. AtomSoA (главный тип)

```cpp
namespace tdmd {

using AtomId    = uint64_t;
using SpeciesId = uint32_t;
using Version   = uint64_t;

struct AtomSoA {
    // Identity (stable across reorder):
    std::vector<AtomId>     id;            // per-atom unique ID
    std::vector<SpeciesId>  type;          // species index

    // Position (in metal units: Å):
    std::vector<double>     x, y, z;

    // Velocity (metal: Å/ps):
    std::vector<double>     vx, vy, vz;

    // Force (metal: eV/Å):
    std::vector<double>     fx, fy, fz;

    // Periodic image counters:
    std::vector<int32_t>    image_x, image_y, image_z;

    // Per-atom bitfield: bit 0 = owned_by_this_rank, bit 1 = boundary_zone_member, ...
    std::vector<uint32_t>   flags;

    size_t  size() const { return id.size(); }
};

} // namespace tdmd
```

**Guarantees:**
- Все vectors имеют одинаковый `.size()`;
- Index `i` в одном vector соответствует тому же atom'у во всех других.

### 2.2. Box

```cpp
struct Box {
    double  xlo, xhi;
    double  ylo, yhi;
    double  zlo, zhi;

    bool    periodic_x, periodic_y, periodic_z;

    // Only orthogonal в v1. Triclinic в post-v1.

    double  lx() const { return xhi - xlo; }
    double  ly() const { return yhi - ylo; }
    double  lz() const { return zhi - zlo; }
    double  volume() const { return lx() * ly() * lz(); }
};
```

### 2.3. Species registry

```cpp
struct SpeciesInfo {
    std::string     name;          // "Al", "Ni", "Cu", ...
    double          mass;          // g/mol (metal units)
    double          charge;        // electron charges (v1: always 0)
    uint32_t        atomic_number; // 13 for Al, 28 for Ni, ...
};

class SpeciesRegistry {
public:
    SpeciesId       register_species(const SpeciesInfo&);
    SpeciesInfo     get_info(SpeciesId) const;
    SpeciesId       id_by_name(const std::string&) const;
    size_t          count() const;
    uint64_t        checksum() const;  // for reproducibility bundle
};
```

Species registered в конфиге или LAMMPS data import. Регистрация — один раз, immutable после init.

### 2.4. StateManager (главный интерфейс)

```cpp
class StateManager {
public:
    // Read-only access:
    virtual const AtomSoA&         atoms() const = 0;
    virtual const Box&             box() const = 0;
    virtual const SpeciesRegistry& species() const = 0;
    virtual Version                version() const = 0;

    // Mutations — explicit, always bump version:
    virtual void  set_positions(size_t atom_idx,
                                 double x, double y, double z) = 0;
    virtual void  set_velocities(size_t atom_idx,
                                  double vx, double vy, double vz) = 0;
    virtual void  set_forces(size_t atom_idx,
                              double fx, double fy, double fz) = 0;
    virtual void  zero_forces() = 0;   // bulk operation
    virtual void  add_to_force(size_t atom_idx,
                                double dfx, double dfy, double dfz) = 0;

    // Bulk batch mutations (для performance — всё еще bump version один раз):
    virtual void  set_positions_batch(const std::vector<double>& x,
                                        const std::vector<double>& y,
                                        const std::vector<double>& z) = 0;

    // Periodic wrapping (returns new image counts):
    virtual void  wrap_to_primary_image(size_t atom_idx) = 0;
    virtual void  wrap_all_to_primary_image() = 0;

    // Adding / removing atoms (rare, e.g. migration):
    virtual AtomId  add_atom(SpeciesId, double x, double y, double z,
                              double vx = 0, double vy = 0, double vz = 0) = 0;
    virtual void   remove_atom(size_t atom_idx) = 0;

    // Atom ↔ zone mapping (cached; recomputed after migration/reorder):
    virtual ZoneId  zone_of_atom(size_t atom_idx) const = 0;
    virtual void    update_zone_mapping(const ZoningPlan&) = 0;

    // Reorder support:
    virtual void  apply_reorder(const ReorderMap&) = 0;

    virtual ~StateManager() = default;
};

class DefaultStateManager final : public StateManager { /*...*/ };
```

### 2.5. DeviceAtomSoA (GPU mirror)

```cpp
struct DeviceAtomSoA {
    AtomId*     id;
    SpeciesId*  type;
    double*     x;  double* y;  double* z;
    double*     vx; double* vy; double* vz;
    double*     fx; double* fy; double* fz;
    int32_t*    image_x; int32_t* image_y; int32_t* image_z;
    uint32_t*   flags;
    size_t      n;

    Version     version_on_device;     // может отставать от host version
};
```

**Relationship:** `DeviceAtomSoA` — mirror `AtomSoA` на GPU. Sync explicit через `StateManager::sync_to_device()` / `sync_from_device()`. Никогда автоматический.

В `Fp64` builds `x, y, z` — double (8 bytes). В `Fp32Experimental` — float (4 bytes), но это compile-time build flavor choice.

---

## 3. Version management

### 3.1. Rules

```
Version = uint64_t, monotonic, never decreasing
Version == 0 is "uninitialized"
First valid version after init = 1
Every state mutation → version++ on that mutation
```

Мутации, которые вызывают `version++`:
- `set_positions`, `set_velocities`, `set_forces`, `add_to_force`;
- `zero_forces`;
- `wrap_to_primary_image`, `wrap_all_to_primary_image`;
- `add_atom`, `remove_atom`;
- `apply_reorder`;
- `update_zone_mapping`.

**Batch mutations bump version once** при завершении всего batch.

### 3.2. Version as dependency token

Другие модули сохраняют `state.version()` как proof, что их cached data consistent. Примеры:

- `NeighborList.associated_state_version` — версия state при last rebuild;
- `SafetyCertificate.version` — версия state при построении certificate;
- `DeviceAtomSoA.version_on_device` — последняя синхронизированная версия.

Invalidation = `cached_version != current state.version`.

### 3.3. Monotonicity invariant

```
forall moments t1 < t2:
    state.version(t2) >= state.version(t1)
```

Strict equality допустима только если state не менялся между `t1` и `t2`.

---

## 4. Periodic boundary conditions

### 4.1. Image flags

Для каждого атома `(image_x, image_y, image_z)` — integers, отслеживающие сколько раз атом пересёк границу:
- `image = 0` — атом в primary image box;
- `image = 1` — атом один раз "перешёл" в positive direction (actual position = stored + `image * box_length`);
- etc.

Это позволяет восстановить **unwrapped** trajectory для диффузии / MSD.

### 4.2. Wrap algorithm

```
function wrap_to_primary_image(atom_idx):
    if periodic_x:
        while atoms.x[atom_idx] >= box.xhi:
            atoms.x[atom_idx] -= box.lx
            atoms.image_x[atom_idx] += 1
        while atoms.x[atom_idx] < box.xlo:
            atoms.x[atom_idx] += box.lx
            atoms.image_x[atom_idx] -= 1
    # same для y, z
```

### 4.3. Когда wrap'ить

**Integrator** wraps после `post_force` half-kick + drift. Это ensures что positions в `AtomSoA` всегда в primary image после integration step.

**Другие модули читают только primary-image positions**. Они **не должны** wrap'ить сами (кроме `neighbor/` для computing periodic distances — но это on-the-fly, без mutation).

### 4.4. Minimum image convention (для distance computation)

Для pair distance:
```
function minimum_image_distance(x_i, x_j, box_length, periodic):
    dx = x_i - x_j
    if periodic:
        if dx > box_length / 2:
            dx -= box_length
        elif dx < -box_length / 2:
            dx += box_length
    return dx
```

Это **query-time operation**, не mutation. Модули `neighbor/` и `potentials/` используют это без изменения `AtomSoA`.

---

## 5. Species и types

### 5.1. Species registration

Species регистрируются **один раз** на startup, из config или LAMMPS data. После registration — immutable.

```yaml
species:
  - name: Al
    mass: 26.9815
    atomic_number: 13
  - name: Ni
    mass: 58.6934
    atomic_number: 28
```

### 5.2. Type indices

`SpeciesId` — dense `uint32_t` index от 0 до `N_species - 1`. Mapping name↔id — через `SpeciesRegistry::id_by_name()`.

**Invariant:** registered species не могут быть удалены или переименованы mid-run. Если нужно — restart.

### 5.3. Checksum для reproducibility

`SpeciesRegistry::checksum()` — hash конкатенации всех `SpeciesInfo` в canonical order (by SpeciesId ascending). Записывается в reproducibility bundle; другой run с тем же checksum гарантированно использует те же species.

---

## 6. Atom ↔ zone mapping

### 6.1. Cached mapping

`StateManager` поддерживает `std::vector<ZoneId> atom_to_zone_` — cached. Обновляется через `update_zone_mapping(zoning_plan)`.

**Invariant after update:** для каждого atom `i`, `atom_to_zone_[i] == zoning_plan.compute_zone_for_position(atoms.x[i], ...)`.

### 6.2. Invalidation

Cache становится stale при:
- mutation позиций атомов (`set_positions`);
- applied reorder (indices изменились);
- migration (explicit through `neighbor/detect_migrations` → `apply_migrations`).

`state.version()` bump — сигнал что cache может быть stale. Caller (scheduler / neighbor) explicitly calls `update_zone_mapping` когда нужна актуальность.

### 6.3. Thread safety

Cached mapping — read-only after `update_zone_mapping`. Multiple threads могут читать одновременно. Update — exclusive lock.

---

## 7. GPU sync

### 7.1. Explicit sync

```cpp
class StateManager {
    virtual void  sync_to_device() = 0;
    virtual void  sync_from_device() = 0;
    virtual const DeviceAtomSoA&  device_atoms() const = 0;
    virtual Version  device_version() const = 0;
};
```

Never implicit. Caller знает когда нужен sync.

### 7.2. Sync semantics

`sync_to_device()`:
- copies all `AtomSoA` fields to device;
- updates `device_version_` = current host version;
- asynchronous using default stream or explicit stream parameter.

`sync_from_device()`:
- copies back from device;
- bumps host version.

### 7.3. Mirrored correctness

After sync, host и device consistent:
- `atoms.x[i]` identical на GPU и CPU;
- Identical bytes, not "similar values".

If sync is async, caller synchronizes stream before reading.

### 7.4. Selective sync (optimization)

v2+: `sync_field_to_device(x)` — sync только один field. Useful когда integrator обновил только velocity, нет нужды sync-ить positions.

В v1: full AtomSoA sync. Simpler, correct.

---

## 8. Atom identity (stable IDs)

### 8.1. ID assignment

Atom получает `AtomId` при первом creation (`add_atom` или import). ID — monotonic uint64, никогда не переиспользуется после `remove_atom`.

### 8.2. Identity guarantees

**AtomId стабилен через:**
- reorder (index меняется, ID — нет);
- migration (между zones — index, ID);
- rebuild (неindex не меняется);
- restart (ID serialized в restart file и restored).

**AtomId меняется только через:**
- `add_atom` (new ID);
- `remove_atom` (ID disappears).

### 8.3. Index vs ID distinction

**Index** (`size_t` в vectors) — position в `AtomSoA`, volatile.
**ID** (`AtomId = uint64_t`) — persistent identifier.

API в других модулях:
- scheduler работает с indices (быстро);
- trajectory dumps пишут IDs (stable для analysis);
- migration records используют IDs (для correctness across reorders).

Caller должен ясно понимать, что он оперирует. Разные API convention — разные параметры. `atom_idx` vs `atom_id` — не взаимозаменяемы.

---

## 9. Add / remove atoms

### 9.1. Use cases

- Import from data file (bulk add на startup);
- Migration cross-subdomain (add на dest, remove на source) — Pattern 2;
- Specialized simulations (depositation, etching) — post-v1.

### 9.2. Semantics

`add_atom(species, x, y, z, vx, vy, vz)`:
- вычисляет new `AtomId`;
- appends to all vectors;
- sets forces to 0, images to 0;
- returns the new `AtomId`;
- version++.

`remove_atom(atom_idx)`:
- moves last atom to removed position (swap-and-pop);
- resizes vectors down by 1;
- invalidates ALL cached mappings (zone, neighbor);
- version++.

**Warning:** `remove_atom` changes indices of at most one atom (the last one). Other modules must check.

### 9.3. Bulk operations

`add_atoms_batch(vector<SpeciesId>, vector<positions>)` — optimized для bulk import. Version++ один раз.

`remove_atoms_batch(vector<atom_idx>)` — harder (need stable indexing during removal). Algorithm: mark-and-compact.

---

## 10. Determinism policy

### 10.1. Reference profile

- Все mutations deterministic (same inputs → same state);
- `add_atom` возвращает monotonic IDs (no random bits);
- `wrap_to_primary_image` — deterministic (finite loops with exact tolerances);
- GPU sync preserves bit identity — copies are exact.

### 10.2. Production

Identical to Reference для correctness. Performance через batch operations and selective sync.

### 10.3. Fast

Может разрешать async GPU sync без explicit wait (overlap). Остальное — как Reference.

---

## 11. Tests

### 11.1. Unit tests

- **AtomSoA invariants:** все fields одинакового size после любого mutation;
- **Version monotonicity:** 1000 случайных mutations → version increases by exactly 1 per mutation;
- **Periodic wrap:**
  - atom в центре box: 0 wraps;
  - atom на `xhi + 0.01`: wraps to `xlo + 0.01`, image_x = 1;
  - negative wraps: symmetric;
- **Image reconstruction:** `unwrapped = x + image_x * box.lx` consistent;
- **Add/remove round-trip:** add → remove → state unchanged (except IDs).

### 11.2. Property tests

```
forall (random_atoms, random_mutations):
    initial_version = state.version()
    apply random_mutations to state
    final_version = state.version()
    assert final_version == initial_version + len(mutations)

    # Invariants:
    assert state.atoms().size() == expected_count
    assert state.atoms().id.size() == state.atoms().x.size()
    # ... для всех fields

    # IDs unique (если не было remove):
    if no removes in mutations:
        assert len(set(atoms.id)) == len(atoms.id)
```

### 11.3. Determinism tests

- **Same mutations twice:** two states with same mutations applied should be byte-identical;
- **Reorder applied twice:** `apply_reorder(R); apply_reorder(R_inverse)` → original state.

### 11.4. Integration tests

- **Round-trip with integrator:** initialize → integrator step → state updated correctly (v, x, images);
- **GPU sync correctness:** CPU state → sync → GPU state → modify GPU → sync back → consistent.

---

## 12. Telemetry

Metrics:
```
state.atom_count
state.version_current
state.mutations_per_iteration_avg
state.gpu_sync_operations_total
state.gpu_sync_bytes_transferred_total
state.species_count
state.box_volume
state.density_atoms_per_a3
```

NVTX ranges:
- `state::sync_to_device`;
- `state::sync_from_device`;
- `state::apply_reorder`.

---

## 13. Roadmap alignment

| Milestone | State deliverable |
|---|---|
| M1 | `AtomSoA`, `Box`, `SpeciesRegistry`, basic `StateManager` on CPU |
| M2 | Checksums, reproducibility bundle integration |
| M3 | Atom↔zone mapping cache |
| M4+ | Version tokens consumed by scheduler |
| **M6** | **`DeviceAtomSoA` + sync** — critical для GPU path |
| M7 | Cross-subdomain add/remove (migration support) |
| v2+ | Selective sync, triclinic box, charged atoms (long-range) |

---

## 14. Open questions

1. **Energy / virial per-atom storage** — хранить в `AtomSoA` или compute on-the-fly? v1: compute on-the-fly (перепакованы в `ForceResult`). v2+ может хранить для analysis outputs.
2. **Per-atom user data** — некоторые users хотят custom fields (charge distribution, orientation). Через `flags` bitfield? Через extension API? Post-v1.
3. **Zero-copy compact removal** — для bulk removal swap-and-pop даёт O(k) но invalidates. Alternative: mark-deleted + periodic compact. Trade-off complexity vs speed.
4. **Triclinic box** — сейчас ортогональный only. Triclinic требует full 9-element transform matrix + тщательно periodic minimum-image. Post-v1.
5. **Atom identity через restart** — current design сохраняет `AtomId` в restart file. Для large-scale parallel restart этот подход нужно проверить на scaling.

---

*Конец state/SPEC.md v1.0, дата: 2026-04-16.*
