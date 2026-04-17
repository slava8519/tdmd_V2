# zoning/SPEC.md

**Module:** `zoning/`
**Status:** master module spec
**Parent:** `TDMD Engineering Spec v2.1` §6.1, §9, §12.3
**Last updated:** 2026-04-16

---

## 1. Purpose и scope

### 1.1. Что делает модуль

`zoning/` решает одну задачу: **как нарезать box модели на расчётные зоны**, и **в каком порядке их обходить**, чтобы:

- удовлетворить constraint `zone_size ≥ r_c + r_skin` по каждой оси;
- минимизировать `N_min` — минимальное число зон, необходимое одному rank'у для устойчивого TD pipeline;
- максимизировать `n_opt = N_zones / N_min` — оптимальное число ranks, на которых метод даёт линейный scaling;
- обеспечить детерминистичный canonical order для Reference profile.

Это **математический ядровой** модуль, имеющий прямое соответствие с формулами (35), (43), (44)-(45) диссертации Андреева. Качество zoning напрямую определяет потолок масштабируемости TDMD.

### 1.2. Scope: что НЕ делает zoning

- **не хранит атомы** (это `state/`);
- **не управляет memory layout атомов в зонах** (это `state/` + `neighbor/` cell bins);
- **не выбирает размер box** (это `io/` + user input);
- **не решает когда продвигать зону** (это `scheduler/`);
- **не знает про ranks и subdomain'ы в Pattern 2** (outer — `runtime/`, inner — `scheduler/`);
- **не предсказывает perf** (это `perfmodel/`, zoning лишь даёт ему input через `N_min`, `n_opt`).

Zoning — **чистая функция**: `plan(box, cutoff, skin, n_ranks, hint) → ZoningPlan`. Stateless, pure, testable.

### 1.3. Отношения с другими модулями

```
                  ┌─────────┐
                  │   io/   │  (user config, potential cutoff)
                  └────┬────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  zoning/ZoningPlanner.plan() │
        └──┬────────────────────────┬──┘
           │                        │
           ▼                        ▼
   ┌───────────────┐        ┌──────────────┐
   │ scheduler/    │        │  perfmodel/  │
   │ consumes      │        │  consumes    │
   │ - N_min       │        │  - n_opt     │
   │ - n_opt       │        │  - n_zones   │
   │ - canonical   │        │  - scheme    │
   │   ordering    │        └──────────────┘
   └───────────────┘
           │
           ▼
   ┌───────────────┐
   │ state/        │
   │ consumes      │
   │ - zone bboxes │
   │ - atom→zone   │
   │   mapping     │
   └───────────────┘
```

---

## 2. Public interface

### 2.1. Базовые типы (из мастер-специи §12.3)

```cpp
namespace tdmd {

enum class ZoningScheme {
    Linear1D,       // разрезание по одной оси (Andreev §2.2)
    Decomp2D,       // разрезание по двум осям, zigzag (Andreev §2.4)
    Hilbert3D,      // TDMD extension: 3D с Hilbert ordering
    Manual          // explicit user-supplied plan (advanced)
};

struct ZoningPlan {
    ZoningScheme                   scheme;
    std::array<uint32_t, 3>        n_zones;
    std::array<double, 3>          zone_size;
    uint64_t                       n_min_per_rank;
    uint64_t                       optimal_rank_count;
    std::vector<ZoneId>            canonical_order;
    std::array<double, 3>          buffer_width;   // обычно = r_skin
    double                         cutoff;
    double                         skin;
    // For Pattern 2 awareness:
    std::optional<Box>             subdomain_box;  // nullopt в Pattern 1
};

struct PerformanceHint {
    double    cost_per_force_evaluation_seconds;  // estimate from perfmodel
    double    bandwidth_peer_to_peer_bytes_per_sec;
    double    atom_record_size_bytes;
    uint32_t  preferred_K_pipeline;               // hint, не contract
};

} // namespace tdmd
```

### 2.2. Главный интерфейс

```cpp
namespace tdmd {

class ZoningPlanner {
public:
    virtual ZoningPlan plan(
        const Box&             box,
        double                 cutoff,
        double                 skin,
        uint64_t               n_ranks,
        const PerformanceHint& hint) const = 0;

    // Альтернатива: пользователь заранее знает схему:
    virtual ZoningPlan plan_with_scheme(
        const Box&             box,
        double                 cutoff,
        double                 skin,
        ZoningScheme           forced_scheme,
        const PerformanceHint& hint) const = 0;

    virtual ~ZoningPlanner() = default;
};

class DefaultZoningPlanner final : public ZoningPlanner {
    // основная реализация; выбирает схему автоматически
};

} // namespace tdmd
```

### 2.3. Вспомогательные запросы

```cpp
class ZoningPlanner {
public:
    // ...
    // Predict-only, без построения полного плана:
    virtual uint64_t estimate_n_min(
        ZoningScheme scheme,
        const Box& box,
        double cutoff, double skin) const = 0;

    virtual uint64_t estimate_optimal_ranks(
        ZoningScheme scheme,
        const Box& box,
        double cutoff, double skin) const = 0;

    // Validation: проверить, подходит ли user-supplied manual plan:
    virtual bool validate_manual_plan(
        const ZoningPlan&,
        std::string& reason_if_invalid) const = 0;
};
```

---

## 3. Математика N_min и n_opt

Это ядро модуля. Ниже — формулы для каждой схемы, с прямой привязкой к диссертации Андреева.

### 3.1. Scheme A — Linear1D (из §2.2 диссертации)

**Описание:** разрезание по одной оси (обычно Z); обход от `zone[1]` до `zone[N_z]` линейный.

**Формула (Andreev eq. 35):**
```
N_min(Linear1D) = 2
```

Два — это минимум: одна зона «внутри», одна «передаётся / принимается». Ровно два достаточны для того, чтобы каждый rank в ring-топологии мог одновременно компутить одну зону и обменивать другую.

**Оптимальное число ranks (Andreev eq. 44):**
```
n_opt(Linear1D) = floor(N_zones_along_axis / N_min) = floor(N_z / 2)
```

**Применимость:** оптимальна для тонких геометрий (плёнки, nanowires вдоль одной оси), где natural extent сильно анизотропный. Для кубических boxes — wasteful (используется только одна ось из трёх).

**Canonical ordering:** `zone_id[i] = i` для `i ∈ [0, N_z)`. Простейший возможный.

### 3.2. Scheme B — Decomp2D (из §2.4 диссертации)

**Описание:** разрезание по двум осям (Y и Z, обычно). Обход — zigzag (змейкой): rows Y fill left-to-right, при смене Z-слоя — right-to-left.

**Формула (Andreev eq. 43, для квадратных 2D zones):**
```
N_min(Decomp2D) = 2 · N_zy + 2 = 2·(N_zy + 1)
```

Для radius-of-influence `r_c` сфера захватывает до 4 соседних зон (Andreev рис. 24), и для ring-обхода нужно держать в памяти один ряд вперёд и один назад плюс текущую позицию.

**Упрощение для quadratic (когда `N_zy ≈ N_zz`):** `N_min ≈ 2·√N_zones`.

**Оптимальное число ranks:**
```
n_opt(Decomp2D) = floor(N_zones / N_min) ≈ N_zones / (2·√N_zones) = √N_zones / 2
```

Это уже **значительно лучше** Linear1D при одинаковом количестве зон, если box близок к квадрату.

**Применимость:** плоские geometries, 2D материалы, thin films с нетривиальным extent в двух направлениях.

**Canonical ordering:** zigzag — строго детерминистичный. Реализация в §5.

### 3.3. Scheme C — Hilbert3D (TDMD extension)

**Описание:** 3D space-filling curve (Hilbert), нумерующая все зоны так, что последовательно пронумерованные зоны **близки в пространстве**.

**Мотивация:** линейная нумерация 3D (lexicographic `i + N_x·j + N_x·N_y·k`) даёт катастрофический `N_min`. Из диссертации §2.5, для модели 16×16×16 с линейной нумерацией `N_min = 274`. Это делает 3D практически непригодным без оптимизации обхода.

**Формула Hilbert 3D (эмпирическая, TDMD-specific):**

Для Hilbert-обхода 3D-куба `N × N × N` с radius-of-influence `R` (в единицах `r_c + r_skin`, обычно `R = 1`):
```
N_min(Hilbert3D, N) ≈ C · N^2
```
где `C ≈ 4..6` в зависимости от конкретной Hilbert mapping и boundary handling.

Для неквадратного box `N_x × N_y × N_z`, pronost:
```
N_min(Hilbert3D, Nx, Ny, Nz) ≈ 4 · max(Nx·Ny, Ny·Nz, Nx·Nz)
```

**Оптимальное число ranks:**
```
n_opt(Hilbert3D) = floor(Nx·Ny·Nz / (4·max(Nx·Ny, Ny·Nz, Nx·Nz)))
              ≈ min(Nx, Ny, Nz) / 4
```

Для `16×16×16`: `n_opt ≈ 16/4 = 4`. Не впечатляет, но Hilbert применим для гораздо большего scale: `64×64×64` → `n_opt ≈ 16`; `256×256×256` → `n_opt ≈ 64`.

**Сравнение (16×16×16 = 4096 zones):**

| Scheme | N_min | n_opt | Notes |
|---|---|---|---|
| Linear1D | 2 | 8 (по Z) | требует `16 · 16 = 256` зон в слое на rank — дорого по памяти |
| Decomp2D | 34 | 120 | хорошо |
| Hilbert3D | ~64 | 64 | best для куба |
| Linear3D (lex) | 274 | 14 | **никогда не использовать!** |

### 3.4. Selection algorithm

```
function select_scheme(box, r_c, r_skin, n_ranks, hint):
    w = r_c + r_skin
    N_x = floor((box.x_hi - box.x_lo) / w)
    N_y = floor((box.y_hi - box.y_lo) / w)
    N_z = floor((box.z_hi - box.z_lo) / w)

    aspect = [N_x, N_y, N_z]
    max_ax = max(aspect)
    min_ax = min(aspect)

    # Critical check: достаточно ли зон вообще?
    N_total = N_x · N_y · N_z
    if  N_total < 3:
        throw "Box too small for TD; use SD-vacuum mode"

    # Decision tree:
    if  max_ax / min_ax > 10  and  min_ax < 4:
        chosen = Linear1D (along max_axis)
    elif  max_ax / min_ax > 3  or  (N_x · N_y) < 16:
        chosen = Decomp2D  (along 2 largest axes)
    else:
        chosen = Hilbert3D

    # Estimate N_min and n_opt for chosen scheme
    N_min = estimate_n_min(chosen, aspect)
    n_opt = floor(N_total / N_min)

    # If n_ranks > n_opt significantly, warn and consider smaller box per rank
    if  n_ranks > 1.2 · n_opt:
        emit_warning("requested ranks exceed optimal; consider Pattern 2")

    return  (chosen, N_min, n_opt)
```

### 3.5. Property tests для формул N_min

**T1:** `N_min(Linear1D) = 2` ∀ N_z ≥ 2.

**T2:** `N_min(Decomp2D, Ny, Nz) = 2·(Ny+1)` ∀ Ny, Nz ≥ 2.

**T3:** `N_min(Hilbert3D, N, N, N) ∈ [3·N², 6·N²]` ∀ N ≥ 4 (эмпирический envelope).

**T4:** Monotonicity: большая модель с той же схемой → `n_opt` не убывает.

**T5:** Corner cases:
- N_x = N_y = N_z = 2 → N_min(Hilbert3D) = N_total = 8; n_opt = 1 (нет смысла TD);
- Linear1D с N_z = 2 → n_opt = 1; нет pipeline depth.

---

## 4. Canonical ordering

### 4.1. Зачем нужен canonical order

В Reference profile (§7 мастер-специи) Scheduler'у требуется **детерминистический** порядок обхода зон. Zoning планер его вычисляет в `plan()` и возвращает в `ZoningPlan::canonical_order`. Scheduler использует его как primary sort key.

Любые два rank'а с идентичным `ZoningPlan` **должны** видеть идентичный `canonical_order`. Это инвариант.

### 4.2. Canonical order per scheme

**Linear1D:** `order[i] = i`, простая последовательность 0..N_z-1.

**Decomp2D (zigzag):**
```
function decomp2d_canonical_order(N_y, N_z):
    order = []
    for  z in 0..N_z - 1:
        if  z % 2 == 0:
            for  y in 0..N_y - 1:
                order.append(z · N_y + y)
        else:
            for  y in N_y-1..0 step -1:
                order.append(z · N_y + y)
    return order
```

Пример для `N_y=3, N_z=2`: `[0, 1, 2, 5, 4, 3]`.

**Hilbert3D:** стандартная Gilbert-Hilbert curve 3D. Реализация — non-trivial, но хорошо изучена; см. [Skilling 2004] или library `libmorton` для reference implementation.

Псевдокод:
```
function hilbert3d_canonical_order(N_x, N_y, N_z):
    # Pad to power-of-2 если нужно
    N_pow = next_power_of_2(max(N_x, N_y, N_z))

    order_padded = []
    for  idx in 0..N_pow³ - 1:
        (x, y, z) = hilbert_d2_xyz(idx, log2(N_pow))
        if  x < N_x and y < N_y and z < N_z:
            order_padded.append(x + N_x·y + N_x·N_y·z)

    return order_padded
```

В TDMD должна быть одна каноническая реализация `hilbert_d2_xyz` (портированная из reference источника с unit tests), без random variations.

### 4.3. Guarantees

- **Permutation property:** `canonical_order` содержит каждый `zone_id` ∈ `[0, N_total)` ровно один раз.
- **Determinism property:** same inputs → same order, bitwise.
- **Locality property (Hilbert only):** среднее расстояние между соседними в `canonical_order` зонами пропорционально `N^(2/3)` (вместо `N` для lex).

### 4.4. Property tests для ordering

```
forall (box, cutoff, skin, n_ranks) in test_corpus:
    plan = planner.plan(box, cutoff, skin, n_ranks, hint)

    # Permutation:
    ids = sorted(plan.canonical_order)
    assert ids == [0, 1, ..., len(plan.canonical_order) - 1]

    # Determinism:
    plan2 = planner.plan(box, cutoff, skin, n_ranks, hint)
    assert plan.canonical_order == plan2.canonical_order

    # Locality (Hilbert only):
    if plan.scheme == Hilbert3D:
        avg_neighbor_distance = measure(plan.canonical_order)
        expected = 1.5 · cube_root(N_total)
        assert  0.5 · expected < avg_neighbor_distance < 2 · expected
```

---

## 5. Buffer width policy

### 5.1. Минимум

Жёсткий минимум: `buffer_width >= r_skin`.

Это условие необходимо и достаточно в consistent neighbor-list контексте: если атом не покинул `r_c + r_skin` между rebuild'ами, он физически не может пересечь границу зоны в пределах одного шага (при условии стандартной TD ширины зоны = `r_c + r_skin`).

### 5.2. Adaptive в Production

В Production profile, `buffer_width` может адаптироваться на основе observed `v_max`:

```
buffer_width = max(r_skin, α · v_max_global · dt · K_pipeline)
```

где `α ∈ [1.5, 3.0]` — safety coefficient.

**Нельзя** уменьшать `buffer_width` ниже `r_skin` ни при каких обстоятельствах. Это hard invariant.

### 5.3. Per-zone vs global buffer

- **v1:** global buffer (одинаковый на все зоны). Упрощает код и анализ.
- **v2+:** per-zone buffer возможен — зоны с медленно-движущимися атомами могут иметь меньший buffer, с быстрыми — больший. Но тогда migration между зонами становится сложнее. Отложено.

---

## 6. Zone ↔ atom mapping

### 6.1. Assignment algorithm

```
function assign_atoms_to_zones(atoms: AtomSoA, plan: ZoningPlan):
    atom_to_zone = []
    for  atom_idx in 0..N_atoms - 1:
        x, y, z = atoms.x[idx], atoms.y[idx], atoms.z[idx]
        zx = min(floor((x - box.x_lo) / plan.zone_size.x), plan.n_zones.x - 1)
        zy = min(floor((y - box.y_lo) / plan.zone_size.y), plan.n_zones.y - 1)
        zz = min(floor((z - box.z_lo) / plan.zone_size.z), plan.n_zones.z - 1)
        zone_id = zx + plan.n_zones.x · zy + plan.n_zones.x · plan.n_zones.y · zz
        atom_to_zone.append(zone_id)
    return  atom_to_zone
```

### 6.2. Edge cases

- **Атом на границе:** `floor` биас'ит в сторону младшей зоны. Атом ровно на границе между зонами `z[i]` и `z[i+1]` приписывается к `z[i]`. Это даёт детерминированное поведение.
- **Атом вне box:** это ошибка; вызов preflight должен её поймать. Scheduler не имеет права видеть атомы вне box.
- **Периодические BC:** атом за пределами периода wrap'ится до assignment (это делает `state/` через `wrap_to_primary_image`).

### 6.3. Reorder для locality

После assignment зон, атомы **должны быть переупорядочены** в `AtomSoA`, так что atoms одной зоны идут последовательно в памяти. Это делает `neighbor/` через stable sort; zoning лишь предоставляет `zone_id[]`.

---

## 7. Subdomain awareness (Pattern 2)

### 7.1. Local zoning в subdomain'е

В Pattern 2 (M7+) каждый subdomain имеет свой локальный `ZoningPlan`. Zoning planner вызывается **дважды**:

1. `Outer plan (SubdomainGrid)` — `runtime/` разрезает box на subdomain'ы;
2. `Inner plan (ZoningPlan per subdomain)` — `zoning/` нарезает каждый subdomain на zones.

Интерфейс:

```cpp
ZoningPlan plan_for_subdomain(
    const Box& subdomain_box,
    double cutoff, double skin,
    uint64_t n_ranks_in_subdomain,
    const PerformanceHint& hint) const;
```

### 7.2. Boundary zones

Zones, лежащие в пределах `r_c + r_skin` от любой границы subdomain'а, маркируются `is_boundary = true`. Эти зоны:

- требуют `SubdomainBoundaryDependency` в scheduler'е (§6.3 мастер-специи);
- участвуют в halo snapshot archive (§4a.8 мастер-специи);
- не должны пересекать границу subdomain'а через migration (этот edge case обрабатывается отдельно runtime'ом).

### 7.3. Interface extension

```cpp
struct ZoningPlan {
    // ...
    std::vector<bool>    is_boundary_zone;  // size == N_total; true = boundary
    std::vector<int>     boundary_peer_subdomain;  // -1 if not boundary; else subdomain_id
};
```

---

## 8. Tests

### 8.1. Unit tests

- Formula correctness: `N_min` для каждой схемы на корпусе тестовых boxes;
- Canonical order permutation property;
- Hilbert curve correctness: reference table из 64 известных `(idx, x, y, z)` пар;
- Zigzag ordering на ручном примере 3×3;
- Edge cases: min box size, high aspect, skin = 0.

### 8.2. Property tests

```
# Shrinker-based fuzzer на входные параметры:
forall (box_extent, cutoff, skin, n_ranks) ∈ fuzz:
    plan = planner.plan(...)

    # Core invariants:
    assert plan.n_min_per_rank >= 1
    assert plan.optimal_rank_count >= 1
    assert plan.canonical_order.size() == product(plan.n_zones)
    assert is_permutation(plan.canonical_order)

    # Consistency:
    assert all(zone_size >= cutoff + skin for zone_size in plan.zone_size)

    # Scheme-specific:
    if plan.scheme == Linear1D:
        assert plan.n_min_per_rank == 2
    if plan.scheme == Hilbert3D:
        assert Hilbert locality metric within envelope
```

**Минимум 10⁵ fuzz cases в CI per PR.**

### 8.3. Anchor test

Из диссертации §2.2–2.5 известны concrete numbers для фиксированных конфигураций. Проверить в TDMD: совпадают ли наши `n_opt` с диссертационными:

| Конфигурация | Dissertation | TDMD expected |
|---|---|---|
| 1D, 16 zones | n_opt = 8 (§2.2) | 8 |
| 2D, 16×5 = 80 zones | n_opt = 13 (eq. 45) | 13 |
| 3D linear, 16³ zones | n_opt = 14 (§2.5) | N/A (мы не используем linear 3D) |
| 3D Hilbert, 16³ | no dissertation ref | 64 (наш target) |

### 8.4. Regression tests

При любом изменении в `plan()` — regression check, что canonical_order не изменился для фиксированного set of inputs. Если изменился — **bump major version** модуля и требование обновить все downstream frozen baselines (scheduler tests, benchmarks).

---

## 9. Roadmap alignment

| Milestone | Zoning deliverable |
|---|---|
| M1 | пустой модуль (скелет) |
| M2 | Linear1D реализация, unit tests |
| **M3** | Decomp2D + Hilbert3D, selection algorithm, full property tests, N_min формулы validated |
| M4+ | consumed by scheduler; no new functionality |
| **M7** | subdomain awareness extension (`plan_for_subdomain`, boundary zones) |
| M9+ | user-supplied manual plans; adaptive re-zoning on dynamic geometry |

---

## 10. Open questions (module-local)

1. **Adaptive re-zoning** — если атомы перераспределяются в процессе run'а (например, в системе с поверхностью, атомы двигаются к границе), нужно ли динамически менять zone sizes? Это **большое расширение**; v1 — static zoning на startup.
2. **Non-orthogonal zones** — для triclinic box (post-v1) нужны параллелепипедные зоны. Математика N_min становится сложнее.
3. **Weighted zoning для load balancing** — зоны с неоднородной плотностью атомов имеют разную стоимость. В v1 zoning однородный; load balancing делает scheduler через task migration.
4. **Hilbert variants** — есть несколько вариантов Hilbert 3D curve (Butz, Skilling, Hamilton). Выбрать и документировать один. Рекомендация: Skilling 2004 (лучшая locality).
5. **Manual plan API** — как именно пользователь может задать custom zoning? YAML extension или programmatic-only?

---

*Конец zoning/SPEC.md v1.0, дата: 2026-04-16.*
