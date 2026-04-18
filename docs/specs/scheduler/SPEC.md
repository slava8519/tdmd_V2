# scheduler/SPEC.md

**Module:** `scheduler/`
**Status:** master module spec
**Parent:** `TDMD Engineering Spec v2.1` §6, §8, §12.4, §12.7a
**Last updated:** 2026-04-16

---

## 1. Purpose и scope

### 1.1. Что делает модуль

`scheduler/` — сердце TDMD. Это **единственный** модуль в системе, который принимает решения о продвижении зон по времени. Весь остальной код — «исполнители» (потенциалы считают силы, integrator двигает атомы, comm передаёт данные), а scheduler решает **что, когда и в каком порядке запустить**.

Формально, scheduler отвечает за пять вещей:

1. **Legality** — ни одна зона не может начать compute без валидного `SafetyCertificate` и удовлетворённых зависимостей.
2. **Bounded progress** — если существуют ready tasks, scheduler обязан их выдать; отсутствие прогресса — hard error, не «normal state».
3. **Commit protocol** — двухфазный commit зоны: сначала local completion, потом global legality (halo received + telemetry snapshot).
4. **Deterministic ordering** (в Reference profile) — одинаковое состояние → одинаковая последовательность задач.
5. **Deadlock detection** — watchdog-инвариант, жёсткая диагностика вместо silent stall.

### 1.2. Scope: что НЕ делает scheduler

Жёсткие границы — нарушение = architectural violation:

- **не вычисляет силы** (это `potentials/`);
- **не двигает атомы** (это `integrator/`);
- **не владеет атомами и box** (это `state/`);
- **не владеет neighbor lists** (это `neighbor/`);
- **не отправляет пакеты в сеть** (это `comm/`);
- **не выбирает unit system или precision** (compile-time `BuildFlavor`);
- **не строит zoning** (это `zoning/`, scheduler лишь потребляет `ZoneningPlan`);
- **не предсказывает performance** (это `perfmodel/`).

Scheduler **читает** state, neighbor, zoning через явные interfaces, **записывает** task queues и cert store, **публикует** events в telemetry. Никакого прямого владения кроме собственных queues и DAG.

### 1.3. Deployment pattern awareness

Scheduler работает в двух конфигурациях (см. мастер-спец §4a, §10.1):

- **Pattern 1 (v1 M4–M6):** `InnerTdScheduler` — весь движок. Нет `OuterSdCoordinator`, boundary dependencies пустые.
- **Pattern 2 (v1 M7+):** `InnerTdScheduler` работает внутри subdomain'а; существует ровно один `OuterSdCoordinator` на run, координирующий subdomain'ы.

Scheduler-код **один и тот же** в обоих паттернах; различие — только в наличии `OuterSdCoordinator*` (nullable dependency) и обработке `SubdomainBoundaryDependency`.

---

## 2. Public interface

### 2.1. Базовые типы (из мастер-специи §12.4)

```cpp
namespace tdmd {

enum class ZoneState {
    Empty,          // память выделена, данных нет
    ResidentPrev,   // содержит данные предыдущего шага
    Ready,          // cert выдан, можно запускать
    Computing,      // идёт force + integrate
    Completed,      // compute завершён, не committed
    PackedForSend,  // данные упакованы в TemporalPacket
    InFlight,       // MPI/NCCL transfer в процессе
    Committed       // подтверждена receiver'ом, можно освобождать
};

struct SafetyCertificate {
    bool        safe;
    uint64_t    cert_id;
    ZoneId      zone_id;
    TimeLevel   time_level;
    Version     version;
    double      v_max_zone, a_max_zone;
    double      dt_candidate;
    double      displacement_bound;
    double      buffer_width;
    double      skin_remaining;
    double      frontier_margin;
    TimeLevel   neighbor_valid_until_step;
    TimeLevel   halo_valid_until_step;
    uint64_t    mode_policy_tag;
};

struct ZoneTask {
    ZoneId      zone_id;
    TimeLevel   time_level;
    Version     local_state_version;
    uint64_t    dep_mask;
    uint64_t    certificate_version;
    uint32_t    priority;
    uint32_t    mode_policy_tag;
};

} // namespace tdmd
```

### 2.2. Главный абстрактный интерфейс

```cpp
namespace tdmd {

class TdScheduler {
public:
    // Lifecycle:
    virtual void  initialize(const ZoningPlan&) = 0;
    virtual void  attach_outer_coordinator(OuterSdCoordinator*) = 0;   // nullable
    virtual void  shutdown() = 0;

    // Certificate management:
    virtual void  refresh_certificates() = 0;
    virtual void  invalidate_certificates_for(ZoneId) = 0;
    virtual void  invalidate_all_certificates(const std::string& reason) = 0;

    // Task selection (one iteration):
    virtual std::vector<ZoneTask>  select_ready_tasks() = 0;

    // Task lifecycle callbacks (called by SimulationEngine):
    virtual void  mark_computing(const ZoneTask&) = 0;
    virtual void  mark_completed(const ZoneTask&) = 0;
    virtual void  mark_packed(const ZoneTask&) = 0;
    virtual void  mark_inflight(const ZoneTask&) = 0;
    virtual void  mark_committed(const ZoneTask&) = 0;

    // Commit protocol (two-phase):
    virtual void  commit_completed() = 0;

    // Introspection:
    virtual bool       finished() const = 0;
    virtual size_t     min_zones_per_rank() const = 0;
    virtual size_t     optimal_rank_count(size_t total_zones) const = 0;
    virtual size_t     current_pipeline_depth() const = 0;
    virtual TimeLevel  local_frontier_min() const = 0;
    virtual TimeLevel  local_frontier_max() const = 0;

    // Events from comm / neighbor:
    virtual void  on_zone_data_arrived(ZoneId, TimeLevel, Version) = 0;
    virtual void  on_halo_arrived(uint32_t peer_subdomain, TimeLevel) = 0;
    virtual void  on_neighbor_rebuild_completed(const std::vector<ZoneId>&) = 0;

    // Watchdog:
    virtual void  check_deadlock(std::chrono::milliseconds T_watchdog) = 0;

    virtual ~TdScheduler() = default;
};

} // namespace tdmd
```

### 2.3. Канонические реализации

В v1 поставляется одна реализация:

```cpp
class CausalWavefrontScheduler final : public TdScheduler {
    // zone DAG, dependency tracking,
    // deterministic / reproducible / fast policy switching
};
```

Именно она покрывает все три ExecProfile через `mode_policy_tag` и PolicyBundle. Не создавать fork'и реализации под каждый профиль — архитектурный запрет (§8.3 мастер-специи).

---

## 3. State machine зоны

### 3.1. Legal transitions

```
Empty
  │
  │ on_zone_data_arrived (receiver side)
  ▼
ResidentPrev  ◄─────────────────────────┐
  │                                      │ commit_completed (no peer needs)
  │ refresh_certificates + deps_ok       │
  ▼                                      │
Ready ─────────────────────┐             │
  │                         │ cert_invalidated
  │ mark_computing          ▼             │
  ▼                      ResidentPrev    │
Computing                                │
  │                                      │
  │ mark_completed                       │
  ▼                                      │
Completed ──────────────────────────────┤
  │
  │ mark_packed (has downstream peer)
  ▼
PackedForSend
  │
  │ mark_inflight
  ▼
InFlight
  │
  │ mark_committed (ACK received / eager protocol)
  ▼
Committed ───► (зона освобождается или возвращается в Empty для следующего цикла)
```

### 3.2. Illegal transitions (I1–I7 из мастер-специи §13.4)

Эти инварианты проверяются property-fuzzing тестами на каждом PR:

- **I1:** `Committed → Ready` без `time_level++` — запрещено.
- **I2:** `Empty → Computing` — запрещено (нет данных).
- **I3:** зона одновременно в `ready_queue` и `inflight_queue` — запрещено.
- **I4:** две активные задачи с одинаковым `(zone_id, time_level, version)` — запрещено.
- **I5:** `Completed ≠ Committed`, commit — отдельная фаза, не побочный эффект completion.
- **I6:** `frontier_min + K_max ≥ max_time_level` любой `Ready` задачи — инвариант каждого `select_ready_tasks()`.
- **I7:** certificate monotonicity — `safe(C[dt1]) ∧ dt2 < dt1 ⟹ safe(C[dt2])`.

### 3.3. Transition guards

Каждый переход состояния происходит только через явный метод scheduler'а. Прямая мутация `ZoneMeta::state` из других модулей — architectural violation и должна диагностироваться clang-tidy custom check'ом.

---

## 4. Safety certificate: формальная модель

### 4.1. Математика (из §6.4 мастер-специи)

**Displacement bound:**
```
δ(dt) = v_max · dt + 0.5 · a_max · dt²
```

**Safety predicate:**
```
safe(C) ⟺ δ(dt_candidate) < min(buffer_width,
                                   skin_remaining,
                                   frontier_margin)
```

### 4.2. Где берутся значения

| Поле | Источник |
|---|---|
| `v_max_zone`, `a_max_zone` | reduction по всем атомам зоны (из `state/`) |
| `skin_remaining` | `neighbor/` (накопленное displacement с последнего rebuild) |
| `buffer_width` | `zoning/` (фиксировано планом), по умолчанию `r_c + r_skin - r_c = r_skin` |
| `frontier_margin` | из сcheduler'а: `K_max · dt - (time_level - frontier_min) · dt` |
| `neighbor_valid_until_step` | `neighbor/` predict, на основе current skin и v_max |
| `halo_valid_until_step` | `comm/` (Pattern 2) или ∞ (Pattern 1) |

### 4.3. Triggers инвалидации

Certificate **обязан** быть invalidated при:

1. Изменении `Version` state зоны (любое изменение positions/velocities атомов зоны);
2. Rebuild neighbor list в любой зоне в пределах `r_c + r_skin` от текущей;
3. Migration атомов (вход или выход атомов зоны);
4. Изменении `dt`;
5. Изменении `ExecProfile` mid-run (редкий, но возможен для emergency stop);
6. Сдвиге `frontier_min` больше чем на 1 шаг (pattern 2: движение соседнего subdomain);
7. Изменении halo snapshot archive в Pattern 2.

### 4.4. Certificate lifecycle

```
build → validate → use (launch decision) → invalidate → rebuild → archive (telemetry)
```

Каждый certificate имеет уникальный `cert_id` (монотонный счётчик на run) и хранится в `CertificateStore` до его инвалидации. После инвалидации — копия в telemetry archive (для debugging и post-mortem).

### 4.5. Strictness по режимам

| ExecProfile | Certificate math | Adaptation allowed |
|---|---|---|
| Reference | максимально консервативный; одинаковый алгоритм на всех платформах; no runtime-adaptive heuristics | — |
| Production | предсказуемая адаптация `buffer_width`; mode-stable heuristics | да, но детерминистично |
| FastExperimental | agressive estimates; cost-aware widening/narrowing; всегда с safety fallback | да, включая non-determinism |

---

## 5. Task selection algorithm

### 5.1. Псевдокод (deterministic policy, из §6.7 мастер-специи)

```
function select_ready_tasks():
    candidates = []

    for  zone z  in  all_zones ordered by canonical_zone_order(z):
        min_level = scheduler.min_advanced_level(z)
        for  t  in  [min_level, min_level + K_max]:
            if  task_already_scheduled(z, t):
                continue

            cert = cert_store.get(z, t)
            if  cert == None  or  not cert.safe:
                continue

            if  not all_spatial_peers_completed(z, t - 1):
                continue

            if  t  >  global_frontier_min + K_max:
                continue   # frontier guard

            if  cert.neighbor_valid_until_step  <  t:
                continue

            # Pattern 2 only:
            if  is_boundary_zone(z):
                if  outer_coordinator == null  or
                   not outer_coordinator.can_advance_boundary_zone(z, t):
                    continue

            candidates.append(ZoneTask(z, t, cert.version, ...))
            break   # не выдаём более одной задачи на зону за iteration

    # Deterministic tie-break (Reference profile):
    sort candidates by (time_level_asc, canonical_zone_order_asc, version_asc)

    return  candidates[: max_tasks_per_iteration]
```

### 5.2. Tie-break по режимам

| ExecProfile | Tie-break ordering |
|---|---|
| Reference | `(time_level, canonical_zone_order, version)` строго детерминистичный |
| Production | Reference + optional cost-aware priority (должен сохранять acceptance thresholds) |
| FastExperimental | adaptive priority: `f(time_level, device_pressure, queue_length, halo_readiness)`; допустим task stealing |

### 5.3. `max_tasks_per_iteration`

- **Reference:** равно числу compute streams (для CPU — число worker threads; для GPU — `num_streams`, обычно 1–3).
- **Production:** до 2× streams (overlap allowance).
- **FastExperimental:** unbounded, ограничено только throughput.

---

## 6. Commit protocol (two-phase)

Коммит зоны всегда происходит в два шага. Это инвариант I5.

### 6.1. Phase A — Local completion

Выполняется в `mark_completed(task)`:

1. Force kernel завершён (проверка через CUDA event или thread fence);
2. Integrator kernel завершён;
3. Local buffers обновлены: новые `x, v, f` записаны в `state/` через явный API;
4. State version incremented;
5. `ZoneMeta::state ← Completed`.

После Phase A зона формально «посчитана», но **ещё не пригодна для передачи / использования другими зонами**.

### 6.2. Phase B — Global legality

Выполняется в `commit_completed()`:

1. Если у зоны есть downstream peer (в Pattern 1 — соседний rank в mesh; в Pattern 2 — может быть соседний subdomain):
   - pack в `TemporalPacket` или `HaloPacket`;
   - state machine → `PackedForSend`;
   - comm.send_async;
   - state machine → `InFlight`;
   - on ACK (или eager protocol): state machine → `Committed`.
2. Если peer'а нет (internal zone): state machine → `Committed` напрямую.
3. Post-step certificate записывается в telemetry archive.
4. Zone version bumped.

### 6.3. Почему two-phase

Причина в **fail-safety**. Если после Phase A мы обнаружили, что certificate был некорректен (например, vmax zone оказался больше ожидаемого после компута), мы можем **откатить** зону в `ResidentPrev` и пересчитать — данные ещё не ушли в сеть. После Phase B откат стоил бы retractions пакетов у peers.

---

## 7. Fail / retry policy

### 7.1. Виды failures

| Failure | Пример | Response |
|---|---|---|
| Certificate invalidated mid-compute | migration произошёл во время compute | abort task, state → ResidentPrev, re-refresh cert |
| Peer data arrived stale | zone_data_arrived с устаревшей version | discard packet, log telemetry event, ждём актуальный |
| Halo timeout | peer subdomain не прислал halo за T_halo_timeout | retry N раз, затем escalate в deadlock watchdog |
| CUDA error в kernel | floating-point error, memory fault | hard abort всего run'а, diagnostics dump |
| MPI error | network failure | retry через backend retry policy, затем hard abort |

### 7.2. Retry policy (Reference)

В Reference profile **число и порядок retry должны быть каноническими** — одинаковые для одинакового стартового состояния. Это значит:

- `retry_count` — детерминистический параметр (фиксированный ceiling);
- запрещены rand-based backoffs;
- сразу после retry `retry_count++` → продолжение обычного flow.

### 7.3. Retry policy (Production / Fast)

Разрешены exponential backoff, jittered retry, но **scientific reproducibility сохраняется**: observables через большой интервал времени идентичны в пределах `2σ`.

---

## 8. Deadlock detection

### 8.1. Watchdog инвариант

```
if  not finished()  and  ready_queue.empty()  and
   inflight_queue.empty_progress_for > T_watchdog:
    → hard_diagnostic_failure("scheduler deadlock")
```

### 8.2. Что считается «прогрессом»

«Progress» = хотя бы одно из:
- задача из `ready_queue` была выдана и перешла в `Computing`;
- задача в `inflight_queue` перешла в `Committed`;
- zone events processed (arrived / rebuild / halo);
- frontier_min увеличился.

Отсутствие всех четырёх признаков в течение `T_watchdog` (по умолчанию 30 s) — хард-диагностика.

### 8.3. Diagnostic dump

При срабатывании watchdog'а scheduler обязан вывести:

- snapshot state всех зон (по count per state);
- `frontier_min`, `frontier_max`;
- длины всех queues;
- список зон в `Computing` дольше 5 s;
- список boundary stalls (Pattern 2);
- последние 100 events из event log;
- советы: «возможно deadlock между subdomain'ами A и B; проверьте `K_max` и halo latency».

---

## 9. Queue semantics

### 9.1. Минимальный набор очередей

- `ready_queue` — задачи с валидным cert и удовлетворёнными deps;
- `blocked_queue` — задачи, которым не хватает cert/halo/neighbor;
- `inflight_queue` — задачи в `Computing | PackedForSend | InFlight`;
- `completed_queue` — задачи в `Completed`, ожидающие commit;
- `retry_queue` — задачи после failure, ожидающие повторной попытки.

### 9.2. Detereministic ordering в Reference

- `ready_queue`: priority queue с compare function = `(time_level_asc, canonical_zone_order_asc, version_asc)`;
- `blocked_queue`: FIFO по времени блокировки;
- `inflight_queue`: FIFO;
- остальные — FIFO.

### 9.3. В Production/Fast

Ready queue может использовать adaptive priority, но с `canonical_zone_order` как fallback tie-break. Task stealing (Fast only) берёт из ready queue другого rank/device.

---

## 10. Events API

Scheduler подписывается на события от других модулей. Все события должны приводить либо к:

- обновлению DAG зависимостей,
- смене состояния зоны,
- выдаче новых ready tasks,
- записи в telemetry.

Если событие не привело ни к чему из этого — это baseline bug (должно диагностироваться тестами).

### 10.1. Event types

| Event | Источник | Эффект |
|---|---|---|
| `ZoneDataArrived` | `comm/` | zone → `ResidentPrev`, cert rebuild queued |
| `HaloArrived` | `comm/` (Pattern 2) | outer boundary deps released |
| `CertificateRefreshed` | internal | возможная разблокировка задач |
| `NeighborRebuildCompleted` | `neighbor/` | invalidate certs в affected zones, rebuild |
| `ForceComputeCompleted` | runtime (via CUDA event) | `mark_completed(task)` |
| `IntegrateCompleted` | runtime | часть `mark_completed` |
| `TemporalPacketCommitted` | `comm/` | `mark_committed(task)` |
| `LongRangeWindowReleased` | `potentials/` (future) | разблокировка зон в LR window |

---

## 11. Policy plumbing

### 11.1. SchedulerPolicy struct

```cpp
struct SchedulerPolicy {
    // Frontier control:
    uint32_t  k_max_pipeline_depth;           // default: 4 (Reference), 16 (Production), 64 (Fast)
    uint32_t  max_tasks_per_iteration;        // default: compute_streams_count

    // Priority:
    bool      use_canonical_tie_break;        // Reference/Production: true; Fast: false
    bool      allow_task_stealing;            // Fast only

    // Certificate:
    bool      allow_adaptive_buffer;          // Production/Fast: true
    bool      deterministic_reduction_cert;   // Reference: true

    // Watchdog:
    std::chrono::milliseconds  t_watchdog;    // default: 30 s

    // Retry:
    uint32_t  max_retries_per_task;           // default: 3
    bool      exponential_backoff;            // Reference: false

    // Commit:
    bool      two_phase_commit;               // always true; here for documentation
};

class PolicyFactory {
public:
    static SchedulerPolicy  for_reference();
    static SchedulerPolicy  for_production();
    static SchedulerPolicy  for_fast_experimental();
};
```

### 11.2. Policy validation

`PolicyValidator::check(SchedulerPolicy, BuildFlavor, ExecProfile)` проверяет compatibility. Примеры:

- `allow_task_stealing = true` с `Reference` profile → **reject**;
- `k_max = 128` с `Fp64Reference` build → **warning** (память);
- `deterministic_reduction_cert = false` с `Reference` → **reject**.

Validator runs один раз на startup, сохраняет результат в reproducibility bundle.

---

## 11a. Load balancing policy

**Критический вопрос для production TDMD.** LAMMPS и GROMACS оба имеют dynamic load balancing (DLB). Без него performance деградирует на 20-50% на неоднородных системах (поверхности, дефекты, фазовые переходы). Для TDMD это **ещё острее**, потому что frontier stall в TD pipeline сильнее зависит от slowest zone.

Этот раздел фиксирует формальную политику load balancing по milestones.

### 11a.1. Источники imbalance в TDMD

Причины неоднородной нагрузки:

1. **Density variation** — free surface, vacancies, dislocations → зоны с разным числом атомов;
2. **Potential heterogeneity** — multi-species system с разными cost_per_atom (Al vs W + SNAP);
3. **ML potential inference cost** — variable по zone composition;
4. **Neighbor list size variation** — pairs per atom variance под cluster formation;
5. **Hardware variation** — GPU A100 + V100 mix, different clock speeds, thermal throttling.

**В TD-specific:** даже 10% imbalance between zones приводит к frontier stalls. Slowest zone становится bottleneck для всего pipeline на своём time_level. Imbalance в SD просто замедляет отдельные ranks; в TD — blocks whole wavefront.

### 11a.2. Три уровня load balancing maturity

TDMD implements DLB в три фазы по milestones:

#### Phase 1 (M5-M6): Measurement-only DLB

Scheduler collects per-zone timing statistics, но **не меняет distribution**. Просто observes and reports.

**Metrics emitted:**
```
scheduler.zone_compute_time_p50_ms
scheduler.zone_compute_time_p95_ms
scheduler.zone_compute_time_p99_ms
scheduler.zone_compute_time_max_ms
scheduler.rank_utilization_avg          # [0, 1]
scheduler.rank_utilization_min
scheduler.imbalance_ratio               # max/mean time per zone
```

**Thresholds:**
- `imbalance_ratio > 1.3` — warning в telemetry log;
- `imbalance_ratio > 2.0` — explicit recommendation в final report: "consider re-zoning для better balance";
- `imbalance_ratio > 3.0` — performance sabotage, likely zoning misconfiguration; error-level log.

**Artefact gate M5:** T3 anchor test должен иметь `imbalance_ratio < 1.2` (uniform Al FCC). Otherwise zoning bug.

#### Phase 2 (M7-M8): Advisory DLB

После measurement phase, runtime emits **recommendations**:

```
tdmd diagnostic --load-balance my_run/
```

Output:
```
Load balance analysis для run_id abc123:

Rank utilization summary:
  Mean: 82%
  Min: 54% (rank 3)
  Max: 98% (rank 7)
  Std: 14%

Imbalance detected: ratio 1.81 (threshold: 1.3)

Recommendations:
  1. Re-zone с Hilbert3D (currently Decomp2D) — estimated imbalance 1.15
  2. Increase zone count from 216 to 512 — finer granularity, imbalance 1.25
  3. Use Pattern 2 two-level (currently Pattern 1) с 2 subdomains — imbalance 1.30

Expected speedup от recommendation 1: 22% (measured-model estimate)
```

Runtime **не** применяет recommendations automatically; user должен ре-запустить с new config. Это preserves bitwise determinism — не изменяется layout mid-run.

**Integration with VerifyLab:** диагностика производится через `verify/harness/load_balance_analyzer/`, reuses measurement infrastructure.

#### Phase 3 (v2+): Active DLB

Runtime **актiveно переназначает zones между ranks** когда imbalance превышает threshold. Это архитектурное расширение — требует preservation of atom identity и consistency of safety certificates через re-assignment.

**Scope v2+:**
- Zone-level migration между ranks (not atom-level);
- Trigger-based (imbalance_ratio > 1.5 для >1000 consecutive steps);
- Smooth transition: новое assignment takes effect после global barrier;
- Only в `FastExperimental` profile — breaks layout-invariant determinism by design;
- Reference/Production profiles: remain static zoning even в v2+.

**Not in scope v2:** atom-level migration cross-rank (это GROMACS approach, complicated); hardware-aware DLB (GPU + CPU mix); ML-predicted imbalance correction.

### 11a.3. Why not earlier than Phase 3

Active DLB breaks two important guarantees:

1. **Layout-invariant determinism** (§7.3 master spec Level 2) — reassignment изменяет reduction order;
2. **Bitwise reproducibility** (§7.3 Level 1) — adaptive behavior по runtime measurements inherently non-deterministic.

В TDMD ценности **scientific rigor первичен**. Static zoning в Reference/Production обеспечивает reproducibility. Active DLB — performance feature, и она заслуженно отнесена в FastExperimental profile где user явно принимает non-determinism trade-off.

Это отличается от LAMMPS/GROMACS, где DLB on by default. TDMD позиция — **scientific defaults, performance opt-in**.

### 11a.4. User workflow

Typical workflow для new simulation:

1. **Initial run:** user запускает `tdmd run case.yaml` с static zoning;
2. **Preflight warning:** runtime предупреждает если expected imbalance high (на основе zoning math, §12.4 master spec);
3. **Measurement phase (Phase 1):** first 1000 steps — collect telemetry;
4. **Mid-run report:** `imbalance_ratio = 1.81, recommendation:...`;
5. **User decision:**
   - Accept current (1.81× slower than optimal) → continue;
   - Stop, re-run с recommended config → restart;
   - v2+ users: enable active DLB via FastExperimental profile.

### 11a.5. Load balancing across subdomains (Pattern 2)

В Pattern 2 два уровня balance:

- **Intra-subdomain** (inner TD): zones должны быть балансированы внутри subdomain'а;
- **Inter-subdomain** (outer SD): subdomains должны иметь равную work.

Inner balance следует §11a.1-11a.4 (per-subdomain zoning).
Inter balance — **дополнительная метрика**:

```
scheduler.subdomain_work_balance_ratio  # max / mean total work per subdomain
```

Recommendations для inter-subdomain imbalance:
- Re-partition subdomain grid (static change, require restart);
- Increase outer decomposition granularity (more subdomains, smaller each);
- v2+: weighted subdomain assignment based на density estimates.

### 11a.6. Policy configuration

```yaml
runtime:
  load_balance:
    # Phase 1 (M5+): always on, measurement only:
    measure_imbalance: true
    imbalance_warning_threshold: 1.3
    imbalance_error_threshold: 3.0

    # Phase 2 (M7+): advisory mode:
    emit_recommendations: true
    recommendation_interval_steps: 10000

    # Phase 3 (v2+): active DLB, opt-in:
    active_dlb: false                       # default off
    active_dlb_trigger_threshold: 1.5       # active_dlb=true requires FastExperimental
    active_dlb_min_stable_steps: 1000
```

**Validation:** `active_dlb=true` с `ExecProfile != FastExperimental` → `PolicyValidator::check` rejects.

### 11a.7. Tests

- **Phase 1 measurement test:** inject synthetic imbalance (2× cost для 1/3 zones), verify metrics report it correctly;
- **Phase 2 recommendation test:** predefined imbalance pattern → expected recommendation;
- **Phase 3 active DLB test (v2+):** imbalanced workload → active DLB восстанавливает balance within 5× measurement_interval;
- **Regression test:** static zoning runs (Reference profile) в Phase 1+ give bit-identical results с и без measurement enabled.

---

## 12. Tests

### 12.1. Unit tests

- **cert math:** δ(dt), safe predicate, все граничные случаи (v=0, a=0, dt=0, буфер=0);
- **state machine transitions:** каждый из ~20 legal transitions проверяется отдельным тестом;
- **queue ordering:** вставить случайные задачи, извлечь, проверить canonical ordering;
- **priority tie-break:** задачи с одинаковым time_level, разным zone_order → проверить Reference ordering.

### 12.2. Property tests (fuzzer)

```
forall seed in fuzz_seeds:
    events = generate_random_events(seed, N=10000)
    scheduler = new CausalWavefrontScheduler(Reference)
    for event in events:
        apply(scheduler, event)
        assert invariants_I1_to_I7(scheduler)
```

**Минимум 10⁶ сгенерированных последовательностей в CI per PR.**

### 12.3. Determinism tests

- **Same run twice bitwise:** два run'а с одинаковым seed должны дать **идентичный** event log и идентичное финальное state.
- **Queue ordering determinism:** 10⁴ случайных вставок в ready_queue в различном temporal order должны давать **идентичную** последовательность извлечения.

### 12.4. Deadlock detection tests

- **Intentional deadlock:** сконструировать сценарий, где две зоны ждут друг друга (cycle в DAG); watchdog должен сработать за `T_watchdog ± 10%`.
- **Long stall without deadlock:** сценарий, где прогресс медленный но существует; watchdog **не должен** срабатывать.

### 12.5. Integration tests

- **Single-rank TD:** canonical T1 benchmark (Al FCC small, Morse), TD mode, check NVE drift.
- **Multi-rank TD Pattern 1:** T3 Al FCC 10⁶, 8 ranks, проверка efficiency ≥ 80%.
- **Pattern 2 integration (M7+):** 2 subdomain × 4 ranks, boundary stalls, halo exchange corrected.

### 12.6. Mandatory merge gates

PR, затрагивающий scheduler, не может быть merged без:

- all unit tests green;
- property fuzzer 10⁶ seeds green;
- T1 differential vs LAMMPS green;
- T3 efficiency ≥ 80% (если меняется policy);
- определение, затрагивает ли PR определенность: если да — determinism tests обязательны.

---

## 13. Telemetry hooks

Scheduler emit'ит следующие метрики в `telemetry/`:

```
scheduler.zones_ready_count
scheduler.zones_inflight_count
scheduler.zones_committed_total
scheduler.certificate_failures_total
scheduler.neighbor_rebuilds_total
scheduler.deadlock_warnings_total
scheduler.boundary_stalls_total     (Pattern 2)
scheduler.task_selection_time_ms    (per iteration)
scheduler.commit_latency_ms         (Phase A → Phase B)
scheduler.current_frontier_min
scheduler.current_frontier_max
scheduler.pipeline_depth            (frontier_max - frontier_min)
```

NVTX ranges для Nsight:
- `TdScheduler::select_ready_tasks`
- `TdScheduler::refresh_certificates`
- `TdScheduler::commit_completed`

---

## 14. Roadmap alignment

| Milestone | Scheduler deliverable |
|---|---|
| M3 | zoning plan consumed, nothing else |
| **M4** | `CausalWavefrontScheduler (Reference)`, K=1, single-rank; all state machine + cert + policy infrastructure |
| M5 | multi-rank через Ring backend; K-batching (K ∈ {1,2,4,8}) |
| M6 | GPU event integration (CUDA streams, events) |
| **M7** | `attach_outer_coordinator`, `SubdomainBoundaryDependency`, Pattern 2 full support |
| M8+ | Adaptive K policy, auto-tuning; FastExperimental profile |

---

## 15. Open questions (module-local)

1. **Cost-aware priority in Production** — какая именно функция даёт scientific reproducibility? Возможно оставить Reference ordering как default и использовать cost-aware только когда telemetry показывает idle > threshold.
2. **Task stealing scope** — только между ranks на одной ноде (через shared memory) или через всю сеть? Первое — безопаснее и разумнее на M8.
3. **Certificate caching** — хранить ли cert `C[dt=x]` если был посчитан для `C[dt=y]` и `x < y` (по monotonicity)? Экономия compute vs память.
4. **Boundary stall fallback** — если Pattern 2 halo не пришёл за `T_halo_max`, откатывать ли зону на предыдущий time_level? Это выход из scope scheduler'а в сторону soft-recovery.

---

*Конец scheduler/SPEC.md v1.0, дата: 2026-04-16.*

---

## 16. Change log

- **2026-04-19** — **M4 landed (Reference, K=1 single-rank)**. T4.1–T4.11
  implemented the full Reference realization of the scheduler: zone state
  machine + I1–I5 fuzzer (T4.4), SafetyCertificate + CertificateStore + I7
  monotonicity fuzzer (T4.3), `CausalWavefrontScheduler` with DAG +
  refresh + select + I6 frontier fuzzer (T4.5–T4.6), two-phase commit +
  retry semantics (T4.7), deadlock watchdog + DiagnosticReport (T4.8),
  `SimulationEngine` TD-mode wiring behind opt-in `scheduler.td_mode`
  YAML flag (T4.9), bitwise determinism tests — same-seed atom SoA + event
  log byte-exact, 10⁴×100 queue-ordering trials (T4.10), M4 integration
  smoke + CI gate (T4.11). D-M4-9 byte-exact acceptance: `td_mode: true`
  on the M3 Ni-Al EAM smoke produces thermo bit-identical to the legacy
  path; the M4 golden IS the M3 golden, copied verbatim.
  - Pattern 1 only (D-M4-2); Pattern 2 and multi-rank remain M5+.
  - `AlwaysSafeCertificateInputSource` stub ships certificates with
    generous safe bounds so every zone is selectable every step; a
    live-state adapter lands with Pattern 2 in M7+.
  - `EventLog::kCapacity = 1024` resolves OQ-M4-4; DiagnosticReport still
    surfaces the last 100 events per §8.3.
