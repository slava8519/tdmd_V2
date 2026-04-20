# runtime/SPEC.md

**Module:** `runtime/`
**Status:** master module spec
**Parent:** `TDMD Engineering Spec v2.1` §6.6, §7, §8.1, §12.8
**Last updated:** 2026-04-19 (T7.9 §2.4 Pattern 2 engine wiring)

---

## 1. Purpose и scope

### 1.1. Что делает модуль

`runtime/` — оркестратор. Единственная точка, где все модули `state/`, `neighbor/`, `potentials/`, `integrator/`, `scheduler/`, `comm/`, `telemetry/`, `perfmodel/` связываются в работающую симуляцию.

Делает четыре вещи:

1. **Lifecycle** — initialize → bootstrap → run → finalize → shutdown;
2. **Policy resolution** — BuildFlavor × ExecProfile → RuntimePolicyBundle;
3. **Module wiring** — создание модулей с правильными dependencies;
4. **Restart / resume** — save/load complete runtime state.

### 1.2. Scope: что НЕ делает

- **не владеет атомами** (это `state/`);
- **не владеет временем** (это `scheduler/`);
- **не вычисляет ничего** (только dispatch);
- **не парсит config** (это `io/`);
- **не делает I/O напрямую** (delegates to `io/`);
- **не владеет CLI** (это `cli/`).

Runtime — **glue layer**. Минимум логики, максимум dispatch.

### 1.3. Единственность оркестрации

Master spec §8.4 fires invariant: **`SimulationEngine` — единственная точка оркестрации**. Никакой другой модуль не вызывает main loop; никто не инициирует compute кроме runtime'а. Нарушение = architectural violation.

Этот contract позволяет test harnesses легко заменять `SimulationEngine` на stub, оставляя все модули functional.

---

## 2. Public interface

### 2.1. Engine state machine

```cpp
namespace tdmd {

enum class EngineState {
    Created,                // constructed, nothing initialized
    Configured,             // RuntimeConfig parsed, ready for policy resolution
    PoliciesResolved,       // RuntimePolicyBundle built, modules not yet created
    StateBootstrapped,      // StateManager initialized, atoms loaded
    ExecutionInitialized,   // all modules created and wired
    Running,                // run loop active
    Paused,                 // run loop paused (e.g. for restart)
    Finalized,              // final outputs written, modules still alive
    Shutdown,               // all modules destroyed
    Failed                  // unrecoverable error
};

} // namespace tdmd
```

**Legal transitions:**
```
Created → Configured → PoliciesResolved → StateBootstrapped
       → ExecutionInitialized → Running ⇄ Paused
       → Finalized → Shutdown

Failed reachable from any state.
```

Transitions enforced by guards in API. Пытка вызвать `run()` из `Configured` → error.

### 2.2. SimulationEngine

```cpp
class SimulationEngine {
public:
    // Lifecycle — in order:
    void  configure(const RuntimeConfig&);
    void  resolve_policies();
    void  bootstrap_state(const SimulationInput&);
    void  initialize_execution();
    void  run(uint64_t n_steps);
    void  pause();
    void  resume();
    void  finalize();
    void  shutdown();

    // Introspection:
    EngineState              state() const;
    BuildFlavorInfo          build_info() const;
    const RuntimePolicyBundle&  policies() const;
    PerformancePrediction    predicted_perf() const;
    uint64_t                 current_step() const;
    double                   elapsed_wall_time_seconds() const;

    // Module access (read-only для tests and cli):
    const StateManager&      state_manager() const;
    const TdScheduler&       scheduler() const;
    const TelemetrySink&     telemetry() const;

    // Restart:
    void  save_restart(const std::string& path);
    void  load_restart(const std::string& path);

    // Graceful interruption:
    void  request_stop();      // safely stops run loop at next iteration boundary

    ~SimulationEngine();

private:
    EngineState              state_;
    RuntimeConfig            config_;
    BuildFlavorInfo          build_info_;
    RuntimePolicyBundle      policies_;
    ReproContext             repro_;

    std::unique_ptr<StateManager>        state_;
    std::unique_ptr<ZoningPlanner>       zoning_planner_;
    ZoningPlan                           zoning_plan_;
    std::unique_ptr<NeighborManager>     neighbor_;
    std::unique_ptr<PotentialModel>      potential_;
    std::unique_ptr<Integrator>          integrator_;
    std::unique_ptr<TdScheduler>         scheduler_;
    std::unique_ptr<OuterSdCoordinator>  outer_coord_;   // nullptr in Pattern 1/3
    std::unique_ptr<CommBackend>         comm_;
    std::unique_ptr<PerfModel>           perfmodel_;
    std::unique_ptr<TelemetrySink>       telemetry_;
    std::unique_ptr<UnitConverter>       unit_converter_;

    std::atomic<bool>        stop_requested_{false};
    uint64_t                 current_step_{0};
    std::chrono::steady_clock::time_point start_time_;
};
```

---

### 2.3. GPU backend wiring (M6, T6.7)

`runtime.backend` — единственный YAML knob, переключающий engine между CPU и GPU путями:

```yaml
runtime:
  backend: cpu | gpu        # default cpu — M1..M5 legacy path сохраняется bit-exact
```

Контракт (D-M6-3, D-M6-7):

- **`backend: cpu`** (default): `SimulationEngine` использует CPU `PotentialModel` + CPU `Integrator`; никаких CUDA зависимостей в runtime path. M1..M5 smokes работают без изменений.
- **`backend: gpu`**: `SimulationEngine::init()` создаёт RAII `runtime::GpuContext` (DevicePool + одна compute stream, D-M6-12), `potentials::EamAlloyGpuAdapter` поверх parsed `EamAlloyPotential::data()`, и `GpuVelocityVerletIntegrator`. Force recompute + pre/post-force integrator branch на `gpu_backend_` флаг; MPI transport остаётся `MpiHostStagingBackend` (host-staged, D-M6-3 — NCCL / GPUDirect отложены на v1.5 M7+).

**Invariant (D-M6-7 engine-level gate):** `backend: gpu` Reference path ≡ `backend: cpu` Reference path byte-for-byte на thermo stream для EAM/alloy (T6.7 1-rank gate, 864-atom Ni-Al, 100 шагов). Multi-rank определённость (D-M5-12) расширяется на GPU эру: GPU K=1 P=1 ≡ GPU K=1 P=2 через тот же Kahan-ring в `comm/` (T6.7 2-rank gate, Ni-Al 10 шагов).

**Guards (preflight + runtime):**
- `io/preflight` отвергает `backend: gpu` + не-EAM potential (M6 scope);
- `GpuContext` ctor бросает `std::runtime_error`, если CUDA device не виден (CPU-only build или отсутствие sm_XX hardware) — engine transparently failfast'ит с чистым сообщением "rebuild с `-DTDMD_BUILD_CUDA=ON` или set `runtime.backend=cpu`".

**Ownership:** `GpuContext` — RAII unique owner пула + stream на engine уровне. Все GPU сущности (`gpu_potential_`, `gpu_integrator_`) получают `DevicePool&` / `DeviceStream&` по ссылке и НЕ владеют ими — lifetime tied к `SimulationEngine`.

Scope для v1.0 (M6):
- Single compute stream (D-M6-13 двух-stream overlap — T6.9);
- Pattern 1/3 только (Pattern 2 GPU planning — M7);
- `BuildFlavor = Fp64ReferenceBuild` гарантирует byte-exact gate; `MixedFast*` flavors активируются в T6.8 с differential поверх того же harness.

---

### 2.4. Pattern 2 engine wiring (M7, T7.9)

`zoning.subdomains` + `comm.backend` — единственные YAML knobs, переключающие engine между Pattern 1 (legacy single-subdomain) и Pattern 2 (TD внутри subdomain × SD между subdomain'ами):

```yaml
zoning:
  subdomains: [Nx, Ny, Nz]    # default [1, 1, 1] — Pattern 1, byte-exact M1..M6 path
comm:
  backend: mpi_host_staging | hybrid    # hybrid требует Pattern 2 (product >= 2)
```

Контракт (D-M7-2):

- **`subdomains: [1, 1, 1]`** (default): `outer_coord_ == nullptr`, scheduler работает в legacy single-subdomain режиме. M1..M6 thermo stream остаётся **byte-for-byte identical** — это Pattern 1 регрессионный инвариант, охраняемый T7.9 wire smoke + всеми M5/M6 2-rank goldens.
- **`subdomains` product ≥ 2**: `SimulationEngine::init()` строит `SubdomainGrid` через равно-тайловую orthogonal-brick декомпозицию (lex id = `ix + Nx*(iy + Ny*iz)`, `rank_of_subdomain[i] = i` — 1:1 биндинг per D-M7-2), создаёт `ConcreteOuterSdCoordinator` (Mode::kReference, std::map for determinism per OC-2) и инициализирует его с `k_max = scheduler.pipeline_depth_cap`. Если `td_mode: true`, coord приклеивается к inner scheduler через `attach_outer_coordinator(coord.get())` (borrowed pointer per master §8.2 — engine остаётся unique owner).

**Always-construct policy:** coord строится при product ≥ 2 независимо от `td_mode`. Это позволяет тестам и introspection-инструментам ассертить ownership и стартовое состояние frontier ([OC-5 ground state: `global_frontier_min == global_frontier_max == 0`](../scheduler/SPEC.md#23-outer-sd-coordinator)) без активации TD pipeline. При `td_mode=false` coord просто не приклеен к scheduler.

**Destruction order:** `outer_coord_` объявлен **после** `td_scheduler_` в class layout — реверсное уничтожение даёт coord first, scheduler second; safe потому что `CausalWavefrontScheduler` dtor defaulted и не разыменовывает borrowed coord pointer.

**Guards (preflight, T7.9):** schema-level Pattern 2 consistency:
- любая `subdomains[i] == 0` → `Error` с key path `zoning.subdomains` ("axis must be >= 1");
- `comm.backend=hybrid` + Pattern 1 (product == 1) → `Error` с key path `comm.backend` ("hybrid requires zoning.subdomains product >= 2");
- Pattern 2 + non-hybrid backend → `Warning` ("consider comm.backend='hybrid' for multi-node — current setting retains inner-only transport"). Сценарист может намеренно зафиксировать `mpi_host_staging` для отладки.

**Out of T7.9 scope** (отложено на T7.14 — full M7 integration smoke):
- runtime CUDA-aware-MPI probe + NCCL probe + transparent fallback chain;
- CLI-level `HybridBackend` construction (T7.5 артефакт уже в `comm/`, but engine-side wiring живёт с full 2-rank smoke);
- cross-subdomain halo traffic verification (текущий wire test упражняет coord construction + attach + 2-step run, но все zone'ы внутренние w.r.t. equal-tile geometry).

Scope для T7.9:
- YAML schema extension (`zoning.subdomains`, `comm.backend: hybrid`);
- preflight Pattern 2 consistency checks;
- engine-internal Pattern 2 wire (coord construct + attach);
- 1-process unit smoke (`tests/runtime/test_pattern2_engine_wire.cpp`) — Pattern 1 регрессия + Pattern 2 opt-in (td_mode=false / true) + 2 preflight rejection cases.

---

## 3. Lifecycle

### 3.1. `configure(RuntimeConfig)`

Проверяет:
- `config.exec_profile` valid;
- `config.backend` available (GPU if requested?);
- `config.pipeline_depth_cap` в разумных пределах `[1, 64]`;
- все required fields присутствуют.

На выходе: `state_ = Configured`, `config_` сохранён.

### 3.2. `resolve_policies()`

Читает `BuildFlavorInfo` (compile-time) + `RuntimeConfig` → строит `RuntimePolicyBundle`:

```
RuntimePolicyBundle = {
    ExecProfilePolicy,
    NumericPolicyView,
    SchedulerPolicy,
    ReductionPolicy,
    CommPolicy,
    ReorderPolicy,
    OverlapPolicy,
    ValidationPolicy
}
```

`PolicyFactory` (из master spec §7.1 / §30.13):

```
policies_ = PolicyFactory::make(build_info_, config_.exec_profile)
```

Проверяет matrix compatibility (§7.2 master spec). Incompatible (e.g. `MixedFast` build + `Reference` profile) → `Failed` state с clear error.

На выходе: `state_ = PoliciesResolved`.

### 3.3. `bootstrap_state(SimulationInput)`

`SimulationInput` — parsed config (из `io/`):

```cpp
struct SimulationInput {
    Box                       box;
    UnitMetadata              units;
    std::vector<AtomSoA>      initial_atoms;   // может быть split по species file
    std::vector<SpeciesInfo>  species;
    std::string               potential_style;
    PotentialConfig           potential_config;
    std::string               integrator_style;
    IntegratorConfig          integrator_config;
    double                    initial_dt;
    uint64_t                  initial_seed;
    // ... other optional fields
};
```

Runtime:
1. Создаёт `state_` (`DefaultStateManager`);
2. Регистрирует species;
3. Bulk-adds initial atoms;
4. Через `unit_converter_` конвертирует в native metal если нужно;
5. Вызывает `wrap_all_to_primary_image()`;
6. Записывает initial state для restart capability.

На выходе: `state_ = StateBootstrapped`.

### 3.4. `initialize_execution()`

Самая критичная transition. Создаёт все модули в правильном порядке с правильными dependencies:

```
1. unit_converter_ = make_unit_converter(input.units)
2. zoning_planner_ = make_zoning_planner()
3. zoning_plan_    = zoning_planner_.plan(
       state_.box(),
       potential_.cutoff(),
       config_.skin,
       comm_.nranks(),
       hint_from_perfmodel)
4. state_.update_zone_mapping(zoning_plan_)
5. neighbor_ = make_neighbor_manager()
6. neighbor_.build_cell_grid(...)
7. neighbor_.build_neighbor_list(...)
8. potential_ = PotentialFactory.create(
       input.potential_style,
       input.potential_config)
9. integrator_ = IntegratorFactory.create(
       input.integrator_style,
       input.integrator_config)
10. comm_ = CommBackendFactory.create(config_.comm_config)
11. perfmodel_ = make_perfmodel()
12. if (pattern == Pattern2):
       outer_coord_ = make_outer_coordinator(subdomain_grid)
13. scheduler_ = make_scheduler(policies_.scheduler)
    scheduler_.initialize(zoning_plan_)
    scheduler_.attach_outer_coordinator(outer_coord_.get())
14. telemetry_ = make_telemetry_sink(config_.telemetry)
```

На выходе: `state_ = ExecutionInitialized`.

**Order matters:** zoning plan нужен перед neighbor (для zone bounds); neighbor нужен перед scheduler (для certificate building); scheduler зависит от comm (outer coordinator); и т.д.

### 3.5. `run(n_steps)`

Main loop:

```
run(n_steps):
    state_ = Running
    start_time_ = now()

    target_step = current_step_ + n_steps

    while  current_step_ < target_step  and  not stop_requested_:
        iteration()
        current_step_++

        if  current_step_ % telemetry_interval == 0:
            telemetry_.flush()

        if  current_step_ % checkpoint_interval == 0:
            save_restart(auto_checkpoint_path)

    if  stop_requested_:
        state_ = Paused
    else:
        state_ = ExecutionInitialized
```

### 3.6. `iteration()` — каноническая одна итерация

Из master spec §6.6, уточнённое для runtime:

```
iteration():
    telemetry_.begin_iteration()

    # 1. Refresh state tracking:
    neighbor_.update_displacement(state_.atoms())
    if  neighbor_.skin_exceeded():
        neighbor_.request_rebuild("skin exceeded")

    if  neighbor_.rebuild_pending():
        neighbor_.execute_rebuild(state_, zoning_plan_)
        scheduler_.on_neighbor_rebuild_completed(affected_zones)

    # 2. Refresh certificates:
    scheduler_.refresh_certificates()

    # 3. Select ready tasks:
    tasks = scheduler_.select_ready_tasks()

    # 4. Execute:
    for task in tasks:
        zone_filter = make_zone_filter(task.zone_id)
        stream = scheduler_.assign_stream(task)

        integrator_.pre_force(state_, zone_filter, current_dt())
        potential_.compute(make_request(task), result)
        integrator_.post_force(state_, zone_filter, current_dt())
        integrator_.end_of_step(state_, zone_filter, current_dt())

        scheduler_.mark_completed(task)

    # 5. Progress communication:
    comm_.progress()
    for packet in comm_.drain_arrived_temporal():
        scheduler_.on_zone_data_arrived(packet.zone_id, packet.time_level, packet.version)
        state_.apply_packet(packet)

    # 6. Commit:
    scheduler_.commit_completed()

    # 7. Migrations (Pattern 2):
    if  current_step_ % migration_check_interval == 0:
        migrations = neighbor_.detect_migrations(state_, zoning_plan_)
        if not migrations.empty():
            neighbor_.apply_migrations(state_, zoning_plan_, migrations)

    # 8. Watchdog:
    scheduler_.check_deadlock(policies_.scheduler.t_watchdog)

    # 9. Telemetry:
    telemetry_.end_iteration(step_summary())
```

### 3.7. `finalize()`

- `scheduler_.shutdown()`;
- final comm drain;
- final telemetry flush;
- write final dump (если configured);
- write reproducibility bundle;
- `state_ = Finalized`.

### 3.8. `shutdown()`

- destroy modules in reverse order;
- comm backend finalize (MPI_Finalize если last rank);
- `state_ = Shutdown`.

---

## 4. Policy bundle

### 4.1. Структура

```cpp
struct RuntimePolicyBundle {
    ExecProfilePolicy    exec;
    NumericPolicyView    numeric;      // view on BuildFlavor
    SchedulerPolicy      scheduler;
    ReductionPolicy      reduction;
    CommPolicy           comm;
    ReorderPolicy        reorder;
    OverlapPolicy        overlap;
    ValidationPolicy     validation;
};

struct ExecProfilePolicy {
    ExecProfile    profile;
    bool           require_bitwise_determinism;
    bool           require_layout_invariant_execution;
    bool           require_scientific_reproducibility;
};

struct NumericPolicyView {
    std::string    build_flavor;              // "Fp64ReferenceBuild", ...
    std::string    numeric_config_id;         // typename hash
    bool           deterministic_reduction;
    bool           allow_device_atomics;
};

struct ReductionPolicy {
    bool           use_kahan_summation;       // Reference: true
    bool           fixed_tree_shape;          // Reference: true
    uint32_t       chunk_size_for_reduction;
};

struct CommPolicy {
    std::string    backend;                   // "auto" | "nccl" | ...
    bool           allow_async_progress;
    uint32_t       pipeline_depth_cap;        // K_max
};

struct ReorderPolicy {
    uint32_t       migration_fraction_threshold;    // % for triggering reorder
    bool           stable_sort_required;
};

struct OverlapPolicy {
    bool           overlap_compute_comm;      // Production/Fast: true
    uint32_t       stream_count;              // 3 default
};

struct ValidationPolicy {
    bool           run_differential_on_startup;  // for smoke test
    double         energy_drift_alarm_threshold;
};
```

### 4.2. PolicyFactory

```cpp
class PolicyFactory {
public:
    static RuntimePolicyBundle  make(
        const BuildFlavorInfo&,
        ExecProfile);

    static RuntimePolicyBundle  make_reference(const BuildFlavorInfo&);
    static RuntimePolicyBundle  make_production(const BuildFlavorInfo&);
    static RuntimePolicyBundle  make_fast_experimental(const BuildFlavorInfo&);

private:
    static bool  validate_compatibility(
        const BuildFlavorInfo&, ExecProfile,
        std::string& reason_if_invalid);
};
```

### 4.3. Compatibility matrix enforcement

Matrix из master spec §7.2:

| BuildFlavor | Reference | Production | FastExperimental |
|---|---|---|---|
| Fp64Reference | ✓ | ✓ | ⚠ warn |
| Fp64Production | ⚠ overkill | ✓ | ✓ |
| MixedFast | ✗ reject | ✓ validated | ✓ |
| Fp32Experimental | ✗ | ✗ | ✓ |

`validate_compatibility` возвращает:
- `true` with empty reason: OK, no warning;
- `true` with reason: warning emitted but proceed;
- `false` with reason: reject (`Failed` state).

---

## 5. Restart / resume

### 5.1. What gets saved

```
RestartBundle = {
    metadata: {
        tdmd_version,
        build_flavor,
        creation_timestamp,
        current_step,
        elapsed_wall_time
    },
    config: RuntimeConfig,
    build_info: BuildFlavorInfo,
    policies: RuntimePolicyBundle,
    state: {
        atoms: AtomSoA (serialized),
        box: Box,
        species: SpeciesRegistry,
        version: Version
    },
    scheduler_state: {
        current_step,
        frontier_min,
        frontier_max,
        zone_states: [(zone_id, state, time_level, version)],
        cert_store: [...]
    },
    telemetry_summary: last_N_iterations
}
```

### 5.2. Save semantics

`save_restart(path)`:
- pauses run loop (drains async ops);
- flushes all pending comm;
- serializes all modules to path (directory with multiple files);
- writes checksum manifest;
- resume OK after return.

Serialization format: **HDF5** (binary, cross-platform, LAMMPS-compatible где возможно). JSON metadata for introspection.

### 5.3. Load semantics

`load_restart(path)`:
- validates:
  - `tdmd_version` matches current binary;
  - `build_flavor` matches current binary (cannot resume Reference run in Mixed build);
  - checksum manifest matches.
- если mismatch — `Failed` with clear error.
- populates all modules from bundle;
- resumes at saved `current_step`.

### 5.4. Auto-checkpoint

```yaml
runtime:
  checkpoint:
    interval: 10000          # steps
    keep_last_n: 3           # rolling buffer
    path: ./checkpoints/
```

Runtime automatically saves every `interval` steps, keeps last N.

### 5.5. Bitwise resume test

Unit test: run N steps → save → shutdown → load → run N more steps. Compare с uninterrupted run of 2N steps. **Must be bitwise identical** в Reference profile.

This is a strong determinism test, catches subtle state-not-captured bugs.

---

## 6. Graceful stop

### 6.1. request_stop()

Safe для call из signal handler (SIGINT) или other thread:

```
request_stop():
    stop_requested_.store(true)
```

Run loop checks flag at each iteration boundary. При срабатывании:
1. Completes current iteration;
2. Commits all in-flight tasks;
3. Final telemetry flush;
4. Transitions to `Paused`.

User может `resume()` после `request_stop()` если change of mind.

### 6.2. Signal handlers

`cli/` регистрирует signal handlers для SIGINT, SIGTERM → `engine.request_stop()`. Это позволяет grapcefully выходить из long runs.

---

## 7. Pattern awareness

### 7.1. Pattern detection

Runtime определяет pattern автоматически:

```
if  n_ranks == 1:
    pattern = Pattern1   # single-rank, TD
elif  outer_coord_grid_configured:
    pattern = Pattern2   # hybrid
elif  td_disabled_in_config:
    pattern = Pattern3   # SD-vacuum
else:
    pattern = Pattern1   # default
```

### 7.2. Pattern-specific initialization

**Pattern 1:** `outer_coord_ = nullptr`, `comm_backend` — single-rank trivial или intra-node NCCL.

**Pattern 2:** `outer_coord_ = make_outer_coordinator(...)`, `comm_backend = HybridBackend`, subdomain grid planned через separate planner.

**Pattern 3:** TD disabled в scheduler policy (K = 1, no temporal packets), comm backend — чистый MPI halo exchange.

### 7.3. Reconfiguration at runtime

Pattern **не меняется mid-run**. Switch Pattern 1 → Pattern 2 требует restart (`save_restart` + new config + `load_restart`).

---

## 8. Reproducibility bundle

### 8.1. Contents

На каждом run, runtime создаёт (в `./repro_bundle_<run_id>/`):

```
repro_bundle_<run_id>/
├── config.yaml              # full effective config after defaults
├── build_info.json          # git_sha, compiler, CUDA version, ...
├── policies.json            # resolved RuntimePolicyBundle
├── hardware_profile.json    # probed hardware stamp
├── initial_state.h5         # state at step 0
├── species_registry.json
├── potential_checksum.txt
├── neighbor_list_first.h5   # first neighbor list (for compare)
├── seed.txt
└── manifest.json            # CRC32 of all above
```

### 8.2. Purpose

Позволяет другому researcher'у в далеком будущем:
1. Проверить, в какой именно конфигурации был run;
2. Воспроизвести результат если имеет doma matching hardware class;
3. Compare results between different runs with confidence.

### 8.3. Invariants

**Bundle immutable after creation.** Если run continues, new run_id создаётся.

**Bundle самодостаточен.** Не ссылается на external files.

---

## 9. Error handling

### 9.1. Categories

1. **Configuration errors** — bad config, invalid units, missing files → clear user message + `Failed`;
2. **Computation errors** — NaN / Inf в forces, negative density → detailed dump + `Failed`;
3. **Network errors** — MPI failure, timeout → retry per policy, then escalate;
4. **Out-of-memory** — log summary, try graceful shutdown;
5. **Unknown errors** — log stack trace, dump state, `Failed`.

### 9.2. Failure mode

`Failed` state — terminal. No recovery без user intervention. `shutdown()` still callable для cleanup.

Diagnostic dump включает:
- last iteration summary;
- zone states count;
- frontier info;
- comm status;
- recent telemetry snapshots.

Saved в `./failure_dump_<timestamp>/`.

---

## 10. Tests

### 10.1. State machine tests

- Legal transitions: walk through all with test harness;
- Illegal transitions: attempt, verify correct error;
- Failed state: verify reachable from each state, verify terminal.

### 10.2. Policy resolution tests

- Each BuildFlavor × ExecProfile combination → verify correct bundle;
- Incompatible combinations → verify `Failed`.

### 10.3. Lifecycle integration tests

- Full run: configure → resolve → bootstrap → init → run 100 steps → finalize → shutdown;
- Restart round-trip (bitwise test above);
- Graceful stop at step 50 → resume → final identical to uninterrupted.

### 10.4. Module wiring tests

- Mock modules that record calls → verify `iteration()` calls them in right order with right params;
- Dependency injection: swap real zoning planner with stub → verify rest works.

### 10.5. Signal handling tests

- SIGINT mid-run → pause gracefully;
- Signal twice → force shutdown;
- Resume after signal.

---

## 11. Telemetry

Runtime-level metrics:
```
runtime.current_step
runtime.elapsed_wall_seconds
runtime.steps_per_second
runtime.iteration_time_ms_avg
runtime.iteration_time_ms_p99
runtime.state_transitions_total
runtime.checkpoint_count
runtime.restart_count
```

NVTX ranges:
- `SimulationEngine::iteration`;
- `SimulationEngine::save_restart`;
- `SimulationEngine::load_restart`.

---

## 12. Roadmap alignment

| Milestone | Runtime deliverable |
|---|---|
| M1 | `SimulationEngine` basic lifecycle; no TD, single-rank |
| M2 | Policy resolution for Reference profile; reproducibility bundle; checkpoint |
| M4 | TD scheduler integration; full iteration function |
| M5 | Restart/resume tests passing |
| M6 | GPU initialization path — `runtime.backend: gpu` wires `GpuContext` + `EamAlloyGpuAdapter` + `GpuVelocityVerletIntegrator`; T6.7 1-rank + 2-rank byte-exact gates |
| **M7** | **Pattern 2 detection + outer coordinator wiring (T7.9 schema + engine wire + preflight; T7.14 transport probe + full 2-rank smoke)** |
| M8 | Auto-tune policies from perfmodel |
| v2+ | Dynamic pattern switching (pause → reconfigure → resume) |

---

## 13. Open questions

1. **Checkpoint granularity** — step-level granularity достаточно? Substep granularity (e.g. зоны) — сложнее но enables finer restart control.
2. **Multi-binary compatibility** — resume saved by binary A в binary B с same build_flavor но different commit SHA? Reject или allow с warning?
3. **Hot config reload** — mid-run config changes (e.g. increase log verbosity)? Post-v1.
4. **Subdomain grid planning** — runtime делает или отдельный модуль? Сейчас — runtime's responsibility в Pattern 2 init. Может стать отдельным `subdomain_planner/` модулем.
5. **Async checkpointing** — чтобы save_restart не pause'ил run loop. Важно для production; overhead-sensitive.

---

*Конец runtime/SPEC.md v1.0.2, дата: 2026-04-19 (T7.9 §2.4 Pattern 2 engine wiring).*

---

## Change log

- **v1.0.3 — 2026-04-20 (T7.14):** §2.4 Pattern 2 engine wiring validated end-to-end by `tests/integration/m7_smoke/` — 2-rank 2-subdomain `runtime.backend: gpu` + `zoning.subdomains: [2,1,1]` + `scheduler.td_mode: true, pipeline_depth_cap: 1` + `comm.backend: mpi_host_staging`. Thermo byte-for-byte matches the M6 golden (D-M7-10 chain: M3 ≡ M4 ≡ M5 ≡ M6 ≡ M7 Pattern 2 K=1 P_space=2). Preflight Pattern 2 consistency (T7.9) confirmed on both the single-rank pre-fail path (step 3/7) and the 2-rank launch path (step 4/7). The runtime GPU wire + M7 scheduler attach run clean on a production-shaped run loop without regressing Pattern 1. M7 milestone closed on this path; HybridBackend transport exercise is T7.5/T7.11 territory (orthogonal; not in the byte-exact gate).
- **v1.0.2 — 2026-04-19 (T7.9):** added §2.4 Pattern 2 engine wiring — `zoning.subdomains` + `comm.backend: hybrid` YAML knobs, equal-tile orthogonal-brick decomposition with lex id + 1:1 rank binding (D-M7-2), `outer_coord_` ownership + attach contract, preflight Pattern 2 consistency, Pattern 1 byte-exact regression invariant. Transport probe + full 2-rank smoke deferred to T7.14.
- **v1.0.1 — 2026-04-19 (T6.7):** added §2.3 GPU backend wiring — `runtime.backend` knob, `GpuContext` RAII ownership, EAM/alloy + VV GPU path, M6 byte-exact gates.
- **v1.0.0 — initial:** baseline runtime/SPEC.
