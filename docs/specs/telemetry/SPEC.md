# telemetry/SPEC.md

**Module:** `telemetry/`
**Status:** master module spec
**Parent:** `TDMD Engineering Spec v2.1` §9 (GPU/NVTX), §12
**Last updated:** 2026-04-16

---

## 1. Purpose и scope

### 1.1. Что делает модуль

`telemetry/` — наблюдатель. Единственный модуль, которому разрешено агрегировать и записывать runtime-метрики из всех других модулей.

Делает четыре вещи:

1. **Metrics collection** — timing breakdown, pipeline stats, step summaries;
2. **Structured logging** — key-value events, JSON lines format;
3. **NVTX ranges** — integration with Nsight Systems для GPU profiling;
4. **Performance reports** — formatted output, LAMMPS-compatible breakdown, comparison tables.

### 1.2. Scope: что НЕ делает

- **не делает decisions** на основе метрик (это scheduler / perfmodel);
- **не меняет state модулей** — read-only observer;
- **не делает file I/O напрямую** (delegates to `io/`);
- **не отвечает за `std::cout` output** (это `cli/`).

### 1.3. Read-only invariant

Master spec §8.2: `telemetry/` — **read-only наблюдатель**. Не имеет права mutate any other module's state. Тест: если telemetry отключена целиком, simulation runs identically (модuloshould not mission-critical logging).

Единственное исключение: telemetry может incrementally update **свои** internal counters (это часть self-observation).

---

## 2. Public interface

### 2.1. Core types

```cpp
namespace tdmd {

struct TimingBreakdown {
    double  t_pair          = 0.0;   // pair potential force compute
    double  t_neigh         = 0.0;   // neighbor list build / update
    double  t_comm          = 0.0;   // total comm
    double  t_comm_inner    = 0.0;   // TD temporal packets
    double  t_comm_outer    = 0.0;   // SD halo (Pattern 2)
    double  t_integrate     = 0.0;   // integrator steps
    double  t_scheduler     = 0.0;   // scheduler overhead
    double  t_output        = 0.0;   // dumps, checkpoints
    double  t_pack          = 0.0;   // pack for send
    double  t_unpack        = 0.0;   // unpack on receive
    double  t_other         = 0.0;

    double  t_total() const {
        return t_pair + t_neigh + t_comm + t_integrate
             + t_scheduler + t_output + t_pack + t_unpack + t_other;
    }
};

struct PipelineStats {
    uint64_t  zones_ready           = 0;
    uint64_t  zones_inflight        = 0;
    uint64_t  zones_committed_total = 0;
    uint64_t  certificate_failures  = 0;
    uint64_t  neighbor_rebuilds     = 0;
    uint64_t  boundary_stalls       = 0;   // Pattern 2
    uint64_t  deadlock_warnings     = 0;
    double    current_pipeline_depth = 0;   // frontier_max - frontier_min
    double    scheduler_idle_fraction = 0;  // [0, 1]
};

struct StepSummary {
    uint64_t       step;
    double         time_ps;
    double         dt;

    // Thermodynamic observables:
    double         temperature_K;
    double         potential_energy;
    double         kinetic_energy;
    double         total_energy;
    double         pressure;
    double         volume;
    double         density;

    // Runtime breakdown:
    TimingBreakdown  timing;
    PipelineStats    pipeline;

    // Quality indicators:
    double         nve_drift_cumulative;
    double         max_displacement;
    uint64_t       neighbor_list_size;
};

} // namespace tdmd
```

### 2.2. TelemetrySink (главный интерфейс)

```cpp
class TelemetrySink {
public:
    // Iteration lifecycle:
    virtual void  begin_iteration(uint64_t step) = 0;
    virtual void  end_iteration(const StepSummary&) = 0;

    // Event emission (zero-cost if disabled):
    virtual void  emit_event(
        const std::string& category,
        const std::string& name,
        std::initializer_list<std::pair<std::string_view, std::string_view>> kv) = 0;

    virtual void  emit_metric(
        const std::string& name,
        double value) = 0;

    virtual void  emit_counter(
        const std::string& name,
        int64_t delta = 1) = 0;

    // NVTX integration (GPU tracing):
    virtual uint64_t  nvtx_range_start(const std::string& name) = 0;
    virtual void      nvtx_range_end(uint64_t handle) = 0;

    // Flush:
    virtual void  flush() = 0;

    // Query / introspection:
    virtual double  get_metric_last(const std::string& name) const = 0;
    virtual int64_t get_counter(const std::string& name) const = 0;

    virtual ~TelemetrySink() = default;
};

class DefaultTelemetrySink final : public TelemetrySink { /*...*/ };
class NullTelemetrySink     final : public TelemetrySink { /*...*/ };  // disabled
```

### 2.3. Scoped helpers

```cpp
// RAII helper: NVTX range + timing measurement in one:
class ScopedRange {
public:
    ScopedRange(TelemetrySink& sink, const std::string& name);
    ~ScopedRange();    // auto-closes range, emits timing metric
};

#define TDMD_TELEMETRY_SCOPE(sink, name) \
    tdmd::ScopedRange __tdmd_scope_##__LINE__(sink, name)
```

Usage:

```cpp
void some_function() {
    TDMD_TELEMETRY_SCOPE(engine.telemetry(), "morse_compute");
    // ... work
    // Timing + NVTX range auto-recorded
}
```

---

## 3. Metrics taxonomy

### 3.1. Namespace conventions

Metrics use dotted namespaces для organizability:

```
<module>.<metric_name>

Examples:
scheduler.zones_ready_count
scheduler.commit_latency_ms
neighbor.rebuilds_total
comm.bytes_sent_total
potentials.morse.force_time_ms
integrator.current_dt
runtime.current_step
perfmodel.prediction_accuracy
```

### 3.2. Metric types

- **Counter** — monotonic integer (total events since start). E.g., `scheduler.zones_committed_total`.
- **Gauge** — instantaneous value (can go up/down). E.g., `scheduler.current_pipeline_depth`.
- **Histogram** — distribution (min, max, avg, p99). E.g., `scheduler.commit_latency_ms_histogram`.
- **Timing** — specialized histogram в time units. E.g., `potentials.morse.compute_time_ms`.

### 3.3. Canonical metrics list

Полный список канонических метрик для всех модулей:

**scheduler:**
```
scheduler.zones_ready_count                 gauge
scheduler.zones_inflight_count              gauge
scheduler.zones_committed_total             counter
scheduler.certificate_failures_total        counter
scheduler.neighbor_rebuilds_total           counter
scheduler.deadlock_warnings_total           counter
scheduler.boundary_stalls_total             counter      (Pattern 2)
scheduler.task_selection_time_ms            timing
scheduler.commit_latency_ms                 timing
scheduler.current_frontier_min              gauge
scheduler.current_frontier_max              gauge
scheduler.pipeline_depth                    gauge
scheduler.idle_fraction                     gauge
```

**neighbor:**
```
neighbor.cell_grid_build_time_ms            timing
neighbor.neighbor_list_build_time_ms        timing
neighbor.neighbor_list_size_total           gauge
neighbor.rebuilds_total                     counter
neighbor.rebuilds_skin_triggered            counter
neighbor.rebuilds_migration_triggered       counter
neighbor.migrations_total                   counter
neighbor.reorder_operations_total           counter
neighbor.displacement_max_current           gauge
```

**potentials:**
```
potentials.<style>.force_time_ms            timing
potentials.<style>.energy                    gauge
potentials.<style>.virial_xx                  gauge       (etc)
potentials.<style>.parameter_checksum         gauge        (static)
```

**comm:**
```
comm.packets_sent_temporal_total            counter
comm.packets_sent_halo_total                counter
comm.bytes_sent_total                       counter
comm.send_latency_ms_avg                    timing
comm.crc_failures_total                     counter
comm.bytes_per_second_measured              gauge
comm.inner_traffic_bytes_total              counter     (Pattern 2)
comm.outer_traffic_bytes_total              counter     (Pattern 2)
```

**integrator:**
```
integrator.steps_total                      counter
integrator.current_dt                       gauge
integrator.dt_adaptive_changes_total        counter
integrator.avg_temperature                  gauge       (NVT)
integrator.nve_drift_cumulative              gauge
```

**state:**
```
state.atom_count                            gauge
state.version_current                       gauge
state.mutations_per_iteration_avg           gauge
state.gpu_sync_operations_total             counter
state.gpu_sync_bytes_total                  counter
```

**runtime:**
```
runtime.current_step                        gauge
runtime.elapsed_wall_seconds                gauge
runtime.steps_per_second                    gauge
runtime.iteration_time_ms                   timing
runtime.checkpoint_count                    counter
```

**perfmodel:**
```
perfmodel.prediction_accuracy               gauge         ([0, 1])
perfmodel.recommended_K                     gauge
perfmodel.validation_errors_total           counter
```

---

## 4. Output formats

### 4.1. Real-time log (JSON lines)

```jsonl
{"ts":"2026-04-16T12:34:56.123Z","step":100,"event":"iteration_end","duration_ms":2.3,"pipeline_depth":4}
{"ts":"...","step":100,"event":"metric","name":"scheduler.zones_ready_count","value":12}
{"ts":"...","step":100,"event":"neighbor_rebuild","trigger":"skin","atoms_moved":150}
```

Easy parsing via `jq`, pandas `read_json(lines=True)`, splunk и др.

### 4.2. LAMMPS-compatible breakdown

На конце run (или каждые N steps при request):

```
Performance: 1234.5 tau/day, 0.0038 ns/day, 2.3 ms/timestep

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    |     1.200  |     1.250  |     1.310  |   2.1 |  54.3
Neigh   |     0.200  |     0.220  |     0.240  |   1.5 |   9.6
Comm    |     0.150  |     0.200  |     0.300  |   5.2 |   8.7
Integrator |  0.100  |     0.120  |     0.130  |   1.1 |   5.2
Scheduler|    0.050  |     0.055  |     0.060  |   0.8 |   2.4
Output  |     0.010  |     0.020  |     0.050  |   7.2 |   0.9
Other   |     0.425  |     0.435  |     0.450  |   0.7 |  18.9
---------------------------------------------------------------
Total   |     2.135  |     2.300  |     2.540  |     - |  100.0

Nlocal:    12500 ave    12500 max    12500 min
Nghost:     3200 ave     3250 max     3150 min
Neighs:   650000 ave   650000 max   650000 min

TDMD-specific pipeline stats:
  Pipeline depth: 4.0 avg, 5 max
  Zones committed: 125000
  Certificate failures: 0
  Boundary stalls (Pattern 2): 0
  Scheduler idle fraction: 3.2%
```

### 4.3. HDF5 archive (long-term storage)

Identically к TrajectoryWriter, telemetry может writes HDF5:

```
/tdmd_telemetry/
  /metrics/
    <metric_name>/
      timestamps            # (N,)
      values                # (N,) or (N, K) for histograms
  /events/
    <category>_<name>/
      timestamps
      payload               # JSON strings
  /metadata/
    run_id
    build_info
```

Benefits: analysis через pandas, numpy. Long-term archival.

### 4.4. Streaming output (future: Prometheus / Grafana)

Post-v1: TelemetrySink publishes metrics to Prometheus-compatible endpoint. Users monitor live runs with Grafana dashboards.

---

## 5. NVTX integration

### 5.1. Purpose

NVIDIA NVTX ranges позволяют видеть performance breakdown в Nsight Systems timeline view — critical для GPU optimization.

### 5.2. Range conventions

Каждый модуль emits NVTX ranges для своих hot functions:

**scheduler:**
- `TdScheduler::select_ready_tasks`
- `TdScheduler::refresh_certificates`
- `TdScheduler::commit_completed`

**potentials:**
- `MorsePotential::compute`
- `EamPotential::density_kernel`
- `EamPotential::force_kernel`

**neighbor:**
- `NeighborManager::build_cell_grid`
- `NeighborManager::build_neighbor_list`

**comm:**
- `CommBackend::send_temporal_packet`
- `CommBackend::drain_arrived_temporal`

### 5.3. Color coding

Convention:
- Green: compute (potentials, integrator);
- Blue: memory (state sync, pack/unpack);
- Orange: communication;
- Red: rebuild / reorder;
- Grey: scheduler, telemetry.

### 5.4. Levels

`ScopedRange` имеет level parameter:
- `Level::Always` — всегда emit (overhead ~100 ns);
- `Level::Verbose` — only when `verbose_nvtx` flag set;
- `Level::Debug` — only в debug builds.

Default: most ranges at `Verbose`, critical hot paths at `Always`.

### 5.5. Disabling

```cpp
#ifdef TDMD_NVTX_ENABLED
    nvtxRangeStartA(name);
#else
    // no-op
#endif
```

Non-GPU builds compile NVTX calls to no-ops.

---

## 6. Performance overhead budget

Telemetry must have **negligible impact** on run time. Budget:

- Counter/metric emit: `< 50 ns` (atomic increment);
- Event emit: `< 500 ns` (formatted string + queue);
- NVTX range (Level::Verbose): `< 200 ns`;
- NVTX range (Level::Always): `< 500 ns`;
- Histogram update: `< 100 ns` (bucket increment).

Total telemetry overhead per iteration: `< 0.1%` of iteration time для typical EAM workload (5 ms/iteration).

**Mandatory benchmark:** перф-gate в CI проверяет что telemetry включенная не регрессирует T3 benchmark > 1%.

### 6.1. Disabled telemetry

When `telemetry.level = off`, все `TelemetrySink` calls — no-op. `NullTelemetrySink` используется. Overhead practically zero.

### 6.2. Async logging

В `DefaultTelemetrySink`:
- metrics / events written в in-memory ring buffer (non-blocking);
- separate thread drains buffer → file/network.

Avoids blocking compute на slow I/O.

---

## 7. Configuration

```yaml
telemetry:
  enabled: true
  level: info                  # off | error | warn | info | debug

  output:
    format: jsonl              # jsonl | hdf5 | console
    path: ./tdmd.log
    rotate_size_mb: 100
    keep_files: 5

  metrics:
    collection_enabled: true
    flush_interval_iterations: 100

  nvtx:
    enabled: true
    level: verbose             # off | always | verbose | debug
    color_scheme: standard

  breakdown_print:
    interval_steps: 1000       # 0 = only at end
    format: lammps_compatible

  prometheus:                   # post-v1
    enabled: false
    port: 9090
```

### 7.1. Level semantics

- `off`: all calls no-op, `NullTelemetrySink` used;
- `error`: only errors emitted;
- `warn`: errors + warnings;
- `info`: errors + warnings + step summaries + major events;
- `debug`: everything including per-iteration verbose.

Production default: `info`. Development: `debug`. Benchmarking: `info` (but metrics enabled).

---

## 8. Testing

### 8.1. Unit tests

- **Metric emit:** counter += 1 after emit; gauge = last value;
- **Histogram:** 1000 values → correct min/max/avg/p99;
- **NVTX:** (без real Nsight, но) range starts и ends called correctly;
- **Null sink:** all methods no-op без side effects.

### 8.2. Performance tests

- **Overhead measurement:** iteration with telemetry vs without. Budget: < 0.1%.
- **Async drain:** burst 10k events in tight loop, drain happens в background, no main-thread blocking.

### 8.3. Correctness tests

- **Thread-safety:** multi-threaded metric emission → no corrupt counters;
- **Flush correctness:** metrics emitted перед flush — all persisted;
- **Shutdown cleanness:** telemetry disposes gracefully when engine shuts down.

### 8.4. Integration tests

- **Full run с telemetry:** expected metrics appear в output log;
- **Breakdown print:** после short run, breakdown section formatted correctly;
- **HDF5 archive:** metrics queryable through h5py.

### 8.5. LAMMPS compat test

Run canonical benchmark, compare TDMD breakdown format с `LAMMPS MPI task timing breakdown` — format должен быть similar enough для pandas / awk scripts chewing both.

---

## 9. Module ownership of metrics

Each module — **author** of its own metrics. Telemetry sink aggregates. Conventions:

1. Each module `.cpp` включает `#include "telemetry/telemetry.hpp"`.
2. Static metric names as `const char*` constants в module header:
   ```cpp
   // scheduler.hpp
   constexpr const char* METRIC_ZONES_READY = "scheduler.zones_ready_count";
   ```
3. Module calls `sink.emit_metric(METRIC_ZONES_READY, value)` from hot loops.
4. New metric = module SPEC update + central metrics list update в этом spec.

---

## 10. Roadmap alignment

| Milestone | Telemetry deliverable |
|---|---|
| **M1** | Skeleton `TelemetrySink`; JSON lines output; basic step summaries |
| M2 | Complete canonical metrics list для all then-existing modules |
| M3 | `ScopedRange` + timing measurement integration |
| M4 | LAMMPS-compatible breakdown format |
| M5 | Histograms для latency-sensitive metrics |
| **M6** | **Full NVTX integration; colored ranges in Nsight** |
| M7 | Pattern 2 metrics (inner/outer traffic ratio, boundary stalls) |
| M8 | HDF5 archive output |
| v2+ | Prometheus endpoint; Grafana dashboards; distributed tracing |

---

## 11. Open questions

1. **Sampling vs full recording** — some metrics (per-atom) are too large to record every step. Use sampling (1 in 100)? How to balance accuracy / overhead?
2. **Retention policy** — how long keep historical metric data? Current: bounded ring buffer. Production: explicit config.
3. **Distributed aggregation** — in Pattern 2 multi-rank, metrics from всех ranks. Aggregation algorithm (sum / mean / max / all separately) per metric?
4. **Structured logging standard** — OpenTelemetry compatible? Useful для future integration с observability stacks.
5. **Privacy / security** — metrics may include path info, user info. Redaction policy для public reports?
6. **Cross-language consumers** — Python analysis tools need easy access. Parquet format для metrics?

---

*Конец telemetry/SPEC.md v1.0, дата: 2026-04-16.*
