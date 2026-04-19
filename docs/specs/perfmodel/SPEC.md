# perfmodel/SPEC.md

**Module:** `perfmodel/`
**Status:** master module spec v1.1 (T6.11 shipped — GPU cost tables + `predict_step_gpu_sec`)
**Parent:** `TDMD Engineering Spec v2.1` §4, §4a, §12.7
**Last updated:** 2026-04-19

---

## 1. Purpose и scope

### 1.1. Что делает модуль

`perfmodel/` — **analytic predictor производительности TDMD**. Отвечает на вопрос «при данной задаче и данном железе, какой deployment pattern, какой `K`, и какое число ranks дадут оптимальную скорость?».

Это **differentiator проекта**, а не nice-to-have. В любой MD-системе быстродействие — эмпирический вопрос; в TDMD мы заявляем, что умеем **предсказывать** производительность до запуска, и наша bench-суита это валидирует. Без perf-model проект не имеет научно воспроизводимого критерия «TDMD работает правильно».

Формально модуль делает пять вещей:

1. **Predict** `T_step` для Pattern 1 / 2 / 3;
2. **Recommend** оптимальные параметры `(pattern, K, P_space, P_time)`;
3. **Explain** в user-facing форме, почему рекомендация именно такая;
4. **Validate** себя — измеряет реальные run'ы и проверяет `|predict - measure| < tolerance`; при превышении — CI gate;
5. **Diagnose** регрессии — если прогноз правильный, а реальность внезапно хуже, указывает на причину (scheduler idle? neighbor rebuild overhead? comm stall?).

### 1.2. Scope: что НЕ делает perfmodel

- **не принимает решения** об исполнении — только рекомендует; окончательно решает `runtime/` + user;
- **не меряет time** — это `telemetry/`; perfmodel потребляет измерения для validation;
- **не оптимизирует kernels** — он anlytic, а не empirical auto-tuner;
- **не заменяет benchmark suite** — дополняет её; benchmarks измеряют ground truth, perfmodel предсказывает.

### 1.3. Философия: analytic first

Perfmodel в TDMD — **analytic model** (параметризованная физикой и топологией), **не ML**. Причины:

- analytic model interpretable: можно объяснить пользователю «вот формула, вот вклады»;
- сложная ML-модель в этом домене переобучается легче чем даёт прогресс (данных мало);
- analytic model дифференцируется по параметрам: можно найти `argmin_K T_step(K)` аналитически.

ML-based correction **может** быть добавлена post-v1 как слой over analytic baseline.

---

## 2. Public interface

### 2.1. Базовые типы (из мастер-специи §12.7)

```cpp
namespace tdmd {

struct HardwareProfile {
    // Compute:
    double   peak_flops_fp64;          // per device
    double   peak_flops_fp32;
    uint32_t cuda_cores;
    uint32_t sm_count;
    double   memory_bw_device_bytes_per_sec;

    // Interconnect:
    double   nvlink_bw_bytes_per_sec;      // 0 if no NVLink
    double   pcie_bw_bytes_per_sec;
    double   mpi_bw_bytes_per_sec;          // effective after protocol overhead
    double   mpi_latency_us;

    // Topology:
    uint32_t  gpus_per_node;
    uint32_t  nodes;
    std::string  interconnect_topology;     // "fat-tree", "torus", ...

    // Derived capabilities:
    bool  has_gpu_aware_mpi;
    bool  has_nccl;
    bool  has_nvshmem;
};

struct PerformancePrediction {
    // Per-pattern predictions:
    double   t_step_pattern1_seconds;   // чистый TD, single-subdomain
    double   t_step_pattern2_seconds;   // two-level TD × SD
    double   t_step_pattern3_seconds;   // SD-vacuum, чистый SD

    // Decomposition breakdown (for the recommended pattern):
    double   t_compute_seconds;         // force + integrate
    double   t_comm_inner_seconds;      // temporal packets
    double   t_comm_outer_seconds;      // SD halo (Pattern 2 only)
    double   t_scheduler_overhead_seconds;
    double   t_neighbor_rebuild_amortized;

    // Speedup analysis:
    double   speedup_td_over_sd;        // Pattern 1 vs Pattern 3
    double   speedup_hybrid_over_pure_sd; // Pattern 2 vs Pattern 3

    // Recommendations:
    std::string  recommended_pattern;   // "Pattern1" | "Pattern2" | "Pattern3"
    uint32_t     recommended_K;
    uint32_t     recommended_P_space;
    uint32_t     recommended_P_time;

    // Confidence:
    double   prediction_uncertainty_pct;   // wider for edge configurations
    std::vector<std::string>  warnings;    // e.g. "box too small for Hilbert3D"

    // Explanation:
    std::string  rationale;              // human-readable
    std::string  rationale_detailed;     // for `tdmd explain --perf --verbose`
};

} // namespace tdmd
```

### 2.2. Главный интерфейс

```cpp
namespace tdmd {

class PerfModel {
public:
    virtual PerformancePrediction predict(
        const ZoningPlan&,
        const PotentialModel&,
        const RuntimeConfig&,
        const HardwareProfile&) const = 0;

    virtual std::string explain(
        const PerformancePrediction&,
        int verbosity_level) const = 0;

    virtual ~PerfModel() = default;
};

class AnalyticPerfModel final : public PerfModel {
    // реализация формул из §3 этой спеки
};

} // namespace tdmd
```

### 2.3. Validation interface (для CI)

```cpp
struct PerfValidationReport {
    double    t_step_predicted;
    double    t_step_measured;
    double    relative_error;
    bool      passed_gate;
    double    gate_threshold;
    std::string  diagnosis;     // если не passed
};

class PerfValidator {
public:
    virtual PerfValidationReport validate(
        const PerformancePrediction&,
        const TimingBreakdown& measured) const = 0;

    virtual ~PerfValidator() = default;
};
```

---

## 3. Модель: формулы

### 3.1. Базовые компоненты

**Per-rank work (common для всех patterns):**
```
N_atoms_per_rank = N_total / (P_space · P_time)
N_neighbors_per_atom = ρ_atoms · V_cutoff   (≈ 50-300 typically)

T_c = (N_atoms_per_rank · C_force_per_atom) / FLOPS_rank
```

где `C_force_per_atom` — FLOP count per atom per force evaluation, зависит от потенциала:

| PotentialKind | C_force_per_atom (approx) |
|---|---|
| Pair (Morse, LJ) | 30-50 × N_neighbors |
| ManyBodyLocal (EAM) | 80-150 × N_neighbors |
| ManyBodyLocal (MEAM) | 300-500 × N_neighbors |
| Descriptor (SNAP) | 2000-10000 × N_neighbors |
| Descriptor (PACE) | 5000-50000 × N_neighbors |

Эти константы — **calibrated empirically** (см. §5.1).

### 3.2. Pattern 3 (чистый SD, baseline)

```
T_step_SD = T_c + T_halo_SD

T_halo_SD = 2 · N_surface_atoms · atom_record_size / B_inter_rank
          = 2 · (N_atoms_per_rank)^(2/3) · atom_record_size / B_inter_rank
```

где `B_inter_rank` — effective bandwidth между ranks (MPI или PCIe).

Значительный overhead при большом `P` из-за surface-to-volume scaling.

### 3.3. Pattern 1 (чистый TD)

```
T_step_TD(K) = T_c + T_comm_inner(K)

T_comm_inner(K) = max(0, T_p / K - T_c_overlap)
T_p = atom_record_size · N_atoms_per_zone / B_intra_rank
```

В K-batched pipeline:
- при `K = 1`, `T_comm_inner` может почти полностью перекрыться с `T_c` через async;
- при `K > 1`, effective cost per step ещё меньше.

**Оптимальное K:**
```
K_opt = argmin_K T_step_TD(K)
     =  sqrt(T_p / T_c_startup_per_K)
     ≈  round_to_nearest_power_of_2
```

Обычно `K_opt ∈ {1, 2, 4, 8}` для small-to-medium, `K_opt ∈ {16, 32, 64}` для large + slow network.

### 3.4. Pattern 2 (two-level hybrid)

```
T_step_hybrid(K, P_space, P_time) = T_c + max(T_h_outer(P_space), T_p_inner / K)

T_h_outer(P_space) = 2 · (N_total / P_space)^(2/3) · atom_record_size / B_outer
T_p_inner(P_time) = atom_record_size · N_atoms_per_zone / B_inner
```

Ключевое свойство: `T_h_outer` зависит только от `P_space`, `T_p_inner/K` зависит только от `K`. Они **независимо** оптимизируются.

**Selection algorithm:**

```
function recommend(config, hw):
    candidates = []

    # Pattern 1: P_space = 1, vary K
    for  K in [1, 2, 4, 8, 16]:
        t = T_step_TD(K, config, hw)
        candidates.append(("Pattern1", K, 1, P_total, t))

    # Pattern 2: vary P_space, K
    for  P_space in [1, 2, 4, 8, ...]  if  P_total % P_space == 0:
        P_time = P_total / P_space
        for  K in [1, 2, 4, 8, 16]:
            t = T_step_hybrid(K, P_space, P_time, config, hw)
            candidates.append(("Pattern2", K, P_space, P_time, t))

    # Pattern 3: reference
    t = T_step_SD(P_total, config, hw)
    candidates.append(("Pattern3", 1, P_total, 1, t))

    best = argmin(candidates by t)
    return best
```

### 3.5. Overhead corrections

Базовые формулы не учитывают:

- **Scheduler overhead** — дополнительное `T_sched ≈ 10-50 μs` на iteration, independent of N_atoms.
- **Neighbor rebuild amortization** — полный rebuild каждые `R` шагов, стоит `T_rebuild ≈ O(N_atoms_per_rank)`. Amortized: `T_rebuild / R`.
- **Migration amortization** — migration атомов между зонами/ranks, `T_migrate · f_migrate`.
- **Startup latency (первые `N_min · P` шагов)** — pipeline fill, где все ranks ждут.

Итоговая формула для Pattern 1:
```
T_step_effective = T_step_TD(K)
                 + T_sched
                 + T_rebuild / R_rebuild
                 + T_migrate · f_migrate
                 + T_startup · (step < N_min · P)
```

### 3.6. Boundary effects для Pattern 2

Дополнительный overhead в Pattern 2: pstall на boundary zones.

```
T_boundary_stall = P_boundary_zones · f_stall · average_stall_duration
P_boundary_zones = (surface_subdomain / volume_subdomain) ≈ 6 · (1/N_zones_per_side)
```

Для большого subdomain'а этот overhead мал (≤5%); для маленьких — может доминировать.

### 3.7. Workload saturation и minimum atoms per rank

**Критическая проверка для GPU-enabled runs.** Если система слишком маленькая, GPU не получает достаточной работы чтобы amortize launch overhead и kernel setup. TDMD в таких сценариях проигрывает простому CPU-only run.

LAMMPS и GROMACS оба документируют эту проблему. GROMACS quote: *"it is normally most efficient to use a single PP rank per GPU and for that rank to have thousands of particles"*.

#### 3.7.1. Модель saturation

Для данного потенциала и hardware существует threshold — минимальное число атомов на rank ниже которого GPU benefit исчезает:

```
N_min_saturation(potential, hw) = max(
    N_kernel_launch_overhead(hw),     # ~1000-10000 atoms для typical A100
    N_memory_bandwidth_saturation(potential, hw),
    N_neighbor_list_overhead_dominated(potential)
)
```

Где:
- `N_kernel_launch_overhead` — минимум чтобы launch overhead amortize, hardware-specific (~1024 на A100, ~2048 на H100);
- `N_memory_bandwidth_saturation` — минимум чтобы достичь >50% peak bandwidth на memory-bound kernels;
- `N_neighbor_list_overhead_dominated` — порог где cell grid build начинает dominate force compute.

#### 3.7.2. Базовые значения (tabulated, initially — calibrated М2+)

| Potential | `N_min_per_rank` A100 | `N_min_per_rank` H100 | Rationale |
|---|---|---|---|
| LJ / Morse (pair) | 10,000 | 20,000 | Memory-bound, нужна saturation |
| EAM/alloy | 5,000 | 10,000 | 2-pass, middle |
| MEAM | 2,000 | 5,000 | Compute-heavy, GPU saturates раньше |
| SNAP J=4 | 1,000 | 2,000 | Compute-bound, small N OK |
| SNAP J=8 | 500 | 1,000 | Heavy compute, small N OK |
| PACE / MLIAP | 500 | 1,000 | Similar SNAP |

Values — **calibrated empirically** в М2-М8. Эти таблицы — starting estimates, уточняются с первыми benchmarks.

#### 3.7.3. Recommendation engine

`PerfModel::recommend_deployment(plan, potential, hw)` возвращает structure:

```cpp
struct DeploymentRecommendation {
    Pattern                   suggested_pattern;
    uint32_t                  suggested_rank_count;
    uint32_t                  suggested_K;
    std::vector<std::string>  warnings;
    std::vector<std::string>  alternatives;      // if primary has caveats
    SaturationVerdict         saturation;
};

enum class SaturationVerdict {
    WellSaturated,            // N_per_rank >> N_min
    Saturated,                // N_per_rank > N_min
    UnderSaturated,           // N_per_rank ~ N_min (warn)
    SeverelyUnderSaturated    // N_per_rank < N_min (recommend alternative)
};
```

#### 3.7.4. Decision logic

```
function recommend_deployment(plan, potential, hw):
    N_total = plan.atom_count
    N_min = minimum_atoms_per_rank_table[potential][hw.gpu_class]

    # Phase 1: saturation check
    if N_total < N_min:
        return DeploymentRecommendation{
            suggested_pattern = CPU_only_or_small_Pattern3,
            warnings = ["System too small for GPU acceleration.
                         Expected speedup: <1.2x vs CPU.
                         Recommendation: use backend=cpu или very small Pattern 3."],
            alternatives = [...]
        }

    # Phase 2: Pattern selection
    if hw.rank_count == 1:
        return recommend_pattern1(plan, potential, hw)
    else:
        N_per_rank = N_total / hw.rank_count
        if N_per_rank < N_min:
            # Too many ranks для such small system:
            optimal_ranks = floor(N_total / N_min)
            return DeploymentRecommendation{
                suggested_pattern = Pattern1 or Pattern2,
                suggested_rank_count = optimal_ranks,
                warnings = ["Using fewer ranks (N={optimal_ranks}) gives better saturation.
                             Per-rank overhead dominated при {hw.rank_count} ranks."]
            }
        else:
            # Well saturated — recommend optimal pattern:
            return optimal_pattern_selection(N_total, N_per_rank, potential, hw)
```

#### 3.7.5. UX integration

`tdmd explain --perf case.yaml` показывает saturation analysis:

```
Workload saturation analysis:
  System size: 5,000 atoms
  Requested configuration: 4 ranks × 1 GPU (H100)
  Atoms per rank: 1,250

  Potential: eam/alloy
  Minimum recommended atoms/rank: 10,000 (H100, EAM)
  Actual: 1,250 (12.5% of minimum) ⚠

  Saturation verdict: SEVERELY UNDER-SATURATED

  Recommendations (sorted by expected performance):
    1. Use 1 rank × 1 GPU (all 5000 atoms on one GPU)
       Expected: 0.8 steps/ms (baseline)
       Still under-saturated: GPU at ~50% utilization

    2. Use CPU backend (8 cores):
       Expected: 0.6 steps/ms (reasonable для small system)
       Better latency, lower overhead

    3. Scale up system (if possible):
       40,000+ atoms would saturate 4 × H100 config
       Expected: ~4 steps/ms (5x current)

  Proceed with current config? Add --ignore-saturation-warnings to override.
```

#### 3.7.6. Preflight enforcement

При `tdmd run` с severely under-saturated config:
- By default: **warning + interactive confirmation** (или exit code 2 в non-interactive);
- `--ignore-saturation-warnings` flag: proceed anyway;
- `--exec-profile reference`: warnings всегда, no exit — Reference allows anything for testing.

#### 3.7.7. Monitoring runtime

Saturation verdict from preflight stored в telemetry:

```
perfmodel.saturation_verdict                # enum
perfmodel.n_atoms_per_rank_actual
perfmodel.n_atoms_per_rank_minimum_recommended
perfmodel.gpu_utilization_measured          # runtime measurement
```

`gpu_utilization_measured < 50%` для >1000 consecutive steps → warning в log: "GPU under-utilized, consider re-configuration".

#### 3.7.8. Roadmap

- **M2:** tabulated starting values для Morse + EAM;
- **M4:** `tdmd explain --perf` показывает saturation analysis;
- **M6:** runtime GPU utilization measurement (nvml sampling);
- **M8:** calibrated values для SNAP + all potentials от measurements;
- **v2+:** auto-refinement table based на telemetry history cross-run.

---

## 4. Calibration

### 4.1. Hardware profile discovery

На startup perfmodel вызывает `probe_hardware()`:

```
function probe_hardware():
    # FLOPS — из CUDA device properties + benchmark
    flops_fp64 = bench_gemm_fp64_small()  # short stamp

    # Bandwidth — micro-benchmarks
    bw_device_mem = bench_device_to_device_copy()
    bw_nvlink = bench_peer_to_peer_copy()     # if available
    bw_pcie = bench_device_to_host_copy()
    bw_mpi = bench_mpi_ring_exchange()

    # MPI latency:
    lat_mpi = bench_mpi_pingpong_minimum()

    # Topology — from NCCL topology API (if available) or MPI_Dims
    topo = query_nccl_topology()

    return HardwareProfile{...}
```

Stamp прогон — менее 1 секунды. Результат кешируется per hardware signature.

### 4.2. Potential cost calibration

`C_force_per_atom` для каждого потенциала — **measured** на первых 100 шагах warm-up при first run, сохраняется в `calibration_cache.json`. Последующие run'ы используют кеш; если кеш миссится — re-measure.

### 4.3. Calibration invariants

- **Same hardware + same potential → same C_force constant** (within 10%);
- **Scaling check:** если увеличили `N_atoms` вдвое, `T_c` должен удвоиться (linear scaling в compute);
- **Missing calibration ⟹ warning в prediction, wider confidence interval.**

---

## 5. Validation и CI gates

### 5.1. Validation harness

После каждого canonical benchmark run'а в CI:

```
predicted = perfmodel.predict(plan, potential, config, hw)
measured = telemetry.get_timing_breakdown()

report = validator.validate(predicted, measured)

if not report.passed_gate:
    fail_ci("perf model gate failed: " + report.diagnosis)
```

### 5.2. Gate thresholds по patterns

| Pattern | Tolerance | Comment |
|---|---|---|
| Pattern 1 (чистый TD) | ±20% | formulas straightforward |
| Pattern 2 (two-level) | ±25% | more parameters, more noise |
| Pattern 3 (SD) | ±15% | classical well-understood model |

Edge cases (N_atoms < 10⁴ или N_atoms > 10⁸) — tolerance удваивается.

### 5.3. Что делать при provoк

Если prediction сильно расходится с measurement **и тест stable** — есть несколько причин:

1. Calibration stale — re-run calibration;
2. Hardware profile incorrect — re-probe;
3. Bug в formulas — issue в `perfmodel/`;
4. Bug в измерении — issue в `telemetry/`;
5. Bug в реализации runtime'а — issue где-то в compute/comm stack.

Diagnosis из `PerfValidationReport::diagnosis` должна направлять в правильную из этих пяти категорий.

### 5.4. Regression gate

Отдельный CI gate: **predicted `T_step` не должно регрессировать between PRs** (для same config). Если новый PR увеличивает predicted time — требует review justification.

---

## 6. `tdmd explain --perf`

Это CLI-команда, которая преобразует `PerformancePrediction` в читаемый отчёт.

### 6.1. Default output

```
$ tdmd explain --perf case.yaml

TDMD Performance Prediction
===========================

Configuration:
  Atoms: 1,000,000 Al FCC (metal)
  Potential: EAM/alloy (Al_mishin.eam.alloy)
  Ranks: 8 (1 node × 8 GPU A100)

Recommended deployment: Pattern 1 (single-subdomain TD)
  K (pipeline depth): 4
  Expected T_step: 2.3 ms
  Expected efficiency: 87%

Alternative patterns:
  Pattern 2 (hybrid):   T_step = 2.8 ms  (slower; overhead for single node)
  Pattern 3 (SD-only):  T_step = 5.1 ms  (baseline; TD gives 2.2× speedup)

Breakdown (Pattern 1, K=4):
  Compute:          1.8 ms (78%)
  Comm (inner):     0.2 ms (8%)
  Scheduler:        0.05 ms (2%)
  Neighbor rebuild: 0.15 ms amortized (7%)
  Other overhead:   0.1 ms (5%)

Confidence: ±18% (high)

Warnings: none
```

### 6.2. Verbose output

`--verbose` добавляет:
- formula expansion с numbers (показать как получилось 1.8 ms);
- sensitivity analysis: что будет если удвоить `N_atoms`, половинить GPU, поменять потенциал;
- ссылки на секции мастер-специи;
- detailed calibration status (когда последний раз мерили).

### 6.3. Machine-readable output

`--json` даёт `PerformancePrediction` в JSON для downstream tooling:

```json
{
  "recommended_pattern": "Pattern1",
  "recommended_K": 4,
  "recommended_P_space": 1,
  "recommended_P_time": 8,
  "t_step_seconds": 0.0023,
  "t_step_pattern1_seconds": 0.0023,
  "t_step_pattern2_seconds": 0.0028,
  "t_step_pattern3_seconds": 0.0051,
  "speedup_td_over_sd": 2.22,
  "confidence_pct": 18,
  "warnings": []
}
```

---

## 7. Use cases

### 7.1. Scientist workflow

Пользователь готовит input, запускает `tdmd explain --perf case.yaml`, видит рекомендацию. Если согласен — `tdmd run case.yaml`. TDMD использует prediction для дефолтных параметров (выбор pattern, K, auto-tuning).

### 7.2. CI benchmark gate

Каждый canonical benchmark (T1-T7) имеет stored baseline prediction. PR должен:
- либо не менять prediction (for unrelated changes);
- либо явно обновить baseline с justification (для performance work).

### 7.3. Runtime auto-tuning

Если `ExecProfile == Production` или `FastExperimental`, `runtime/` может периодически (раз в N steps) re-query perfmodel с current runtime telemetry и подстраивать `K`. Это **soft recommendation**, не override user settings.

### 7.4. Hardware shopping

«У меня есть бюджет на A100 или H100, что выгоднее для моей задачи?» Perfmodel прогоняет два hypothetical `HardwareProfile`, сравнивает, даёт рекомендацию. Это расширение post-v1.

---

## 8. Tests

### 8.1. Unit tests

- Base formulas: `T_c` для каждой PotentialKind на фиксированных input;
- `T_comm_inner`, `T_halo_SD` для фиксированных topology параметров;
- `K_opt` computation: должно быть power of 2, в диапазоне [1, 64];
- `recommend()` selection algorithm на table-driven тестах.

### 8.2. Property tests

```
forall (N_atoms, potential, hw) ∈ fuzz:
    pred = perfmodel.predict(...)

    # Ordering invariants:
    if hw has slow network (bw < 10 GB/s):
        assert pred.speedup_td_over_sd >= 1.0  # TD should win

    if N_atoms < 10^4:
        assert pred.recommended_pattern != "Pattern2"  # overhead not worth it

    # Monotonicity:
    pred_larger = predict(N_atoms * 2)
    assert pred_larger.t_compute > pred.t_compute  # more atoms, more work

    # Pattern 1 never worse than Pattern 3 on dense network:
    if hw.bw_inner > hw.bw_outer:
        # always at least comparable
        assert pred.t_step_pattern1_seconds <= 1.1 · pred.t_step_pattern3_seconds
```

### 8.3. Calibration tests

- **Reproducibility:** same hardware + potential → same calibration within ±5%;
- **Scaling check:** 2× N_atoms → 2× T_c (linear, within 10%);
- **Cache coherency:** modified потенциал invalidates cache;
- **Missing calibration graceful:** warning emitted, default conservative constants used.

### 8.4. Validation tests

Это особая категория — **integration**, не unit. В CI для canonical benchmarks (T1-T7):

```
for benchmark in [T1, T2, T3, T4, T5, T6, T7]:
    prediction = perfmodel.predict(benchmark.config)
    measurement = run_benchmark(benchmark)
    report = validator.validate(prediction, measurement)
    assert report.passed_gate, f"Gate failed: {report.diagnosis}"
```

### 8.5. Anchor test

Для T3 (Al FCC 10⁶), `perfmodel.predict()` должен давать результат, близкий к числам из диссертации §3.5:

- 1 node × 4 T800 (2007 Alpha equivalent) предсказать efficiency ~95%;
- Scaling curve: performance linear in N_ranks up to n_opt.

Это validates, что формулы model'а совместимы с реальным experimental baseline из первой научной демонстрации TD.

---

## 9. Roadmap alignment

| Milestone | Perfmodel deliverable |
|---|---|
| **M2** | Skeleton `PerfModel`, predict() для Pattern 3 (чистый SD); CLI `explain --perf` basic |
| M3 | Pattern 1 predictions; K_opt; first formulas for TD |
| M4 | Calibration system; potential cost measurement; calibration cache |
| M5 | Anchor test vs Andreev numbers; full Pattern 1 validation |
| M6 | GPU HardwareProfile; NCCL/NVLink probing. **T6.11 shipped (v1.1):** GPU cost tables (`gpu_cost_tables.hpp`) + `predict_step_gpu_sec` + Reference/MixedFast factories with placeholder coefficients; ±20% calibration gate deferred to T6.11b (pending Nsight run on target GPU). |
| **M7** | Pattern 2 full support; `recommend()` для hybrid; validation gates |
| M8+ | Uncertainty quantification; ML correction layer; hardware-shopping use case |

---

## 10. Open questions (module-local)

1. **Cost constants для ML potentials (SNAP/PACE/MLIAP)** — calibration по-особому, т.к. cost зависит от complexity потенциала (числа descriptors). Нужно расширение `PotentialModel` для self-describing cost.
2. **ML correction layer** — есть ли практический смысл добавлять ML over analytic в v1.5? Вероятно да для edge cases, но нужен corpus данных.
3. **Stochastic predictions** — имеет ли смысл выдавать distribution вместо point estimate? `P(T_step < X) = 0.95`. Post-v1.
4. **User-supplied overrides** — пользователь видит рекомендацию K=4, но хочет K=8. Должен ли perfmodel принять override и пересчитать другие показатели, или требовать явный `--override-prediction` флаг?
5. **Multi-objective optimization** — иногда пользователь хочет не минимизировать time, а максимизировать atoms per dollar (throughput/cost). Post-v1.

---

## 11. Change log

| Date       | Version | Change                                                                    |
|------------|---------|---------------------------------------------------------------------------|
| 2026-04-16 | v1.0    | Initial авторство. Pattern 3 predict() skeleton (M2/T2.10). Scope: CPU cost tables, `PotentialCost`, `HardwareProfile::modern_x86_64`, `predict_step_sec` single-rank baseline. |
| 2026-04-19 | v1.1    | **T6.11 landed (M6)** — GPU cost-table infrastructure. New public header `tdmd/perfmodel/gpu_cost_tables.hpp`: `GpuKernelCost {a_sec, b_sec_per_atom}` with `predict(n_atoms) = a + b·n` linear model; `GpuCostTables` aggregate (`h2d_atom`, `nl_build`, `eam_force`, `vv_pre`, `vv_post`, `d2h_force` + `provenance` string) with `step_total_sec(n_atoms)` sum. Factory functions `gpu_cost_tables_fp64_reference()` + `gpu_cost_tables_mixed_fast()` ship **placeholder coefficients** (Ampere/Ada consumer-GPU estimates) — provenance strings tag them for replacement via T6.11b calibration harness. **New method** `PerfModel::predict_step_gpu_sec(n_atoms, tables)` divides `n_atoms` by `HardwareProfile::n_ranks` then sums `tables.step_total_sec(n_per_rank) + hw.scheduler_overhead_sec`. Scope limit: this ships the **shape**, not the **accuracy** — ±20 % gate vs measured Nsight data is deferred to T6.11b (needs profiling run on target GPU, which Option A CI cannot automate on a public repo without a self-hosted runner). When T6.11b lands, a JSON fixture will carry measured coefficients and a new test case will assert `predict_step_gpu_sec` within ±20 % of measured step wall-time. Tests: 8 new Catch2 cases in `tests/perfmodel/test_gpu_cost_tables.cpp` cover linear-model math, structural sanity bands, MixedFast ≤ Reference EAM per-atom cost invariant, and `predict_step_gpu_sec` wiring through `n_ranks`. |

---

*Конец perfmodel/SPEC.md v1.1, дата: 2026-04-19.*
