# integrator/SPEC.md

**Module:** `integrator/`
**Status:** master module spec
**Parent:** `TDMD Engineering Spec v2.1` §5.1, §6.5, §12
**Last updated:** 2026-04-16

---

## 1. Purpose и scope

### 1.1. Что делает модуль

`integrator/` двигает атомы во времени. Единственная задача: given `(positions, velocities, forces, dt)`, produce `(new_positions, new_velocities)` согласно заданному numerical scheme.

Делает три вещи:

1. **Integration schemes** — velocity-Verlet (v1), symplectic variants (post-v1);
2. **Thermostat / barostat** — NVT (Nosé-Hoover chains v1.5), NPT (v1.5), Langevin (post-v1);
3. **Timestep policy** — global `dt`, adaptive `dt` в Production profile.

### 1.2. Scope: что НЕ делает

- **не вычисляет силы** (это `potentials/`);
- **не владеет атомами** (это `state/` через `StateManager`);
- **не решает когда считать** — вызывается `scheduler/`;
- **не делает migration** (это `neighbor/`);
- **не знает про zones в смысле scheduler'а** — работает per-atom.

### 1.3. Interaction с scheduler

Integrator — simple executor. Scheduler вызывает `pre_force` → `potential.compute` → `post_force` в последовательности, определяемой zone pipeline. Integrator сам не инициирует compute.

---

## 2. Public interface

### 2.1. Базовый интерфейс

```cpp
namespace tdmd {

enum class IntegratorKind {
    VelocityVerlet,        // NVE, v1 baseline
    NoseHooverNVT,         // v1.5
    NoseHooverNPT,         // v1.5
    LangevinNVT,           // post-v1
    CustomSymplectic       // extension point
};

class Integrator {
public:
    virtual std::string     name() const = 0;
    virtual IntegratorKind  kind() const = 0;

    // Two-phase VV:
    virtual void  pre_force(StateManager&, const ZoneFilter&, double dt) = 0;
    virtual void  post_force(StateManager&, const ZoneFilter&, double dt) = 0;

    // Single-phase (для thermostats, доп. ops за iteration):
    virtual void  end_of_step(StateManager&, const ZoneFilter&, double dt) = 0;

    // Introspection для explain:
    virtual std::string  parameter_summary() const = 0;

    virtual ~Integrator() = default;
};

struct ZoneFilter {
    const ZoneId*  zone_ids;       // nullptr = все atoms
    uint32_t       n_zones;
};

} // namespace tdmd
```

### 2.2. Конкретные классы

```cpp
class VelocityVerletIntegrator final : public Integrator {
    // NVE; v1
};

class NoseHooverNvtIntegrator  final : public Integrator {
    // NVT с Nosé-Hoover chains (M_chain = 3 default); v1.5
    // target_temperature, damping_time_ps
};

class NoseHooverNptIntegrator  final : public Integrator {
    // NPT; v1.5
    // target_temperature, target_pressure, damping_time_T, damping_time_P
};

class LangevinIntegrator       final : public Integrator {
    // stochastic Langevin dynamics; post-v1
};
```

---

## 3. Velocity-Verlet (v1 reference)

### 3.1. Scheme

Standard velocity-Verlet:

```
Step 1 (pre_force): half-kick + drift
    v(t + dt/2)  =  v(t) + (f(t) / m) · dt/2
    x(t + dt)    =  x(t) + v(t + dt/2) · dt

Step 2: compute f(t + dt)    [done by potential]

Step 3 (post_force): half-kick
    v(t + dt)    =  v(t + dt/2) + (f(t + dt) / m) · dt/2

Step 4 (end_of_step): wrap positions to primary image (periodic BC)
```

### 3.2. Symplectic property

Velocity-Verlet — **symplectic** second-order integrator. Сохраняет phase space volume и approximate energy over long runs. Это critical для NVE drift behavior (master spec §5.2).

**Property:** long NVE run (10⁶ steps) должен show energy drift `|ΔE|/E < 1e-6` для FP64 Reference, `< 1e-4` для mixed precision.

### 3.3. Implementation pseudocode

```
pre_force(state, filter, dt):
    for atom_idx in state.atoms filtered by zone_filter:
        m = species.mass(atoms.type[atom_idx])
        vx_half = atoms.vx[atom_idx] + (atoms.fx[atom_idx] / m) * dt/2
        vy_half = atoms.vy[atom_idx] + (atoms.fy[atom_idx] / m) * dt/2
        vz_half = atoms.vz[atom_idx] + (atoms.fz[atom_idx] / m) * dt/2

        state.set_velocities(atom_idx, vx_half, vy_half, vz_half)

        new_x = atoms.x[atom_idx] + vx_half * dt
        new_y = atoms.y[atom_idx] + vy_half * dt
        new_z = atoms.z[atom_idx] + vz_half * dt

        state.set_positions(atom_idx, new_x, new_y, new_z)

post_force(state, filter, dt):
    for atom_idx in state.atoms filtered by zone_filter:
        m = species.mass(atoms.type[atom_idx])
        vx_new = atoms.vx[atom_idx] + (atoms.fx[atom_idx] / m) * dt/2
        vy_new = atoms.vy[atom_idx] + (atoms.fy[atom_idx] / m) * dt/2
        vz_new = atoms.vz[atom_idx] + (atoms.fz[atom_idx] / m) * dt/2

        state.set_velocities(atom_idx, vx_new, vy_new, vz_new)

end_of_step(state, filter, dt):
    # Wrap all positions (deferred periodic boundary):
    for atom_idx in state.atoms filtered by zone_filter:
        state.wrap_to_primary_image(atom_idx)
```

### 3.4. Unit correctness

Все операции в metal units:
- `m` — g/mol;
- `f` — eV/Å;
- `v` — Å/ps;
- `x` — Å;
- `dt` — ps.

Conversion factor: `f / m` имеет units `(eV/Å)/(g/mol) = 9.648... × 10^{-3} Å/ps²` (после multiplication на `N_A / (1 g/mol)`).

**В TDMD metal units это unity** благодаря правильному выбору unit definitions в `state/SPEC §5`. Никакого multiplier — просто `v += f/m * dt/2`.

### 3.5. GPU-resident execution mode (М6+)

**Критическая архитектурная политика** для GPU backend. Избегаем "traditional offload" pattern который оставляет GPU idle.

#### 3.5.1. Проблема traditional offload

Typical legacy pattern (NAMD до 3.0, LAMMPS GPU package):
```
GPU computes forces → transfer forces to CPU → CPU integrates → transfer positions back
```

Observed issue: GPU idle **30-50% времени** пока CPU выполняет integration. Data movement D2H + H2D добавляет PCIe traversal каждый step (~microseconds, но per-step × millions of steps = значительно).

NAMD 3.0 paper quote: *"GPUs are idling for a large fraction of the simulation time, since integration is a critical step and must be performed before the next launch of GPU force kernels."*

GROMACS решение: `-update gpu` flag — integrator на GPU. Это **mandatory** для modern GPU workloads.

#### 3.5.2. TDMD policy

**При `runtime.backend=cuda`, integrator kernels обязательно выполняются на GPU.** Это не optional — это требование архитектуры.

Следствия:
- `pre_force`, `post_force`, `end_of_step` — все на GPU stream;
- `atoms.x`, `atoms.y`, `atoms.z`, `atoms.vx`, `atoms.vy`, `atoms.vz`, `atoms.fx`, `atoms.fy`, `atoms.fz` — device memory, not host;
- CPU-GPU data transfer **только** при:
  - initial bootstrap (one-time);
  - periodic dumps (interval from dump.config);
  - checkpoints (explicit user request);
  - final finalize.

**Never per-step data transfer.**

#### 3.5.3. Implementation

```cpp
[[tdmd::hot_kernel]]
template<typename NumericConfig>
__global__ void velocity_verlet_pre_force_kernel_gpu(
    typename NumericConfig::StateReal* __restrict__ x,
    typename NumericConfig::StateReal* __restrict__ y,
    typename NumericConfig::StateReal* __restrict__ z,
    typename NumericConfig::StateReal* __restrict__ vx,
    typename NumericConfig::StateReal* __restrict__ vy,
    typename NumericConfig::StateReal* __restrict__ vz,
    const typename NumericConfig::AccumReal* __restrict__ fx,
    const typename NumericConfig::AccumReal* __restrict__ fy,
    const typename NumericConfig::AccumReal* __restrict__ fz,
    const double* __restrict__ mass,
    const SpeciesId* __restrict__ type,
    double dt_half, int n_atoms)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;

    double m = mass[type[i]];
    double inv_m = 1.0 / m;

    // Half-kick:
    vx[i] += fx[i] * inv_m * dt_half;
    vy[i] += fy[i] * inv_m * dt_half;
    vz[i] += fz[i] * inv_m * dt_half;

    // Drift:
    x[i] += vx[i] * 2.0 * dt_half;    // 2*dt_half == dt
    y[i] += vy[i] * 2.0 * dt_half;
    z[i] += vz[i] * 2.0 * dt_half;
}
```

Kernel runs на `stream_compute` (см. master spec §9.2), coordinate with force kernels через CUDA events.

#### 3.5.4. Validation: GPU idle time

Performance gate в VerifyLab T6 (SNAP benchmark):

```
Required: gpu_idle_fraction < 0.10   # <10% idle время
```

Measured через NVTX ranges + `nvidia-smi` sampling. If >10% idle — likely CPU-GPU ping-pong bug, regression alert.

`telemetry.gpu_idle_fraction_avg` — mandatory metric в GPU builds.

#### 3.5.5. Consequences для other modules

- `state/SPEC` §7 GPU sync — `sync_to_device` и `sync_from_device` называются **только** при bootstrap/dump/checkpoint, не per-step;
- `neighbor/SPEC` §6 — neighbor rebuild kernel тоже на GPU (нельзя rebuild на CPU и sync);
- `potentials/SPEC` §9 — force kernels на GPU (уже было);
- `scheduler/SPEC` §9 — scheduler orchestration CPU, но issues GPU commands, не reads data;
- `comm/SPEC` §6.4 — `HybridBackend` обязательно GPU-aware для inter-rank transfers.

**Один принцип: data lives on GPU; CPU orchestrates but never touches hot data per-step.**

#### 3.5.6. CPU backend consistency

При `runtime.backend=cpu`, integrator на CPU — очевидно. Но same kernel signature (template over NumericConfig, same `__restrict__` pattern) — это единый код, просто compiled для CPU threads вместо CUDA. Simplifies testing: CPU run + GPU run должны give same results.

#### 3.5.7. Roadmap

- **M1-M5:** CPU integrator only, `backend=cpu`;
- **M6:** GPU-resident mode introduced, `backend=cuda` obligatory puts integrator on GPU;
- **M7+:** Pattern 2 extends — each subdomain runs GPU-resident independently;
- **v2+:** heterogeneous CPU+GPU hybrid (CPU для dumps/output, GPU для compute) — scope decision post-v1.

---

## 4. NVT thermostat (v1.5 — M9+)

### 4.1. Nosé-Hoover chains

Для canonical ensemble (constant T), используется Nosé-Hoover thermostat с chains of M = 3 (default):

```
Extended Hamiltonian:
    H = H_physical + sum_{k=1..M} [p_ξk² / (2 Q_k) + N_f · k_B · T · ξ_k ]

Where:
    ξ_k — thermostat variable
    p_ξk — its momentum
    Q_k — thermostat mass
    N_f — number of degrees of freedom (3N for unconstrained)
```

### 4.2. Integration scheme

Trotter decomposition за step:

```
1. Update thermostat chain (Yoshida-Suzuki 3rd-order, dt/2)
2. Rescale velocities by exp(-ξ_1 · dt/2)
3. Velocity-Verlet pre_force
4. compute forces
5. Velocity-Verlet post_force
6. Rescale velocities by exp(-ξ_1 · dt/2)
7. Update thermostat chain (dt/2)
```

### 4.3. Parameters

```yaml
integrator:
  style: nvt
  temperature: 300.0         # K
  damping_time: 0.1          # ps (τ_T)
  chain_length: 3            # M
```

`Q_k = N_f · k_B · T · τ_T²` (standard choice).

### 4.4. Validation

- **Equipartition:** ⟨KE⟩ = (3/2) N k_B T within `±2σ` over 10⁵ steps.
- **Temperature distribution:** Gaussian histogram centered at target T.
- **Canonical ensemble test:** радиальная функция распределения на matched LJ liquid match LAMMPS NVT within noise.

### 4.5. Coupling intervals (performance critical)

**Неочевидная производительность-проблема.** GROMACS documentation warns: *"Frequent temperature or pressure coupling can have significant overhead; to reduce this, make sure to have as infrequent coupling as your algorithms allow (typically >=50-100 steps)"*.

Причина: thermostat chain update требует **global reduction** (total kinetic energy over all atoms). На multi-rank systems это **MPI Allreduce каждый шаг** = ~10-100 μs на large clusters. За 10⁶ steps = ~10-100 seconds overhead только на thermostat sync.

#### 4.5.1. Политика TDMD

Thermostat coupling делается **раз в N steps**, not each step:

```yaml
integrator:
  style: nvt
  temperature: 300.0
  damping_time: 0.1
  thermostat_update_interval: 50      # default
```

**Effect:**
- Каждые 50 шагов: full Nosé-Hoover chain update (includes global KE reduction);
- Между updates: NVE propagation с frozen thermostat variables (`ξ_k` constant);
- **Canonical ensemble preserved** — doc proven для wide range of coupling intervals [1, 1000 steps].

#### 4.5.2. Trade-off

| `thermostat_update_interval` | Effect |
|---|---|
| 1 | Max accuracy, significant overhead (10-30% на large systems) |
| 10 | Small accuracy cost, ~3-5% overhead |
| **50 (default)** | Negligible accuracy cost, ~1% overhead |
| 100 | Still OK for most systems, <1% overhead |
| 500 | May affect temperature stability в coupled systems |
| 1000+ | Risk of temperature drift между updates |

Default `50` — conservative choice, safe для всех common workloads.

#### 4.5.3. Validation constraint

При изменении `thermostat_update_interval`, validation:

```
constraint: damping_time / (dt * thermostat_update_interval) > 2.0
```

Т.е. update interval должен быть **much shorter** чем damping time для stability. Violation → preflight error:

```
Config error: thermostat_update_interval too coarse.
  damping_time = 0.1 ps
  dt = 0.001 ps
  thermostat_update_interval = 500
  Characteristic coupling period: 500 * 0.001 = 0.5 ps (5x damping_time)
  Ratio damping_time / coupling_period = 0.2 (requires > 2.0)

Suggestion: decrease interval (<50) or increase damping_time (>0.25 ps).
```

#### 4.5.4. Pressure coupling (NPT)

Same policy:

```yaml
integrator:
  style: npt
  temperature: 300.0
  pressure: 1.0
  temperature_damping: 0.1
  pressure_damping: 1.0
  thermostat_update_interval: 50
  barostat_update_interval: 100      # longer — pressure slower
```

Default `barostat_update_interval: 100` — pressure response intrinsically slower than temperature.

---

## 5. NPT barostat (v1.5 — M9+)

### 5.1. Parrinello-Rahman-like с Nosé-Hoover

Volume flexes along orthogonal axes (isotropic в v1; anisotropic в v2+).

### 5.2. Parameters

```yaml
integrator:
  style: npt
  temperature: 300.0
  pressure: 1.0              # bar (metal units: bar)
  temperature_damping: 0.1   # ps
  pressure_damping: 1.0      # ps (longer — important для stability)
  chain_length: 3
```

### 5.3. Implementation

Требует box flexibility — `StateManager::set_box(new_box)`. Это invalidates zoning plan (zone sizes change), triggers neighbor rebuild. Expensive но infrequent.

**v1.5 decision:** NPT только изотропный (single `isoenthalpic scaling factor`). Anisotropic (stress tensor) — v2+.

---

## 6. Timestep policy

### 6.1. Global vs local

**v1 принципиальное решение:** только **global `dt`** для всех atoms. Нет per-zone dt.

Причина: per-zone dt создаёт задачу re-synchronization scales, эквивалентную задаче Parareal, и выходит за scope TDMD v1 (master spec §6.5).

### 6.2. Fixed dt (Reference)

В Reference profile: `dt` fixed на весь run. Задаётся в config:

```yaml
integrator:
  dt: 0.001                  # ps = 1 fs
```

### 6.3. Adaptive dt (Production)

В Production profile: `dt` adapts based на global `v_max`:

```
dt_candidate = min(dt_cap,
                   α_safety · buffer_width / max(v_max_global, ε))

dt_new = min(dt_candidate, 1.05 · dt_current)   # slow increase
       = dt_candidate                             # fast decrease (if needed)
```

Where:
- `α_safety ∈ [0.3, 0.7]`, default 0.5;
- `dt_cap` — maximum dt (physics limit, не вычисления), default 0.005 ps = 5 fs;
- update раз в R шагов, default R = 50.

### 6.4. Synchronization

Изменение `dt` — global event. Все ranks должны согласованно знать new `dt` прежде чем integrator step начнётся. Это coordinated через `scheduler/` certificate refresh.

В Pattern 2 (M7+): все subdomains согласовывают `dt` через `comm.global_max_double(1.0 / dt_candidate)` — берётся наиболее conservative.

### 6.5. Validation

- **Fixed dt:** reproducibility test — two runs same dt → same trajectory bitwise.
- **Adaptive dt:** NVE drift с adaptive dt < drift с fixed dt (при правильной реализации adaptive should not degrade conservation).

---

## 7. TD integration pattern

### 7.1. Per-zone execution

Scheduler вызывает integrator per-zone, not per-all-atoms:

```
# In scheduler iteration:
for zone in scheduler.ready_zones():
    zone_filter = ZoneFilter{ zone.atoms_begin, zone.atoms_end }

    integrator.pre_force(state, zone_filter, dt)
    potential.compute(request(zone), result)
    integrator.post_force(state, zone_filter, dt)
    integrator.end_of_step(state, zone_filter, dt)

    scheduler.mark_completed(zone)
```

Integrator **не знает**, что обрабатывается только одна зона — просто получает filter. Same code path для single-zone TD и all-atoms SD-vacuum.

### 7.2. Consistency при K > 1

Если `K > 1` (K-batched pipeline), integrator вызывается **K раз последовательно для одной зоны** прежде чем packet отправляется peer'у. State version bumps на каждом call — certificate validation между invocations корректно triggered.

### 7.3. Thermostat global state в TD

NVT/NPT имеют **global state** — thermostat variables `ξ_k` и их momenta `p_ξk`. В стандартном SD каждый integrator step делает два thermostat half-updates (Trotter decomposition, §4.2), и global state обновляется synchronously со всеми atoms.

В TD, где zones процессируются independently at разных `time_level`, возникает фундаментальная проблема: **какое значение thermostat state видит зона, находящаяся на time_level `h+5`, в то время как другая зона на `h+3`?**

Три концептуальных варианта решения:

#### 7.3.1. Вариант A — Global frozen state (v1.5 policy)

Thermostat update делается **один раз на iteration** в scheduler'е, после всех zone commits. Все zones в current iteration видят **одно и то же** `ξ_k` value.

Consequence: **требуется `K = 1`** для NVT/NPT, чтобы все zones в iteration были на одном time_level. K > 1 невозможен с этим подходом.

**Это v1.5 default solution.** Strict enforcement в `PolicyValidator::check`: если `integrator.style != nve` и `pipeline_depth_cap > 1`, → reject config с clear error:

```
Config error: NVT/NPT integrator requires K=1 (pipeline_depth_cap=1).
Current config has pipeline_depth_cap=4, style=nvt.

Rationale: Nosé-Hoover chains require global synchronization at each step.
K>1 batched pipeline would use inconsistent thermostat state across zones.

Options:
  - Set pipeline_depth_cap: 1 (slower, but correct NVT)
  - Use style: nve (NVE with K>1 fully supported)
  - Wait for v2.0 NVT-in-TD research (see integrator/SPEC §7.3)
```

**Consequence для TDMD value proposition:** NVT/NPT workloads **не получают full TD speedup** в v1.5. Это известный trade-off — scientific correctness первична.

Для users это означает:
- NVE production в TD: full benefit из K-batched pipeline (M5+);
- NVT production: работает, но as SD effectively (K=1, no time decomposition benefit);
- NVT + TD — v2+ research target.

#### 7.3.2. Вариант B — Per-zone thermostat (rejected в v1.5)

Каждая zone имеет свой local `ξ_k`. Zones эволюционируют независимо.

**Проблема:** это **не эквивалентно** стандартному NVT ensemble. Разные zones могут drift в разные thermodynamic states. Unclear как это relates к canonical ensemble statistics.

**Decision:** отвергнуто для v1.5. Может быть revisited в v2+ после research.

#### 7.3.3. Вариант C — Lazy thermostat synchronization (v2+ research target)

Zone продвигается на `K` local steps **с snapshot'ом** `ξ_k` из момента старта batch'а. В конце K-шагов global thermostat update делается atomically, с accumulated kinetic energy от всех zones.

**Теоретическая основа:** similar к k-step Verlet + thermostat correction. Несколько papers (Tuckerman 2010, Eastwood 2010) обсуждают RESPA-like multi-timescale thermostat techniques которые потенциально адаптируемы.

**Research requirements перед implementation:**

1. **Формальный proof correctness** — что получаемый ensemble статистически эквивалентен canonical (Maxwell-Boltzmann distribution, equipartition);
2. **Reference paper identification** — один из Tuckerman-like approaches, но не все adaptable к TD context;
3. **Numerical validation** — comparison с LAMMPS NVT на canonical Al FCC тесте, observables match within statistical envelope;
4. **Scaling study** — даёт ли это meaningful speedup? (тонкость: possibly lazy thermostat создаёт extra synchronization overhead который nullifies TD benefit).

#### 7.3.4. Research roadmap

**M9 (v1.5 delivery):** Вариант A implemented. NVT/NPT работают correctly но без TD speedup. Добавляет policy validation блокирующую invalid configs. Добавляет CI test проверяющий что NVT не принимает K>1.

**M11-M12 (post-v1, research window):** Validation Engineer + Physics Engineer исследуют Вариант C. Deliverables:
- Literature survey (Tuckerman, Eastwood, других);
- Prototype implementation на отдельной branch;
- Comparison suite: LAMMPS NVT baseline vs TDMD lazy-thermostat on canonical Al FCC;
- Go/no-go decision based на наличии speedup >10% при equipartition match;
- If go: full integration в mainline integrator.

**M13+ (possibly v2.0):** Если Вариант C успешен — production integration. Если нет — Вариант A остаётся, documented как fundamental TD limitation для NVT.

**Это explicit research program, не "TBD".** Owner: Physics Engineer + Validation Engineer. Budget: ~8-12 weeks estimated.

Documented здесь чтобы users знали что именно блокирует NVT-in-TD, и чтобы research не был dropped if project накапливает immediate product pressure.

---

## 8. Numeric precision

> Источник истины — мастер-спец Приложение D. Здесь — integrator-specific применение.

### 8.1. BuildFlavor-aware

```cpp
template<typename NumericConfig>
class VelocityVerletImpl : public Integrator {
    using StateReal = typename NumericConfig::StateReal;
    using ForceReal = typename NumericConfig::ForceReal;
    using AccumReal = typename NumericConfig::AccumReal;
    // ...
};
```

Positions и velocities stored в `StateReal` (always double в v1).

Velocity update `v += (f/m) * dt/2`:
- `f` читается как `ForceReal` (может быть float);
- `m` — double;
- intermediate `(f/m) * dt/2` — computed в `AccumReal` (double в Philosophy B);
- `v += ...` — double += double.

### 8.2. Energy conservation precision

NVE drift main-metric для integrator correctness (thresholds из `verify/thresholds.yaml`):
- **Fp64Reference**: drift `< 1e-8 per ns`;
- **Fp64Production**: drift `< 1e-6 per ns`;
- **MixedFast** (Philosophy B): drift `< 1e-4 per ns`;
- **MixedFastAggressive** (Philosophy A): **gate disabled** — expected behaviour;
- **Fp32Experimental**: gate disabled.

### 8.3. Pointer aliasing

Integrator kernels (pre_force, post_force) — hot kernels, следуют §D.16 мастер-специи. Canonical signature:

```cpp
[[tdmd::hot_kernel]]
template<typename NumericConfig>
__device__ void velocity_verlet_pre_force_kernel(
    typename NumericConfig::StateReal* __restrict__ x,
    typename NumericConfig::StateReal* __restrict__ y,
    typename NumericConfig::StateReal* __restrict__ z,
    typename NumericConfig::StateReal* __restrict__ vx,
    typename NumericConfig::StateReal* __restrict__ vy,
    typename NumericConfig::StateReal* __restrict__ vz,
    const typename NumericConfig::AccumReal* __restrict__ fx,
    const typename NumericConfig::AccumReal* __restrict__ fy,
    const typename NumericConfig::AccumReal* __restrict__ fz,
    const double* __restrict__ mass,
    const SpeciesId* __restrict__ type,
    double dt, int n_atoms);
```

Все pointer'ы mark'нутые `__restrict__` — независимые arrays в `AtomSoA`, никогда не алиасятся. Expected integrator speedup: 10-15% (§D.16.2).

---

## 9. Tests

### 9.1. Unit tests

- **Single-atom free particle:** no forces, const v, x advances linearly. Exact match.
- **Single-atom harmonic:** analytic solution known; integrator error scales as `O(dt²)` — verify.
- **Symmetry:** start at `(x, v)`, run N steps forward, reverse velocity, run N steps back → return to `(x, -v)` within numerical precision.

### 9.2. NVE conservation tests

- Al FCC 256 atoms, Morse, `dt = 1 fs`, 10⁴ steps:
  - FP64 Reference: `|ΔE|/E < 1e-8`;
  - FP64 Production: `|ΔE|/E < 1e-6`;
  - Mixed: `|ΔE|/E < 1e-4`.

### 9.3. Thermostat validation

- NVT Al FCC 512 atoms, T = 300 K, 10⁵ steps:
  - ⟨T⟩ = 300 ± 3 K;
  - ⟨KE⟩ = (3/2) · 512 · k_B · 300 within `±2σ`;
  - velocity distribution — Maxwell-Boltzmann within chi-square test.

### 9.4. Differential vs LAMMPS

- **VelocityVerlet:** 100 steps Al FCC, compare positions / velocities at end — match to `1e-11` in FP64.
- **NVT:** long run; observables match LAMMPS within statistical noise.
- **NPT:** box evolution match LAMMPS within `2%` relative (slower convergence expected).

### 9.5. Determinism tests

- Same integrator, same state, same dt → byte-identical evolution.
- Integration fused with state version bumps: each call bumps version exactly once.

---

## 10. Telemetry

Metrics:
```
integrator.steps_total
integrator.current_dt
integrator.dt_adaptive_changes_total
integrator.avg_temperature
integrator.nve_drift_cumulative
integrator.thermostat_energy             (NVT/NPT)
integrator.barostat_volume_change        (NPT)
```

NVTX ranges:
- `integrator::pre_force`;
- `integrator::post_force`;
- `integrator::end_of_step`.

---

## 11. Roadmap alignment

| Milestone | Integrator deliverable |
|---|---|
| M1 | `VelocityVerletIntegrator` CPU; fixed dt; NVE |
| M2 | TD scheduler integration (per-zone calls) |
| M6 | GPU kernels для VV (half-kick, drift kernels) |
| M7 | Adaptive dt in Production profile |
| **M9** | `NoseHooverNvtIntegrator` CPU + GPU |
| **M10** | `NoseHooverNptIntegrator` |
| v2+ | `LangevinIntegrator`, anisotropic NPT, per-zone dt research |

---

## 12. Open questions

1. **SHAKE/RATTLE constraints** — rigid bonds (water models etc). Post-v1, если biomolecules становятся scope.
2. **Rigid body dynamics** — для polymers, granular simulations. Post-v1.
3. **Multiple-timestep integrators (rRESPA)** — для interfacing с long-range service. v2+.
4. **Stochastic integrators** (Langevin) — NVT alternative, simpler than Nosé-Hoover but adds noise. Post-v1.
5. **Per-zone adaptive dt** — огромное архитектурное расширение. Strictly post-v1.

---

*Конец integrator/SPEC.md v1.0, дата: 2026-04-16.*
