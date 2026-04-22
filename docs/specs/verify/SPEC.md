# verify/SPEC.md

**Module:** `verify/`
**Status:** master module spec
**Parent:** `TDMD Engineering Spec v2.1` §13 (testing), §3.2 (positioning)
**Version:** v1.1 (M9 T9.1 SPEC delta: §4 extended to T0–T9; §4.8 T8 NVT + §4.9 T9 NPT authored; §14 roadmap extended through M12; §16 change log introduced)
**Last updated:** 2026-04-22 (M9 T9.1 SPEC delta — T8 NVT + T9 NPT canonical benchmarks registered)

---

## 1. Purpose и scope

### 1.1. Что делает модуль

`verify/` — **cross-module scientific validation layer** TDMD. Единственная зона ответственности всех тестов, которые касаются физической/математической корректности и пересекают границы модулей:

1. **Canonical benchmarks** (T0-T7) как runnable tests;
2. **Threshold registry** — единое хранилище всех допусков проекта;
3. **Differential harness** — side-by-side сравнение TDMD с LAMMPS / GROMACS / reference implementations;
4. **Physics invariant tests** — законы сохранения, термодинамические инварианты, detailed balance;
5. **Anchor-test framework** — воспроизведение эксперимента диссертации Андреева;
6. **Reference data management** — golden trajectories, expected outputs, параметрические reference tables;
7. **Regression baseline versioning** — snapshot'ы которые сравниваются при каждом PR;
8. **Acceptance report generator** — diagnostic reports для CI и scientific reviewers.

### 1.2. Scope: что НЕ делает

Жёсткие границы — иначе VerifyLab становится dumping ground:

- **не содержит unit tests модулей** — те остаются в `tests/<module>/` per-module;
- **не содержит property fuzz tests** — остаются в module-owned tests (`scheduler/SPEC §12.2`, и т.д.);
- **не делает performance measurement** — это `perfmodel/` job; verify *использует* perfmodel для validation Pattern 2, но не меряет сам;
- **не владеет CI infrastructure** — использует существующие pipelines из master spec §11, не дублирует;
- **не меняет runtime** — VerifyLab запускает SimulationEngine через public API как клиент, не intrusive.

### 1.3. Философия: scientific credibility

Master spec §3 fire product positioning: TDMD — «high-performance engine для EAM/MEAM/SNAP». Для academic users «correctness» — не техническая метрика, а **gate acceptance**. Размытая валидация → размытое доверие → проект не используется.

VerifyLab — **ответ TDMD на вопрос «а как вы доказываете что оно работает?»**. Не должен быть скрытым; должен быть публичным, документированным, воспроизводимым.

### 1.4. Role ownership

Из `claude_code_playbook.md` §2.7: **Validation / Reference Engineer**. Все PR в `verify/` требуют review from this role; threshold changes требуют дополнительного review Architect / Spec Steward.

---

## 2. Architecture

### 2.1. Место в системе

```
┌─────────────────────────────────────┐
│   SimulationEngine (runtime)        │
│   ├─ state/                          │
│   ├─ scheduler/                      │
│   ├─ potentials/                     │
│   ├─ integrator/                     │
│   └─ ...                             │
└─────────────────────────────────────┘
          │
          │ public API
          │ (run, compare, observe)
          │
          ▼
┌─────────────────────────────────────┐
│   VerifyLab (verify/)                │
│                                      │
│   inputs:   benchmark configs        │
│             threshold registry       │
│             reference data           │
│   outputs:  pass/fail verdicts       │
│             diagnostic reports       │
│             acceptance certificates  │
└─────────────────────────────────────┘
          │
          │
          ▼
     ┌────────────┐        ┌────────────────┐
     │  CI gates  │        │ Scientific     │
     │  (PR / CD) │        │ reviewer tools │
     └────────────┘        └────────────────┘
```

VerifyLab — **pure consumer** SimulationEngine'а. Она ничего не мутирует в runtime. Это позволяет полностью отключать VerifyLab в production runs без влияния на симуляцию.

### 2.2. Внутренняя структура

```
verify/
├── thresholds/                 # single source of all tolerances
│   └── thresholds.yaml
│
├── benchmarks/                 # canonical T0-T7 benchmarks
│   ├── t0_morse_analytic/
│   ├── t1_al_fcc_small/
│   ├── t2_al_fcc_medium/
│   ├── t3_al_fcc_large_anchor/    # Andreev reproduction
│   ├── t4_nial_alloy/
│   ├── t5_si_meam/
│   ├── t6_w_snap/
│   └── t7_mixed_scaling/
│
├── reference/                  # golden data
│   ├── trajectories/            # reference .h5 files
│   ├── lammps_scripts/          # LAMMPS input for each benchmark
│   ├── analytic/                # closed-form reference (for T0)
│   └── baselines/               # current stored baselines, versioned
│
├── harness/                    # executable validation framework
│   ├── differential_runner/     # runs TDMD + LAMMPS, compares
│   ├── conservation_checker/    # NVE drift, momentum, ...
│   ├── observables_comparator/  # T, P, MSD, RDF, g(r)
│   ├── anchor_test_runner/      # Andreev §3.5 replication
│   └── perfmodel_validator/     # integrates with perfmodel/
│
├── third_party/
│   └── lammps/                  # git submodule (see §5)
│
├── reports/                    # generated acceptance reports
│   └── <run_id>/
│
└── tiers/                      # tier configurations
    ├── fast.yaml                # <5 min, runs per PR
    ├── medium.yaml              # ~30 min, runs nightly
    └── slow.yaml                # ~8 hours, runs per release
```

### 2.3. Тестовые слои (из master spec §13.1)

| Layer | Owner | VerifyLab? |
|---|---|---|
| 1. Unit tests | module | no |
| 2. Property tests | module | no |
| 3. Differential tests (vs LAMMPS) | **VerifyLab** | **yes** |
| 4. Determinism tests | **VerifyLab** | **yes** (cross-module) |
| 5. Performance tests | perfmodel + VerifyLab | partially (validator) |
| 6. Perfmodel validation | **VerifyLab** | **yes** |

VerifyLab purely handles layers 3, 4, 6. Layer 5 — delegated to `perfmodel/`, but VerifyLab integrates results.

---

## 3. Threshold registry

### 3.1. Единый источник истины

Все численные допуски проекта живут в **одном файле**: `verify/thresholds/thresholds.yaml`. Никакие hardcoded magic numbers в остальном коде — нарушение = merge reject.

### 3.2. Format

```yaml
# verify/thresholds/thresholds.yaml
#
# Единое хранилище всех числовых допусков TDMD.
# Изменения ТОЛЬКО через отдельный PR с rationale и Architect review.
# См. verify/SPEC.md §3.

schema_version: 1
last_updated: 2026-04-16

tolerances:

  # ====================
  # Force correctness
  # ====================
  forces:
    # TDMD vs analytical formula (2-atom, harmonic):
    reference_fp64_vs_analytic:
      absolute: 1.0e-12       # |Δf|
      relative: 1.0e-12       # |Δf| / |f|
      rationale: "FP64 arithmetic precision floor"

    # TDMD vs LAMMPS (5 BuildFlavor'ов, см. мастер-спец §D.13):
    reference_fp64_vs_lammps:
      relative: 1.0e-10
      rationale: "Accounts for different reduction orders in FP64"
    production_fp64_vs_lammps:
      relative: 1.0e-10
    mixed_fast_vs_fp64_reference:
      relative: 1.0e-5
      rationale: "Philosophy B: float compute, double accumulate, safe mixed"
    mixed_fast_aggressive_vs_fp64_reference:
      relative: 1.0e-4
      rationale: "Philosophy A: float throughout, opt-in research"
    fp32_vs_fp64_reference:
      relative: 1.0e-3
      rationale: "FP32 experimental build; research only"

  # ====================
  # Energy (potential + kinetic)
  # ====================
  energy:
    reference_fp64_vs_lammps:
      relative: 1.0e-10
    mixed_vs_fp64_reference:
      relative: 1.0e-6
    # NVE drift over long run (см. §D.13 мастер-специи):
    nve_drift_per_ns:
      reference_fp64:             1.0e-8
      production_fp64:            1.0e-6
      mixed_fast:                 1.0e-4     # Philosophy B
      mixed_fast_aggressive:      gate_disabled   # Philosophy A — explicit opt-out
      fp32_experimental:          gate_disabled
      rationale: "Symplectic Verlet scheme typical drift; Philosophy A gates disabled by design"

  # ====================
  # Virial / pressure
  # ====================
  virial:
    reference_fp64_vs_lammps:
      relative: 1.0e-9
    # Statistical consistency:
    pressure_mean_deviation_sigma: 2.5    # 2.5σ threshold on mean

  # ====================
  # Observables (statistical)
  # ====================
  observables:
    temperature_mean_sigma:           2.0
    temperature_distribution_chi2:    0.05    # p-value threshold
    msd_slope_relative:               0.05    # ±5% for diffusion
    rdf_peak_position_relative:       0.02    # ±2% for structure

  # ====================
  # Conservation laws
  # ====================
  conservation:
    total_momentum_magnitude:         1.0e-10   # should be zero
    angular_momentum_drift_nvt:       1.0e-6
    # Energy drift (NVE) — see energy.nve_drift

  # ====================
  # Performance model (perfmodel/ validation)
  # ====================
  perfmodel:
    pattern1_prediction_error:        0.20
    pattern2_prediction_error:        0.25
    pattern3_prediction_error:        0.15
    edge_case_multiplier:             2.0     # double tolerance for N<10^4 or N>10^8

  # ====================
  # Anchor test (reproduce Andreev §3.5)
  # ====================
  anchor_test:
    efficiency_deviation:             0.10
    scaling_linear_r2:                0.95    # linearity R² threshold

  # ====================
  # Determinism
  # ====================
  determinism:
    bitwise_reference_same_hardware:  exact      # byte-for-byte
    bitwise_restart_resume:            exact
    layout_invariant_reference:        exact      # target (stretch for M7)
    scientific_reproducibility:        as per observables.*

  # ====================
  # Numerical stability
  # ====================
  stability:
    max_force_magnitude:              1.0e6   # eV/Å; abort if exceeded (runaway)
    min_bond_length:                  0.5     # Å; warn if atoms overlap
    max_velocity_magnitude:           1000.0  # Å/ps; abort if exceeded

  # ====================
  # Cutoff treatment (см. potentials/SPEC §2.4)
  # ====================
  cutoff_treatment:
    shifted_force_energy_continuity:        1.0e-12
      # |E(r_c - ε) - E(r_c + ε)| должно быть near machine epsilon
    shifted_energy_force_jump_acceptable:   1.0e-3
      # Strategy B: относительный jump force при r = r_c
    smoothed_derivative_continuity:         1.0e-10
      # Strategy D: производная smoothing function continuous
    morse_hard_cutoff_reference_only:        true
      # Strategy A запрещена в production, только для unit tests

  # ====================
  # Auto-K policy (см. master spec §6.5a)
  # ====================
  auto_k:
    convergence_cycles_max:                 5
      # Maximum retune cycles до convergence на synthetic workload
    oscillation_hysteresis:                 0.05
      # 5% threshold для hysteresis (prevent K ping-pong)
    measurement_overhead_max:               0.05
      # Auto-K measurement должна тратить <5% overall time

  # ====================
  # FMA / Toolchain binding (см. master spec §D.10, §7.3)
  # ====================
  toolchain_binding:
    bitwise_same_toolchain:            exact
      # Same compiler version + CUDA version + arch → bitwise identity
    cross_compiler_envelope:           layout_invariant
      # Different compiler → Level 2 guarantee only
    environment_fingerprint_required:   true
      # repro bundle must записывать: compiler, CUDA ver, arch, BLAS, HW class
```

### 3.3. Access API

```cpp
namespace tdmd::verify {

class ThresholdRegistry {
public:
    static ThresholdRegistry& instance();

    double  get(const std::string& path) const;
    // e.g. get("forces.reference_fp64_vs_lammps.relative") → 1e-10

    bool  check(const std::string& path, double measured, double reference) const;

    // For structured lookups:
    struct ForceTolerance {
        double  absolute;
        double  relative;
        std::string rationale;
    };
    ForceTolerance get_force_tolerance(
        const std::string& build_flavor,
        const std::string& reference_kind) const;
};

} // namespace tdmd::verify
```

### 3.4. Change policy

Threshold change процедура (из playbook §9.1 adapted):

1. PR затрагивает **только** `thresholds.yaml` + docs с rationale;
2. Commit message должен включать:
   - что меняется;
   - почему (physical justification или measurement evidence);
   - impacted benchmarks (list);
3. Required reviewers: **Validation / Reference Engineer** + **Architect / Spec Steward**;
4. CI runs **full slow tier** проверить что изменение консистентно;
5. Merge только после both approvals.

**Никаких silent tolerance loosening**. Если test провалился — либо починить код, либо явно оправдать threshold change.

---

## 4. Benchmarks (T0-T9)

### 4.1. Обзор

Из master spec §13.2 + M9 T9.1 SPEC delta:

| Tier | Name | Description | Primary purpose |
|---|---|---|---|
| T0 | `morse-analytic` | 2 atoms, аналитика | unit sanity |
| T1 | `al-fcc-small` | Al 64-512, Morse | correctness + determinism |
| T2 | `al-fcc-medium` | Al 10⁴-10⁵, Morse | NVE drift, repro |
| T3 | `al-fcc-large` | Al 10⁶, Morse | **TD anchor-test (Andreev §3.5)** |
| T4 | `nial-alloy` | Ni/Al EAM | EAM correctness vs LAMMPS |
| T5 | `meam-angular` | Si MEAM | many-body TD target |
| T6 | `snap-tungsten` | W SNAP (`W_2940_2017_2.snap`, Wood+Thompson 2017, 128-atom BCC) | **ML niche proof-of-value** — M8 artifact gate |
| T7 | `mixed-scaling` | T4 + T6 parallel | multi-GPU TD×SD |
| **T8** | `al-fcc-nvt` | Al FCC 512, EAM/alloy, Nosé-Hoover NVT | **NVT canonical** — M9 artifact gate (equipartition + Maxwell-Boltzmann) |
| **T9** | `al-fcc-npt` | Al FCC 512, EAM/alloy, Nosé-Hoover NPT isotropic | **NPT canonical** — M9 artifact gate (⟨V⟩ + fluctuations vs LAMMPS) |

> **Registry headroom:** T10+ reserved for M12 (PACE/MLIAP) and future ML
> potential additions. Current active range is T0–T9; any new benchmark
> follows §4.10 *Adding new benchmark* procedure.

### 4.2. Структура одного benchmark

Each benchmark — self-contained directory:

```
verify/benchmarks/t4_nial_alloy/
├── README.md               # описание, purpose, expected behavior
├── config.yaml             # TDMD config
├── Ni_Al.data              # LAMMPS data file (init state)
├── NiAl.eam.alloy          # potential parameters
├── lammps_script.in        # equivalent LAMMPS input
├── reference_output.h5     # golden trajectory (5000 steps)
├── reference_observables.json  # T, P, E averages
├── checks.yaml             # what to validate
├── tier_assignment.yaml    # fast/medium/slow
└── metadata.yaml           # version, creation date, source
```

### 4.3. `checks.yaml` — что проверяется

```yaml
# verify/benchmarks/t4_nial_alloy/checks.yaml
checks:
  - name: run0_forces
    type: differential
    reference: lammps
    metric: forces
    threshold_path: forces.reference_fp64_vs_lammps
    required: true

  - name: nve_drift_10k_steps
    type: conservation
    metric: energy_drift
    threshold_path: energy.nve_drift_per_ns
    steps: 10000
    required: true

  - name: temperature_distribution
    type: observable
    metric: temperature_histogram
    threshold_path: observables.temperature_distribution_chi2
    steps: 50000
    warmup: 10000
    required: false   # medium tier only

  - name: msd_diffusion
    type: observable
    metric: msd_slope
    threshold_path: observables.msd_slope_relative
    steps: 100000
    warmup: 50000
    required: false   # slow tier only
```

### 4.4. Benchmark T3 (anchor-test) — особое место

T3 не просто benchmark — это **existence proof** проекта. Из master spec §13.3:

**Что:** Al 10⁶ atoms, Lennard-Jones `r_c = 8 Å`, 85 atoms in sphere, cyclic BC, Z-axis linear zoning, ring topology.

**Успех:** численное соответствие таблицы performance/efficiency vs N_processors из рис. 29-30 диссертации с погрешностью ≤ 10% на equivalent hardware class.

**Gate:** M5 **cannot merge** без passing T3.

**Status:** fixture landed at M5 (T5.10, commit lands this section). Harness
(`anchor_test_runner`) arrives at T5.11.

`verify/benchmarks/t3_al_fcc_large_anchor/` содержит:

- `README.md` — experiment description, potential-proxy rationale (Morse
  Girifalco-Weizer stands in for the dissertation's LJ at M5 — native LJ is
  post-M5), and the "setup.data is regenerated, not LFS-blob'd" decision;
- `config.yaml` — TDMD config (linear_1d zoning, ring backend + Kahan
  ring reduction, NVE, dt = 1 fs, 1000 steps, Al FCC 10⁶ atoms);
- `lammps_script.in` — LAMMPS parity script; cross-checks LJ physics only,
  not scaling;
- `dissertation_reference_data.csv` — извлечённые числа из рис. 29-30
  (initial shipment is a placeholder per R-M5-8; replacement before T5.11
  runs as a real gate);
- `hardware_normalization.py` — stdlib-only script нормализации
  performance с 2007 железа на текущее; emits `ghz_flops_ratio` scalar;
- `checks.yaml` — per-gate thresholds (efficiency ±10 %, absolute perf
  ±25 %, LAMMPS parity 1e-10 force + 1e-6 thermo, NVE drift 1e-6);
- `acceptance_criteria.md` — точные условия pass/fail + failure-mode
  catalogue + escalation path.

Companion data lives at `verify/data/t3_al_fcc_large_anchor/`:

- `regen_setup.sh` — one-shot LAMMPS invocation that produces `setup.data`
  (10⁶-atom Al FCC lattice at 300 K). Idempotent; `--force` overwrites.
  Not-committed-as-LFS decision follows the T1 precedent
  (`t1_al_morse_500`) — keeps the repo git-LFS-free at the cost of a
  ~30 s regeneration per fresh CI workspace.

### 4.5. T1 (`t1_al_morse_500`) — landed at M1

**Status:** implemented at M1 (commit lands this section). Thermo-only differential
(`run_differential.py`); full trajectory comparison lands at M2 with `DifferentialRunner`.

**System:**

- Al FCC 5×5×5 = 500 atoms, lattice 4.05 Å;
- Morse (Girifalco-Weizer): `D = 0.2703 eV`, `alpha = 1.1646 Å⁻¹`, `r0 = 3.253 Å`,
  `cutoff = 6.0 Å`, `cutoff_strategy: hard_cutoff` (matches LAMMPS `pair_style morse` Strategy A);
- NVE, `dt = 0.001 ps`, 100 steps, thermo every 10 steps;
- Seed 12345, initial T = 300 K via `velocity create ... loop geom dist gaussian` (LAMMPS
  writes the resulting atoms+velocities via `write_data ... nocoeff`, TDMD reads the same file).

**Layout:** `verify/benchmarks/t1_al_morse_500/{README.md, config_metal.yaml, config_lj.yaml, lammps_script_metal.in, checks.yaml}`.
Ground-state data file is produced by LAMMPS at harness run-time rather than committed
as a golden artifact — keeps the benchmark self-regenerating and avoids a 500-atom binary
blob in-tree. The `_metal` / `_lj` suffix pattern distinguishes the two unit variants —
see the **lj variant** subsection below.

**Residual budget (measured at M1 on x86_64 + gcc-13):**

| Column | Threshold | Measured residual | Source |
|---|---|---|---|
| `pe`     | 1.0e-10 rel | 0.0 (bit-exact) | `energy.reference_fp64_vs_lammps` |
| `ke`     | 1.0e-10 rel | 0.0 (bit-exact) | `energy.reference_fp64_vs_lammps` |
| `etotal` | 1.0e-10 rel | 0.0 (bit-exact) | `energy.reference_fp64_vs_lammps` |
| `temp`   | 2.0e-6 rel  | 1.13e-6 | **kB definitional gap (see below)** |
| `press`  | 1.0e-8 rel  | 1.23e-11 | FP64 reduction-order roundoff |

**Irreducible kB residual.** LAMMPS hard-codes `boltz = 8.617343e-5` (older CODATA, truncated);
TDMD uses CODATA 2018 `kB = 8.617333262e-5`. The ratio contributes a flat ~1.13e-6 relative
offset on any temperature comparison. Tightening `thermo_temperature_relative` below this
floor would require degrading TDMD's physical constant, which is explicitly rejected on
fidelity grounds. The threshold is therefore set at 2.0e-6 (1.7× the observed residual)
with this rationale block committed alongside.

**Pressure conversion.** `compare.py` converts LAMMPS `press` (bar) → eV/Å³ using LAMMPS's
truncated internal constant `nktv2p = 1.6021765e+6`, not modern CODATA 2018
(`1.602176634e+6`). The two constants differ by ~2e-8 — using modern CODATA would plant
a systematic residual that the harness would incorrectly attribute to physics. Matching
LAMMPS's constant makes the comparison definition-identical and leaves only reduction-order
roundoff (observed: 1.23e-11 rel).

**CI wiring.** Runs on `ubuntu-latest` under the `differential-t1` job. The job executes the
full pipeline end-to-end (compile harness, run driver) but the LAMMPS submodule is not
fetched on the public runner (Option A policy), so the Catch2 wrapper detects the absent
oracle binary and exits `77` (`SKIP_RETURN_CODE`). Oracle-gated validation is part of the
local pre-push protocol until an isolated runner lands (revisit at M6).

#### 4.5.1. lj variant — D-M1-6 cross-check (landed at M2, T2.4)

**Purpose.** Prove `UnitConverter` is numerically transparent — i.e. the same
physical system expressed in `units: lj` (with identity reference σ=ε=m_ref=1)
and in `units: metal` produces bit-identical thermo through the full
force/integrator pipeline over 100 NVE steps. This closes invariant D-M1-6
(`verify/SPEC.md` §2.2) and gates any future `UnitConverter` refactor.

**How it works.** `run_differential.py --variant both` runs LAMMPS-metal once
(produces `setup.data` + oracle thermo), then two TDMD passes:

1. **TDMD-metal** ingests `setup.data` directly with `config_metal.yaml` (existing
   M1 flow, renamed from `config.yaml`).
2. **TDMD-lj** reads `config_lj.yaml`. Because TDMD's `velocity_from_lj`
   multiplies by `1/sqrt(mvv2e_metal) ≈ 98.23` on ingest (see
   `src/runtime/unit_converter.cpp` "LJ_TIME and LJ_VEL" block), the harness
   pre-scales the Velocities block of `setup.data` by `sqrt(mvv2e_metal)
   ≈ 0.01018` and writes `setup_lj.data`. Length/mass/energy columns are
   untouched — σ=ε=m_ref=1 makes them valid lj numerics bit-for-bit. The lj
   `dt` field carries the inverse scaling: `dt_lj = 0.001 / sqrt(mvv2e) =
   0.09822695059540948` (round-trips to `dt_metal = 0.001` exactly).

Both TDMD runs emit metal-unit thermo (internal representation is always metal,
master spec §5.3), so the cross-diff is a direct column-by-column comparison
at `benchmarks.t1_al_morse_500.cross_unit_relative = 1.0e-10`.

**Residual budget (measured at M2 on x86_64 + gcc-13):**

| Check | Threshold | Measured residual |
|---|---|---|
| TDMD-metal vs LAMMPS | same as §4.5 table above | same as §4.5 |
| TDMD-lj    vs LAMMPS | same as §4.5 table above | same as §4.5 |
| TDMD-metal ≡ TDMD-lj | 1.0e-10 rel per column    | **0.0 (bit-exact) on every column at every step** |

The bit-exact cross-check is not coincidence: with identity reference every
non-dt/non-v conversion is a multiply-by-1, and the sqrt(mvv2e) round-trip
through dt and velocity scaling hits exactly the same floating-point pattern
UnitConverter uses internally, so the two code paths reduce to the same
byte-level state immediately after ingest. The 1e-10 threshold is a generous
safety margin — a future refactor that drifts even a single ulp per step would
show up as ~1e-14 rel residual, still comfortably passing but detectable in
the `max_rel` column of the harness report.

**Not exercised by this variant (deliberate scope).** LAMMPS's own `units lj`
path is NOT validated here — that would test LAMMPS, not TDMD. A full
TDMD-lj vs LAMMPS-lj differential requires LAMMPS-lj to produce a `setup.data`
with equivalent initial conditions (same seed under `units lj` draws
differently) and `compare.py` to handle lj→metal thermo conversion on the
oracle side. Scheduled for T5 (post-M2) if a future benchmark needs it;
D-M1-6 is already fully closed without it.

### 4.7. T4 (`t4_nial_alloy`) — M2 acceptance gate

**Status:** landed at M2 (T2.9). The first end-to-end TDMD-vs-LAMMPS
differential against a many-body **EAM/alloy** potential, and the single
benchmark whose failure blocks M2 closure (master spec §14 M2).

**System:**

- Ni-Al FCC 6×6×6 = 864 atoms, lattice `a₀ = 3.52 Å` (Ni native), 50:50
  Ni:Al random shuffle (`random.Random(12345)`);
- Types: `1 = Ni, m = 58.71`, `2 = Al, m = 26.982` — both taken **verbatim
  from the Mishin 2004 setfl header**, not from tabulated atomic weights.
  LAMMPS `pair_eam_alloy::coeff()` silently overrides atom masses with the
  setfl values on `pair_coeff`, so `setup.data` must match or KE at step 0
  diverges by ~1.7e-5 relative between engines;
- Potential: Mishin 2004 Ni-Al EAM/alloy (`pair_style eam/alloy`),
  `cutoff = 6.72488 Å` from setfl tabulation;
- NVE Velocity-Verlet, `dt = 0.001 ps`, 100 steps, thermo every 10 steps;
- Initial velocities: Maxwell-Boltzmann at 300 K, COM momentum subtracted,
  KE rescaled to hit exactly 300 K under `dof = 3N − 3`;
- Fully periodic box, skin = 0.3 Å.

**Layout:** `verify/benchmarks/t4_nial_alloy/{README.md, generate_setup.py,
setup.data, config_metal.yaml, lammps_script_metal.in, checks.yaml}`. Unlike
T1, the ground-state file `setup.data` is **committed** (not regenerated at
harness time) — the LAMMPS script reads it verbatim via a `-var setup_data
<abs_path>` variable. This keeps both engines on bit-identical bytes and
avoids the LAMMPS `velocity create` PRNG path (which differs from Python's
`random.Random` Gauss sampler). Re-running `generate_setup.py` with the
unchanged seed reproduces the committed file byte-for-byte.

The setfl file `verify/third_party/potentials/NiAl_Mishin_2004.eam.alloy`
(1.9 MB) is committed verbatim from the NIST Interatomic Potentials
Repository — it is not part of the LAMMPS examples tree and we refuse a
runtime dependency on `nist.gov` for reproducibility. Provenance and
license live in `third_party/potentials/README.md`.

**No lj variant.** EAM setfl tables are dimensional by convention
(length in Å, energy in eV), and `SimulationEngine::init` rejects
`units: lj` for `style: eam/alloy` with an explicit error. A cross-unit
T4 variant is out of scope.

**Residual budget (measured at M2 on x86_64 + gcc-13):**

| Column / check | Threshold | Measured residual | Source |
|---|---|---|---|
| `pe`       | 1.0e-10 rel | 0.0 (bit-exact) | `energy.reference_fp64_vs_lammps` |
| `ke`       | 1.0e-10 rel | 0.0 (bit-exact) | `energy.reference_fp64_vs_lammps` |
| `etotal`   | 1.0e-10 rel | 0.0 (bit-exact) | `energy.reference_fp64_vs_lammps` |
| `temp`     | 2.0e-6 rel  | 1.13e-6 | kB definitional gap (§4.5) |
| `press`    | 1.0e-8 rel  | 3.65e-11 | FP64 reduction-order roundoff |
| **forces (per-atom per-component)** | **1.0e-10 rel** | **3.25e-11** | **FP64 reduction-order roundoff** |

Forces residual headroom is 3.08× (3.25e-11 / 1.0e-10). Tighter thresholds
would not add value — the remainder is genuine FP64 cancellation in the
two-pass EAM reduction order, not a TDMD/LAMMPS semantic gap. All thermo
columns reproduce T1's residual pattern (kB gap for temp, bit-exact for
energies) because nothing in the EAM addition changes the thermo pipeline.

**CI wiring.** Runs on `ubuntu-latest` under the `differential-t4` job (same
Option A template as `differential-t1`). Public runner does not fetch the
LAMMPS submodule, so the Catch2 wrapper returns `77` (`SKIP_RETURN_CODE`)
and the job is green without executing the oracle. Local pre-push runs
exercise the full differential against the built-in LAMMPS oracle.

**Known non-issues.** Listed in `verify/benchmarks/t4_nial_alloy/README.md`
under "Known non-issues"; the setfl-mass-override and the temperature kB
gap are the two with persistent cross-release implications.

### 4.6. T7 (`t7_mixed_scaling`) — M7 Pattern 2 strong-scaling probe

**Status:** landed at M7 (T7.11). Strong-scaling probe of the Pattern 2
TD×SD stack on a Ni-Al EAM 131k-atom (~10⁵) system. Per Option A, multi-rank
runs execute locally pre-push (single-node mandatory, 2-node opportunistic
via cloud burst); CI itself does not exercise the harness because public
runners have no GPU.

**System:**

- Ni-Al FCC 32×32×32 = 131,072 atoms, lattice `a₀ = 3.52 Å`, 50:50 random
  Ni:Al shuffle (`random.Random(12345)`) — same algorithmic chain as T4
  (the `generate_setup.py` here delegates to T4's `generate()` symbol);
- Same Mishin 2004 EAM/alloy potential as T4, T6.7, T6.13;
- NVE Velocity-Verlet, `dt = 0.001 ps`, 100 steps, thermo every step;
- Pattern 2 base config: `comm.backend: hybrid` (T7.5),
  `scheduler.pipeline_depth_cap: 1`, `zoning.scheme: linear_1d`. The
  harness injects `zoning.subdomains: [N, 1, 1]` per probe point so the
  fixture config is N-agnostic.
- Fully periodic box, skin = 0.3 Å.

**Layout:** `verify/benchmarks/t7_mixed_scaling/{README.md, config.yaml,
checks.yaml, generate_setup.py, hardware_normalization.py}`. Unlike T4,
`setup.data` is **not committed** (~7.5 MB at 131k atoms) — the harness
lazily regenerates it via `generate_setup.py` on first invocation. The
default output path is `verify/data/t7_mixed_scaling/setup.data`,
mirroring T3's data-vs-fixture split.

**Driver:** `verify/harness/scaling_runner/` (Python, stdlib + PyYAML
only — same dependency profile as `anchor_test_runner`). Public API:

- `RunnerConfig` — paths + knobs;
- `ScalingRunner.run() → ScalingReport` — orchestrator;
- `ScalingProbePoint` — per-N record (steps/sec, efficiency, gate match,
  status);
- `python -m verify.harness.scaling_runner` — CLI front-end.

**Gates:**

- **Efficiency `E(N) = rate(N) / (rate(1) × N) × 100`** vs per-tier gate
  from `checks.yaml::efficiency_gates`. Default tiers:
  - `single_node` (N ∈ [2,8]): ≥80% (D-M7-8 single-node target),
  - `two_node` (N ∈ [9,16]): ≥70% (D-M7-8 2-node target, opportunistic).
- **Pattern 1 byte-exact regression at N=1.** When
  `checks.yaml::pattern1_baseline_byte_exact: true` AND the harness is
  invoked with `--baseline-thermo <path>`, the N=1 thermo trace is
  byte-compared against the supplied baseline. Mirrors D-M7-10 chain
  extension (M3 ≡ M4 ≡ M5 ≡ M6 ≡ M7 thermo).

**Out of scope at M7:**

- Inter-node NCCL physics (M8+); the 2-node tier exercises HybridBackend
  composition (intra-node NCCL + inter-node GpuAwareMPI) but the absolute
  inter-node SNAP-class workload is reserved for M8.
- Dissertation Morse fidelity (M9+ per D-M7-16); T7 ships an EAM
  substitute. The dissertation efficiency-curve match remains a CPU
  property of the T3 anchor.
- PerfModel-based hardware normalisation: `hardware_normalization.py` is
  a stub returning `perfmodel_calibration_ratio: 1.0`. The real loader
  (read JSON measured-coefficient fixture, apply per-(GPU,
  n_atoms_per_subdomain) scaling) lands with T7.13.

**CI wiring.** Per Option A (`project_option_a_ci.md`), no GPU CI runner
exists; the harness is exercised pre-push via
`tests/integration/t7_scaling_local/run_t7_scaling.sh`, which auto-skips
(exit 77) when `nvidia-smi -L` reports no enumerated device. The Python
unit suite (`verify/harness/scaling_runner/test_scaling_runner.py`) runs
in every CI flavor — covers efficiency formula, gate dispatch, augmented
config injection, Pattern 1 byte-exact gate, launch-failure handling, and
report serialisation. 13 cases, ~25 ms wall.

### 4.7. T6 (`t6_snap_tungsten`) — canonical fixture **shipped M8 T8.10**

**Status:** fully shipped at M8 T8.10 (2026-04-21). Fixture choice and oracle
subset gate landed at M8 T8.2; scaffold (`README.md`, `checks.yaml`,
`lammps_script.in`, threshold entries) landed at T8.10a (2026-04-20);
D-M8-7 CPU FP64 byte-exact gate landed at T8.5 (2026-04-20 — 250-atom
differential at max_rel ≈ 8.8e-13 < 1e-12 budget); D-M6-7 extension to GPU
FP64 landed at T8.7 (2026-04-20); T8.10 proper adds the canonical
1024-atom `config.yaml.template`, 8192-atom `config_8192.yaml.template`
scaling variant, and the `tests/integration/m8_smoke_t6/` integration
smoke (D-M8-8 NVE-drift gate). T8.11 cloud-burst strong-scaling campaign
consumes the 8192-atom variant; T8.12 slow-tier exercises
MixedFastSnapOnly differential on all T6 variants.

**Canonical fixture (D-M8-3):**

- **Coefficient set:** `W_2940_2017_2.snap` — SNAP include file that wires
  `pair_style hybrid/overlay zbl 4.0 4.8 snap` + loads
  `W_2940_2017_2.snapcoeff` (30 bispectrum coefficients) +
  `W_2940_2017_2.snapparam` (twojmax, rcutfac, ...).
- **Reference:** Wood & Thompson, "Quantum-Accurate Molecular Dynamics Potential
  for Tungsten" arXiv:1702.07042 (2017). Pure W single-species BCC, 2940 DFT
  training configurations.
- **Path:** `verify/third_party/lammps/examples/snap/W_2940_2017_2.snap`
  (resolved via M1-landed submodule; no binary tracked by tdmd repo).
- **Driver example:** `in.snap.W.2940` — 128-atom BCC W, 100-step NVE,
  `dt = 0.0005 ps`, T₀ = 300 K.
- **Upstream reference log:** `log.15Jun20.snap.W.2940.g++.1` (1-rank) — sanity
  check that local LAMMPS build is byte-exact с upstream to LAMMPS float
  precision; NOT the TDMD acceptance gate (TDMD gate is D-M8-7/D-M8-8 per
  m8 exec pack).

**Why this fixture:** Pure W single-species is simpler для first-pass ML-niche
proof-of-value than W-Be alloy (`WBe_Wood_PRB2019`, reserved for M9+ SNAP
alloy gate). 128 atoms fit single-GPU profiling workflow; 8×8×8 BCC (1024-atom
— two atoms per conventional BCC cell × 512 cells) and 16×16×16 (8192-atom)
variants register at T8.10 for scaling probes. Note: the M8 exec pack prose
labels these "2048-atom" and "16384-atom" respectively; that is a
per-unit-cell-counting arithmetic slip (BCC has 2 atoms/cell) — this SPEC
section and `verify/benchmarks/t6_snap_tungsten/generate_setup.py` are
authoritative.

**Oracle subset verification (landed T8.2):**

`verify/third_party/lammps/install_tdmd/bin/lmp -h | grep ML-SNAP` reports
non-empty; `in.snap.W.2940` executes cleanly against local build; thermo
output matches upstream `log.15Jun20.snap.W.2940.g++.1` byte-for-byte at
5-decimal precision (Step 0: TotEng = −10.98985, Step 100: TotEng = −10.989847).
Path-existence Catch2 gate:
`tests/potentials/test_lammps_oracle_snap_fixture` — self-skips (exit 77) if
submodule not initialized; fails if fixture files go missing from a correctly
initialized submodule.

### 4.8. T8 (`t8_al_fcc_nvt`) — **registered M9 T9.1, lands T9.10**

**Status:** registered at T9.1 (2026-04-22) — fixture and threshold calibration
scheduled for T9.10 per `docs/development/m9_execution_pack.md`. This section
describes the **canonical form** the fixture MUST take when it lands; any
deviation requires Architect + Validation Engineer review.

**Canonical fixture (D-M9-8):**

- **Lattice:** Al FCC, 5×5×5 conventional cells × 4 atoms/cell = **500 atoms**
  (master spec §14 M9 "512 atoms" rounds the canonical cubic count; 500 is the
  exact FCC 5×5×5 number — fixture README SHOULD state both so the arithmetic
  is not lost).
- **Potential:** `eam/alloy` single-species Al (reuses T4 Mishin 2004 `Al.eam.alloy`
  asset). Morse path not selected — per D-M9-8 — because Morse GPU kernel is
  M10+ scope; `eam/alloy` leverages existing M6 T6.5 GPU kernel and gives T8
  GPU coverage without new kernel work.
- **Integrator:** Nosé-Hoover NVT, M = 3 chains, `damping_time = 0.1 ps`,
  `thermostat_update_interval = 50` (integrator/SPEC §4.5 defaults).
- **Thermodynamic setpoint:** T = 300 K, `dt = 1 fs`, 10⁵ steps (plus 10⁴
  equilibration discarded from statistics).
- **Assets at `verify/benchmarks/t8_al_fcc_nvt/`:**

  ```
  t8_al_fcc_nvt/
  ├── README.md              # fixture purpose, atom count arithmetic, provenance
  ├── generate_setup.py      # `--structure fcc_al --nrep 5` → setup.data
  ├── setup.data             # committed (same policy as T1/T4)
  ├── Al.eam.alloy           # reuse from T4 (symlink or fetched from submodule)
  ├── config.yaml            # TDMD config: eam/alloy + nvt
  ├── lammps_script.in       # `fix nvt temp 300 300 0.1` LAMMPS parity
  ├── checks.yaml            # equipartition + Maxwell-Boltzmann + LAMMPS diff
  └── metadata.yaml          # version, creation date, fixture hash
  ```

**Acceptance gates (D-M9-9 statistical, не byte-exact):**

1. **Equipartition** — ⟨KE⟩ = (3/2)·N·k_B·T within **±2σ** over the sampling
   window (10⁵ steps after 10⁴ equilibration). Threshold: `t8.equipartition_rel_band = 0.03`
   (3% envelope on mean kinetic energy);
2. **Velocity distribution Maxwell-Boltzmann** — χ² goodness-of-fit with
   `p > 0.05` on per-component v_x, v_y, v_z histograms (bin count per
   integrator/SPEC §9.3);
3. **LAMMPS parity (pseudo-deterministic)** — 100-step with identical velocity
   seed: positions + velocities match LAMMPS `fix nvt` to ≤ 1e-10 relative at
   step 100 (pre-divergence window, deterministic-stochastic-free gate);
4. **LAMMPS statistical match** — 10⁴-step: ⟨T⟩ within ±3 K (1% rel at 300 K),
   ⟨KE⟩ within ±2σ of analytic equipartition;
5. **NVE-sanity (regression)** — NVT fix disabled in a parallel run: NVE drift
   `< 1e-5 / 1000 steps` holds (reaffirms §4.2 Trotter ordering correctness).

**Tier assignment:** medium (equipartition + LAMMPS parity, ~30 min) + slow
(full 10⁵-step statistical campaign). Fast-tier excluded — 10⁵ steps is
nightly budget.

**Threshold registry entries (reserved at T9.1, activated at T9.10):**

```
benchmarks.t8_al_fcc_nvt.equipartition_rel_band              status: reserved
benchmarks.t8_al_fcc_nvt.maxwell_boltzmann_chi2_p_min        status: reserved
benchmarks.t8_al_fcc_nvt.lammps_step100_forces_rel           status: reserved
benchmarks.t8_al_fcc_nvt.lammps_statistical_temp_k_tol       status: reserved
```

See `verify/thresholds/thresholds.yaml` for full schema. Values calibrated
from the first successful 10⁵-step run at T9.10.

### 4.9. T9 (`t9_al_fcc_npt`) — **registered M9 T9.1, lands T9.11**

**Status:** registered at T9.1 (2026-04-22) — fixture scheduled for T9.11. Same
canonical-form contract as T8.

**Canonical fixture (D-M9-8 shared):**

- **Lattice + potential + chain_length + dt:** identical to T8 (Al FCC 500
  atoms, `eam/alloy` Al, M=3 chains, `dt = 1 fs`).
- **Integrator:** Nosé-Hoover NPT isotropic (Parrinello-Rahman-like per
  integrator/SPEC §5.1), `temperature = 300 K`, `pressure = 1 bar`,
  `temperature_damping = 0.1 ps`, `pressure_damping = 1.0 ps`,
  `barostat_update_interval = 100` (integrator/SPEC §4.5.4 default).
- **Run length:** 10⁵ steps (10⁴ equilibration discarded).
- **Scope:** isotropic volume flex only; anisotropic (stress tensor) is v2+
  per integrator/SPEC §5.3.
- **Assets at `verify/benchmarks/t9_al_fcc_npt/`:**

  ```
  t9_al_fcc_npt/
  ├── README.md              # shares T8's arithmetic rationale; NPT-specific setpoint
  ├── generate_setup.py      # delegates to T8 — identical lattice
  ├── setup.data             # may be symlink to T8's setup.data (same lattice)
  ├── Al.eam.alloy           # shared asset
  ├── config.yaml            # TDMD config: eam/alloy + npt isotropic
  ├── lammps_script.in       # `fix npt temp 300 300 0.1 iso 1 1 1` parity
  ├── checks.yaml            # volume match + fluctuation + LAMMPS diff
  └── metadata.yaml
  ```

**Acceptance gates (D-M9-9):**

1. **Equilibrium volume** — ⟨V⟩ matches LAMMPS NPT within **2% relative** over
   the sampling window (consequence of matched pressure setpoint + matched EAM
   parameters + Variant A thermostat policy). Threshold:
   `t9.equilibrium_volume_rel_tol = 0.02`;
2. **Volume fluctuation** — σ_V matches LAMMPS within **5% relative**. The
   fluctuation magnitude is ensemble-statistical-mechanics predicted from
   κ_T·k_B·T·V (isothermal compressibility); LAMMPS baseline under identical
   params is the reference;
3. **Temperature control** — while box flexes, ⟨T⟩ remains within ±3 K of
   setpoint (D-M9-9 inherited from T8);
4. **LAMMPS 100-step parity** — positions + velocities + box volume match
   LAMMPS `fix npt iso` to ≤ 1e-10 relative at step 100 (deterministic seeds);
5. **Stability sanity** — 10⁵-step trajectory does not exhibit box-variable
   exponential runaway (`barostat_mass_damping > 0` well-formed per §5.2
   validation bands).

**Tier assignment:** same as T8 (medium + slow). 10⁵-step NPT is comparable
cost to 10⁵-step NVT (single Allreduce per `barostat_update_interval` over
the NVT cost).

**Threshold registry entries (reserved at T9.1, activated at T9.11):**

```
benchmarks.t9_al_fcc_npt.equilibrium_volume_rel_tol          status: reserved
benchmarks.t9_al_fcc_npt.volume_fluctuation_rel_tol          status: reserved
benchmarks.t9_al_fcc_npt.lammps_step100_box_rel              status: reserved
```

**Shared with T8:** `setup.data` file identity (isotropic NPT drifts volume
**away** from initial lattice; NVT holds volume fixed — same *starting* state
is intentional and simplifies the "flip one knob at a time" debugging
principle).

### 4.10. Adding new benchmark

При добавлении нового потенциала или расширении (e.g. новый ML model в wave 3), добавляется новый Txx benchmark. Процедура:

1. Создать `verify/benchmarks/t<N>_<descriptive_name>/`;
2. Записать config, reference, checks.yaml;
3. Регистрировать в `verify/tiers/` соответствующего tier;
4. Добавить в `thresholds.yaml` если нужны benchmark-specific tolerances;
5. Документировать в master spec §13.2.

---

## 5. LAMMPS integration

### 5.1. Как LAMMPS используется

LAMMPS — **external oracle**, не runtime dependency. Используется только в VerifyLab для differential comparison:

1. `verify/harness/differential_runner/` генерирует LAMMPS input от TDMD yaml;
2. Запускает LAMMPS subprocess;
3. Запускает TDMD с identical config;
4. Сравнивает outputs (forces, energy, trajectories).

В production runs LAMMPS **не нужен** — TDMD самостоятельный.

### 5.2. Submodule build

**Решение:** LAMMPS как **git submodule** в `verify/third_party/lammps/`.

Основание:
- Reproducibility: every TDMD commit имеет pinned LAMMPS version;
- No external dependency issues в CI;
- AI-агенты могут собрать LAMMPS сами из submodule при first invocation;
- Full control над LAMMPS version (не зависим от системного apt / conda).

### 5.3. Setup

```bash
# В корне tdmd_project:
git submodule add https://github.com/lammps/lammps.git verify/third_party/lammps
git submodule update --init --recursive

# Pinned version (обновляется периодически):
cd verify/third_party/lammps
git checkout stable_23Jun2022_update4   # example
cd ../../..
git add verify/third_party/lammps
```

### 5.4. Build script

`verify/third_party/build_lammps.sh`:

```bash
#!/bin/bash
set -euo pipefail

LAMMPS_DIR="verify/third_party/lammps"
BUILD_DIR="$LAMMPS_DIR/build"

# Minimum packages TDMD needs for differential tests:
PACKAGES="-D PKG_EXTRA-PAIR=yes \
          -D PKG_MANYBODY=yes \
          -D PKG_ML-SNAP=yes \
          -D PKG_KOKKOS=yes \
          -D Kokkos_ARCH_NATIVE=yes"

cmake -S "$LAMMPS_DIR/cmake" -B "$BUILD_DIR" \
      -D CMAKE_BUILD_TYPE=Release \
      $PACKAGES

cmake --build "$BUILD_DIR" -j

echo "LAMMPS binary: $BUILD_DIR/lmp"
```

### 5.5. Version pinning

`verify/third_party/lammps_version.txt`:

```
# LAMMPS version currently used as reference oracle.
# Updates require Architect approval и full slow-tier re-validation.

version_tag: stable_23Jun2022_update4
commit: abc1234...
last_updated: 2026-04-16
reason: "Initial setup"
```

Change of LAMMPS version — отдельный PR (`verify/` change) с full tier re-run.

### 5.6. Agent-buildable

Claude Code agent при first encounter VerifyLab:

```bash
# Agent command:
./verify/third_party/build_lammps.sh
# → binary at verify/third_party/lammps/build/lmp (or lmp_kokkos)
```

Binary cached between runs. Agent не rebuilds если version не менялась.

### 5.7. Fallback: LAMMPS not available

Если build failed или binary missing — VerifyLab:
- emits clear warning;
- skips differential tests (marks as "SKIPPED: LAMMPS unavailable");
- все **self-reference** tests (anchor-test, analytic T0, conservation) продолжают работать;
- CI gate: diagnostic-only в этом случае, не hard fail (чтобы не блокировать dev без LAMMPS).

---

## 6. Reference data

### 6.1. Storage: в самом проекте

**Решение:** все reference data лежат в `verify/reference/`, committed в git.

Rationale:
- Единый clone → все нужные данные сразу;
- Reproducibility: reference bound к commit SHA;
- No external infrastructure dependencies;
- Git LFS если файлы большие (> 10 MB).

### 6.2. Size estimate

Для всех 8 benchmarks:

| Benchmark | Size | Format |
|---|---|---|
| T0 analytic | 100 KB | JSON + text |
| T1 small | 500 KB | HDF5 (1000 steps) |
| T2 medium | 50 MB | HDF5 (5000 steps) |
| T3 anchor | 200 MB | HDF5 (1000 steps, 10⁶ atoms) |
| T4 alloy | 10 MB | HDF5 |
| T5 MEAM | 20 MB | HDF5 (v1.5) |
| T6 SNAP | 100 MB | HDF5 |
| T7 scaling | 5 MB | metadata only (generated on-the-fly) |

**Total: ~400 MB.**

Git LFS для файлов > 10 MB. `verify/reference/trajectories/` configured as LFS path.

### 6.3. Structure

```
verify/reference/
├── trajectories/              # Git LFS
│   ├── t1_al_fcc_small_ref.h5
│   ├── t2_al_fcc_medium_ref.h5
│   ├── t3_al_fcc_large_ref.h5
│   ├── t4_nial_alloy_ref.h5
│   └── t6_w_snap_ref.h5
│
├── analytic/                  # не LFS, small
│   ├── t0_morse_2atom.json
│   └── harmonic_oscillator_ref.json
│
├── observables/               # pre-computed averages
│   ├── t2_medium_nve_drift.json
│   └── t4_alloy_diffusion.json
│
├── lammps_scripts/            # LAMMPS inputs
│   ├── t1_al_fcc_small.in
│   ├── t2_al_fcc_medium.in
│   └── ...
│
└── baselines/                 # current stored baselines
    ├── version_manifest.json
    └── snapshots/
        └── baseline_v2.1.0/
```

### 6.4. Baseline regeneration

Reference data стала stale (e.g. LAMMPS updated, or potential file refined):

```bash
# Regenerate all references:
./verify/harness/regenerate_references.sh --tier slow

# Regenerate one:
./verify/harness/regenerate_references.sh --benchmark t4_nial_alloy
```

Script:
1. Runs LAMMPS с current lammps_script.in;
2. Saves trajectory to `reference/trajectories/<tN>_<name>_ref.h5`;
3. Computes observables, saves to `reference/observables/`;
4. Updates `reference/baselines/version_manifest.json` с new hash.

**Regeneration — отдельный PR**. Never mid-flight в другом PR. Требует approval Architect'а + Validation Engineer'а.

### 6.5. Version manifest

```json
{
  "baseline_version": "v2.1.0",
  "created": "2026-04-16T12:00:00Z",
  "lammps_version": "stable_23Jun2022_update4",
  "tdmd_version_used_for_generation": "N/A (only LAMMPS for v2.1.0)",
  "references": {
    "t1_al_fcc_small": {
      "trajectory_hash": "sha256:abc123...",
      "observables_hash": "sha256:def456..."
    },
    "t4_nial_alloy": {
      "trajectory_hash": "sha256:...",
      "observables_hash": "sha256:..."
    }
  }
}
```

---

## 7. Harness — executable framework

### 7.1. Differential runner

`verify/harness/differential_runner/`:

```python
# Pseudo-code; can be C++, Python, or shell
class DifferentialRunner:
    def run(self, benchmark: Benchmark, mode: str = "full") -> DifferentialReport:
        # 1. Prepare environment
        workspace = mkdtemp()

        # 2. Run LAMMPS
        lammps_output = self.run_lammps(benchmark.lammps_script, workspace)

        # 3. Run TDMD
        tdmd_output = self.run_tdmd(benchmark.config, workspace)

        # 4. Compare
        report = self.compare_outputs(
            lammps_output, tdmd_output,
            checks=benchmark.checks,
            thresholds=ThresholdRegistry.instance())

        return report
```

### 7.2. Conservation checker

`verify/harness/conservation_checker/`:

```python
class ConservationChecker:
    def check_nve_drift(self, trajectory: Trajectory, tolerance: float) -> Result
    def check_momentum_conservation(self, trajectory: Trajectory) -> Result
    def check_angular_momentum_nvt(self, trajectory: Trajectory) -> Result
```

### 7.3. Observables comparator

```python
class ObservablesComparator:
    def compare_temperature(self, tdmd, reference, tolerance_sigma) -> Result
    def compare_pressure(self, tdmd, reference, tolerance_sigma) -> Result
    def compare_msd(self, tdmd, reference, relative_tolerance) -> Result
    def compare_rdf(self, tdmd, reference, peak_tolerance) -> Result
```

### 7.4. Anchor test runner

`verify/harness/anchor_test_runner/` (shipped T5.11, 2026-04-19):

```python
class AnchorTestRunner:
    """
    Reproduces Andreev dissertation §3.5 experiment.
    Runs TDMD with Ring topology for N_procs in {4, 8, 16, ...}.
    Compares:
      - per-processor performance curves
      - efficiency curves
      - scaling linearity (R² of linear fit)
    against dissertation_reference_data.csv.
    """
    def run(self) -> AnchorTestReport
```

**Files:**

- `runner.py` — `AnchorTestRunner.run()` orchestrator + telemetry parse.
- `report.py` — `AnchorTestReport` / `AnchorTestPoint` dataclasses;
  `STATUS_{GREEN,YELLOW,RED}` tri-state; JSON round-trip.
- `hardware_probe.py` — wraps T5.10 `hardware_normalization.py`;
  24h on-disk cache at `~/.cache/tdmd/hardware_flops.json`;
  `--force-probe` bypass flag.
- `__main__.py` — CLI: `python -m verify.harness.anchor_test_runner
  --tdmd-bin … --output report.json`. Exit codes 0/1/2/3 =
  GREEN/YELLOW/RED/infra-error.
- `test_anchor_runner.py` — mocked-TDMD smoke suite (8 cases).
- `tests/integration/m5_anchor_test/` — local-only wrapper
  (`run_anchor_test.sh` + README); pre-push mandatory per D-M5-13.

The runner NEVER invokes LAMMPS for comparison (T3 is dissertation
match, not oracle diff); LAMMPS is only consulted transparently when
`setup.data` regeneration is required on a fresh workspace (T1
precedent; see `verify/data/t3_al_fcc_large_anchor/regen_setup.sh`).

### 7.5. Perfmodel validator

Integrates с `perfmodel/`:

```python
class PerfmodelValidator:
    def validate_prediction(self,
                            benchmark: Benchmark,
                            predicted: PerformancePrediction,
                            measured: TimingBreakdown) -> ValidationReport:
        error = abs(predicted.t_step - measured.t_step) / measured.t_step
        threshold = thresholds.get(f"perfmodel.pattern{predicted.pattern_num}_prediction_error")
        return Result(passed=(error < threshold), error=error, ...)
```

---

## 8. Tiers (fast / medium / slow)

### 8.1. Tier definition

Tests sorted by cost:

| Tier | Time budget | Triggered by | Benchmark subset |
|---|---|---|---|
| **fast** | < 5 minutes | every PR | T0, T1 (run 0 + 1000 steps) |
| **medium** | < 30 minutes | nightly + on-demand | T0-T4 (full observables) |
| **slow** | < 8 hours | release + threshold change | T0-T7 (full), anchor-test, scaling |

### 8.2. `verify/tiers/fast.yaml`

```yaml
tier: fast
max_duration: 300s
benchmarks:
  - t0_morse_analytic:
      checks: [run0_forces, run0_energy]
  - t1_al_fcc_small:
      checks: [run0_forces, nve_drift_1k_steps]
      steps: 1000
required_for_merge: true
lammps_required: false    # T0 и T1 мы could сделать self-reference
```

### 8.3. `verify/tiers/medium.yaml`

```yaml
tier: medium
max_duration: 1800s
benchmarks:
  - t1_al_fcc_small:
      full_suite: true
  - t2_al_fcc_medium:
      checks: [run0_forces, nve_drift_10k, temperature_distribution]
      steps: 10000
  - t3_al_fcc_large_anchor:
      mode: smoke_test         # abbreviated anchor
  - t4_nial_alloy:
      full_suite: true
trigger: nightly
```

### 8.4. `verify/tiers/slow.yaml`

```yaml
tier: slow
max_duration: 28800s
benchmarks:
  - t1_al_fcc_small:  full
  - t2_al_fcc_medium: full
  - t3_al_fcc_large_anchor:
      mode: full_reproduction     # real Andreev scaling experiment
      processor_counts: [1, 2, 4, 8, 16, 32, 64]
  - t4_nial_alloy: full
  - t5_si_meam: full             # M9+ only
  - t6_w_snap: full              # M8+ only
  - t7_mixed_scaling: full       # M7+ only
trigger:
  - release
  - threshold_change
  - lammps_version_change
  - monthly_comprehensive
```

### 8.5. CI integration

Master spec §11 pipelines расширяются:

```
Pipeline D — Differential
  → runs verify tier "fast"

Nightly Pipeline
  → runs verify tier "medium"

Release Pipeline
  → runs verify tier "slow"
```

Fast tier — mandatory merge gate. Medium tier — informational (failures notify but не блокируют merge, но require fix в течение N дней). Slow tier — blocking for release.

---

## 9. Diagnostic mode (не только pass/fail)

### 9.1. Goal

CI binary pass/fail — удобно для automation, но **недостаточно** для scientific understanding. VerifyLab всегда emits **detailed diagnostic report**, даже когда все passed.

### 9.2. Report structure

```
verify/reports/<run_id>/
├── summary.json                 # machine-readable overall verdict
├── summary.md                    # human-readable overview
├── per_benchmark/
│   ├── t1_al_fcc_small/
│   │   ├── report.md
│   │   ├── forces_diff.png       # visualizations
│   │   ├── energy_drift.png
│   │   ├── raw_data.h5            # raw measurements
│   │   └── diagnosis.md          # если failed — что делать
│   └── ...
└── environment.json              # hardware, LAMMPS version, TDMD commit
```

### 9.3. Diagnostic content per benchmark

```markdown
# T4 NiAl Alloy Benchmark Report

**Benchmark:** t4_nial_alloy
**Tier:** medium
**TDMD commit:** abc1234
**LAMMPS version:** stable_23Jun2022_update4
**Status:** ✅ PASS (all checks)

## Checks

### run0_forces: ✅ PASS
- Maximum |Δf| / |f|: 3.2e-12
- Threshold: 1e-10
- Margin: 31 orders of magnitude below threshold
- Atoms with largest deviation: #42, #128 (near crystal boundary — expected)

### nve_drift_10k_steps: ✅ PASS
- Measured drift: 2.1e-7 per ns
- Threshold: 1e-6 per ns
- Interpretation: well within bounds, typical для VV FP64

### temperature_distribution: ✅ PASS
- Chi-square p-value: 0.42
- Threshold: 0.05
- Distribution: визуально Maxwell-Boltzmann [see plot]

## Visualizations
[forces_diff.png — scatter plot TDMD vs LAMMPS]
[energy_drift.png — drift over time]

## Conclusion
Benchmark passes all checks. No action required.
```

### 9.4. Diagnostic for failures

Когда test fails:

```markdown
# T2 Al FCC Medium — ❌ FAIL

**Status:** FAILED on check `nve_drift_10k_steps`

## Failed check details
- Measured drift: 3.8e-6 per ns
- Threshold: 1e-6 per ns
- Margin: 3.8× over threshold ❌

## Likely causes
1. Integrator regression (check recent changes в integrator/)
2. Force calculation issue (check potentials/)
3. Threshold too tight для current precision policy

## Suggested diagnosis steps
1. Run `tdmd compare case.yaml --with lammps --step-by-step` для identification где divergence started
2. Check `potentials.morse.parameter_checksum` — parameters not corrupted?
3. Check telemetry — есть ли anomalies в neighbor_rebuilds_total?

## Related information
- Similar historic failures: git log for "nve drift regression"
- Related issues: #N/A
```

---

## 10. Acceptance report

### 10.1. PR acceptance report

При каждом PR run VerifyLab generates:

```markdown
# VerifyLab Acceptance Report — PR #1234

**Commit:** abc1234
**Base:** main@def5678
**Tier:** fast

## Summary: ✅ ACCEPTED

- All 2 benchmarks PASSED
- Duration: 3 minutes 42 seconds
- Merge recommendation: APPROVED (VerifyLab perspective)

## Details
[tables, links to per-benchmark reports]

## Comparison with baseline
- Forces deviation vs baseline: no change (±5% acceptable)
- Energy drift vs baseline: slight improvement (-12%)
- No regression detected

## Next tier runs
Medium tier will run tonight at 02:00 UTC as part of nightly pipeline.
```

### 10.2. Release acceptance certificate

Для release:

```markdown
# TDMD Release Certificate — v2.2.0-rc1

**Commit:** abc1234
**Date:** 2026-04-16
**Tier coverage:** slow (all benchmarks, full suite)

## Certification status: ✅ CERTIFIED FOR RELEASE

## Benchmarks passed (8/8)
- T0 Morse analytic: PASS
- T1 Al FCC small: PASS (10⁴ steps observables within envelope)
- T2 Al FCC medium: PASS
- T3 Al FCC large (anchor): PASS (efficiency 92.4%, within 10% of dissertation)
- T4 NiAl alloy: PASS
- T5 Si MEAM: PASS
- T6 W SNAP: PASS (**3.2× faster than LAMMPS on 16 ranks**)
- T7 Mixed scaling: PASS (efficiency 82% on 2 nodes × 8 GPU)

## Scientific claims certified
- ✅ TD method implementation correctness (via anchor-test)
- ✅ LAMMPS-level correctness for EAM/alloy
- ✅ SNAP ML niche proof-of-value (T6 > LAMMPS threshold)
- ✅ Pattern 2 multi-node scaling

## Signature
- Validation Engineer: <signature/approver>
- Architect: <signature/approver>

Certificate issued: 2026-04-16T14:23:00Z
```

Этот certificate становится частью reproducibility bundle release'а.

---

## 11. Public interface

### 11.1. Main API

```cpp
namespace tdmd::verify {

enum class Tier {
    Fast,           // < 5 min
    Medium,         // < 30 min
    Slow            // < 8 hours
};

struct VerifyConfig {
    Tier                       tier;
    std::vector<std::string>   benchmark_filter;   // empty = all in tier
    std::string                tdmd_binary_path;
    std::string                lammps_binary_path;  // empty = build from submodule
    std::string                output_dir;
    bool                       parallel_benchmarks = true;
    bool                       generate_visualizations = true;
};

struct BenchmarkResult {
    std::string           name;
    bool                  passed;
    double                duration_seconds;
    std::vector<CheckResult>  checks;
    std::string           diagnostic_report_path;
};

struct VerifyReport {
    bool                  overall_passed;
    Tier                  tier_used;
    std::vector<BenchmarkResult>  benchmarks;
    std::string           summary_path;
    std::string           environment_fingerprint;
};

class VerifyLab {
public:
    virtual VerifyReport run(const VerifyConfig&) = 0;

    // Regenerate reference data from LAMMPS (only run explicitly):
    virtual void regenerate_references(
        const std::vector<std::string>& benchmarks) = 0;

    // Threshold introspection:
    virtual const ThresholdRegistry& thresholds() const = 0;

    virtual ~VerifyLab() = default;
};

} // namespace tdmd::verify
```

### 11.2. CLI integration

`cli/SPEC.md` §6 `tdmd compare` — delegate to VerifyLab:

```
tdmd verify <config>              # alias for verify fast tier
tdmd verify <config> --tier medium
tdmd verify --all-benchmarks --tier slow

tdmd verify --list                # list benchmarks
tdmd verify --threshold-registry  # show all thresholds

tdmd verify --regenerate <benchmark>  # regenerate reference
```

Existing `tdmd compare` остаётся для backwards compatibility, becomes wrapper around `tdmd verify`.

---

## 12. Tests (verify module's own tests)

### 12.1. Unit tests для harness itself

VerifyLab тоже имеет bugs. Unit tests для:

- `ThresholdRegistry.get()` — correct path resolution;
- `DifferentialRunner.compare_forces()` — synthetic data (known deltas) → correct verdict;
- `ConservationChecker.check_nve_drift()` — analytic test case;
- `AnchorTestRunner` — mock trajectories → expected efficiency computation.

### 12.2. Meta-tests (tests of tests)

- **False positive test:** intentionally broken TDMD (injected force error) → VerifyLab detects it.
- **False negative test:** correct TDMD → VerifyLab не делает spurious failures.
- **Threshold sensitivity test:** change threshold by 1% → predictable change в pass/fail.

### 12.3. Determinism

VerifyLab **сам по себе** детерминистичен:
- Same TDMD commit + same reference data → same report byte-for-byte;
- Timestamps в reports — optional (for reproducibility of reports themselves).

---

## 13. Telemetry

Metrics:

```
verify.benchmark_runs_total
verify.benchmark_pass_rate
verify.benchmark_duration_seconds
verify.threshold_lookups_total
verify.lammps_invocations_total
verify.lammps_build_time_seconds
verify.reference_data_hits_total
verify.reference_data_misses_total
verify.diagnostic_reports_generated
```

Integrated в standard TDMD telemetry (`telemetry/SPEC.md` §3).

---

## 14. Roadmap alignment

| Milestone | VerifyLab deliverable |
|---|---|
| M0 | Skeleton module, `thresholds.yaml` initial version |
| **M1** | T0, T1 benchmarks; fast tier CI integration; `ThresholdRegistry` (closed 2026-04-18) |
| **M2** | T4 EAM benchmark; LAMMPS submodule; DifferentialRunner MVP; medium tier (closed 2026-04-18) |
| M3 | `ConservationChecker`, observable comparators (closed 2026-04-18) |
| M4 | Fast/medium tiers fully operational for all then-existing benchmarks (closed 2026-04-19) |
| **M5** | **AnchorTestRunner**; **T3 full Andreev reproduction**; slow tier (closed 2026-04-19) |
| M6 | GPU-aware benchmarks; mixed-precision tolerance handling (closed 2026-04-19) |
| **M7** | T7 mixed-scaling (Pattern 2); `PerfmodelValidator` integrated (closed 2026-04-20) |
| **M8** | **T6 SNAP benchmark**; release certificate process (**closed 2026-04-22** via Case B honest-documentation per D-M8-6; `v1.0.0-alpha1` shipped) |
| **M9** | **T8 NVT + T9 NPT Al FCC canonical benchmarks** (fixtures at T9.10/T9.11); NVT/NPT differential harness vs LAMMPS (statistical gates per D-M9-9); PolicyValidator K=1-for-thermostatted CI gate; `v1.0.0-beta1` target tag at T9.13 |
| **M10** | T5 MEAM Si benchmark; MEAM differential vs LAMMPS `meam/c` |
| **M11** | Variant C NVT-in-TD research-window validation suite (if go/no-go passes) |
| **M12** | T10 PACE/MLIAP benchmarks (Cu copper, Python-plugin harness) |
| v2+ | Statistical uncertainty quantification; continuous fuzzing; community benchmark submissions |

---

## 15. Open questions

1. **Alternative oracles beyond LAMMPS** — GROMACS? ASE? Useful для cross-validation but adds maintenance. Post-v1 if community benchmark requests.
2. **Python bindings** — many scientific users work в Python. VerifyLab Python API (`pytdmd_verify`)? Nice-to-have, post-v1.
3. **Visualization output** — what plotting library? matplotlib (safe, widely available) vs plotly (interactive, heavier). Recommendation: matplotlib for CI, plotly optional для interactive reviewers.
4. **Statistical tests sophistication** — current approach uses simple thresholds. Advanced: bootstrap confidence intervals, Bayesian model comparison. Post-v1 research.
5. **Community-submitted benchmarks** — allow users to contribute new benchmarks (e.g. their own potential / system)? Requires review process, но polezen для ecosystem. Post-v1.
6. **Reference data compression** — 400 MB может become 4 GB с больше benchmarks. LFS vs external hosting (Zenodo? S3?)? Probably LFS до 10 GB, потом migrate.
7. **Real-time monitoring** — для long slow-tier runs useful live dashboard. Integration with Prometheus (см. telemetry/SPEC §4.4).
8. **Mutation testing** — инъектировать bugs в TDMD code, проверить что VerifyLab их ловит. Advanced, но может выявить слабые места. Post-v1.

---

---

## 16. Change log

### v1.1 — 2026-04-22 (M9 T9.1 SPEC delta)

- **§4.1 benchmark table** extended to T0–T9: new rows T8 `al-fcc-nvt` and
  T9 `al-fcc-npt` registered per master spec §14 M9. Headroom notice added:
  T10+ reserved for M12 PACE/MLIAP.
- **§4.8 T8 NVT** section authored (mirrors §4.7 T6 structure): canonical
  fixture (Al FCC 5×5×5 = 500 atoms, EAM/alloy Al, Nosé-Hoover chains M=3,
  300 K, 10⁵ steps), five acceptance gates (equipartition, Maxwell-Boltzmann
  χ², LAMMPS 100-step parity, LAMMPS statistical, NVE-sanity regression),
  threshold registry entries reserved (activated at T9.10).
- **§4.9 T9 NPT** section authored: same fixture contract, isotropic NPT
  (1 bar, `pressure_damping=1 ps`, `barostat_update_interval=100`), five
  acceptance gates (⟨V⟩ within 2% vs LAMMPS, σ_V within 5%, ⟨T⟩ within ±3 K,
  100-step box parity, stability sanity); reserved threshold entries
  activated at T9.11.
- **§4.10 Adding new benchmark** section renumbered from §4.8.
- **§14 roadmap** extended: M1–M8 rows annotated with closure dates (including
  M8 CLOSED 2026-04-22 with `v1.0.0-alpha1` shipped per Case B D-M8-6). New
  rows: M9 (T8 + T9 + PolicyValidator K=1 gate + `v1.0.0-beta1` target),
  M10 (T5 MEAM), M11 (Variant C research validation), M12 (T10 PACE/MLIAP).

### v1.0 — 2026-04-16

- Initial VerifyLab module SPEC landed at M0 kickoff. T0–T7 benchmark table,
  threshold registry schema, tier policy (fast/medium/slow), LAMMPS-as-oracle
  procedure.

---

*Конец verify/SPEC.md v1.1, дата: 2026-04-22.*
