# TDMD BuildFlavors — когда какой использовать

Этот документ — scientist-facing руководство по выбору `TDMD_BUILD_FLAVOR`
для production / development / research runs. Технические детали numerical
semantics каждого флейвора — master spec §D (Приложение D precision policy);
compat matrix + acceptance thresholds — §D.12 + §D.13.

**Last updated:** 2026-04-20 (M8 T8.8 — `MixedFastSnapOnlyBuild` landed).

---

## TL;DR decision tree

```
Нужна ли битовая воспроизводимость через runs на разном layout / ranks?
│
├── ДА (научная публикация, bitwise comparison с LAMMPS, CI oracle)
│   └── Fp64ReferenceBuild
│
└── НЕТ (production runs с observables-level reproducibility)
    │
    ├── Workload содержит SNAP (или другой ML-IAP) как dominant cost?
    │   │
    │   ├── ДА + только SNAP (без secondary EAM на ML-not-covered regions)
    │   │   └── MixedFastSnapOnlyBuild
    │   │
    │   ├── ДА + hybrid SNAP+EAM где EAM precision критична
    │   │   └── MixedFastSnapOnlyBuild
    │   │
    │   └── ДА + hybrid SNAP+EAM где обе stороны могут терпеть FP32
    │       └── MixedFastBuild
    │
    ├── Workload EAM/pair only (без SNAP)?
    │   └── MixedFastBuild
    │
    ├── Чистая FP64 production без performance concern?
    │   └── Fp64ProductionBuild
    │
    ├── Ensemble screening / research, NVE drift gate можно отключить?
    │   └── MixedFastAggressiveBuild (⚠ research-only)
    │
    └── Extreme throughput, все numerical gates off?
        └── Fp32ExperimentalBuild (⚠⚠ research-only)
```

---

## Полный перечень (шесть canonical flavors)

### 1. `Fp64ReferenceBuild` — каноническая битовая oracle

**Когда использовать:**
- Научная публикация, где нужна bitwise reproducibility через layout changes
  (различное число ranks, GPU, разные схемы разбиения).
- CI reference runs (oracle для differential tests против LAMMPS).
- Отладка numerical regression — bitwise comparison прежней и новой versions.
- Верификация новых BuildFlavor'ов (§D.17 step 5 slow-tier pass).

**Numerical guarantees (§D.13):**
- Forces vs LAMMPS: ≤ 1e-10 rel
- Energy vs LAMMPS: ≤ 1e-10 rel
- NVE drift per 1000 steps: ≤ 1e-8
- Layout-invariant determinism: **exact** (bitwise)
- Bitwise same-run reproduce: **exact**

**Стоимость:** все force kernels в FP64; нет atomic reductions; static
reduction trees; task stealing запрещён. На GPU работа в FP64 throughput
(~2x медленнее FP32 peak FLOPs на non-HPC cards вроде RTX 5080).

**Не использовать для:** production throughput runs (overhead значителен).

**ExecProfile compat:** `Reference` (canonical) / `Production` (allowed).
`FastExperimental` с этим флейвором выдаёт warning (FP64 overkill для fast).

---

### 2. `Fp64ProductionBuild` — scientific production FP64

**Когда использовать:**
- Production runs где precision критична, но layout-invariant determinism
  не обязателен (один и тот же hardware + layout будет использован).
- Длинные NVE runs (10⁶+ steps) где drift budget 1e-6 per 1000 steps
  требуется.
- Публикация энергетики и forces где FP32 noise недопустимо, но строгая
  bitwise reproducibility не нужна.

**Numerical guarantees (§D.13):**
- Forces vs LAMMPS: ≤ 1e-10 rel
- Energy vs LAMMPS: ≤ 1e-10 rel
- NVE drift per 1000 steps: ≤ 1e-6
- Layout-invariant determinism: exact (stretch — не гарантируется абсолютно)
- Bitwise same-run reproduce: **exact**

**Стоимость:** как Reference, но разрешён FMA contraction (допустимый
~3-5% speedup без numerical cost).

**ExecProfile compat:** `Reference` (allowed, emits "identical to Ref"
warning) / `Production` (canonical) / `FastExperimental` (allowed).

---

### 3. `MixedFastBuild` — default fast mixed (Philosophy B)

**Когда использовать:**
- **Default throughput target** для production TDMD runs.
- EAM-dominated workloads (или чистый pair-style) на GPU.
- Hybrid SNAP+EAM workloads где обе стороны могут терпеть FP32 force
  envelope (1e-5 rel / 1e-7 rel).

**Numerical guarantees (§D.13):**
- Forces vs LAMMPS (dense-cutoff): ≤ 1e-5 rel
- Energy vs LAMMPS: ≤ 1e-7 rel
- NVE drift per 1000 steps: ≤ 1e-5
- Layout-invariant determinism: **observables only** (не bitwise)
- Bitwise same-run reproduce: exact

**Стоимость:** force kernels в FP32, accumulators в FP64, state в FP64.
Типовой GPU speedup 2-3x vs Fp64Production (на RTX 5080 для EAM).

**ExecProfile compat:** `Reference` **REJECTED** (philosophy mismatch) /
`Production` (allowed — validated-only) / `FastExperimental` (canonical).

---

### 4. `MixedFastSnapOnlyBuild` — SNAP-dominant heterogeneous (Philosophy B-het) — **M8 T8.8 landed**

**Когда использовать:**
- **Pure SNAP workload на GPU** (tungsten, Ta06A, C_SNAP single-species).
  SNAP force evaluation = 85% step budget; FP32 SNAP ниже ML fit noise
  floor (~3 orders of magnitude).
- **Hybrid SNAP+EAM workload где SNAP dominates cost но EAM precision
  matters** (alloy workflows, ML-covers-most + EAM-for-edge-cases runs).
  EAM остаётся в FP64 — сохраняет D-M6-8 1e-5 rel force ceiling.
- Production SNAP runs где M8 acceptance gate ≥ 20% speedup vs LAMMPS SNAP
  является целью.

**Когда НЕ использовать:**
- Pure EAM workload (без SNAP): используй `MixedFastBuild` — SNAP branch
  никогда не запускается, heterogeneous bookkeeping не даёт wins.
- Workload требует bitwise layout-invariance: используй
  `Fp64ReferenceBuild`.
- Dense-cutoff FP32 EAM envelope приемлем и throughput критически важен:
  используй `MixedFastBuild` (unified FP32 — чуть меньше код-пути
  heterogeneity overhead).

**Numerical guarantees (§D.13):**
- SNAP force vs LAMMPS: ≤ 1e-5 rel (**D-M8-8 dense-cutoff analog**)
- SNAP energy vs LAMMPS: ≤ 1e-7 rel
- EAM force vs LAMMPS: ≤ 1e-5 rel (inherited; EAM в FP64 — residual pure
  reduction-order roundoff, published ceiling at MixedFastBuild level)
- EAM energy vs LAMMPS: ≤ 1e-7 rel
- EAM virial (rel-to-max): ≤ 5e-6
- NVE drift per 1000 steps: ≤ 1e-5
- Layout-invariant determinism: observables only
- Bitwise same-run reproduce: exact

**Стоимость:** SNAP force в FP32 (throughput gain), EAM в FP64
(precision preserved), state + accumulators + reductions в FP64. Рабочая
combination для M8 acceptance gate.

**Обоснование существования:** master spec §D.11 запрещает per-kernel
precision overrides через runtime flags ("if potential==snap use_fp32").
Единственный approved path к heterogeneous precision — explicit новый
BuildFlavor, прошедший полную §D.17 7-step procedure. Формальный rationale:
[`docs/specs/potentials/mixed_fast_snap_only_rationale.md`](../specs/potentials/mixed_fast_snap_only_rationale.md).

**Activation status:**
- [x] CMake option + flavor function (T8.8, 2026-04-20)
- [x] §D.11/§D.12/§D.13/§D.14 SPEC entries (T8.8)
- [x] Threshold registry (T8.8 —
  `verify/thresholds/thresholds.yaml` `benchmarks.gpu_mixed_fast_snap_only`)
- [ ] Kernel split emission (T8.9 — requires T8.4b SNAP force body port)
- [ ] Slow-tier VerifyLab pass (T8.12 — **hard gate before M8 closure**)
- [ ] Architect + Validation Engineer joint review signoff (T8.8 PR thread)

До T8.9 + T8.12 флейвор configures cleanly но не emits heterogeneous kernel
paths. Используй `MixedFastBuild` для production SNAP runs на текущий
момент; переключайся на `MixedFastSnapOnlyBuild` после T8.12 gate close.

**ExecProfile compat:** `Reference` **REJECTED** (philosophy mismatch) /
`Production` (**canonical** — SNAP-dominant production target) /
`FastExperimental` (allowed но не канон).

---

### 5. `MixedFastAggressiveBuild` — opt-in research (Philosophy A)

**⚠ Research-only flavor. Не для production runs.**

**Когда использовать:**
- Ensemble screening где drift budget не важен (short runs < 10⁴ steps).
- Research ablation studies где нужен максимальный throughput.
- Explicit performance baseline для comparison с Philosophy B.

**Numerical guarantees (§D.13):**
- Forces vs LAMMPS: ≤ 1e-4 rel
- Energy vs LAMMPS: ≤ 1e-4 rel
- NVE drift per 1000 steps: **gate disabled** (no guarantee)
- Layout-invariant determinism: **gate disabled**
- Bitwise same-run reproduce: exact (same binary + hardware)

**Стоимость:** force kernels + accumulators в FP32 (единственная
difference от MixedFastBuild — accumulators down-cast до FP32). Типовой
GPU speedup 3-8% vs MixedFastBuild; economically marginal outside
extreme-throughput research scenarios.

**⚠⚠ NVE drift gate отключён.** Пользователи этого флейвора обязаны
**сами валидировать energy conservation** для своих specific systems через
shorter runs ИЛИ принять потенциальный drift как research cost.

**Output warning:** `tdmd_mixed_agg --version` печатает explicit warning
о disabled gates.

**ExecProfile compat:** `Reference` REJECTED / `Production` (warning —
NVE gates disabled) / `FastExperimental` (canonical).

---

### 6. `Fp32ExperimentalBuild` — extreme single-precision

**⚠⚠⚠ Extreme research-only. НЕ для научной публикации.**

**Когда использовать:**
- Исключительно research throughput exploration (FP32 memory bandwidth).
- ML-training with TDMD as dataset generator где noise is tolerable.
- Performance upper-bound reference для TDMD throughput analysis.

**Numerical guarantees (§D.13):**
- Forces vs LAMMPS: ≤ 1e-3 rel
- Energy vs LAMMPS: ≤ 1e-4 rel
- NVE drift per 1000 steps: **gate disabled**
- Layout-invariant determinism: **gate disabled**
- Bitwise same-run reproduce: exact

**Стоимость:** всё в FP32 включая state (position deltas!). На GPU ~4x
throughput vs MixedFastBuild теоретически, но для большинства MD
workloads memory-bound character limits wins до ~1.5-2x.

**⚠⚠⚠ Position delta accuracy compromised.** Catastrophic cancellation
в pair-distance subtraction возможна при больших box sizes. Не использовать
для runs с box > 100 Å side length.

**ExecProfile compat:** `Reference` REJECTED / `Production` REJECTED /
`FastExperimental` (canonical — единственный разрешённый режим).

---

## Как выбрать правильный флейвор: quick matrix

| Workload | Recommended flavor |
|---|---|
| EAM/pair production runs | `MixedFastBuild` |
| Pure SNAP production (tungsten, Ta06A) | `MixedFastSnapOnlyBuild` (post-T8.12) |
| SNAP+EAM hybrid where EAM precision matters | `MixedFastSnapOnlyBuild` (post-T8.12) |
| SNAP+EAM hybrid where both can use FP32 | `MixedFastBuild` |
| Научная публикация FP64 | `Fp64ProductionBuild` |
| Bitwise oracle для differential tests | `Fp64ReferenceBuild` |
| LAMMPS-comparison CI | `Fp64ReferenceBuild` |
| Long NVE (10⁶+ steps) publication-grade | `Fp64ProductionBuild` |
| Ensemble screening (short runs) | `MixedFastAggressiveBuild` ⚠ |
| Throughput research / ML dataset generation | `Fp32ExperimentalBuild` ⚠⚠ |

---

## Как configure + build

```bash
# Example: MixedFastBuild — default production throughput target
cmake -B build-mixed -DTDMD_BUILD_FLAVOR=MixedFastBuild .
cmake --build build-mixed

# Example: MixedFastSnapOnlyBuild — SNAP-dominant (post-T8.12)
cmake -B build-snap-only -DTDMD_BUILD_FLAVOR=MixedFastSnapOnlyBuild .
cmake --build build-snap-only

# Example: Fp64ReferenceBuild — bitwise oracle
cmake -B build-ref -DTDMD_BUILD_FLAVOR=Fp64ReferenceBuild .
cmake --build build-ref
```

`cmake --preset default` выбирает `Fp64ReferenceBuild` по умолчанию. Для
production переключайся на `MixedFastBuild`:

```bash
cmake --preset mixed-fast
```

Active flavor всегда печатается при run:

```
$ tdmd --version
TDMD v1.0.0-alpha1
Build flavor: MixedFastSnapOnlyBuild (Philosophy B-het: SNAP=FP32, EAM=FP64)
Numerical guarantees: see docs/specs/verify/thresholds.yaml
```

---

## См. также

- Master spec `TDMD_Engineering_Spec.md` §7 (BuildFlavor × ExecProfile),
  §D (полная precision policy), §D.13 (acceptance thresholds),
  §D.17 (процедура добавления нового BuildFlavor).
- `docs/specs/potentials/mixed_fast_snap_only_rationale.md` — формальный
  rationale для `MixedFastSnapOnlyBuild` (§D.17 step 1 artifact).
- `docs/specs/verify/SPEC.md` §3 — threshold registry.
- `verify/thresholds/thresholds.yaml` — single source of truth для всех
  numerical tolerances.
