# potentials/SPEC.md

**Module:** `potentials/`
**Status:** master module spec
**Parent:** `TDMD Engineering Spec v2.1` §3.2, §5.1, §12.5
**Last updated:** 2026-04-20 (T8.3 — §6 SNAP body authored; interface
contract + force-evaluation algorithm + LAMMPS USER-SNAP attribution chain +
MixedFastSnapOnly placeholder + validation gate matrix D-M8-7/D-M8-8)

---

## 1. Purpose и scope

### 1.1. Что делает модуль

`potentials/` — модуль **физики взаимодействия**. Единственная цель: для данной конфигурации атомов вычислить **силы, энергию и вириал**. Ничего больше.

Модуль содержит:

1. Абстрактный интерфейс `PotentialModel`;
2. Конкретные реализации (Morse, EAM, MEAM, SNAP, PACE, MLIAP);
3. Parameter file parsers (EAM .alloy/.fs, SNAP coefficients);
4. Numeric reference path (CPU, FP64) для каждого потенциала;
5. GPU kernels (Mixed/Fast builds, позже);
6. Validation harness для differential vs LAMMPS.

### 1.2. Scope: что НЕ делает

- **не владеет атомами** (читает via `ForceRequest`);
- **не меняет позиции** — записывает только `fx, fy, fz`;
- **не строит neighbor lists** (потребляет готовый);
- **не решает когда считать** — вызывается scheduler'ом;
- **не делает migration** — перемещения атомов делает `integrator/` и `neighbor/`;
- **не знает про zones** в смысле scheduler'а — получает zone_ids как hint для filtering, но работает на per-atom basis.

### 1.3. Почему потенциалы — центральная ниша TDMD

Как обсуждалось в §3 мастер-специи: **TDMD существует потому что есть дорогие локальные потенциалы**. Если Morse/LJ было бы единственной целевой аудиторией, LAMMPS-SD победил бы нас по всем параметрам. Поэтому потенциалы для нас — не "один из модулей", а **главный customer** TD-метода.

Приоритеты реализации (wave 1 → wave 4):

**Wave 1 (v1 M1-M8):** `Morse` (reference), `EAM/alloy`, `EAM/FS`, `SNAP`.
**Wave 2 (v1.5):** `MEAM`, `MEAM-like angular`.
**Wave 3 (post-v1):** `PACE`, `MLIAP`.
**Wave 4 (v2+):** `ReaxFF` как новый track.

---

## 2. Public interface

### 2.1. Базовые типы (из мастер-специи §12.5)

```cpp
namespace tdmd {

enum class PotentialKind {
    Pair,                 // Morse, LJ — простые pairwise
    ManyBodyLocal,        // EAM, EAM/FS, MEAM — локальные multi-body
    Descriptor,           // SNAP, MLIAP, PACE — ML-based, descriptor + regression
    Reactive,             // ReaxFF, COMB (v2+)
    Hybrid                // classical + ML correction (v2+)
};

struct ComputeMask {
    bool  force;
    bool  energy;
    bool  virial;
};

struct ForceRequest {
    const AtomSoA*          atoms;
    const NeighborList*     neigh;
    const Box*              box;
    const ZoneId*           zone_ids;       // optional filter; nullptr = all atoms
    uint32_t                n_zones;
    ComputeMask             mask;
};

struct ForceResult {
    double  potential_energy;     // total PE, sum over computed atoms
    double  virial[6];            // xx, yy, zz, xy, xz, yz (Voigt)
    // forces written in-place into atoms->fx/fy/fz
};

class PotentialModel {
public:
    virtual std::string     name() const = 0;
    virtual PotentialKind   kind() const = 0;
    virtual double          cutoff() const = 0;
    virtual double          effective_skin() const = 0;   // recommended skin for this potential
    virtual bool            is_local() const = 0;         // TD applicability: false → no TD
    virtual void            compute(const ForceRequest&, ForceResult&) = 0;

    // Calibration support (для perfmodel):
    virtual double          estimated_flops_per_atom() const = 0;

    // Introspection для explain:
    virtual std::string     parameter_summary() const = 0;
    virtual uint64_t        parameter_checksum() const = 0;  // for repro bundle

    virtual ~PotentialModel() = default;
};

} // namespace tdmd
```

### 2.2. Регистрация и фабрика

```cpp
class PotentialFactory {
public:
    static std::unique_ptr<PotentialModel>  create(
        const std::string& style,         // "morse" | "eam/alloy" | "snap" | ...
        const PotentialConfig& config);

    static std::vector<std::string>  available_styles();
};
```

Пользователь в `tdmd.yaml` указывает `style`; фабрика создаёт соответствующий класс. Registry extensible через plugin system (v2+).

### 2.3. Concrete classes (v1)

```cpp
class MorsePotential          final : public PotentialModel { /*...*/ };
class EamAlloyPotential       final : public PotentialModel { /*...*/ };
class EamFsPotential          final : public PotentialModel { /*...*/ };
class SnapPotential           final : public PotentialModel { /*...*/ };

// v1.5:
class MeamPotential           final : public PotentialModel { /*...*/ };

// v2+:
class PacePotential           final : public PotentialModel { /*...*/ };
class MliapPotential          final : public PotentialModel { /*...*/ };
class ReaxffPotential         final : public PotentialModel { /*...*/ };
```

### 2.4. Cutoff and smoothing policy (cross-potential)

Все потенциалы в TDMD **обязаны** реализовывать одну из следующих cutoff treatment strategies. Это **не free choice** — выбор влияет на energy conservation, force continuity, и совместимость с LAMMPS.

#### 2.4.1. Три canonical treatments

**Strategy A — Hard cutoff (только для reference math):**
```
E(r) = E_raw(r) * step(r_c - r)
F(r) = F_raw(r) * step(r_c - r)
```
Discontinuous в `r = r_c`. Дает drift в NVE runs. **Используется только** для analytic reference (2-атом test в T0) и unit tests. **Запрещено в production**.

**Strategy B — Shifted-energy (LAMMPS default для pair_style many):**
```
E(r) = E_raw(r) - E_raw(r_c)   if r < r_c, else 0
F(r) = F_raw(r)                  if r < r_c, else 0
```
Energy continuous, force **не** continuous в `r_c` (jump `F_raw(r_c)`). Дает slight NVE drift proportional к `|F_raw(r_c)|`.

**Strategy C — Shifted-force (TDMD default для all short-range):**
```
E(r) = E_raw(r) - E_raw(r_c) - (r - r_c) · F_raw(r_c)    if r < r_c, else 0
F(r) = F_raw(r) - F_raw(r_c)                              if r < r_c, else 0
```
Obe energy и force continuous в `r_c`. **Best energy conservation** в NVE. Рекомендуется для всех новых potential implementations.

**Strategy D — Explicit smoothing function (для ML potentials, SNAP, PACE):**
```
E(r) = E_raw(r) · smooth(r, r_c, r_cin)
F = - dE/dr
```
где `smooth(r, r_c, r_cin)` — cosine / polynomial с support `[r_cin, r_c]`:
```
smooth(r) = 1                                          if r < r_cin
          = 0.5 * (1 + cos(π * (r - r_cin)/(r_c - r_cin)))   if r_cin ≤ r < r_c
          = 0                                           if r ≥ r_c
```
`r_cin` — inner cutoff (typical `r_cin = 0.8 · r_c`). Smooth transition zone. Best accuracy для ML descriptors чувствительных к cutoff artifacts.

#### 2.4.2. Matrix: потенциал → strategy

| Potential | Default strategy | Rationale |
|---|---|---|
| Morse (T0 reference) | **A** (analytic) + **C** (production) | A для unit tests с аналитической формулой, C для canonical benchmarks |
| LJ | **C** | Standard shifted-force, matches LAMMPS `pair_style lj/cut/smooth` |
| EAM/alloy, EAM/FS | **C** (pair part) + natural cutoff ρ(r) и φ(r) | EAM tables уже имеют `F(ρ(r_c)) = 0` и `φ(r_c) = 0` build-in |
| MEAM | **D** (explicit cosine smoothing) | Published MEAM papers specify cosine cutoff; must match для LAMMPS diff |
| SNAP | **D** (cosine) | Published SNAP convention |
| PACE | **D** (specific ACE smoothing) | ACE reference implementation prescribes exact form |
| MLIAP | Per-model (user-configurable) | Depends on ML architecture |

**Critical:** для differential test vs LAMMPS наш treatment должен **точно совпадать** с LAMMPS `pair_style` convention. Проверяется в VerifyLab differential tests.

#### 2.4.3. Numerical considerations near cutoff

Даже при правильном smoothing, pairs с `r` очень близко к `r_c` вносят numerical noise:
- `F_raw(r_c) - F_raw(r_c)` в Strategy C — catastrophic cancellation;
- `smooth(r ≈ r_c) ≈ 0` в Strategy D — multiplication на near-zero number.

**Мitigation:**
1. Inner loop проверяет `r² ≤ cutoff_sq` (squared distances avoids sqrt);
2. В Strategy C: precompute `F_c_shift = F_raw(r_c)` один раз, не в inner loop;
3. В Strategy D: предвычислить `smooth()` и `d_smooth/dr` if tables applicable (избегать cos в inner loop для Mixed precision);
4. Для ML potentials в float precision: test что `smooth(0.999 * r_c)` remains representable (not underflow to denormal).

#### 2.4.4. API integration

Каждый `PotentialModel::compute` возвращает forces already с applied cutoff treatment. Strategy — internal detail потенциала, выбирается при construction через `PotentialConfig`:

```yaml
potential:
  style: morse
  cutoff_treatment: shifted_force   # A|shifted_energy|shifted_force|smoothed
  params: ...
```

Если user не указал — потенциал использует default из матрицы §2.4.2. Warning emitted при config parse если strategy не default.

#### 2.4.5. Validation

Каждое potential реализует tests для своей cutoff treatment:
- **Analytical check:** at `r = r_c - ε`, `r = r_c`, `r = r_c + ε` — force continuous в Strategy C, discontinuous в A и B;
- **Energy continuity:** `E(r_c - ε) - E(r_c + ε) ≤ threshold` (Strategy C: `10⁻¹²`, Strategy B: `|F(r_c)| · ε`);
- **NVE drift:** long run — drift proportional к `|F(r_c)|` в Strategy B, near zero в Strategy C.

Thresholds в `verify/thresholds.yaml`:

```yaml
tolerances:
  cutoff_treatment:
    shifted_force_energy_continuity: 1.0e-12
    shifted_energy_force_jump_acceptable: 1.0e-3   # relative to max |F|
    smoothed_derivative_continuity: 1.0e-10
```

---

## 3. Morse (reference)

### 3.1. Форма потенциала

```
E_pair(r)  =  D · [1 - exp(-α·(r - r_0))]² - D
```

Где:
- `D` — depth of well (eV);
- `α` — width parameter (1/Å);
- `r_0` — equilibrium distance (Å).

Cutoff с smoothing (опционально):
```
E(r) = E_pair(r) - E_pair(r_c)  -  (r - r_c) · dE_pair/dr|_{r_c}    if r < r_c
     = 0                                                                if r ≥ r_c
```

Это "shifted-force" smoothing — делает force continuous в `r_c` (важно для energy conservation).

### 3.2. Force formula

```
F_ij = -dE/dr · r_hat = -2·D·α·[1 - exp(-α·(r - r_0))]·exp(-α·(r - r_0)) · (r_ij / r)
```

### 3.3. Parameter format

YAML:
```yaml
potential:
  style: morse
  params:
    - species_pair: [Al, Al]
      D: 0.2703           # eV
      alpha: 1.1646       # 1/Å
      r0: 3.253           # Å
      cutoff: 8.0         # Å
    # multi-species: дополнительные pair entries
```

### 3.4. Implementation outline

```
class MorsePotential:
    private:
        vector<MorseParams>  params_per_pair;   // indexed by (type_i, type_j)
        double               r_c;

    compute(request, result):
        for atom_i in request.atoms (filtered by zones):
            for j in request.neigh.get_neighbors(i):
                dx, dy, dz = periodic_delta(atoms, i, j)
                r² = dx² + dy² + dz²
                if r² > r_c²:  continue
                r = sqrt(r²)
                p = params_per_pair[type_i, type_j]
                exp_term = exp(-p.alpha · (r - p.r0))
                factor = 2·p.D·p.alpha · (1 - exp_term) · exp_term / r

                fx = factor · dx
                fy = factor · dy
                fz = factor · dz

                atoms.fx[i] += fx
                atoms.fy[i] += fy
                atoms.fz[i] += fz
                if newton and j is owned:
                    atoms.fx[j] -= fx
                    atoms.fy[j] -= fy
                    atoms.fz[j] -= fz

                if request.mask.energy:
                    result.potential_energy += p.D · (1 - exp_term)² - p.D
                if request.mask.virial:
                    result.virial[0] += fx · dx  # xx
                    result.virial[1] += fy · dy  # yy
                    # ... xy, xz, yz
```

### 3.5. Validation

- **Analytic test:** 2 atoms at known `r`, force match closed-form formula.
- **Diff vs LAMMPS:** `pair_style morse` at matching cutoff + shift — forces identical to FP64 within `1e-12`.
- **Energy conservation:** NVE run 10⁴ steps, ΔE/E < `1e-6`.

---

## 4. EAM (alloy и Finnis-Sinclair)

### 4.1. Форма

EAM: potential energy split into embedding term и pairwise term:

```
E_i = F_{α(i)}(ρ_i) + (1/2) · Σ_j  φ_{α(i),α(j)}(r_ij)
ρ_i = Σ_{j≠i}  ρ_{α(j)}(r_ij)
```

Где:
- `F_α` — embedding function для species α (eV);
- `ρ_α` — electron density contribution function от species α;
- `φ_{αβ}` — pair interaction function для species pair (α, β);
- `ρ_i` — total electron density at atom i.

### 4.2. EAM/alloy vs EAM/FS

Различие — только в **формате parameter file**:

- **EAM/alloy** (`.eam.alloy`): table-format; single `ρ_α(r)` per species;
- **EAM/FS** (`.eam.fs`, Finnis-Sinclair): `ρ_{αβ}(r)` — density contribution зависит от обеих species.

Математика force evaluation одинакова для обоих, параметризация разная.

### 4.3. Force evaluation algorithm

Two-pass algorithm:

**Pass 1 — compute densities:**
```
for i: ρ[i] = 0
for each pair (i, j) within cutoff:
    ρ[i] += ρ_funcs[type_j].eval(r_ij)
    if newton: ρ[j] += ρ_funcs[type_i].eval(r_ij)
```

**Pass 2 — compute forces:**
```
for i:
    dF_i = F_funcs[type_i].derivative(ρ[i])   # dF/dρ at atom i
for each pair (i, j):
    r = distance(i, j)
    # Embedding contribution:
    dρ_j_dr = ρ_funcs[type_j].derivative(r)
    dρ_i_dr = ρ_funcs[type_i].derivative(r)
    F_embed = -(dF_i · dρ_j_dr + dF_j · dρ_i_dr) / r

    # Pair contribution:
    dφ_dr = φ_funcs[type_i, type_j].derivative(r) / r

    F_total = (F_embed + dφ_dr) · r_hat
    apply F to i, j (with newton)
```

### 4.4. Interpolation of tabulated functions

EAM parameters приходят как **tabulated** `F(ρ)`, `ρ(r)`, `φ(r)` в `.alloy` / `.fs` файле. Для force evaluation нужны `F'(ρ)`, `ρ'(r)`, `φ'(r)`.

**Interpolation scheme:** natural cubic spline с derivative extraction.

```
class TabulatedFunction:
    private:
        vector<double>  x_grid;        // usually uniform
        vector<double>  y_values;
        vector<double>  y_derivatives; // precomputed at init

    eval(x):  # cubic spline eval
    derivative(x):  # cubic spline derivative (analytically computed from spline coeffs)
```

**Match LAMMPS:** LAMMPS использует specific interpolation в `pair_eam.cpp`. Для differential test bitwise match мы должны использовать **идентичную** формулу. Reference implementation в `eam_alloy_potential.cpp` копируется 1:1 из LAMMPS (с credit + license).

### 4.5. Parameter file format (LAMMPS-compatible)

**EAM/alloy (`.eam.alloy`):**
```
<comment line 1>
<comment line 2>
<comment line 3>
<N_species> <Al> <Ni> <Cu>               # species names
<N_rho> <d_rho> <N_r> <d_r> <r_cutoff>   # grid parameters
<Z_Al> <mass_Al> <a_Al> <lattice_Al>     # per-species info
<F_Al values ...>                         # N_rho values: embedding function
<ρ_Al values ...>                         # N_r values: electron density
<Z_Ni> <mass_Ni> ...
<φ values ...>                            # N_r · N_species · (N_species + 1) / 2
```

`PotentialFactory` parses это файл → `EamAlloyPotential`.

### 4.6. Validation

- **Analytic test:** single atom, ρ = 0 → F = F(0), dF = F'(0);
- **Diff vs LAMMPS:** canonical potential file (e.g. `Al99.eam.alloy` Mendelev 2008) → forces match to `1e-10`.
- **Multi-species:** Ni-Al alloy FCC, both species populated, diff vs LAMMPS `pair_style eam/alloy`.

---

## 5. MEAM (angular extension)

### 5.1. Форма

MEAM extends EAM с **angular dependence**:

```
ρ_i = sum of ρ_(0), ρ_(1), ρ_(2), ρ_(3)
        (partial densities включая angular moments)

ρ_(0)_i = Σ_j  ρ^(0)(r_ij)                                    # spherical, как EAM
ρ_(1)_i²  = Σ_α  [Σ_j  ρ^(1)(r_ij) · x_ij^α / r_ij]²         # dipole
ρ_(2)_i² = ... (3×3 matrix, trace subtracted)                 # quadrupole
ρ_(3)_i² = ... (3×3×3 tensor)                                 # octupole
```

Это делает MEAM существенно дороже EAM (3-4× больше FLOPs).

### 5.2. Почему MEAM — sweet spot для TDMD

MEAM — dense local computation, с halo dependency через angular moments. В SD это создаёт overhead (halo нужно передать 4 different density channels). В TD **отсутствует** этот overhead, потому что zone считает своё full MEAM за один локальный шаг, и темпоральный pipeline передаёт только финальное state.

Это — **main TDMD pitch для металлургов**: MEAM at scale без deteriorating scaling.

### 5.3. Implementation

Скелет (wave 2, M9-M10):

```
class MeamPotential:
    private:
        // reference implementation из LAMMPS meam package
        vector<MeamParams>  params_per_triplet;  // потенциал зависит от (i, j, k)

    compute(request, result):
        # 3-pass algorithm:
        # Pass 1: partial densities (spherical ρ^(0))
        # Pass 2: angular contributions (ρ^(1), ρ^(2), ρ^(3))
        # Pass 3: forces (complex multi-term formula)
```

Конкретная numeric implementation — масштабный проект, ~5000 lines C++. Отложен на wave 2. В v1 M8 — только skeleton и stub.

### 5.4. Validation

- Diff vs LAMMPS `pair_style meam/c` на Si.meam или Fe.meam test cases;
- Angular force terms: single triplet of atoms → force match to analytic (если есть closed form) или LAMMPS numerical.

---

## 6. SNAP (ML-based, Wave 1)

### 6.1. Форма

SNAP (Spectral Neighbor Analysis Potential, Thompson et al. J. Comp. Phys.
2015) — machine-learned potential, где атомная энергия представляется как
linear regression на bispectrum descriptors of the local neighbor density:

```
E_i = β_0^{(α_i)} + Σ_k  β_k^{(α_i)} · B_k(r_i)         (linear, default)

или (quadratic extension, `quadraticflag = 1` в LAMMPS):

E_i = β_0 + Σ_k β_k · B_k + (1/2) · Σ_{k ≤ l} β_{kl} · B_k · B_l
```

Где:

- `B_k(r_i)` — k-ый bispectrum component для окружения атома `i` внутри
  cutoff `r_c = rcutfac · Σ_j R_j`, где `R_j` — per-species radius;
- `β_k^{(α_i)}` — ML-learned coefficients для species `α_i` (из `.snapcoeff`);
- `β_0^{(α_i)}` — per-species offset;
- `B_k` строится из sum expansion of spherical harmonics `Y_{l,m}(r̂_ij)`
  × radial basis `f(r_ij)` × Clebsch-Gordan coefficients `C^{j1,j2,j}_{m1,m2,m}`.

Размер basis: `k_max = (J_max+1)(J_max+2)(J_max+3)/6` — typical
values: `J_max = 4` → `k = 31`, `J_max = 8` → `k = 165`.

### 6.2. Why SNAP in Wave 1

В §3.2 master spec: **TDMD's proof-of-value niche**. SNAP — flagship ML
potential, широко используемый для металлов (W, Ta, Zr, Nb, Mo) и доступен в
LAMMPS для differential test. Если TDMD не обгоняет LAMMPS на SNAP — проект
не имеет raison d'être. M8 acceptance gate формализует это (мастер-спец §14):
«TDMD либо обгоняет LAMMPS на ≥ 20% на ≥ 8 ranks, либо честно документирует
почему нет».

### 6.3. Cost characteristics

SNAP — **dramatically expensive** compared to EAM:

| Potential | FLOP / neighbor pair | FLOP / atom (N_nbr ≈ 50) |
|---|---|---|
| Morse | ~20 | ~1 000 |
| EAM/alloy | ~100 | ~5 000 |
| SNAP `J_max=4` | ~5 000 | ~250 000 |
| SNAP `J_max=8` | ~50 000 | ~2 500 000 |

Это именно та ниша, где TD даёт драматический выигрыш: `T_compute ≫ T_comm`
absolutely, так что perfect overlap (см. perfmodel/SPEC §3.7 saturation
tables: `N_min_saturation` для SNAP на A100 ≈ 1 000, для EAM ≈ 5 000 — SNAP
**saturates GPU at 5× fewer atoms**).

### 6.4. Interface contract

```cpp
namespace tdmd::potentials {

// LAMMPS-compatible hyperparameters (из *.snapparam).
struct SnapParams {
    int     twojmax;           // 2·J_max (even integer; J_max typical 4, 6, 8)
    double  rcutfac;           // global cutoff scaling factor
    double  rfac0;             // inner-to-outer radial basis ratio (default 0.99363)
    double  rmin0;             // minimum radial basis (typical 0.0)
    bool    switchflag;        // cosine smooth turned on (Strategy D §2.4)
    bool    bzeroflag;         // subtract B_k^{empty} reference (LAMMPS default: on)
    bool    quadraticflag;     // linear (0) vs quadratic (1) SNAP
    bool    chemflag;          // multi-species chem SNAP (M9+; false in v1)
    // Additional fields: bnormflag, wselfallflag, switchinnerflag —
    // mapped 1:1 из LAMMPS ComputeSNA constructor per §6.6.
};

// LAMMPS-compatible per-species data (из *.snapcoeff).
struct SnapSpecies {
    std::string          name;            // "W", "Ta", ...
    double               radius_elem;     // per-species R_j
    double               weight_elem;     // per-species w_j (default 1.0)
    std::vector<double>  beta;            // β_0 + k_max linear coeffs
                                          // (+ k_max·(k_max+1)/2 quadratic coeffs если quadraticflag)
};

// Full SNAP parameter set.
struct SnapData {
    SnapParams                params;
    std::vector<SnapSpecies>  species;      // size == n_species
    // Derived / cached:
    int                       k_max;        // number of bispectrum components
    std::vector<double>       rcut_sq_ab;   // pairwise squared cutoffs (symmetric n×n)
    uint64_t                  checksum;     // parameter_checksum() payload
};

class SnapPotential final : public PotentialModel {
public:
    SnapPotential(SnapData data, const PotentialConfig& config);

    std::string     name()    const override { return "snap"; }
    PotentialKind   kind()    const override { return PotentialKind::Descriptor; }
    double          cutoff()  const override;    // max over all species-pair rcuts
    double          effective_skin() const override;     // recommended skin
    bool            is_local() const override { return true; }

    void            compute(const ForceRequest&, ForceResult&) override;

    double          estimated_flops_per_atom() const override;    // derived from k_max
    std::string     parameter_summary() const override;
    uint64_t        parameter_checksum() const override;

private:
    SnapData                         data_;
    // Canonical LAMMPS-port scratch buffers (см. §6.5):
    //   bispectrum_components_[i][k]
    //   beta_times_B_[i]
    //   force_contributions_[i]  // per-atom output accumulator
    // Concrete layout mandated by D-M8-7 byte-exactness: must be identical
    // to LAMMPS pair_snap.cpp scratch ordering for per-atom force match.
};

}  // namespace tdmd::potentials
```

**Invariants:**

1. `SnapPotential::is_local() == true` (SNAP is strictly local despite being
   ML — cutoff bounded, no global coupling). TD applicability gated.
2. `cutoff()` returns the species-pair-maximum effective cutoff
   `max_{α,β} (rcutfac · (R_α + R_β))` — this feeds neighbor/SPEC §3.2 skin
   computation.
3. `compute()` MUST zero-out `fx/fy/fz` before accumulation per §7.2
   force-zero-out invariant (cross-potential contract).
4. D-M8-7 byte-exactness requires internal scratch allocation and reduction
   order identical to LAMMPS `ComputeSNA::compute_sna_atom` / `pair_snap.cpp`
   — see §6.5 implementation strategy.

### 6.5. Force evaluation algorithm

SNAP force evaluation **is ported from LAMMPS USER-SNAP** (explicit attribution
per LICENSE + source header). Reimplementation from scratch is rejected (risk
of subtle bispectrum basis function bugs too high; canonical reference exists
and is well-validated).

**Three-pass algorithm** (mirroring `pair_snap.cpp` + `sna.cpp` upstream):

```
Pass 1 — compute bispectrum B_k per atom:
  for i in filter:
      B[i] := ComputeSNA::compute_sna_atom(i, neighbors_of_i, data.params)
            // includes:
            //   (a) radial basis f_cut(r_ij) via switching function §2.4 Strategy D;
            //   (b) spherical harmonics Y_{j,m1,m2}(r̂_ij);
            //   (c) 4D array U_{j,m1,m2,iatom} accumulation;
            //   (d) bispectrum B = Σ U × U × C (Clebsch-Gordan contraction);
            //   (e) subtract B_k^{empty} reference if bzeroflag.

Pass 2 — compute energy and per-atom β·B:
  for i in filter:
      E[i]       := β_0[α_i] + Σ_k β_k[α_i] · B_k[i]
                    (+ quadratic term if quadraticflag)
      beta_B[i]  := cached linear combination for force pass

Pass 3 — compute forces (inverse chain rule through bispectrum):
  for each pair (i, j) within r_cut:
      dB_k/dr_j for k = 0..k_max-1    // most expensive loop
      F_j += -Σ_k beta_B[i]_k · dB_k/dr_j
      F_i += +(Newton third law contribution)
```

**Byte-exactness contract (D-M8-7).** TDMD CPU FP64 SNAP MUST match LAMMPS FP64
SNAP to ≤ 1e-12 rel per-atom force on T6 fixture (see `verify/SPEC.md §4.7`).
This dictates **scratch array layout + reduction order** be inherited from
LAMMPS verbatim — no "cleaner" reimplementation allowed for M8. Post-M8
micro-optimizations (kernel fusion, SoA repacking) may diverge IFF they pass
a regenerated byte-exact gate. Auto-reject (master spec §11.4): any
"optimization" that abandons the byte-exact chain without an explicit
thresholds-registry entry justifying the divergence.

**License chain.** LAMMPS USER-SNAP is GPLv2. TDMD integrates the port under
the GPL compatibility clause per LICENSE; source headers include SPDX tag
`SPDX-License-Identifier: GPL-2.0-or-later` и attribution block citing
`src/SNAP/sna.cpp` + `src/SNAP/pair_snap.cpp` + Thompson et al. JCP 2015 +
Wood & Thompson arXiv:1702.07042 (T6 fixture authors).

### 6.6. Parameter file format (LAMMPS-compatible)

TDMD consumes LAMMPS-native SNAP files unchanged. Three artefacts per potential:

**`.snap` include file (entry point):**

```lammps
# DATE: 2017-02-20 CONTRIBUTOR: Mitchell Wood
variable zblcutinner equal 4
variable zblcutouter equal 4.8
pair_style hybrid/overlay &
  zbl ${zblcutinner} ${zblcutouter} &
  snap
pair_coeff 1 1 zbl 74 74
pair_coeff * * snap W_2940_2017_2.snapcoeff W_2940_2017_2.snapparam W
```

TDMD's `PotentialFactory::create("snap", config)` parser extracts the two
filename arguments of the `pair_coeff * * snap ...` line + the species map
(trailing `W` above) and loads the two sidecar files.

**`.snapcoeff` (per-species coefficient file):**

```
# comment
<n_species> <k_max_plus_1>
<species_name> <radius_elem> <weight_elem>
<β_0>
<β_1>
...
<β_{k_max}>
<next species...>
```

**`.snapparam` (hyperparameters):**

```
# comment
rcutfac        4.67637
twojmax        6
rfac0          0.99363
rmin0          0.0
switchflag     1
bzeroflag      1
quadraticflag  0
```

Parser: `parse_snap_files(coeff_path, param_path, species_map) → SnapData`
— analogous to `parse_eam_alloy` / `parse_eam_fs` (§4.5). Lives in
`src/potentials/snap_file.cpp` (T8.4 scope).

### 6.7. Precision policy и MixedFastSnapOnlyBuild

Per §8 (numeric precision) + §D.11 (one-precision rule + mixed policy):

- **Fp64Reference / Fp64Production:** SNAP runs in FP64 throughout (bispectrum
  accumulation, β·B inner product, force kernel). Bit-exact vs LAMMPS FP64.
- **MixedFastBuild (default mixed):** SNAP inherits `MixedPrecision<ForceReal=float>`
  — whole force path runs in FP32, including bispectrum. Expected deviation
  vs Fp64Reference: ≤ 1e-5 rel force / ≤ 1e-7 rel PE per D-M6-8 dense-cutoff
  analog (D-M8-8 in m8 exec pack). Motivation: SNAP ML coefficients are fit
  against DFT energies with RMSE ≈ 1e-3 eV/atom >> FP32 ULP — FP32 is
  numerically appropriate for the physics.
- **MixedFastSnapOnlyBuild (new at M8 T8.8):** heterogeneous precision —
  SNAP kernels в FP32, EAM / pair kernels в FP64, State в FP64 (matches
  MixedFastBuild state policy). Only approved per-kernel precision mix (see
  §8.7). Введение этого BuildFlavor проходит через формальную §D.17 7-step
  procedure (T8.8 scope); до T8.8 landing этот flavor **не доступен**.

Auto-reject (master spec §D.11):

- SNAP в FP32 под MixedFastBuild без explicit D-M8-8 threshold registry
  entry;
- Any introduction of per-kernel precision dispatch (`if (potential == snap)
  use_fp32`) that does NOT go through §D.17 BuildFlavor path.

### 6.8. GPU kernel strategy (wave 1.5)

Per §9.1 kernel taxonomy, SNAP decomposes into three GPU kernels:

1. `snap_bispectrum_kernel` — Pass 1; dominant cost. Launch grid = one block
   per atom или per (atom, J_max)-shard depending on `k_max`. Shared-memory
   cache for Clebsch-Gordan coefficients (constant per-potential).
2. `snap_energy_kernel` — Pass 2; thin kernel, per-atom β·B dot product.
3. `snap_force_kernel` — Pass 3; dominant over Pass 1 when `J_max ≥ 6`.
   Atom-parallel force accumulation via canonical gather-to-single-block
   Kahan per D-M6-7 в Reference profile; atomic adds allowed в Production.

D-M8-7 byte-exact extends D-M6-7 SNAP: GPU FP64 SNAP ≤ 1e-12 rel vs CPU FP64
SNAP on T6 fixture (`W_2940_2017_2.snap`, 2048-atom BCC W). Implementation
lands T8.6 (GPU) + T8.7 (byte-exact gate) per m8 exec pack.

Per §9.4 NVTX policy, kernels are instrumented with explicit ranges:

```
SnapPotential::bispectrum_kernel
SnapPotential::energy_kernel
SnapPotential::force_kernel
SnapPotential::rebuild_species_tables   // init-time, не per-step
```

Performance target: SNAP throughput (atoms/sec) ≥ LAMMPS `pair_style snap` +
`package gpu` on the M8 reference hardware (RTX 5080). M8 acceptance gate
(master spec §14): ≥ 20% speedup on ≥ 8 ranks vs LAMMPS — cloud-burst-gated
per D-M8-5 (see m8 exec pack §3).

### 6.9. Validation

Full validation chain per m8 exec pack §4:

- **T8.4 unit:** CPU FP64 SnapPotential analytic single-atom single-neighbor
  force check + `parse_snap_files` fixture load (covers `.snapcoeff`,
  `.snapparam`, `.snap` include wrapper).
- **T8.5 CPU differential (D-M8-7 byte-exact):** `t6_snap_cpu_vs_lammps`
  benchmark — 2048-atom BCC W, `W_2940_2017_2.snap` + sidecars resolved via
  M1-landed LAMMPS submodule; per-atom force ≤ 1e-12 rel vs LAMMPS FP64
  `pair_style snap`; total PE ≤ 1e-12 rel.
- **T8.7 GPU byte-exact gate (D-M6-7 SNAP extension):** GPU FP64 SNAP
  ≤ 1e-12 rel per-atom force vs CPU FP64 SNAP.
- **T8.9 MixedFast threshold gate (D-M8-8 dense-cutoff analog for SNAP):**
  `MixedFastBuild` / `MixedFastSnapOnlyBuild` SNAP ≤ 1e-5 rel force / ≤ 1e-7
  rel PE vs Fp64Reference SNAP (10x margin на per-step; 1000-step global cap
  ≤ 1e-4 rel force / ≤ 1e-6 rel PE per NVE drift budget).
- **T8.10 T6 benchmark (`t6_snap_tungsten`):** single-rank Reference +
  Mixed single-subdomain; 2-rank Pattern 2 K=1 Reference byte-exact chain
  extension; 4-rank opportunistic; D-M7-10 chain (M3 ≡ M4 ≡ M5 ≡ M6 ≡ M7 ≡ M8
  P_space=N K=1 Reference thermo byte-exact) extension point.
- **T8.11 scaling probe (cloud-burst-gated):** TDMD vs LAMMPS на ≥ 8 ranks,
  артефакт в `verify/benchmarks/t6_snap_scaling/results_<date>.json`. Both
  "≥ 20% speedup achieved" and "honest documentation of why not" outcomes
  close the M8 acceptance gate (master spec §14; m8 exec pack D-M8-6).
- **T8.12 slow-tier VerifyLab pass:** full §D.17 7-step procedure gate for
  `MixedFastSnapOnlyBuild` — 1000-step drift + energy conservation + T4
  regression parity.

Threshold registry entries (anchor: `verify/thresholds.yaml`):

```yaml
# D-M8-7 (byte-exact SNAP):
snap_cpu_vs_lammps_force_per_atom_rel_max:  1.0e-12
snap_cpu_vs_lammps_total_pe_rel:             1.0e-12
snap_gpu_vs_cpu_fp64_force_per_atom_rel:    1.0e-12
snap_gpu_vs_cpu_fp64_total_pe_rel:           1.0e-12

# D-M8-8 (MixedFast dense-cutoff analog for SNAP):
snap_mixedfast_vs_fp64_force_per_atom_rel_max:  1.0e-5
snap_mixedfast_vs_fp64_total_pe_rel:             1.0e-7
# 1000-step NVE drift cap:
snap_mixedfast_1000step_force_rel_max:           1.0e-4
snap_mixedfast_1000step_pe_rel:                   1.0e-6
```

---

## 7. Integrator integration

### 7.1. Relationship

`potentials/` produces `fx, fy, fz`. `integrator/` consumes them, advances positions and velocities. They communicate через `AtomSoA` (no direct coupling).

Стандартный flow (Velocity Verlet):
```
integrator.pre_force(atoms, dt):    # half-kick + drift
    v += 0.5 · dt · f/m
    x += dt · v

potential.compute(request, result): # compute new forces
    populate atoms.fx/fy/fz

integrator.post_force(atoms, dt):   # half-kick
    v += 0.5 · dt · f/m
```

### 7.2. Force zero-out

**Потенциалы ДОЛЖНЫ обнулять `fx, fy, fz` перед accumulation**. Это инвариант. Запрещено полагаться, что "кто-то другой это сделает".

```
compute(request, result):
    for atom_i in filtered atoms:
        atoms.fx[i] = 0
        atoms.fy[i] = 0
        atoms.fz[i] = 0
    # ... then accumulate
```

### 7.3. Multiple potentials (hybrid)

v1 supports **один потенциал за раз**. Hybrid (классический + ML correction) — wave 4.

---

## 8. Numeric policy (precision)

> **Источник истины для precision policy — мастер-спец Приложение D.** Эта секция — module-specific применение общих правил. Любые изменения общей политики — в Приложение D с SPEC delta, не здесь.

### 8.1. BuildFlavor awareness

`PotentialModel` templated по `NumericConfig` (5 вариантов, см. мастер-спец §D.2):

```cpp
template<typename NumericConfig>
class MorsePotentialImpl : public PotentialModel {
    using StateReal = typename NumericConfig::StateReal;      // double, double, double, double, float
    using ForceReal = typename NumericConfig::ForceReal;      // double, double, float,  float,  float
    using AccumReal = typename NumericConfig::AccumReal;      // double, double, double, float,  float
    // ...
};

// Instantiated per BuildFlavor (5 targets):
using MorsePotential_Fp64Ref     = MorsePotentialImpl<NumericConfigFp64Reference>;
using MorsePotential_Fp64Prod    = MorsePotentialImpl<NumericConfigFp64Production>;
using MorsePotential_Mixed       = MorsePotentialImpl<NumericConfigMixedFast>;            // Philosophy B
using MorsePotential_MixedAggr   = MorsePotentialImpl<NumericConfigMixedFastAggressive>;  // Philosophy A
using MorsePotential_Fp32        = MorsePotentialImpl<NumericConfigFp32Experimental>;
```

### 8.2. Accumulation rules (каноническая anatomy)

Следуем canonical force kernel template из мастер-спец §D.4:

- **Position delta ALWAYS в double** (StateReal), потом cast к ForceReal — см. §D.5;
- **Inner compute в ForceReal** (float в Mixed, double в Fp64);
- **Per-atom accumulation в AccumReal** (double в Philosophy B, float в Philosophy A);
- **Cross-atom reductions в ReductionReal** (всегда double, кроме Fp32Experimental).

**EAM table lookups всегда double** (кроме Fp32Experimental) — см. §D.6. Это non-negotiable для energy conservation.

### 8.3. Reductions policy

Энергия, virial на уровне zone / global — следуют §D.9:

| BuildFlavor × ExecProfile | Reduction |
|---|---|
| Fp64Ref × Reference | Deterministic tree + Kahan |
| Fp64Prod × Production | Deterministic tree без Kahan |
| Mixed × Production | Deterministic tree без Kahan |
| Mixed × FastExp | Block reduction, non-deterministic OK |
| MixedAggr, Fp32 × FastExp | Atomics OK |

### 8.4. Atomics policy

Следуем §D.8. Для force accumulation (Newton's third law `fx[j] -= Δfx`):

- Reference: **forbidden** — per-block reductions + tree merge;
- Production (Fp64Prod, Mixed): **allowed only for force pair**;
- FastExperimental (Mixed, MixedAggr, Fp32): unrestricted.

### 8.5. Force validation per BuildFlavor

Thresholds из `verify/thresholds.yaml` (см. мастер-спец §D.13):

| BuildFlavor | Force vs LAMMPS | Force vs analytic |
|---|---|---|
| Fp64Reference | 1e-10 | 1e-12 |
| Fp64Production | 1e-10 | 1e-12 |
| MixedFast | 1e-5 | 1e-5 |
| MixedFastAggressive | 1e-4 | 1e-4 |
| Fp32Experimental | 1e-3 | 1e-3 |

### 8.6. Compile flags per BuildFlavor

Потенциалы compiled с flags из §D.10:

- Reference/Production: **no fast-math, FTZ off**;
- Mixed/MixedAggr: fast-math allowed, FTZ on;
- Fp32: fast-math + FTZ, максимум aggressive.

### 8.7. Per-kernel precision overrides — запрещены

Из §D.11: все kernels в одном BuildFlavor используют одинаковую precision policy. Нельзя делать `MorsePotential` в double, а `SnapPotential` в float **внутри одного binary**.

**Исключение через explicit BuildFlavor:** `MixedFastSnapOnlyBuild` (M8 T8.8) — явный BuildFlavor где SNAP в float, EAM в double. Это единственный approved путь per-kernel precision разнообразия. Full specification — §6.7 + master spec §D.11 / §D.17. Formal 7-step §D.17 procedure lands T8.8 per m8 exec pack.

---

## 9. Compute optimization (GPU, wave 1.5)

### 9.1. Kernel taxonomy

Per potential, минимум один kernel:

- **Morse:** single kernel, simple pair loop;
- **EAM:** 3 kernels (density, embedding derivative, force);
- **SNAP:** 3 kernels (bispectrum components, energy, force), + batched matrix ops.

### 9.2. Occupancy и launch config

Целевой occupancy: 50%+ на A100/H100. Too-low occupancy = wasted SMs.

Launch config: `threadsPerBlock = 128` or `256`, `blocksPerSM ≥ 4`. Adjust empirically per kernel.

### 9.3. Memory layout

- `AtomSoA` stores arrays contiguously (SoA good for coalescing);
- Neighbor list — CSR format;
- Per-atom scratch (forces, energies) — in-place accumulation;
- Parameter arrays — `__constant__` memory для pair potentials (small); `__global__` для EAM tables (larger, accessed via texture).

### 9.4. NVTX instrumentation

Каждый compute() call — отдельный NVTX range:
- `MorsePotential::compute`;
- `EamAlloyPotential::density_kernel`;
- `EamAlloyPotential::force_kernel`;
- `SnapPotential::bispectrum_kernel`;
- etc.

Позволяет видеть профиль в Nsight Systems без guessing.

### 9.5. Atomic usage policy

В Reference (+ `deterministic_reduction=true`):
- **zero atomicAdd** — все accumulations через explicit per-block reductions + tree merge.

В Production: atomicAdd OK для forces (Newton's third law pair), но deterministic reduction для global energy/virial.

В FastExperimental: atomics везде.

### 9.6. Pointer aliasing (`__restrict__`)

Следуем мастер-спец §D.16. Все force kernels (CPU и GPU) **обязаны** иметь `__restrict__` qualifiers на всех pointer parameters:

```cpp
[[tdmd::hot_kernel]]
template<typename NumericConfig>
__device__ void eam_force_kernel(
    const AtomSoA* __restrict__ atoms,
    const NeighborList* __restrict__ neigh,
    const EamTables* __restrict__ tables,
    const double* __restrict__ density,      // from density pass
    typename NumericConfig::AccumReal* __restrict__ fx_out,
    typename NumericConfig::AccumReal* __restrict__ fy_out,
    typename NumericConfig::AccumReal* __restrict__ fz_out);
```

Enforcement:
- clang-tidy custom check `tdmd-missing-restrict` на all `[[tdmd::hot_kernel]]` functions;
- Correctness test с `__restrict__` enabled и disabled — results identical;
- Missing `__restrict__` without explicit `NOLINT` rationale = merge block.

Expected speedup: 15-35% (§D.16.2).

### 9.7. Hot kernel annotation

Все performance-critical functions annotated с `[[tdmd::hot_kernel]]`:

```cpp
[[tdmd::hot_kernel]]
void MorsePotential::compute(const ForceRequest& req, ForceResult& result);
```

Attribute служит:
- marker для static analysis tools;
- documentation для reviewers (kernel considered performance-critical);
- hook для future optimizations (auto-apply specific compile flags per hot kernel).

---

## 10. Validation suite

### 10.1. Per-potential mandatory tests

| Test type | Morse | EAM | MEAM | SNAP |
|---|---|---|---|---|
| Analytic 2-atom | ✓ | (indirect via density) | N/A | N/A |
| Diff vs LAMMPS run 0 | ✓ | ✓ | ✓ | ✓ |
| NVE drift 10⁴ steps | ✓ | ✓ | ✓ | ✓ |
| Parameter file round-trip | ✓ | ✓ | ✓ | ✓ |
| Checksum stability | ✓ | ✓ | ✓ | ✓ |
| Multi-species | N/A | ✓ | ✓ | ✓ |
| Periodic BC correctness | ✓ | ✓ | ✓ | ✓ |

### 10.2. Force finite-difference check

Для любого нового потенциала: симметричная конечно-разностная проверка силы против градиента энергии:

```
F_x(i) should equal -(E(x_i + δ) - E(x_i - δ)) / (2δ)
```

Tolerance: `|F_analytic - F_numerical| < 1e-6 · |F|` для δ = `1e-5` Å.

Это catches bugs в force derivation (забытый минус, пропущенный term, etc).

### 10.3. Conservation tests

Long NVE run:
- Total energy drift < `1e-6 per ns`;
- Total momentum conservation: `|P_total(t)| < 1e-12` (should be zero);
- Angular momentum conservation (if no periodic BC): similar.

### 10.4. Differential vs LAMMPS

Centerpiece of validation. CI gate:

```
for potential in [Morse, EAM/alloy, EAM/FS, SNAP]:
    for benchmark in canonical_cases[potential]:
        tdmd_result = run_tdmd(benchmark)
        lammps_result = run_lammps(benchmark)

        force_diff = max(|tdmd.f - lammps.f|)
        assert  force_diff / max(|lammps.f|) < tolerance[precision]

        energy_diff = |tdmd.PE - lammps.PE|
        assert  energy_diff / |lammps.PE| < tolerance[precision]
```

Canonical cases:
- Morse: Al FCC 256, 1024, 4096;
- EAM: Ni, NiAl, Al FCC same sizes;
- SNAP: W_BCC 128, 1024 with published SNAP potential.

Tolerances по précision:
- Reference FP64: `1e-11`;
- Production FP64: `1e-10`;
- Mixed: `1e-5`.

---

## 11. Performance cost hints (для perfmodel)

Каждый `PotentialModel` декларирует cost:

```
virtual double estimated_flops_per_atom() const = 0;
```

Baseline values (calibrated initially, refined after benchmarks):

| Potential | FLOPs/atom | Source |
|---|---|---|
| Morse | ~100 × N_neighbors | analytic |
| LJ | ~50 × N_neighbors | analytic |
| EAM/alloy | ~500 × N_neighbors | measured |
| EAM/FS | ~600 × N_neighbors | measured |
| MEAM | ~2000 × N_neighbors | measured (wave 2) |
| SNAP (J=4) | ~8000 × N_neighbors | measured |
| SNAP (J=8) | ~80000 × N_neighbors | measured |
| PACE | ~20000 × N_neighbors | measured (wave 3) |
| MLIAP | depends on model | measured |

Consumer: `perfmodel/` uses эти numbers для TD vs SD decision (§3.1 perfmodel/SPEC.md).

---

## 12. Roadmap alignment

| Milestone | Potentials deliverable |
|---|---|
| M1 | `MorsePotential` CPU FP64 |
| **M2** | `EamAlloyPotential`, `EamFsPotential` CPU FP64; differential vs LAMMPS |
| M4+ | integration with TD scheduler; per-zone compute masks |
| M6 | GPU kernels для Morse, EAM (Reference and Mixed builds) |
| **M8** | `SnapPotential` CPU + GPU; **T6 benchmark** — proof-of-value for TDMD |
| Post-v1 | `MeamPotential`, `PacePotential`, `MliapPotential` (wave 2-3) |

---

## 13. Open questions

1. **SNAP implementation** — port from LAMMPS vs rewrite? Porting is faster; rewriting может дать лучшую GPU utilization, но risks numerical divergence. Recommendation: **port in M8**, optimize in M8.5.
2. **MLIAP pluggability** — пользователи хотят custom ML models. Как expose C++ extension API? Pybind11 для Python-defined models? Отложено на v2.
3. **Hybrid potentials** (classical + ML correction) — в v2 отдельный `HybridCorrectionPotential` class? Или composition?
4. **Plugin/extension API** — как третьи лица могут добавлять потенциалы без modification TDMD core? Shared library loading + ABI stability — нетривиально.
5. **Parameter file version compatibility** — EAM tables бывают разных формулировок (LAMMPS setfl, DYNAMO, pot.sav); поддерживать все или только LAMMPS-совместимый subset? Recommendation: только LAMMPS-совместимый в v1.

---

*Конец potentials/SPEC.md v1.0, дата: 2026-04-16.*
