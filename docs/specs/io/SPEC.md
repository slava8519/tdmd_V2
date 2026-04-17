# io/SPEC.md

**Module:** `io/`
**Status:** master module spec
**Parent:** `TDMD Engineering Spec v2.1` §5.1, §14
**Last updated:** 2026-04-16

---

## 1. Purpose и scope

### 1.1. Что делает модуль

`io/` — единственная точка **файлового ввода-вывода**. Все чтение и запись на диск проходит через этот модуль.

Делает пять вещей:

1. **Parsing** `tdmd.yaml` → `SimulationInput`;
2. **LAMMPS data import** — совместимость с LAMMPS data file format;
3. **Trajectory dumps** — запись траекторий (LAMMPS-compatible + TDMD native HDF5);
4. **Restart files** — serialization `SimulationEngine` state (работает с `runtime/`);
5. **Reproducibility bundle** — собирает все metadata в один packaged output.

### 1.2. Scope: что НЕ делает

- **не инициирует compute** (только read/write);
- **не владеет state** — пишет и читает, но не хранит;
- **не парсит potential files** (EAM .alloy etc.) — это `potentials/` (которые могут использовать `io/` для низкоуровневого file access);
- **не логирует telemetry** (это `telemetry/`);
- **не печатает в stdout** (это `cli/`).

### 1.3. Философия

**Файловые форматы — это контракт с пользователем.** Любое изменение format'а — breaking change, требует version bump + migration tool. `io/` — один из самых stable модулей проекта.

**LAMMPS-compatibility first.** TDMD читает LAMMPS input files без modification. Пишет dump'ы в LAMMPS format'е + native HDF5. Это делает TDMD drop-in альтернативой для workflows уже использующих LAMMPS tooling (VMD, OVITO, LAMMPS-tools).

---

## 2. Public interface

### 2.1. Parsing tdmd.yaml

```cpp
namespace tdmd::io {

struct ConfigParseResult {
    SimulationInput     input;
    RuntimeConfig       runtime;
    std::vector<std::string>  warnings;    // non-fatal issues
};

class ConfigParser {
public:
    virtual ConfigParseResult  parse(const std::string& yaml_path) const = 0;
    virtual ConfigParseResult  parse_string(const std::string& yaml_content) const = 0;

    virtual ~ConfigParser() = default;
};

} // namespace tdmd::io
```

### 2.2. LAMMPS data import

```cpp
class LammpsDataImporter {
public:
    // Parse LAMMPS data file, returning partial state.
    // Requires species registry to be pre-configured (via yaml).
    virtual void  import(
        const std::string& data_file_path,
        const SpeciesRegistry& species,
        const UnitMetadata& units,
        AtomSoA& out_atoms,
        Box& out_box) const = 0;

    virtual ~LammpsDataImporter() = default;
};
```

### 2.3. Trajectory dumps

```cpp
enum class DumpFormat {
    LammpsDumpText,           // тhe classic LAMMPS dump format
    LammpsDumpBinary,          // binary variant, smaller
    Hdf5Native,                // TDMD native HDF5, rich metadata
    Xyz                         // simplest format, for simple viz
};

struct DumpConfig {
    DumpFormat      format;
    std::string     path;
    uint64_t        interval_steps;
    std::vector<std::string>  fields;     // ["id", "type", "x", "y", "z", "vx", ...]
    bool            wrapped;               // primary image или unwrapped (image flags)
    UnitSystem      output_units;          // metal | lj (for LAMMPS compat)
};

class TrajectoryWriter {
public:
    virtual void  write_step(
        uint64_t step, double time_ps,
        const AtomSoA&, const Box&,
        const SpeciesRegistry&) = 0;

    virtual void  flush() = 0;
    virtual void  finalize() = 0;   // close files properly

    virtual ~TrajectoryWriter() = default;
};

class DumpFactory {
public:
    static std::unique_ptr<TrajectoryWriter>  create(const DumpConfig&);
};
```

### 2.4. Restart I/O (работает с runtime/)

```cpp
class RestartIO {
public:
    virtual void  save(
        const std::string& path,
        const RestartBundle&) = 0;

    virtual RestartBundle  load(
        const std::string& path) = 0;

    virtual bool  verify_manifest(const std::string& path) const = 0;

    virtual ~RestartIO() = default;
};
```

`RestartBundle` — структура из `runtime/SPEC §5.1`.

### 2.5. Reproducibility bundle

```cpp
class ReproBundleWriter {
public:
    virtual void  write(
        const std::string& dir_path,
        const ReproContext&,
        const BuildFlavorInfo&,
        const RuntimePolicyBundle&,
        const SimulationInput&,
        const AtomSoA& initial_state,
        const SpeciesRegistry&) = 0;

    virtual ~ReproBundleWriter() = default;
};
```

---

## 3. tdmd.yaml format

### 3.1. Top-level structure

```yaml
simulation:
  units: metal                   # required: metal | lj  (v1)
  seed: 42                       # required for reproducibility
  lj_reference:                  # required if units == lj
    sigma: 1.0
    epsilon: 1.0
    mass: 1.0

box:
  xlo: 0.0
  xhi: 50.0
  # ... ylo/yhi/zlo/zhi
  periodic: [true, true, true]   # per axis

species:
  - name: Al
    mass: 26.9815
    atomic_number: 13

atoms:
  source: lammps_data             # path | inline | lammps_data | generate
  path: ./Al_fcc.data             # if source is lammps_data

potential:
  style: morse                    # or eam/alloy, eam/fs, snap, ...
  params:
    # style-specific

integrator:
  style: velocity_verlet          # or nvt, npt
  dt: 0.001                       # ps
  # style-specific:
  # temperature: 300.0
  # damping_time: 0.1

runtime:
  exec_profile: reference         # reference | production | fast_experimental
  backend: cpu                    # cpu | cuda | auto
  device_count: 1                 # GPUs (if cuda)
  pipeline_depth_cap: 4           # K_max

scheduler:
  preset: auto                    # auto | conservative | aggressive

comm:
  backend: auto                    # auto | mpi_host | gpu_mpi | nccl | hybrid

neighbor:
  skin: 2.0                        # Å (or σ in lj)
  rebuild_interval_hint: 0         # 0 = automatic via displacement tracking

dump:
  - format: lammps_dump_text
    path: traj.lammpstrj
    interval: 100
    fields: [id, type, x, y, z, vx, vy, vz]
    wrapped: true

checkpoint:
  interval: 10000
  keep_last_n: 3
  path: ./checkpoints/

telemetry:
  log_path: ./tdmd.log
  level: info                      # debug | info | warn | error

run:
  n_steps: 100000
```

### 3.2. Required vs optional fields

**Required (preflight error если missing):**
- `simulation.units`;
- `simulation.seed`;
- `box.*` (all 6 bounds + periodic);
- at least one species;
- `atoms.source` + associated path/inline data;
- `potential.style` + params;
- `integrator.style` + `dt`;
- `run.n_steps`.

**Implicit defaults:**
- `runtime.exec_profile = reference`;
- `runtime.backend = auto`;
- `runtime.pipeline_depth_cap = 4`;
- `neighbor.skin = 2.0`;
- `scheduler.preset = auto`;
- `comm.backend = auto`;
- `checkpoint.interval = 0` (disabled);
- `dump = []` (no dumps).

### 3.3. Validation (preflight)

`ConfigParser::parse` выполняет:

1. Syntactic validation (YAML well-formed);
2. Schema validation (required fields present);
3. Semantic validation:
   - `box.xhi > box.xlo` и т.д.;
   - `cutoff < min(lx, ly, lz) / 2` (periodic requirement);
   - `species.mass > 0`;
   - `integrator.dt > 0`;
   - units сonsistency (if lj, lj_reference required);
   - potential params match style.

Failure → raises `ConfigError` с actionable message:

```
ConfigError: неверная box configuration
  box.xlo = 0.0, box.xhi = -5.0
  ожидается: xhi > xlo
  исправление: проверьте единицы и знаки в box bounds
  см. также: docs/user/units.md
```

### 3.4. Extensibility

YAML format будет evolve. Versioning:

```yaml
tdmd_schema_version: 1        # optional; implicit = latest
```

Missing version → current. Explicit old version → parser использует compatible path + emit deprecation warnings.

---

## 4. LAMMPS data import

### 4.1. Supported LAMMPS data file sections

Minimum viable (v1):

- **Header:** atoms count, atom types count, box bounds;
- **Atoms** (atom_style `atomic` или `molecular`):
  ```
  id type x y z  [image_x image_y image_z]
  ```
- **Velocities** (optional):
  ```
  id vx vy vz
  ```
- **Masses** (required if not in yaml):
  ```
  type mass
  ```

### 4.2. Unsupported / v2+ sections

- `Bonds`, `Angles`, `Dihedrals` — pairwise-only v1;
- `Molecules`, `Ellipsoids` — rigid body dynamics post-v1;
- `PairIJ` overrides в data file — expected в pair_coeff в yaml;
- triclinic box (non-orthogonal) — post-v1;
- `atom_style charge` — с post-v1 electrostatics.

Encountering these в data file → **preflight warning** (не error): TDMD читает пропуская unsupported sections, но уведомляет user.

### 4.3. Unit system awareness

LAMMPS data files **сами по себе не указывают единицы**. Единицы определяются:
1. Explicit `simulation.units` в yaml (если указан);
2. Default `metal` если не указан в yaml.

Если пользователь imports LAMMPS data что был написан для `real` или `lj`, но в yaml указал `metal` — values будут interpreted в metal (потенциально wrong!). `UnitConverter` преобразует если `lj` → `metal`; для `real` → `metal` — v2+ feature.

**Preflight check:** если atom positions unrealistic для declared units (е.g. `x = 0.5` в metal means 0.5 Å что подозрительно близко) → warning.

### 4.4. Parse algorithm (outline)

```
function parse_lammps_data(path, species, units):
    lines = read file; strip comments; split into tokens
    state = HEADER

    for line in lines:
        if line matches "Atoms" section header:
            state = ATOMS
            continue
        elif line matches "Velocities":
            state = VELOCITIES
            continue
        # ... other sections

        if state == HEADER:
            parse_header_line(line) → update counts, box bounds
        elif state == ATOMS:
            parse_atom_line(line) → atoms.add(...)
        elif state == VELOCITIES:
            parse_velocity_line(line) → atoms.set_velocity(...)
        # ...

    validate all atoms have velocities (or set to 0 если absent)
    apply unit conversion if needed
    return (atoms, box)
```

**Deterministic parsing:** atoms appear в `AtomSoA` в том же порядке, что в data file. `AtomId` assignment — последовательный starting from some offset (usually 1 для LAMMPS compat).

### 4.5. Validation tests

- Known small LAMMPS data file (Al FCC 64 atoms) → 64 atoms imported, box correct;
- Compare import result с LAMMPS `read_data` output (by printing atom_style data back);
- Corrupted file (missing section header) → clear error;
- Very large file (10⁶ atoms) → reasonable performance (< 10s parsing).

---

## 5. Trajectory dumps

### 5.1. LAMMPS dump text format (canonical для interop)

```
ITEM: TIMESTEP
<step>
ITEM: NUMBER OF ATOMS
<N>
ITEM: BOX BOUNDS pp pp pp
<xlo> <xhi>
<ylo> <yhi>
<zlo> <zhi>
ITEM: ATOMS id type x y z [vx vy vz] [fx fy fz] ...
<atom records>
```

Field selection в `DumpConfig::fields`. Default fields: `[id, type, x, y, z]`.

### 5.2. LAMMPS dump binary format

Compact binary; reduces file size by ~2x; same information content. OVITO, VMD read both.

### 5.3. Native HDF5 format

```
/tdmd_trajectory/
  /metadata/
    tdmd_version
    build_flavor
    run_id
    units
  /box/
    bounds              # (N_timesteps, 6)
    periodic            # (3,)
  /atoms/
    id                  # (N_timesteps, N_atoms) — should be invariant
    type                # (N_timesteps, N_atoms)
    position            # (N_timesteps, N_atoms, 3)
    velocity            # (N_timesteps, N_atoms, 3)
    force               # (N_timesteps, N_atoms, 3)
    image               # (N_timesteps, N_atoms, 3) int32
  /timesteps/
    step                # (N_timesteps,) uint64
    time                # (N_timesteps,) double, ps
```

Benefits:
- compressed (zlib/lz4);
- random access (no need to scan from start);
- rich metadata;
- compatible с pandas / xarray / numpy для analysis.

### 5.4. Output frequency policy

`interval: 100` means every 100 steps. Other options:
- `interval: 0` — only first and last step (initial + final state);
- `on_event: rebuild` — dump на каждом neighbor rebuild;
- `time_interval: 1.0` — every 1 ps (requires dt tracking).

### 5.5. Multi-file dumps (rollover)

For long runs, single file impractical. Rollover options:

```yaml
dump:
  - format: hdf5
    path: traj_{rollover:06d}.h5
    interval: 100
    rollover_every_steps: 100000   # new file every 100k steps
```

`{rollover:06d}` is path formatter: 6-digit zero-padded index.

### 5.6. Unwrapped coordinates

`wrapped: false` writes:
```
actual_position = atoms.x + image_x * box.lx
```

Useful для diffusion analysis где wrapped positions misleading.

---

## 6. Restart I/O

### 6.1. HDF5 structure

```
/tdmd_restart/
  /manifest/
    version
    checksum
    timestamp
  /config/
    yaml_serialized
  /build_info/
    git_sha
    compiler_id
    ...
  /state/
    atoms/
      id, type, x, y, z, vx, vy, vz, fx, fy, fz, image_*, flags
    box/
      ...
    species/
      ...
    version
  /scheduler/
    current_step
    frontier_min/max
    zone_states/
      zone_id, state, time_level, version
    cert_store/
      ...
  /telemetry/
    last_N_iterations/
```

### 6.2. Manifest and integrity

At save:
1. Write all data в HDF5;
2. Compute CRC32 over content;
3. Write `manifest.json`:
   ```json
   {
       "tdmd_version": "2.1.0",
       "git_sha": "...",
       "timestamp": "2026-04-16T12:34:56Z",
       "crc32": "deadbeef",
       "file": "restart.h5",
       "file_size": 12345678
   }
   ```

At load: verify CRC32 matches. Mismatch → reject.

### 6.3. Version compatibility

Restart saved by TDMD 2.1.x → can be loaded by TDMD 2.1.y (y ≥ x). Major version change (2.x → 3.0) — migration tool required.

Build flavor must match exactly. `Fp64Reference` restart cannot load в `MixedFast` binary.

### 6.4. Async save (post-v1)

In v1, save_restart blocks run loop. Post-v1: async save — runtime copies state to shadow buffer, continues run, I/O thread writes в background. Requires careful consistency.

---

## 7. Reproducibility bundle

Из `runtime/SPEC §8`, implemented here.

### 7.1. Directory layout

```
repro_bundle_<run_id>/
├── manifest.json              # CRC32 of all files
├── README.md                   # human-readable описание
├── config.yaml                 # full effective config
├── build_info.json
├── policies.json
├── hardware_profile.json
├── initial_state.h5
├── species_registry.json
├── potential_checksum.txt
├── seed.txt
└── commands_reproduce.sh       # shell script to reproduce the run
```

### 7.2. commands_reproduce.sh

```bash
#!/bin/bash
# Generated reproduction script для run_id = <run_id>
# Generated on 2026-04-16 by TDMD v2.1.0

# Check build flavor:
tdmd --version --json | jq -e '.build_flavor == "Fp64ReferenceBuild"' || {
    echo "ERROR: incorrect build flavor"
    exit 1
}

# Run with bundle config:
tdmd run config.yaml --load-initial-state initial_state.h5 --seed $(cat seed.txt)
```

### 7.3. README.md содержит

- Short description of what this run was;
- Key parameters summary (N atoms, potential, ensemble, steps);
- Commit SHA of TDMD when run started;
- Timestamp и contact info если user provided.

---

## 8. Error handling

### 8.1. Categories

- **File not found:** clear error message включающий path, suggests `tdmd validate` for preflight.
- **Permission denied:** suggests checking permissions, including parent directory.
- **Corrupted file (CRC mismatch):** explicit warning о data integrity, suggests re-download or backup.
- **Format version mismatch:** explicit error с required version.
- **Parse errors (YAML, LAMMPS data):** line number + context + likely fix.

### 8.2. Error format

```
IO Error: invalid YAML at line 42:
  box:
    xlo: 0.0
  >>> xhi 50.0
       ^-- expected colon after 'xhi'

Suggestion: 'xhi: 50.0' (add colon)
See: docs/user/config_format.md
```

---

## 9. Tests

### 9.1. Unit tests

- **YAML parsing:** valid yaml → expected `ConfigParseResult`;
- **LAMMPS data import:** canonical Al_fcc.data → atoms count, box, types correct;
- **Unit conversion:** lj input → metal state с known sigma/epsilon/mass;
- **Dump write/read round-trip:** write 100 steps → read back → identical data;
- **Restart round-trip:** save → load → identical internal state.

### 9.2. Format compatibility tests

- **LAMMPS dump interop:**
  - TDMD writes dump, parse by LAMMPS `read_dump` command (через external tool);
  - LAMMPS writes dump, TDMD reads it;
  - Output compared byte-by-byte для specific fields.
- **HDF5 integrity:** h5check tool reports no errors.

### 9.3. Stress tests

- **Large file:** 10⁸ atoms dump в HDF5 → < 1 minute;
- **Many dumps:** 10⁴ dump files + metadata → filesystem-friendly;
- **Long YAML:** 10k lines config parses reasonably.

### 9.4. Error handling tests

- Missing required field → clear error message;
- Invalid YAML syntax → line number in error;
- Corrupted restart file → refuse to load with clear error;
- LAMMPS data with unsupported section → warning, continue.

---

## 10. Telemetry

Metrics:
```
io.config_parse_time_ms
io.lammps_data_parse_time_ms
io.dump_write_count
io.dump_write_bytes_total
io.restart_save_time_ms
io.restart_load_time_ms
```

NVTX ranges:
- `io::parse_yaml`;
- `io::import_lammps_data`;
- `io::dump_write`;
- `io::save_restart`;
- `io::load_restart`.

---

## 11. Roadmap alignment

| Milestone | IO deliverable |
|---|---|
| **M1** | YAML parser; LAMMPS data importer minimal; LAMMPS dump text writer |
| M2 | LJ import support; full LAMMPS data sections; HDF5 writer |
| M3 | Reproducibility bundle writer |
| M4 | Restart save/load; manifest integrity |
| M5 | Stress-tested на large systems (10⁶ atoms) |
| v2+ | Async save; triclinic box; `real` unit support; extensible dump fields |

---

## 12. Open questions

1. **Dump compression** — zlib vs lz4 vs zstd? Trade-off speed vs ratio. Recommendation: lz4 default (fast, ~2x ratio), option для zstd (better ratio).
2. **Partial / incremental dump** — append-mode HDF5? Avoid rewriting whole file каждый dump.
3. **Metadata schema evolution** — HDF5 groups добавляем new fields in v2.x. Backward compat: older readers skip unknown groups.
4. **LAMMPS scripting DSL import** — TDMD yaml ≠ LAMMPS script. Worth writing converter? Maybe post-v1 tool `tdmd-convert-lammps` external.
5. **Custom user fields** — allow user-defined extra per-atom fields (e.g. `charge`, `orientation`)? Requires extensions в `AtomSoA`. Post-v1.
6. **Network-mounted storage** — large restart files on network filesystem have different performance characteristics. Should io/ behave differently?

---

*Конец io/SPEC.md v1.0, дата: 2026-04-16.*
