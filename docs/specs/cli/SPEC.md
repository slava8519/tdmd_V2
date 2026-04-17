# cli/SPEC.md

**Module:** `cli/`
**Status:** master module spec
**Parent:** `TDMD Engineering Spec v2.1` §14 (UX), §5.1
**Last updated:** 2026-04-16

---

## 1. Purpose и scope

### 1.1. Что делает модуль

`cli/` — user-facing слой. Превращает TDMD из движка в **инструмент для учёного**. Один исполняемый `tdmd` с подкомандами.

Делает пять вещей:

1. **Command dispatch** — `tdmd run`, `tdmd validate`, `tdmd explain`, `tdmd compare`, `tdmd resume`, `tdmd repro-bundle`;
2. **Argument parsing** — flags, positional args, help text;
3. **Preflight** — check config before running, emit actionable errors / warnings;
4. **Human-readable output** — formatted breakdown, progress indicators, error messages;
5. **Exit codes** — conventional Unix exit codes для automation.

### 1.2. Scope: что НЕ делает

- **не парсит config** (это `io/`);
- **не выполняет симуляцию** (delegates to `runtime/SimulationEngine`);
- **не владеет state** (no state of its own);
- **не меняет данные** — только reads и orchestrates.

### 1.3. Философия: scientist-first

Master spec §14: **scientist can run TDMD without understanding its internals**. CLI должен быть дружелюбным, с actionable errors и встроенными safeguards.

Каждая команда имеет:
- `--help` с примером использования;
- preflight validation перед destructive operations;
- progress indicators для long operations;
- meaningful exit codes.

Цель: от `$ tdmd run case.yaml` до научного результата — минимум friction для типичного use case.

---

## 2. Command taxonomy

### 2.1. Canonical commands

```
tdmd --version                  Print version info and exit
tdmd --help                     Show general help

tdmd run          <config>      Run simulation
tdmd validate     <config>      Validate config без running
tdmd explain      <config>      Explain runtime plan, perf prediction
tdmd compare      <config>      Run + compare с reference (LAMMPS)
tdmd resume       <dir>         Resume from checkpoint
tdmd repro-bundle <dir>         Create reproducibility bundle
tdmd convert      <...>         Format conversions (post-v1)
tdmd benchmark    <name>        Run canonical benchmarks
```

### 2.2. Hidden / developer commands

Для debugging, не первичные:

```
tdmd debug scheduler-dump     Dump scheduler internal state
tdmd debug neighbor-visualize Visualize neighbor structure
tdmd debug profile-kernels    Run per-kernel profiling
```

Hidden — не показываются в `tdmd --help`, только в `tdmd --help --verbose`.

---

## 3. `tdmd run` — main command

### 3.1. Usage

```
tdmd run <config.yaml> [options]

Required:
  config.yaml               Path to TDMD configuration file

Options:
  --n-steps N               Override run.n_steps from config
  --exec-profile PROFILE    Override runtime.exec_profile (reference|production|fast_experimental)
  --backend BACKEND         Override runtime.backend (cpu|cuda|auto)
  --seed N                  Override simulation.seed
  --output-dir DIR          Directory for outputs (default: ./tdmd_run_<run_id>)
  --checkpoint-interval N   Override checkpoint.interval
  --dry-run                 Go through init, print plan, don't run
  --verbose, -v             Increase verbosity (stacks: -vv, -vvv)
  --quiet, -q               Suppress non-essential output
  --no-color                Disable ANSI colors в output
```

### 3.2. Execution flow

```
1. Parse args
2. Load and validate config (io/)
3. Preflight checks (see §5)
4. Create SimulationEngine
5. configure() → resolve_policies() → bootstrap_state() → initialize_execution()
6. Register signal handlers (SIGINT, SIGTERM → request_stop)
7. Print startup banner with summary:
     - run_id, build_flavor, exec_profile
     - box dimensions, atom count, species
     - potential style, integrator style
     - predicted performance (from perfmodel)
8. Run loop с progress indicator
9. Final breakdown report
10. Write reproducibility bundle
11. finalize() → shutdown()
12. Exit with code 0 (success)
```

### 3.3. Startup banner

```
TDMD v2.1.0 (build: Fp64ReferenceBuild, git: abc1234)
Starting run: <run_id>

Configuration summary:
  Box: 50.0 × 50.0 × 50.0 Å (orthogonal, periodic)
  Atoms: 4,000 (species: Al)
  Potential: morse (D=0.27, α=1.16, r₀=3.25, r_c=8.0 Å)
  Integrator: velocity-Verlet, NVE, dt=1.0 fs
  Target: 100,000 steps (~100 ps)

Runtime:
  Exec profile: reference
  Backend: cpu (1 thread)
  Pattern: Pattern 1 (single-subdomain TD)
  Pipeline depth K: 4

Performance prediction:
  Expected throughput: 12,500 steps/s
  Expected wall time: ~8 s
  Expected efficiency: 87%

Starting simulation...
```

### 3.4. Progress indicator

```
 [=====>                    ]  23.5% | step 23500/100000 | 12,489 steps/s | ETA 6s
```

Updates every ~1s. Suppressed with `-q`. Also shows temperature / energy if NVT/NPT.

### 3.5. Final report

```
Simulation completed.

Statistics:
  Total steps: 100,000
  Wall time: 8.01 s
  Throughput: 12,485 steps/s
  Performance efficiency: 87.3% of predicted

Thermodynamics (averages):
  T: 300.1 ± 2.3 K
  PE: -3.4025 eV/atom
  KE: 0.03876 eV/atom
  E_total: -3.3637 eV/atom (drift: 2.1e-7 relative)

[LAMMPS-compatible breakdown here — see telemetry SPEC §4.2]

Outputs:
  Trajectory: ./tdmd_run_20260416_123456/traj.lammpstrj (15.2 MB)
  Log: ./tdmd_run_20260416_123456/tdmd.log
  Reproducibility bundle: ./tdmd_run_20260416_123456/repro_bundle/
  Final checkpoint: ./tdmd_run_20260416_123456/final.h5

Exit: success
```

---

## 4. `tdmd validate`

### 4.1. Purpose

Preflight validation **без** running simulation. Для CI pipelines, для users чтобы catch errors cheaply.

### 4.2. Usage

```
tdmd validate <config.yaml> [options]

Options:
  --strict                  Treat warnings as errors
  --output-format FORMAT    human (default) | json | sarif
  --check ITEM              Specific check (schema | units | potential | memory | all)
```

### 4.3. Checks performed

1. **Schema:** YAML valid, required fields present;
2. **Units:** consistent, valid для chosen potential;
3. **Box:** positive extents, cutoff < half min extent для periodic;
4. **Atoms:** data file exists, parseable, atom types match species;
5. **Potential:** style supported, params file exists (EAM alloy file etc);
6. **Integrator:** style supported, parameters reasonable;
7. **Runtime:** exec_profile × build_flavor compatibility;
8. **Memory:** estimated memory usage fits в hardware (warning if > 80% available);
9. **Performance:** predicted throughput reasonable (warning if < 100 steps/s suggests misconfiguration).

### 4.4. Output (human)

```
$ tdmd validate case.yaml

Validating case.yaml...

✓ Schema valid
✓ Units: metal (consistent)
✓ Box: 50×50×50 Å, periodic XYZ (cutoff 8.0 < half extent 25.0)
✓ Atoms: 4,000 loaded from ./Al_fcc.data
✓ Species: Al (mass 26.9815)
✓ Potential: morse (params valid)
✓ Integrator: velocity-Verlet, dt=0.001 ps
✓ Runtime: Fp64ReferenceBuild + reference (compatible)
⚠ Memory estimate: 450 MB (60% of 756 MB available)
✓ Performance prediction: 12,500 steps/s

Result: PASS (1 warning)

Run: `tdmd run case.yaml`
```

### 4.5. Output (JSON)

```json
{
  "result": "pass",
  "warnings_count": 1,
  "errors_count": 0,
  "checks": [
    {"name": "schema", "status": "pass"},
    {"name": "memory", "status": "warning", "message": "..."},
    ...
  ]
}
```

### 4.6. Exit codes

- `0`: PASS (no warnings);
- `1`: PASS with warnings;
- `2`: FAIL (errors);
- `3`: Config file not found;
- `4`: Internal validator error.

В `--strict` mode, warnings → exit 2.

---

## 5. `tdmd explain`

### 5.1. Purpose

Human-readable explanation of **что TDMD собирается сделать**. Pedagogical tool для students, transparency tool для researchers.

### 5.2. Usage

```
tdmd explain <config.yaml> [options]

Options:
  --perf                    Focus на performance analysis
  --runtime                 Focus на runtime architecture
  --scheduler               Focus на scheduler plan
  --verbose                 Show all details
  --format FORMAT           human | json | markdown
```

### 5.3. Sections

**General explain:**

```
$ tdmd explain case.yaml

TDMD Execution Plan
===================

Physical system:
  Al crystal (FCC), 4,000 atoms, 50×50×50 Å
  Morse potential (cutoff 8 Å)

Domain decomposition:
  Zoning scheme: Linear1D (for box aspect 1:1:1 may want Hilbert3D...
                 but atom count too small, Linear1D simpler)
  Zones: 6 × 6 × 6 = 216 total
  Zone size: 8.3 × 8.3 × 8.3 Å (= cutoff + skin)

Deployment pattern: Pattern 1 (single-subdomain TD)
  Ranks: 1 (no MPI needed)
  Pipeline depth K: 4
  N_min per rank: 2

Runtime characteristics:
  Exec profile: reference (deterministic)
  Numerical: Fp64 throughout
  Scheduler: CausalWavefrontScheduler, fixed priority

Performance prediction:
  (see `tdmd explain --perf` for details)
  Expected throughput: 12,500 steps/s
  Expected efficiency: 87%

Output plan:
  Trajectory every 100 steps → traj.lammpstrj
  Checkpoint every 10,000 steps
  Total run: 100,000 steps, ~8 seconds

Ready to run: `tdmd run case.yaml`
```

**--perf variant:**

Output from `perfmodel/SPEC §6.1`. Full breakdown of предсказаний.

**--scheduler variant:**

```
Scheduler Plan:
  Zone count: 216
  Canonical ordering: linear z-y-x (zone_id = x + 6y + 36z)
  Certificate refresh: every iteration
  Pipeline depth K_max: 4

Expected zone activity pattern (first few iterations):
  Iter 1: zones 0, 1, 2, ... Ready → Computing → Completed
  Iter 2: zones 0 commits for t=1; zones 216-17 enter t=2 pipeline stage
  ...

Expected steady-state pipeline depth: 2.5 zones in flight on average
Expected rebuild interval: ~50 steps (based on skin/dt/v_max)
```

---

## 6. `tdmd compare`

### 6.1. Purpose

Run TDMD + reference (LAMMPS) side-by-side и compare. First-class tool для scientific validation.

### 6.2. Usage

```
tdmd compare <config.yaml> [options]

Options:
  --with lammps             Reference engine (currently only lammps supported)
  --lammps-bin PATH         LAMMPS binary path (autodetect if not provided)
  --metrics METRIC,METRIC   Metrics to compare (forces,energy,nve_drift,msd,rdf)
  --tolerance NAME          Tolerance preset (strict | default | loose)
  --output-dir DIR          Where to save comparison report
```

### 6.3. Workflow

```
1. Parse config.yaml.
2. Generate equivalent LAMMPS input script from tdmd.yaml.
3. Run LAMMPS in subprocess.
4. Run TDMD с identical config.
5. Both write trajectories и thermodynamic outputs.
6. Load both results.
7. Compute metric deltas (forces, energy, etc).
8. Apply tolerances, produce pass/fail report.
```

### 6.4. Output report

```
$ tdmd compare case.yaml --with lammps

Running reference (LAMMPS 23Jun2022)...
  Completed in 8.5s

Running TDMD...
  Completed in 7.9s (1.08× faster)

Comparison results:
========================================
Metric          | Target      | Actual     | Status
----------------------------------------
Max |Δf| / |f|  | < 1e-10     | 3.2e-12    | ✓ PASS
Max |ΔE| / |E|  | < 1e-10     | 7.1e-13    | ✓ PASS
NVE drift       | < 1e-6      | 2.1e-7     | ✓ PASS
T average       | 300 ± 3 K   | 300.1 K    | ✓ PASS
MSD slope       | ± 5%        | -1.2%      | ✓ PASS
========================================
Overall: PASS (all metrics within tolerance)

Report saved to: ./compare_20260416/report.md
```

### 6.5. JSON output для CI

```json
{
  "reference_engine": "lammps",
  "reference_version": "23Jun2022",
  "tdmd_version": "2.1.0",
  "wall_time_lammps": 8.5,
  "wall_time_tdmd": 7.9,
  "speedup": 1.08,
  "metrics": [
    {"name": "max_force_rel_error", "target": 1e-10, "actual": 3.2e-12, "pass": true},
    ...
  ],
  "overall": "pass"
}
```

---

## 7. `tdmd resume`

### 7.1. Usage

```
tdmd resume <checkpoint_dir> [options]

Options:
  --n-additional-steps N    Run N more steps (default: from original config)
  --override-config PATH    Use different config (с caveats)
```

### 7.2. Workflow

```
1. Validate checkpoint directory (manifest, CRC).
2. Verify tdmd_version и build_flavor match current binary.
3. Call SimulationEngine.load_restart(path).
4. Continue run loop from saved state.
5. Final reports / cleanup как в `tdmd run`.
```

### 7.3. Errors

- Version mismatch: clear error, suggest correct binary;
- CRC failure: reject с suggestion to use backup;
- Build flavor mismatch: reject, explain why (numerical consistency).

---

## 8. `tdmd repro-bundle`

### 8.1. Usage

```
tdmd repro-bundle <run_dir> [options]

Options:
  --output PATH             Where to create bundle (default: <run_dir>/repro_bundle/)
  --include trajectory      Include trajectory (default: only metadata + initial state)
  --compress                Compress bundle into .tar.gz
```

### 8.2. What's in bundle

From `io/SPEC §7`. Плюс:

- Shell script `commands_reproduce.sh`;
- README.md с summary;
- Optional trajectory (if `--include trajectory`).

### 8.3. Use case

Researcher shares bundle с collaborator / reviewer. Collaborator:
```
$ tar xzf repro_bundle.tar.gz
$ cd repro_bundle/
$ bash commands_reproduce.sh
```

Gets same-run output (within reproducibility guarantees of build flavor).

---

## 9. Error handling и exit codes

### 9.1. Exit code convention

- `0`: success;
- `1`: general failure;
- `2`: config validation failed;
- `3`: file not found;
- `4`: permission denied;
- `5`: out of memory;
- `6`: computation error (NaN, physics violation);
- `7`: network / MPI error;
- `8`: CUDA / GPU error;
- `9`: interrupted by signal (graceful);
- `10`: interrupted by signal (forced);
- `>= 100`: user-specified custom exit codes.

### 9.2. Error message format

```
Error: <category>
  <specific message>

Details:
  <key>: <value>
  ...

Suggestion: <actionable advice>
See also: <doc path / URL>
```

Example:

```
Error: Box too small for periodic cutoff
  cutoff = 8.0 Å, but half min box extent = 6.5 Å

Details:
  box.xhi - box.xlo = 13.0 Å
  cutoff required: 8.0 Å
  rule: cutoff < 0.5 · min(box extents)

Suggestion: either reduce cutoff (potential.cutoff)
           or enlarge box (box.*)
See also: docs/user/periodic_bc.md
```

### 9.3. Color coding (unless `--no-color`)

- Errors: red;
- Warnings: yellow;
- Successes (✓): green;
- Info (ℹ): blue;
- Headers: bold;
- Metric deltas: red (bad) / green (good).

---

## 10. Logging vs CLI output

### 10.1. Двойная output

- **stdout**: human-readable CLI output (banners, progress, reports);
- **stderr**: errors, warnings;
- **log file** (via telemetry): structured JSON lines for machine parsing.

### 10.2. `-q` (quiet) mode

Suppress stdout non-essential:
- No banner;
- No progress;
- Only final summary и errors.

Useful for scripting.

### 10.3. `-v` (verbose)

- `-v`: some extra info (per-step temperature prints);
- `-vv`: scheduler debug messages;
- `-vvv`: everything including NVTX events.

---

## 11. Shell integration

### 11.1. Tab completion

Ship bash/zsh/fish completion scripts:

```
$ tdmd <TAB>
run  validate  explain  compare  resume  repro-bundle  benchmark

$ tdmd run <TAB>
case.yaml  other_case.yaml  ...
```

### 11.2. Man page

`man tdmd` shows comprehensive reference с examples.

### 11.3. JSON output mode для scripting

Most commands support `--output-format json` для integration с pipelines:

```bash
$ tdmd validate case.yaml --output-format json | jq '.result'
"pass"

$ tdmd explain case.yaml --format json | jq '.predicted_steps_per_second'
12500
```

---

## 12. Tests

### 12.1. Unit tests

- Argument parsing: various valid / invalid argv;
- Exit code correctness: simulated failures → correct exit code;
- Color handling: `--no-color` disables ANSI;
- Format output: JSON parses correctly.

### 12.2. Integration tests

- `tdmd validate <canonical configs>` → all pass;
- `tdmd validate <intentionally broken>` → clear error;
- `tdmd run` small config → complete, produce expected outputs;
- `tdmd resume` round-trip;
- `tdmd compare` с LAMMPS stub → fake compare report.

### 12.3. User acceptance tests

Scripted scenarios:
- "New user: `tdmd --help` → `tdmd validate example.yaml` → `tdmd run example.yaml`" passes with minimal friction;
- "Resume workflow: interrupted run → `tdmd resume`" works;
- "Repro: `tdmd repro-bundle` → send to peer → they reproduce";

### 12.4. Performance tests

- `tdmd --help` starts up < 100ms (no heavy init for lightweight commands);
- `tdmd validate` complete < 1s для typical config;
- `tdmd run` startup overhead (config parse to first step) < 2s.

---

## 13. Documentation requirements

Each command должна иметь:

- `tdmd <command> --help` text (reference);
- `docs/user/<command>.md` (tutorial);
- example в `examples/<usecase>/` directory.

Documentation ownership: `cli/` owns command help + user tutorials. Technical specs (what command actually does internally) remain в other module SPECs.

---

## 14. Roadmap alignment

| Milestone | CLI deliverable |
|---|---|
| **M1** | `tdmd run`, `tdmd validate` — basic functionality, minimal flags |
| M2 | `tdmd explain`; full preflight checks; JSON output formats |
| M3 | Tab completion; better error messages |
| M4 | Progress indicator; banner; final report |
| M5 | `tdmd resume`, `tdmd repro-bundle` |
| **M8** | `tdmd compare --with lammps` |
| M9+ | `tdmd benchmark`; `tdmd convert` (format conversions) |
| v2+ | Interactive mode (`tdmd shell`)? Web UI? |

---

## 15. Open questions

1. **Output directory policy** — per-run dir (good for isolation) or single dir with timestamps (good для aggregation)? Default: per-run с timestamp.
2. **Configuration inheritance** — can `case.yaml` include другой YAML? Useful для sharing common settings. Consider for v2.
3. **Environment variable overrides** — `TDMD_EXEC_PROFILE=production tdmd run ...` should override yaml? Yes, but explicit flag always wins.
4. **Interactive shell** — useful для debugging, но heavy engineering task. Post-v1.
5. **Localization** — English only в v1. Russian support? Other languages? Probably community-driven post-v1.
6. **REST API mode** — `tdmd serve` as HTTP daemon? Niche use case, but requested for integration в workflows. Post-v1.

---

*Конец cli/SPEC.md v1.0, дата: 2026-04-16.*
