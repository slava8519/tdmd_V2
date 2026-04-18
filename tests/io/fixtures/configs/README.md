# `tests/io/fixtures/configs/` — `tdmd.yaml` fixtures

Per-fixture expected outcome for T1.4 parser + preflight tests.

| File                         | Stage         | Expected outcome                                                   |
| ---------------------------- | ------------- | ------------------------------------------------------------------ |
| `valid_nve_al.yaml`          | parse + preflight | Parse succeeds; preflight returns empty vector (config is good). |
| `missing_units.yaml`         | parse         | `YamlParseError` at `simulation.units` — required-field missing. |
| `missing_atoms_source.yaml`  | parse         | `YamlParseError` at `atoms.source` — required-field missing.    |
| `bad_timestep.yaml`          | preflight     | Error `integrator.dt` — must be finite and > 0.                  |
| `bad_steps.yaml`             | preflight     | Error `run.n_steps` — must be >= 1.                              |
| `missing_atoms_file.yaml`    | preflight     | Error `atoms.path` — file does not exist.                         |

`valid_nve_al.yaml` references the 32-atom Al FCC fixture from T1.3 at
`../al_fcc_small.data`. Paths are relative to the fixture file — tests
resolve them against `TDMD_TEST_FIXTURES_DIR`.

Add a new fixture only when it exercises a distinct failure mode or a new
accepted-value literal. Inline YAML via `parse_yaml_config_string` covers
line-specific error-message tests; fixtures cover the end-to-end path.
