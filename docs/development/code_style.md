# TDMD code style

Binding conventions for all TDMD C++ and CUDA source. Enforcement: `.clang-format`, `.clang-tidy`, and pre-commit hooks (playbook §5.1). This doc is the human-readable narrative; the machine-readable truth lives in the config files.

**Scope:** `src/`, `tests/`, `benchmarks/`, `tools/` (C++/CUDA). Third-party code (LAMMPS submodule, FetchContent deps) is exempt.

---

## 1. Language and standards

- **C++20**, extensions off. Features: concepts, ranges, `[[likely]]`, `constinit`, `consteval`.
- **Namespace:** flat `namespace tdmd { ... }` for all library code. Module identity lives in include paths (`tdmd/<module>/<header>.hpp`), not namespace nesting — matches authoritative module SPECs.
- **CUDA C++17** (nvcc 12.x ceiling). Host code remains C++20.
- Headers use `#pragma once`, never include guards.
- Never `using namespace std` at file scope; allowed inside functions or `.cpp` for short-lived scopes.

## 2. Naming

| Kind                        | Style               | Example                            |
| --------------------------- | ------------------- | ---------------------------------- |
| Namespaces                  | `lower_case`        | `tdmd` (flat — see §1)             |
| Classes / structs / enums   | `CamelCase`         | `SafetyCertificate`, `BuildFlavor` |
| Free / member functions     | `lower_case`        | `advance_to`, `rebuild_if_stale`   |
| Variables (local, param)    | `lower_case`        | `zone_id`, `safe_until`            |
| Private members             | `lower_case_`       | `pending_count_`                   |
| Compile-time constants      | `lower_case`        | `max_neighbor_count`               |
| Global / linkage constants  | `UPPER_SNAKE`       | `TDMD_MAX_ZONES`                   |
| Macros                      | `UPPER_SNAKE`       | `TDMD_DEVICE_CHECK(...)`           |
| Template type parameters    | `CamelCase`         | `Scalar`, `RealType`               |
| Enum values                 | `CamelCase`         | `BuildFlavor::Fp64Reference`       |

Enforced by `readability-identifier-naming` (see `.clang-tidy`).

## 3. Layout

- Indent 2 spaces. No tabs.
- Column limit 100 (soft).
- Pointer/reference left-binding: `float* p`, `const Atom& a`. Consistency with CUDA: `float* __restrict__ forces`.
- Attach braces: `if (cond) {`, no Egyptian-style newline before `{`.
- No single-line `if`, `for`, `while` — always braces.
- Includes in priority order (enforced by clang-format `IncludeCategories`):
  1. `"tdmd/<module>/<header>.h"` — internal
  2. external / third-party
  3. `<cstdio>` etc. C++ standard
  4. `<some.h>` C headers

## 4. CUDA-specific

- Every hot kernel pointer parameter uses `__restrict__` unless rationalized with `// NOLINT(tdmd-missing-restrict): <why>`. Master spec §D.16. Violations are auto-rejected (playbook §5.1).
- Kernels marked `[[tdmd::hot_kernel]]` by code-review convention until custom clang-tidy check lands in M2-M3.
- Launch bounds (`__launch_bounds__`) on every kernel — occupancy is intentional, not accidental.
- Never `cudaDeviceSynchronize()` in production paths; use events.
- All `cudaMalloc*` / `cudaMemcpy*` paths go through `state/` or `comm/` allocators — direct calls outside these modules flagged by review.

## 5. Error handling philosophy

- **Programming errors → assertions.** `TDMD_ASSERT(cond, "message")` in debug/reference builds; compiled out in production.
- **Physical / configuration errors → exceptions.** Invalid YAML, infeasible schedule, failed preflight. Caught at CLI boundary and pretty-printed.
- **Numerical divergence → certificate failure.** Scheduler signals via `SafetyCertificate::kInvalid`, caller decides retry / escalate.
- No silent fallbacks. If a config is ambiguous, fail loudly.

## 6. Units and numeric types

- Everything in **SI-derived TDMD units** (eV, Å, ps, amu). Hardcoded constants are auto-rejected (playbook §5.1); use `UnitConverter`.
- Numeric types per `BuildFlavor`:
  - `Scalar` is `double` in `Fp64ReferenceBuild` / `Fp64ProductionBuild`.
  - Mixed flavors explicitly switch via `NumericConfig`; no implicit `float`/`double` contraction.
- Never mix `float` and `double` implicitly. Cast or promote deliberately.

## 7. Invariants of hot loops

Playbook §5.4:

- No heap allocation in the inner force loop (`operator new`, `std::vector::push_back`).
- No virtual dispatch across the neighbor traversal.
- No `std::sort` in a rebuild-frequency path; use `radix_sort` or pre-sorted output of the cell grid.
- Branch-free where possible; use `std::copysign`, `std::fma`.

## 8. Comments

- Comments explain **why**, not **what**. Names carry the "what".
- Every module public header starts with a 1-paragraph purpose block and a `// SPEC: docs/specs/<module>/SPEC.md` pointer.
- TODO format: `// TODO(<role or milestone>): <description>` (e.g. `// TODO(M4): ...`).
- NOLINT comments include rationale: `// NOLINTNEXTLINE(check-name): <why>`.

## 9. Tests

- Framework: Catch2 v3. One test file per source translation unit by default.
- Test names describe behavior, not structure: `TEST_CASE("safe_until is monotone under zone growth")`.
- Property tests use ≥10⁵ cases per invariant (master spec §13.4).
- Differential tests cite threshold by name (`TDMD_THR(forces_relative_max)`), never bare numerics.

## 10. Pre-commit enforcement

Local install:

```bash
pip install --user pre-commit
pre-commit install
# one-time sanity check
pre-commit run --all-files
```

Hooks run on every `git commit`. To bypass (e.g. WIP commit on a local branch) use `git commit --no-verify` — but merged PRs must pass `pre-commit run --all-files` in CI regardless. Playbook forbids `--no-verify` on any pushed commit.

Required binaries (install separately):

- `clang-format` ≥ 17 (`apt install clang-format-18`; also pulled by pre-commit mirror)
- `clang-tidy` ≥ 17 (for `tools/lint/run_clang_tidy.sh`; not invoked by pre-commit)
- `python3` ≥ 3.10 (for local STUB hooks)

## 11. Deviations

If a file legitimately cannot follow these rules (e.g. generated code, vendored snippet), wrap it with `// clang-format off ... // clang-format on` and annotate why. Wholesale exemptions require `// NOLINT(...)` at file level and SPEC reference.

---

*M0/T0.2 deliverable. Updates require Core Runtime Engineer review + playbook §15 compliance.*
