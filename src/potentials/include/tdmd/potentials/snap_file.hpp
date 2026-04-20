#pragma once

// SPEC: docs/specs/potentials/SPEC.md §6.6 (SNAP parameter file format,
// LAMMPS-compatible). Exec pack: docs/development/m8_execution_pack.md T8.4.
//
// Parsers for LAMMPS-compatible SNAP parameter files (`.snapcoeff`,
// `.snapparam`). Returns a `SnapData` struct keyed to the hyperparameters
// declared в the param file plus the per-species coefficient blocks из the
// coeff file. Downstream `SnapPotential` (T8.4b — force body port) consumes
// `SnapData` directly.
//
// Three artefacts coexist per SNAP potential (см. SPEC §6.6):
//   * `<name>.snap`         — LAMMPS include-style entry point:
//                              `pair_style hybrid/overlay zbl ... snap`
//                              `pair_coeff * * snap <coeff> <param> <species...>`
//                             Used только как hint для config loader: fixture
//                             locator pulls the two sidecar filenames out.
//   * `<name>.snapcoeff`    — per-species coefficients: `<n_species>
//                              <n_coeffs>` header, then per-species block
//                              `<name> <radius> <weight>` + (`n_coeffs` β
//                              values);
//   * `<name>.snapparam`    — hyperparameters: key-value pairs
//                              (`rcutfac`, `twojmax`, `rfac0`, `rmin0`,
//                              `switchflag`, `bzeroflag`, `quadraticflag`,
//                              `chemflag`, `bnormflag`, `wselfallflag`,
//                              `switchinnerflag`). Missing optional keys
//                              resolve to LAMMPS defaults per `pair_snap.cpp`.
//
// Malformed input raises `std::runtime_error` с `path:line: message`
// diagnostic format. All numeric values preserved at FP64 — the parsed
// struct should be bit-identical to what LAMMPS reads from the same file,
// modulo LAMMPS's optional unit_convert scaling (NOT applied here — metal
// units assumed per SNAP convention).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tdmd::potentials {

// LAMMPS-compatible hyperparameters (из *.snapparam). Defaults track
// LAMMPS `pair_snap.cpp` constructor.
struct SnapParams {
  int twojmax = 0;               // 2·J_max (even; required)
  double rcutfac = 0.0;          // global cutoff scaling factor (required)
  double rfac0 = 0.99363;        // inner-to-outer radial basis ratio
  double rmin0 = 0.0;            // minimum radial basis
  bool switchflag = true;        // cosine smooth turned on (Strategy D §2.4)
  bool bzeroflag = true;         // subtract B_k^{empty} reference
  bool quadraticflag = false;    // linear (false) vs quadratic (true) SNAP
  bool chemflag = false;         // multi-species chem SNAP (M9+; false в v1)
  bool bnormflag = false;        // normalise B by number of neighbours
  bool wselfallflag = false;     // include self-interaction on all species
  bool switchinnerflag = false;  // inner cutoff switching (M9+)
};

// Per-species data (из *.snapcoeff). One `SnapSpecies` per row в the
// coefficient file.
struct SnapSpecies {
  std::string name;          // "W", "Ta", ...
  double radius_elem = 0.0;  // per-species R_j
  double weight_elem = 1.0;  // per-species w_j (default 1.0)

  // β coefficients:
  //   * linear:    size == k_max + 1  (β_0 offset + k_max bispectrum coeffs)
  //   * quadratic: size == k_max + 1 + k_max·(k_max+1)/2
  // Where k_max = (twojmax+1)(twojmax+2)(twojmax+3)/6 — computed by the
  // parser and cross-checked against the coefficient count declared в the
  // `.snapcoeff` header.
  std::vector<double> beta;
};

// Full SNAP parameter set.
struct SnapData {
  SnapParams params;
  std::vector<SnapSpecies> species;

  // Derived / cached (populated by parser):
  int k_max = 0;                   // number of bispectrum components
  std::vector<double> rcut_sq_ab;  // pairwise squared cutoffs (symmetric
                                   // n×n, row-major; rcut_ab = rcutfac
                                   // · (R_α + R_β))
  uint64_t checksum = 0;           // parameter_checksum() payload

  [[nodiscard]] static std::size_t pair_index(std::size_t alpha,
                                              std::size_t beta,
                                              std::size_t n_species) noexcept {
    return alpha * n_species + beta;
  }

  // Maximum pairwise cutoff (max over species pairs of rcutfac · (R_α + R_β)).
  // Used by SnapPotential::cutoff() and neighbor-list sizing.
  [[nodiscard]] double max_pairwise_cutoff() const noexcept;
};

// Parse a LAMMPS `.snapcoeff` file. Returns populated species vector (leaves
// `params`, `k_max`, `rcut_sq_ab`, `checksum` untouched — set by
// `parse_snap_files` after cross-checking against param file).
//
// Throws `std::runtime_error` on any format error.
[[nodiscard]] std::vector<SnapSpecies> parse_snap_coeff(const std::string& path);

// Parse a LAMMPS `.snapparam` file. Returns populated hyperparameters.
// Missing optional keys resolve to documented defaults.
//
// Throws `std::runtime_error` on any format error.
[[nodiscard]] SnapParams parse_snap_param(const std::string& path);

// Compose both parsers + derive cached fields. Cross-checks that the
// coefficient-count declared in the `.snapcoeff` header matches `k_max`
// computed from `twojmax` (+ quadratic expansion if `quadraticflag`).
//
// Throws `std::runtime_error` on any format error or mismatch.
[[nodiscard]] SnapData parse_snap_files(const std::string& coeff_path,
                                        const std::string& param_path);

// Compute SNAP k_max from twojmax per LAMMPS `SNA::build_indexlist`.
[[nodiscard]] int snap_k_max(int twojmax) noexcept;

}  // namespace tdmd::potentials
