#pragma once

// SPEC: docs/specs/potentials/SPEC.md §4.5 (EAM parameter file format,
// LAMMPS-compatible). Exec pack: docs/development/m2_execution_pack.md T2.6.
//
// Parsers for LAMMPS-compatible `setfl` text files (the DYNAMO format read
// by `pair_eam/alloy` and `pair_eam/fs`). Returns `TabulatedFunction`
// instances keyed to the grid declared in the file header — no unit
// conversion, no resampling. Downstream EAM force classes (T2.7) consume
// these structs directly.
//
// The two variants differ only in how the density contribution `ρ(r)` is
// indexed:
//   * **alloy** stores one `ρ_α(r)` per species α (N tables total);
//   * **fs**    stores one `ρ_{αβ}(r)` per ordered pair (α, β) — density
//     seen by species α from species β (N² tables total, not symmetric).
//
// Both variants store the pair part as `z2r[α][β] = r · φ_{αβ}(r)` for
// α ≥ β (lower-triangular packing), matching LAMMPS's on-disk convention.
// Per-species embedding F(ρ) is common to both.
//
// Malformed input raises `std::runtime_error` with `path:line: message`
// diagnostic format. Numeric values are preserved at FP64 precision — the
// parsed struct is bit-compatible with what LAMMPS reads from the same
// file (modulo LAMMPS's optional `unit_convert` scaling, which is NOT
// applied here).

#include "tdmd/potentials/tabulated.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace tdmd::potentials {

struct EamAlloyData {
  std::vector<std::string> species_names;  // length N
  std::vector<double> masses;              // length N
  int nrho;                                // number of ρ grid points
  double drho;                             // ρ grid step
  int nr;                                  // number of r grid points
  double dr;                               // r grid step
  double cutoff;                           // file-declared pair cutoff

  // F_rho[α] — embedding function for species α, sampled on ρ = k·drho
  // for k ∈ [0, nrho). Length N.
  std::vector<TabulatedFunction> F_rho;

  // rho_r[α] — density contribution from a single species-α neighbour at
  // distance r, sampled on r = k·dr for k ∈ [0, nr). Length N.
  std::vector<TabulatedFunction> rho_r;

  // z2r[pair_index(α, β)] — r · φ_{αβ}(r) for α ≥ β (lower-triangular
  // packing: `α·(α+1)/2 + β`), sampled on r = k·dr. Length N·(N+1)/2.
  std::vector<TabulatedFunction> z2r;

  // Lower-triangular pair packing index. Symmetric in the unordered (α, β).
  [[nodiscard]] static std::size_t pair_index(std::size_t alpha, std::size_t beta) noexcept;
};

struct EamFsData {
  std::vector<std::string> species_names;
  std::vector<double> masses;
  int nrho;
  double drho;
  int nr;
  double dr;
  double cutoff;

  std::vector<TabulatedFunction> F_rho;

  // rho_ij[row_major_index(α, β)] — density seen by species α from a
  // single species-β neighbour. Row-major packing: `α·N + β`. Length N².
  // **Not** symmetric in (α, β) in the general Finnis-Sinclair case.
  std::vector<TabulatedFunction> rho_ij;

  std::vector<TabulatedFunction> z2r;

  [[nodiscard]] static std::size_t pair_index(std::size_t alpha, std::size_t beta) noexcept;

  [[nodiscard]] static std::size_t rho_ij_index(std::size_t alpha,
                                                std::size_t beta,
                                                std::size_t n) noexcept {
    return alpha * n + beta;
  }
};

// Parse a LAMMPS `.eam.alloy` (setfl) file. Returns a populated data
// struct; throws `std::runtime_error` on any format error.
[[nodiscard]] EamAlloyData parse_eam_alloy(const std::string& path);

// Parse a LAMMPS `.eam.fs` (Finnis-Sinclair setfl) file. Returns a
// populated data struct; throws `std::runtime_error` on any format error.
[[nodiscard]] EamFsData parse_eam_fs(const std::string& path);

}  // namespace tdmd::potentials
