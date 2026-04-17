#pragma once

// SPEC: docs/specs/state/SPEC.md §2.3, §5 (Species registry), §5.3 (checksum)
// Exec pack: docs/development/m1_execution_pack.md T1.1
//
// `SpeciesRegistry` is the one-time-registered, immutable-after-init mapping
// from human-readable species name (e.g. "Al") to a dense `SpeciesId`. Species
// are registered on startup from config / LAMMPS data and never removed mid-
// run (SPEC §5.1).
//
// Units: `mass` in g/mol (metal). `charge` stored but unused in M1 (SPEC §2.3
// "charge: electron charges (v1: always 0)"). `atomic_number` is metadata used
// by reproducibility bundles and — in later milestones — by parameterization
// lookups.

#include "tdmd/state/atom_soa.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tdmd {

struct SpeciesInfo {
  std::string name;
  double mass = 0.0;
  double charge = 0.0;
  std::uint32_t atomic_number = 0;

  [[nodiscard]] bool operator==(const SpeciesInfo& other) const noexcept;
  [[nodiscard]] bool operator!=(const SpeciesInfo& other) const noexcept {
    return !(*this == other);
  }
};

class SpeciesRegistry {
public:
  SpeciesRegistry() = default;

  // Registers a new species and returns its dense SpeciesId. Throws
  // `std::invalid_argument` if a species with the same name is already
  // registered or if the info is malformed (empty name, non-positive mass).
  SpeciesId register_species(const SpeciesInfo& info);

  // Returns the stored info for `id`. Throws `std::out_of_range` on invalid
  // id. Non-throwing alternative: `try_get_info`.
  [[nodiscard]] const SpeciesInfo& get_info(SpeciesId id) const;

  [[nodiscard]] std::optional<std::reference_wrapper<const SpeciesInfo>> try_get_info(
      SpeciesId id) const noexcept;

  // Name → id lookup. Throws `std::out_of_range` if the name is not
  // registered. Non-throwing alternative: `find_id_by_name`.
  [[nodiscard]] SpeciesId id_by_name(std::string_view name) const;

  [[nodiscard]] std::optional<SpeciesId> find_id_by_name(std::string_view name) const noexcept;

  [[nodiscard]] std::size_t count() const noexcept { return species_.size(); }
  [[nodiscard]] bool empty() const noexcept { return species_.empty(); }
  [[nodiscard]] bool contains(std::string_view name) const noexcept {
    return find_id_by_name(name).has_value();
  }

  // Reproducibility checksum (SPEC §5.3): FNV-1a 64 over the canonical byte
  // representation of each `SpeciesInfo` in ascending-SpeciesId order.
  // Deterministic across runs on the same platform.
  [[nodiscard]] std::uint64_t checksum() const noexcept;

private:
  std::vector<SpeciesInfo> species_;
};

}  // namespace tdmd
