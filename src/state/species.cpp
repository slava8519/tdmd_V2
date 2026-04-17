#include "tdmd/state/species.hpp"

#include <cstring>
#include <stdexcept>

namespace tdmd {

namespace {

constexpr std::uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
constexpr std::uint64_t kFnvPrime = 0x100000001b3ULL;

inline std::uint64_t fnv1a_append(std::uint64_t h, const void* data, std::size_t len) noexcept {
  const auto* const bytes = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < len; ++i) {
    h ^= bytes[i];
    h *= kFnvPrime;
  }
  return h;
}

template <typename T>
inline std::uint64_t fnv1a_append_pod(std::uint64_t h, const T& value) noexcept {
  static_assert(std::is_trivially_copyable_v<T>,
                "fnv1a_append_pod requires trivially copyable type");
  return fnv1a_append(h, &value, sizeof(T));
}

}  // namespace

bool SpeciesInfo::operator==(const SpeciesInfo& other) const noexcept {
  return name == other.name && mass == other.mass && charge == other.charge &&
         atomic_number == other.atomic_number;
}

SpeciesId SpeciesRegistry::register_species(const SpeciesInfo& info) {
  if (info.name.empty()) {
    throw std::invalid_argument("SpeciesRegistry: species name must not be empty");
  }
  if (!(info.mass > 0.0)) {
    throw std::invalid_argument("SpeciesRegistry: species mass must be positive");
  }
  if (find_id_by_name(info.name).has_value()) {
    throw std::invalid_argument("SpeciesRegistry: species '" + info.name +
                                "' is already registered");
  }
  const auto id = static_cast<SpeciesId>(species_.size());
  species_.push_back(info);
  return id;
}

const SpeciesInfo& SpeciesRegistry::get_info(SpeciesId id) const {
  if (id >= species_.size()) {
    throw std::out_of_range("SpeciesRegistry::get_info — id out of range");
  }
  return species_[id];
}

std::optional<std::reference_wrapper<const SpeciesInfo>> SpeciesRegistry::try_get_info(
    SpeciesId id) const noexcept {
  if (id >= species_.size())
    return std::nullopt;
  return std::cref(species_[id]);
}

SpeciesId SpeciesRegistry::id_by_name(std::string_view name) const {
  if (const auto maybe = find_id_by_name(name); maybe.has_value()) {
    return *maybe;
  }
  throw std::out_of_range("SpeciesRegistry::id_by_name — unknown species '" + std::string(name) +
                          "'");
}

std::optional<SpeciesId> SpeciesRegistry::find_id_by_name(std::string_view name) const noexcept {
  for (std::size_t i = 0; i < species_.size(); ++i) {
    if (species_[i].name == name) {
      return static_cast<SpeciesId>(i);
    }
  }
  return std::nullopt;
}

std::uint64_t SpeciesRegistry::checksum() const noexcept {
  std::uint64_t h = kFnvOffset;
  const auto count_value = static_cast<std::uint64_t>(species_.size());
  h = fnv1a_append_pod(h, count_value);
  for (const auto& s : species_) {
    const auto name_len = static_cast<std::uint64_t>(s.name.size());
    h = fnv1a_append_pod(h, name_len);
    h = fnv1a_append(h, s.name.data(), s.name.size());
    h = fnv1a_append_pod(h, s.mass);
    h = fnv1a_append_pod(h, s.charge);
    h = fnv1a_append_pod(h, s.atomic_number);
  }
  return h;
}

}  // namespace tdmd
