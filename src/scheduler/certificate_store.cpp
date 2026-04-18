// SPEC: docs/specs/scheduler/SPEC.md §4.3, §4.4
// Master spec: §6.4
// Exec pack: docs/development/m4_execution_pack.md T4.3

#include "tdmd/scheduler/certificate_store.hpp"

#include <cstdint>

namespace tdmd::scheduler {

std::size_t CertKeyHash::operator()(const CertKey& k) const noexcept {
  // splitmix-style hash of (zone_id, time_level). Using a 64-bit mix and
  // returning size_t. ZoneId is u32 so we can pack {zone_id, time_level}
  // into a 96-bit logical key; we compress by multiplying time_level by a
  // large odd constant (golden-ratio prime) and XOR'ing zone_id. Collisions
  // on same zone with distinct time_levels (the hot pattern in the store)
  // are trivially non-collisional because of the multiplier.
  constexpr std::uint64_t kGoldenOdd = 0x9E3779B97F4A7C15ULL;
  std::uint64_t h = k.time_level * kGoldenOdd;
  h ^= static_cast<std::uint64_t>(k.zone_id) + 0x9E3779B9ULL + (h << 6) + (h >> 2);
  h ^= h >> 33;
  h *= 0xFF51AFD7ED558CCDULL;
  h ^= h >> 33;
  return static_cast<std::size_t>(h);
}

CertificateStore::CertificateStore(std::uint64_t mode_policy_tag) noexcept
    : mode_policy_tag_(mode_policy_tag) {}

SafetyCertificate CertificateStore::build(const CertificateInputs& in) {
  const std::uint64_t cert_id = next_cert_id_++;
  CertificateInputs stamped = in;
  stamped.mode_policy_tag = mode_policy_tag_;
  SafetyCertificate c = build_certificate(cert_id, stamped);
  map_[CertKey{in.zone_id, in.time_level}] = c;
  return c;
}

const SafetyCertificate* CertificateStore::get(ZoneId zone, TimeLevel time_level) const noexcept {
  auto it = map_.find(CertKey{zone, time_level});
  if (it == map_.end()) {
    return nullptr;
  }
  return &it->second;
}

std::size_t CertificateStore::invalidate_for(ZoneId zone) {
  std::size_t removed = 0;
  for (auto it = map_.begin(); it != map_.end();) {
    if (it->first.zone_id == zone) {
      it = map_.erase(it);
      ++removed;
    } else {
      ++it;
    }
  }
  return removed;
}

std::size_t CertificateStore::invalidate_all(std::string_view /*reason*/) {
  const std::size_t n = map_.size();
  map_.clear();
  return n;
}

std::size_t CertificateStore::size() const noexcept {
  return map_.size();
}

std::uint64_t CertificateStore::last_cert_id() const noexcept {
  return next_cert_id_ - 1;
}

std::uint64_t CertificateStore::mode_policy_tag() const noexcept {
  return mode_policy_tag_;
}

}  // namespace tdmd::scheduler
