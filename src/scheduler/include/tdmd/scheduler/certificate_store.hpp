#pragma once

// SPEC: docs/specs/scheduler/SPEC.md §4.4 (lifecycle) + §4.3 (invalidation)
// Master spec: §6.4
// Exec pack: docs/development/m4_execution_pack.md T4.3
//
// In-memory (zone, time_level) → SafetyCertificate store with explicit
// invalidation. M4 is single-threaded by contract (SPEC §14 M4); mutex-less
// implementation is intentional and will be revisited at M5.
//
// The store allocates cert_id monotonically on build(). mode_policy_tag is
// fixed at construction (mirrors SchedulerPolicy::mode_policy_tag) and
// stamped onto every certificate it issues — call sites never have to
// thread the tag through by hand.

#include "tdmd/scheduler/safety_certificate.hpp"
#include "tdmd/scheduler/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <unordered_map>

namespace tdmd::scheduler {

struct CertKey {
  ZoneId zone_id = 0;
  TimeLevel time_level = 0;
  friend bool operator==(const CertKey&, const CertKey&) noexcept = default;
};

struct CertKeyHash {
  std::size_t operator()(const CertKey& k) const noexcept;
};

class CertificateStore {
public:
  explicit CertificateStore(std::uint64_t mode_policy_tag) noexcept;

  // Build a certificate from inputs, allocate cert_id, stamp mode_policy_tag
  // (overriding whatever the caller put in `in.mode_policy_tag` — the store
  // is the authoritative source), insert into the map (replacing any prior
  // cert for that (zone, time_level)), return a copy.
  //
  // Inserting over an existing key is allowed: it's how refresh after
  // invalidation works at §4.4's `rebuild` edge. The returned certificate
  // is the newly-issued one.
  SafetyCertificate build(const CertificateInputs& in);

  // Returns a pointer into the store or nullptr if not present.
  // Valid until the next mutating call on this store.
  const SafetyCertificate* get(ZoneId zone, TimeLevel time_level) const noexcept;

  // Remove every cert whose zone_id matches. Returns count removed.
  std::size_t invalidate_for(ZoneId zone);

  // Remove every cert. `reason` reserved for telemetry archive (M5 scope);
  // currently discarded but part of the stable API surface.
  std::size_t invalidate_all(std::string_view reason);

  std::size_t size() const noexcept;
  std::uint64_t last_cert_id() const noexcept;
  std::uint64_t mode_policy_tag() const noexcept;

private:
  std::uint64_t mode_policy_tag_;
  std::uint64_t next_cert_id_ = 1;
  std::unordered_map<CertKey, SafetyCertificate, CertKeyHash> map_;
};

}  // namespace tdmd::scheduler
