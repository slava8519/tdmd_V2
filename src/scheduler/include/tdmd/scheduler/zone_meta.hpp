#pragma once

// SPEC: docs/specs/scheduler/SPEC.md §3 (state machine)
// Master spec: §6.2, §13.4 I1-I5
// Exec pack: docs/development/m4_execution_pack.md T4.4
//
// Per-zone book-keeping record threaded through the state machine and, in
// T4.5+, the scheduler's queues. POD-ish: trivially copyable, default
// constructor yields a fresh Empty zone at time_level 0. Ownership of
// mutation is the scheduler's — direct external mutation of `state` is
// an architectural violation (SPEC §3.3) which a custom clang-tidy check
// will flag at M5+.

#include "tdmd/scheduler/types.hpp"

#include <cstdint>

namespace tdmd::scheduler {

struct ZoneMeta {
  ZoneState state = ZoneState::Empty;
  TimeLevel time_level = 0;
  Version version = 0;

  // 0 sentinel = no certificate currently associated. ZoneStateMachine uses
  // the zero-vs-nonzero distinction to enforce I2 ("cannot go Computing
  // without a certificate"); the scheduler (T4.5) additionally checks that
  // the cert is still present in the CertificateStore (cert can be
  // invalidated between mark_ready and mark_computing).
  std::uint64_t cert_id = 0;

  // I3 tracking: a zone must never be simultaneously enqueued in both.
  // Flags are flipped by the state machine as zones enter/exit the
  // respective logical queues.
  bool in_ready_queue = false;
  bool in_inflight_queue = false;
};

}  // namespace tdmd::scheduler
