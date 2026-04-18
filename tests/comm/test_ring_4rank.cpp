// Exec pack: docs/development/m5_execution_pack.md T5.5
// SPEC: docs/specs/comm/SPEC.md §6.5, §7.1
//
// 4-rank ring smoke for RingBackend. Every rank sends to (rank+1) % 4 and
// receives from (rank-1+4) % 4. Ring-sum is checked for both correctness
// and bit-exact reproducibility across 10 repeat calls. Entry point in
// main_mpi.cpp (shared with T5.4). Assert-on-non-ring-dest is exercised
// via the backend's `is_ring_next()` predicate, which is identical to the
// abort-triggering check in send_temporal_packet — no process-death test
// required.

#include "tdmd/comm/comm_config.hpp"
#include "tdmd/comm/packet_serializer.hpp"
#include "tdmd/comm/ring_backend.hpp"
#include "tdmd/comm/types.hpp"

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include <mpi.h>

namespace tc = tdmd::comm;

namespace {

std::vector<std::uint8_t> make_payload(std::uint32_t n, std::uint64_t seed) {
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(n) * tc::kAtomRecordSize, 0u);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>((seed * 0x9E3779B97F4A7C15ull + i) & 0xFFu);
  }
  return bytes;
}

std::vector<tc::TemporalPacket> drain_until(tc::RingBackend& backend,
                                            std::size_t expected,
                                            int max_polls = 100000) {
  std::vector<tc::TemporalPacket> accumulated;
  for (int i = 0; i < max_polls; ++i) {
    backend.progress();
    auto batch = backend.drain_arrived_temporal();
    for (auto& p : batch) {
      accumulated.push_back(std::move(p));
    }
    if (accumulated.size() >= expected) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return accumulated;
}

}  // namespace

TEST_CASE("RingBackend — 4-rank ring ping (K=10 packets each)", "[comm][mpi][mpi4rank][ring]") {
  int rank = -1;
  int nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  REQUIRE(nranks == 4);

  tc::RingBackend backend;
  backend.initialize(tc::CommConfig{});

  REQUIRE(backend.rank() == rank);
  REQUIRE(backend.nranks() == 4);

  constexpr int kPackets = 10;
  const int next = (rank + 1) % 4;
  const int prev = (rank - 1 + 4) % 4;

  // Each packet carries zone_id=sender_rank and time_level=packet_index so
  // the receiver can verify identity + ordering without extra metadata.
  for (int i = 0; i < kPackets; ++i) {
    tc::TemporalPacket pkt;
    pkt.zone_id = static_cast<tc::ZoneId>(rank);
    pkt.time_level = static_cast<tc::TimeLevel>(i);
    pkt.version = 0;
    pkt.atom_count = static_cast<std::uint32_t>(i + 1);
    pkt.box_snapshot = tc::Box{0, 8, 0, 8, 0, 8};
    pkt.certificate_hash = 0xC0FFEE0000000000ull | static_cast<std::uint64_t>(rank);
    pkt.payload = make_payload(pkt.atom_count, /*seed=*/0x2000 + rank * 100 + i);
    backend.send_temporal_packet(pkt, next);
  }

  const auto arrived = drain_until(backend, kPackets);
  REQUIRE(arrived.size() == kPackets);

  // All received packets must originate from ring-prev and arrive in send
  // order (MPI guarantees ordering within a (src, dst, tag) triple, and the
  // backend only accepts from ring-prev anyway).
  for (std::size_t i = 0; i < arrived.size(); ++i) {
    const auto& p = arrived[i];
    REQUIRE(p.zone_id == static_cast<tc::ZoneId>(prev));
    REQUIRE(p.time_level == i);
    REQUIRE(p.atom_count == static_cast<std::uint32_t>(i + 1));
    REQUIRE(p.protocol_version == tc::kCommProtocolVersion);
  }

  REQUIRE(backend.dropped_crc_count() == 0u);
  REQUIRE(backend.dropped_version_count() == 0u);

  backend.barrier();
  backend.shutdown();
}

TEST_CASE("RingBackend — ring-sum bit-exact on 4 ranks", "[comm][mpi][mpi4rank][ring]") {
  int rank = -1;
  int nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  REQUIRE(nranks == 4);

  tc::RingBackend backend;
  backend.initialize(tc::CommConfig{});

  // Each rank contributes 1.0 -> the ring-sum is 4.0 everywhere.
  const double contribution = 1.0;

  double first_call = 0.0;
  for (int iter = 0; iter < 10; ++iter) {
    const double got = backend.global_sum_double(contribution);
    if (iter == 0) {
      first_call = got;
    }
    REQUIRE(got == 4.0);
    REQUIRE(got == first_call);
    backend.barrier();
  }

  backend.shutdown();
}

TEST_CASE("RingBackend — ring-sum with non-integer contributions", "[comm][mpi][mpi4rank][ring]") {
  int rank = -1;
  int nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  REQUIRE(nranks == 4);

  tc::RingBackend backend;
  backend.initialize(tc::CommConfig{});

  // Contributions chosen so the exact sum is representable in double
  // (0.5 * 4 = 2.0). Purpose: verify ring-sum doesn't introduce a spurious
  // offset for small-magnitude inputs.
  const double got = backend.global_sum_double(0.5);
  REQUIRE(got == 2.0);

  backend.barrier();
  backend.shutdown();
}

TEST_CASE("RingBackend — is_ring_next() guards non-ring dest",
          "[comm][mpi][mpi4rank][ring][guardrail]") {
  // Exercises the predicate that backs the send_temporal_packet abort, so
  // we verify the guard without tripping std::abort on the live MPI world.
  int rank = -1;
  int nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  REQUIRE(nranks == 4);

  tc::RingBackend backend;
  backend.initialize(tc::CommConfig{});

  const int legal = (rank + 1) % 4;
  REQUIRE(backend.is_ring_next(legal));

  // Every other destination should be rejected — exhaustive on 4 ranks.
  for (int d = 0; d < 4; ++d) {
    if (d != legal) {
      REQUIRE_FALSE(backend.is_ring_next(d));
    }
  }

  backend.barrier();
  backend.shutdown();
}

TEST_CASE("RingBackend — BackendInfo reports RingTopologyNative", "[comm][mpi][mpi4rank][ring]") {
  int rank = -1;
  int nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  REQUIRE(nranks == 4);

  tc::RingBackend backend;
  backend.initialize(tc::CommConfig{});

  const auto info = backend.info();
  REQUIRE(info.name == "RingBackend");
  REQUIRE(info.protocol_version == tc::kCommProtocolVersion);

  bool has_ring_capability = false;
  for (const auto c : info.capabilities) {
    if (c == tc::BackendCapability::RingTopologyNative) {
      has_ring_capability = true;
    }
  }
  REQUIRE(has_ring_capability);

  backend.barrier();
  backend.shutdown();
}
