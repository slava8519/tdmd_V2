// Exec pack: docs/development/m5_execution_pack.md T5.4
// SPEC: docs/specs/comm/SPEC.md §6.1, §7.2
//
// 2-rank ping-pong + deterministic reduction smoke for MpiHostStagingBackend.
// Entry point lives in main_mpi.cpp; Catch2 is launched after MPI_Init, so
// `MPI_COMM_WORLD` is live by the time any TEST_CASE runs. Every test is
// written to pass on both ranks — otherwise Catch2 prints a failure on
// one rank only and the harness deadlocks on the final MPI_Barrier.

#include "tdmd/comm/comm_config.hpp"
#include "tdmd/comm/deterministic_reduction.hpp"
#include "tdmd/comm/mpi_host_staging_backend.hpp"
#include "tdmd/comm/packet_serializer.hpp"
#include "tdmd/comm/types.hpp"

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include <mpi.h>

namespace tc = tdmd::comm;

namespace {

// Deterministic synthetic atom-record payload of `n` atoms, value driven
// by `(seed, atom_index)`. The content doesn't matter for ping-pong — we
// check the serializer roundtrip exactly — but keeping it deterministic
// means a future cross-rank byte-hash check is a one-line addition.
std::vector<std::uint8_t> make_payload(std::uint32_t n, std::uint64_t seed) {
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(n) * tc::kAtomRecordSize, 0u);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>((seed * 0x9E3779B97F4A7C15ull + i) & 0xFFu);
  }
  return bytes;
}

// Drains arrived packets until `expected` have landed, or a cap (100k
// polls) is hit. Each miss sleeps ~0.1 ms so the test doesn't burn a core.
// Per comm/SPEC §5.4, progress() + drain is the standard idiom.
std::vector<tc::TemporalPacket> drain_until(tc::MpiHostStagingBackend& backend,
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

TEST_CASE("MpiHostStaging — 2-rank ping-pong (10 packets each way)", "[comm][mpi][mpi2rank]") {
  int rank = -1;
  int nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  REQUIRE(nranks == 2);

  tc::MpiHostStagingBackend backend;
  tc::CommConfig cfg;
  backend.initialize(cfg);

  REQUIRE(backend.rank() == rank);
  REQUIRE(backend.nranks() == 2);

  constexpr int kPackets = 10;
  const int peer = (rank + 1) % 2;

  // Each rank sends 10 packets with atom_count = rank*100 + i. This makes
  // the received atom_counts uniquely identify sender+index.
  for (int i = 0; i < kPackets; ++i) {
    tc::TemporalPacket pkt;
    pkt.zone_id = static_cast<tc::ZoneId>(rank);
    pkt.time_level = static_cast<tc::TimeLevel>(i);
    pkt.version = 0;
    pkt.atom_count = static_cast<std::uint32_t>(rank * 100 + i);
    pkt.box_snapshot = tc::Box{0, 10, 0, 10, 0, 10};
    pkt.certificate_hash = 0xA5A5'5A5A'0000'0000ull | static_cast<std::uint64_t>(rank);
    pkt.payload = make_payload(pkt.atom_count, /*seed=*/0x1000 + rank * 100 + i);
    backend.send_temporal_packet(pkt, peer);
  }

  const auto arrived = drain_until(backend, kPackets);
  REQUIRE(arrived.size() == kPackets);

  // Each received packet must be well-formed and originate from `peer`
  // (identified via zone_id, which we set to sender rank above).
  for (const auto& p : arrived) {
    REQUIRE(p.protocol_version == tc::kCommProtocolVersion);
    REQUIRE(p.zone_id == static_cast<tc::ZoneId>(peer));
    const std::uint32_t expected_atom_count = static_cast<std::uint32_t>(peer * 100 + p.time_level);
    REQUIRE(p.atom_count == expected_atom_count);
  }

  // Zero CRC / version drops in clean MPI loopback.
  REQUIRE(backend.dropped_crc_count() == 0u);
  REQUIRE(backend.dropped_version_count() == 0u);

  backend.barrier();
  backend.shutdown();
}

TEST_CASE("MpiHostStaging — deterministic global_sum_double (D-M5-9)",
          "[comm][mpi][mpi2rank][reduction]") {
  int rank = -1;
  int nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  REQUIRE(nranks == 2);

  tc::MpiHostStagingBackend backend;
  tc::CommConfig cfg;
  // Use_deterministic_reductions default is true (D-M5-9); pin it here so
  // if the default ever flips, the intent of this test is explicit.
  cfg.use_deterministic_reductions = true;
  backend.initialize(cfg);

  const double contribution = static_cast<double>(rank + 1);
  const double expected = static_cast<double>(nranks * (nranks + 1)) / 2.0;

  // Run 10 times; every rank must observe bitwise-identical output.
  double first_call = 0.0;
  for (int iter = 0; iter < 10; ++iter) {
    const double got = backend.global_sum_double(contribution);
    if (iter == 0) {
      first_call = got;
    }
    REQUIRE(got == expected);
    // Bit-exact across repeated invocations on the same rank — deterministic
    // sum is associativity-free at the byte level.
    REQUIRE(got == first_call);
  }

  backend.barrier();
  backend.shutdown();
}

TEST_CASE("MpiHostStaging — global_max_double works", "[comm][mpi][mpi2rank][reduction]") {
  int rank = -1;
  int nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  REQUIRE(nranks == 2);

  tc::MpiHostStagingBackend backend;
  backend.initialize(tc::CommConfig{});

  // Rank 0 contributes 3.14, rank 1 contributes 2.71 → max 3.14.
  const double contribution = (rank == 0) ? 3.14 : 2.71;
  const double got = backend.global_max_double(contribution);
  REQUIRE(got == 3.14);

  backend.barrier();
  backend.shutdown();
}

TEST_CASE("kahan_sum_ordered — associative result independent of repeat", "[comm][reduction]") {
  // Non-MPI scalar guard: the Kahan helper alone must yield the same
  // result across repeated calls with identical input. Guards against any
  // hidden globals in the helper.
  std::vector<double> v{1.0, 1e16, -1e16, 1.0};
  const double a = tc::kahan_sum_ordered(v);
  const double b = tc::kahan_sum_ordered(v);
  REQUIRE(a == b);

  // Rough correctness: with Neumaier compensation, the result is ≈ 2.0 (the
  // naive left-to-right sum in double returns 0.0 for this sequence).
  REQUIRE(a >= 1.5);
  REQUIRE(a <= 2.5);
}
