// Exec pack: docs/development/m5_execution_pack.md T5.7
// SPEC: docs/specs/scheduler/SPEC.md §3.1 (state machine), §6.2 Phase B
// Master spec: §10.4 temporal packet protocol
//
// 2-rank peer dispatch integration test. Each rank runs its own scheduler
// over a 2-zone plan; rank R owns zone R, peer-routing forwards zone R to
// rank (1 - R) on commit. We drive a full Completed→Packed→InFlight→
// Committed cycle on the sender side, and verify that poll_arrivals() on
// the receiver fires on_zone_data_arrived() for the arriving zone. CRC
// drops and cert-hash drops are counted distinctly.

#include "tdmd/comm/mpi_host_staging_backend.hpp"
#include "tdmd/scheduler/causal_wavefront_scheduler.hpp"
#include "tdmd/scheduler/policy.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include <mpi.h>

namespace ts = tdmd::scheduler;
namespace tz = tdmd::zoning;
namespace tc = tdmd::comm;

namespace {

// 2-zone plan with disjoint peer radii (cutoff + skin = 0.2, zone_size = 1)
// so scheduler's spatial DAG is zero — commit_completed's peer branch fires
// solely on peer_routing, not on spatial dependencies.
tz::ZoningPlan make_plan() {
  tz::ZoningPlan plan;
  plan.scheme = tz::ZoningScheme::Linear1D;
  plan.n_zones = {2, 1, 1};
  plan.zone_size = {1.0, 1.0, 1.0};
  plan.cutoff = 0.1;
  plan.skin = 0.1;
  plan.buffer_width = {0.1, 0.1, 0.1};
  plan.canonical_order = {0, 1};
  plan.n_min_per_rank = 1;
  plan.optimal_rank_count = 2;
  return plan;
}

struct SafeSource : ts::CertificateInputSource {
  void fill_inputs(ts::ZoneId zone,
                   ts::TimeLevel time_level,
                   ts::CertificateInputs& out) const override {
    out.zone_id = zone;
    out.time_level = time_level;
    out.v_max_zone = 0.1;
    out.a_max_zone = 0.2;
    out.dt_candidate = 0.001;
    out.buffer_width = 1.0;
    out.skin_remaining = 1.0;
    out.frontier_margin = 1.0;
    out.neighbor_valid_until_step = time_level + 1000;
    out.halo_valid_until_step = time_level + 1000;
  }
};

// Spin `poll_arrivals()` until the receiver's owned zone reports a state
// transition, or the poll budget is exhausted. Mirrors the backend-level
// drain_until helper in test_mpi_host_staging_2rank, scaled down to the
// scheduler's per-rank ownership.
bool poll_until_zone_arrived(ts::CausalWavefrontScheduler& sched,
                             ts::ZoneId watched,
                             int max_polls = 100000) {
  for (int i = 0; i < max_polls; ++i) {
    sched.poll_arrivals();
    if (sched.zone_meta(watched).state != ts::ZoneState::Empty) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  return false;
}

}  // namespace

TEST_CASE("Peer dispatch — Completed→Packed→InFlight→Committed + receiver on_zone_data_arrived",
          "[scheduler][peer_dispatch][mpi][mpi2rank]") {
  int rank = -1;
  int nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  REQUIRE(nranks == 2);

  // Backend — MpiHostStaging (mesh topology, ANY_SOURCE on receive).
  tc::MpiHostStagingBackend backend;
  backend.initialize(tc::CommConfig{});

  auto policy = ts::PolicyFactory::for_reference();
  policy.k_max_pipeline_depth = 1;
  policy.max_tasks_per_iteration = 2;

  ts::CausalWavefrontScheduler sched{policy};
  sched.initialize(make_plan());
  sched.set_comm_backend(&backend);

  // Peer routing: zone 0 → rank 1, zone 1 → rank 0. Each rank owns and
  // dispatches one zone, and receives the other from its peer.
  sched.set_peer_routing({/*zone 0 → rank*/ 1, /*zone 1 → rank*/ 0});

  SafeSource src;
  sched.set_certificate_input_source(&src);

  // Prime only the zone this rank owns. The OTHER zone stays Empty, which
  // is the precondition for poll_arrivals() → on_zone_data_arrived().
  const auto owned = static_cast<ts::ZoneId>(rank);
  const auto peer_zone = static_cast<ts::ZoneId>(1 - rank);
  sched.on_zone_data_arrived(owned, /*step=*/0, /*version=*/0);

  sched.refresh_certificates();
  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 1);
  REQUIRE(tasks[0].zone_id == owned);

  // Drive the owned zone through Phase A.
  sched.mark_computing(tasks[0]);
  sched.mark_completed(tasks[0]);

  // At this point state should be Completed; the zone has a peer route,
  // so commit_completed must walk the full Packed → InFlight → Committed
  // cycle on the OWNED zone (and Empty-stays-Empty on the peer zone).
  REQUIRE(sched.zone_meta(owned).state == ts::ZoneState::Completed);
  REQUIRE(sched.zone_meta(peer_zone).state == ts::ZoneState::Empty);

  sched.commit_completed();

  REQUIRE(sched.zone_meta(owned).state == ts::ZoneState::Committed);
  REQUIRE(sched.zone_meta(peer_zone).state == ts::ZoneState::Empty);

  // Receiver side — spin poll_arrivals() until the peer zone transitions
  // out of Empty. The MpiHostStagingBackend's progress() reaps completed
  // Isends; drain_arrived_temporal() returns the inbound packet; the
  // scheduler fires on_zone_data_arrived(peer_zone, ...).
  REQUIRE(poll_until_zone_arrived(sched, peer_zone));
  REQUIRE(sched.zone_meta(peer_zone).state == ts::ZoneState::ResidentPrev);

  // No CRC or cert-hash drops expected on the happy path.
  REQUIRE(backend.dropped_crc_count() == 0u);
  REQUIRE(backend.dropped_version_count() == 0u);
  REQUIRE(sched.dropped_cert_hash_count() == 0u);

  backend.barrier();
  sched.shutdown();
  backend.shutdown();
}

TEST_CASE("Peer dispatch — without backend (peer_routing default), Pattern 1 short-circuit",
          "[scheduler][peer_dispatch][mpi][mpi2rank][regression]") {
  int rank = -1;
  int nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  REQUIRE(nranks == 2);
  (void) rank;

  // Do NOT inject a comm backend — commit_completed must fall through to
  // commit_completed_no_peer, mirroring the M4 D-M4-6 regression shape.
  // This guards against the peer-dispatch code unconditionally touching
  // the backend pointer.
  auto policy = ts::PolicyFactory::for_reference();
  policy.k_max_pipeline_depth = 1;
  policy.max_tasks_per_iteration = 2;

  ts::CausalWavefrontScheduler sched{policy};
  sched.initialize(make_plan());
  // No set_comm_backend, no set_peer_routing — defaults apply.

  SafeSource src;
  sched.set_certificate_input_source(&src);
  sched.on_zone_data_arrived(0, 0, 0);
  sched.on_zone_data_arrived(1, 0, 0);
  sched.refresh_certificates();

  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 2);
  for (const auto& t : tasks) {
    sched.mark_computing(t);
    sched.mark_completed(t);
  }
  sched.commit_completed();

  // Short-circuit path — zones go Completed → Committed without touching
  // Packed / InFlight. After release_committed they return to Empty.
  for (ts::ZoneId z = 0; z < 2; ++z) {
    REQUIRE(sched.zone_meta(z).state == ts::ZoneState::Committed);
  }
  sched.release_committed();
  for (ts::ZoneId z = 0; z < 2; ++z) {
    REQUIRE(sched.zone_meta(z).state == ts::ZoneState::Empty);
  }

  sched.shutdown();

  // Important: the MpiHostStagingBackend is not used by this case but the
  // MPI environment is active; a final barrier keeps the 2-rank Catch2
  // session balanced across both ranks.
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_CASE("Peer dispatch — poll_arrivals counts cert-hash drops on out-of-range zone_id",
          "[scheduler][peer_dispatch][mpi][mpi2rank][retry]") {
  int rank = -1;
  int nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  REQUIRE(nranks == 2);

  // Send a packet whose zone_id is outside the receiver's total_zones.
  // The backend passes it through (CRC + version OK); the scheduler's
  // poll_arrivals must count it as dropped_cert_hash — the local scheduler
  // doesn't recognise the zone so firing on_zone_data_arrived is not safe.
  tc::MpiHostStagingBackend backend;
  backend.initialize(tc::CommConfig{});

  auto policy = ts::PolicyFactory::for_reference();
  policy.k_max_pipeline_depth = 1;
  policy.max_tasks_per_iteration = 2;

  ts::CausalWavefrontScheduler sched{policy};
  sched.initialize(make_plan());  // total_zones == 2
  sched.set_comm_backend(&backend);

  // Both ranks participate: rank 0 sends a bogus packet to rank 1; rank 1
  // drains + counts. Rank 1 sends nothing. After exchange both ranks run
  // the same assertion on their own scheduler instance (rank 0 expects
  // zero drops since it never received anything; rank 1 expects exactly
  // one cert-hash drop).
  if (rank == 0) {
    tc::TemporalPacket pkt;
    pkt.protocol_version = tc::kCommProtocolVersion;
    pkt.zone_id = 99;  // out of range for the 2-zone plan on rank 1
    pkt.time_level = 1;
    pkt.version = 0;
    pkt.atom_count = 0;
    pkt.certificate_hash = 0xDEADBEEFull;
    pkt.payload.clear();
    backend.send_temporal_packet(pkt, /*dest_rank=*/1);
  }

  if (rank == 1) {
    // Spin poll_arrivals until we see the drop counter tick.
    bool observed = false;
    for (int i = 0; i < 100000 && !observed; ++i) {
      sched.poll_arrivals();
      if (sched.dropped_cert_hash_count() > 0) {
        observed = true;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    REQUIRE(observed);
    REQUIRE(sched.dropped_cert_hash_count() == 1u);
  } else {
    // Rank 0 — no inbound traffic, no drops.
    sched.poll_arrivals();
    REQUIRE(sched.dropped_cert_hash_count() == 0u);
  }

  backend.barrier();
  sched.shutdown();
  backend.shutdown();
}
