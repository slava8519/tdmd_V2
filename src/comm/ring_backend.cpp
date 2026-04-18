// SPEC: docs/specs/comm/SPEC.md §6.5, §7.1
// Exec pack: docs/development/m5_execution_pack.md T5.5

#include "tdmd/comm/ring_backend.hpp"

#if defined(TDMD_ENABLE_MPI) && TDMD_ENABLE_MPI

#include "tdmd/comm/packet_serializer.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace tdmd::comm {

namespace {

// Distinct from MpiHostStaging's kTagTemporal (1001) so the two backends
// can coexist on the same communicator without cross-talk. Ring-sum uses
// its own tag band to avoid false matches against temporal-packet traffic.
constexpr int kTagTemporal = 1002;
constexpr int kTagRingSum = 1101;
constexpr int kTagRingBcast = 1102;

// Kahan-Neumaier pair-sum: identical math to kahan_sum_ordered's inner loop,
// but exposed as a standalone binary op so the ring-sum phase can extend
// the accumulator one-peer-at-a-time. Keeping (sum, c) paired and
// re-injecting c on the next step would need an extra scalar hop around
// the ring; in practice, the one-shot round-trip loses at most one ulp,
// well below the reduction's O(P·eps) drift budget.
double kahan_add(double a, double b) noexcept {
  const double t = a + b;
  double c;
  if (std::abs(a) >= std::abs(b)) {
    c = (a - t) + b;
  } else {
    c = (b - t) + a;
  }
  return t + c;
}

}  // namespace

RingBackend::~RingBackend() {
  if (comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&comm_);
    comm_ = MPI_COMM_NULL;
  }
  if (owns_mpi_init_) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
    owns_mpi_init_ = false;
  }
}

void RingBackend::initialize(const CommConfig& config) {
  config_ = config;

  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited) {
    int provided = 0;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SINGLE, &provided);
    owns_mpi_init_ = true;
  }

  MPI_Comm_dup(MPI_COMM_WORLD, &comm_);
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &nranks_);
}

void RingBackend::shutdown() {
  for (auto& send : pending_sends_) {
    MPI_Wait(&send.request, MPI_STATUS_IGNORE);
  }
  pending_sends_.clear();

  if (comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&comm_);
    comm_ = MPI_COMM_NULL;
  }
  if (owns_mpi_init_) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
    owns_mpi_init_ = false;
  }
}

bool RingBackend::is_ring_next(int dest) const noexcept {
  if (nranks_ <= 0) {
    return false;
  }
  return dest == ((rank_ + 1) % nranks_);
}

void RingBackend::reap_completed_sends() {
  for (auto it = pending_sends_.begin(); it != pending_sends_.end();) {
    int flag = 0;
    MPI_Test(&it->request, &flag, MPI_STATUS_IGNORE);
    if (flag) {
      it = pending_sends_.erase(it);
    } else {
      ++it;
    }
  }
}

void RingBackend::send_temporal_packet(const TemporalPacket& packet, int dest_rank) {
  if (!is_ring_next(dest_rank)) {
    // Reference profile — fatal. A ring backend forwarding to anything
    // other than (rank+1)%P is a scheduler bug; keep failing loudly so the
    // bug is caught by the test that produced it, not propagated further.
    std::fprintf(stderr,
                 "[RingBackend] illegal dest_rank=%d from rank=%d (expected %d)\n",
                 dest_rank,
                 rank_,
                 (rank_ + 1) % nranks_);
    std::abort();
  }

  reap_completed_sends();

  PendingSend pending;
  pending.buffer = pack_temporal_packet(packet);
  pending_sends_.push_back(std::move(pending));
  auto& back = pending_sends_.back();

  MPI_Isend(back.buffer.data(),
            static_cast<int>(back.buffer.size()),
            MPI_BYTE,
            dest_rank,
            kTagTemporal,
            comm_,
            &back.request);
}

std::vector<TemporalPacket> RingBackend::drain_arrived_temporal() {
  std::vector<TemporalPacket> out;
  const int prev = (rank_ - 1 + nranks_) % nranks_;

  // Accept only from ring-prev — MPI_Iprobe with explicit source.
  for (;;) {
    int flag = 0;
    MPI_Status status;
    MPI_Iprobe(prev, kTagTemporal, comm_, &flag, &status);
    if (!flag) {
      break;
    }

    int byte_count = 0;
    MPI_Get_count(&status, MPI_BYTE, &byte_count);

    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(byte_count));
    MPI_Recv(buffer.data(), byte_count, MPI_BYTE, prev, kTagTemporal, comm_, MPI_STATUS_IGNORE);

    auto result = unpack_temporal_packet(buffer);
    if (!result.ok()) {
      if (result.error.find("CRC") != std::string::npos) {
        ++dropped_crc_;
      } else if (result.error.find("protocol version") != std::string::npos) {
        ++dropped_version_;
      } else {
        ++dropped_crc_;
      }
      continue;
    }
    out.push_back(std::move(*result.packet));
  }
  return out;
}

void RingBackend::send_subdomain_halo(const HaloPacket&, int) {}
std::vector<HaloPacket> RingBackend::drain_arrived_halo() {
  return {};
}
void RingBackend::send_migration_packet(const MigrationPacket&, int) {}
std::vector<MigrationPacket> RingBackend::drain_arrived_migrations() {
  return {};
}

double RingBackend::global_sum_double(double local) {
  // Sequential add around the ring with a fixed rank-ascending accumulation
  // order + ring-broadcast (comm/SPEC §7.1). Single-rank degenerate case:
  // the sum is the local value.
  if (nranks_ == 1) {
    return local;
  }

  const int prev = (rank_ - 1 + nranks_) % nranks_;
  const int next = (rank_ + 1) % nranks_;

  double acc = 0.0;
  if (rank_ == 0) {
    // Phase 1: seed with rank 0's contribution and pass it to rank 1.
    acc = local;
    MPI_Send(&acc, 1, MPI_DOUBLE, next, kTagRingSum, comm_);
    // Sum travels the ring, accumulating via Kahan at each stop, and
    // returns to rank 0 with the full total.
    MPI_Recv(&acc, 1, MPI_DOUBLE, prev, kTagRingSum, comm_, MPI_STATUS_IGNORE);
  } else {
    double incoming = 0.0;
    MPI_Recv(&incoming, 1, MPI_DOUBLE, prev, kTagRingSum, comm_, MPI_STATUS_IGNORE);
    acc = kahan_add(incoming, local);
    MPI_Send(&acc, 1, MPI_DOUBLE, next, kTagRingSum, comm_);
  }

  // Phase 2: ring-broadcast. Rank 0 holds the total from Phase 1; forward
  // it once around the ring. The last rank (P-1) receives the value and
  // stops — no need to loop back to rank 0 (rank 0 already has acc).
  if (rank_ == 0) {
    MPI_Send(&acc, 1, MPI_DOUBLE, next, kTagRingBcast, comm_);
  } else {
    MPI_Recv(&acc, 1, MPI_DOUBLE, prev, kTagRingBcast, comm_, MPI_STATUS_IGNORE);
    if (rank_ != nranks_ - 1) {
      MPI_Send(&acc, 1, MPI_DOUBLE, next, kTagRingBcast, comm_);
    }
  }

  return acc;
}

double RingBackend::global_max_double(double local) {
  // MAX is order-invariant; MPI_Allreduce is fine here even in a strictly
  // ring-flavored backend. This keeps the reduction O(log P) instead of
  // a redundant ring traversal.
  double result = 0.0;
  MPI_Allreduce(&local, &result, 1, MPI_DOUBLE, MPI_MAX, comm_);
  return result;
}

void RingBackend::barrier() {
  MPI_Barrier(comm_);
}

void RingBackend::progress() {
  reap_completed_sends();
}

BackendInfo RingBackend::info() const {
  BackendInfo i;
  i.name = "RingBackend";
  i.protocol_version = kCommProtocolVersion;
  i.capabilities = {BackendCapability::RingTopologyNative};
  return i;
}

int RingBackend::rank() const {
  return rank_;
}

int RingBackend::nranks() const {
  return nranks_;
}

}  // namespace tdmd::comm

#endif  // TDMD_ENABLE_MPI
