#include "tdmd/comm/hybrid_backend.hpp"

// SPEC: docs/specs/comm/SPEC.md §6.4 (HybridBackend)
// Master spec: §14 M7 (Pattern 2 default)
// Exec pack: docs/development/m7_execution_pack.md T7.5

#include <stdexcept>
#include <utility>

namespace tdmd::comm {

HybridBackend::HybridBackend(std::unique_ptr<CommBackend> inner,
                             std::unique_ptr<CommBackend> outer,
                             CartesianGrid grid)
    : inner_(std::move(inner)), outer_(std::move(outer)), topology_(grid) {
  if (!inner_) {
    throw std::invalid_argument("HybridBackend: inner backend must be non-null");
  }
  if (!outer_) {
    throw std::invalid_argument("HybridBackend: outer backend must be non-null");
  }
}

HybridBackend::~HybridBackend() = default;
HybridBackend::HybridBackend(HybridBackend&&) noexcept = default;
HybridBackend& HybridBackend::operator=(HybridBackend&&) noexcept = default;

void HybridBackend::initialize(const CommConfig& config) {
  inner_->initialize(config);
  outer_->initialize(config);
  // D-M7-2: inner and outer must agree on rank space. If they don't, the
  // composition is broken — fail fast rather than corrupt traffic later.
  if (inner_->rank() != outer_->rank() || inner_->nranks() != outer_->nranks()) {
    throw std::runtime_error("HybridBackend: inner/outer disagree on world rank or size");
  }
  if (inner_->nranks() != topology_.grid().total()) {
    throw std::runtime_error(
        "HybridBackend: backend nranks does not match topology subdomain count (D-M7-2 violation)");
  }
}

void HybridBackend::shutdown() {
  // Reverse construction order — outer first, then inner — so collectives
  // remain available during outer teardown if the outer backend needs one.
  outer_->shutdown();
  inner_->shutdown();
}

void HybridBackend::send_temporal_packet(const TemporalPacket& packet, int dest_rank) {
  inner_->send_temporal_packet(packet, dest_rank);
}

std::vector<TemporalPacket> HybridBackend::drain_arrived_temporal() {
  return inner_->drain_arrived_temporal();
}

void HybridBackend::send_subdomain_halo(const HaloPacket& packet, int dest_subdomain) {
  outer_->send_subdomain_halo(packet, dest_subdomain);
}

std::vector<HaloPacket> HybridBackend::drain_arrived_halo() {
  return outer_->drain_arrived_halo();
}

void HybridBackend::send_migration_packet(const MigrationPacket& packet, int dest_subdomain) {
  outer_->send_migration_packet(packet, dest_subdomain);
}

std::vector<MigrationPacket> HybridBackend::drain_arrived_migrations() {
  return outer_->drain_arrived_migrations();
}

double HybridBackend::global_sum_double(double local) {
  // Inner owns the reduction tree to keep D-M5-12 byte-exact thermo chain
  // through the same path M5/M6 used. The outer backend's collective is
  // never invoked — if it were, two different reduction paths could
  // interfere when both paths happen to share a communicator.
  return inner_->global_sum_double(local);
}

double HybridBackend::global_max_double(double local) {
  return inner_->global_max_double(local);
}

void HybridBackend::barrier() {
  inner_->barrier();
}

void HybridBackend::progress() {
  // One tick on each — neither backend should starve. Order is fixed
  // (inner then outer) for determinism of any progress-driven side effect.
  inner_->progress();
  outer_->progress();
}

BackendInfo HybridBackend::info() const {
  BackendInfo i = inner_->info();
  const BackendInfo o = outer_->info();
  i.name = "HybridBackend(inner=" + i.name + ",outer=" + o.name + ")";
  // Union of capabilities — a downstream consumer that asks "is GPU-aware
  // available?" should see "yes" if either path supports it.
  for (const auto& cap : o.capabilities) {
    bool present = false;
    for (const auto& existing : i.capabilities) {
      if (existing == cap) {
        present = true;
        break;
      }
    }
    if (!present) {
      i.capabilities.push_back(cap);
    }
  }
  return i;
}

int HybridBackend::rank() const {
  return inner_->rank();
}
int HybridBackend::nranks() const {
  return inner_->nranks();
}

}  // namespace tdmd::comm
