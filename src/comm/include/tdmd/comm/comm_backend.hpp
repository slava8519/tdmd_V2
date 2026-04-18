#pragma once

// SPEC: docs/specs/comm/SPEC.md §2.2 (main interface), §5 (async model)
// Master spec: §10 (parallel model), §12.6 (comm interfaces)
// Exec pack: docs/development/m5_execution_pack.md T5.2
//
// Abstract CommBackend — the single point through which TDMD ranks talk
// to each other. comm/SPEC §1.2 states the scope explicitly: comm does
// not decide what to transfer (scheduler owns that), does not know about
// physics (payload is opaque bytes), and does not manage GPU memory
// (state/zoning own buffers). It is pure transport.
//
// Every send_* method is non-blocking (returns immediately after posting
// the underlying MPI_Isend / ncclSend); the matching drain_arrived_*
// method returns whatever has already arrived. progress() drives manual
// progression for MPI implementations that need it. See §5.1 and §5.4.
//
// T5.2 ships the abstract class only. Concrete backends (MpiHostStaging,
// Ring) land in T5.4 and T5.5 respectively.

#include "tdmd/comm/comm_config.hpp"
#include "tdmd/comm/types.hpp"

#include <vector>

namespace tdmd::comm {

class CommBackend {
public:
  // Lifecycle.
  virtual void initialize(const CommConfig& config) = 0;
  virtual void shutdown() = 0;

  // Inner-level (intra-subdomain, TD temporal packets). `dest_rank` is an
  // absolute rank in the global communicator. Non-blocking: returns once
  // the send is posted. Completion is surfaced via progress() ticks and
  // the caller's own tracking — see comm/SPEC §4.5 eager protocol.
  virtual void send_temporal_packet(const TemporalPacket& packet, int dest_rank) = 0;

  // Returns whatever TemporalPackets have arrived since the last drain.
  // Non-blocking: may return an empty vector. Ownership of the returned
  // payload bytes transfers to the caller (who is responsible for
  // release via return_buffer if backends adopt a pooled scheme).
  virtual std::vector<TemporalPacket> drain_arrived_temporal() = 0;

  // Outer-level (inter-subdomain, Pattern 2 halo — M7+). Declared in M5
  // so the interface shape is frozen; M5 backends either implement as
  // no-op or throw "not implemented".
  virtual void send_subdomain_halo(const HaloPacket& packet, int dest_subdomain) = 0;
  virtual std::vector<HaloPacket> drain_arrived_halo() = 0;

  // Cross-subdomain atom migration (Pattern 2 — M7+). Same M5 status as
  // halo: declared, stubbed.
  virtual void send_migration_packet(const MigrationPacket& packet, int dest_subdomain) = 0;
  virtual std::vector<MigrationPacket> drain_arrived_migrations() = 0;

  // Collectives (comm/SPEC §7). In Reference profile, global_sum_double
  // MUST use a deterministic Kahan-compensated ring reduction — see
  // D-M5-9. Raw MPI_Allreduce is not reproducible across runs and is
  // therefore forbidden in Reference.
  virtual double global_sum_double(double local) = 0;
  virtual double global_max_double(double local) = 0;
  virtual void barrier() = 0;

  // Drives outstanding async operations. Cheap and non-blocking — callable
  // many times per compute iteration on idle threads. MPI backends
  // (OpenMPI) often need manual progression; NCCL / NVSHMEM — less so.
  virtual void progress() = 0;

  // Query.
  virtual BackendInfo info() const = 0;
  virtual int rank() const = 0;
  virtual int nranks() const = 0;

  virtual ~CommBackend() = default;
};

}  // namespace tdmd::comm
