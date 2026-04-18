// Exec pack: docs/development/m5_execution_pack.md T5.2
// SPEC: docs/specs/comm/SPEC.md §2.1 (core types), §3.3 (CommConfig)
//
// T5.2 lands only the abstract CommBackend interface and dependency-free
// value types. No concrete backend exists yet, so the tests here mirror
// T3.2 / T2.2 style: instantiate the types, verify defaults match the SPEC,
// and assert a handful of compile-time shape invariants that downstream
// backends (T5.4 / T5.5) will rely on. Serialization smoke + CRC arrive
// with T5.3; ping-pong / ring backend tests arrive with T5.4 / T5.5.

#include "tdmd/comm/comm_backend.hpp"
#include "tdmd/comm/comm_config.hpp"
#include "tdmd/comm/types.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <type_traits>

namespace tc = tdmd::comm;

TEST_CASE("kCommProtocolVersion pinned to v1", "[comm][types]") {
  // comm/SPEC §4.3 — v1 is the initial wire format (D-M5-10). Any bump of
  // this constant requires a coordinated SPEC delta + receiver update.
  STATIC_REQUIRE(tc::kCommProtocolVersion == 1);
}

TEST_CASE("Box — default-constructible with zero extents", "[comm][types]") {
  tc::Box box;
  REQUIRE(box.xlo == 0.0);
  REQUIRE(box.xhi == 0.0);
  REQUIRE(box.ylo == 0.0);
  REQUIRE(box.yhi == 0.0);
  REQUIRE(box.zlo == 0.0);
  REQUIRE(box.zhi == 0.0);
}

TEST_CASE("TemporalPacket — default-constructible, SPEC-compliant defaults", "[comm][types]") {
  tc::TemporalPacket pkt;
  REQUIRE(pkt.protocol_version == tc::kCommProtocolVersion);
  REQUIRE(pkt.zone_id == 0u);
  REQUIRE(pkt.time_level == 0u);
  REQUIRE(pkt.version == 0u);
  REQUIRE(pkt.atom_count == 0u);
  REQUIRE(pkt.certificate_hash == 0u);
  REQUIRE(pkt.payload.empty());
  REQUIRE(pkt.crc32 == 0u);
}

TEST_CASE("HaloPacket — default-constructible (M7+ payload, declared in M5)", "[comm][types]") {
  tc::HaloPacket pkt;
  REQUIRE(pkt.protocol_version == tc::kCommProtocolVersion);
  REQUIRE(pkt.source_subdomain_id == 0u);
  REQUIRE(pkt.dest_subdomain_id == 0u);
  REQUIRE(pkt.time_level == 0u);
  REQUIRE(pkt.atom_count == 0u);
  REQUIRE(pkt.payload.empty());
  REQUIRE(pkt.crc32 == 0u);
}

TEST_CASE("MigrationPacket — default-constructible (M7+, Pattern 2)", "[comm][types]") {
  tc::MigrationPacket pkt;
  REQUIRE(pkt.protocol_version == tc::kCommProtocolVersion);
  REQUIRE(pkt.source_subdomain_id == 0u);
  REQUIRE(pkt.dest_subdomain_id == 0u);
  REQUIRE(pkt.atom_count == 0u);
  REQUIRE(pkt.payload.empty());
  REQUIRE(pkt.crc32 == 0u);
}

TEST_CASE("CommConfig — Reference-safe defaults (D-M5-9, D-M5-11)", "[comm][config]") {
  tc::CommConfig cfg;
  REQUIRE(cfg.backend == tc::BackendKind::Auto);
  REQUIRE(cfg.inner_topology == tc::InnerTopology::Auto);
  REQUIRE(cfg.outer_topology == tc::OuterTopology::Auto);

  // D-M5-9: deterministic Kahan reductions are mandatory in Reference.
  REQUIRE(cfg.use_deterministic_reductions);
  // D-M5-11: CRC32 validation on every packet in M5.
  REQUIRE(cfg.use_crc32);

  REQUIRE(cfg.send_buffer_pool_size == 32u);
  REQUIRE_FALSE(cfg.use_gpu_aware);
  REQUIRE_FALSE(cfg.use_nccl_intranode);
  REQUIRE_FALSE(cfg.auto_bench_on_init);
  REQUIRE(cfg.subdomain_layout.empty());
}

TEST_CASE("BackendKind enum — M5 lands MpiHostStaging + Ring only", "[comm][config]") {
  // Order is part of the public ABI for YAML parsing (io/). If enumerators
  // are reordered the YAML config loader in T5.8 breaks silently.
  STATIC_REQUIRE(static_cast<std::uint8_t>(tc::BackendKind::Auto) == 0);
  STATIC_REQUIRE(static_cast<std::uint8_t>(tc::BackendKind::MpiHostStaging) == 1);
  STATIC_REQUIRE(static_cast<std::uint8_t>(tc::BackendKind::Ring) == 2);
}

TEST_CASE("BackendCapability — distinct values", "[comm][types]") {
  // Capabilities are advertised by backends at init and consumed by
  // HybridBackend routing logic in M7+. Distinctness is load-bearing.
  REQUIRE(tc::BackendCapability::GpuAwarePointers != tc::BackendCapability::RemoteDirectMemory);
  REQUIRE(tc::BackendCapability::CollectiveOptimized != tc::BackendCapability::RingTopologyNative);
  REQUIRE(tc::BackendCapability::GpuAwarePointers != tc::BackendCapability::RingTopologyNative);
}

TEST_CASE("BackendInfo — default-constructible with protocol_version pinned", "[comm][types]") {
  tc::BackendInfo info;
  REQUIRE(info.name.empty());
  REQUIRE(info.capabilities.empty());
  REQUIRE(info.protocol_version == tc::kCommProtocolVersion);
  REQUIRE(info.measured_bw_bytes_per_sec == 0.0);
  REQUIRE(info.measured_latency_us == 0.0);
}

TEST_CASE("CommBackend — abstract interface compiles", "[comm][interface]") {
  // The abstract class cannot be instantiated directly; this test asserts
  // the SPEC-required shape without touching vtable invocation (which needs
  // a concrete backend, T5.4+). The abstract-class check is what gives
  // downstream tasks a stable hook: if a pure-virtual method is dropped
  // from the header the compiler will let this file compile but other
  // modules will fail to link once concrete backends appear.
  STATIC_REQUIRE(std::is_abstract_v<tc::CommBackend>);
  STATIC_REQUIRE(std::has_virtual_destructor_v<tc::CommBackend>);
}
