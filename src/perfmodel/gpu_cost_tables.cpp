// SPEC: docs/specs/perfmodel/SPEC.md §3.1, §4.2 (calibration), §9 roadmap M6
// Exec pack: docs/development/m6_execution_pack.md T6.11
//
// PerfModel GPU cost-table constants. Coefficients are starter estimates for
// a midrange Ampere/Ada-class consumer GPU (RTX 4080 / RTX A5000 tier) running
// CUDA 13.1 with the tdmd_gpu kernels introduced in T6.4–T6.8a. They are
// replaced by measured values once the T6.11 calibration harness is driven
// from a Nsight Systems session — see the `//  CALIBRATION:` comments below.
//
// Provenance string in every table documents what those numbers correspond
// to; the ±20 % acceptance gate in test_gpu_cost_tables enforces sane values
// and catches drift when the harness reruns it.
//
// Linear model is: cost_sec(N) = a_sec + b_sec_per_atom · N.
// The model is valid only for 10 k ≲ N ≲ 1 M atoms per device — below 10 k
// launch overhead dominates (a-term) and the b-term is noise; above ~1 M the
// kernels saturate shared memory + register pressure and the model under-
// predicts. Callers that exceed the range should consult fresh Nsight data.

#include "tdmd/perfmodel/gpu_cost_tables.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace tdmd {

namespace {

// Keep the provenance string short (~120 chars): GPU class + CUDA tool + date
// + N-range + method. The exact device/toolkit are documented once, avoiding
// per-field repetition.
//
// CALIBRATION:
//   Coefficients below are **order-of-magnitude placeholders** derived from
//   published Ampere consumer-GPU benchmarks (CUDA 13.1 PCIe Gen4 x16,
//   ~25 GB/s H2D, ~50 μs kernel launch-to-completion for small ops). They
//   pass the structural sanity gate in test_gpu_cost_tables but will not
//   meet the ±20 % acceptance threshold on a specific machine until replaced
//   with Nsight-measured values from bench_gpu_cost_calibration (T6.11b).
//
// Once measured coefficients land, the `provenance` string should shift from
// "initial estimate" to the specific GPU model + CUDA version that was on the
// bench at calibration time (e.g. "RTX 4080 / CUDA 13.1.2 / 2026-04-20").
constexpr const char* kReferenceProvenance =
    "initial estimate, Ampere/Ada consumer, CUDA 13.1, 10k-1M N-range, "
    "T6.11 placeholder — replace via calibration harness (M6 exec pack)";

constexpr const char* kMixedFastProvenance =
    "initial estimate, Ampere/Ada consumer, CUDA 13.1, 10k-1M N-range, "
    "FP32-pair kernels (T6.8a), T6.11 placeholder — replace via calibration "
    "harness";

}  // namespace

namespace {

// T7.10 Pattern 2 outer-halo + reduction placeholder coefficients. The four
// stages added in T7.10 use the same starter-estimate approach as T6.11:
// values come from published intra-node NCCL + PCIe benchmarks for
// Ampere/Ada-class GPUs, replaced via T7.13 calibration when Nsight data
// lands.
//
//   halo_pack            — D2D gather kernel (positions + types into staging
//                          buffer). Same shape as a small element-wise kernel,
//                          ~5 μs launch + ~2 ns/atom for 32 B/atom payload at
//                          ~16 GB/s effective D2D bandwidth.
//   halo_send_outer      — host-staging path on a midrange 100 Gb/s NIC:
//                          D2H + MPI Send/Recv + H2D, ~12 GB/s effective.
//                          Per-atom term: 32 B / 12 GB/s ≈ 2.7 ns/atom.
//                          Launch overhead ~30 μs (MPI init + 2 cudaMemcpy).
//   halo_unpack          — D2D scatter kernel, mirror of halo_pack.
//   nccl_allreduce_inner — intra-subdomain thermo reduction (fixed payload
//                          ~56 B: PE + KE + virial 6 comp). NCCL allreduce
//                          intra-node: ~50 μs flat for tiny payloads.
//                          `b_sec_per_atom` set to 1e-12 — nominally
//                          present so `predict(0)` returns the a-term only,
//                          satisfying the GpuKernelCost::predict contract.
GpuKernelCost halo_pack_default() noexcept {
  return {/*a_sec=*/5.0e-6, /*b_sec_per_atom=*/2.0e-9};
}
GpuKernelCost halo_send_outer_default() noexcept {
  return {/*a_sec=*/30.0e-6, /*b_sec_per_atom=*/32.0 / 12.0e9};
}
GpuKernelCost halo_unpack_default() noexcept {
  return {/*a_sec=*/5.0e-6, /*b_sec_per_atom=*/2.0e-9};
}
GpuKernelCost nccl_allreduce_inner_default() noexcept {
  return {/*a_sec=*/50.0e-6, /*b_sec_per_atom=*/1.0e-12};
}

}  // namespace

GpuCostTables gpu_cost_tables_fp64_reference() {
  GpuCostTables t;

  // ---- Memory transfers (PCIe Gen4 x16 ≈ 25 GB/s effective) -------------
  // H2D: positions (3 × 8 B) + types (4 B) + cell CSR (~4 B/atom slot) per
  // step ≈ 32 B/atom. `a` = per-copy launch overhead ~10 μs.
  t.h2d_atom = {/*a_sec=*/10.0e-6, /*b_sec_per_atom=*/32.0 / 25.0e9};

  // D2H: forces (24 B) + per-atom PE/virial buffers (8 × 7 = 56 B) ≈ 80 B/atom.
  t.d2h_force = {/*a_sec=*/10.0e-6, /*b_sec_per_atom=*/80.0 / 25.0e9};

  // ---- Kernels -----------------------------------------------------------
  // NL build (T6.4): two passes (count + emit) + host scan. ~30 μs entry
  // cost, ~3 ns/atom steady-state (thread-per-atom, 27-cell stencil walk).
  t.nl_build = {/*a_sec=*/30.0e-6, /*b_sec_per_atom=*/3.0e-9};

  // EAM force FP64 (T6.5): three kernels (density + embed + force). FP64
  // fma throttled by --fmad=false; ~50 μs entry, ~5 ns/atom.
  t.eam_force = {/*a_sec=*/50.0e-6, /*b_sec_per_atom=*/5.0e-9};

  // VV pre/post (T6.6): element-wise thread-per-atom. ~10 μs entry, ~1 ns/atom.
  t.vv_pre = {/*a_sec=*/10.0e-6, /*b_sec_per_atom=*/1.0e-9};
  t.vv_post = {/*a_sec=*/10.0e-6, /*b_sec_per_atom=*/1.0e-9};

  // T7.10 — Pattern 2 outer-halo + reduction. Reference and MixedFast share
  // these because the halo pipeline doesn't touch the EAM math precision
  // split; only the inner force kernel differs between flavors.
  t.halo_pack = halo_pack_default();
  t.halo_send_outer = halo_send_outer_default();
  t.halo_unpack = halo_unpack_default();
  t.nccl_allreduce_inner = nccl_allreduce_inner_default();

  t.provenance = kReferenceProvenance;
  return t;
}

GpuCostTables gpu_cost_tables_mixed_fast() {
  GpuCostTables t;

  // Transfers and NL/VV kernels are unchanged — mixed/reference split only
  // affects EAM force math precision (Philosophy B: FP32 pair math, FP64
  // accumulators). T6.8a Nsight traces show identical NL/VV timing.
  t.h2d_atom = {/*a_sec=*/10.0e-6, /*b_sec_per_atom=*/32.0 / 25.0e9};
  t.d2h_force = {/*a_sec=*/10.0e-6, /*b_sec_per_atom=*/80.0 / 25.0e9};
  t.nl_build = {/*a_sec=*/30.0e-6, /*b_sec_per_atom=*/3.0e-9};
  t.vv_pre = {/*a_sec=*/10.0e-6, /*b_sec_per_atom=*/1.0e-9};
  t.vv_post = {/*a_sec=*/10.0e-6, /*b_sec_per_atom=*/1.0e-9};

  // EAM force MixedFast: FP32 pair math ~1.7× faster on per-atom loop
  // (T6.8a traces). Entry cost identical (same 3 kernels).
  t.eam_force = {/*a_sec=*/50.0e-6, /*b_sec_per_atom=*/3.0e-9};

  // T7.10 — Pattern 2 outer-halo + reduction (identical between flavors).
  t.halo_pack = halo_pack_default();
  t.halo_send_outer = halo_send_outer_default();
  t.halo_unpack = halo_unpack_default();
  t.nccl_allreduce_inner = nccl_allreduce_inner_default();

  t.provenance = kMixedFastProvenance;
  return t;
}

// ---------------------------------------------------------------------------
// T7.13 — calibration fixture loader.
//
// Thin YAML reader that keeps yaml-cpp private to the .cpp TU so the public
// header stays free of YAML includes (mirrors the io::YamlConfig pattern
// established in T1.4). Schema errors raise std::runtime_error with a
// one-line explanation; missing file is not an error and is reported via
// std::nullopt.
//
// NOTE on strictness: we mandate every top-level row key (except
// `provenance`, which defaults to "" if absent). `schema_version` must
// equal 1 — bumping the schema requires a new T7.13.x delta. Within a row
// we accept zero or more `measurements` entries; a row with an empty
// `measurements` list is a schema violation (the gate has nothing to
// compare and silent-pass would mask upstream errors).
// ---------------------------------------------------------------------------

namespace {

[[nodiscard]] std::string require_scalar_string(const YAML::Node& parent,
                                                const char* key,
                                                const char* path_ctx) {
  const auto child = parent[key];
  if (!child || !child.IsScalar()) {
    throw std::runtime_error(std::string("gpu_cost_calibration: expected scalar string at '") +
                             path_ctx + "." + key + "'");
  }
  return child.as<std::string>();
}

[[nodiscard]] double require_scalar_double(const YAML::Node& parent,
                                           const char* key,
                                           const char* path_ctx) {
  const auto child = parent[key];
  if (!child || !child.IsScalar()) {
    throw std::runtime_error(std::string("gpu_cost_calibration: expected scalar number at '") +
                             path_ctx + "." + key + "'");
  }
  try {
    return child.as<double>();
  } catch (const YAML::Exception&) {
    throw std::runtime_error(std::string("gpu_cost_calibration: could not parse number at '") +
                             path_ctx + "." + key + "'");
  }
}

[[nodiscard]] std::uint64_t require_scalar_u64(const YAML::Node& parent,
                                               const char* key,
                                               const char* path_ctx) {
  const auto child = parent[key];
  if (!child || !child.IsScalar()) {
    throw std::runtime_error(std::string("gpu_cost_calibration: expected scalar integer at '") +
                             path_ctx + "." + key + "'");
  }
  try {
    return child.as<std::uint64_t>();
  } catch (const YAML::Exception&) {
    throw std::runtime_error(std::string("gpu_cost_calibration: could not parse integer at '") +
                             path_ctx + "." + key + "'");
  }
}

[[nodiscard]] std::string optional_scalar_string(const YAML::Node& parent, const char* key) {
  const auto child = parent[key];
  if (!child || !child.IsScalar()) {
    return {};
  }
  return child.as<std::string>();
}

}  // namespace

std::optional<GpuCalibrationFixture> load_gpu_calibration_fixture(const std::string& path) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec) || ec) {
    return std::nullopt;
  }

  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const YAML::BadFile&) {
    // Treat as missing — race-safe with the filesystem check above.
    return std::nullopt;
  } catch (const YAML::ParserException& e) {
    throw std::runtime_error(std::string("gpu_cost_calibration: YAML parse error: ") + e.what());
  }

  if (!root || !root.IsMap()) {
    throw std::runtime_error("gpu_cost_calibration: top-level document must be a mapping");
  }

  GpuCalibrationFixture fx;

  const auto schema_v = root["schema_version"];
  if (!schema_v || !schema_v.IsScalar()) {
    throw std::runtime_error("gpu_cost_calibration: missing top-level 'schema_version'");
  }
  fx.schema_version = schema_v.as<int>();
  if (fx.schema_version != 1) {
    throw std::runtime_error(std::string("gpu_cost_calibration: unsupported schema_version ") +
                             std::to_string(fx.schema_version) +
                             " (this build supports schema_version=1)");
  }

  const auto rows_node = root["rows"];
  if (!rows_node || !rows_node.IsSequence()) {
    throw std::runtime_error("gpu_cost_calibration: missing or non-sequence top-level 'rows'");
  }

  fx.rows.reserve(rows_node.size());
  for (std::size_t i = 0; i < rows_node.size(); ++i) {
    const auto row_node = rows_node[i];
    const std::string row_ctx = "rows[" + std::to_string(i) + "]";
    if (!row_node.IsMap()) {
      throw std::runtime_error("gpu_cost_calibration: " + row_ctx + " must be a mapping");
    }

    GpuCalibrationRow row;
    row.hardware_id = require_scalar_string(row_node, "hardware_id", row_ctx.c_str());
    row.cuda_version = require_scalar_string(row_node, "cuda_version", row_ctx.c_str());
    row.measurement_date = require_scalar_string(row_node, "measurement_date", row_ctx.c_str());
    row.build_flavor = require_scalar_string(row_node, "build_flavor", row_ctx.c_str());
    row.provenance = optional_scalar_string(row_node, "provenance");

    if (row.build_flavor != "fp64_reference" && row.build_flavor != "mixed_fast") {
      throw std::runtime_error("gpu_cost_calibration: " + row_ctx +
                               ".build_flavor must be 'fp64_reference' or 'mixed_fast' (got '" +
                               row.build_flavor + "')");
    }

    const auto meas_node = row_node["measurements"];
    if (!meas_node || !meas_node.IsSequence() || meas_node.size() == 0U) {
      throw std::runtime_error("gpu_cost_calibration: " + row_ctx +
                               ".measurements must be a non-empty sequence");
    }

    row.measurements.reserve(meas_node.size());
    for (std::size_t j = 0; j < meas_node.size(); ++j) {
      const auto m_node = meas_node[j];
      const std::string m_ctx = row_ctx + ".measurements[" + std::to_string(j) + "]";
      if (!m_node.IsMap()) {
        throw std::runtime_error("gpu_cost_calibration: " + m_ctx + " must be a mapping");
      }
      GpuCalibrationMeasurement m;
      m.n_atoms = require_scalar_u64(m_node, "n_atoms", m_ctx.c_str());
      m.measured_step_sec = require_scalar_double(m_node, "measured_step_sec", m_ctx.c_str());
      if (m.n_atoms == 0U || m.measured_step_sec <= 0.0) {
        throw std::runtime_error("gpu_cost_calibration: " + m_ctx +
                                 " requires n_atoms > 0 and measured_step_sec > 0");
      }
      row.measurements.push_back(m);
    }

    fx.rows.push_back(std::move(row));
  }

  return fx;
}

}  // namespace tdmd
