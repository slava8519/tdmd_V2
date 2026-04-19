#include "tdmd/perfmodel/perfmodel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tdmd {

namespace {

// Halo / temporal-packet payload size per atom. SPEC §3.2 / §3.3 reference it
// as `atom_record_size` without fixing a value; for M2 we use 32 bytes — 3
// doubles of position plus a 4-byte type/tag header, which matches the Pattern
// 3 ghost-atom record shipped across ranks. M6 revisits when GPU packs add
// velocities to the halo for NH/NPT.
constexpr double kAtomRecordBytes = 32.0;

// SPEC §6.5a + exec pack T2.10 clamp: K swept only over {1, 2, 4, 8, 16}. The
// analytic minimum from §3.3 is K_opt ≈ sqrt(T_p / T_c_startup) rounded to
// power of 2; iterating the candidate set also captures the overlap clamp
// `max(0, T_p/K - T_c)` which the pure sqrt formula elides.
constexpr std::array<std::uint32_t, 5> kKCandidates = {1U, 2U, 4U, 8U, 16U};

bool profile_is_sane(const HardwareProfile& hw) noexcept {
  return std::isfinite(hw.cpu_flops_per_sec) && hw.cpu_flops_per_sec > 0.0 &&
         std::isfinite(hw.intra_bw_bytes_per_sec) && hw.intra_bw_bytes_per_sec > 0.0 &&
         std::isfinite(hw.inter_bw_bytes_per_sec) && hw.inter_bw_bytes_per_sec > 0.0 &&
         std::isfinite(hw.scheduler_overhead_sec) && hw.scheduler_overhead_sec >= 0.0 &&
         hw.n_ranks > 0U;
}

bool potential_is_sane(const PotentialCost& pc) noexcept {
  return std::isfinite(pc.flops_per_pair) && pc.flops_per_pair > 0.0 &&
         pc.n_neighbors_per_atom > 0U;
}

}  // namespace

PerfModel::PerfModel(HardwareProfile hw, PotentialCost potential) : hw_(hw), potential_(potential) {
  if (!profile_is_sane(hw_)) {
    throw std::invalid_argument(
        "PerfModel: HardwareProfile must have finite, positive FLOPS, bandwidths, "
        "non-negative scheduler overhead, and n_ranks > 0");
  }
  if (!potential_is_sane(potential_)) {
    throw std::invalid_argument(
        "PerfModel: PotentialCost must have finite flops_per_pair > 0 and "
        "n_neighbors_per_atom > 0");
  }
}

double PerfModel::compute_time_sec(std::uint64_t n_atoms_per_rank) const noexcept {
  // SPEC §3.1: T_c = (N_per_rank · C_force) / FLOPS_rank.
  // C_force = flops_per_pair · N_neighbors (per atom per step).
  const double c_force =
      potential_.flops_per_pair * static_cast<double>(potential_.n_neighbors_per_atom);
  return static_cast<double>(n_atoms_per_rank) * c_force / hw_.cpu_flops_per_sec;
}

PerfPrediction PerfModel::predict_pattern3(std::uint64_t n_atoms) const {
  // Per-rank work decomposition (§3.1). Pattern 3: P_space = n_ranks, P_time = 1.
  // Integer division matches the SPEC formula exactly — residual atoms fold
  // into the trailing rank at M3+ when zoning plans the exact split.
  const std::uint64_t n_per_rank = n_atoms / hw_.n_ranks;

  const double t_c = compute_time_sec(n_per_rank);

  // §3.2: T_halo_SD = 2 · N_per_rank^(2/3) · atom_record_size / B_inter.
  // Single-rank (n_ranks == 1) has no halo — the cbrt factor is still
  // N^(2/3) but there's no neighbor to send to, so halo time drops to 0.
  const double t_halo = (hw_.n_ranks > 1U)
                            ? 2.0 * std::pow(static_cast<double>(n_per_rank), 2.0 / 3.0) *
                                  kAtomRecordBytes / hw_.inter_bw_bytes_per_sec
                            : 0.0;

  // §3.5 correction: scheduler overhead per iteration (10-50 μs midpoint
  // stored in the profile). Neighbor rebuild and migration amortizations
  // need live telemetry, which lands with T2.12; omitted here.
  const double t_step = t_c + t_halo + hw_.scheduler_overhead_sec;

  PerfPrediction p;
  p.pattern_name = "Pattern3_SD";
  p.t_step_sec = t_step;
  p.recommended_K = 1U;
  p.speedup_vs_baseline = 1.0;
  return p;
}

PerfPrediction PerfModel::predict_pattern1(std::uint64_t n_atoms, std::uint32_t K) const {
  if (K == 0U) {
    throw std::invalid_argument("PerfModel::predict_pattern1: K must be >= 1");
  }

  // Pattern 1 single-subdomain: P_space = 1, P_time = n_ranks. Each rank owns
  // one TD pipeline stage, so zone size == rank size.
  const std::uint64_t n_per_rank = n_atoms / hw_.n_ranks;

  const double t_c = compute_time_sec(n_per_rank);

  // §3.3: T_p = atom_record_size · N_per_zone / B_intra.
  // For single-subdomain Pattern 1, zone == rank, so N_per_zone = N_per_rank.
  const double t_p =
      kAtomRecordBytes * static_cast<double>(n_per_rank) / hw_.intra_bw_bytes_per_sec;

  // §3.3: T_comm_inner(K) = max(0, T_p/K - T_c_overlap). We take T_c_overlap =
  // T_c (async can hide comm up to one full compute's worth, per the §3.3
  // "almost fully overlap" note). For K=1 compute-bound case this yields
  // T_comm_inner = 0; for comm-bound large-N, a finite residual remains.
  const double t_comm_inner = std::max(0.0, t_p / static_cast<double>(K) - t_c);

  // §3.5 scheduler overhead scales with K — each pipeline batch incurs one
  // dispatch. A single TD iteration with K batches pays K · T_sched. This
  // matches the SPEC §3.3 K_opt = sqrt(T_p / T_c_startup) derivation where
  // T_c_startup is the per-batch scheduler cost.
  const double t_sched_total = static_cast<double>(K) * hw_.scheduler_overhead_sec;

  const double t_step = t_c + t_comm_inner + t_sched_total;

  // Speedup vs Pattern 3 baseline — uses the same `n_atoms` so the ratio is
  // the CLI-facing "how much faster TD is than SD on this config".
  const PerfPrediction p3 = predict_pattern3(n_atoms);
  const double speedup = (t_step > 0.0) ? p3.t_step_sec / t_step : 1.0;

  PerfPrediction p;
  p.pattern_name = "Pattern1_TD";
  p.t_step_sec = t_step;
  p.recommended_K = K;
  p.speedup_vs_baseline = speedup;
  return p;
}

std::uint32_t PerfModel::K_opt(std::uint64_t n_atoms) const {
  // SPEC §3.4 candidate sweep — pick the K ∈ {1,2,4,8,16} that minimizes
  // T_step_TD(K). The analytic shortcut K_opt ≈ sqrt(T_p / T_sched) from
  // §3.3 agrees with this on the ramp-up regime (T_p > T_c) but overshoots
  // in the overlap-saturated regime where max(0, T_p/K - T_c) = 0; iterating
  // is the robust choice and matches what `rank()` needs internally.
  std::uint32_t best_k = 1U;
  double best_t = std::numeric_limits<double>::infinity();
  for (const auto k : kKCandidates) {
    const auto p = predict_pattern1(n_atoms, k);
    if (p.t_step_sec < best_t) {
      best_t = p.t_step_sec;
      best_k = k;
    }
  }
  return best_k;
}

double PerfModel::predict_step_gpu_sec(std::uint64_t n_atoms,
                                       const GpuCostTables& tables) const noexcept {
  // Same per-rank decomposition as Pattern 3 — single-subdomain single-rank
  // owns the full n_atoms/n_ranks slice. Pattern 1 / Pattern 2 GPU variants
  // land with M7 when heterogeneous cost modeling arrives.
  const std::uint64_t n_per_rank = n_atoms / hw_.n_ranks;
  return tables.step_total_sec(n_per_rank) + hw_.scheduler_overhead_sec;
}

// T7.10 — Pattern 2 cost model. Pure formula, no hardware probing — the
// caller provides the per-subdomain partition (via `Pattern2CostInputs`)
// and PerfModel composes the per-step seconds from the four cost-table
// stages introduced in T7.10 (halo_pack/send/unpack + nccl_allreduce_inner).
//
// Halo atom count per face: cubic-subdomain face area = N^(2/3). For
// non-cubic subdomains this is an upper bound (max-face dominates),
// adequate for the placeholder-coefficient era. T7.13 calibration revisits
// the geometry approximation if a tighter bound is warranted.
double PerfModel::predict_step_hybrid_seconds(const Pattern2CostInputs& inputs,
                                              const GpuCostTables& tables) const noexcept {
  const double t_inner = tables.step_total_sec(inputs.n_atoms_per_subdomain);

  // Face-area approximation: n_halo ≈ (n_per_subdomain)^(2/3). Cast to
  // uint64 for the cost-table API; floor is fine — placeholder coefficients
  // dominate the rounding error.
  const auto n_per_sd = static_cast<double>(inputs.n_atoms_per_subdomain);
  const auto n_halo_per_face =
      (n_per_sd > 0.0) ? static_cast<std::uint64_t>(std::pow(n_per_sd, 2.0 / 3.0)) : 0U;

  const double t_outer_per_neighbor = tables.outer_halo_sec_per_neighbor(n_halo_per_face);
  const double t_outer = static_cast<double>(inputs.n_face_neighbors) * t_outer_per_neighbor;

  // Reduction: thermo allreduce, fixed payload — `predict(0)` returns the
  // a-term (launch + execute) only; the b-term is set to ~0 in the table
  // factories so n_atoms-scaling is structurally absent for this stage.
  const double t_reduction = tables.nccl_allreduce_inner.predict(0U);

  return t_inner + t_outer + t_reduction + hw_.scheduler_overhead_sec;
}

std::uint32_t face_neighbors_count(const std::array<std::uint32_t, 3>& subdomains) noexcept {
  std::uint32_t n = 0U;
  for (const auto axis : subdomains) {
    if (axis >= 2U) {
      n += 2U;  // periodic interior subdomain has two neighbors per partitioned axis
    }
  }
  return n;
}

Pattern2Recommendation PerfModel::recommend_pattern2(std::uint64_t n_atoms_total,
                                                     const std::array<std::uint32_t, 3>& subdomains,
                                                     const GpuCostTables& tables) const noexcept {
  Pattern2Recommendation rec;

  // Pattern 1 baseline: all atoms on one subdomain on one device — bypass
  // hw_.n_ranks because that field reflects the *current* deployment, not
  // the hypothetical Pattern-1-collapse scenario we're comparing against.
  rec.t_pattern1_sec = tables.step_total_sec(n_atoms_total) + hw_.scheduler_overhead_sec;

  const std::uint64_t product = static_cast<std::uint64_t>(subdomains[0]) *
                                static_cast<std::uint64_t>(subdomains[1]) *
                                static_cast<std::uint64_t>(subdomains[2]);

  if (product < 2U) {
    // Degenerate input — caller passed [1,1,1]. No Pattern 2 to compare.
    rec.t_pattern2_sec = rec.t_pattern1_sec;
    rec.margin_fraction = 0.0;
    rec.recommended_pattern = "Pattern1";
    return rec;
  }

  Pattern2CostInputs p2;
  p2.n_atoms_per_subdomain = n_atoms_total / product;
  p2.n_face_neighbors = face_neighbors_count(subdomains);
  rec.t_pattern2_sec = predict_step_hybrid_seconds(p2, tables);

  rec.margin_fraction = (rec.t_pattern1_sec > 0.0)
                            ? (rec.t_pattern1_sec - rec.t_pattern2_sec) / rec.t_pattern1_sec
                            : 0.0;

  // OQ-M7-9: 5% margin threshold. Below this Pattern 1 is the safe default
  // (less coordination overhead, simpler debugging path).
  rec.recommended_pattern = (rec.margin_fraction >= 0.05) ? "Pattern2" : "Pattern1";
  return rec;
}

std::vector<PerfPrediction> PerfModel::rank(std::uint64_t n_atoms) const {
  std::vector<PerfPrediction> out;
  out.reserve(2U);
  out.push_back(predict_pattern3(n_atoms));
  out.push_back(predict_pattern1(n_atoms, K_opt(n_atoms)));

  // Ascending by `t_step_sec`. Deterministic tiebreak: lexicographic
  // `pattern_name` (so Pattern1_TD < Pattern3_SD on exact ties, matching the
  // SPEC §2.2 "recommend TD-first when comparable" bias).
  std::stable_sort(out.begin(), out.end(), [](const PerfPrediction& a, const PerfPrediction& b) {
    if (a.t_step_sec != b.t_step_sec) {
      return a.t_step_sec < b.t_step_sec;
    }
    return a.pattern_name < b.pattern_name;
  });
  return out;
}

}  // namespace tdmd
