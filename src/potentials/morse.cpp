#include "tdmd/potentials/morse.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace tdmd {

namespace {

bool all_finite_and_positive(const MorsePotential::PairParams& p) noexcept {
  return std::isfinite(p.D) && std::isfinite(p.alpha) && std::isfinite(p.r0) &&
         std::isfinite(p.cutoff) && p.D > 0.0 && p.alpha > 0.0 && p.r0 > 0.0 && p.cutoff > p.r0;
}

}  // namespace

MorsePotential::MorsePotential(const PairParams& params, CutoffStrategy strategy)
    : params_(params), strategy_(strategy) {
  if (!all_finite_and_positive(params)) {
    throw std::invalid_argument(
        "MorsePotential: parameters must be finite with D>0, alpha>0, r0>0, cutoff>r0");
  }
  // Shift constants are nonzero only for Strategy C. With both left at 0, the
  // compute() body's `g - g_rc` and `(r - cutoff)·g_rc` terms degenerate to the
  // raw Strategy A expressions — no per-pair branching needed on the hot path.
  if (strategy_ == CutoffStrategy::ShiftedForce) {
    const double e_rc = std::exp(-params_.alpha * (params_.cutoff - params_.r0));
    const double one_minus_e_rc = 1.0 - e_rc;
    g_at_rc_ = 2.0 * params_.D * params_.alpha * one_minus_e_rc * e_rc;
    e_pair_at_rc_ = params_.D * one_minus_e_rc * one_minus_e_rc - params_.D;
  }
}

MorsePotential::Result MorsePotential::compute(AtomSoA& atoms,
                                               const NeighborList& neighbors,
                                               const Box& box) const {
  Result result;
  const std::size_t n = atoms.size();
  if (n == 0) {
    return result;
  }

  const double* __restrict__ x_ptr = atoms.x.data();
  const double* __restrict__ y_ptr = atoms.y.data();
  const double* __restrict__ z_ptr = atoms.z.data();
  double* __restrict__ fx_ptr = atoms.fx.data();
  double* __restrict__ fy_ptr = atoms.fy.data();
  double* __restrict__ fz_ptr = atoms.fz.data();

  const auto& offsets = neighbors.page_offsets();
  const auto& ids = neighbors.neigh_ids();
  // NOTE: neighbors.neigh_r2() caches r² at list-build time. Atoms drift
  // between rebuilds, so we MUST recompute r² from live positions inside
  // the pair loop (see T1.11 differential-harness commit for the trace
  // that surfaced this). The cached value is still valid as a candidate
  // filter at build time but not as the physical r² passed to E and F.

  const double cutoff_sq = params_.cutoff * params_.cutoff;
  const double D = params_.D;
  const double alpha = params_.alpha;
  const double r0 = params_.r0;
  const double g_rc = g_at_rc_;
  const double e_pair_rc = e_pair_at_rc_;

  double pe = 0.0;
  double v_xx = 0.0;
  double v_yy = 0.0;
  double v_zz = 0.0;
  double v_xy = 0.0;
  double v_xz = 0.0;
  double v_yz = 0.0;

  for (std::size_t i = 0; i < n; ++i) {
    const std::uint64_t begin = offsets[i];
    const std::uint64_t end = offsets[i + 1];
    for (std::uint64_t k = begin; k < end; ++k) {
      const std::uint32_t j = ids[k];

      const auto delta =
          box.unwrap_minimum_image(x_ptr[j] - x_ptr[i], y_ptr[j] - y_ptr[i], z_ptr[j] - z_ptr[i]);
      const double dx = delta[0];
      const double dy = delta[1];
      const double dz = delta[2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > cutoff_sq) {
        continue;
      }
      const double r = std::sqrt(r2);
      const double e = std::exp(-alpha * (r - r0));
      const double one_minus_e = 1.0 - e;
      const double g = 2.0 * D * alpha * one_minus_e * e;
      const double g_shifted = g - g_rc;
      const double inv_r = 1.0 / r;
      const double factor = g_shifted * inv_r;

      const double fx = factor * dx;
      const double fy = factor * dy;
      const double fz = factor * dz;

      fx_ptr[i] += fx;
      fy_ptr[i] += fy;
      fz_ptr[i] += fz;
      fx_ptr[j] -= fx;
      fy_ptr[j] -= fy;
      fz_ptr[j] -= fz;

      // Strategy C (shifted-force) energy:
      //   E_shifted(r) = E_pair(r) - E_pair(r_c) - (r - r_c) · G(r_c)
      // For Strategy A (hard cutoff), e_pair_rc and g_rc are 0 (set in ctor),
      // and this line reduces to `pe += e_pair`.
      const double e_pair = D * one_minus_e * one_minus_e - D;
      pe += e_pair - e_pair_rc - (r - params_.cutoff) * g_rc;

      // Virial (Clausius): W_αβ = Σ_pairs F_i_α · r_ij_β, r_ij = r_j - r_i.
      v_xx += fx * dx;
      v_yy += fy * dy;
      v_zz += fz * dz;
      v_xy += fx * dy;
      v_xz += fx * dz;
      v_yz += fy * dz;
    }
  }

  result.potential_energy = pe;
  result.virial[0] = v_xx;
  result.virial[1] = v_yy;
  result.virial[2] = v_zz;
  result.virial[3] = v_xy;
  result.virial[4] = v_xz;
  result.virial[5] = v_yz;
  return result;
}

}  // namespace tdmd
