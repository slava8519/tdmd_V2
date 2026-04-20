// SPEC: docs/specs/potentials/SPEC.md §6 (SNAP). Exec pack:
// docs/development/m8_execution_pack.md T8.4a/T8.4b.
//
// SnapPotential force body (T8.4b). Port of LAMMPS USER-SNAP
// `pair_snap.cpp::compute()` (~150 lines) on top of the SnaEngine port
// (`snap/sna_engine.cpp`). Together these two reproduce the full force path
// из LAMMPS's ML-SNAP package, with FP summation ordering preserved so the
// D-M8-7 acceptance gate (TDMD Fp64Reference ≡ LAMMPS FP64 ≤ 1e-12 rel) is
// load-bearing.
//
// The only structural deviation from upstream is the half-list → full-list
// bridge: TDMD's NeighborList is `newton on` half-list (each pair once with
// j > i), whereas LAMMPS SNAP requests `REQ_FULL` (each pair twice). We
// materialise a CSR full-list scratch from the half list inside compute()
// before entering SnaEngine's outer loop, which matches upstream 1:1.
//
// Virial sign convention. TDMD stores virial as `Σ F_i · (r_j - r_i)` — see
// eam_alloy.cpp:171 для precedent. This is the OPPOSITE sign from LAMMPS's
// internal `Σ F_i · (r_i - r_j)`, но the TDMD pressure formula
// `P = (2·KE - virial) / 3V` (runtime/simulation_engine.cpp:574) compensates
// so thermo pressure agrees with LAMMPS at the 1e-10 threshold.

#include "tdmd/potentials/snap.hpp"

#include "tdmd/potentials/snap/sna_engine.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace tdmd {

namespace {

void validate(const potentials::SnapData& data) {
  if (data.species.empty()) {
    throw std::invalid_argument("SnapPotential: SnapData has no species");
  }
  if (data.params.twojmax < 0 || (data.params.twojmax % 2) != 0) {
    throw std::invalid_argument("SnapPotential: twojmax must be non-negative and even, got " +
                                std::to_string(data.params.twojmax));
  }
  const std::size_t k = static_cast<std::size_t>(data.k_max);
  const std::size_t expected_linear = k + 1;
  const std::size_t expected_quad = expected_linear + k * (k + 1) / 2;
  const std::size_t expected = data.params.quadraticflag ? expected_quad : expected_linear;
  for (const auto& sp : data.species) {
    if (sp.beta.size() != expected) {
      std::ostringstream oss;
      oss << "SnapPotential: species '" << sp.name << "' has " << sp.beta.size()
          << " β coefficients but " << expected << " expected (twojmax=" << data.params.twojmax
          << ", k_max=" << data.k_max << ", quadraticflag=" << (data.params.quadraticflag ? 1 : 0)
          << ")";
      throw std::invalid_argument(oss.str());
    }
  }
  if (data.params.rcutfac <= 0.0) {
    throw std::invalid_argument("SnapPotential: rcutfac must be > 0");
  }
}

}  // namespace

SnapPotential::SnapPotential(potentials::SnapData data) : data_(std::move(data)) {
  validate(data_);

  // Engine construction: match sna.cpp constructor arg order. M8 parser rejects
  // chemflag=1 / switchinnerflag=1 / quadraticflag=1 (see snap_file.cpp), so
  // the corresponding engine paths stay dormant on this milestone. They are
  // retained in the port for M9+ expansion.
  engine_ = std::make_unique<snap_detail::SnaEngine>(data_.params.rfac0,
                                                     data_.params.twojmax,
                                                     data_.params.rmin0,
                                                     data_.params.switchflag ? 1 : 0,
                                                     data_.params.bzeroflag ? 1 : 0,
                                                     data_.params.chemflag ? 1 : 0,
                                                     data_.params.bnormflag ? 1 : 0,
                                                     data_.params.wselfallflag ? 1 : 0,
                                                     static_cast<int>(data_.species.size()),
                                                     data_.params.switchinnerflag ? 1 : 0);
  engine_->init();
}

SnapPotential::~SnapPotential() = default;
SnapPotential::SnapPotential(SnapPotential&&) noexcept = default;
SnapPotential& SnapPotential::operator=(SnapPotential&&) noexcept = default;

ForceResult SnapPotential::compute(AtomSoA& atoms, const NeighborList& neighbors, const Box& box) {
  ForceResult result;
  const std::size_t n = atoms.size();
  if (n == 0) {
    return result;
  }

  const auto& half_offsets = neighbors.page_offsets();
  const auto& half_ids = neighbors.neigh_ids();

  // --- Build symmetric full-list CSR from half-list.
  //
  // Pass A: count. Pass B: prefix-sum. Pass C: scatter.
  //
  // Within each atom's block we sort ascending after scatter so the resulting
  // iteration order is a deterministic function of (atom positions, cell
  // hashing, half-list layout). LAMMPS's own neighbor-list traversal order
  // is likewise deterministic but built from a different data structure; the
  // two orderings diverge at the last bits of FP accumulation in compute_ui
  // и compute_zi. For per-atom neighbour counts ≤ ~50 (bcc W with
  // rcut ≈ 4.73 Å → ~30 neighbours in the WSS shell) the accumulation error
  // stays below ~1e-14 relative per atom, comfortably inside the 1e-12
  // D-M8-7 threshold. Sort itself adds O(k·log k) per atom — dominated by
  // compute_uarray's O(jnum·idxu_max²) work per step.
  full_offsets_.assign(n + 1, 0);
  for (std::size_t i = 0; i < n; ++i) {
    const auto begin = half_offsets[i];
    const auto end = half_offsets[i + 1];
    full_offsets_[i + 1] += (end - begin);
    for (auto k = begin; k < end; ++k) {
      const auto j = half_ids[k];
      full_offsets_[static_cast<std::size_t>(j) + 1] += 1;
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    full_offsets_[i + 1] += full_offsets_[i];
  }
  full_ids_.assign(full_offsets_[n], 0);
  full_cursor_.assign(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    const auto begin = half_offsets[i];
    const auto end = half_offsets[i + 1];
    for (auto k = begin; k < end; ++k) {
      const std::uint32_t j = half_ids[k];
      full_ids_[full_offsets_[i] + full_cursor_[i]++] = j;
      full_ids_[full_offsets_[j] + full_cursor_[j]++] = static_cast<std::uint32_t>(i);
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    std::sort(full_ids_.begin() + static_cast<std::ptrdiff_t>(full_offsets_[i]),
              full_ids_.begin() + static_cast<std::ptrdiff_t>(full_offsets_[i + 1]));
  }

  // --- Grow engine scratch to the max per-atom neighbour count.
  std::size_t max_jnum = 0;
  for (std::size_t i = 0; i < n; ++i) {
    max_jnum =
        std::max(max_jnum, static_cast<std::size_t>(full_offsets_[i + 1] - full_offsets_[i]));
  }
  if (max_jnum > 0) {
    engine_->grow_rij(static_cast<int>(max_jnum));
  }

  // --- Outer loop over atoms (pair_snap.cpp lines 116–236, straightforward impl).
  const double* __restrict__ x_ptr = atoms.x.data();
  const double* __restrict__ y_ptr = atoms.y.data();
  const double* __restrict__ z_ptr = atoms.z.data();
  double* __restrict__ fx_ptr = atoms.fx.data();
  double* __restrict__ fy_ptr = atoms.fy.data();
  double* __restrict__ fz_ptr = atoms.fz.data();
  const SpeciesId* __restrict__ type_ptr = atoms.type.data();

  const std::size_t n_species = data_.species.size();
  const double rcutfac = data_.params.rcutfac;

  double pe = 0.0;
  double v_xx = 0.0;
  double v_yy = 0.0;
  double v_zz = 0.0;
  double v_xy = 0.0;
  double v_xz = 0.0;
  double v_yz = 0.0;

  for (std::size_t i = 0; i < n; ++i) {
    const auto itype = static_cast<std::size_t>(type_ptr[i]);
    // Species mapping. M8 parser rejects chemflag=1, so nelements_=1 inside
    // the engine и the only valid ielem is 0. We still pass `itype` through
    // data_.species[] lookups для radius_elem/weight_elem/beta because SnapData
    // is indexed by type directly (matches EAM convention — see eam_alloy.hpp
    // §"Species mapping").
    const int ielem = 0;
    const double radi = data_.species[itype].radius_elem;
    const double xtmp = x_ptr[i];
    const double ytmp = y_ptr[i];
    const double ztmp = z_ptr[i];

    const auto full_begin = full_offsets_[i];
    const auto full_end = full_offsets_[i + 1];

    // Filter by per-pair cutoff, fill engine scratch.
    int ninside = 0;
    for (auto k = full_begin; k < full_end; ++k) {
      const std::uint32_t j = full_ids_[k];
      const auto jtype = static_cast<std::size_t>(type_ptr[j]);
      const double radj = data_.species[jtype].radius_elem;
      const double wjelem = data_.species[jtype].weight_elem;

      const auto delta =
          box.unwrap_minimum_image(x_ptr[j] - xtmp, y_ptr[j] - ytmp, z_ptr[j] - ztmp);
      const double dx = delta[0];
      const double dy = delta[1];
      const double dz = delta[2];
      const double rsq = dx * dx + dy * dy + dz * dz;

      const double cutsq_ij =
          data_.rcut_sq_ab[potentials::SnapData::pair_index(itype, jtype, n_species)];
      // Mirror LAMMPS's `rsq > 1e-20` guard — defends against atoms that ended
      // up effectively co-located (zero distance would NaN the CG recursion).
      if (rsq < cutsq_ij && rsq > 1e-20) {
        engine_->rij[ninside][0] = dx;
        engine_->rij[ninside][1] = dy;
        engine_->rij[ninside][2] = dz;
        engine_->inside[ninside] = static_cast<int>(j);
        engine_->wj[ninside] = wjelem;
        engine_->rcutij[ninside] = (radi + radj) * rcutfac;
        ++ninside;
      }
    }

    // Three-pass bispectrum kernel: Ui → Yi → (per-neighbour) deidrj.
    engine_->compute_ui(ninside, ielem);
    // β vector: linear SNAP, β_k = coeff[k+1] (k ∈ [0, idxb_max)). β[0] is the
    // constant offset used only for energy — never touches compute_yi.
    const double* __restrict__ beta_ptr = data_.species[itype].beta.data() + 1;
    engine_->compute_yi(beta_ptr);

    double fij[3];
    for (int jj = 0; jj < ninside; ++jj) {
      const int j = engine_->inside[jj];
      engine_->compute_duidrj(jj);
      engine_->compute_deidrj(fij);

      fx_ptr[i] += fij[0];
      fy_ptr[i] += fij[1];
      fz_ptr[i] += fij[2];
      fx_ptr[j] -= fij[0];
      fy_ptr[j] -= fij[1];
      fz_ptr[j] -= fij[2];

      // Virial: TDMD convention F_i · (r_j - r_i). rij[jj] = r_j - r_i as
      // filled above.
      const double rij_x = engine_->rij[jj][0];
      const double rij_y = engine_->rij[jj][1];
      const double rij_z = engine_->rij[jj][2];
      v_xx += fij[0] * rij_x;
      v_yy += fij[1] * rij_y;
      v_zz += fij[2] * rij_z;
      v_xy += fij[0] * rij_y;
      v_xz += fij[0] * rij_z;
      v_yz += fij[1] * rij_z;
    }

    // Per-atom energy (eflag==always in TDMD — thermo reads PE every step).
    // E_i = β_0 + Σ_k β_k · B_{k,i}, summed linearly matches LAMMPS order
    // из pair_snap.cpp:216–217.
    engine_->compute_zi();
    engine_->compute_bi(ielem);
    double evdwl_i = data_.species[itype].beta[0];
    const double* __restrict__ blist = engine_->blist;
    const int idxb_max = engine_->idxb_max();
    for (int kidx = 0; kidx < idxb_max; ++kidx) {
      evdwl_i += beta_ptr[kidx] * blist[kidx];
    }
    pe += evdwl_i;
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
