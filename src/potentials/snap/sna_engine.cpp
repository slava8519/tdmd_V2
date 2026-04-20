// -----------------------------------------------------------------------------
// SnaEngine — verbatim port of LAMMPS USER-SNAP `SNA` class (ML-SNAP package).
//
//   Original: verify/third_party/lammps/src/ML-SNAP/sna.cpp (1597 lines,
//     pin stable_22Jul2025_update4). Upstream authors: Aidan Thompson,
//     Christian Trott (Sandia National Labs). Upstream licence: GNU General
//     Public License v2 (GPLv2). The original LAMMPS header block is
//     reproduced immediately below.
//
// The port is structural: member names, loop ordering, FP accumulation
// sequence, and array layout are preserved 1:1 with upstream so that the
// D-M8-7 byte-exact differential (TDMD Fp64Reference ≡ LAMMPS FP64 ≤ 1e-12
// rel) is load-bearing. Only LAMMPS infrastructure is swapped for C++
// stdlib equivalents:
//   * `Pointers` base class         → none (dropped).
//   * `memory->create / destroy`    → raw new[]/delete[] with inline 2D/3D
//                                     pointer-array helpers mirroring
//                                     LAMMPS `memory.cpp`.
//   * `error->one/all` / warnings   → `throw std::runtime_error(...)`.
//   * `MathConst::MY_PI, MY_PI2`    → `M_PI`, `M_PI_2`.
//   * `MathSpecial::factorial`      → local `factorial(n)` helper.
//
// Flags `chemflag`, `switchinnerflag`, `quadraticflag` are accepted at the
// interface for parity with upstream but rejected at parse time in
// `snap_file.cpp` (T8.4a) с clear "M9+" message. The code paths are ported
// for future expansion, not exercised in M8.
//
// SPEC: docs/specs/potentials/SPEC.md §6 (SNAP); master spec §14 M8.
// Exec pack: docs/development/m8_execution_pack.md T8.4b.
//
// -----------------------------------------------------------------------------
// LAMMPS GPLv2 ATTRIBUTION (preserved verbatim from sna.cpp upstream header):
//
//   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
//   https://www.lammps.org/, Sandia National Laboratories
//   LAMMPS development team: developers@lammps.org
//
//   Copyright (2003) Sandia Corporation. Under the terms of Contract
//   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
//   certain rights in this software. This software is distributed under
//   the GNU General Public License.
//
//   Contributing authors: Aidan Thompson, Christian Trott, SNL
//
//   This implementation is based on the method outlined in Bartok[1] using
//   formulae from VMK[2].
//     [1] A. Bartok-Partay, "Gaussian Approximation..." Doctoral Thesis,
//         Cambridge University (2009).
//     [2] D. A. Varshalovich, A. N. Moskalev, V. K. Khersonskii, "Quantum
//         Theory of Angular Momentum," World Scientific (1988).
// -----------------------------------------------------------------------------

#include "tdmd/potentials/snap/sna_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 (M_PI * 0.5)
#endif

namespace tdmd::snap_detail {

namespace {

// ----- LAMMPS-compatible allocation helpers ---------------------------------
// Equivalents of `memory->create(ptr, n)` / `create(ptr, n, m)` /
// `create(ptr, n, m, p)`. Pointer arrays indexed with LAMMPS idiom
// (`arr[i][j]`, `arr[i][j][k]`) are contiguous for the deepest dimension.
// Zero-initialised to match LAMMPS `memory->create` behaviour (which calls
// `memset` under the hood).

template <typename T>
void create1d(T*& arr, std::size_t n) {
  arr = new T[n]{};
}

template <typename T>
void destroy1d(T*& arr) {
  delete[] arr;
  arr = nullptr;
}

template <typename T>
void create2d(T**& arr, std::size_t n, std::size_t m) {
  arr = new T*[n];
  T* data = new T[n * m]{};
  for (std::size_t i = 0; i < n; ++i) {
    arr[i] = data + i * m;
  }
}

template <typename T>
void destroy2d(T**& arr) {
  if (arr != nullptr) {
    delete[] arr[0];
    delete[] arr;
    arr = nullptr;
  }
}

template <typename T>
void create3d(T***& arr, std::size_t n, std::size_t m, std::size_t p) {
  arr = new T**[n];
  T** mid = new T*[n * m];
  T* data = new T[n * m * p]{};
  for (std::size_t i = 0; i < n; ++i) {
    arr[i] = mid + i * m;
    for (std::size_t j = 0; j < m; ++j) {
      arr[i][j] = data + (i * m + j) * p;
    }
  }
}

template <typename T>
void destroy3d(T***& arr) {
  if (arr != nullptr) {
    delete[] arr[0][0];
    delete[] arr[0];
    delete[] arr;
    arr = nullptr;
  }
}

// ----- math helpers --------------------------------------------------------

// LAMMPS `MathSpecial::factorial` uses a precomputed table up to ~170. The
// SNAP Clebsch-Gordan init only needs factorial up to (3*twojmax/2 + 1). For
// twojmax = 8 (our T6 fixture) that is 13; for any realistic twojmax (≤ 20)
// we stay within double-precision exactly-representable integers. Plain
// multiplication is enough and matches LAMMPS semantics for all reachable
// arguments.
inline double factorial(int n) {
  double f = 1.0;
  for (int k = 2; k <= n; ++k) {
    f *= static_cast<double>(k);
  }
  return f;
}

inline int imin(int a, int b) {
  return a < b ? a : b;
}
inline int imax(int a, int b) {
  return a > b ? a : b;
}

}  // namespace

// ----------------------------------------------------------------------------
// Constructor — mirrors sna.cpp `SNA::SNA(...)` lines 115–169.
// ----------------------------------------------------------------------------
SnaEngine::SnaEngine(double rfac0_in,
                     int twojmax_in,
                     double rmin0_in,
                     int switch_flag_in,
                     int bzero_flag_in,
                     int chem_flag_in,
                     int bnorm_flag_in,
                     int wselfall_flag_in,
                     int nelements_in,
                     int switch_inner_flag_in) {
  wself_ = 1.0;

  rfac0_ = rfac0_in;
  rmin0_ = rmin0_in;
  switch_flag_ = switch_flag_in;
  switch_inner_flag_ = switch_inner_flag_in;
  bzero_flag_ = bzero_flag_in;
  chem_flag_ = chem_flag_in;
  bnorm_flag_ = bnorm_flag_in;
  wselfall_flag_ = wselfall_flag_in;

  // LAMMPS warns (not errors) when bnorm_flag != chem_flag. We silently
  // respect user intent — parser already rejects chem_flag=1, so this branch
  // cannot trigger in M8 practice.

  if (chem_flag_) {
    nelements_ = nelements_in;
  } else {
    nelements_ = 1;
  }

  twojmax_ = twojmax_in;

  compute_ncoeff();

  build_indexlist();
  create_twojmax_arrays();

  if (bzero_flag_) {
    double www = wself_ * wself_ * wself_;
    for (int j = 0; j <= twojmax_; ++j) {
      if (bnorm_flag_) {
        bzero[j] = www;
      } else {
        bzero[j] = www * (j + 1);
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Destructor.
// ----------------------------------------------------------------------------
SnaEngine::~SnaEngine() {
  destroy2d(rij);
  destroy1d(inside);
  destroy1d(wj);
  destroy1d(rcutij);
  destroy1d(sinnerij);
  destroy1d(dinnerij);
  if (chem_flag_) {
    destroy1d(element);
  }
  destroy2d(ulist_r_ij);
  destroy2d(ulist_i_ij);
  delete[] idxz;
  idxz = nullptr;
  delete[] idxb;
  idxb = nullptr;
  destroy_twojmax_arrays();
}

// ----------------------------------------------------------------------------
// build_indexlist — sna.cpp lines 189–307, verbatim apart from memory-
// helper substitutions.
// ----------------------------------------------------------------------------
void SnaEngine::build_indexlist() {
  const int jdim = twojmax_ + 1;
  create3d(idxcg_block,
           static_cast<std::size_t>(jdim),
           static_cast<std::size_t>(jdim),
           static_cast<std::size_t>(jdim));

  int idxcg_count = 0;
  for (int j1 = 0; j1 <= twojmax_; ++j1) {
    for (int j2 = 0; j2 <= j1; ++j2) {
      for (int j = j1 - j2; j <= imin(twojmax_, j1 + j2); j += 2) {
        idxcg_block[j1][j2][j] = idxcg_count;
        for (int m1 = 0; m1 <= j1; ++m1) {
          for (int m2 = 0; m2 <= j2; ++m2) {
            idxcg_count++;
          }
        }
      }
    }
  }
  idxcg_max_ = idxcg_count;

  // index list for uarray (both halves).
  create1d(idxu_block, static_cast<std::size_t>(jdim));

  int idxu_count = 0;
  for (int j = 0; j <= twojmax_; ++j) {
    idxu_block[j] = idxu_count;
    for (int mb = 0; mb <= j; ++mb) {
      for (int ma = 0; ma <= j; ++ma) {
        idxu_count++;
      }
    }
  }
  idxu_max_ = idxu_count;

  // index list for beta and B.
  int idxb_count = 0;
  for (int j1 = 0; j1 <= twojmax_; ++j1) {
    for (int j2 = 0; j2 <= j1; ++j2) {
      for (int j = j1 - j2; j <= imin(twojmax_, j1 + j2); j += 2) {
        if (j >= j1) {
          idxb_count++;
        }
      }
    }
  }
  idxb_max_ = idxb_count;
  idxb = new SnaBIndices[idxb_max_];

  idxb_count = 0;
  for (int j1 = 0; j1 <= twojmax_; ++j1) {
    for (int j2 = 0; j2 <= j1; ++j2) {
      for (int j = j1 - j2; j <= imin(twojmax_, j1 + j2); j += 2) {
        if (j >= j1) {
          idxb[idxb_count].j1 = j1;
          idxb[idxb_count].j2 = j2;
          idxb[idxb_count].j = j;
          idxb_count++;
        }
      }
    }
  }

  // reverse index list for beta and b.
  create3d(idxb_block,
           static_cast<std::size_t>(jdim),
           static_cast<std::size_t>(jdim),
           static_cast<std::size_t>(jdim));
  idxb_count = 0;
  for (int j1 = 0; j1 <= twojmax_; ++j1) {
    for (int j2 = 0; j2 <= j1; ++j2) {
      for (int j = j1 - j2; j <= imin(twojmax_, j1 + j2); j += 2) {
        if (j >= j1) {
          idxb_block[j1][j2][j] = idxb_count;
          idxb_count++;
        }
      }
    }
  }

  // index list for zlist.
  int idxz_count = 0;
  for (int j1 = 0; j1 <= twojmax_; ++j1) {
    for (int j2 = 0; j2 <= j1; ++j2) {
      for (int j = j1 - j2; j <= imin(twojmax_, j1 + j2); j += 2) {
        for (int mb = 0; 2 * mb <= j; ++mb) {
          for (int ma = 0; ma <= j; ++ma) {
            idxz_count++;
          }
        }
      }
    }
  }
  idxz_max_ = idxz_count;
  idxz = new SnaZIndices[idxz_max_];

  create3d(idxz_block,
           static_cast<std::size_t>(jdim),
           static_cast<std::size_t>(jdim),
           static_cast<std::size_t>(jdim));

  idxz_count = 0;
  for (int j1 = 0; j1 <= twojmax_; ++j1) {
    for (int j2 = 0; j2 <= j1; ++j2) {
      for (int j = j1 - j2; j <= imin(twojmax_, j1 + j2); j += 2) {
        idxz_block[j1][j2][j] = idxz_count;

        for (int mb = 0; 2 * mb <= j; ++mb) {
          for (int ma = 0; ma <= j; ++ma) {
            idxz[idxz_count].j1 = j1;
            idxz[idxz_count].j2 = j2;
            idxz[idxz_count].j = j;
            idxz[idxz_count].ma1min = imax(0, (2 * ma - j - j2 + j1) / 2);
            idxz[idxz_count].ma2max = (2 * ma - j - (2 * idxz[idxz_count].ma1min - j1) + j2) / 2;
            idxz[idxz_count].na =
                imin(j1, (2 * ma - j + j2 + j1) / 2) - idxz[idxz_count].ma1min + 1;
            idxz[idxz_count].mb1min = imax(0, (2 * mb - j - j2 + j1) / 2);
            idxz[idxz_count].mb2max = (2 * mb - j - (2 * idxz[idxz_count].mb1min - j1) + j2) / 2;
            idxz[idxz_count].nb =
                imin(j1, (2 * mb - j + j2 + j1) / 2) - idxz[idxz_count].mb1min + 1;

            const int jju = idxu_block[j] + (j + 1) * mb + ma;
            idxz[idxz_count].jju = jju;

            idxz_count++;
          }
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
// init — sna.cpp line 311.
// ----------------------------------------------------------------------------
void SnaEngine::init() {
  init_clebsch_gordan();
  init_rootpqarray();
}

// ----------------------------------------------------------------------------
// grow_rij — sna.cpp lines 318–342. Idempotent shrink, full realloc on grow.
// ----------------------------------------------------------------------------
void SnaEngine::grow_rij(int newnmax) {
  if (newnmax <= nmax_) {
    return;
  }

  nmax_ = newnmax;

  destroy2d(rij);
  destroy1d(inside);
  destroy1d(wj);
  destroy1d(rcutij);
  destroy1d(sinnerij);
  destroy1d(dinnerij);
  if (chem_flag_) {
    destroy1d(element);
  }
  destroy2d(ulist_r_ij);
  destroy2d(ulist_i_ij);

  create2d(rij, static_cast<std::size_t>(nmax_), 3);
  create1d(inside, static_cast<std::size_t>(nmax_));
  create1d(wj, static_cast<std::size_t>(nmax_));
  create1d(rcutij, static_cast<std::size_t>(nmax_));
  create1d(sinnerij, static_cast<std::size_t>(nmax_));
  create1d(dinnerij, static_cast<std::size_t>(nmax_));
  if (chem_flag_) {
    create1d(element, static_cast<std::size_t>(nmax_));
  }
  create2d(ulist_r_ij, static_cast<std::size_t>(nmax_), static_cast<std::size_t>(idxu_max_));
  create2d(ulist_i_ij, static_cast<std::size_t>(nmax_), static_cast<std::size_t>(idxu_max_));
}

// ----------------------------------------------------------------------------
// compute_ui — sna.cpp lines 348–375.
// ----------------------------------------------------------------------------
void SnaEngine::compute_ui(int jnum, int ielem) {
  double rsq, r, x, y, z, z0, theta0;

  zero_uarraytot(ielem);

  for (int j = 0; j < jnum; ++j) {
    x = rij[j][0];
    y = rij[j][1];
    z = rij[j][2];
    rsq = x * x + y * y + z * z;
    r = std::sqrt(rsq);

    theta0 = (r - rmin0_) * rfac0_ * M_PI / (rcutij[j] - rmin0_);
    z0 = r / std::tan(theta0);

    compute_uarray(x, y, z, z0, r, j);
    add_uarraytot(r, j);
  }
}

// ----------------------------------------------------------------------------
// compute_zi — sna.cpp lines 381–448.
// ----------------------------------------------------------------------------
void SnaEngine::compute_zi() {
  int idouble = 0;
  double* zptr_r;
  double* zptr_i;
  for (int elem1 = 0; elem1 < nelements_; ++elem1) {
    for (int elem2 = 0; elem2 < nelements_; ++elem2) {
      zptr_r = &zlist_r[idouble * idxz_max_];
      zptr_i = &zlist_i[idouble * idxz_max_];

      for (int jjz = 0; jjz < idxz_max_; ++jjz) {
        const int j1 = idxz[jjz].j1;
        const int j2 = idxz[jjz].j2;
        const int j = idxz[jjz].j;
        const int ma1min = idxz[jjz].ma1min;
        const int ma2max = idxz[jjz].ma2max;
        const int na = idxz[jjz].na;
        const int mb1min = idxz[jjz].mb1min;
        const int mb2max = idxz[jjz].mb2max;
        const int nb = idxz[jjz].nb;

        const double* cgblock = cglist + idxcg_block[j1][j2][j];

        zptr_r[jjz] = 0.0;
        zptr_i[jjz] = 0.0;

        int jju1 = idxu_block[j1] + (j1 + 1) * mb1min;
        int jju2 = idxu_block[j2] + (j2 + 1) * mb2max;
        int icgb = mb1min * (j2 + 1) + mb2max;
        for (int ib = 0; ib < nb; ++ib) {
          double suma1_r = 0.0;
          double suma1_i = 0.0;

          const double* u1_r = &ulisttot_r[elem1 * idxu_max_ + jju1];
          const double* u1_i = &ulisttot_i[elem1 * idxu_max_ + jju1];
          const double* u2_r = &ulisttot_r[elem2 * idxu_max_ + jju2];
          const double* u2_i = &ulisttot_i[elem2 * idxu_max_ + jju2];

          int ma1 = ma1min;
          int ma2 = ma2max;
          int icga = ma1min * (j2 + 1) + ma2max;

          for (int ia = 0; ia < na; ++ia) {
            suma1_r += cgblock[icga] * (u1_r[ma1] * u2_r[ma2] - u1_i[ma1] * u2_i[ma2]);
            suma1_i += cgblock[icga] * (u1_r[ma1] * u2_i[ma2] + u1_i[ma1] * u2_r[ma2]);
            ma1++;
            ma2--;
            icga += j2;
          }

          zptr_r[jjz] += cgblock[icgb] * suma1_r;
          zptr_i[jjz] += cgblock[icgb] * suma1_i;

          jju1 += j1 + 1;
          jju2 -= j2 + 1;
          icgb += j2;
        }
        if (bnorm_flag_) {
          zptr_r[jjz] /= (j + 1);
          zptr_i[jjz] /= (j + 1);
        }
      }
      idouble++;
    }
  }
}

// ----------------------------------------------------------------------------
// compute_yi — sna.cpp lines 454–563.
// ----------------------------------------------------------------------------
void SnaEngine::compute_yi(const double* beta) {
  int jju;
  double betaj;
  int itriple;

  for (int ielem1 = 0; ielem1 < nelements_; ++ielem1) {
    for (int j = 0; j <= twojmax_; ++j) {
      jju = idxu_block[j];
      for (int mb = 0; 2 * mb <= j; ++mb) {
        for (int ma = 0; ma <= j; ++ma) {
          ylist_r[ielem1 * idxu_max_ + jju] = 0.0;
          ylist_i[ielem1 * idxu_max_ + jju] = 0.0;
          jju++;
        }
      }
    }
  }

  for (int elem1 = 0; elem1 < nelements_; ++elem1) {
    for (int elem2 = 0; elem2 < nelements_; ++elem2) {
      for (int jjz = 0; jjz < idxz_max_; ++jjz) {
        const int j1 = idxz[jjz].j1;
        const int j2 = idxz[jjz].j2;
        const int j = idxz[jjz].j;
        const int ma1min = idxz[jjz].ma1min;
        const int ma2max = idxz[jjz].ma2max;
        const int na = idxz[jjz].na;
        const int mb1min = idxz[jjz].mb1min;
        const int mb2max = idxz[jjz].mb2max;
        const int nb = idxz[jjz].nb;

        const double* cgblock = cglist + idxcg_block[j1][j2][j];

        double ztmp_r = 0.0;
        double ztmp_i = 0.0;

        int jju1 = idxu_block[j1] + (j1 + 1) * mb1min;
        int jju2 = idxu_block[j2] + (j2 + 1) * mb2max;
        int icgb = mb1min * (j2 + 1) + mb2max;
        for (int ib = 0; ib < nb; ++ib) {
          double suma1_r = 0.0;
          double suma1_i = 0.0;

          const double* u1_r = &ulisttot_r[elem1 * idxu_max_ + jju1];
          const double* u1_i = &ulisttot_i[elem1 * idxu_max_ + jju1];
          const double* u2_r = &ulisttot_r[elem2 * idxu_max_ + jju2];
          const double* u2_i = &ulisttot_i[elem2 * idxu_max_ + jju2];

          int ma1 = ma1min;
          int ma2 = ma2max;
          int icga = ma1min * (j2 + 1) + ma2max;

          for (int ia = 0; ia < na; ++ia) {
            suma1_r += cgblock[icga] * (u1_r[ma1] * u2_r[ma2] - u1_i[ma1] * u2_i[ma2]);
            suma1_i += cgblock[icga] * (u1_r[ma1] * u2_i[ma2] + u1_i[ma1] * u2_r[ma2]);
            ma1++;
            ma2--;
            icga += j2;
          }

          ztmp_r += cgblock[icgb] * suma1_r;
          ztmp_i += cgblock[icgb] * suma1_i;

          jju1 += j1 + 1;
          jju2 -= j2 + 1;
          icgb += j2;
        }

        if (bnorm_flag_) {
          ztmp_i /= j + 1;
          ztmp_r /= j + 1;
        }

        jju = idxz[jjz].jju;
        for (int elem3 = 0; elem3 < nelements_; ++elem3) {
          if (j >= j1) {
            const int jjb = idxb_block[j1][j2][j];
            itriple = ((elem1 * nelements_ + elem2) * nelements_ + elem3) * idxb_max_ + jjb;
            if (j1 == j) {
              if (j2 == j) {
                betaj = 3 * beta[itriple];
              } else {
                betaj = 2 * beta[itriple];
              }
            } else {
              betaj = beta[itriple];
            }
          } else if (j >= j2) {
            const int jjb = idxb_block[j][j2][j1];
            itriple = ((elem3 * nelements_ + elem2) * nelements_ + elem1) * idxb_max_ + jjb;
            if (j2 == j) {
              betaj = 2 * beta[itriple];
            } else {
              betaj = beta[itriple];
            }
          } else {
            const int jjb = idxb_block[j2][j][j1];
            itriple = ((elem2 * nelements_ + elem3) * nelements_ + elem1) * idxb_max_ + jjb;
            betaj = beta[itriple];
          }

          if (!bnorm_flag_ && j1 > j) {
            betaj *= (j1 + 1) / (j + 1.0);
          }

          ylist_r[elem3 * idxu_max_ + jju] += betaj * ztmp_r;
          ylist_i[elem3 * idxu_max_ + jju] += betaj * ztmp_i;
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
// compute_deidrj — sna.cpp lines 569–630.
// ----------------------------------------------------------------------------
void SnaEngine::compute_deidrj(double* dedr) {
  for (int k = 0; k < 3; ++k) {
    dedr[k] = 0.0;
  }

  const int jelem = elem_duarray_;
  for (int j = 0; j <= twojmax_; ++j) {
    int jju = idxu_block[j];

    for (int mb = 0; 2 * mb < j; ++mb) {
      for (int ma = 0; ma <= j; ++ma) {
        double* dudr_r = dulist_r[jju];
        double* dudr_i = dulist_i[jju];
        double jjjmambyarray_r = ylist_r[jelem * idxu_max_ + jju];
        double jjjmambyarray_i = ylist_i[jelem * idxu_max_ + jju];

        for (int k = 0; k < 3; ++k) {
          dedr[k] += dudr_r[k] * jjjmambyarray_r + dudr_i[k] * jjjmambyarray_i;
        }
        jju++;
      }
    }

    // For j even, handle middle column.
    if (j % 2 == 0) {
      int mb = j / 2;
      for (int ma = 0; ma < mb; ++ma) {
        double* dudr_r = dulist_r[jju];
        double* dudr_i = dulist_i[jju];
        double jjjmambyarray_r = ylist_r[jelem * idxu_max_ + jju];
        double jjjmambyarray_i = ylist_i[jelem * idxu_max_ + jju];

        for (int k = 0; k < 3; ++k) {
          dedr[k] += dudr_r[k] * jjjmambyarray_r + dudr_i[k] * jjjmambyarray_i;
        }
        jju++;
      }

      double* dudr_r = dulist_r[jju];
      double* dudr_i = dulist_i[jju];
      double jjjmambyarray_r = ylist_r[jelem * idxu_max_ + jju];
      double jjjmambyarray_i = ylist_i[jelem * idxu_max_ + jju];

      for (int k = 0; k < 3; ++k) {
        dedr[k] += (dudr_r[k] * jjjmambyarray_r + dudr_i[k] * jjjmambyarray_i) * 0.5;
      }
    }
  }

  for (int k = 0; k < 3; ++k) {
    dedr[k] *= 2.0;
  }
}

// ----------------------------------------------------------------------------
// compute_bi — sna.cpp lines 636–717.
// ----------------------------------------------------------------------------
void SnaEngine::compute_bi(int ielem) {
  int itriple = 0;
  int idouble = 0;
  for (int elem1 = 0; elem1 < nelements_; ++elem1) {
    for (int elem2 = 0; elem2 < nelements_; ++elem2) {
      double* zptr_r = &zlist_r[idouble * idxz_max_];
      double* zptr_i = &zlist_i[idouble * idxz_max_];

      for (int elem3 = 0; elem3 < nelements_; ++elem3) {
        for (int jjb = 0; jjb < idxb_max_; ++jjb) {
          const int j1 = idxb[jjb].j1;
          const int j2 = idxb[jjb].j2;
          const int j = idxb[jjb].j;

          int jjz = idxz_block[j1][j2][j];
          int jju = idxu_block[j];
          double sumzu = 0.0;
          for (int mb = 0; 2 * mb < j; ++mb) {
            for (int ma = 0; ma <= j; ++ma) {
              sumzu += ulisttot_r[elem3 * idxu_max_ + jju] * zptr_r[jjz] +
                       ulisttot_i[elem3 * idxu_max_ + jju] * zptr_i[jjz];
              jjz++;
              jju++;
            }
          }

          if (j % 2 == 0) {
            int mb = j / 2;
            for (int ma = 0; ma < mb; ++ma) {
              sumzu += ulisttot_r[elem3 * idxu_max_ + jju] * zptr_r[jjz] +
                       ulisttot_i[elem3 * idxu_max_ + jju] * zptr_i[jjz];
              jjz++;
              jju++;
            }

            sumzu += 0.5 * (ulisttot_r[elem3 * idxu_max_ + jju] * zptr_r[jjz] +
                            ulisttot_i[elem3 * idxu_max_ + jju] * zptr_i[jjz]);
          }

          blist[itriple * idxb_max_ + jjb] = 2.0 * sumzu;
        }
        itriple++;
      }
      idouble++;
    }
  }

  // apply bzero shift
  if (bzero_flag_) {
    if (!wselfall_flag_) {
      itriple = (ielem * nelements_ + ielem) * nelements_ + ielem;
      for (int jjb = 0; jjb < idxb_max_; ++jjb) {
        const int j = idxb[jjb].j;
        blist[itriple * idxb_max_ + jjb] -= bzero[j];
      }
    } else {
      int it = 0;
      for (int elem1 = 0; elem1 < nelements_; ++elem1) {
        for (int elem2 = 0; elem2 < nelements_; ++elem2) {
          for (int elem3 = 0; elem3 < nelements_; ++elem3) {
            for (int jjb = 0; jjb < idxb_max_; ++jjb) {
              const int j = idxb[jjb].j;
              blist[it * idxb_max_ + jjb] -= bzero[j];
            }
            it++;
          }
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
// compute_duidrj — sna.cpp lines 962–984.
// ----------------------------------------------------------------------------
void SnaEngine::compute_duidrj(int jj) {
  double rsq, r, x, y, z, z0, theta0, cs, sn;
  double dz0dr;
  const double rcut = rcutij[jj];

  x = rij[jj][0];
  y = rij[jj][1];
  z = rij[jj][2];
  rsq = x * x + y * y + z * z;
  r = std::sqrt(rsq);
  const double rscale0 = rfac0_ * M_PI / (rcut - rmin0_);
  theta0 = (r - rmin0_) * rscale0;
  cs = std::cos(theta0);
  sn = std::sin(theta0);
  z0 = r * cs / sn;
  dz0dr = z0 / r - (r * rscale0) * (rsq + z0 * z0) / rsq;

  if (chem_flag_) {
    elem_duarray_ = element[jj];
  } else {
    elem_duarray_ = 0;
  }

  compute_duarray(x, y, z, z0, r, dz0dr, wj[jj], rcut, jj);
}

// ----------------------------------------------------------------------------
// zero_uarraytot — sna.cpp lines 988–1006.
// ----------------------------------------------------------------------------
void SnaEngine::zero_uarraytot(int ielem) {
  for (int jelem = 0; jelem < nelements_; ++jelem) {
    for (int j = 0; j <= twojmax_; ++j) {
      int jju = idxu_block[j];
      for (int mb = 0; mb <= j; ++mb) {
        for (int ma = 0; ma <= j; ++ma) {
          ulisttot_r[jelem * idxu_max_ + jju] = 0.0;
          ulisttot_i[jelem * idxu_max_ + jju] = 0.0;

          // utot(j,ma,ma) = wself, sometimes.
          if (jelem == ielem || wselfall_flag_) {
            if (ma == mb) {
              ulisttot_r[jelem * idxu_max_ + jju] = wself_;
            }
          }
          jju++;
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
// add_uarraytot — sna.cpp lines 1012–1037.
// ----------------------------------------------------------------------------
void SnaEngine::add_uarraytot(double r, int jj) {
  double sfac = compute_sfac(r, rcutij[jj], sinnerij[jj], dinnerij[jj]);
  sfac *= wj[jj];

  int jelem;
  if (chem_flag_) {
    jelem = element[jj];
  } else {
    jelem = 0;
  }

  const double* ulist_r = ulist_r_ij[jj];
  const double* ulist_i = ulist_i_ij[jj];

  for (int j = 0; j <= twojmax_; ++j) {
    int jju = idxu_block[j];
    for (int mb = 0; mb <= j; ++mb) {
      for (int ma = 0; ma <= j; ++ma) {
        ulisttot_r[jelem * idxu_max_ + jju] += sfac * ulist_r[jju];
        ulisttot_i[jelem * idxu_max_ + jju] += sfac * ulist_i[jju];
        jju++;
      }
    }
  }
}

// ----------------------------------------------------------------------------
// compute_uarray — sna.cpp lines 1043–1126.
// ----------------------------------------------------------------------------
void SnaEngine::compute_uarray(double x, double y, double z, double z0, double r, int jj) {
  double r0inv;
  double a_r, b_r, a_i, b_i;
  double rootpq;

  r0inv = 1.0 / std::sqrt(r * r + z0 * z0);
  a_r = r0inv * z0;
  a_i = -r0inv * z;
  b_r = r0inv * y;
  b_i = -r0inv * x;

  double* ulist_r = ulist_r_ij[jj];
  double* ulist_i = ulist_i_ij[jj];

  ulist_r[0] = 1.0;
  ulist_i[0] = 0.0;

  for (int j = 1; j <= twojmax_; ++j) {
    int jju = idxu_block[j];
    int jjup = idxu_block[j - 1];

    // fill in left side of matrix layer from previous layer.
    for (int mb = 0; 2 * mb <= j; ++mb) {
      ulist_r[jju] = 0.0;
      ulist_i[jju] = 0.0;

      for (int ma = 0; ma < j; ++ma) {
        rootpq = rootpqarray[j - ma][j - mb];
        ulist_r[jju] += rootpq * (a_r * ulist_r[jjup] + a_i * ulist_i[jjup]);
        ulist_i[jju] += rootpq * (a_r * ulist_i[jjup] - a_i * ulist_r[jjup]);

        rootpq = rootpqarray[ma + 1][j - mb];
        ulist_r[jju + 1] = -rootpq * (b_r * ulist_r[jjup] + b_i * ulist_i[jjup]);
        ulist_i[jju + 1] = -rootpq * (b_r * ulist_i[jjup] - b_i * ulist_r[jjup]);
        jju++;
        jjup++;
      }
      jju++;
    }

    // copy left side to right side with inversion symmetry VMK 4.4(2).
    jju = idxu_block[j];
    jjup = jju + (j + 1) * (j + 1) - 1;
    int mbpar = 1;
    for (int mb = 0; 2 * mb <= j; ++mb) {
      int mapar = mbpar;
      for (int ma = 0; ma <= j; ++ma) {
        if (mapar == 1) {
          ulist_r[jjup] = ulist_r[jju];
          ulist_i[jjup] = -ulist_i[jju];
        } else {
          ulist_r[jjup] = -ulist_r[jju];
          ulist_i[jjup] = ulist_i[jju];
        }
        mapar = -mapar;
        jju++;
        jjup--;
      }
      mbpar = -mbpar;
    }
  }
}

// ----------------------------------------------------------------------------
// compute_duarray — sna.cpp lines 1133–1286.
// ----------------------------------------------------------------------------
void SnaEngine::compute_duarray(double x,
                                double y,
                                double z,
                                double z0,
                                double r,
                                double dz0dr,
                                double wj_local,
                                double rcut,
                                int jj) {
  double r0inv;
  double a_r, a_i, b_r, b_i;
  double da_r[3], da_i[3], db_r[3], db_i[3];
  double dz0[3], dr0inv[3], dr0invdr;
  double rootpq;

  const double rinv = 1.0 / r;
  const double ux = x * rinv;
  const double uy = y * rinv;
  const double uz = z * rinv;

  r0inv = 1.0 / std::sqrt(r * r + z0 * z0);
  a_r = z0 * r0inv;
  a_i = -z * r0inv;
  b_r = y * r0inv;
  b_i = -x * r0inv;

  dr0invdr = -std::pow(r0inv, 3.0) * (r + z0 * dz0dr);

  dr0inv[0] = dr0invdr * ux;
  dr0inv[1] = dr0invdr * uy;
  dr0inv[2] = dr0invdr * uz;

  dz0[0] = dz0dr * ux;
  dz0[1] = dz0dr * uy;
  dz0[2] = dz0dr * uz;

  for (int k = 0; k < 3; ++k) {
    da_r[k] = dz0[k] * r0inv + z0 * dr0inv[k];
    da_i[k] = -z * dr0inv[k];
  }
  da_i[2] += -r0inv;

  for (int k = 0; k < 3; ++k) {
    db_r[k] = y * dr0inv[k];
    db_i[k] = -x * dr0inv[k];
  }
  db_i[0] += -r0inv;
  db_r[1] += r0inv;

  const double* ulist_r = ulist_r_ij[jj];
  const double* ulist_i = ulist_i_ij[jj];

  dulist_r[0][0] = 0.0;
  dulist_r[0][1] = 0.0;
  dulist_r[0][2] = 0.0;
  dulist_i[0][0] = 0.0;
  dulist_i[0][1] = 0.0;
  dulist_i[0][2] = 0.0;

  for (int j = 1; j <= twojmax_; ++j) {
    int jju = idxu_block[j];
    int jjup = idxu_block[j - 1];
    for (int mb = 0; 2 * mb <= j; ++mb) {
      dulist_r[jju][0] = 0.0;
      dulist_r[jju][1] = 0.0;
      dulist_r[jju][2] = 0.0;
      dulist_i[jju][0] = 0.0;
      dulist_i[jju][1] = 0.0;
      dulist_i[jju][2] = 0.0;

      for (int ma = 0; ma < j; ++ma) {
        rootpq = rootpqarray[j - ma][j - mb];
        for (int k = 0; k < 3; ++k) {
          dulist_r[jju][k] += rootpq * (da_r[k] * ulist_r[jjup] + da_i[k] * ulist_i[jjup] +
                                        a_r * dulist_r[jjup][k] + a_i * dulist_i[jjup][k]);
          dulist_i[jju][k] += rootpq * (da_r[k] * ulist_i[jjup] - da_i[k] * ulist_r[jjup] +
                                        a_r * dulist_i[jjup][k] - a_i * dulist_r[jjup][k]);
        }

        rootpq = rootpqarray[ma + 1][j - mb];
        for (int k = 0; k < 3; ++k) {
          dulist_r[jju + 1][k] = -rootpq * (db_r[k] * ulist_r[jjup] + db_i[k] * ulist_i[jjup] +
                                            b_r * dulist_r[jjup][k] + b_i * dulist_i[jjup][k]);
          dulist_i[jju + 1][k] = -rootpq * (db_r[k] * ulist_i[jjup] - db_i[k] * ulist_r[jjup] +
                                            b_r * dulist_i[jjup][k] - b_i * dulist_r[jjup][k]);
        }
        jju++;
        jjup++;
      }
      jju++;
    }

    // copy left side to right side with inversion symmetry VMK 4.4(2).
    jju = idxu_block[j];
    jjup = jju + (j + 1) * (j + 1) - 1;
    int mbpar = 1;
    for (int mb = 0; 2 * mb <= j; ++mb) {
      int mapar = mbpar;
      for (int ma = 0; ma <= j; ++ma) {
        if (mapar == 1) {
          for (int k = 0; k < 3; ++k) {
            dulist_r[jjup][k] = dulist_r[jju][k];
            dulist_i[jjup][k] = -dulist_i[jju][k];
          }
        } else {
          for (int k = 0; k < 3; ++k) {
            dulist_r[jjup][k] = -dulist_r[jju][k];
            dulist_i[jjup][k] = dulist_i[jju][k];
          }
        }
        mapar = -mapar;
        jju++;
        jjup--;
      }
      mbpar = -mbpar;
    }
  }

  double sfac = compute_sfac(r, rcut, sinnerij[jj], dinnerij[jj]);
  double dsfac = compute_dsfac(r, rcut, sinnerij[jj], dinnerij[jj]);

  sfac *= wj_local;
  dsfac *= wj_local;
  for (int j = 0; j <= twojmax_; ++j) {
    int jju = idxu_block[j];
    for (int mb = 0; 2 * mb <= j; ++mb) {
      for (int ma = 0; ma <= j; ++ma) {
        dulist_r[jju][0] = dsfac * ulist_r[jju] * ux + sfac * dulist_r[jju][0];
        dulist_i[jju][0] = dsfac * ulist_i[jju] * ux + sfac * dulist_i[jju][0];
        dulist_r[jju][1] = dsfac * ulist_r[jju] * uy + sfac * dulist_r[jju][1];
        dulist_i[jju][1] = dsfac * ulist_i[jju] * uy + sfac * dulist_i[jju][1];
        dulist_r[jju][2] = dsfac * ulist_r[jju] * uz + sfac * dulist_r[jju][2];
        dulist_i[jju][2] = dsfac * ulist_i[jju] * uz + sfac * dulist_i[jju][2];
        jju++;
      }
    }
  }
}

// ----------------------------------------------------------------------------
// create_twojmax_arrays / destroy_twojmax_arrays — sna.cpp lines 1335–1384.
// ----------------------------------------------------------------------------
void SnaEngine::create_twojmax_arrays() {
  const int jdimpq = twojmax_ + 2;
  create2d(rootpqarray, static_cast<std::size_t>(jdimpq), static_cast<std::size_t>(jdimpq));
  create1d(cglist, static_cast<std::size_t>(idxcg_max_));
  create1d(ulisttot_r, static_cast<std::size_t>(idxu_max_) * static_cast<std::size_t>(nelements_));
  create1d(ulisttot_i, static_cast<std::size_t>(idxu_max_) * static_cast<std::size_t>(nelements_));
  create2d(dulist_r, static_cast<std::size_t>(idxu_max_), 3);
  create2d(dulist_i, static_cast<std::size_t>(idxu_max_), 3);
  create1d(zlist_r, static_cast<std::size_t>(idxz_max_) * static_cast<std::size_t>(ndoubles_));
  create1d(zlist_i, static_cast<std::size_t>(idxz_max_) * static_cast<std::size_t>(ndoubles_));
  create1d(blist, static_cast<std::size_t>(idxb_max_) * static_cast<std::size_t>(ntriples_));
  create2d(dblist, static_cast<std::size_t>(idxb_max_) * static_cast<std::size_t>(ntriples_), 3);
  create1d(ylist_r, static_cast<std::size_t>(idxu_max_) * static_cast<std::size_t>(nelements_));
  create1d(ylist_i, static_cast<std::size_t>(idxu_max_) * static_cast<std::size_t>(nelements_));

  if (bzero_flag_) {
    create1d(bzero, static_cast<std::size_t>(twojmax_ + 1));
  } else {
    bzero = nullptr;
  }
}

void SnaEngine::destroy_twojmax_arrays() {
  destroy2d(rootpqarray);
  destroy1d(cglist);
  destroy1d(ulisttot_r);
  destroy1d(ulisttot_i);
  destroy2d(dulist_r);
  destroy2d(dulist_i);
  destroy1d(zlist_r);
  destroy1d(zlist_i);
  destroy1d(blist);
  destroy2d(dblist);
  destroy1d(ylist_r);
  destroy1d(ylist_i);

  destroy3d(idxcg_block);
  destroy1d(idxu_block);
  destroy3d(idxz_block);
  destroy3d(idxb_block);

  if (bzero_flag_) {
    destroy1d(bzero);
  }
}

// ----------------------------------------------------------------------------
// deltacg — sna.cpp lines 1390–1396. VMK Eq. 8.2(1).
// ----------------------------------------------------------------------------
double SnaEngine::deltacg(int j1, int j2, int j) const {
  const double sfaccg = factorial((j1 + j2 + j) / 2 + 1);
  return std::sqrt(factorial((j1 + j2 - j) / 2) * factorial((j1 - j2 + j) / 2) *
                   factorial((-j1 + j2 + j) / 2) / sfaccg);
}

// ----------------------------------------------------------------------------
// init_clebsch_gordan — sna.cpp lines 1403–1461. VMK 8.2.1(3).
// ----------------------------------------------------------------------------
void SnaEngine::init_clebsch_gordan() {
  double sum, dcg, sfaccg;
  int m, aa2, bb2, cc2;
  int ifac;

  int idxcg_count = 0;
  for (int j1 = 0; j1 <= twojmax_; ++j1) {
    for (int j2 = 0; j2 <= j1; ++j2) {
      for (int j = j1 - j2; j <= imin(twojmax_, j1 + j2); j += 2) {
        for (int m1 = 0; m1 <= j1; ++m1) {
          aa2 = 2 * m1 - j1;

          for (int m2 = 0; m2 <= j2; ++m2) {
            bb2 = 2 * m2 - j2;
            m = (aa2 + bb2 + j) / 2;

            if (m < 0 || m > j) {
              cglist[idxcg_count] = 0.0;
              idxcg_count++;
              continue;
            }

            sum = 0.0;

            for (int zz = imax(0, imax(-(j - j2 + aa2) / 2, -(j - j1 - bb2) / 2));
                 zz <= imin((j1 + j2 - j) / 2, imin((j1 - aa2) / 2, (j2 + bb2) / 2));
                 ++zz) {
              ifac = (zz % 2) ? -1 : 1;
              sum +=
                  ifac / (factorial(zz) * factorial((j1 + j2 - j) / 2 - zz) *
                          factorial((j1 - aa2) / 2 - zz) * factorial((j2 + bb2) / 2 - zz) *
                          factorial((j - j2 + aa2) / 2 + zz) * factorial((j - j1 - bb2) / 2 + zz));
            }

            cc2 = 2 * m - j;
            dcg = deltacg(j1, j2, j);
            sfaccg = std::sqrt(factorial((j1 + aa2) / 2) * factorial((j1 - aa2) / 2) *
                               factorial((j2 + bb2) / 2) * factorial((j2 - bb2) / 2) *
                               factorial((j + cc2) / 2) * factorial((j - cc2) / 2) * (j + 1));

            cglist[idxcg_count] = sum * dcg * sfaccg;
            idxcg_count++;
          }
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
// init_rootpqarray — sna.cpp lines 1502–1507.
// ----------------------------------------------------------------------------
void SnaEngine::init_rootpqarray() {
  for (int p = 1; p <= twojmax_; ++p) {
    for (int q = 1; q <= twojmax_; ++q) {
      rootpqarray[p][q] = std::sqrt(static_cast<double>(p) / q);
    }
  }
}

// ----------------------------------------------------------------------------
// compute_ncoeff — sna.cpp lines 1511–1529.
// ----------------------------------------------------------------------------
void SnaEngine::compute_ncoeff() {
  int ncount = 0;

  for (int j1 = 0; j1 <= twojmax_; ++j1) {
    for (int j2 = 0; j2 <= j1; ++j2) {
      for (int j = j1 - j2; j <= imin(twojmax_, j1 + j2); j += 2) {
        if (j >= j1) {
          ncount++;
        }
      }
    }
  }

  ndoubles_ = nelements_ * nelements_;
  ntriples_ = nelements_ * nelements_ * nelements_;
  if (chem_flag_) {
    ncoeff = ncount * ntriples_;
  } else {
    ncoeff = ncount;
  }
}

// ----------------------------------------------------------------------------
// compute_sfac / compute_dsfac — sna.cpp lines 1533–1596.
// ----------------------------------------------------------------------------
double SnaEngine::compute_sfac(double r, double rcut, double sinner, double dinner) const {
  double sfac;

  if (switch_flag_ == 0) {
    sfac = 1.0;
  } else if (r <= rmin0_) {
    sfac = 1.0;
  } else if (r > rcut) {
    sfac = 0.0;
  } else {
    const double rcutfac_local = M_PI / (rcut - rmin0_);
    sfac = 0.5 * (std::cos((r - rmin0_) * rcutfac_local) + 1.0);
  }

  if (switch_inner_flag_ == 1 && r < sinner + dinner) {
    if (r > sinner - dinner) {
      const double rcutfac_local = M_PI_2 / dinner;
      sfac *= 0.5 * (1.0 - std::cos(M_PI_2 + (r - sinner) * rcutfac_local));
    } else {
      sfac = 0.0;
    }
  }

  return sfac;
}

double SnaEngine::compute_dsfac(double r, double rcut, double sinner, double dinner) const {
  double dsfac;
  double sfac_outer = 0.0;
  double dsfac_outer;
  double sfac_inner;
  double dsfac_inner;
  if (switch_flag_ == 0) {
    dsfac_outer = 0.0;
  } else if (r <= rmin0_) {
    dsfac_outer = 0.0;
  } else if (r > rcut) {
    dsfac_outer = 0.0;
  } else {
    const double rcutfac_local = M_PI / (rcut - rmin0_);
    dsfac_outer = -0.5 * std::sin((r - rmin0_) * rcutfac_local) * rcutfac_local;
  }

  if (switch_inner_flag_ == 1 && r < sinner + dinner) {
    if (r > sinner - dinner) {
      if (switch_flag_ == 0) {
        sfac_outer = 1.0;
      } else if (r <= rmin0_) {
        sfac_outer = 1.0;
      } else if (r > rcut) {
        sfac_outer = 0.0;
      } else {
        const double rcutfac_local = M_PI / (rcut - rmin0_);
        sfac_outer = 0.5 * (std::cos((r - rmin0_) * rcutfac_local) + 1.0);
      }

      const double rcutfac_local = M_PI_2 / dinner;
      sfac_inner = 0.5 * (1.0 - std::cos(M_PI_2 + (r - sinner) * rcutfac_local));
      dsfac_inner = 0.5 * rcutfac_local * std::sin(M_PI_2 + (r - sinner) * rcutfac_local);
      dsfac = dsfac_outer * sfac_inner + sfac_outer * dsfac_inner;
    } else {
      dsfac = 0.0;
    }
  } else {
    dsfac = dsfac_outer;
  }

  return dsfac;
}

}  // namespace tdmd::snap_detail
