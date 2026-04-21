// SPEC: docs/specs/gpu/SPEC.md §7.5 (SNAP GPU — T8.6b), §1.1 (data-oblivious gpu/)
// Module SPEC: docs/specs/potentials/SPEC.md §6 (SNAP)
// Exec pack: docs/development/m8_execution_pack.md T8.6b
//
// Host-side SNAP index-table builder. See snap_gpu_tables.cuh for the rationale
// and data layout. This TU duplicates the ~150 lines of integer-recurrence /
// Clebsch-Gordan math from src/potentials/snap/sna_engine.cpp so that the gpu/
// module can produce flat device-ready tables without having to #include
// src/potentials/snap/sna_engine.hpp (keeps gpu/ data-oblivious per
// module SPEC §1.1).
//
// The algorithm is pure integer + factorial arithmetic; byte-exactness with
// the CPU oracle at the level of `cglist[]` follows by construction (same
// arithmetic, no FP accumulation subtlety).
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
// -----------------------------------------------------------------------------

#include "snap_gpu_tables.cuh"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tdmd::gpu::snap_detail {

namespace {

inline int imin(int a, int b) {
  return a < b ? a : b;
}
inline int imax(int a, int b) {
  return a > b ? a : b;
}

inline double factorial(int n) {
  double f = 1.0;
  for (int k = 2; k <= n; ++k) {
    f *= static_cast<double>(k);
  }
  return f;
}

// Flat-array helpers: address idxcg_block[j1][j2][j] in a jdim³ vector.
inline std::size_t jkk_index(int j1, int j2, int j, int jdim) {
  return static_cast<std::size_t>(j1) * static_cast<std::size_t>(jdim) *
             static_cast<std::size_t>(jdim) +
         static_cast<std::size_t>(j2) * static_cast<std::size_t>(jdim) +
         static_cast<std::size_t>(j);
}

// Port of SnaEngine::deltacg.
double deltacg(int j1, int j2, int j) {
  const double sfaccg = factorial((j1 + j2 + j) / 2 + 1);
  return std::sqrt(factorial((j1 + j2 - j) / 2) * factorial((j1 - j2 + j) / 2) *
                   factorial((-j1 + j2 + j) / 2) / sfaccg);
}

// Port of SnaEngine::build_indexlist, lines 233-364.
void build_indexlist_into(FlatIndexTables& t) {
  const int jdim = t.jdim;
  const std::size_t jdim_sz = static_cast<std::size_t>(jdim);
  const std::size_t jdim_cube = jdim_sz * jdim_sz * jdim_sz;

  // idxcg_block + count idxcg_max.
  t.idxcg_block.assign(jdim_cube, 0);
  int idxcg_count = 0;
  for (int j1 = 0; j1 <= t.twojmax; ++j1) {
    for (int j2 = 0; j2 <= j1; ++j2) {
      for (int j = j1 - j2; j <= imin(t.twojmax, j1 + j2); j += 2) {
        t.idxcg_block[jkk_index(j1, j2, j, jdim)] = idxcg_count;
        for (int m1 = 0; m1 <= j1; ++m1) {
          for (int m2 = 0; m2 <= j2; ++m2) {
            idxcg_count++;
          }
        }
      }
    }
  }
  t.idxcg_max = idxcg_count;

  // idxu_block.
  t.idxu_block.assign(static_cast<std::size_t>(jdim), 0);
  int idxu_count = 0;
  for (int j = 0; j <= t.twojmax; ++j) {
    t.idxu_block[static_cast<std::size_t>(j)] = idxu_count;
    for (int mb = 0; mb <= j; ++mb) {
      for (int ma = 0; ma <= j; ++ma) {
        idxu_count++;
      }
    }
  }
  t.idxu_max = idxu_count;

  // idxb_max — count first pass.
  int idxb_count = 0;
  for (int j1 = 0; j1 <= t.twojmax; ++j1) {
    for (int j2 = 0; j2 <= j1; ++j2) {
      for (int j = j1 - j2; j <= imin(t.twojmax, j1 + j2); j += 2) {
        if (j >= j1) {
          idxb_count++;
        }
      }
    }
  }
  t.idxb_max = idxb_count;
  t.idxb_packed.assign(static_cast<std::size_t>(idxb_count) * kSnaBIndicesStride, 0);

  idxb_count = 0;
  for (int j1 = 0; j1 <= t.twojmax; ++j1) {
    for (int j2 = 0; j2 <= j1; ++j2) {
      for (int j = j1 - j2; j <= imin(t.twojmax, j1 + j2); j += 2) {
        if (j >= j1) {
          const std::size_t base = static_cast<std::size_t>(idxb_count) * kSnaBIndicesStride;
          t.idxb_packed[base + 0] = j1;
          t.idxb_packed[base + 1] = j2;
          t.idxb_packed[base + 2] = j;
          idxb_count++;
        }
      }
    }
  }

  // idxb_block (reverse index).
  t.idxb_block.assign(jdim_cube, 0);
  idxb_count = 0;
  for (int j1 = 0; j1 <= t.twojmax; ++j1) {
    for (int j2 = 0; j2 <= j1; ++j2) {
      for (int j = j1 - j2; j <= imin(t.twojmax, j1 + j2); j += 2) {
        if (j >= j1) {
          t.idxb_block[jkk_index(j1, j2, j, jdim)] = idxb_count;
          idxb_count++;
        }
      }
    }
  }

  // idxz_max — count first pass.
  int idxz_count = 0;
  for (int j1 = 0; j1 <= t.twojmax; ++j1) {
    for (int j2 = 0; j2 <= j1; ++j2) {
      for (int j = j1 - j2; j <= imin(t.twojmax, j1 + j2); j += 2) {
        for (int mb = 0; 2 * mb <= j; ++mb) {
          for (int ma = 0; ma <= j; ++ma) {
            idxz_count++;
          }
        }
      }
    }
  }
  t.idxz_max = idxz_count;
  t.idxz_packed.assign(static_cast<std::size_t>(idxz_count) * kSnaZIndicesStride, 0);

  t.idxz_block.assign(jdim_cube, 0);

  idxz_count = 0;
  for (int j1 = 0; j1 <= t.twojmax; ++j1) {
    for (int j2 = 0; j2 <= j1; ++j2) {
      for (int j = j1 - j2; j <= imin(t.twojmax, j1 + j2); j += 2) {
        t.idxz_block[jkk_index(j1, j2, j, jdim)] = idxz_count;

        for (int mb = 0; 2 * mb <= j; ++mb) {
          for (int ma = 0; ma <= j; ++ma) {
            const std::size_t base = static_cast<std::size_t>(idxz_count) * kSnaZIndicesStride;
            t.idxz_packed[base + 0] = j1;
            t.idxz_packed[base + 1] = j2;
            t.idxz_packed[base + 2] = j;

            const int ma1min = imax(0, (2 * ma - j - j2 + j1) / 2);
            const int ma2max = (2 * ma - j - (2 * ma1min - j1) + j2) / 2;
            const int na = imin(j1, (2 * ma - j + j2 + j1) / 2) - ma1min + 1;
            const int mb1min = imax(0, (2 * mb - j - j2 + j1) / 2);
            const int mb2max = (2 * mb - j - (2 * mb1min - j1) + j2) / 2;
            const int nb = imin(j1, (2 * mb - j + j2 + j1) / 2) - mb1min + 1;
            const int jju = t.idxu_block[static_cast<std::size_t>(j)] + (j + 1) * mb + ma;

            t.idxz_packed[base + 3] = ma1min;
            t.idxz_packed[base + 4] = ma2max;
            t.idxz_packed[base + 5] = mb1min;
            t.idxz_packed[base + 6] = mb2max;
            t.idxz_packed[base + 7] = na;
            t.idxz_packed[base + 8] = nb;
            t.idxz_packed[base + 9] = jju;

            idxz_count++;
          }
        }
      }
    }
  }
}

// Port of SnaEngine::init_clebsch_gordan, lines 1101-1148.
void init_clebsch_gordan_into(FlatIndexTables& t) {
  double sum, dcg, sfaccg;
  int m, aa2, bb2, cc2;
  int ifac;

  t.cglist.assign(static_cast<std::size_t>(t.idxcg_max), 0.0);

  int idxcg_count = 0;
  for (int j1 = 0; j1 <= t.twojmax; ++j1) {
    for (int j2 = 0; j2 <= j1; ++j2) {
      for (int j = j1 - j2; j <= imin(t.twojmax, j1 + j2); j += 2) {
        for (int m1 = 0; m1 <= j1; ++m1) {
          aa2 = 2 * m1 - j1;

          for (int m2 = 0; m2 <= j2; ++m2) {
            bb2 = 2 * m2 - j2;
            m = (aa2 + bb2 + j) / 2;

            if (m < 0 || m > j) {
              t.cglist[static_cast<std::size_t>(idxcg_count)] = 0.0;
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

            t.cglist[static_cast<std::size_t>(idxcg_count)] = sum * dcg * sfaccg;
            idxcg_count++;
          }
        }
      }
    }
  }
}

// Port of SnaEngine::init_rootpqarray, lines 1153-1159.
void init_rootpqarray_into(FlatIndexTables& t) {
  const std::size_t jdimpq = static_cast<std::size_t>(t.jdimpq);
  t.rootpq.assign(jdimpq * jdimpq, 0.0);
  for (int p = 1; p <= t.twojmax; ++p) {
    for (int q = 1; q <= t.twojmax; ++q) {
      t.rootpq[static_cast<std::size_t>(p) * jdimpq + static_cast<std::size_t>(q)] =
          std::sqrt(static_cast<double>(p) / q);
    }
  }
}

// T-opt-2: build the per-jju CSR bucket of jjz indices (see header comment on
// FlatIndexTables). The jju of each jjz is idxz_packed[jjz*10 + 9].
// Bucketing is stable-by-jjz: within a bucket, jjz values are strictly
// ascending — matches the legacy tid==0 Phase B loop's += order at that jju.
// Implementation: counting-sort style two-pass over idxz_max.
void build_idxz_jju_buckets_into(FlatIndexTables& t) {
  t.idxz_jju_bucket_begin.assign(t.idxu_max + 1, 0);
  t.idxz_by_jju.assign(t.idxz_max, 0);
  if (t.idxz_max == 0 || t.idxu_max == 0) {
    return;
  }
  for (int jjz = 0; jjz < t.idxz_max; ++jjz) {
    const int jju = t.idxz_packed[jjz * 10 + 9];
    ++t.idxz_jju_bucket_begin[jju + 1];
  }
  for (int jju = 0; jju < t.idxu_max; ++jju) {
    t.idxz_jju_bucket_begin[jju + 1] += t.idxz_jju_bucket_begin[jju];
  }
  std::vector<int> cursor(t.idxu_max, 0);
  for (int jjz = 0; jjz < t.idxz_max; ++jjz) {
    const int jju = t.idxz_packed[jjz * 10 + 9];
    const int slot = t.idxz_jju_bucket_begin[jju] + cursor[jju]++;
    t.idxz_by_jju[slot] = jjz;
  }
}

}  // namespace

FlatIndexTables build_flat_tables(int twojmax) {
  if (twojmax < 0 || (twojmax % 2) != 0) {
    throw std::invalid_argument(
        "tdmd::gpu::snap_detail::build_flat_tables: twojmax must be non-negative and even");
  }

  FlatIndexTables t;
  t.twojmax = twojmax;
  t.jdim = twojmax + 1;
  t.jdimpq = twojmax + 2;

  build_indexlist_into(t);
  init_clebsch_gordan_into(t);
  init_rootpqarray_into(t);
  build_idxz_jju_buckets_into(t);

  return t;
}

}  // namespace tdmd::gpu::snap_detail
