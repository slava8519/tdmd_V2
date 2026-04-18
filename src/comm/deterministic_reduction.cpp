// SPEC: docs/specs/comm/SPEC.md §7.2
// Exec pack: docs/development/m5_execution_pack.md T5.4

#include "tdmd/comm/deterministic_reduction.hpp"

#include <cmath>

namespace tdmd::comm {

double kahan_sum_ordered(const std::vector<double>& values) noexcept {
  // Neumaier's improved Kahan ("Kahan-Babuska-Neumaier") — more robust when
  // the running sum dwarfs the next addend and vice versa. The iteration
  // order is exactly `values[0..N)`, so as long as the caller passes
  // rank-ordered data, the result is deterministic.
  double sum = 0.0;
  double c = 0.0;
  for (const double v : values) {
    const double t = sum + v;
    // Pick the branch that loses less precision for this addend magnitude.
    if (std::abs(sum) >= std::abs(v)) {
      c += (sum - t) + v;
    } else {
      c += (v - t) + sum;
    }
    sum = t;
  }
  return sum + c;
}

#if defined(TDMD_ENABLE_MPI) && TDMD_ENABLE_MPI

double deterministic_sum_double(double local, MPI_Comm comm) {
  int nranks = 0;
  MPI_Comm_size(comm, &nranks);

  std::vector<double> gathered(static_cast<std::size_t>(nranks));
  MPI_Allgather(&local, 1, MPI_DOUBLE, gathered.data(), 1, MPI_DOUBLE, comm);

  return kahan_sum_ordered(gathered);
}

#endif

}  // namespace tdmd::comm
