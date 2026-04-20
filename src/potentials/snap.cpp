// SPEC: docs/specs/potentials/SPEC.md §6 (SNAP). Exec pack:
// docs/development/m8_execution_pack.md T8.4a/T8.4b.
//
// SnapPotential skeleton (T8.4a). Constructor validates the SnapData layout
// (species count > 0; twojmax even and non-negative; every species carries
// k_max + 1 linear β coefficients, or `k_max + 1 + k_max·(k_max+1)/2` when
// `quadraticflag` is set — cross-checked by parse_snap_files, re-checked here
// so that hand-assembled SnapData literals (unit tests) also get screened).
// `compute()` throws `std::logic_error`: the force-evaluation body is a
// verbatim port of LAMMPS USER-SNAP `sna.cpp` + `pair_snap.cpp` and lands в
// T8.4b as a follow-up PR (size: ~1600 lines of dense Clebsch-Gordan code —
// not session-sized).

#include "tdmd/potentials/snap.hpp"

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
}

ForceResult SnapPotential::compute(AtomSoA& /*atoms*/,
                                   const NeighborList& /*neighbors*/,
                                   const Box& /*box*/) {
  // T8.4b will port the three-pass bispectrum → energy → force evaluator
  // from LAMMPS USER-SNAP (`sna.cpp` + `pair_snap.cpp`). Keeping the
  // skeleton loud + fail-fast so that any runtime path that accidentally
  // wires this potential in before T8.4b surfaces the gap immediately
  // instead of silently producing zero forces.
  throw std::logic_error(
      "SnapPotential::compute: force evaluation not yet implemented "
      "(T8.4b — LAMMPS USER-SNAP port)");
}

}  // namespace tdmd
