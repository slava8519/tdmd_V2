#pragma once

// SPEC: docs/specs/integrator/SPEC.md
//
// Integrator is the abstract interface for time integration schemes.
// Concrete implementations (VelocityVerlet, NoseHoover NVT/NPT, Langevin)
// live in separate TUs and arrive in M1+.

namespace tdmd {

class Integrator {
 public:
  Integrator() = default;
  virtual ~Integrator() = default;

  Integrator(const Integrator&) = delete;
  Integrator& operator=(const Integrator&) = delete;
  Integrator(Integrator&&) = delete;
  Integrator& operator=(Integrator&&) = delete;

  // TODO(M1): interface methods per integrator/SPEC.md §2.1 —
  // prepare(), advance(dt), finalize(), name(), kind() (enum), etc.
};

}  // namespace tdmd
