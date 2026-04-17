#include "tdmd/integrator/velocity_verlet.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace tdmd {

namespace {

void validate_dt(double dt) {
  if (!std::isfinite(dt)) {
    throw std::invalid_argument("VelocityVerletIntegrator: dt must be finite");
  }
  if (dt <= 0.0) {
    throw std::invalid_argument("VelocityVerletIntegrator: dt must be strictly positive");
  }
}

}  // namespace

void VelocityVerletIntegrator::pre_force_step(AtomSoA& atoms,
                                              const SpeciesRegistry& species,
                                              double dt) {
  validate_dt(dt);
  const double half_dt = 0.5 * dt;
  const std::size_t n = atoms.size();
  for (std::size_t i = 0; i < n; ++i) {
    const double m = species.get_info(atoms.type[i]).mass;
    const double accel_factor = kMetalFtm2v / m;
    atoms.vx[i] += atoms.fx[i] * accel_factor * half_dt;
    atoms.vy[i] += atoms.fy[i] * accel_factor * half_dt;
    atoms.vz[i] += atoms.fz[i] * accel_factor * half_dt;
    atoms.x[i] += atoms.vx[i] * dt;
    atoms.y[i] += atoms.vy[i] * dt;
    atoms.z[i] += atoms.vz[i] * dt;
  }
}

void VelocityVerletIntegrator::post_force_step(AtomSoA& atoms,
                                               const SpeciesRegistry& species,
                                               double dt) {
  validate_dt(dt);
  const double half_dt = 0.5 * dt;
  const std::size_t n = atoms.size();
  for (std::size_t i = 0; i < n; ++i) {
    const double m = species.get_info(atoms.type[i]).mass;
    const double accel_factor = kMetalFtm2v / m;
    atoms.vx[i] += atoms.fx[i] * accel_factor * half_dt;
    atoms.vy[i] += atoms.fy[i] * accel_factor * half_dt;
    atoms.vz[i] += atoms.fz[i] * accel_factor * half_dt;
  }
}

double kinetic_energy(const AtomSoA& atoms, const SpeciesRegistry& species) {
  double raw_sum = 0.0;
  const std::size_t n = atoms.size();
  for (std::size_t i = 0; i < n; ++i) {
    const double m = species.get_info(atoms.type[i]).mass;
    const double v2 =
        atoms.vx[i] * atoms.vx[i] + atoms.vy[i] * atoms.vy[i] + atoms.vz[i] * atoms.vz[i];
    raw_sum += m * v2;
  }
  return 0.5 * raw_sum * kMetalMvv2e;
}

}  // namespace tdmd
