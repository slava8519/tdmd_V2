// SPEC: docs/specs/gpu/SPEC.md §7.3, §1.1; docs/specs/integrator/SPEC.md §3
// Exec pack: docs/development/m6_execution_pack.md T6.6

#include "tdmd/integrator/gpu_velocity_verlet.hpp"

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/integrator_vv_gpu.hpp"
#include "tdmd/integrator/velocity_verlet.hpp"  // kMetalFtm2v

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace tdmd {

namespace {

void validate_dt(double dt) {
  if (!std::isfinite(dt)) {
    throw std::invalid_argument("GpuVelocityVerletIntegrator: dt must be finite");
  }
  if (dt <= 0.0) {
    throw std::invalid_argument("GpuVelocityVerletIntegrator: dt must be strictly positive");
  }
}

}  // namespace

GpuVelocityVerletIntegrator::GpuVelocityVerletIntegrator(const SpeciesRegistry& species)
    : gpu_(std::make_unique<tdmd::gpu::VelocityVerletGpu>()) {
  const std::size_t n_species = species.count();
  if (n_species == 0) {
    throw std::invalid_argument("GpuVelocityVerletIntegrator: empty SpeciesRegistry");
  }
  accel_by_species_.reserve(n_species);
  for (std::size_t s = 0; s < n_species; ++s) {
    const double m = species.get_info(static_cast<SpeciesId>(s)).mass;
    if (!std::isfinite(m) || m <= 0.0) {
      throw std::invalid_argument(
          "GpuVelocityVerletIntegrator: species mass must be finite and positive");
    }
    accel_by_species_.push_back(kMetalFtm2v / m);
  }
}

GpuVelocityVerletIntegrator::~GpuVelocityVerletIntegrator() = default;
GpuVelocityVerletIntegrator::GpuVelocityVerletIntegrator(GpuVelocityVerletIntegrator&&) noexcept =
    default;
GpuVelocityVerletIntegrator& GpuVelocityVerletIntegrator::operator=(
    GpuVelocityVerletIntegrator&&) noexcept = default;

std::uint64_t GpuVelocityVerletIntegrator::compute_version() const noexcept {
  return gpu_ ? gpu_->compute_version() : 0;
}

void GpuVelocityVerletIntegrator::pre_force_step(AtomSoA& atoms,
                                                 double dt,
                                                 tdmd::gpu::DevicePool& pool,
                                                 tdmd::gpu::DeviceStream& stream) {
  validate_dt(dt);
  const std::size_t n = atoms.size();
  gpu_->pre_force_step(n,
                       dt,
                       accel_by_species_.size(),
                       accel_by_species_.data(),
                       atoms.type.data(),
                       atoms.fx.data(),
                       atoms.fy.data(),
                       atoms.fz.data(),
                       atoms.x.data(),
                       atoms.y.data(),
                       atoms.z.data(),
                       atoms.vx.data(),
                       atoms.vy.data(),
                       atoms.vz.data(),
                       pool,
                       stream);
}

void GpuVelocityVerletIntegrator::post_force_step(AtomSoA& atoms,
                                                  double dt,
                                                  tdmd::gpu::DevicePool& pool,
                                                  tdmd::gpu::DeviceStream& stream) {
  validate_dt(dt);
  const std::size_t n = atoms.size();
  gpu_->post_force_step(n,
                        dt,
                        accel_by_species_.size(),
                        accel_by_species_.data(),
                        atoms.type.data(),
                        atoms.fx.data(),
                        atoms.fy.data(),
                        atoms.fz.data(),
                        atoms.vx.data(),
                        atoms.vy.data(),
                        atoms.vz.data(),
                        pool,
                        stream);
}

}  // namespace tdmd
