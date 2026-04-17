#include "tdmd/integrator/integrator.hpp"

// TODO(M1): concrete integrators per integrator/SPEC.md §5 —
// VelocityVerletIntegrator, NoseHooverNvtIntegrator, NoseHooverNptIntegrator,
// LangevinIntegrator.

namespace tdmd {

extern const char* const kIntegratorModuleTag;
const char* const kIntegratorModuleTag = "tdmd::integrator (M0 skeleton)";

}  // namespace tdmd
