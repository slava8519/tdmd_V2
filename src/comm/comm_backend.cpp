// Translation unit for CommBackend — present so the abstract class has a
// stable vtable + typeinfo emission unit. Concrete backends (T5.4/T5.5)
// live alongside this file.
//
// The virtual destructor is inlined in the header; this TU exists to give
// the linker a home for any future out-of-line infrastructure (e.g.
// a shared error-formatting helper for all backends).

#include "tdmd/comm/comm_backend.hpp"

namespace tdmd::comm {

// Explicit instantiation anchor — ensures CommBackend's vtable is emitted
// here rather than spread across every user of the header.
void* comm_backend_translation_unit_anchor() {
  return nullptr;
}

}  // namespace tdmd::comm
