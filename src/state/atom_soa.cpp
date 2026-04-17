#include "tdmd/state/atom_soa.hpp"

#include <cassert>

namespace tdmd {

void AtomSoA::reserve_all_fields(std::size_t new_capacity) {
  id.reserve(new_capacity);
  type.reserve(new_capacity);
  x.reserve(new_capacity);
  y.reserve(new_capacity);
  z.reserve(new_capacity);
  vx.reserve(new_capacity);
  vy.reserve(new_capacity);
  vz.reserve(new_capacity);
  fx.reserve(new_capacity);
  fy.reserve(new_capacity);
  fz.reserve(new_capacity);
  image_x.reserve(new_capacity);
  image_y.reserve(new_capacity);
  image_z.reserve(new_capacity);
  flags.reserve(new_capacity);
}

void AtomSoA::resize_all_fields(std::size_t new_size) {
  id.resize(new_size);
  type.resize(new_size);
  x.resize(new_size);
  y.resize(new_size);
  z.resize(new_size);
  vx.resize(new_size);
  vy.resize(new_size);
  vz.resize(new_size);
  fx.resize(new_size);
  fy.resize(new_size);
  fz.resize(new_size);
  image_x.resize(new_size);
  image_y.resize(new_size);
  image_z.resize(new_size);
  flags.resize(new_size);
}

void AtomSoA::reserve(std::size_t new_capacity) {
  reserve_all_fields(new_capacity);
}

void AtomSoA::resize(std::size_t new_size) {
  resize_all_fields(new_size);
}

void AtomSoA::clear() noexcept {
  id.clear();
  type.clear();
  x.clear();
  y.clear();
  z.clear();
  vx.clear();
  vy.clear();
  vz.clear();
  fx.clear();
  fy.clear();
  fz.clear();
  image_x.clear();
  image_y.clear();
  image_z.clear();
  flags.clear();
}

AtomId AtomSoA::add_atom(const AtomInit& init) {
  const AtomId assigned = next_id_++;
  id.push_back(assigned);
  type.push_back(init.type);
  x.push_back(init.x);
  y.push_back(init.y);
  z.push_back(init.z);
  vx.push_back(init.vx);
  vy.push_back(init.vy);
  vz.push_back(init.vz);
  fx.push_back(0.0);
  fy.push_back(0.0);
  fz.push_back(0.0);
  image_x.push_back(0);
  image_y.push_back(0);
  image_z.push_back(0);
  flags.push_back(0);
  return assigned;
}

AtomId AtomSoA::add_atom(SpeciesId atom_type,
                         double ax,
                         double ay,
                         double az,
                         double avx,
                         double avy,
                         double avz) {
  return add_atom(AtomInit{atom_type, ax, ay, az, avx, avy, avz});
}

void AtomSoA::remove_atom(std::size_t atom_idx) {
  assert(atom_idx < size() && "AtomSoA::remove_atom — index out of range");
  const std::size_t last = size() - 1;
  if (atom_idx != last) {
    id[atom_idx] = id[last];
    type[atom_idx] = type[last];
    x[atom_idx] = x[last];
    y[atom_idx] = y[last];
    z[atom_idx] = z[last];
    vx[atom_idx] = vx[last];
    vy[atom_idx] = vy[last];
    vz[atom_idx] = vz[last];
    fx[atom_idx] = fx[last];
    fy[atom_idx] = fy[last];
    fz[atom_idx] = fz[last];
    image_x[atom_idx] = image_x[last];
    image_y[atom_idx] = image_y[last];
    image_z[atom_idx] = image_z[last];
    flags[atom_idx] = flags[last];
  }
  id.pop_back();
  type.pop_back();
  x.pop_back();
  y.pop_back();
  z.pop_back();
  vx.pop_back();
  vy.pop_back();
  vz.pop_back();
  fx.pop_back();
  fy.pop_back();
  fz.pop_back();
  image_x.pop_back();
  image_y.pop_back();
  image_z.pop_back();
  flags.pop_back();
}

bool AtomSoA::invariants_hold() const noexcept {
  const std::size_t n = id.size();
  return type.size() == n && x.size() == n && y.size() == n && z.size() == n && vx.size() == n &&
         vy.size() == n && vz.size() == n && fx.size() == n && fy.size() == n && fz.size() == n &&
         image_x.size() == n && image_y.size() == n && image_z.size() == n && flags.size() == n;
}

}  // namespace tdmd
