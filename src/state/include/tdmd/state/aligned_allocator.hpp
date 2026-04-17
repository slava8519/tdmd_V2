#pragma once

// SPEC: docs/specs/state/SPEC.md §2 (AtomSoA layout)
// Exec pack: docs/development/m1_execution_pack.md — T1.1 mandatory invariant
// "SoA fields aligned на 64 bytes".
//
// Minimal C++17 allocator that returns buffers aligned to `Alignment` bytes
// using `::operator new(std::size_t, std::align_val_t)`. Intended for
// `std::vector<T, AlignedAllocator<T, 64>>` so that `&vec[0]` is always
// 64-byte aligned — a prerequisite for vectorized kernels (M6) and GPU
// transfer paths.
//
// The allocator is stateless: all instances compare equal. This lets the
// standard library elide moves across same-typed containers.

#include <cstddef>
#include <limits>
#include <new>
#include <type_traits>

namespace tdmd {

template <typename T, std::size_t Alignment = 64>
class AlignedAllocator {
  static_assert(Alignment > 0 && (Alignment & (Alignment - 1)) == 0,
                "Alignment must be a positive power of two");
  static_assert(Alignment >= alignof(T), "Alignment must be at least alignof(T)");

public:
  using value_type = T;
  static constexpr std::size_t alignment = Alignment;

  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;
  using is_always_equal = std::true_type;

  constexpr AlignedAllocator() noexcept = default;

  template <typename U>
  constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  [[nodiscard]] T* allocate(std::size_t n) {
    if (n == 0) {
      return nullptr;
    }
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }
    void* const p = ::operator new(n * sizeof(T), std::align_val_t{Alignment});
    return static_cast<T*>(p);
  }

  void deallocate(T* p, std::size_t /*n*/) noexcept {
    ::operator delete(p, std::align_val_t{Alignment});
  }
};

template <typename T, typename U, std::size_t A>
constexpr bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept {
  return true;
}

template <typename T, typename U, std::size_t A>
constexpr bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept {
  return false;
}

}  // namespace tdmd
