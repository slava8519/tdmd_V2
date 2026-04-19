// Exec pack: docs/development/m6_execution_pack.md T6.2
// SPEC: docs/specs/gpu/SPEC.md §2 (core types), §11 (GpuConfig defaults)
//
// T6.2 skeleton tests — compile-time shape invariants + runtime defaults.
// Concrete kernel and pool behaviour lands at T6.3+ with CUDA-bound Impl.
// These tests are pure C++ and run on every build (CI included), regardless
// of whether CUDA toolkit is present.

#include "tdmd/gpu/device_allocator.hpp"
#include "tdmd/gpu/gpu_config.hpp"
#include "tdmd/gpu/types.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace tg = tdmd::gpu;

// -----------------------------------------------------------------------------
// GpuConfig — defaults locked to D-M6-13 / D-M6-14 / D-M6-18.
// -----------------------------------------------------------------------------
TEST_CASE("GpuConfig — defaults match D-M6-18", "[gpu][types]") {
  tg::GpuConfig cfg;
  REQUIRE(cfg.device_id == 0);
  REQUIRE(cfg.streams == 2u);                      // D-M6-13 (compute + mem)
  REQUIRE(cfg.memory_pool_init_size_mib == 256u);  // D-M6-12 initial pool
  REQUIRE(cfg.enable_nvtx == true);                // D-M6-14
}

TEST_CASE("Stream ordinal constants — pinned", "[gpu][types]") {
  STATIC_REQUIRE(tg::kStreamCompute == 0u);
  STATIC_REQUIRE(tg::kStreamMem == 1u);
}

// -----------------------------------------------------------------------------
// DevicePtr<T> — move-only, custom deleter invoked exactly once.
// -----------------------------------------------------------------------------
TEST_CASE("DevicePtr — move-only; not copy-constructible", "[gpu][types]") {
  STATIC_REQUIRE(!std::is_copy_constructible_v<tg::DevicePtr<int>>);
  STATIC_REQUIRE(!std::is_copy_assignable_v<tg::DevicePtr<int>>);
  STATIC_REQUIRE(std::is_nothrow_move_constructible_v<tg::DevicePtr<int>>);
  STATIC_REQUIRE(std::is_nothrow_move_assignable_v<tg::DevicePtr<int>>);
}

TEST_CASE("DevicePtr — default is null; get/bool agree", "[gpu][types]") {
  tg::DevicePtr<int> ptr;
  REQUIRE(ptr.get() == nullptr);
  REQUIRE_FALSE(static_cast<bool>(ptr));
}

TEST_CASE("DevicePtr — deleter invoked exactly once on destruction", "[gpu][types]") {
  int delete_count = 0;
  int storage = 42;
  {
    tg::DevicePtr<int> p(
        &storage,
        [](void* ptr, void* ctx) noexcept {
          (void) ptr;
          ++(*static_cast<int*>(ctx));
        },
        &delete_count);
    REQUIRE(p.get() == &storage);
    REQUIRE(static_cast<bool>(p));
  }
  REQUIRE(delete_count == 1);
}

TEST_CASE("DevicePtr — move transfers ownership; source becomes null", "[gpu][types]") {
  int delete_count = 0;
  int storage = 42;
  auto deleter = [](void* /*ptr*/, void* ctx) noexcept { ++(*static_cast<int*>(ctx)); };
  {
    tg::DevicePtr<int> a(&storage, deleter, &delete_count);
    tg::DevicePtr<int> b(std::move(a));
    REQUIRE(a.get() == nullptr);  // NOLINT(bugprone-use-after-move) — asserting moved-from state
    REQUIRE(b.get() == &storage);
  }
  REQUIRE(delete_count == 1);
}

TEST_CASE("DevicePtr — move-assignment releases existing resource", "[gpu][types]") {
  int delete_count = 0;
  int a_storage = 1;
  int b_storage = 2;
  auto deleter = [](void* /*ptr*/, void* ctx) noexcept { ++(*static_cast<int*>(ctx)); };
  {
    tg::DevicePtr<int> a(&a_storage, deleter, &delete_count);
    tg::DevicePtr<int> b(&b_storage, deleter, &delete_count);
    a = std::move(b);
    REQUIRE(delete_count == 1);  // a's original released
    REQUIRE(a.get() == &b_storage);
    REQUIRE(b.get() == nullptr);  // NOLINT(bugprone-use-after-move)
  }
  REQUIRE(delete_count == 2);  // remaining released on scope exit
}

TEST_CASE("DevicePtr — release() transfers raw pointer; deleter not invoked", "[gpu][types]") {
  int delete_count = 0;
  int storage = 42;
  auto deleter = [](void* /*ptr*/, void* ctx) noexcept { ++(*static_cast<int*>(ctx)); };
  int* raw = nullptr;
  {
    tg::DevicePtr<int> p(&storage, deleter, &delete_count);
    raw = p.release();
    REQUIRE(p.get() == nullptr);
  }
  REQUIRE(raw == &storage);
  REQUIRE(delete_count == 0);  // release() opts out of deletion
}

TEST_CASE("DevicePtr — reset() on null handle is a no-op", "[gpu][types]") {
  tg::DevicePtr<int> p;
  p.reset();  // should not crash / deref null deleter
  REQUIRE(p.get() == nullptr);
}

// -----------------------------------------------------------------------------
// DeviceStream / DeviceEvent — move-only RAII, default-constructed is null.
// -----------------------------------------------------------------------------
TEST_CASE("DeviceStream — move-only; not copy-constructible", "[gpu][types]") {
  STATIC_REQUIRE(!std::is_copy_constructible_v<tg::DeviceStream>);
  STATIC_REQUIRE(!std::is_copy_assignable_v<tg::DeviceStream>);
  STATIC_REQUIRE(std::is_nothrow_move_constructible_v<tg::DeviceStream>);
  STATIC_REQUIRE(std::is_nothrow_move_assignable_v<tg::DeviceStream>);
}

TEST_CASE("DeviceStream — default-constructed is valid() == false", "[gpu][types]") {
  tg::DeviceStream s;
  REQUIRE_FALSE(s.valid());
  REQUIRE(s.impl() == nullptr);
}

TEST_CASE("DeviceStream — move of null stream is null; no UB", "[gpu][types]") {
  tg::DeviceStream a;
  tg::DeviceStream b(std::move(a));
  REQUIRE_FALSE(b.valid());
  REQUIRE_FALSE(a.valid());  // NOLINT(bugprone-use-after-move)
}

TEST_CASE("DeviceEvent — move-only; not copy-constructible", "[gpu][types]") {
  STATIC_REQUIRE(!std::is_copy_constructible_v<tg::DeviceEvent>);
  STATIC_REQUIRE(!std::is_copy_assignable_v<tg::DeviceEvent>);
  STATIC_REQUIRE(std::is_nothrow_move_constructible_v<tg::DeviceEvent>);
  STATIC_REQUIRE(std::is_nothrow_move_assignable_v<tg::DeviceEvent>);
}

TEST_CASE("DeviceEvent — default-constructed is valid() == false", "[gpu][types]") {
  tg::DeviceEvent e;
  REQUIRE_FALSE(e.valid());
  REQUIRE(e.impl() == nullptr);
}

// -----------------------------------------------------------------------------
// DeviceInfo — struct layout + defaults.
// -----------------------------------------------------------------------------
TEST_CASE("DeviceInfo — default-constructible with warp_size == 32", "[gpu][types]") {
  tg::DeviceInfo info;
  REQUIRE(info.device_id == 0);
  REQUIRE(info.name.empty());
  REQUIRE(info.compute_capability_major == 0u);
  REQUIRE(info.compute_capability_minor == 0u);
  REQUIRE(info.total_global_memory_bytes == 0u);
  REQUIRE(info.multiprocessor_count == 0u);
  REQUIRE(info.warp_size == 32u);
  REQUIRE(info.max_threads_per_block == 0u);
  REQUIRE(info.capabilities.empty());
}

// -----------------------------------------------------------------------------
// DeviceAllocator — abstract with virtual destructor.
// -----------------------------------------------------------------------------
TEST_CASE("DeviceAllocator — abstract class with virtual destructor", "[gpu][types]") {
  STATIC_REQUIRE(std::is_abstract_v<tg::DeviceAllocator>);
  STATIC_REQUIRE(std::has_virtual_destructor_v<tg::DeviceAllocator>);
}
