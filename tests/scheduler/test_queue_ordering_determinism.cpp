// Exec pack: docs/development/m4_execution_pack.md T4.10
// SPEC: docs/specs/scheduler/SPEC.md §5.2 (Reference tie-break), §12.3 (determinism)
// Master spec: §6.7, §13.4
//
// T4.10 (b) — queue ordering determinism. Generate 10⁴ TaskCandidates with
// unique (time_level, canonical_index, version) triples, shuffle them 100
// different ways, sort each shuffle with ReferenceTaskCompare. All 100
// sorted sequences must be byte-identical.
//
// This is the scheduling-side analogue of a stable-sort fuzzer: it locks
// the Reference tie-break so that refactors touching select_ready_tasks()
// can't silently introduce arrival-order dependence. A dependence here
// would show up in M5 as a cross-run layout divergence — cheaper to catch
// here with 10⁴×100 = 10⁶ trials than in an anchor-test reproduction.

#include "tdmd/scheduler/queues.hpp"
#include "tdmd/scheduler/types.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <random>
#include <unordered_set>
#include <vector>

namespace {

namespace ts = tdmd::scheduler;

// Build `n` TaskCandidates with unique (time_level, canonical_index, version)
// tuples so ReferenceTaskCompare induces a total order (no ties). Using a
// hash set deduplicates against accidental collisions from the RNG.
std::vector<ts::TaskCandidate> make_unique_candidates(std::size_t n, std::uint64_t seed) {
  std::vector<ts::TaskCandidate> out;
  out.reserve(n);

  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<std::uint32_t> tl_dist(0, 63);
  std::uniform_int_distribution<std::size_t> idx_dist(0, 1024);
  std::uniform_int_distribution<std::uint64_t> ver_dist(0, 65535);

  struct TupleHash {
    std::size_t operator()(const std::tuple<ts::TimeLevel, std::size_t, ts::Version>& t) const {
      auto h1 = std::hash<ts::TimeLevel>{}(std::get<0>(t));
      auto h2 = std::hash<std::size_t>{}(std::get<1>(t));
      auto h3 = std::hash<ts::Version>{}(std::get<2>(t));
      return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
  };
  std::unordered_set<std::tuple<ts::TimeLevel, std::size_t, ts::Version>, TupleHash> seen;

  while (out.size() < n) {
    ts::TaskCandidate c;
    c.zone_id = static_cast<ts::ZoneId>(out.size());
    c.time_level = tl_dist(rng);
    c.canonical_index = idx_dist(rng);
    c.version = ver_dist(rng);
    c.cert_id = static_cast<std::uint64_t>(out.size());
    const auto key = std::make_tuple(c.time_level, c.canonical_index, c.version);
    if (seen.insert(key).second) {
      out.push_back(c);
    }
  }
  return out;
}

bool candidates_byte_equal(const std::vector<ts::TaskCandidate>& a,
                           const std::vector<ts::TaskCandidate>& b) {
  if (a.size() != b.size())
    return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].time_level != b[i].time_level)
      return false;
    if (a[i].canonical_index != b[i].canonical_index)
      return false;
    if (a[i].version != b[i].version)
      return false;
    if (a[i].zone_id != b[i].zone_id)
      return false;
    // cert_id is metadata only; ignored by ReferenceTaskCompare.
  }
  return true;
}

}  // namespace

TEST_CASE("T4.10 (b) — 10^4 candidates × 100 shuffles: sorted order byte-identical",
          "[scheduler][determinism][queue-ordering]") {
  constexpr std::size_t kCandidates = 10000;
  constexpr std::size_t kShuffles = 100;

  const auto base = make_unique_candidates(kCandidates, /*seed=*/0xC0FFEEULL);
  REQUIRE(base.size() == kCandidates);

  // Reference ordering: sort once without shuffling.
  std::vector<ts::TaskCandidate> reference = base;
  std::sort(reference.begin(), reference.end(), ts::ReferenceTaskCompare{});

  // Each shuffle uses its own RNG seed, so the shuffle permutations are
  // themselves deterministic across runs (Level 1 determinism all the way
  // down — the test itself must be reproducible).
  for (std::size_t s = 0; s < kShuffles; ++s) {
    INFO("shuffle index " << s);
    std::vector<ts::TaskCandidate> shuffled = base;
    std::mt19937_64 rng(0xBEEFULL + s);
    std::shuffle(shuffled.begin(), shuffled.end(), rng);

    std::sort(shuffled.begin(), shuffled.end(), ts::ReferenceTaskCompare{});

    REQUIRE(candidates_byte_equal(shuffled, reference));
  }
}

TEST_CASE("T4.10 (b) — stable_sort on unique keys matches sort (arrival-order independence)",
          "[scheduler][determinism][queue-ordering][stable]") {
  // Defense-in-depth: even if a future refactor swaps sort→stable_sort, the
  // Reference tie-break should keep producing identical output on unique
  // keys. Verifying explicitly guards against someone quietly introducing a
  // ties-depend-on-arrival-order regression downstream.
  constexpr std::size_t kCandidates = 1000;

  const auto base = make_unique_candidates(kCandidates, /*seed=*/0xFACEULL);

  std::vector<ts::TaskCandidate> sorted = base;
  std::sort(sorted.begin(), sorted.end(), ts::ReferenceTaskCompare{});

  std::vector<ts::TaskCandidate> stable = base;
  std::stable_sort(stable.begin(), stable.end(), ts::ReferenceTaskCompare{});

  REQUIRE(candidates_byte_equal(sorted, stable));
}

TEST_CASE("T4.10 (b) — empty and single-element inputs are no-op determ.",
          "[scheduler][determinism][queue-ordering][edge]") {
  std::vector<ts::TaskCandidate> empty;
  std::sort(empty.begin(), empty.end(), ts::ReferenceTaskCompare{});
  CHECK(empty.empty());

  std::vector<ts::TaskCandidate> single(1);
  single[0].zone_id = 42;
  single[0].time_level = 7;
  single[0].canonical_index = 3;
  single[0].version = 11;
  std::sort(single.begin(), single.end(), ts::ReferenceTaskCompare{});
  REQUIRE(single.size() == 1);
  CHECK(single[0].zone_id == 42);
}
