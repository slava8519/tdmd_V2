// SPEC: docs/specs/gpu/SPEC.md §9 (NVTX), D-M6-14
// Exec pack: docs/development/m6_execution_pack.md T6.11
//
// Grep-based NVTX audit. Walks each .cu translation unit under src/gpu/ and
// asserts every `<<< ... >>>` kernel launch is inside a block that contains
// at least one `TDMD_NVTX_RANGE(...)` macro. This enforces D-M6-14 without
// requiring an Nsight capture in CI.
//
// The audit is deliberately structural, not semantic: it detects that a
// range exists within the surrounding `{ ... }` scope, not that the range
// name is specific to that kernel. Meaningful name assignments are reviewed
// in PR; the CI gate stops the regression of "forgot the range entirely".
//
// On TDMD_BUILD_CUDA=OFF the .cu files compile as C++ and contain no `<<<`
// markers, so the audit trivially passes — the test still runs to protect
// the audit harness itself from bitrot.

#include "tdmd/gpu/gpu_config.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef TDMD_GPU_SRC_DIR
#error "TDMD_GPU_SRC_DIR must be defined by the build system"
#endif

namespace {

std::string slurp(const std::filesystem::path& path) {
  std::ifstream f(path);
  std::ostringstream oss;
  oss << f.rdbuf();
  return oss.str();
}

// Find the opening brace of the enclosing `{ ... }` scope that contains the
// character at `pos` in `src`. Returns npos if `pos` is at file scope (no
// enclosing brace) — that case is only hit for macros expanded at file scope,
// which we don't use for kernel launches, so it's a valid failure indicator.
std::size_t find_enclosing_brace(const std::string& src, std::size_t pos) {
  int depth = 0;
  for (std::size_t i = pos; i-- > 0;) {
    if (src[i] == '}') {
      ++depth;
    } else if (src[i] == '{') {
      if (depth == 0) {
        return i;
      }
      --depth;
    }
  }
  return std::string::npos;
}

// Returns true iff there is at least one `TDMD_NVTX_RANGE` token between
// `scope_open` (an opening brace position) and `launch_pos` (the `<<<`
// position). The macro expansion site is what matters for Nsight — a range
// opened in an outer scope still covers the nested kernel, but we require
// "same-scope or ancestor-scope" coverage, which the brace walk enforces.
bool scope_has_nvtx_range(const std::string& src, std::size_t scope_open, std::size_t launch_pos) {
  const std::string marker = "TDMD_NVTX_RANGE";
  std::size_t cur = scope_open;
  while (true) {
    const std::size_t found = src.find(marker, cur);
    if (found == std::string::npos || found > launch_pos) {
      return false;
    }
    // Check the found position is not inside a comment or string literal. A
    // rough filter: walk back to the line start, skip leading whitespace,
    // reject if line starts with '//' or '*'. Sufficient for this codebase.
    std::size_t line_start = src.rfind('\n', found);
    line_start = (line_start == std::string::npos) ? 0 : line_start + 1;
    std::string leading = src.substr(line_start, found - line_start);
    auto is_ws = [](char c) { return c == ' ' || c == '\t'; };
    const auto first_non_ws = std::find_if_not(leading.begin(), leading.end(), is_ws);
    const std::string trimmed(first_non_ws, leading.end());
    const bool is_comment =
        trimmed.rfind("//", 0) == 0 || trimmed.rfind("*", 0) == 0 || trimmed.rfind("/*", 0) == 0;
    if (!is_comment) {
      return true;
    }
    cur = found + marker.size();
  }
}

struct LaunchAudit {
  std::string file;
  std::size_t line;
  bool wrapped;
};

std::size_t line_of(const std::string& src, std::size_t pos) {
  return std::count(src.begin(), src.begin() + static_cast<std::ptrdiff_t>(pos), '\n') + 1;
}

std::vector<LaunchAudit> audit_file(const std::filesystem::path& path) {
  const std::string src = slurp(path);
  std::vector<LaunchAudit> out;
  const std::string needle = "<<<";
  std::size_t cur = 0;
  while (true) {
    const std::size_t found = src.find(needle, cur);
    if (found == std::string::npos) {
      break;
    }
    cur = found + needle.size();
    // Skip `<<<` occurrences inside comment lines (e.g. NVTX example docs).
    const std::size_t line_start = src.rfind('\n', found);
    const std::size_t ls = (line_start == std::string::npos) ? 0 : line_start + 1;
    const std::string leading = src.substr(ls, found - ls);
    auto is_ws = [](char c) { return c == ' ' || c == '\t'; };
    const auto first_non_ws = std::find_if_not(leading.begin(), leading.end(), is_ws);
    const std::string trimmed(first_non_ws, leading.end());
    if (trimmed.rfind("//", 0) == 0 || trimmed.rfind("*", 0) == 0) {
      continue;
    }

    const std::size_t scope = find_enclosing_brace(src, found);
    const bool wrapped = scope != std::string::npos && scope_has_nvtx_range(src, scope, found);
    out.push_back({path.filename().string(), line_of(src, found), wrapped});
  }
  return out;
}

}  // namespace

TEST_CASE("NVTX audit — every kernel launch wrapped in a TDMD_NVTX_RANGE scope",
          "[gpu][nvtx][audit]") {
  const std::filesystem::path root = TDMD_GPU_SRC_DIR;
  REQUIRE(std::filesystem::is_directory(root));

  std::vector<LaunchAudit> failures;
  std::size_t total = 0;
  for (const auto& entry : std::filesystem::directory_iterator(root)) {
    if (entry.path().extension() != ".cu") {
      continue;
    }
    const auto launches = audit_file(entry.path());
    total += launches.size();
    for (const auto& a : launches) {
      if (!a.wrapped) {
        failures.push_back(a);
      }
    }
  }

  if (!failures.empty()) {
    std::ostringstream msg;
    msg << "NVTX audit found " << failures.size() << " unwrapped kernel launches:\n";
    for (const auto& f : failures) {
      msg << "  " << f.file << ":" << f.line << " — launch not inside TDMD_NVTX_RANGE scope\n";
    }
    FAIL(msg.str());
  }

#if TDMD_BUILD_CUDA
  // Sanity: we expect at least a few kernels to exist in a CUDA build. If the
  // count collapses to zero, the audit wouldn't fail but the file globbing
  // would silently be missing everything.
  REQUIRE(total > 0U);
#else
  SUCCEED("CPU-only build — .cu files contain no kernel launches");
#endif
}
