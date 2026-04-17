#!/usr/bin/env python3
"""
tdmd-check-restrict — STUB.

Purpose (future): enforce master spec §D.16 and playbook §5.1:
    Hot kernels (functions marked [[tdmd::hot_kernel]] or residing in a
    .cu file with pointer parameters) MUST qualify those pointers with
    `__restrict__`, unless an adjacent comment `NOLINT(tdmd-missing-restrict): <rationale>`
    documents why the restriction is unsafe.

Current state: STUB.
    - Always exits 0.
    - Does not parse C++ AST. A real implementation requires libclang or a
      custom clang-tidy C++ plugin (bundled with the LLVM dev headers and
      exposed via matchers::parmVarDecl / hasType / pointerType).
    - Scheduled for M2-M3, once we actually have hot kernels to protect.

Why a stub now?
    - Locks the hook name and invocation path into the pre-commit pipeline,
      so switching to the real implementation is a one-line change.
    - Documents the rule visibly in the repo, so reviewers catch missing
      __restrict__ by eye until the automated check lands.

Invocation:
    tools/lint/check_restrict_stub.py <path>...
"""

import sys


def main(paths: list[str]) -> int:
    # No-op for now. Never fails.
    _ = paths  # deliberately unused
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
