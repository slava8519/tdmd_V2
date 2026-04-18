"""Cross-benchmark VerifyLab harness package.

Contains the generic differential runner that T1, T4, and future benchmarks
share. Per-benchmark drivers (e.g. ``verify/t1/run_differential.py``) are
thin wrappers that configure this runner with their own files and variant
policy.
"""
