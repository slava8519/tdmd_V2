"""Hardware FLOPs probe with 24h on-disk cache.

Wraps ``verify/benchmarks/t3_al_fcc_large_anchor/hardware_normalization.py``
so the anchor-test harness amortises a ~5-second micro-benchmark across
repeat runs. Cache location: ``~/.cache/tdmd/hardware_flops.json``.

The exec pack (T5.11, line 808-809) originally called for a C++ LJ
micro-kernel. T5.10 shipped Python-only; this probe wraps that proxy and
records the fact in the report (``HardwareProbeResult.probe_tool``
isn't stored, but ``normalization_log`` carries the note). Replacing the
proxy with a native TDMD micro-benchmark is the documented T5.11
refinement and will drop right into this file without touching the
runner.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import subprocess
import sys

from .report import HardwareProbeResult

# 24h TTL — re-probe once per day so hot machine-state drift (thermal
# throttling, DVFS re-pinning) eventually resolves. Per-run re-probing
# is available via ``--force-probe``.
CACHE_TTL_SECONDS = 24 * 60 * 60


def _cache_path() -> pathlib.Path:
    """~/.cache/tdmd/hardware_flops.json — honours XDG_CACHE_HOME."""
    xdg = os.environ.get("XDG_CACHE_HOME", "")
    if xdg:
        root = pathlib.Path(xdg)
    else:
        root = pathlib.Path.home() / ".cache"
    return root / "tdmd" / "hardware_flops.json"


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(tz=datetime.timezone.utc)


def _cache_is_fresh(entry: dict) -> bool:
    try:
        ts = datetime.datetime.fromisoformat(entry["probe_timestamp_utc"])
    except (KeyError, ValueError):
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=datetime.timezone.utc)
    age = (_now_utc() - ts).total_seconds()
    return 0 <= age < CACHE_TTL_SECONDS


def run_fresh_probe(
    baseline_gflops: float | None = None,
    hardware_normalization_script: pathlib.Path | None = None,
) -> dict:
    """Invoke the stdlib-only Python probe and return its JSON payload.

    Resolves the T5.10 script by default so callers that construct the
    runner with the bundled benchmark directory never need to hand-plumb
    the path. Unit tests override ``hardware_normalization_script`` with
    a tmp-scratch equivalent.
    """
    if hardware_normalization_script is None:
        repo_root = pathlib.Path(__file__).resolve().parents[3]
        hardware_normalization_script = (
            repo_root
            / "verify"
            / "benchmarks"
            / "t3_al_fcc_large_anchor"
            / "hardware_normalization.py"
        )
    if not hardware_normalization_script.is_file():
        raise FileNotFoundError(
            f"hardware_normalization.py not found at {hardware_normalization_script}"
        )

    cmd = [sys.executable, str(hardware_normalization_script), "--json"]
    if baseline_gflops is not None:
        cmd += ["--baseline-gflops", f"{baseline_gflops:.6f}"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"hardware_normalization.py exited with code {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    try:
        payload = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"hardware_normalization.py did not emit JSON: '{result.stdout.strip()}'"
        ) from exc
    payload["probe_timestamp_utc"] = _now_utc().isoformat(timespec="seconds")
    return payload


def probe(
    force_probe: bool = False,
    baseline_gflops: float | None = None,
    hardware_normalization_script: pathlib.Path | None = None,
    cache_path: pathlib.Path | None = None,
) -> HardwareProbeResult:
    """Return a fresh or cached :class:`HardwareProbeResult`.

    Tries the on-disk cache first (unless ``force_probe`` is set); if the
    cache is missing, stale (>24h), or corrupt, re-runs the micro-kernel
    and writes back.
    """
    if cache_path is None:
        cache_path = _cache_path()

    cached_payload: dict | None = None
    if not force_probe and cache_path.is_file():
        try:
            cached_payload = json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            cached_payload = None

    if cached_payload is not None and _cache_is_fresh(cached_payload):
        return HardwareProbeResult(
            local_gflops=float(cached_payload["local_gflops"]),
            baseline_gflops=float(cached_payload["baseline_gflops"]),
            ghz_flops_ratio=float(cached_payload["ghz_flops_ratio"]),
            probe_timestamp_utc=str(cached_payload["probe_timestamp_utc"]),
            cached=True,
        )

    fresh = run_fresh_probe(
        baseline_gflops=baseline_gflops,
        hardware_normalization_script=hardware_normalization_script,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(fresh, indent=2, sort_keys=False))
    return HardwareProbeResult(
        local_gflops=float(fresh["local_gflops"]),
        baseline_gflops=float(fresh["baseline_gflops"]),
        ghz_flops_ratio=float(fresh["ghz_flops_ratio"]),
        probe_timestamp_utc=str(fresh["probe_timestamp_utc"]),
        cached=False,
    )
