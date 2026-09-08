from __future__ import annotations

import math
import statistics
from typing import Sequence


BenchmarkStats = dict[str, float | int]


def summarize_samples(samples: Sequence[float]) -> BenchmarkStats:
    """Return mean/median/min/max/p95/stddev statistics for a set of timing samples in milliseconds."""
    values = [float(value) for value in samples]
    if not values:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "p95_ms": 0.0,
            "stddev_ms": 0.0,
        }

    sorted_values = sorted(values)
    mean_ms = statistics.fmean(sorted_values)
    median_ms = statistics.median(sorted_values)
    min_ms = min(sorted_values)
    max_ms = max(sorted_values)
    p95_ms = _percentile_nearest_rank(sorted_values, 95.0)
    stddev_ms = statistics.pstdev(sorted_values) if len(sorted_values) > 1 else 0.0
    return {
        "count": len(sorted_values),
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "p95_ms": p95_ms,
        "stddev_ms": stddev_ms,
    }


def _percentile_nearest_rank(sorted_values: Sequence[float], percent: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = max(1, math.ceil((percent / 100.0) * len(sorted_values))) - 1
    return float(sorted_values[min(rank, len(sorted_values) - 1)])


def calculate_overhead(direct_ms: float, mcp_ms: float) -> dict[str, float]:
    """Compute absolute and relative overhead between a direct baseline and the MCP path."""
    direct_value = float(direct_ms)
    mcp_value = float(mcp_ms)
    absolute_ms = max(0.0, mcp_value - direct_value)
    if direct_value == 0:
        relative_pct = 0.0
    else:
        relative_pct = (absolute_ms / direct_value) * 100.0
    return {"absolute_ms": absolute_ms, "relative_pct": relative_pct}


def benchmark_result(*, benchmark: str, direct_ms: float | None = None, mcp_ms: float | None = None, iterations: int, **extra: object) -> dict[str, object]:
    """Create a JSON-friendly benchmark record with normalized overhead metadata."""
    direct_value = float(direct_ms) if direct_ms is not None else None
    mcp_value = float(mcp_ms) if mcp_ms is not None else None

    if direct_value is not None and mcp_value is not None:
        overhead = calculate_overhead(direct_value, mcp_value)
        result = {
            "benchmark": benchmark,
            "iterations": iterations,
            "direct_ms": direct_value,
            "mcp_ms": mcp_value,
            "absolute_overhead_ms": overhead["absolute_ms"],
            "relative_overhead_pct": overhead["relative_pct"],
        }
    else:
        result = {
            "benchmark": benchmark,
            "iterations": iterations,
            "direct_ms": direct_value,
            "mcp_ms": mcp_value,
            "absolute_overhead_ms": None,
            "relative_overhead_pct": None,
        }

    result.update(extra)
    return result
