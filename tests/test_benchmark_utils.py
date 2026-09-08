import math

import pytest

from benchmarks.benchmark_utils import BenchmarkStats, benchmark_result, calculate_overhead, summarize_samples


def test_summarize_samples_basic():
    samples = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = summarize_samples(samples)

    assert stats["count"] == 5
    assert stats["mean_ms"] == pytest.approx(30.0)
    assert stats["median_ms"] == pytest.approx(30.0)
    assert stats["min_ms"] == pytest.approx(10.0)
    assert stats["max_ms"] == pytest.approx(50.0)
    assert stats["p95_ms"] == pytest.approx(50.0)
    assert stats["stddev_ms"] == pytest.approx(15.811388300841896, rel=1e-6)


def test_summarize_samples_empty_input():
    stats = summarize_samples([])

    assert stats == {
        "count": 0,
        "mean_ms": 0.0,
        "median_ms": 0.0,
        "min_ms": 0.0,
        "max_ms": 0.0,
        "p95_ms": 0.0,
        "stddev_ms": 0.0,
    }


def test_calculate_overhead():
    direct = 50.0
    mcp = 120.0
    overhead = calculate_overhead(direct, mcp)

    assert overhead["absolute_ms"] == pytest.approx(70.0)
    assert overhead["relative_pct"] == pytest.approx(140.0)


def test_benchmark_result_serializes():
    result = benchmark_result(
        benchmark="create_task",
        direct_ms=50.0,
        mcp_ms=120.0,
        iterations=10,
    )

    assert result["benchmark"] == "create_task"
    assert result["iterations"] == 10
    assert result["direct_ms"] == pytest.approx(50.0)
    assert result["mcp_ms"] == pytest.approx(120.0)
    assert result["absolute_overhead_ms"] == pytest.approx(70.0)
    assert result["relative_overhead_pct"] == pytest.approx(140.0)
