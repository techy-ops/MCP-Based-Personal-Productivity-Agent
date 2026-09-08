from __future__ import annotations

import asyncio
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import BASE_DIR
from app.database.connection import Base
from app.services import calendar_service, note_service, task_service
from benchmarks.benchmark_utils import benchmark_result, summarize_samples
from mcp_client import MCPClient


DEFAULT_OUTPUT = Path(tempfile.gettempdir()) / "mcp_productivity_benchmark.json"


@contextmanager
def patched_session_factory_for(db_path: Path) -> Iterator[None]:
    import app.database.connection as db_connection

    engine = create_engine(f"sqlite:///{db_path.as_posix()}", future=True, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    original_db_session = db_connection.SessionLocal
    original_task_session = task_service.SessionLocal
    original_calendar_session = calendar_service.SessionLocal
    original_note_session = note_service.SessionLocal

    db_connection.SessionLocal = session_factory
    task_service.SessionLocal = session_factory
    calendar_service.SessionLocal = session_factory
    note_service.SessionLocal = session_factory

    try:
        yield
    finally:
        db_connection.SessionLocal = original_db_session
        task_service.SessionLocal = original_task_session
        calendar_service.SessionLocal = original_calendar_session
        note_service.SessionLocal = original_note_session


def _benchmark_synchronous(operation_name: str, direct_fn: Callable[[], Any], *, iterations: int, warmup: int = 3) -> dict[str, Any]:
    times: list[float] = []
    for _ in range(warmup):
        direct_fn()
    for _ in range(iterations):
        start = __import__("time").perf_counter()
        direct_fn()
        elapsed_ms = (__import__("time").perf_counter() - start) * 1000.0
        times.append(elapsed_ms)
    stats = summarize_samples(times)
    result = benchmark_result(benchmark=operation_name, direct_ms=stats["mean_ms"], iterations=iterations, warmup=warmup, **stats)
    result["direct_ms"] = stats["mean_ms"]
    return result


async def _benchmark_mcp_call(operation_name: str, tool_name: str, payload: dict[str, Any], *, iterations: int, warmup: int = 3) -> dict[str, Any]:
    db_path = Path(tempfile.mkdtemp(prefix="mcp_bench_")) / "bench.db"
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{db_path.as_posix()}"}
    client = MCPClient(server_path=BASE_DIR / "mcp_servers" / "unified_server.py", env=env)
    await client.connect()
    await client.list_tools()

    try:
        for _ in range(warmup):
            await client.call_tool(tool_name, payload)
        times: list[float] = []
        for _ in range(iterations):
            start = __import__("time").perf_counter()
            await client.call_tool(tool_name, payload)
            elapsed_ms = (__import__("time").perf_counter() - start) * 1000.0
            times.append(elapsed_ms)
        stats = summarize_samples(times)
        result = benchmark_result(benchmark=operation_name, mcp_ms=stats["mean_ms"], iterations=iterations, warmup=warmup, **stats)
        result["mcp_ms"] = stats["mean_ms"]
        return result
    finally:
        await client.close()


async def _measure_connect_latency(iterations: int, warmup: int = 3) -> dict[str, Any]:
    times: list[float] = []
    for _ in range(warmup):
        client = MCPClient(server_path=BASE_DIR / "mcp_servers" / "unified_server.py")
        await client.connect(); await client.close()
    for _ in range(iterations):
        client = MCPClient(server_path=BASE_DIR / "mcp_servers" / "unified_server.py")
        start = __import__("time").perf_counter()
        await client.connect()
        elapsed_ms = (__import__("time").perf_counter() - start) * 1000.0
        times.append(elapsed_ms)
        await client.close()
    stats = summarize_samples(times)
    return {"benchmark": "connection_initialization", "iterations": iterations, "warmup": warmup, **stats}


async def _measure_discovery_latency(iterations: int, warmup: int = 3) -> dict[str, Any]:
    times: list[float] = []
    client = MCPClient(server_path=BASE_DIR / "mcp_servers" / "unified_server.py")
    await client.connect()
    try:
        for _ in range(warmup):
            await client.list_tools()
        for _ in range(iterations):
            start = __import__("time").perf_counter()
            await client.list_tools()
            elapsed_ms = (__import__("time").perf_counter() - start) * 1000.0
            times.append(elapsed_ms)
    finally:
        await client.close()
    stats = summarize_samples(times)
    return {"benchmark": "tool_discovery", "iterations": iterations, "warmup": warmup, **stats}


async def _measure_session_reuse(iterations: int) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for _ in range(iterations):
        client = MCPClient(server_path=BASE_DIR / "mcp_servers" / "unified_server.py")
        await client.connect()
        start = __import__("time").perf_counter()
        for index in range(5):
            await client.call_tool("create_task", {"title": f"session-task-{index}-{_}", "description": "reuse"})
        elapsed_ms = (__import__("time").perf_counter() - start) * 1000.0
        details.append({"duration_ms": elapsed_ms})
        await client.close()
    stats = summarize_samples([item["duration_ms"] for item in details])
    return {"benchmark": "session_reuse_5_calls", "iterations": iterations, **stats}


async def _measure_reconnect(iterations: int) -> dict[str, Any]:
    times: list[float] = []
    for _ in range(iterations):
        client = MCPClient(server_path=BASE_DIR / "mcp_servers" / "unified_server.py")
        await client.connect(); await client.close()
        start = __import__("time").perf_counter(); await client.connect(); elapsed_ms = (__import__("time").perf_counter() - start) * 1000.0; times.append(elapsed_ms); await client.close()
    stats = summarize_samples(times)
    return {"benchmark": "reconnect", "iterations": iterations, **stats}


async def _measure_cross_domain_workflow(iterations: int) -> dict[str, Any]:
    times: list[float] = []
    for index in range(iterations):
        client = MCPClient(server_path=BASE_DIR / "mcp_servers" / "unified_server.py")
        await client.connect()
        start = __import__("time").perf_counter()
        await client.call_tool("create_task", {"title": f"wf-task-{index}", "description": "workflow"})
        await client.call_tool("create_event", {"title": f"wf-event-{index}", "start_time": "2026-09-10T09:00:00", "end_time": "2026-09-10T10:00:00"})
        await client.call_tool("create_note", {"title": f"wf-note-{index}", "content": "workflow benchmark"})
        elapsed_ms = (__import__("time").perf_counter() - start) * 1000.0
        times.append(elapsed_ms)
        await client.close()
    stats = summarize_samples(times)
    return {"benchmark": "cross_domain_workflow", "iterations": iterations, **stats}


def _direct_create_task() -> Any:
    db_path = Path(tempfile.mkdtemp(prefix="bench_direct_")) / "task.db"
    with patched_session_factory_for(db_path):
        task = task_service.create_task(title="benchmark-task", description="direct benchmark", priority="high")
        return task


def _direct_create_event() -> Any:
    db_path = Path(tempfile.mkdtemp(prefix="bench_direct_")) / "event.db"
    with patched_session_factory_for(db_path):
        start = datetime.now(timezone.utc)
        end = start + timedelta(hours=1)
        return calendar_service.create_event(title="benchmark-event", description="direct benchmark", start_time=start, end_time=end)


def _direct_create_note() -> Any:
    db_path = Path(tempfile.mkdtemp(prefix="bench_direct_")) / "note.db"
    with patched_session_factory_for(db_path):
        return note_service.create_note(title="benchmark-note", content="direct benchmark")


async def main() -> None:
    output_path = Path(os.environ.get("MCP_BENCHMARK_OUTPUT", str(DEFAULT_OUTPUT)))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    benchmark_data: dict[str, Any] = {
        "environment": {
            "python_version": os.sys.version.split()[0],
            "os": os.name,
            "database": "sqlite",
            "dataset": "deterministic synthetic task/event/note payloads",
            "warmup_iterations": 3,
            "measured_iterations": 20,
            "benchmark_command": "python -m benchmarks.benchmark_mcp",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "benchmarks": {},
    }

    benchmark_data["benchmarks"]["connection_initialization"] = await _measure_connect_latency(20, warmup=3)
    benchmark_data["benchmarks"]["tool_discovery"] = await _measure_discovery_latency(20, warmup=3)

    direct_create_task = _benchmark_synchronous("direct_create_task", _direct_create_task, iterations=20, warmup=3)
    mcp_create_task = await _benchmark_mcp_call("mcp_create_task", "create_task", {"title": "bench-task", "description": "mcp benchmark"}, iterations=20, warmup=3)
    benchmark_data["benchmarks"]["create_task"] = {
        "direct": direct_create_task,
        "mcp": mcp_create_task,
        "overhead": benchmark_result(
            benchmark="create_task_overhead",
            direct_ms=direct_create_task["mean_ms"],
            mcp_ms=mcp_create_task["mean_ms"],
            iterations=20,
        ),
    }

    direct_create_event = _benchmark_synchronous("direct_create_event", _direct_create_event, iterations=20, warmup=3)
    mcp_create_event = await _benchmark_mcp_call("mcp_create_event", "create_event", {"title": "bench-event", "start_time": "2026-09-10T09:00:00", "end_time": "2026-09-10T10:00:00"}, iterations=20, warmup=3)
    benchmark_data["benchmarks"]["create_event"] = {
        "direct": direct_create_event,
        "mcp": mcp_create_event,
        "overhead": benchmark_result(
            benchmark="create_event_overhead",
            direct_ms=direct_create_event["mean_ms"],
            mcp_ms=mcp_create_event["mean_ms"],
            iterations=20,
        ),
    }

    direct_create_note = _benchmark_synchronous("direct_create_note", _direct_create_note, iterations=20, warmup=3)
    mcp_create_note = await _benchmark_mcp_call("mcp_create_note", "create_note", {"title": "bench-note", "content": "mcp benchmark"}, iterations=20, warmup=3)
    benchmark_data["benchmarks"]["create_note"] = {
        "direct": direct_create_note,
        "mcp": mcp_create_note,
        "overhead": benchmark_result(
            benchmark="create_note_overhead",
            direct_ms=direct_create_note["mean_ms"],
            mcp_ms=mcp_create_note["mean_ms"],
            iterations=20,
        ),
    }

    benchmark_data["benchmarks"]["session_reuse"] = await _measure_session_reuse(10)
    benchmark_data["benchmarks"]["reconnect"] = await _measure_reconnect(10)
    benchmark_data["benchmarks"]["cross_domain_workflow"] = await _measure_cross_domain_workflow(10)

    output_path.write_text(json.dumps(benchmark_data, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "summary": benchmark_data["benchmarks"]}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
