# MCP-Based Personal Productivity Agent

## Project status

This repository contains a complete Phase 2 MCP backend foundation and the Phase 3.1 LLM integration foundation for a personal productivity system. It includes the validated service layer, real MCP stdio transport integration, dynamic tool discovery, generic invocation, integration and security checks, a reproducible benchmarking harness, and a provider-isolated LLM client.

## Phase completion status

- Phase 1 — COMPLETE
- Phase 2.1 — COMPLETE
- Phase 2.2 — COMPLETE
- Phase 2.3 — COMPLETE
- Phase 2.4 — COMPLETE
- Phase 2.5 — COMPLETE
- Phase 2.6 — ABSORBED INTO 2.5
- Phase 2.7.1 — COMPLETE
- Phase 2.7.2 — COMPLETE
- Phase 2.7.3 — COMPLETE
- Phase 2.7.4 — COMPLETE
- Phase 2.8 — COMPLETE
- Phase 3.1 — COMPLETE
- Phase 3.2 — NEXT

## Technology stack

- Python 3.11+
- SQLite
- SQLAlchemy ORM
- Pydantic
- FastMCP
- MCP SDK
- pytest
- python-dotenv
- OpenAI SDK (provider adapter for Phase 3.1)

## Architecture

The verified MCP architecture is:

MCP Client
    ↓
MCP stdio transport
    ↓
Unified MCP Server
    ↓
Task / Calendar / Notes tools
    ↓
Service layer
    ↓
SQLite database

The service layer remains the source of business logic; the MCP layer is intentionally thin and exposes the same validated behavior through tool adapters.

## Phase 3.1 LLM foundation

Phase 3.1 adds a small, provider-neutral LLM abstraction without connecting it to MCP, databases, tools, or agent orchestration.

Current Phase 3.1 architecture:

USER / FUTURE AGENT
    ↓
LLM abstraction (LLMClient)
    ↓
Provider adapter (OpenAIProvider)
    ↓
Configured LLM API

The client returns a normalized response containing text, model information, and safe metadata. Credentials and model settings come from environment variables; deterministic tests inject a fake provider. No LangGraph graph, tool calling, planning, ReAct loop, or productivity-data pipeline is implemented yet.

The MCP architecture remains independent. These layers are intentionally kept separate until a future Phase 3.2 integration.

### LLM configuration

Copy the variable names from `.env.example` into a local `.env` and provide values through the environment. `.env` is ignored by Git. `LLM_MODEL` and `LLM_API_KEY` are required when constructing `LLMClient.from_env()`.

## Functional scope

The project includes:

- SQLAlchemy database models for tasks, calendar events, and notes
- SQLite configuration and session handling
- Pydantic validation schemas
- service-layer business logic
- MCP server adapters for all domains
- unified MCP server with exactly 17 tools
- dynamic tool discovery via the real MCP client
- generic tool invocation over the active session
- integration testing and reliability validation
- benchmarking utilities for direct-vs-MCP comparisons

## Unified MCP tool inventory

The Unified MCP Server exposes exactly 17 tools:

### Task tools
- create_task
- get_task
- list_tasks
- update_task
- complete_task
- delete_task

### Calendar tools
- create_event
- get_event
- list_events
- update_event
- delete_event

### Notes tools
- create_note
- get_note
- list_notes
- update_note
- delete_note
- search_notes

## Getting started

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run tests

```powershell
pytest -q
```

## Benchmarking

The benchmark suite is intentionally small and reproducible. It compares direct service-layer access against the real MCP stdio path while using isolated SQLite databases and deterministic synthetic payloads.

Run the benchmark with:

```powershell
python -m benchmarks.benchmark_mcp
```

Optional output override:

```powershell
$env:MCP_BENCHMARK_OUTPUT = "C:\temp\mcp_productivity_benchmark.json"
python -m benchmarks.benchmark_mcp
```

The default output location is the system temp directory, and the script writes a JSON artifact containing measured latencies, summary statistics, and overhead deltas.

## Benchmark interpretation

The benchmark distinguishes:

- connection initialization latency
- dynamic discovery latency
- representative task/calendar/note invocation latency
- warm-session repeated operation behavior
- reconnect costs
- cross-domain workflow timing

This is an experimental baseline for the architecture itself. It does not claim universal production performance and intentionally excludes LLM reasoning, planning, and frontend concerns.

## Security and reliability

The repository includes validation and reliability checks for:

- invalid tool names
- non-dictionary arguments
- server connection issues
- reconnect after closure
- lifecycle cleanup
- database isolation during tests
- repeated operations under controlled conditions

## Known limitations

- The benchmark is limited to the current local SQLite-backed architecture.
- The results are environment-dependent and should be interpreted as a reproducible baseline rather than a universal performance guarantee.
- Concurrency benchmarking is intentionally outside the current Phase 2 MCP contract.
- No LLM, LangGraph, frontend, or autonomous planning layer is implemented in this phase.

## Repository structure

```text
mcp-productivity-agent/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   └── models.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── task.py
│   │   ├── calendar.py
│   │   └── note.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── task_service.py
│   │   ├── calendar_service.py
│   │   └── note_service.py
│   └── utils/
│       ├── __init__.py
│       └── validators.py
├── benchmarks/
│   ├── __init__.py
│   ├── benchmark_utils.py
│   └── benchmark_mcp.py
├── mcp_client/
│   ├── __init__.py
│   ├── client.py
│   └── exceptions.py
├── llm/
│   ├── __init__.py
│   ├── client.py
│   ├── config.py
│   └── exceptions.py
├── mcp_servers/
│   ├── __init__.py
│   ├── task_server.py
│   ├── calendar_server.py
│   ├── notes_server.py
│   └── unified_server.py
├── tests/
│   ├── __init__.py
│   ├── test_benchmark_utils.py
│   ├── test_tasks.py
│   ├── test_calendar.py
│   ├── test_notes.py
│   ├── test_task_mcp_server.py
│   ├── test_calendar_mcp_server.py
│   ├── test_notes_mcp_server.py
│   ├── test_unified_mcp_server.py
│   ├── test_mcp_client.py
│   └── test_reliability.py
├── data/
├── scripts/
│   └── seed_database.py
├── .gitignore
├── requirements.txt
├── README.md
├── main.py
└── .env.example
```

## Final note

Phase 3.1 is finalized as an LLM foundation. The present repository does not implement LangGraph, LLM reasoning, agent planning, MCP tool calling, frontend UIs, or autonomous orchestration. Phase 3.2 is next.


## Phase 2.7.1 — FULL SYSTEM INTEGRATION TESTING

Status: COMPLETE

This phase verifies the complete MCP architecture as an integrated system rather than testing individual domain components in isolation. The implementation confirms the real end-to-end flow:

MCP Client
    ↓
MCP stdio transport
    ↓
Unified MCP Server
    ↓
Task / Calendar / Notes tools
    ↓
Task / Calendar / Notes services
    ↓
SQLite database

The verified integration coverage includes:

- 17-tool unified discovery and registration
- MCP Client → Unified MCP Server connection, initialization, discovery, and generic invocation
- end-to-end Task CRUD and completion workflow
- end-to-end Calendar create/list/update/delete workflow with conflict validation
- end-to-end Notes create/list/update/search/delete workflow
- cross-domain workflows across task, calendar, and note records in one MCP session
- isolated database test fixtures to prevent production data changes
- empty-data handling for fresh databases
- persistence verification across repeated MCP calls

Validation command:

```powershell
pytest -q
```

The current integration status is complete for Phase 2.7.1. Final Phase 2.7 verification is documented below.

## Phase 2.7.2 — SECURITY & INPUT VALIDATION AUDIT

Status: COMPLETE

This phase performed a focused security and input-validation audit of the current MCP architecture. The audit covered:

- MCP client tool-name, argument, discovery, and disconnected-state handling
- Task, Calendar, and Notes malformed-input and boundary validation
- invalid IDs, strict MCP argument types, calendar range integrity, and cross-domain isolation
- SQL injection resistance using isolated SQLite databases and parameterized ORM queries
- SQL-like, HTML-like, wildcard, Unicode, and other user text handling as data
- controlled exception responses without exposing unexpected internal exception details
- configuration, environment, dependency, command-execution, filesystem, and repository secret review
- security regression tests using isolated in-memory or temporary databases

The audit fixed the identified input-boundary, calendar-integrity, search-pattern, and error-leakage issues. It does not claim absolute security; authentication, authorization, rate limiting, deployment hardening, and broader recovery architecture remain outside this sub-phase.

Validation command:

```powershell
pytest -q
```

Next: 2.7.3 — Reliability, Failure & Recovery Testing

## Phase 2.7.3 — RELIABILITY, FAILURE & RECOVERY TESTING

Status: COMPLETE

This phase tested the existing MCP lifecycle under controlled connection, transport, server, tool, and database failures. Coverage included:

- failed startup, partial session initialization, repeated connect, repeated close, and client state transitions
- real MCP server termination, transport failure detection, cleanup, reconnect, fresh 17-tool discovery, and persisted-data recovery
- expected tool failure followed by successful cross-domain operations in the same session
- isolated SQLite commit failures, rollback behavior, update-state preservation, and no partial task persistence
- subprocess and async session cleanup after normal and failed lifecycle paths
- repeated sequential and cross-domain workflows already covered by the integration suite

The reliability fix clears stale client session and transport state when an MCP invocation fails at the transport boundary, allowing deterministic reconnect. No new dependencies were required. Concurrent client use is not part of the current contract and was not forced into the architecture. This phase does not claim guaranteed recovery for every operating-system, process, or network failure mode.

Validation command:

```powershell
pytest -q
```

Next: 2.7.4 — Final Integration & Security Verification

## Phase 2.7.4 — FINAL INTEGRATION & SECURITY VERIFICATION

Status: COMPLETE

This final verification confirmed the complete current MCP architecture through the real client and stdio transport using isolated temporary SQLite databases. The verification covered:

- dynamic discovery of exactly 17 unique Unified MCP tools with descriptions and input schemas
- successful invocation of every Task, Calendar, and Notes tool
- complete Task, Calendar, and Notes workflows, including validation, conflicts, search, deletion, and missing-record behavior
- cross-domain operations in one MCP session and persistence across close/reconnect
- standalone Task, Calendar, Notes, and Unified server startup and expected tool counts of 6, 5, 6, and 17
- final security regression coverage for IDs, boolean IDs, invalid input, SQL-like text, literal note wildcards, calendar update ranges, and safe errors
- final reliability coverage for startup failure, partial initialization, server termination, reconnect, tool failure recovery, rollback, state consistency, and subprocess cleanup
- repository secret/configuration review, dependency review, isolated test databases, and artifact hygiene

The final live verification produced successful results for Task, Calendar, Notes, cross-domain operations, persistence/reconnect, and clean shutdown. No production database was used or modified. No new dependencies were added.

Validation command:

```powershell
pytest -q
```

Phase 2.7 is fully verified. Next: 2.8 — Finalization & Benchmarking.

- list_tasks
- update_task
- complete_task
- delete_task

Local startup command:

```powershell
python -m mcp_servers.task_server
```

This starts the FastMCP task server over stdio so it can be connected by an MCP client in later phases.

## Calendar MCP Integration (Phase 2.2)

The calendar domain is now exposed through a dedicated FastMCP server that stays thin and delegates to the existing Phase 1 calendar business logic.

Architecture:

Calendar MCP Tools
    ↓
Calendar Service
    ↓
Database

This preserves the existing calendar validation and overlap conflict detection while exposing the tools needed by an MCP-capable client.

The Calendar MCP server exposes these tools:

- create_event
- get_event
- list_events
- update_event
- delete_event

Local startup command:

```powershell
python -m mcp_servers.calendar_server
```

The MCP layer preserves the existing calendar service behavior, including:

- input and time-range validation
- date-range filtering
- overlap conflict detection during creation and updates
- event updates
- event deletion

## Notes MCP Integration (Phase 2.3)

The notes domain is exposed through a dedicated FastMCP server that delegates all operations to the existing Phase 1 notes service.

Architecture:

Notes MCP Tool
    ↓
Notes Service
    ↓
Database

The Notes MCP server exposes:

- create_note
- get_note
- list_notes
- update_note
- delete_note
- search_notes

Local startup command:

```powershell
python -m mcp_servers.notes_server
```

All Notes MCP tools preserve the existing service-layer validation, persistence, missing-note handling, and case-insensitive title/content search behavior.

## Unified MCP Architecture (Phase 2.4)

The Unified MCP architecture is complete. The unified server exposes the existing Task, Calendar, and Notes MCP tools through one FastMCP application while preserving the individual domain servers.

Architecture:

Unified MCP Server
       ↓
    ┌──────────────┼──────────────┐
    ↓              ↓              ↓
Task Tools    Calendar Tools    Notes Tools
    │              │              │
    ↓              ↓              ↓
Task Service  Calendar Service  Notes Service
    │              │              │
    └──────────────┼──────────────┘
                   ↓
                Database

The unified server exposes 17 tools in total:

Task tools:

- create_task
- get_task
- list_tasks
- update_task
- complete_task
- delete_task

Calendar tools:

- create_event
- get_event
- list_events
- update_event
- delete_event

Notes tools:

- create_note
- get_note
- list_notes
- update_note
- delete_note
- search_notes

Local startup command:

```powershell
python -m mcp_servers.unified_server
```

The individual Task, Calendar, and Notes MCP servers remain available. The unified server is an additional consolidated interface, and all three domains continue to delegate to their existing service layers. AI and LLM integration has not been implemented.

Phase 2.4 sub-phases:

- 2.4.1 Unified Server Foundation — complete
- 2.4.2 Calendar + Notes Integration — complete
- 2.4.3 Final Integration + Verification — complete

## Current limitations

- No AI or LLM integration
- No Streamlit frontend
- No authentication or user accounts
- No external productivity APIs
- No semantic search or embeddings
- SQLite is used locally for development and demonstration

## Future phases

- Phase 2:
    - 2.1 Task MCP Server — complete
    - 2.2 Calendar MCP Server — complete
    - 2.3 Notes MCP Server — complete
    - 2.4 Unified MCP Architecture — complete
      - 2.4.1 Unified Server Foundation — complete
      - 2.4.2 Calendar + Notes Integration — complete
      - 2.4.3 Final Integration + Verification — complete
- Phase 2.5: MCP Client
  - 2.5.1 MCP Client Foundation & Lifecycle — complete
  - 2.5.2 MCP Client Tool Discovery — complete
    - 2.5.3 Generic Tool Invocation & Error Handling — complete
- Phase 2.6: Dynamic Discovery & Invocation — absorbed into Phase 2.5
- Phase 2.7:
    - 2.7.1 Full System Integration Testing — complete
    - 2.7.2 Security & Input Validation Audit — complete
    - 2.7.3 Reliability, Failure & Recovery Testing — complete
    - 2.7.4 Final Integration & Security Verification — complete
- Phase 2.8: Finalization & Benchmarking — next

## Important note

Phase 1 through Phase 2.7.4 are complete. Phase 2.8 — Finalization & Benchmarking — is next. LangGraph, LLM integration, agent orchestration, AI features, and frontend interfaces remain future work. The backend is deliberately designed so those layers can be added later without rewriting the core services.
