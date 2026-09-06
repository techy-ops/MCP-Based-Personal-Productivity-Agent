# MCP-Based Personal Productivity Agent

## Project description

This project is a local backend foundation for a personal productivity system. It manages tasks, calendar events, and notes using a lightweight SQLAlchemy + SQLite architecture that is intentionally designed to be easy to wrap later with MCP tools.

## Phase 1 scope

Phase 1 focuses only on the backend foundation and database layer. It includes:

- SQLAlchemy database models
- SQLite configuration and session handling
- Pydantic validation schemas
- Service-layer business logic for tasks, events, and notes
- Local CLI verification
- Seed data script
- Pytest-based test coverage

MCP tools, LangGraph, AI features, frontend interfaces, external APIs, and cloud deployment are intentionally not implemented in this phase.

## Technology stack

- Python 3.11+
- SQLite
- SQLAlchemy ORM
- Pydantic
- pytest
- python-dotenv

## Architecture

The application follows a clean service-oriented flow:

DATABASE
    ↓
MODELS
    ↓
SCHEMAS
    ↓
SERVICES
    ↓
CLI DEMO / FUTURE MCP TOOL WRAPPERS

The service layer is the key boundary for future Phase 2 MCP exposure.

## Folder structure

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
├── tests/
│   ├── __init__.py
│   ├── test_tasks.py
│   ├── test_calendar.py
│   └── test_notes.py
├── data/
│   └── .gitkeep
├── scripts/
│   └── seed_database.py
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── main.py
└── data/productivity.db
```

## Setup instructions

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
cd path\to\mcp-productivity-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

The default configuration is:

```env
DATABASE_URL=sqlite:///data/productivity.db
APP_ENV=development
```

## Database initialization

The database tables are created automatically when the application starts using the database connection module. The default SQLite file is stored in the data folder.

## Seed data instructions

To add realistic demo data:

```powershell
python scripts/seed_database.py
```

The script creates demo tasks, calendar events, and notes if the database is empty or not previously seeded.

## How to run the CLI demo

```powershell
python main.py
```

This verifies that:

- a task can be created
- tasks can be listed
- a calendar event can be created
- events can be listed
- a note can be created
- notes can be searched

## How to run tests

```powershell
pytest
```

## Phase 2.5 — MCP Client

Phase 2.5.1 implements the MCP client foundation and lifecycle management for the existing Unified MCP Server. Phase 2.5.2 adds dynamic tool discovery through that same active MCP session.

The client establishes a real MCP protocol session over stdio, manages clean connect/disconnect behavior, and can request the Unified MCP Server's tool metadata without importing server functions or maintaining a local tool registry.

### 2.5.1 MCP Client Foundation & Lifecycle — COMPLETE

The client follows the project’s real MCP protocol flow:

MCP Client
    ↓
MCP Transport (stdio)
    ↓
Unified MCP Server
    ↓
Phase 1 services / database

The reusable client is implemented in the `mcp_client` package and exposes an async `MCPClient` with `connect()` and `close()` lifecycle methods. It uses the installed `mcp` SDK’s `stdio_client` transport and `ClientSession.initialize()` handshake to validate a real session before marking the client as connected.

Verification includes:

- client creation and initial disconnected state
- successful stdio connection to the Unified MCP Server
- protocol session initialization
- clean shutdown and repeated close safety
- idempotent repeated connect behavior
- invalid server path handling via client-level exceptions
- async context manager support

### 2.5.2 MCP Client Tool Discovery — COMPLETE

After `await client.connect()`, `await client.list_tools()` uses the MCP SDK's `ClientSession.list_tools()` operation over the already-active stdio session. It returns the SDK's native tool metadata objects, including each tool's name, description, and input schema.

The Unified MCP Server currently exposes 17 dynamically discovered tools:

- Task: `create_task`, `get_task`, `list_tasks`, `update_task`, `complete_task`, `delete_task`
- Calendar: `create_event`, `get_event`, `list_events`, `update_event`, `delete_event`
- Notes: `create_note`, `get_note`, `list_notes`, `update_note`, `delete_note`, `search_notes`

Tool invocation is intentionally not implemented yet. The next phase is Phase 2.5.3 — Generic Tool Invocation & Error Handling.

This phase intentionally does not implement:
- tool invocation
- `call_tool()` abstraction
- dynamic tool routing
- LangGraph, LLM, or frontend layers

The next phase is Phase 2.5.3 — Generic Tool Invocation & Error Handling.

## Example operations

```python
from app.services.task_service import create_task, list_tasks, complete_task
from app.services.calendar_service import create_event
from app.services.note_service import create_note, search_notes

create_task(title="Prepare presentation", priority="high")
list_tasks(status="pending")
complete_task(1)

create_event(title="Project meeting", start_time=some_datetime, end_time=another_datetime)
create_note(title="Python notes", content="Type hints and service-based design help readability.")
search_notes("python")
```

## MCP Integration

Phase 2.1 introduces a lightweight MCP layer for the Task domain. The purpose of this layer is to expose the existing task management functionality as standardized MCP tools without duplicating business logic or bypassing the Phase 1 service layer.

The architecture remains intentionally thin:

MCP Tool
    ↓
Task Service
    ↓
Database

This allows a future AI agent or MCP client to discover and call productivity tools in a structured way, while the business rules continue to live in the proven Phase 1 services.

The Task MCP server exposes these tools:

- create_task
- get_task
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
  - 2.5.3 Generic Tool Invocation & Error Handling — next
- Phase 3: LangGraph agent orchestration and AI workflows
- Phase 4: Intelligence, security, and testing enhancements
- Phase 5: Frontend integration and deployment

## Important note

Phase 1, Phase 2.1, Phase 2.2, Phase 2.3, Phase 2.4, and Phase 2.5.1–2.5.2 are complete. Tool invocation, agent orchestration, AI features, and frontend interfaces remain future work. The backend is deliberately designed so those layers can be added later without rewriting the core services.
