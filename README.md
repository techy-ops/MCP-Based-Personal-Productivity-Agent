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

## Calendar MCP expansion (Phase 2.2.1)

The calendar domain is now exposed through a dedicated FastMCP server that stays thin and delegates to the existing Phase 1 calendar business logic.

Architecture:

Calendar MCP Tool
    ↓
Calendar Service
    ↓
Database

This preserves the existing calendar validation and overlap conflict detection while exposing the tools needed by an MCP-capable client.

Current Calendar MCP tools:

- create_event
- get_event

Local startup command:

```powershell
python -m mcp_servers.calendar_server
```

The calendar MCP server intentionally does not add list, update, or delete event tools yet; those remain for the next calendar MCP sub-phase.

## Current limitations

- No AI or LLM integration
- No Notes MCP server yet
- No Streamlit frontend
- No authentication or user accounts
- No external productivity APIs
- No semantic search or embeddings
- SQLite is used locally for development and demonstration

## Future phases

- Phase 2: MCP tool servers exposing the backend service layer
- Phase 3: LangGraph agent orchestration and AI workflows
- Phase 4: Intelligence, security, and testing enhancements
- Phase 5: Frontend integration and deployment

## Important note

This project is intentionally not implementing FastMCP, LangGraph, AI features, or frontend interfaces in Phase 1. The backend is deliberately designed so those layers can be added later without rewriting the core services.
