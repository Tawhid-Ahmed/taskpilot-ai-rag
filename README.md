# TaskPilot AI — RAG-enabled Agent for Task Management

A demo project that combines Retrieval-Augmented Generation (RAG), a small agent toolset, and conversational state to manage tasks via a Spring Boot backend. The system routes user queries either to a fast backend fetch path (for simple retrievals) or to an LLM-driven agent (for CRUD and complex workflows). Designed to showcase a production-ready architecture for interviewers and visitors.

## Highlights

- Hybrid architecture: RAG (FAISS + sentence-transformers) + agent tools for CRUD via authenticated Spring Boot APIs.
- Smart decision layer: an LLM-backed classifier (with a conservative heuristic fallback) decides whether a query should call tools or be answered from memory/vector search.
- Tooling: agent-accessible tools wrap the Spring Boot endpoints (create/get/update/delete/get-by-id) with idempotency and friendly confirmations.
- Conversation memory: chat messages persist in the DB and seed the agent/RAG context.
- Developer-friendly: small codebase, easy-to-run FastAPI entrypoint, and clear extension points.

## Repo structure (important files)

- `main.py` — FastAPI app and `/chat` endpoint.
- `app/api/task_tools.py` — Tool wrappers around Spring Boot APIs: `create_task`, `get_tasks`, `update_task`, `delete_task`, `get_task_by_id`.
- `app/rag/graph.py` — Core workflow graph: input handling, decision node (LLM "brain"), backend-fetch node, agent node (RAG + tools), output handler.
- `app/rag/vector_store.py` — In-memory FAISS indexes + sentence-transformers embeddings (RAG).
- `app/memory` — Persistence helpers for conversation memory (`app/memory/manager.py`, `app/memory/models.py`).
- `app/db.py` — SQLAlchemy DB setup.
- `app/schemas.py` — Pydantic request models (e.g., `ChatRequest`).
- `requirements.txt` — Python dependencies.

> Tip: Browse the code in `app/rag/graph.py` and `app/api/task_tools.py` to see the decision flow and how tools call your Spring Boot API.

## Architecture (short)

1. HTTP /chat request -> `main.py` extracts Authorization header and sets runtime token.
2. A `StateGraph` pipeline runs:
   - `input_handler`: load last N messages from DB memory.
   - `decision_node`: classify intent (LLM classifier -> 'agent' or 'backend').
   - `backend_fetch_node`: quick backend calls and compact natural-language formatting (used for lists, counts, due dates).
   - `agent_node`: initialize agent with LLM + tools + RAG context and run the chain for complex CRUD or reasoning.
   - `output_handler`: saves the user and AI messages into DB memory and returns response.
3. Response returned to client.

## Key design decisions

- LLM classifier as the "brain": avoids brittle hard-coded trigger lists and can handle nuanced phrasing.
- Conservative fallback heuristic: ensures safety and predictable behavior if LLM fails or is ambiguous.
- Tools return `return_direct=True` and human-friendly confirmations so the agent can finish immediately after an API action.
- `create_task` includes an idempotency check (match on title + due date) to avoid accidental duplicates.
- RAG uses `sentence-transformers` (`all-MiniLM-L6-v2`) + FAISS for fast similarity search.

## Tech stack

This project uses the following technologies and libraries:

- Python 3.10+
- FastAPI + Uvicorn (web server)
- LangChain / langchain-core and LangGraph for agent orchestration
- Ollama (via `langchain_ollama`) for LLM calls (configurable with `OLLAMA_MODEL`)
- sentence-transformers (`all-MiniLM-L6-v2`) for embeddings
- FAISS for vector similarity search
- httpx for HTTP tooling to Spring Boot APIs
- SQLAlchemy + PostgreSQL (psycopg2) for conversation memory
- dotenv for environment configuration
- pytest for tests (optional)

The architecture is modular so you can swap the LLM provider, vector store, or DB for alternatives in production.

## Environment & configuration

Required env variables (example names used in code):

- `SPRINGBOOT_BASE_URL` — Base URL for your Spring Boot API (default: `http://localhost:8080/api`).
- `SPRINGBOOT_JWT_TOKEN` — Optional static token for backend calls. The `/chat` endpoint also accepts Authorization header and sets the token during request handling.
- `OLLAMA_MODEL` — LLM model identifier used by `ChatOllama` (default seen in code: `llama3.1:latest`).
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` — Postgres DB connection for memory/persistence.

Put these in a `.env` file or export them in your shell.

## Quick start (Windows PowerShell)

1. Create and activate a Python environment (recommended Python 3.10+):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Add environment variables (example using PowerShell):

```powershell
$env:SPRINGBOOT_BASE_URL = 'http://localhost:8080/api'
$env:OLLAMA_MODEL = 'llama3.1:latest'
# DB vars as needed
```

3. Run the app:

```powershell
uvicorn main:app --reload --port 8001
```

4. Example chat request (include Authorization header if your Spring Boot API uses JWT):

```powershell
curl -X POST http://localhost:8001/chat -H "Content-Type: application/json" -H "Authorization: Bearer <token>" -d '{"user_id":1,"session_id":"session-123","message":"What tasks are due this week?"}'
```

Example responses:
- Retrieval: `{"response": "Tasks due this week:\nTask A is due on 2025-09-10"}`
- Count: `{"response": "You have 3 tasks with status NEW."}`
- Create: `{"response": "Created task 'Finish AI project' (id: 42) due 2025-09-15."}`

## Example flows

- Natural retrieval (e.g. "Which tasks are pending?")
  - Classifier -> backend
  - `get_tasks()` -> filter/present -> short answer

- Create or update (e.g. "Create a task called X by Friday")
  - Classifier -> agent
  - Agent uses `create_task` (idempotent) -> receives confirmation string -> finishes and returns chat response

- Complex reasoning + RAG (e.g. "Which tasks relate to the client meeting last month?")
  - Classifier -> agent
  - Agent uses `rag_query` to get relevant tasks into prompt, may ask follow-ups, and can call tools if needed.

## Internals — important snippets

- Decision: LLM-backed classifier + fallback heuristic. See `app/rag/graph.py::decision_node`.
- Tool wrappers: `app/api/task_tools.py` — defensive parsing, idempotency, token header handling via `set_auth_token()`.
- RAG: `app/rag/vector_store.py` builds FAISS indexes per user and returns the top-k similar tasks.

## Running tests & local verification

- There's a small test file `test_ollama.py` (adjust it to your environment if you have an Ollama server or local LLM).
- For integration tests, mock the Spring Boot API endpoints or point `SPRINGBOOT_BASE_URL` to a test instance.

## Troubleshooting

- Stuck requests: check `main.py` flow: token is set from incoming Authorization header and cleared after `app_graph.invoke`.
- LLM hangs or slow: classifier uses `temperature=0` and a short prompt; if you still see latency, switch to a local lightweight classifier or cache decisions.
- Duplicate create calls: `create_task` has an idempotency check but relies on backend GET working; ensure the backend returns tasks as expected.
- DB/Memory issues: check `app/db.py` env var configuration and that Postgres is reachable.

## Security & production notes

- Never store global tokens in long-lived globals for multi-tenant apps; current demo sets a transient token per request in `task_tools.set_auth_token()` and clears it after invocation.
- Validate and sanitize all user inputs; agents can be prompted to synthesize JSON and may produce malformed payloads. The tool wrappers attempt to parse safely but consider stricter validation in production.
- Add rate limits and auth checks on `/chat` to prevent abuse.

## Future improvements (good interview talking points)

- Replace the classifier LLM call with a lightweight local classifier for low-latency routing.
- Add more structured prompts and tool response schemas to reduce parsing ambiguity.
- Persist FAISS indexes to disk or use a managed vector DB for scaling across restarts.
- Add a small test harness with mocked backend for CI integration tests.
- Add metrics/observability for branch decisions, tool usage, and latency.

