# Task Assistant

A task-management backend for learning the Python AI stack, built one stage at a time.

| Stage | What to build | Tech | Status |
|---|---|---|---|
| 1 | CRUD APIs for tasks | FastAPI, Pydantic, SQLAlchemy, Postgres | ✅ done |
| 2 | Manage tasks with natural language | Groq (OpenAI-compatible API), MCP, RAG (pgvector) | 🟡 read-only `/assistant` |
| 3 | Fetch tasks and calendar availability concurrently | asyncio, HTTPX, timeouts | ⬜ |
| 4 | Propose changes, then ask for approval before applying them | LangGraph, state, human approval | ⬜ |
| 5 | Test ambiguous requests, bad tool arguments, API failures | pytest, evals, tracing | ⬜ |

## How `/assistant` works

```
POST /assistant {"question": "..."}
        │
        ▼
1. Plan     gpt-oss-20b returns a JSON plan: every MCP tool call needed  ◀─┐ again only if a
        │                                                                 │ call needs another's
        ▼                                                                 │ result (max 3 rounds)
2. Execute  TaskAssistant runs them in parallel ──MCP over stdio──▶ app/mcp_server.py
                                                         ├─ list_tasks ─────────────┐ HTTP GET /tasks
                                                         ├─ get_task ───────────────┘ (the Stage 1 API, Postgres)
                                                         └─ search_knowledge_base ──▶ Postgres + pgvector (FAQ)
        │   results go back to step 1 only when the plan asked for them ──┘
        ▼
3. Compose  gpt-oss-120b writes the answer from the question + tool results
```

- **MCP server** (`app/mcp_server.py`): `list_tasks` and `get_task` are thin wrappers over the existing GET endpoints. `list_tasks` returns at most 5 tasks, the most recently created, plus `more_tasks_exist`, so a large task list can't flood the model's context; the answer then says the list is partial. `search_knowledge_base` embeds the question and runs a cosine-similarity search in pgvector.
- **RAG store** (`app/rag/`): the 40 FAQ entries in `data/faq.json` are embedded locally with fastembed (`BAAI/bge-small-en-v1.5`, 384 dimensions) and stored in the `faq_entries` table.
- **Assistant** (`app/services/assistant.py`): on startup, FastAPI launches the MCP server as a subprocess and keeps one connection to it open. For each question, a small model (`ROUTER_MODEL`) returns a plan as structured JSON output: every tool call needed, for all parts of the question at once. The app runs them in parallel. (gpt-oss on Groq makes only one function call per turn, so a JSON plan is what lets a multi-part question take one router call.) Only when a call needs another's result, such as `create_task` for someone whose id comes from `list_users`, does the plan set `needs_results`, and the router plans again with the results, for up to 3 rounds. Writes in such a plan wait for the next round, so they never run with a guessed argument. Then the main model (`ANSWER_MODEL`) writes the answer. The composer has no tools, so most questions cost one small-model call plus one main-model call. The response includes the answer and the list of tool calls, so you can see what happened.
- **Grounded task values**: the answer model never writes task values itself. It writes placeholders like `{{task:1.status}}`, and the app fills them in from the `list_tasks`/`get_task` output, so status, priority, dates and titles always match the API exactly. A placeholder for a task that wasn't fetched, or a field that doesn't exist, gets one retry with feedback; otherwise the assistant returns an error message instead of an unverified answer.

## Setup

You need **Python 3.10+** (the `mcp` SDK requires it) and **Postgres with pgvector**.

```bash
# 1. Python 3.12 + virtualenv
brew install python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env              # then set LLM_API_KEY (a free Groq key) in .env

# 2. Postgres + pgvector (tasks, users and the knowledge base). Pick one:
docker compose up -d              # a) Docker: matches the default DATABASE_URL and KNOWLEDGE_DB_URL
#   or
brew install pgvector             # b) Homebrew Postgres
brew services start postgresql@18
createdb task_assistant           #    then set DATABASE_URL and KNOWLEDGE_DB_URL to
                                  #    postgresql+psycopg://$USER@localhost:5432/task_assistant

# 3. Load the FAQ into pgvector (downloads the ~70 MB embedding model the first time)
python -m app.rag.seed
```

## Run

```bash
uvicorn app.main:app --reload
```

Interactive API docs are at http://127.0.0.1:8000/docs.

```bash
curl -X POST localhost:8000/assistant -H 'content-type: application/json' -H 'X-User-Id: 1' \
  -d '{"question": "When is a task considered overdue, and do I have any overdue tasks?"}'
```

That question should trigger both `search_knowledge_base` (the rule) and `list_tasks(overdue=true)` (your data).

Every request needs an `X-User-Id` header (see [Users and permissions](#users-and-permissions)). In the interactive docs, fill it in on each endpoint.

You can also run the MCP server on its own and connect another MCP client to it, such as Claude Code or the MCP Inspector:

```bash
MCP_DEFAULT_USER_ID=1 python -m app.mcp_server    # stdio transport, acting as Alice
```

Task tools need a user. The assistant sends one in each request's metadata; other clients don't, so `MCP_DEFAULT_USER_ID` sets who they act as.

## Users and permissions

Authentication is dev-grade: the caller sends `X-User-Id: <id>` and the API trusts it, so anyone can claim any id. To use real auth (JWT, OAuth), replace `get_current_user` in `app/auth.py`; everything else only depends on the resulting user.

Startup seeds five sample users:

| Id | Name | Role | Team |
|---|---|---|---|
| 1 | Alice | employee | platform |
| 2 | Bob | employee | platform |
| 3 | Carol | manager | platform |
| 4 | Dave | employee | sales |
| 5 | Erin | admin | it |

Rules (`app/services/permissions.py`):

- **See and change:** employees, the tasks they created or are assigned to; managers, also every task assigned to their team; admins, everything. A task you can't see returns 404, so ids don't leak.
- **Create:** employees only for themselves, managers for anyone on their team, admins for anyone. A task is assigned to its creator unless `assignee_id` is given.
- **Delete:** the team manager or an admin, or the creator while the task is still `todo`.
- **Bulk reschedule:** managers and admins, limited to the tasks they can see.
- **Create through the assistant (MCP `create_task`):** stricter than the API. Only the roles in `MCP_CREATE_TASK_ROLES` (default `manager,admin`) may use it, so employees create their own tasks through the REST API, not the assistant. The MCP server looks up the caller's role via `GET /users/me` and refuses before anything is created. The assistant also doesn't offer the tool to other roles, so the model doesn't try it and the answer can explain why.

The assistant can't get around these rules. `/assistant` knows who is asking, and every MCP tool call carries that user's id in the request metadata (`_meta`), outside the tool arguments, so the LLM can't see or change it. The MCP server forwards it to the API, which checks permissions and returns 403 when an action isn't allowed. The assistant then explains the denial. It also never runs the same `create_task` call twice for one question.

## Evals

Two kinds of evals check the quality of the assistant's answers. See [evals/README.md](evals/README.md) for details.

- **Offline evals** (`python -m evals.run`): fixed questions with known correct answers, run before changing a prompt, model or tool, to catch regressions.
- **Online evals:** every live `/assistant` request is scored in the background without slowing it down. Code checks run on every request and an LLM judge on a sample. They find problems you didn't anticipate, like knowledge-base gaps. Admins read the results at `GET /evals/online/summary` and `/evals/online/traces`.

## Test

```bash
pytest -v
```

The tests need Postgres but no API key. They use their own database on the same server, `DATABASE_URL`'s name plus `_test` (created if missing; override with `TEST_DATABASE_URL`), and empty its tables before each test, so your data is never touched. The MCP tests connect a real MCP client to the server in-process, and route its HTTP calls into the FastAPI app. The knowledge-base search and the LLM are replaced with fakes.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/assistant` | Ask a question in plain English; returns `answer` + `tool_calls` |
| `GET` | `/users`, `/users/me` | List users; the current user |
| `POST` | `/tasks` | Create a task (`assignee_id` optional) |
| `GET` | `/tasks` | List tasks. Filters: `status`, `overdue`, `due_on`, `completed_since`; `newest_first`, `limit` |
| `GET` | `/tasks/{id}` | Get one task |
| `PATCH` | `/tasks/{id}` | Partially update a task (setting `status=done` stamps `completed_at`) |
| `DELETE` | `/tasks/{id}` | Delete a task |
| `POST` | `/tasks/reschedule` | Move unfinished tasks from one date to another |

## Layout

```
app/
  main.py               FastAPI app; startup sets up the DB and connects the assistant to MCP
  bootstrap.py          create tables, add new columns to an existing DB, seed sample users
  auth.py               X-User-Id authentication (CurrentUser dependency)
  config.py             settings from env / .env (pydantic-settings)
  database.py           SQLAlchemy engine + session for tasks and users (Postgres)
  models.py             Task and User ORM models
  schemas.py            Pydantic request/response models
  mcp_server.py         MCP server: list_tasks, get_task, create_task, list_users, search_knowledge_base
  rag/
    embeddings.py       fastembed text embeddings
    store.py            pgvector table, upsert, and similarity search
    seed.py             loads data/faq.json into pgvector
  services/
    tasks.py            task business logic (no HTTP)
    permissions.py      who may see, create, change and delete which tasks
    online_evals.py     background scoring of live /assistant requests
    users.py            sample users and user queries
    assistant.py        plan (small model) → parallel MCP tool calls → compose (main model)
  routers/
    tasks.py            /tasks endpoints
    users.py            /users endpoints
    evals.py            /evals/online endpoints (admin)
    assistant.py        /assistant endpoint
data/faq.json           knowledge-base content for RAG
evals/                  offline evals (run.py, cases.jsonl) and the LLM judge; see evals/README.md
tests/                  pytest
```

## Notes and next steps

- Tasks, users and the knowledge base all live in Postgres, in the same database by default (`DATABASE_URL` and `KNOWLEDGE_DB_URL` can point to different ones).
- The FAQ mentions `blocked` status and comments, which the task model doesn't have yet. Adding them is a good exercise.
- The assistant can read and create tasks. Next: expose `update_task` and `reschedule_unfinished` as MCP tools (the API already enforces their permissions). Stage 4 will put an approval step in front of writes.
- Both LLM calls use the OpenAI SDK against any Responses-API-compatible provider: Groq by default (free tier, keys at https://console.groq.com/keys). Change provider and models with `LLM_BASE_URL`, `ROUTER_MODEL` and `ANSWER_MODEL`.
