from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings, read from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Task Assistant"
    # Tasks and users (Postgres). Same database as the knowledge base by default.
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/task_assistant"

    # RAG knowledge base (Postgres + pgvector)
    knowledge_db_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/task_assistant"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # The MCP server reads tasks through the REST API at this address
    task_api_url: str = "http://127.0.0.1:8000"
    # User the MCP server acts as when a client sends no user in the request metadata,
    # e.g. the MCP Inspector. The assistant always sends one. Leave unset outside local dev.
    mcp_default_user_id: Optional[int] = None
    # Roles allowed to create tasks through the MCP create_task tool, i.e. via the
    # assistant. Stricter than the REST API, which lets employees create their own tasks.
    mcp_create_task_roles: str = "manager,admin"

    # Any provider that speaks the OpenAI Responses API. Defaults to Groq's free tier;
    # for OpenAI itself, set LLM_BASE_URL=https://api.openai.com/v1.
    llm_base_url: str = "https://api.groq.com/openai/v1"
    # Read from .env here and passed to the client (the SDK alone only reads os.environ).
    llm_api_key: Optional[str] = None
    # Small, fast model that picks the tools; the main model writes the answer.
    router_model: str = "openai/gpt-oss-20b"
    answer_model: str = "openai/gpt-oss-120b"
    # Grades the assistant's answers in the evals (evals/run.py). A different model
    # family from answer_model, since a model tends to rate its own writing higher.
    judge_model: str = "qwen/qwen3.8-27b"

    # Online evals: score live /assistant traffic in the background (app/services/online_evals.py).
    online_eval_enabled: bool = True
    # Share of answers the judge grades. Code checks run on every request. Each judged
    # answer is one judge_model call, so mind the provider's daily limits.
    online_eval_judge_sample_rate: float = 0.2
    # A knowledge-base search whose best hit scores lower is flagged: the FAQ likely
    # has no good answer, i.e. an entry worth adding.
    online_eval_min_retrieval_score: float = 0.6
    online_eval_slow_ms: int = 20_000
    # Traces waiting to be evaluated. When full, new ones are dropped (and counted),
    # so a slow judge never holds up requests.
    online_eval_queue_size: int = 100


settings = Settings()
