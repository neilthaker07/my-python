"""Answers natural-language questions in three steps, using the tools from our MCP server.

1. Plan:    a small, fast model reads the question and returns a JSON plan: every MCP
            tool call needed to answer it, for all parts of the question at once.
2. Execute: we run the planned calls in parallel (task API or the pgvector knowledge base).
            Only if a call needs another's result first (e.g. list_users before
            create_task) does the model see the results and plan again, up to
            MAX_PLANNING_ROUNDS times.
3. Compose: the main model writes the answer from the question and the tool results.
            It writes placeholders such as {{task:1.status}} instead of task values,
            and we fill them in from the MCP data, so facts can't be altered by the LLM.
"""

import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional, Union

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.server import MCPServer
from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel, Field, ValidationError

from app.auth import USER_META_KEY
from app.config import settings
from app.schemas import AssistantResponse, ToolCall, UserRead
from app.services.permissions import can_create_via_mcp

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFUSAL_ANSWER = "Sorry, I can't help with that request."
UNVERIFIED_ANSWER = "Sorry, I couldn't produce an answer that matches the task data. Please try again."

# Task data comes from these tools; their outputs are TaskRead JSON (one object or a list).
TASK_TOOLS = {"list_tasks", "get_task", "create_task"}
TASK_FIELDS = (
    "title", "description", "status", "priority", "due_date", "completed_at",
    "created_at", "updated_at", "created_by_id", "assignee_id",
)
# Tools that change data. The same write is never run twice for one question.
WRITE_TOOLS = {"create_task"}
PLACEHOLDER = re.compile(r"\{\{task:(\d+)\.(\w+)\}\}")

# Each round is one small-model call that plans tool calls, then runs them in parallel.
# Most questions need one round; more only when a call needs another call's result.
MAX_PLANNING_ROUNDS = 3

# Groq's error codes when the router's output isn't a usable plan: it doesn't match the
# JSON schema, or gpt-oss sent the plan as a call to a tool named "plan".
PLAN_REJECTED = {"json_validate_failed", "tool_use_failed"}

ROUTER_PROMPT = """\
You plan the data lookups for a task-management assistant. Reply with a plan: \
every tool call needed to answer the question. The calls run in parallel, and \
another model writes the answer from their results.

Tools:
{tools}

- Questions about the user's own tasks (what's due, overdue, done, a specific \
task) → list_tasks or get_task.
- Questions about policies, rules or how things work (statuses, priorities, \
permissions, overdue rules, what to do when blocked) → search_knowledge_base.
- "Overdue" is a filter (list_tasks with overdue=true), not a status.
- Only when the user explicitly asks to create a task → create_task. If it's for \
someone else and you don't know their user id, plan only list_users (not \
create_task) and set needs_results to true; you'll create the task in the next \
plan. Create each task once; if create_task returns an error (e.g. not allowed), \
don't retry with other arguments.
- Questions often have several parts. Plan a call for every part in this one \
reply: "what's the leave policy, and do I have overdue tasks?" needs both \
search_knowledge_base and list_tasks.
- needs_results is almost always false: the other model reads the results, not \
you. Set it to true only when a call you can't plan yet needs a value (such as a \
user id) that you'll only learn from a result.
- If no tool is relevant (e.g. a greeting), plan no calls. Don't answer the \
question yourself.

{user}
Today's date is {today}."""

COMPOSER_PROMPT = """\
You are a task-management assistant for an enterprise team. Answer the user's \
question using only the tool results provided with it. If they don't contain \
the answer, say so instead of guessing. Keep answers short.

Never write a task's field values yourself. Wherever the answer states one, write \
a placeholder {{{{task:<id>.<field>}}}} and the app fills in the exact value. \
Fields: {fields}. Refer to tasks by id as plain text.
Example: Task 1 "{{{{task:1.title}}}}" is {{{{task:1.status}}}} with \
{{{{task:1.priority}}}} priority, due {{{{task:1.due_date}}}}.

If a tool returned an error, such as a permission denial, explain it plainly. \
Never claim a task was created unless a create_task result shows it.

list_tasks returns only the most recently created matching tasks. If its result \
has "more_tasks_exist": true, say these are the newest ones and that more match; \
don't present them as the full list or count them as the total.

{user}
Today's date is {today}."""


def stdio_server_params() -> StdioServerParameters:
    """Launch app/mcp_server.py as a subprocess that talks MCP over stdin/stdout."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        cwd=PROJECT_ROOT,
        env=dict(os.environ),  # pass through config such as KNOWLEDGE_DB_URL
    )


class TaskAssistant:
    """Holds one MCP connection open for the app's lifetime (use as `async with`)."""

    def __init__(
        self,
        mcp_server: Union[MCPServer, StdioServerParameters],
        openai_client: Optional[AsyncOpenAI] = None,
    ) -> None:
        self._mcp = Client(mcp_server)
        # Raises at startup if no key is set, so POST /assistant returns 503 instead of 500.
        self._openai = openai_client or AsyncOpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key
        )

    async def __aenter__(self) -> "TaskAssistant":
        await self._mcp.__aenter__()
        mcp_tools = (await self._mcp.session.list_tools()).tools
        # MCP tool definitions map directly onto OpenAI function tools.
        self._tools = [
            {"type": "function", "name": t.name, "description": t.description or "", "parameters": t.input_schema}
            for t in mcp_tools
        ]
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._mcp.__aexit__(*exc_info)

    async def ask(self, question: str, user: UserRead) -> AssistantResponse:
        """Answer `question` for `user`. Tools run with that user's permissions."""
        response, _ = await self.ask_with_results(question, user)
        return response

    async def ask_with_results(
        self, question: str, user: UserRead
    ) -> tuple[AssistantResponse, list[tuple[ToolCall, str]]]:
        """Like ask(), but also returns each tool call with its output: the data the
        answer is based on. The evals (evals/run.py) grade the answer against it."""
        today = date.today().isoformat()
        context = PromptContext(today=today, user=user)

        results = await self._plan_and_execute(question, context)
        if results is None:
            return AssistantResponse(answer=REFUSAL_ANSWER, tool_calls=[]), []

        tool_calls = [call for call, _ in results]
        tasks = _fetched_tasks(results)

        # Step 3, plus a second try if the model referenced data that wasn't fetched.
        problems: list[str] = []
        for _ in range(2):
            template = await self._compose(question, context, results, problems)
            if template is None:
                return AssistantResponse(answer=REFUSAL_ANSWER, tool_calls=tool_calls), results
            answer, problems = fill_placeholders(template, tasks)
            if not problems:
                return AssistantResponse(answer=answer or REFUSAL_ANSWER, tool_calls=tool_calls), results
        return AssistantResponse(answer=UNVERIFIED_ANSWER, tool_calls=tool_calls), results

    async def _plan_and_execute(
        self, question: str, context: "PromptContext"
    ) -> Optional[list[tuple[ToolCall, str]]]:
        """Steps 1-2: the small model plans tool calls and we run them all in parallel.

        Plans again only when the model asked to see results first (chained calls).
        Returns each tool call with its output, or None if the model refused.
        """
        results: list[tuple[ToolCall, str]] = []

        for _ in range(MAX_PLANNING_ROUNDS):
            plan = await self._plan(question, context, results)
            if plan is None:
                return None
            calls = _runnable_calls(plan, done=[call for call, _ in results])
            if calls:
                outputs = await asyncio.gather(*(self._call_tool(call, context.user.id) for call in calls))
                results += zip(calls, outputs)
            if not (calls and plan.needs_results):
                break

        return results

    async def _plan(
        self, question: str, context: "PromptContext", results: list[tuple[ToolCall, str]]
    ) -> Optional["Plan"]:
        """Step 1: one small-model call that returns a plan of tool calls (None if it refused)."""
        tools = self._tools_for(context.user)
        request = dict(
            model=settings.router_model,
            instructions=ROUTER_PROMPT.format(
                tools=_describe_tools(tools), today=context.today, user=context.describe_user()
            ),
            input=_planner_input(question, results),
            # Not strict: strict mode needs every property required, and MCP tool
            # arguments have optional ones. The MCP server validates the arguments anyway.
            text={"format": {"type": "json_schema", "name": "plan", "schema": _plan_schema(tools), "strict": False}},
            reasoning={"effort": "low"},
            max_output_tokens=1024,
        )
        try:
            response = await self._openai.responses.create(**request)
        except BadRequestError as exc:
            # e.g. status="overdue", which the schema doesn't allow. Small models
            # usually get it right on a retry.
            if exc.code not in PLAN_REJECTED:
                raise
            response = await self._openai.responses.create(**request)
        if _refused(response):
            return None
        try:
            return Plan.model_validate_json(response.output_text)
        except ValidationError:
            # Rare: the provider checks the schema. Composing from no data is safe, since
            # the composer says when the results don't contain the answer.
            logger.warning("Router returned an invalid plan: %r", response.output_text)
            return Plan()

    def _tools_for(self, user: UserRead) -> list[dict]:
        """The tools this user may use. Hiding create_task just keeps the model from trying;
        the MCP server enforces the rule either way."""
        if can_create_via_mcp(user.role):
            return self._tools
        return [tool for tool in self._tools if tool["name"] != "create_task"]

    async def _call_tool(self, call: ToolCall, user_id: int) -> str:
        """Step 2: run one MCP tool as `user_id` and return its text output."""
        # The user id goes in the request metadata, not the arguments: the model chose
        # the arguments, but whose permissions apply is decided here, in code.
        result = await self._mcp.session.call_tool(call.name, call.input, meta={USER_META_KEY: user_id})
        text = "\n".join(block.text for block in result.content if block.type == "text")
        return f"ERROR: {text}" if result.is_error else text

    async def _compose(
        self, question: str, context: "PromptContext", results: list[tuple[ToolCall, str]], problems: list[str]
    ) -> Optional[str]:
        """Step 3: the main model writes the answer, with placeholders for task values.

        Returns the unfilled answer, or None if the model refused.
        """
        content = (
            f"<question>\n{question}\n</question>\n\n"
            f"<tool_results>\n{_render_results(results) or '(no tools were called)'}\n</tool_results>"
        )
        if problems:
            content += (
                "\n\nYour previous answer used placeholders that don't match the fetched tasks: "
                + "; ".join(problems)
                + ". Use only tasks and fields from the tool results."
            )

        response = await self._openai.responses.create(
            model=settings.answer_model,
            instructions=COMPOSER_PROMPT.format(
                today=context.today, user=context.describe_user(), fields=", ".join(TASK_FIELDS)
            ),
            input=content,
            reasoning={"effort": "low"},
            max_output_tokens=16000,
        )
        if _refused(response):
            return None
        return response.output_text


@dataclass(frozen=True)
class PromptContext:
    today: str
    user: UserRead

    def describe_user(self) -> str:
        u = self.user
        text = f"The current user is {u.name} (user id {u.id}, role {u.role.value}, team {u.team})."
        if not can_create_via_mcp(u.role):
            text += " Their role can't create tasks through the assistant; if asked, say so."
        return text


class PlannedCall(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    """The router's output: tool calls to run now, and whether to plan again after them."""

    calls: list[PlannedCall] = Field(default_factory=list)
    needs_results: bool = False


def _plan_schema(tools: list[dict]) -> dict:
    """JSON schema for a Plan, where each call's arguments follow that tool's own schema."""
    defs: dict = {}
    branches = []
    for tool in tools:
        arguments = dict(tool["parameters"])
        # Arguments can $ref shared types ("#/$defs/TaskStatus"). Refs resolve from the
        # schema's root, so the definitions move up to the plan's root.
        defs.update(arguments.pop("$defs", {}))
        branches.append({
            "type": "object",
            "properties": {"tool": {"const": tool["name"]}, "arguments": arguments},
            "required": ["tool", "arguments"],
        })
    schema = {
        "type": "object",
        "properties": {
            "calls": {"type": "array", "items": {"anyOf": branches}},
            "needs_results": {
                "type": "boolean",
                "description": "True only if a call still to be planned needs a value from these calls' results.",
            },
        },
        "required": ["calls", "needs_results"],
    }
    if defs:
        schema["$defs"] = defs
    return schema


def _describe_tools(tools: list[dict]) -> str:
    """Tool names and descriptions for the router prompt (a plan has no function tools)."""
    return "\n".join(f"- {tool['name']}: {tool['description']}" for tool in tools)


def _planner_input(question: str, results: list[tuple[ToolCall, str]]) -> str:
    if not results:
        return question
    return (
        f"<question>\n{question}\n</question>\n\n"
        f"<tool_results>\n{_render_results(results)}\n</tool_results>\n\n"
        "Plan only the calls still needed. Don't repeat calls that already have results."
    )


def _render_results(results: list[tuple[ToolCall, str]]) -> str:
    return "\n".join(
        f"<result tool={json.dumps(call.name)} input={json.dumps(json.dumps(call.input))}>\n"
        f"{output}\n</result>"
        for call, output in results
    )


def _runnable_calls(plan: Plan, done: list[ToolCall]) -> list[ToolCall]:
    """The plan's calls that may run now.

    - A write already run in an earlier round, or planned twice, runs only once,
      so a question can't create duplicates.
    - If the plan waits on results, its writes wait too and only its reads run: a write
      planned next to the read it depends on (create_task beside list_users) would
      otherwise run with a guessed argument. The next plan, made with the results, has it.
    """
    calls: list[ToolCall] = []
    for planned in plan.calls:
        # The model writes null for arguments it isn't setting; every tool's default means
        # the same, and some arguments (e.g. overdue: bool) would reject null.
        arguments = {name: value for name, value in planned.arguments.items() if value is not None}
        call = ToolCall(name=planned.tool, input=arguments)
        if call.name in WRITE_TOOLS and (call in done or call in calls):
            continue
        calls.append(call)
    reads = [call for call in calls if call.name not in WRITE_TOOLS]
    return reads if plan.needs_results and reads else calls


def _fetched_tasks(results: list[tuple[ToolCall, str]]) -> dict[int, dict]:
    """Task records from successful list_tasks / get_task / create_task calls, keyed by task id."""
    tasks: dict[int, dict] = {}
    for call, output in results:
        if call.name not in TASK_TOOLS or output.startswith("ERROR:"):
            continue
        data = json.loads(output)
        # list_tasks returns {"tasks": [...], "more_tasks_exist": ...}; the others one task.
        for task in data["tasks"] if "tasks" in data else [data]:
            tasks[task["id"]] = task
    return tasks


def fill_placeholders(template: str, tasks: dict[int, dict]) -> tuple[str, list[str]]:
    """Replace {{task:<id>.<field>}} with the exact value from the fetched task data.

    Returns the filled answer and a list of problems (placeholders naming a task that
    wasn't fetched or a field that doesn't exist). The answer is only usable if there are none.
    """
    problems: list[str] = []

    def value(match: re.Match) -> str:
        task_id, field = int(match.group(1)), match.group(2)
        if task_id not in tasks:
            problems.append(f"{match.group(0)}: task {task_id} was not fetched")
            return match.group(0)
        if field not in TASK_FIELDS:
            problems.append(f"{match.group(0)}: unknown field {field!r}")
            return match.group(0)
        raw = tasks[task_id][field]
        return "none" if raw is None else str(raw)

    return PLACEHOLDER.sub(value, template), problems


def _refused(response) -> bool:
    """True if the model declined the request (a refusal block instead of an answer)."""
    return any(
        part.type == "refusal"
        for item in response.output
        if item.type == "message"
        for part in item.content
    )
