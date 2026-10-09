"""Offline evals for the assistant: real models, fixed task data, an LLM judge.

Each case in cases.jsonl is a question asked as one of the sample users. For each case:
1. A fresh in-memory task database is filled with SEED_TASKS (due dates relative to today).
2. The assistant answers it with the real router and answer models, the real MCP
   server (in-process) and the real pgvector knowledge base.
3. Code checks the tool calls and the answer (expect_calls, forbid_tools,
   max_tool_calls, answer_excludes).
4. The judge model (JUDGE_MODEL) grades the answer against the tool results and the
   case's expectations: grounded, complete, expectations.

A case passes when the checks and all three grades pass. Results are printed and saved
to evals/results/. Needs LLM_API_KEY and the seeded knowledge base (python -m app.rag.seed).

    python -m evals.run                       # every case
    python -m evals.run --case greeting       # only some cases (repeat the flag)
    python -m evals.run --repeat 3            # run each case 3 times, to spot flaky ones
"""

import argparse
import asyncio
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Optional

import httpx
from fastapi.testclient import TestClient
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, computed_field
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import mcp_server
from app.auth import USER_HEADER
from app.config import settings
from app.database import Base, get_db
from app.main import app
from app.rag import store
from app.schemas import ToolCall, UserRead
from app.services.assistant import PromptContext, TaskAssistant
from app.services.users import SAMPLE_USERS, seed_sample_users
from evals.judge import CRITERIA, Verdict, judge

EVALS_DIR = Path(__file__).resolve().parent
CASES_FILE = EVALS_DIR / "cases.jsonl"
RESULTS_DIR = EVALS_DIR / "results"

USERS = {u.id: UserRead.model_validate(u) for u in SAMPLE_USERS}
ALICE, BOB, CAROL, DAVE, ERIN = 1, 2, 3, 4, 5


@dataclass(frozen=True)
class SeedTask:
    title: str
    assignee: int
    priority: str
    due_in_days: Optional[int]
    status: str = "todo"
    created_by: Optional[int] = None  # defaults to the assignee


# Created in this order, so task ids are 1, 2, ... and cases can refer to them by id.
# Alice sees 1-4 and 6; Carol (platform manager) 1-6; Dave 7-8; Erin (admin) all.
SEED_TASKS = [
    SeedTask("Fix login timeout bug", ALICE, "high", due_in_days=-3, status="in_progress"),  # overdue
    SeedTask("Write Q4 roadmap draft", ALICE, "medium", due_in_days=0),
    SeedTask("Update onboarding docs", ALICE, "low", due_in_days=7),
    SeedTask("Rotate API keys", ALICE, "high", due_in_days=-10, status="done"),  # completed today
    SeedTask("Migrate CI to new runners", BOB, "medium", due_in_days=-1),  # overdue
    SeedTask("Review incident postmortem", ALICE, "medium", due_in_days=1, created_by=CAROL),
    SeedTask("Prepare Acme renewal proposal", DAVE, "high", due_in_days=2),
    SeedTask("Send Q3 invoices", DAVE, "medium", due_in_days=-5),  # overdue, sales team
]


class ExpectedCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    # Only these arguments are checked; the call may have others.
    arguments: dict[str, Any] = Field(default_factory=dict)


class Case(BaseModel):
    # Unknown keys are an error, so a typo in cases.jsonl doesn't silently skip a check.
    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: int
    question: str
    # What a correct answer says, for the judge. Refer to the seeded tasks by id and title.
    expectations: str
    expect_calls: list[ExpectedCall] = Field(default_factory=list)
    forbid_tools: list[str] = Field(default_factory=list)
    max_tool_calls: Optional[int] = None
    # Text that must not appear in the answer (case-insensitive), e.g. another user's task.
    answer_excludes: list[str] = Field(default_factory=list)


class CaseResult(BaseModel):
    case_id: str
    run: int
    question: str
    answer: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    check_problems: list[str] = Field(default_factory=list)
    verdict: Optional[Verdict] = None
    error: Optional[str] = None
    seconds: float = 0.0  # the assistant's time to answer, without judging

    @computed_field
    @property
    def passed(self) -> bool:
        return (
            self.error is None
            and not self.check_problems
            and self.verdict is not None
            and all(grade.passed for grade in self.verdict.grades().values())
        )


class RunReport(BaseModel):
    today: date
    router_model: str
    answer_model: str
    judge_model: str
    results: list[CaseResult]


def load_cases(path: Path = CASES_FILE) -> list[Case]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    cases = [Case.model_validate_json(line) for line in lines]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate case ids in {path.name}")
    return cases


def check_case(case: Case, answer: str, calls: list[ToolCall]) -> list[str]:
    """The checks that need no judge. Returns the problems found (empty if none)."""
    problems: list[str] = []
    for expected in case.expect_calls:
        if not any(
            call.name == expected.tool
            and all(call.input.get(name) == value for name, value in expected.arguments.items())
            for call in calls
        ):
            problems.append(f"no {expected.tool} call with {expected.arguments}" if expected.arguments
                            else f"no {expected.tool} call")
    for call in calls:
        if call.name in case.forbid_tools:
            problems.append(f"called {call.name}, which this case forbids")
    if case.max_tool_calls is not None and len(calls) > case.max_tool_calls:
        problems.append(f"made {len(calls)} tool calls, at most {case.max_tool_calls} expected")
    for text in case.answer_excludes:
        if text.lower() in answer.lower():
            problems.append(f"answer mentions {text!r}")
    return problems


@contextmanager
def task_database() -> Iterator[None]:
    """A fresh in-memory task database with the sample users and SEED_TASKS.

    The API, and so the MCP server, uses it until the block exits. Each case gets its
    own, so a task created by one case doesn't change the data another case sees.
    """
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with Session() as session:
        seed_sample_users(session)
        app.dependency_overrides[get_db] = lambda: session
        try:
            _seed_tasks(TestClient(app))  # not entered, so the app lifespan doesn't run
            yield
        finally:
            app.dependency_overrides.pop(get_db, None)
    engine.dispose()


def _seed_tasks(api: TestClient) -> None:
    """Create SEED_TASKS through the API, so they get the same defaults and checks as real ones."""
    today = date.today()
    for task_id, task in enumerate(SEED_TASKS, start=1):
        headers = {USER_HEADER: str(task.created_by or task.assignee)}
        body: dict[str, Any] = {"title": task.title, "priority": task.priority, "assignee_id": task.assignee}
        if task.due_in_days is not None:
            body["due_date"] = (today + timedelta(days=task.due_in_days)).isoformat()
        created = api.post("/tasks", json=body, headers=headers)
        created.raise_for_status()
        assert created.json()["id"] == task_id, "seed tasks must get ids 1, 2, ... in order"
        if task.status != "todo":
            api.patch(f"/tasks/{task_id}", json={"status": task.status}, headers=headers).raise_for_status()


async def run_case(assistant: TaskAssistant, client: AsyncOpenAI, case: Case, run: int) -> CaseResult:
    result = CaseResult(case_id=case.id, run=run, question=case.question)
    user = USERS[case.user_id]
    try:
        with task_database():
            start = time.monotonic()
            response, results = await assistant.ask_with_results(case.question, user)
            result.seconds = round(time.monotonic() - start, 1)
        result.answer = response.answer
        result.tool_calls = response.tool_calls
        result.check_problems = check_case(case, response.answer, response.tool_calls)
        result.verdict = await judge(
            client,
            question=case.question,
            answer=response.answer,
            results=results,
            context=PromptContext(today=date.today().isoformat(), user=user),
            expectations=case.expectations,
        )
    except Exception as exc:  # one broken case shouldn't stop the run
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def print_result(result: CaseResult) -> None:
    status = "PASS" if result.passed else ("ERROR" if result.error else "FAIL")
    calls = ", ".join(call.name for call in result.tool_calls) or "none"
    print(f"{status:<5} {result.case_id:<30} {result.seconds:>5.1f}s  calls: {calls}")
    if result.passed:
        return
    if result.error:
        print(f"      error: {result.error}")
        return
    print(f"      answer: {result.answer!r}")
    for problem in result.check_problems:
        print(f"      check: {problem}")
    for name, grade in result.verdict.grades().items():
        if not grade.passed:
            print(f"      {name}: {grade.reasoning}")


def print_summary(results: list[CaseResult]) -> None:
    graded = [r for r in results if r.verdict is not None]
    print(f"\n{sum(r.passed for r in results)}/{len(results)} passed")
    print(f"  {'checks':<14} {sum(not r.check_problems for r in graded)}/{len(graded)}")
    for name in CRITERIA:
        print(f"  {name:<14} {sum(r.verdict.grades()[name].passed for r in graded)}/{len(graded)}")
    errors = len(results) - len(graded)
    if errors:
        print(f"  {'errors':<14} {errors}")

    by_case: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        by_case[result.case_id].append(result.passed)
    flaky = {case_id: runs for case_id, runs in by_case.items() if len(set(runs)) > 1}
    if flaky:
        print("\nFlaky (passed some runs, failed others):")
        for case_id, runs in flaky.items():
            print(f"  {case_id}: {sum(runs)}/{len(runs)}")


def save_results(results: list[CaseResult]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{datetime.now():%Y-%m-%dT%H-%M-%S}.json"
    report = RunReport(
        today=date.today(),
        router_model=settings.router_model,
        answer_model=settings.answer_model,
        judge_model=settings.judge_model,
        results=results,
    )
    path.write_text(report.model_dump_json(indent=2))
    return path



async def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the assistant evals with an LLM judge.")
    parser.add_argument("--case", action="append", default=[], help="Run only this case id (repeatable).")
    parser.add_argument("--repeat", type=int, default=1, help="Run each case this many times.")
    args = parser.parse_args(argv)

    cases = load_cases()
    if args.case:
        unknown = set(args.case) - {case.id for case in cases}
        if unknown:
            parser.error(f"unknown case ids: {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case.id in args.case]
    if not settings.llm_api_key:
        print("LLM_API_KEY is not set (see .env.example).", file=sys.stderr)
        return 2
    try:
        store.search("overdue", top_k=1)
    except Exception as exc:
        print(f"Knowledge base unavailable ({type(exc).__name__}). Start Postgres and run "
              "python -m app.rag.seed.", file=sys.stderr)
        return 2

    # The MCP server reaches the task API in-process, on the per-case database.
    mcp_server._task_api = lambda: httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://evals"
    )
    # More retries than the default 2: the free tier's rate limits are low, and the
    # SDK waits as long as the provider's retry-after header says.
    client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key, max_retries=6)

    print(f"router {settings.router_model}, answer {settings.answer_model}, judge {settings.judge_model}\n")
    results: list[CaseResult] = []
    async with TaskAssistant(mcp_server.mcp, openai_client=client) as assistant:
        for run in range(1, args.repeat + 1):
            for case in cases:
                result = await run_case(assistant, client, case, run)
                print_result(result)
                results.append(result)

    print_summary(results)
    print(f"\nSaved to {save_results(results)}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
