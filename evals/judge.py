"""LLM as a judge: a second model grades an assistant answer against the data it was based on.

The judge sees what the assistant saw (the question, the user, today's date and the
tool results), the final answer, and, for offline evals, the case author's expectations
for a correct answer. It grades each criterion pass/fail, with its reasoning first.

Online evals (app/services/online_evals.py) grade live traffic, which has no
expectations, so they get only the grounded and complete criteria.
"""

from typing import Optional, Union

from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.schemas import ToolCall
from app.services.assistant import PromptContext, _render_results

JUDGE_PROMPT = """\
You grade answers written by a task-management assistant. You get the user's \
question, the tool results the assistant had (task data from the API and \
knowledge-base entries){answer_and_expectations}.

Judge only against the tool results{expectations_basis}, not your own idea of \
how a task system should work. Task field values in the answer (titles, statuses, \
dates) were copied from the tool results by code, so they are exact; check that \
the right tasks were picked and that the claims about them hold. A tool result \
starting with ERROR is a failed call, e.g. a permission denial.

Grade each criterion separately. Write your reasoning first, then decide.
- grounded: every factual claim in the answer (tasks, counts, dates, statuses, \
policies, names, actions taken) is supported by the tool results. Greetings, \
offers to help, and saying the data isn't available are fine.
- complete: the answer addresses every part of the question, or says plainly \
that the data to answer it isn't available.
{expectations_criterion}
Be strict: one unsupported claim fails grounded{expectations_strict}.

{user}
Today's date is {today}."""


class Grade(BaseModel):
    reasoning: str
    passed: bool


class LiveVerdict(BaseModel):
    """Grades for live traffic, which has no expectations to check against."""

    grounded: Grade
    complete: Grade

    def grades(self) -> dict[str, Grade]:
        return {name: getattr(self, name) for name in type(self).model_fields}


class Verdict(LiveVerdict):
    """Grades for an offline eval case."""

    expectations: Grade


CRITERIA = tuple(Verdict.model_fields)


class JudgeError(Exception):
    """The judge model didn't return a usable verdict."""


async def judge(
    client: AsyncOpenAI,
    *,
    question: str,
    answer: str,
    results: list[tuple[ToolCall, str]],
    context: PromptContext,
    expectations: Optional[str] = None,
) -> Union[Verdict, LiveVerdict]:
    """Grade an answer. With `expectations` (offline evals) returns a Verdict; without
    (online evals on live traffic) a LiveVerdict, which has no expectations grade."""
    offline = expectations is not None
    content = (
        f"<question>\n{question}\n</question>\n\n"
        f"<tool_results>\n{_render_results(results) or '(no tools were called)'}\n</tool_results>\n\n"
        f"<answer>\n{answer}\n</answer>"
    )
    if offline:
        content += f"\n\n<expectations>\n{expectations}\n</expectations>"
    verdict_type = Verdict if offline else LiveVerdict
    instructions = JUDGE_PROMPT.format(
        user=context.describe_user(),
        today=context.today,
        answer_and_expectations=(
            ", the assistant's final answer, and the eval author's expectations for a correct answer"
            if offline else ", and the assistant's final answer"
        ),
        expectations_basis=" and the expectations" if offline else "",
        expectations_criterion="- expectations: the answer meets every point in the expectations.\n" if offline else "",
        expectations_strict=", one missed point fails expectations" if offline else "",
    )
    request = dict(
        model=settings.judge_model,
        instructions=instructions,
        input=content,
        text={"format": {"type": "json_schema", "name": "verdict", "schema": verdict_type.model_json_schema(), "strict": False}},
        reasoning={"effort": "medium"},
        # A verdict, reasoning included, is ~500 tokens. Groq's free tier allows 1000
        # output tokens per minute for some models and rejects any request whose limit
        # is above that, so keep this at or below the judge model's per-minute limit.
        max_output_tokens=1000,
    )
    # One retry: the provider rejects output that doesn't match the schema
    # (json_validate_failed), and a retry usually gets it right.
    for _ in range(2):
        try:
            response = await client.responses.create(**request)
            return verdict_type.model_validate_json(response.output_text)
        except BadRequestError as exc:
            if exc.code != "json_validate_failed":
                raise
            error: Exception = exc
        except ValidationError as exc:
            error = exc
    raise JudgeError(f"No valid verdict after 2 tries: {error}")
