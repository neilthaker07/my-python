from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import AssistantResponse, ToolCall
from tests.conftest import ADMIN_ID


class FakeAssistant:
    def __init__(self) -> None:
        self.questions: list[str] = []
        self.users: list[int] = []

    async def ask_with_results(self, question: str, user):
        self.questions.append(question)
        self.users.append(user.id)
        call = ToolCall(name="list_tasks", input={"overdue": True})
        response = AssistantResponse(answer="You have 1 overdue task.", tool_calls=[call])
        return response, [(call, '{"tasks": [], "more_tasks_exist": false}')]


@pytest.fixture
def fake_assistant() -> Iterator[FakeAssistant]:
    fake = FakeAssistant()
    app.state.assistant = fake
    yield fake
    del app.state.assistant


def test_assistant_returns_answer_and_tool_calls(client: TestClient, fake_assistant: FakeAssistant):
    response = client.post("/assistant", json={"question": "Show my overdue tasks"})

    assert response.status_code == 200
    body = response.json()
    assert len(body.pop("trace_id")) == 32
    assert body == {
        "answer": "You have 1 overdue task.",
        "tool_calls": [{"name": "list_tasks", "input": {"overdue": True}}],
    }
    assert fake_assistant.questions == ["Show my overdue tasks"]
    assert fake_assistant.users == [ADMIN_ID]  # the test client acts as the admin


def test_assistant_rejects_empty_question(client: TestClient, fake_assistant: FakeAssistant):
    assert client.post("/assistant", json={"question": ""}).status_code == 422


def test_assistant_unavailable_returns_503(client: TestClient):
    response = client.post("/assistant", json={"question": "hi"})
    assert response.status_code == 503
