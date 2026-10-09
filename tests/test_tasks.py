from datetime import date, timedelta

from fastapi.testclient import TestClient

TODAY = date.today()
YESTERDAY = TODAY - timedelta(days=1)
TOMORROW = TODAY + timedelta(days=1)


def create(client: TestClient, **fields) -> dict:
    response = client.post("/tasks", json={"title": "A task", **fields})
    assert response.status_code == 201, response.text
    return response.json()


def test_create_task_uses_defaults(client: TestClient):
    task = create(client, title="Prepare for interview", due_date=str(TOMORROW))

    assert task["title"] == "Prepare for interview"
    assert task["status"] == "todo"
    assert task["priority"] == "medium"
    assert task["due_date"] == str(TOMORROW)
    assert task["completed_at"] is None


def test_create_task_rejects_empty_title(client: TestClient):
    response = client.post("/tasks", json={"title": ""})
    assert response.status_code == 422


def test_get_missing_task_returns_404(client: TestClient):
    assert client.get("/tasks/999").status_code == 404


def test_update_task_partially(client: TestClient):
    task = create(client, title="Old", description="keep me")

    response = client.patch(f"/tasks/{task['id']}", json={"title": "New"})

    assert response.status_code == 200
    assert response.json()["title"] == "New"
    assert response.json()["description"] == "keep me"


def test_update_rejects_null_title(client: TestClient):
    task = create(client)
    response = client.patch(f"/tasks/{task['id']}", json={"title": None})
    assert response.status_code == 422


def test_update_can_clear_due_date(client: TestClient):
    task = create(client, due_date=str(TODAY))
    response = client.patch(f"/tasks/{task['id']}", json={"due_date": None})
    assert response.json()["due_date"] is None


def test_marking_done_sets_and_clears_completed_at(client: TestClient):
    task = create(client)

    done = client.patch(f"/tasks/{task['id']}", json={"status": "done"}).json()
    assert done["completed_at"] is not None

    reopened = client.patch(f"/tasks/{task['id']}", json={"status": "todo"}).json()
    assert reopened["completed_at"] is None


def test_delete_task(client: TestClient):
    task = create(client)

    assert client.delete(f"/tasks/{task['id']}").status_code == 204
    assert client.get(f"/tasks/{task['id']}").status_code == 404


def test_list_overdue_excludes_done_and_future(client: TestClient):
    overdue = create(client, title="overdue", due_date=str(YESTERDAY))
    finished = create(client, title="finished", due_date=str(YESTERDAY))
    client.patch(f"/tasks/{finished['id']}", json={"status": "done"})
    create(client, title="future", due_date=str(TOMORROW))
    create(client, title="no date")

    titles = [t["title"] for t in client.get("/tasks", params={"overdue": True}).json()]

    assert titles == [overdue["title"]]


def test_list_completed_since(client: TestClient):
    done = create(client, title="done")
    client.patch(f"/tasks/{done['id']}", json={"status": "done"})
    create(client, title="not done")

    response = client.get("/tasks", params={"completed_since": str(TODAY - timedelta(days=7))})

    assert [t["title"] for t in response.json()] == ["done"]


def test_reschedule_moves_only_unfinished_tasks(client: TestClient):
    unfinished = create(client, title="unfinished", due_date=str(TODAY))
    finished = create(client, title="finished", due_date=str(TODAY))
    client.patch(f"/tasks/{finished['id']}", json={"status": "done"})

    response = client.post(
        "/tasks/reschedule", json={"from_date": str(TODAY), "to_date": str(TOMORROW)}
    )

    assert response.json()["moved"] == 1
    assert client.get(f"/tasks/{unfinished['id']}").json()["due_date"] == str(TOMORROW)
    assert client.get(f"/tasks/{finished['id']}").json()["due_date"] == str(TODAY)
