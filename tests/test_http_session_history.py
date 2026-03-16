"""HTTP persistence tests focused on session history behavior."""

import time

from fastapi.testclient import TestClient

from src.server.app import create_app
from src.store.repository import AppStore


def wait_for_run(client: TestClient, run_id: str, timeout: float = 2.0) -> dict:
    """Poll until one run reaches a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/api/v1/runs/{run_id}").json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"Run {run_id} did not finish")


def test_http_session_snapshot_matches_compacted_context(temp_dir, http_runtime_config, patch_http_runtime):
    """Persisted session history should be replaced by the compacted context snapshot."""
    app = create_app(
        runtime_config=http_runtime_config,
        repo_root=temp_dir,
    )

    with TestClient(app) as client:
        session = client.post("/api/v1/sessions", json={"title": "HTTP"}).json()
        first_run = client.post(
            f"/api/v1/sessions/{session['id']}/runs",
            json={"input": "hello one"},
        ).json()
        wait_for_run(client, first_run["id"])

        compact_run = client.post(
            f"/api/v1/sessions/{session['id']}/runs",
            json={"input": "compact session"},
        ).json()
        wait_for_run(client, compact_run["id"])

        session_detail = client.get(f"/api/v1/sessions/{session['id']}").json()
        assert session_detail["summary_text"] == "Compacted summary"
        assert [message["content"] for message in session_detail["messages"]] == [
            "compact session",
            "Echo: compact session",
        ]


def test_server_startup_marks_incomplete_runs_failed(temp_dir, http_runtime_config, patch_http_runtime):
    """Creating the app should fail stale queued/running runs from prior processes."""
    store = AppStore(temp_dir / "state.db")
    store.init_db()
    session = store.create_session("Restart")
    queued = store.create_run(session.id, "queued")
    running = store.create_run(session.id, "running")
    store.set_run_running(running.id)

    app = create_app(
        runtime_config=http_runtime_config,
        repo_root=temp_dir,
    )
    with TestClient(app):
        pass

    assert store.get_run(queued.id).status == "failed"
    assert store.get_run(running.id).status == "failed"


def test_http_non_compaction_turns_use_append_snapshot_path(
    temp_dir,
    http_runtime_config,
    patch_http_runtime,
):
    """Regular turns should append transcript deltas without full snapshot replacement."""
    app = create_app(
        runtime_config=http_runtime_config,
        repo_root=temp_dir,
    )

    append_calls = 0
    replace_calls = 0
    original_append = app.state.store.append_session_messages
    original_replace = app.state.store.replace_session_snapshot

    def append_wrapper(*args, **kwargs):
        nonlocal append_calls
        append_calls += 1
        return original_append(*args, **kwargs)

    def replace_wrapper(*args, **kwargs):
        nonlocal replace_calls
        replace_calls += 1
        return original_replace(*args, **kwargs)

    app.state.store.append_session_messages = append_wrapper
    app.state.store.replace_session_snapshot = replace_wrapper

    with TestClient(app) as client:
        session = client.post("/api/v1/sessions", json={"title": "HTTP"}).json()
        first = client.post(
            f"/api/v1/sessions/{session['id']}/runs",
            json={"input": "first"},
        ).json()
        wait_for_run(client, first["id"])

        second = client.post(
            f"/api/v1/sessions/{session['id']}/runs",
            json={"input": "second"},
        ).json()
        wait_for_run(client, second["id"])

    assert append_calls == 2
    assert replace_calls == 0
