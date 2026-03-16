"""HTTP route tests for the simplified local server."""

import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from src.context import Context
from src.server.app import create_app
from src.server import routes as server_routes


def wait_for_run_completion(client: TestClient, run_id: str, timeout: float = 2.0):
    """Poll the run endpoint until the run finishes."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/v1/runs/{run_id}")
        payload = response.json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"Run {run_id} did not complete")


def test_health_and_session_routes(temp_dir, http_runtime_config, patch_http_runtime):
    """The app should expose basic health and session CRUD endpoints."""
    app = create_app(
        runtime_config=http_runtime_config,
        repo_root=temp_dir,
    )

    with TestClient(app) as client:
        assert client.get("/api/v1/health").json() == {"status": "ok"}

        created = client.post("/api/v1/sessions", json={})
        assert created.status_code == 201
        session_payload = created.json()
        assert session_payload["title"].startswith("Session ")

        listed = client.get("/api/v1/sessions")
        assert listed.status_code == 200
        assert listed.json()[0]["id"] == session_payload["id"]


def test_run_validation_and_detail_endpoints(temp_dir, http_runtime_config, patch_http_runtime):
    """Run creation should validate inputs and expose final run detail."""
    app = create_app(
        runtime_config=http_runtime_config,
        repo_root=temp_dir,
    )

    with TestClient(app) as client:
        session = client.post("/api/v1/sessions", json={"title": "HTTP"}).json()

        empty = client.post(f"/api/v1/sessions/{session['id']}/runs", json={"input": "   "})
        assert empty.status_code == 400

        slash = client.post(f"/api/v1/sessions/{session['id']}/runs", json={"input": "/help"})
        assert slash.status_code == 400
        assert slash.json()["detail"] == "Slash commands are CLI-only in HTTP v1."

        queued = client.post(
            f"/api/v1/sessions/{session['id']}/runs",
            json={"input": "hello http"},
        )
        assert queued.status_code == 202
        run_id = queued.json()["id"]

        run_detail = wait_for_run_completion(client, run_id)
        assert run_detail["final_output"] == "Echo: hello http"
        assert "llm_call_count" not in run_detail
        assert "tool_call_count" not in run_detail
        assert client.get(f"/api/v1/runs/{run_id}/events").status_code == 404

        session_detail = client.get(f"/api/v1/sessions/{session['id']}").json()
        assert [message["role"] for message in session_detail["messages"]] == ["user", "assistant"]


def test_build_agent_for_http_run_closes_mcp_on_assembly_failure(monkeypatch, temp_dir, http_runtime_config):
    """MCP manager should be closed when runtime assembly fails after MCP init."""

    class FakeMCPManager:
        def __init__(self) -> None:
            self.closed_count = 0

        def close_all(self) -> None:
            self.closed_count += 1

    fake_mcp_manager = FakeMCPManager()

    monkeypatch.setattr(server_routes, "_build_mcp_manager", lambda _runtime_config: fake_mcp_manager)
    monkeypatch.setattr(
        server_routes,
        "SkillManager",
        lambda repo_root: SimpleNamespace(discover=lambda: None),
    )
    monkeypatch.setattr(
        server_routes,
        "SubagentManager",
        lambda runtime_config=None: object(),
    )

    def failing_registry_builder(**_kwargs):
        raise RuntimeError("synthetic registry failure")

    monkeypatch.setattr(server_routes, "build_tool_registry", failing_registry_builder)

    fake_app = SimpleNamespace(
        state=SimpleNamespace(
            runtime_config=http_runtime_config,
            repo_root=temp_dir,
        )
    )
    context = Context(cwd=temp_dir, session_id="sess_test")

    with pytest.raises(RuntimeError, match="synthetic registry failure"):
        server_routes._build_agent_for_http_run(fake_app, context, "run_test")

    assert fake_mcp_manager.closed_count == 1
