"""Tests for local subagent runtime and agent batching."""

import json
from pathlib import Path
from types import SimpleNamespace

from src.agent import Agent
from src.commands import builtin
from src.commands.registry import CommandRegistry
from src.context import Context
from src.logger import SessionLogger
from src.skills import SkillManager
from src.subagents import SubagentManager, SubagentRequest, SubagentResult
from src.tool_builder import build_tool_registry
from src.tools import ToolRegistry
from src.tools.subagent import RunSubagentTool


class DummyParentLLM:
    """Minimal parent-LLM stub used to seed child configuration."""

    def __init__(self) -> None:
        self.provider = "ollama"
        self.model = "llama3"
        self.base_url = "http://localhost:11434/v1"
        self.logger = None


class FakeChildLLM:
    """A deterministic child LLM used by subagent runtime tests."""

    def __init__(self, provider=None, model=None, base_url=None, logger=None):
        self.provider = provider or "ollama"
        self.model = model or "llama3"
        self.base_url = base_url or "http://localhost:11434/v1"
        self.logger = logger

    def chat(self, messages, tools=None, log_context=None):
        if self.logger and log_context:
            request_payload = {"model": self.model, "messages": messages, "tools": tools, "stream": False}
            self.logger.log_llm_request(
                turn_id=log_context["turn_id"],
                iteration=log_context["iteration"],
                request_payload=request_payload,
                provider=self.provider,
                model=self.model,
                stream=False,
                request_kind=log_context.get("request_kind", "agent_turn"),
            )

        metrics = SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=6,
            total_tokens=18,
            cached_tokens=0,
            duration=0.1,
            iteration=None,
        )
        response_payload = {
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Executive summary.\n\nFull delegated report.",
                        "tool_calls": None,
                    },
                }
            ],
        }
        if self.logger and log_context:
            self.logger.log_llm_response(
                turn_id=log_context["turn_id"],
                iteration=log_context["iteration"],
                response_payload=response_payload,
                provider=self.provider,
                model=self.model,
                stream=False,
                metrics={
                    "prompt_tokens": 12,
                    "completion_tokens": 6,
                    "total_tokens": 18,
                    "cached_tokens": 0,
                    "ttft": 0.0,
                    "duration": 0.1,
                    "tokens_per_second": 60.0,
                    "tpot": 0.0,
                },
                request_kind=log_context.get("request_kind", "agent_turn"),
            )
        return {"role": "assistant", "content": "Executive summary.\n\nFull delegated report."}, metrics


def test_build_tool_registry_includes_and_excludes_run_subagent(temp_dir):
    """Parent registries should expose run_subagent while child ones should not."""
    skill_manager = SkillManager(repo_root=Path(temp_dir))
    skill_manager.discover()
    subagent_manager = SubagentManager()

    parent_tools = build_tool_registry(
        skill_manager=skill_manager,
        subagent_manager=subagent_manager,
        include_subagent_tool=True,
    )
    child_tools = build_tool_registry(
        skill_manager=skill_manager,
        subagent_manager=subagent_manager,
        include_subagent_tool=False,
    )

    assert "run_subagent" in parent_tools.list_tools()
    assert "run_subagent" not in child_tools.list_tools()


def test_subagent_run_creates_nested_child_session(monkeypatch, temp_dir):
    """A completed child run should return structured metadata and nested logs."""
    monkeypatch.setattr("src.subagents.LLMClient", FakeChildLLM)

    context = Context.create(cwd=str(temp_dir))
    skill_manager = SkillManager(repo_root=Path(temp_dir))
    skill_manager.discover()
    subagent_manager = SubagentManager()
    tools = build_tool_registry(
        skill_manager=skill_manager,
        subagent_manager=subagent_manager,
        include_subagent_tool=True,
    )
    parent_logger = SessionLogger(context.session_id, log_dir=str(temp_dir), enabled=True)
    parent_agent = Agent(
        DummyParentLLM(),
        tools,
        context,
        skill_manager=skill_manager,
        logger=parent_logger,
        subagent_manager=subagent_manager,
    )

    request = SubagentRequest(
        task="Inspect the logging flow and summarize it.",
        label="research-logging",
        context="Focus on parent and child session logging.",
        success_criteria="Return one concise report.",
        files=["src/logger.py"],
        output_hint="short report",
    )
    result = subagent_manager.run_one(parent_agent, request, parent_turn_id=1, iteration=0)
    parent_agent.logger.close()

    assert result.status == "completed"
    assert result.summary == "Executive summary."
    assert result.report.startswith("Executive summary.")
    assert Path(result.session_dir).exists()
    assert Path(result.llm_log).exists()
    assert Path(result.events_log).exists()

    session_json = json.loads((Path(result.session_dir) / "session.json").read_text())
    assert session_json["session_kind"] == "subagent"
    assert session_json["parent_session_id"] == context.session_id
    assert session_json["parent_turn_id"] == 1
    assert session_json["subagent_label"] == "research-logging"

    child_llm_log = Path(result.llm_log).read_text()
    assert "Request Kind: subagent_turn" in child_llm_log
    assert "run_subagent" not in child_llm_log

    latest_session = (Path(temp_dir) / "latest-session").resolve()
    assert latest_session == parent_agent.logger.session_dir.resolve()


def test_agent_batches_consecutive_run_subagent_calls(temp_dir):
    """The agent should batch consecutive run_subagent tool calls through SubagentManager."""

    class FakeSubagentManager:
        def __init__(self):
            self.seen_requests = []

        def build_request(self, arguments):
            return SubagentRequest(
                task=arguments["task"],
                label=arguments.get("label", ""),
                context="",
                success_criteria="",
                files=[],
                output_hint="",
            )

        def run_batch(self, parent_agent, requests, *, parent_turn_id, iteration, on_event=None):
            self.seen_requests = list(requests)
            return [
                SubagentResult(
                    subagent_id=f"sa_000{i+1}",
                    label=request.label or f"subagent-{i+1}",
                    status="completed",
                    summary=f"summary {i+1}",
                    report=f"report {i+1}",
                    session_dir=f"/tmp/sa_{i+1}",
                    llm_log=f"/tmp/sa_{i+1}/llm.log",
                    events_log=f"/tmp/sa_{i+1}/events.jsonl",
                    llm_call_count=1,
                    tool_call_count=0,
                    tools_used=[],
                )
                for i, request in enumerate(requests)
            ]

        def result_to_payload(self, result):
            return {
                "subagent_id": result.subagent_id,
                "label": result.label,
                "status": result.status,
                "summary": result.summary,
                "report": result.report,
                "session_dir": result.session_dir,
                "llm_log": result.llm_log,
                "events_log": result.events_log,
                "llm_call_count": result.llm_call_count,
                "tool_call_count": result.tool_call_count,
                "tools_used": result.tools_used,
                "error": result.error,
            }

    fake_manager = FakeSubagentManager()
    context = Context.create(cwd=str(temp_dir))
    logger = SessionLogger(context.session_id, log_dir=str(temp_dir), enabled=True)
    tools = ToolRegistry()
    tools.register(RunSubagentTool(fake_manager))
    agent = Agent(
        DummyParentLLM(),
        tools,
        context,
        logger=logger,
        subagent_manager=fake_manager,
    )

    turn_id = agent.logger.start_turn(raw_user_input="delegate", normalized_user_input="delegate")
    messages = []
    processed = agent._process_tool_calls(
        [
            {"id": "call_1", "name": "run_subagent", "arguments": json.dumps({"task": "a", "label": "one"})},
            {"id": "call_2", "name": "run_subagent", "arguments": json.dumps({"task": "b", "label": "two"})},
        ],
        messages,
        turn_id,
        0,
    )
    agent.logger.close()

    assert processed == 2
    assert [request.task for request in fake_manager.seen_requests] == ["a", "b"]
    assert len(messages) == 2
    assert json.loads(messages[0]["content"])["subagent_id"] == "sa_0001"
    assert json.loads(messages[1]["content"])["subagent_id"] == "sa_0002"


def test_builtin_subagent_command_show_and_run_help():
    """The built-in registry should expose /subagent help and targeted help."""
    registry = CommandRegistry()
    builtin.register_all(registry)

    assert registry.get_command("subagent") is not None
