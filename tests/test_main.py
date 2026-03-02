"""Test main.py functions."""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch
from rich.console import Console
from src.turn_activity import TurnActivityEvent

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def make_recording_console() -> Console:
    """Create a console suitable for deterministic output assertions."""
    return Console(record=True, force_terminal=False, width=100)


def normalize_output(output: str) -> str:
    """Normalize Rich exported text for stable assertions."""
    return "\n".join(line.rstrip() for line in output.splitlines())


def test_print_banner_no_error():
    """Test that print_banner runs without errors."""
    from src.main import print_banner

    console = Console()
    # Should not raise any exceptions
    print_banner(console)


def test_print_banner_with_context_window(monkeypatch):
    """Test banner displays with context_window set."""
    # Enable test mode
    monkeypatch.setenv("NANO_CODER_TEST", "true")

    # Set test values
    monkeypatch.setenv("LLM_PROVIDER", "test-provider")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://test.example.com")
    monkeypatch.setenv("LLM_CONTEXT_WINDOW", "128000")

    # Reload config to pick up env vars
    from src.config import Config
    Config.reload()

    # Import after reload to get fresh config
    from src.main import print_banner

    console = Console()
    # Should not raise any exceptions
    print_banner(console)


def test_print_banner_minimal_config(monkeypatch):
    """Test banner with minimal config."""
    # Enable test mode
    monkeypatch.setenv("NANO_CODER_TEST", "true")

    # Clear any existing LLM env vars
    for key in list(os.environ.keys()):
        if key.startswith(("LLM_", "API_KEY")):
            monkeypatch.delenv(key, raising=False)

    # Reload config to pick up changes
    from src.config import Config
    Config.reload()

    # Import after reload
    from src.main import print_banner

    console = Console()
    # Should not raise any exceptions even with minimal config
    print_banner(console)


class StubContext:
    """Simple context stub for turn runner tests."""

    def __init__(self, messages=None):
        self._messages = messages or []

    def get_messages(self):
        return list(self._messages)


def test_run_agent_turn_streaming_shows_summary_and_answer():
    """Streaming turns should persist a concise summary and render the final answer once."""
    from src.main import run_agent_turn

    class StubAgent:
        def __init__(self):
            self.context = StubContext([{"role": "assistant", "content": "## Answer\n\n- one"}])

        def run_stream(self, user_input, on_tool_call=None, on_event=None):
            assert user_input == "hello"
            on_event(TurnActivityEvent("llm_call_started", iteration=0, details={
                "stream": True,
                "message_count": 2,
                "tool_schema_count": 4,
            }))
            on_event(TurnActivityEvent("llm_call_finished", iteration=0, details={
                "stream": True,
                "duration_s": 0.2,
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
                "cached_tokens": 10,
                "has_tool_calls": True,
                "tool_call_count": 1,
                "result_kind": "tool_calls",
            }))
            on_event(TurnActivityEvent("tool_call_started", iteration=0, details={
                "tool_name": "deepwiki:ask_question",
                "tool_call_id": "call_1",
                "arguments": {
                    "repoName": "repo",
                    "question": "what?",
                },
            }))
            on_event(TurnActivityEvent("tool_call_finished", iteration=0, details={
                "tool_name": "deepwiki:ask_question",
                "tool_call_id": "call_1",
                "arguments": {
                    "repoName": "repo",
                    "question": "what?",
                },
                "success": True,
                "duration_s": 12.73,
            }))
            on_event(TurnActivityEvent("llm_call_started", iteration=1, details={
                "stream": True,
                "message_count": 4,
                "tool_schema_count": 4,
            }))
            yield "## Answer\n\n- one"
            on_event(TurnActivityEvent("llm_call_finished", iteration=1, details={
                "stream": True,
                "duration_s": 0.4,
                "prompt_tokens": 30,
                "completion_tokens": 7,
                "total_tokens": 37,
                "cached_tokens": 12,
                "has_tool_calls": False,
                "tool_call_count": 0,
                "result_kind": "final_answer",
            }))
            on_event(TurnActivityEvent("turn_completed", details={
                "status": "completed",
                "llm_call_count": 2,
                "tool_call_count": 1,
                "tools_used": ["deepwiki:ask_question"],
                "skills_used": [],
            }))

    console = make_recording_console()
    response = run_agent_turn(
        console,
        StubAgent(),
        "hello",
        enable_streaming=True,
        skill_debug=False,
    )
    normalized = normalize_output(console.export_text())

    assert response == "## Answer\n\n- one"
    assert "LLM call 1 requested 1 tool" in normalized
    assert "Tool finished: deepwiki:ask_question(repoName='repo', question='what?') (12.73s)" in normalized
    assert "LLM call 2 produced final answer" in normalized
    assert normalized.count("LLM call 1 requested 1 tool") == 1
    assert normalized.count("Tool finished: deepwiki:ask_question(repoName='repo', question='what?') (12.73s)") == 1
    assert normalized.count("LLM call 2 produced final answer") == 1
    assert "Answer" in normalized
    assert "• one" in normalized
    assert normalized.count("Answer") == 1
    assert "→ deepwiki:ask_question" not in normalized
    assert normalized.index("LLM call 2 produced final answer") < normalized.index("Answer")


def test_run_agent_turn_non_streaming_persists_summary_and_answer():
    """Non-streaming turns should use the same live activity path."""
    from src.main import run_agent_turn

    class StubAgent:
        context = StubContext()

        def run(self, user_input, on_tool_call=None, on_event=None):
            assert user_input == "hello"
            on_event(TurnActivityEvent("llm_call_started", iteration=0, details={
                "stream": False,
                "message_count": 2,
                "tool_schema_count": 4,
            }))
            on_event(TurnActivityEvent("llm_call_finished", iteration=0, details={
                "stream": False,
                "duration_s": 0.1,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cached_tokens": 4,
                "has_tool_calls": False,
                "tool_call_count": 0,
                "result_kind": "final_answer",
            }))
            on_event(TurnActivityEvent("turn_completed", details={
                "status": "completed",
                "llm_call_count": 1,
                "tool_call_count": 0,
                "tools_used": [],
                "skills_used": [],
            }))
            return "Plain answer"

    console = make_recording_console()
    response = run_agent_turn(
        console,
        StubAgent(),
        "hello",
        enable_streaming=False,
        skill_debug=False,
    )
    normalized = normalize_output(console.export_text())

    assert response == "Plain answer"
    assert "LLM call 1 produced final answer" in normalized
    assert normalized.count("LLM call 1 produced final answer") == 1
    assert "Plain answer" in normalized
    assert normalized.count("Plain answer") == 1
    assert "Thinking..." not in normalized


def test_run_agent_turn_includes_skill_preload_summary():
    """Skill preload events should be visible in the persisted summary."""
    from src.main import run_agent_turn

    class StubAgent:
        context = StubContext()

        def run(self, user_input, on_tool_call=None, on_event=None):
            on_event(TurnActivityEvent("skill_preload", details={
                "skill_name": "pdf",
                "reason": "explicit",
                "source": "user",
                "catalog_visible": True,
            }))
            on_event(TurnActivityEvent("llm_call_finished", iteration=0, details={
                "stream": False,
                "duration_s": 0.1,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cached_tokens": 4,
                "has_tool_calls": False,
                "tool_call_count": 0,
                "result_kind": "final_answer",
            }))
            on_event(TurnActivityEvent("turn_completed", details={
                "status": "completed",
                "llm_call_count": 1,
                "tool_call_count": 0,
                "tools_used": [],
                "skills_used": ["pdf"],
            }))
            return "Done"

    console = make_recording_console()
    run_agent_turn(
        console,
        StubAgent(),
        "hello",
        enable_streaming=False,
        skill_debug=False,
    )
    normalized = normalize_output(console.export_text())
    assert "Skill preloaded: pdf (explicit)" in normalized


def test_run_agent_turn_includes_subagent_summary_lines():
    """Subagent lifecycle events should be summarized without generic tool noise."""
    from src.main import run_agent_turn

    class StubAgent:
        context = StubContext()

        def run(self, user_input, on_tool_call=None, on_event=None):
            on_event(TurnActivityEvent("subagent_started", details={
                "subagent_id": "sa_0001_abcd1234",
                "label": "research-logging",
                "task": "inspect logging",
            }))
            on_event(TurnActivityEvent("subagent_completed", details={
                "subagent_id": "sa_0001_abcd1234",
                "label": "research-logging",
                "duration_s": 1.25,
                "summary": "summary",
            }))
            on_event(TurnActivityEvent("llm_call_finished", iteration=0, details={
                "stream": False,
                "duration_s": 0.1,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cached_tokens": 4,
                "has_tool_calls": False,
                "tool_call_count": 0,
                "result_kind": "final_answer",
            }))
            on_event(TurnActivityEvent("turn_completed", details={
                "status": "completed",
                "llm_call_count": 1,
                "tool_call_count": 1,
                "tools_used": ["run_subagent"],
                "skills_used": [],
            }))
            return "Plain answer"

    console = make_recording_console()
    response = run_agent_turn(
        console,
        StubAgent(),
        "hello",
        enable_streaming=False,
        skill_debug=False,
    )
    normalized = normalize_output(console.export_text())

    assert response == "Plain answer"
    assert "Subagent finished: research-logging" in normalized
    assert normalized.count("Subagent finished: research-logging") == 1
    assert "Tool finished: run_subagent" not in normalized


def test_run_agent_turn_includes_compaction_summary():
    """Context compaction should appear in the persisted CLI summary."""
    from src.main import run_agent_turn

    class StubAgent:
        context = StubContext()

        def run(self, user_input, on_tool_call=None, on_event=None):
            on_event(TurnActivityEvent("context_compaction_started", details={
                "reason": "threshold_reached",
                "covered_turn_count": 4,
                "retained_turn_count": 2,
            }))
            on_event(TurnActivityEvent("context_compaction_completed", details={
                "reason": "threshold_reached",
                "covered_turn_count": 4,
                "retained_turn_count": 2,
                "before_tokens": 180000,
                "after_tokens": 110000,
            }))
            on_event(TurnActivityEvent("llm_call_finished", iteration=0, details={
                "stream": False,
                "duration_s": 0.1,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cached_tokens": 4,
                "has_tool_calls": False,
                "tool_call_count": 0,
                "result_kind": "final_answer",
            }))
            on_event(TurnActivityEvent("turn_completed", details={
                "status": "completed",
                "llm_call_count": 1,
                "tool_call_count": 0,
                "tools_used": [],
                "skills_used": [],
            }))
            return "Done"

    console = make_recording_console()
    run_agent_turn(
        console,
        StubAgent(),
        "hello",
        enable_streaming=False,
        skill_debug=False,
    )
    normalized = normalize_output(console.export_text())
    assert "Context compacted: 4 older turns summarized" in normalized
    assert normalized.count("Context compacted: 4 older turns summarized") == 1


def test_run_agent_turn_shows_tool_failure_clearly():
    """Failed tools should remain visible in the persisted summary."""
    from src.main import run_agent_turn

    class StubAgent:
        context = StubContext()

        def run(self, user_input, on_tool_call=None, on_event=None):
            on_event(TurnActivityEvent("llm_call_finished", iteration=0, details={
                "stream": False,
                "duration_s": 0.1,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cached_tokens": 4,
                "has_tool_calls": True,
                "tool_call_count": 1,
                "result_kind": "tool_calls",
            }))
            on_event(TurnActivityEvent("tool_call_finished", iteration=0, details={
                "tool_name": "deepwiki:ask_question",
                "tool_call_id": "call_1",
                "arguments": {
                    "repoName": "repo",
                    "question": "what?",
                },
                "success": False,
                "duration_s": 0.5,
                "error": "upstream failure",
            }))
            on_event(TurnActivityEvent("turn_completed", details={
                "status": "error",
                "llm_call_count": 1,
                "tool_call_count": 1,
                "tools_used": ["deepwiki:ask_question"],
                "skills_used": [],
            }))
            return "Sorry."

    console = make_recording_console()
    run_agent_turn(
        console,
        StubAgent(),
        "hello",
        enable_streaming=False,
        skill_debug=False,
    )
    normalized = normalize_output(console.export_text())
    assert "Tool failed: deepwiki:ask_question(repoName='repo', question='what?') (0.50s)" in normalized
    assert normalized.count("Tool failed: deepwiki:ask_question(repoName='repo', question='what?') (0.50s)") == 1


def test_run_agent_turn_distinguishes_repeated_tools_by_arguments():
    """Tool summary lines should include compact argument details."""
    from src.main import run_agent_turn

    class StubAgent:
        context = StubContext()

        def run(self, user_input, on_tool_call=None, on_event=None):
            on_event(TurnActivityEvent("llm_call_finished", iteration=0, details={
                "stream": False,
                "duration_s": 0.1,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cached_tokens": 4,
                "has_tool_calls": True,
                "tool_call_count": 2,
                "result_kind": "tool_calls",
            }))
            on_event(TurnActivityEvent("tool_call_finished", iteration=0, details={
                "tool_name": "run_command",
                "tool_call_id": "call_1",
                "arguments": {"cmd": "pwd"},
                "success": True,
                "duration_s": 0.04,
            }))
            on_event(TurnActivityEvent("tool_call_finished", iteration=0, details={
                "tool_name": "run_command",
                "tool_call_id": "call_2",
                "arguments": {"cmd": "ls src"},
                "success": True,
                "duration_s": 0.01,
            }))
            on_event(TurnActivityEvent("llm_call_finished", iteration=1, details={
                "stream": False,
                "duration_s": 0.1,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cached_tokens": 4,
                "has_tool_calls": False,
                "tool_call_count": 0,
                "result_kind": "final_answer",
            }))
            on_event(TurnActivityEvent("turn_completed", details={
                "status": "completed",
                "llm_call_count": 2,
                "tool_call_count": 2,
                "tools_used": ["run_command"],
                "skills_used": [],
            }))
            return "Done"

    console = make_recording_console()
    run_agent_turn(
        console,
        StubAgent(),
        "hello",
        enable_streaming=False,
        skill_debug=False,
    )
    normalized = normalize_output(console.export_text())
    assert "Tool finished: run_command(cmd='pwd') (0.04s)" in normalized
    assert "Tool finished: run_command(cmd='ls src') (0.01s)" in normalized


def test_run_agent_turn_streaming_does_not_show_answer_twice():
    """Streaming turns should not keep a preview copy plus the final answer."""
    from src.main import run_agent_turn

    class StubAgent:
        def __init__(self):
            self.context = StubContext([{"role": "assistant", "content": "Plain final answer"}])

        def run_stream(self, user_input, on_tool_call=None, on_event=None):
            on_event(TurnActivityEvent("llm_call_started", iteration=0, details={
                "stream": True,
                "message_count": 2,
                "tool_schema_count": 0,
            }))
            yield "Plain "
            yield "final "
            yield "answer"
            on_event(TurnActivityEvent("llm_call_finished", iteration=0, details={
                "stream": True,
                "duration_s": 0.1,
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
                "cached_tokens": 0,
                "has_tool_calls": False,
                "tool_call_count": 0,
                "result_kind": "final_answer",
            }))
            on_event(TurnActivityEvent("turn_completed", details={
                "status": "completed",
                "llm_call_count": 1,
                "tool_call_count": 0,
                "tools_used": [],
                "skills_used": [],
            }))

    console = make_recording_console()
    run_agent_turn(
        console,
        StubAgent(),
        "hello",
        enable_streaming=True,
        skill_debug=False,
    )
    normalized = normalize_output(console.export_text())

    assert normalized.count("Plain final answer") == 1


def test_run_agent_turn_enables_live_auto_refresh_in_terminal():
    """Terminal consoles should enable auto-refresh so the spinner animates."""
    from src.main import run_agent_turn

    live_kwargs = {}

    class StubLive:
        def __init__(self, *args, **kwargs):
            live_kwargs.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, renderable, refresh=False):
            return None

    class StubAgent:
        context = StubContext()

        def run(self, user_input, on_tool_call=None, on_event=None):
            on_event(TurnActivityEvent("llm_call_started", iteration=0, details={
                "stream": False,
                "message_count": 2,
                "tool_schema_count": 1,
            }))
            on_event(TurnActivityEvent("llm_call_finished", iteration=0, details={
                "stream": False,
                "duration_s": 0.1,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cached_tokens": 0,
                "has_tool_calls": False,
                "tool_call_count": 0,
                "result_kind": "final_answer",
            }))
            on_event(TurnActivityEvent("turn_completed", details={
                "status": "completed",
                "llm_call_count": 1,
                "tool_call_count": 0,
                "tools_used": [],
                "skills_used": [],
            }))
            return "Done"

    console = Console(record=True, force_terminal=True, width=100)
    with patch("src.main.Live", StubLive):
        run_agent_turn(
            console,
            StubAgent(),
            "hello",
            enable_streaming=False,
            skill_debug=False,
        )

    assert live_kwargs["auto_refresh"] is True
    assert live_kwargs["refresh_per_second"] == 12
