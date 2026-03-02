"""Test main.py functions."""

import pytest
import os
import sys
from pathlib import Path
from rich.console import Console

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class FakeTimer:
    """Deterministic timer stub for streaming tests."""

    def __init__(self, interval, callback, args=None, kwargs=None):
        self.interval = interval
        self.callback = callback
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.daemon = False
        self._started = False
        self._cancelled = False

    def start(self):
        self._started = True

    def cancel(self):
        self._cancelled = True

    def is_alive(self):
        return self._started and not self._cancelled


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


def test_display_streaming_response_prints_text_once(monkeypatch):
    """Plain text streaming should still render once without duplicates."""
    from src import main as main_module

    monkeypatch.setattr(main_module.threading, "Timer", FakeTimer)

    class StubAgent:
        def run_stream(self, user_input, on_tool_call=None):
            assert user_input == "hello"
            yield "Hello"
            yield " world"

    console = make_recording_console()
    response = main_module.display_streaming_response(console, StubAgent(), "hello")
    output = console.export_text()
    normalized = normalize_output(output)

    assert response == "Hello world"
    assert "Hello world" in normalized
    assert normalized.count("Hello") == 1
    assert output.endswith("\n")


def test_display_streaming_response_empty_stream_has_no_output(monkeypatch):
    """Empty streams should not leave loading text behind."""
    from src import main as main_module

    monkeypatch.setattr(main_module.threading, "Timer", FakeTimer)

    class StubAgent:
        def run_stream(self, user_input, on_tool_call=None):
            if False:
                yield user_input

    console = make_recording_console()
    response = main_module.display_streaming_response(console, StubAgent(), "hello")

    assert response == ""
    assert console.export_text() == ""


def test_display_streaming_response_renders_markdown(monkeypatch):
    """Streaming markdown should be rendered, not printed as raw syntax."""
    from src import main as main_module

    monkeypatch.setattr(main_module.threading, "Timer", FakeTimer)

    class StubAgent:
        def run_stream(self, user_input, on_tool_call=None):
            yield "# Title"
            yield "\n\n- one"
            yield "\n- two"

    console = make_recording_console()
    response = main_module.display_streaming_response(console, StubAgent(), "hello")
    normalized = normalize_output(console.export_text())

    assert response == "# Title\n\n- one\n- two"
    assert "Title" in normalized
    assert "• one" in normalized
    assert "• two" in normalized
    assert "# Title" not in normalized


def test_display_streaming_response_preserves_tool_call_output(monkeypatch):
    """Tool call notices should still appear before a later markdown answer."""
    from src import main as main_module

    monkeypatch.setattr(main_module.threading, "Timer", FakeTimer)

    class StubAgent:
        def run_stream(self, user_input, on_tool_call=None):
            if on_tool_call:
                on_tool_call("deepwiki:ask_question", {"repoName": "sgl-project/sglang"})
            yield "## Answer\n\n- one"

    console = make_recording_console()
    response = main_module.display_streaming_response(console, StubAgent(), "hello")
    normalized = normalize_output(console.export_text())

    assert response == "## Answer\n\n- one"
    assert "→ deepwiki:ask_question(repoName='sgl-project/sglang')" in normalized
    assert "Answer" in normalized
    assert "• one" in normalized
    assert "## Answer" not in normalized
    assert "Thinking..." not in normalized


def test_display_streaming_response_preserves_tool_call_chronology(monkeypatch):
    """Tool calls should remain ordered between assistant markdown segments."""
    from src import main as main_module

    monkeypatch.setattr(main_module.threading, "Timer", FakeTimer)

    class StubAgent:
        def run_stream(self, user_input, on_tool_call=None):
            yield "I'll ask **DeepWiki** first."
            if on_tool_call:
                on_tool_call("deepwiki:ask_question", {"repoName": "sgl-project/sglang"})
            yield "\n\n## Answer\n\n- one"

    console = make_recording_console()
    response = main_module.display_streaming_response(console, StubAgent(), "hello")
    normalized = normalize_output(console.export_text())

    assert response == "I'll ask **DeepWiki** first.\n\n## Answer\n\n- one"
    assert normalized.index("I'll ask DeepWiki first.") < normalized.index(
        "→ deepwiki:ask_question(repoName='sgl-project/sglang')"
    )
    assert normalized.index("→ deepwiki:ask_question(repoName='sgl-project/sglang')") < normalized.index(
        "Answer"
    )
    assert "• one" in normalized


def test_display_streaming_response_appends_trailing_newline(monkeypatch):
    """Final rendered streaming output should end with a newline."""
    from src import main as main_module

    monkeypatch.setattr(main_module.threading, "Timer", FakeTimer)

    class StubAgent:
        def run_stream(self, user_input, on_tool_call=None):
            yield "No newline yet"

    console = make_recording_console()
    response = main_module.display_streaming_response(console, StubAgent(), "hello")
    output = console.export_text()

    assert response == "No newline yet"
    assert output.endswith("\n")
