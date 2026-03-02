"""Tests for slash command registry execution rules."""

import io

from rich.console import Console

from src.commands.registry import CommandRegistry


def create_console(buffer: io.StringIO) -> Console:
    """Create a deterministic Rich console for tests."""
    return Console(file=buffer, force_terminal=False, color_system=None)


def test_execute_runs_command_with_leading_whitespace():
    """Commands should execute when slash is the first non-space character."""
    registry = CommandRegistry()
    calls = []

    @registry.register("help", "Show help")
    def cmd_help(console, args, context):
        calls.append((args, context))

    output = io.StringIO()
    console = create_console(output)

    executed = registry.execute("   /help topic", console, {"mode": "test"})

    assert executed is True
    assert calls == [("topic", {"mode": "test"})]


def test_execute_ignores_inline_slash_text():
    """Slash text later in the line should be treated as normal user input."""
    registry = CommandRegistry()
    calls = []

    @registry.register("help", "Show help")
    def cmd_help(console, args, context):
        calls.append(args)

    output = io.StringIO()
    console = create_console(output)

    executed = registry.execute("show me /help", console, {})

    assert executed is False
    assert calls == []


def test_execute_unknown_command_with_leading_whitespace():
    """Unknown slash commands should still show the existing error text."""
    registry = CommandRegistry()
    output = io.StringIO()
    console = create_console(output)

    executed = registry.execute("   /unknown", console, {})

    assert executed is True
    text = output.getvalue()
    assert "Unknown command: /unknown" in text
    assert "Type /help for available commands" in text
