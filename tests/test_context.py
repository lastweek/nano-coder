"""Test Context class."""

import pytest
from src.context import Context
from pathlib import Path


class TestContext:
    """Test Context dataclass."""

    def test_create_context(self):
        """Test creating a context with default values."""
        context = Context(cwd=Path("/tmp"))
        assert context.cwd == Path("/tmp")
        assert isinstance(context.session_id, str)
        assert len(context.session_id) > 0
        assert context.messages == []

    def test_create_from_string(self):
        """Test Context.create classmethod."""
        context = Context.create(".")
        assert context.cwd.is_absolute()
        assert isinstance(context.cwd, Path)

    def test_add_message(self):
        """Test adding a message to history."""
        context = Context(cwd=Path("/tmp"))
        context.add_message("user", "Hello")
        assert len(context.messages) == 1
        assert context.messages[0] == {"role": "user", "content": "Hello"}

    def test_add_multiple_messages(self):
        """Test adding multiple messages."""
        context = Context(cwd=Path("/tmp"))
        context.add_message("user", "Hello")
        context.add_message("assistant", "Hi there")
        assert len(context.messages) == 2

    def test_get_messages(self):
        """Test getting messages returns the list."""
        context = Context(cwd=Path("/tmp"))
        context.add_message("user", "test")
        messages = context.get_messages()
        assert len(messages) == 1
        # Should return same list (not copy, based on our implementation)
        assert messages is context.messages

    def test_clear_messages(self):
        """Test clearing messages."""
        context = Context(cwd=Path("/tmp"))
        context.add_message("user", "test")
        context.clear_messages()
        assert len(context.messages) == 0
