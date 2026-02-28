"""Test ChatLogger."""

import pytest
import tempfile
import json
import uuid
from pathlib import Path
from src.logger import ChatLogger


class TestChatLogger:
    """Test ChatLogger functionality."""

    def test_creates_log_file(self, temp_dir):
        """Test that logger creates log file."""
        session_id = str(uuid.uuid4())
        logger = ChatLogger(session_id, log_dir=str(temp_dir), enabled=True)
        logger.close()

        # Filter out the symlink, count only actual session files
        log_files = [f for f in Path(temp_dir).glob("*.jsonl") if f.name != "latest.jsonl"]
        assert len(log_files) == 1
        # Verify latest.jsonl symlink exists
        assert (Path(temp_dir) / "latest.jsonl").exists()

    def test_logs_llm_request(self, temp_dir):
        """Test logging LLM request."""
        logger = ChatLogger("test-session", log_dir=str(temp_dir), enabled=True)

        logger.log_llm_request(
            messages=[{"role": "user", "content": "test"}],
            tools=[{"name": "test"}],
            model="gpt-4",
            provider="openai"
        )
        logger.close()

        log_path = list(Path(temp_dir).glob("*.jsonl"))[0]
        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert entry["type"] == "llm_request"
        assert entry["model"] == "gpt-4"

    def test_logs_llm_response(self, temp_dir):
        """Test logging LLM response."""
        logger = ChatLogger("test-session", log_dir=str(temp_dir), enabled=True)

        logger.log_llm_response({
            "role": "assistant",
            "content": "Hello!"
        })
        logger.close()

        log_path = list(Path(temp_dir).glob("*.jsonl"))[0]
        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert entry["type"] == "llm_response"

    def test_logs_tool_calls(self, temp_dir):
        """Test logging tool calls and results."""
        logger = ChatLogger("test-session", log_dir=str(temp_dir), enabled=True)

        logger.log_tool_call("read_file", {"file_path": "test.txt"})
        logger.log_tool_result("read_file", {"success": True, "data": "content"})
        logger.close()

        log_path = list(Path(temp_dir).glob("*.jsonl"))[0]
        content = log_path.read_text()
        lines = content.strip().split("\n")

        assert len(lines) == 2
        entry1 = json.loads(lines[0])
        entry2 = json.loads(lines[1])

        assert entry1["type"] == "tool_call"
        assert entry2["type"] == "tool_result"

    def test_disabled_when_env_false(self, monkeypatch, temp_dir):
        """Test logger respects ENABLE_LOGGING env var."""
        monkeypatch.setenv("ENABLE_LOGGING", "false")
        logger = ChatLogger("test-session", log_dir=str(temp_dir))

        logger.log_user_message("test")
        logger.close()

        log_files = list(Path(temp_dir).glob("*.jsonl"))
        assert len(log_files) == 0

    def test_logs_user_and_agent_messages(self, temp_dir):
        """Test logging user and agent messages."""
        logger = ChatLogger("test-session", log_dir=str(temp_dir), enabled=True)

        logger.log_user_message("Hello, agent!")
        logger.log_agent_response("Hello! How can I help?")
        logger.close()

        log_path = list(Path(temp_dir).glob("*.jsonl"))[0]
        content = log_path.read_text()
        lines = content.strip().split("\n")

        assert len(lines) == 2
        entry1 = json.loads(lines[0])
        entry2 = json.loads(lines[1])

        assert entry1["type"] == "user_message"
        assert entry1["content"] == "Hello, agent!"
        assert entry2["type"] == "agent_response"
        assert entry2["content"] == "Hello! How can I help?"

    def test_includes_session_id(self, temp_dir):
        """Test that session_id is included in log entries."""
        logger = ChatLogger("my-session-123", log_dir=str(temp_dir), enabled=True)

        logger.log_user_message("test")
        logger.close()

        log_path = list(Path(temp_dir).glob("*.jsonl"))[0]
        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert entry["session_id"] == "my-session-123"

    def test_includes_timestamp(self, temp_dir):
        """Test that timestamp is included in log entries."""
        logger = ChatLogger("test-session", log_dir=str(temp_dir), enabled=True)

        logger.log_user_message("test")
        logger.close()

        log_path = list(Path(temp_dir).glob("*.jsonl"))[0]
        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert "timestamp" in entry
        assert entry["timestamp"] is not None

    def test_context_manager(self, temp_dir):
        """Test that logger works as context manager."""
        with ChatLogger("test-session", log_dir=str(temp_dir), enabled=True) as logger:
            logger.log_user_message("test")

        # Filter out the symlink, count only actual session files
        log_files = [f for f in Path(temp_dir).glob("*.jsonl") if f.name != "latest.jsonl"]
        assert len(log_files) == 1

    def test_buffering(self, temp_dir):
        """Test that log entries are buffered and flushed."""
        # Use small buffer size for testing
        logger = ChatLogger("test-session", log_dir=str(temp_dir), enabled=True, buffer_size=3)

        # Log 2 entries (should not flush yet)
        logger.log_user_message("message 1")
        logger.log_user_message("message 2")

        # Read file content - should be empty since buffer not flushed
        log_path = [f for f in Path(temp_dir).glob("*.jsonl") if f.name != "latest.jsonl"][0]
        content = log_path.read_text()
        assert content == ""

        # Log third entry (should trigger flush)
        logger.log_user_message("message 3")

        # Now file should have content
        content = log_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3

        logger.close()
