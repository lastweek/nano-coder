"""Chat completion logging for Nano-Coder."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from threading import Lock


class ChatLogger:
    """Logs chat completions and tool executions to JSONL files."""

    def __init__(self, session_id: str, log_dir: str = "logs", enabled: bool = True, buffer_size: int = 10):
        """Initialize the logger.

        Args:
            session_id: Unique session identifier
            log_dir: Directory to store log files
            enabled: Whether logging is enabled
            buffer_size: Number of log entries to buffer before flushing (default: 10)
        """
        self.session_id = session_id
        self.log_dir = Path(log_dir)
        # Check ENABLE_LOGGING env var, default to true
        if enabled and os.environ.get("ENABLE_LOGGING", "true").lower() != "true":
            enabled = False
        self.enabled = enabled
        self._lock = Lock()
        self._file = None
        self._buffer = []
        self._buffer_size = buffer_size

        if self.enabled:
            self._open_log_file()

    def _open_log_file(self):
        """Open/create the log file for this session."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = self.log_dir / f"session-{self.session_id[:8]}-{timestamp}.jsonl"

        self._file = open(log_path, "a", encoding="utf-8")

        # Create/update symlink to latest
        latest_link = self.log_dir / "latest.jsonl"
        if latest_link.exists():
            latest_link.unlink()
        try:
            latest_link.symlink_to(log_path.name)
        except OSError:
            # Symlinks may not work on all systems, skip if fails
            pass

    def log(self, entry: Dict[str, Any]) -> None:
        """Write a log entry to the file.

        Args:
            entry: Dictionary containing log data
        """
        if not self.enabled or self._file is None:
            return

        with self._lock:
            entry["timestamp"] = datetime.now().isoformat()
            entry["session_id"] = self.session_id
            self._buffer.append(json.dumps(entry))

            # Flush when buffer reaches threshold
            if len(self._buffer) >= self._buffer_size:
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush the buffer to file."""
        if self._buffer and self._file:
            self._file.write("\n".join(self._buffer) + "\n")
            self._file.flush()
            self._buffer.clear()

    def log_llm_request(self, messages: List[Dict], tools: Optional[List[Dict]], model: str, provider: str) -> None:
        """Log an LLM request."""
        entry = {
            "type": "llm_request",
            "model": model,
            "provider": provider,
            "messages": messages,
        }
        if tools:
            entry["tools"] = tools
        self.log(entry)

    def log_llm_response(self, response: Dict) -> None:
        """Log an LLM response."""
        entry = {
            "type": "llm_response",
            **response
        }
        self.log(entry)

    def log_tool_call(self, tool_name: str, arguments: Dict) -> None:
        """Log a tool execution request."""
        entry = {
            "type": "tool_call",
            "tool_name": tool_name,
            "arguments": arguments
        }
        self.log(entry)

    def log_tool_result(self, tool_name: str, result: Dict) -> None:
        """Log a tool execution result."""
        entry = {
            "type": "tool_result",
            "tool_name": tool_name,
            **result
        }
        self.log(entry)

    def log_user_message(self, message: str) -> None:
        """Log a user message."""
        entry = {
            "type": "user_message",
            "content": message
        }
        self.log(entry)

    def log_agent_response(self, response: str) -> None:
        """Log the final agent response to the user."""
        entry = {
            "type": "agent_response",
            "content": response
        }
        self.log(entry)

    def close(self) -> None:
        """Close the log file."""
        if self._file:
            with self._lock:
                # Flush any remaining entries
                self._flush_buffer()
                self._file.close()
                self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
