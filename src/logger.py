"""Chat completion logging for Nano-Coder."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from threading import Lock, Thread
from queue import Queue
from src.config import config


class ChatLogger:
    """Logs chat completions and tool executions to JSONL files."""

    def __init__(self, session_id: str, log_dir: Optional[str] = None, enabled: Optional[bool] = None, buffer_size: Optional[int] = None, async_mode: Optional[bool] = None):
        """Initialize the logger.

        Args:
            session_id: Unique session identifier
            log_dir: Directory to store log files (defaults to config.logging.log_dir)
            enabled: Whether logging is enabled (defaults to config.logging.enabled)
            buffer_size: Number of log entries to buffer before flushing (defaults to config.logging.buffer_size)
            async_mode: If True, use background thread for non-blocking logging (defaults to config.logging.async_mode)
        """
        self.session_id = session_id

        # Use config defaults if not specified
        if log_dir is None:
            log_dir = config.logging.log_dir
        if enabled is None:
            # Check env var directly for backward compatibility
            import os
            env_value = os.environ.get("ENABLE_LOGGING", "").lower()
            if env_value == "false":
                enabled = False
            elif env_value == "true":
                enabled = True
            else:
                enabled = config.logging.enabled
        if buffer_size is None:
            buffer_size = config.logging.buffer_size
        if async_mode is None:
            async_mode = config.logging.async_mode

        self.log_dir = Path(log_dir)
        self.enabled = enabled
        self.async_mode = async_mode

        self._lock = Lock()
        self._file = None
        self._buffer = []
        self._buffer_size = buffer_size

        # Async mode: queue and writer thread
        self._queue = None
        self._writer_thread = None
        self._stop_requested = False

        if self.enabled and self.async_mode:
            self._queue = Queue()
            self._start_writer_thread()

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

    def _ensure_file_open(self) -> None:
        """Open log file on first write (lazy initialization)."""
        if self.enabled and self._file is None:
            self._open_log_file()

    def _start_writer_thread(self):
        """Start background thread for async logging."""
        def writer():
            while not self._stop_requested:
                try:
                    # Get entry with timeout to allow checking stop_requested
                    entry = self._queue.get(timeout=0.1)
                    if entry is None:  # Poison pill
                        break
                    self._ensure_file_open()
                    self._write_entry(entry)
                except (Queue.Empty, TimeoutError):
                    continue  # Timeout or empty queue, continue loop
                except Exception as e:
                    # Log unexpected errors but continue running
                    import logging
                    logging.warning(f"Writer thread error: {e}")

        self._writer_thread = Thread(target=writer, daemon=True)
        self._writer_thread.start()

    def _write_entry(self, entry: Dict[str, Any]) -> None:
        """Write a single log entry (called by sync or async mode)."""
        if not self.enabled:
            return

        self._ensure_file_open()

        if self._file is None:
            return

        with self._lock:
            entry["timestamp"] = datetime.now().isoformat()
            entry["session_id"] = self.session_id
            self._buffer.append(json.dumps(entry))

            # Flush when buffer reaches threshold
            if len(self._buffer) >= self._buffer_size:
                self._flush_buffer()

    def log(self, entry: Dict[str, Any]) -> None:
        """Write a log entry to the file.

        Args:
            entry: Dictionary containing log data
        """
        if not self.enabled:
            return

        self._ensure_file_open()

        if self.async_mode:
            # Non-blocking: queue for background thread
            self._queue.put(entry)
        else:
            # Blocking: write immediately
            self._write_entry(entry)

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
        """Close the log file and cleanup resources."""
        # Stop writer thread if in async mode
        if self.async_mode and self._writer_thread:
            self._stop_requested = True
            # Send poison pill to unblock the thread
            self._queue.put(None)
            # Wait for thread to finish (with timeout)
            self._writer_thread.join(timeout=1.0)
            self._writer_thread = None

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
