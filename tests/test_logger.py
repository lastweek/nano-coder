"""Tests for SessionLogger."""

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

from src.logger import SessionLogger


class TestSessionLogger:
    """Test SessionLogger functionality."""

    def _build_logger(self, temp_dir, **kwargs):
        logger = SessionLogger(str(uuid.uuid4()), log_dir=str(temp_dir), enabled=True, **kwargs)
        logger.start_session(
            cwd=str(temp_dir),
            provider="openai",
            model="gpt-4.1",
            base_url="https://example.invalid/v1",
            streaming_enabled=True,
        )
        return logger

    def test_does_not_create_session_without_writes(self, temp_dir):
        """No session directory should be created if nothing is written."""
        logger = self._build_logger(temp_dir)
        logger.close()

        assert list(Path(temp_dir).glob("session-*")) == []
        assert not (Path(temp_dir) / "latest-session").exists()
        assert not (Path(temp_dir) / "latest.log").exists()

    def test_creates_session_directory_and_manifest(self, temp_dir):
        """A real write should create session.json, llm.log, and events.jsonl."""
        logger = self._build_logger(temp_dir)
        turn_id = logger.start_turn(raw_user_input="hello", normalized_user_input="hello")
        logger.finish_turn(turn_id, "world", [], status="completed")
        logger.close()

        session_dirs = list(Path(temp_dir).glob("session-*"))
        assert len(session_dirs) == 1
        session_dir = session_dirs[0]
        assert (session_dir / "session.json").exists()
        assert (session_dir / "llm.log").exists()
        assert (session_dir / "events.jsonl").exists()
        assert (session_dir / "artifacts").exists()
        assert (Path(temp_dir) / "latest-session").exists()
        assert (Path(temp_dir) / "latest.log").exists()

        session = json.loads((session_dir / "session.json").read_text())
        assert session["status"] == "completed"
        assert session["turn_count"] == 1
        assert session["llm_log"] == "llm.log"
        assert session["events_log"] == "events.jsonl"
        assert session["artifacts_dir"] == "artifacts"

    def test_logs_llm_request_and_response_to_llm_log(self, temp_dir):
        """llm.log should contain full JSON request/response timeline blocks."""
        logger = self._build_logger(temp_dir)
        turn_id = logger.start_turn(raw_user_input="hello", normalized_user_input="hello")

        logger.log_llm_request(
            turn_id=turn_id,
            iteration=0,
            provider="openai",
            model="gpt-4.1",
            stream=False,
            request_payload={
                "model": "gpt-4.1",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "hello"},
                ],
                "tools": [{"type": "function", "function": {"name": "read_file"}}],
                "stream": False,
            },
        )
        logger.log_llm_response(
            turn_id=turn_id,
            iteration=0,
            provider="openai",
            model="gpt-4.1",
            stream=False,
            response_payload={
                "id": "resp_1",
                "object": "chat.completion",
                "model": "gpt-4.1",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Hi there", "tool_calls": None},
                    }
                ],
                "usage": None,
            },
            metrics={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15, "cached_tokens": 8},
        )
        logger.finish_turn(
            turn_id,
            "Hi there",
            [
                SimpleNamespace(
                    prompt_tokens=12,
                    completion_tokens=3,
                    total_tokens=15,
                    cached_tokens=8,
                )
            ],
        )
        logger.close()

        session_dir = next(Path(temp_dir).glob("session-*"))
        llm_log = (session_dir / "llm.log").read_text()
        assert "STEP 0001 | SESSION START" in llm_log
        assert "STEP 0002 | TURN 0001 | TURN START" in llm_log
        assert "STEP 0003 | TURN 0001 | ITERATION 01 | LLM REQUEST | STREAM=false" in llm_log
        assert "REQUEST JSON" in llm_log
        assert "\"messages\"" in llm_log
        assert "\"tools\"" in llm_log
        assert "STEP 0004 | TURN 0001 | ITERATION 01 | LLM RESPONSE | STREAM=false" in llm_log
        assert "RESPONSE JSON" in llm_log
        assert "Hi there" in llm_log
        assert "\"prompt_tokens\": 12" in llm_log
        assert "STEP 0005 | TURN 0001 | TURN END" in llm_log
        assert "STEP 0006 | SESSION END" in llm_log

    def test_logs_structured_events_and_spills_large_payloads(self, temp_dir):
        """Tool and skill events should be inline in llm.log and structured in events."""
        logger = self._build_logger(temp_dir)
        turn_id = logger.start_turn(raw_user_input="hello", normalized_user_input="hello")

        logger.log_skill_event(turn_id, "preload", skill_name="pdf", reason="explicit")
        logger.log_tool_call(
            turn_id=turn_id,
            iteration=0,
            tool_name="read_file",
            arguments={"file_path": "README.md"},
            tool_call_id="call_1",
        )
        logger.log_tool_result(
            turn_id=turn_id,
            iteration=0,
            tool_name="read_file",
            result={"output": "x" * 9000},
            tool_call_id="call_1",
        )
        logger.log_error(turn_id=turn_id, phase="agent.run", message="boom", details={"kind": "ValueError"})
        logger.finish_turn(turn_id, "done", [], status="error", error={"message": "boom"})
        logger.close(status="error")

        session_dir = next(Path(temp_dir).glob("session-*"))
        events = [
            json.loads(line)
            for line in (session_dir / "events.jsonl").read_text().splitlines()
            if line.strip()
        ]
        kinds = [event["kind"] for event in events]
        assert "session_started" in kinds
        assert "turn_started" in kinds
        assert "skill_event" in kinds
        assert "tool_call" in kinds
        assert "tool_result" in kinds
        assert "error" in kinds
        assert "turn_completed" in kinds
        assert "session_completed" in kinds

        tool_result = next(event for event in events if event["kind"] == "tool_result")
        assert tool_result["tool_name"] == "read_file"
        assert "payload_path" in tool_result
        assert "timeline_seq" in tool_result
        artifact_path = session_dir / tool_result["payload_path"]
        assert artifact_path.exists()
        assert artifact_path.read_text().startswith("{")

        llm_log = (session_dir / "llm.log").read_text()
        assert "SKILL EVENT" in llm_log
        assert "TOOL CALL" in llm_log
        assert "TOOL RESULT" in llm_log
        assert "\"file_path\": \"README.md\"" in llm_log
        assert "\"reason\": \"explicit\"" in llm_log

    def test_async_mode_preserves_valid_output(self, temp_dir):
        """Async logging should still produce readable files."""
        logger = self._build_logger(temp_dir, async_mode=True)
        turn_id = logger.start_turn(raw_user_input="hi", normalized_user_input="hi")
        logger.log_llm_request(
            turn_id=turn_id,
            iteration=0,
            provider="openai",
            model="gpt-4.1",
            stream=True,
            request_payload={
                "model": "gpt-4.1",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        logger.log_llm_response(
            turn_id=turn_id,
            iteration=0,
            provider="openai",
            model="gpt-4.1",
            stream=True,
            response_payload={
                "object": "chat.completion.stream.reconstructed",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hello"}}],
            },
            metrics={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7, "cached_tokens": 0},
        )
        logger.finish_turn(turn_id, "hello", [], status="completed")
        logger.close()

        session_dir = next(Path(temp_dir).glob("session-*"))
        llm_log = (session_dir / "llm.log").read_text()
        assert "LLM REQUEST" in llm_log
        assert "LLM RESPONSE" in llm_log

        events = [
            json.loads(line)
            for line in (session_dir / "events.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert events[0]["kind"] == "session_started"
        assert events[-1]["kind"] == "session_completed"
