"""Test streaming functionality."""

import pytest
from unittest.mock import Mock, patch
from src.llm import LLMClient
from src.agent import Agent
from src.context import Context
from src.tools import ToolRegistry


class TestStreaming:
    """Test streaming functionality."""

    def test_llm_chat_stream_yields_tokens(self):
        """Test that chat_stream yields tokens."""
        llm_client = LLMClient(provider="ollama")

        # Mock the streaming response
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(role="assistant", content="Hello", tool_calls=None), finish_reason=None)]),
            Mock(choices=[Mock(delta=Mock(role=None, content=" world", tool_calls=None), finish_reason=None)]),
            Mock(choices=[Mock(delta=Mock(role=None, content="!", tool_calls=None), finish_reason="stop")]),
        ]

        with patch.object(llm_client.client.chat.completions, 'create', return_value=iter(mock_chunks)):
            tokens = list(llm_client.chat_stream([{"role": "user", "content": "test"}]))

            # Should yield delta tokens (3 deltas + role + finish_reason = 5 total)
            assert len(tokens) == 5
            assert tokens[0]["role"] == "assistant"
            assert tokens[1]["delta"] == "Hello"
            assert tokens[2]["delta"] == " world"
            assert tokens[3]["delta"] == "!"
            assert tokens[4]["finish_reason"] == "stop"

    def test_llm_chat_stream_with_logger(self, temp_dir):
        """Test that chat_stream logs properly."""
        from src.logger import ChatLogger

        # Create logger
        logger = ChatLogger("test-session", log_dir=str(temp_dir), enabled=True)
        llm_client = LLMClient(provider="ollama", logger=logger)

        # Mock the streaming response
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(role="assistant", content="Hi", tool_calls=None), finish_reason=None)]),
            Mock(choices=[Mock(delta=Mock(role=None, content=" there", tool_calls=None), finish_reason="stop")]),
        ]

        with patch.object(llm_client.client.chat.completions, 'create', return_value=iter(mock_chunks)):
            list(llm_client.chat_stream([{"role": "user", "content": "hello"}]))

        # Close logger to flush
        logger.close()

        # Verify logging happened
        from pathlib import Path
        log_files = [f for f in Path(temp_dir).glob("*.jsonl") if f.name != "latest.jsonl"]
        assert len(log_files) == 1

        import json
        content = log_files[0].read_text()
        entries = [json.loads(line) for line in content.strip().split("\n")]
        assert any(e["type"] == "llm_request" for e in entries)
        assert any(e["type"] == "llm_response" for e in entries)

    def test_agent_run_stream_yields_tokens(self, temp_dir):
        """Test that agent.run_stream yields tokens."""
        context = Context.create(cwd=str(temp_dir))
        tools = ToolRegistry()
        llm_client = LLMClient(provider="ollama")
        agent = Agent(llm_client, tools, context)

        # Mock streaming response
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(role="assistant", content="Test", tool_calls=None), finish_reason=None)]),
            Mock(choices=[Mock(delta=Mock(role=None, content=" response", tool_calls=None), finish_reason="stop")]),
        ]

        # Mock non-streaming call for tool check
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None

        with patch.object(llm_client.client.chat.completions, 'create') as mock_create:
            # First call is streaming, second is non-streaming for tool check
            mock_create.side_effect = [iter(mock_chunks), mock_response]

            tokens = list(agent.run_stream("hello"))

            assert len(tokens) == 2
            assert "".join(tokens) == "Test response"
