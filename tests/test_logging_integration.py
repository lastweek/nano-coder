"""Test logging integration between Agent and LLMClient."""

import pytest
import tempfile
from pathlib import Path
from src.agent import Agent
from src.llm import LLMClient
from src.context import Context
from src.tools import ToolRegistry
from tools.read import ReadTool
from unittest.mock import Mock, patch


class TestLoggingIntegration:
    """Test that logging is properly integrated across components."""

    def test_agent_shares_logger_with_llm_client(self, temp_dir):
        """Test that Agent injects its logger into LLMClient."""
        # Create components
        context = Context.create(cwd=str(temp_dir))
        tools = ToolRegistry()
        tools.register(ReadTool())

        # Create LLMClient without logger (use ollama to avoid API key requirement)
        llm_client = LLMClient(provider="ollama")

        # Create Agent (should share its logger with LLMClient)
        agent = Agent(llm_client, tools, context)

        # Verify logger is shared
        assert llm_client.logger is not None
        assert llm_client.logger is agent.logger
        assert llm_client.logger.session_id == context.session_id

    def test_llm_client_logs_requests_and_responses(self, temp_dir):
        """Test that LLMClient actually logs when called through Agent."""
        # Create components
        context = Context.create(cwd=str(temp_dir))
        tools = ToolRegistry()
        tools.register(ReadTool())

        # Create LLMClient (use ollama to avoid API key requirement)
        llm_client = LLMClient(provider="ollama")

        # Create Agent with logger
        agent = Agent(llm_client, tools, context)

        # Get the session_id to find the specific log file
        session_id = context.session_id[:8]  # First 8 chars as used in filename

        # Mock the chat.completions.create to avoid real API call
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.tool_calls = None

        with patch.object(llm_client.client.chat.completions, 'create', return_value=mock_response):
            # Call chat through LLMClient
            result = llm_client.chat([{"role": "user", "content": "test"}])

            # Close logger to flush buffered entries
            agent.logger.close()

            # Get the log directory from the logger
            log_dir = agent.logger.log_dir

            # Find the specific log file for this session
            log_files = list(log_dir.glob(f"session-{session_id}-*.jsonl"))
            assert len(log_files) == 1, f"Expected 1 log file for session {session_id}, found {len(log_files)}"

            # Check log contains llm_request and llm_response
            content = log_files[0].read_text()
            assert '"type": "llm_request"' in content
            assert '"type": "llm_response"' in content

            # Clean up
            log_files[0].unlink()
            latest_link = log_dir / "latest.jsonl"
            if latest_link.exists():
                latest_link.unlink()
            if log_dir.exists() and not list(log_dir.iterdir()):
                log_dir.rmdir()
