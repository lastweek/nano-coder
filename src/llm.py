"""LLM integration for Nano-Coder."""

import os
from typing import Optional, List, Dict, TYPE_CHECKING
from openai import OpenAI
from src.config import config

if TYPE_CHECKING:
    from src.logger import ChatLogger
    from src.metrics import LLMMetrics


class LLMClient:
    """Handles communication with OpenAI-compatible APIs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
        logger: Optional['ChatLogger'] = None,
    ):
        """Initialize the LLM client.

        Args:
            api_key: API key (defaults to provider-specific env var)
            model: Model name (defaults to MODEL env var or provider default)
            base_url: Custom base URL (defaults to BASE_URL env var)
            provider: Provider name for env var lookup (defaults to LLM_PROVIDER env var)
            logger: Optional ChatLogger instance for logging requests/responses
        """
        # Determine provider
        provider = provider or config.llm.provider

        # Get API key based on provider
        if api_key is None:
            api_key = self._get_api_key(provider)

        if not api_key and provider not in ("ollama", "local"):
            raise ValueError(
                f"API key required for provider '{provider}'. "
                f"Set {self._get_api_key_env_var(provider)}"
            )

        # Get base URL
        if base_url is None:
            base_url = config.llm.base_url
            # Set default URLs for known providers
            if not base_url and provider == "ollama":
                base_url = "http://localhost:11434/v1"

        # Get model name with provider defaults
        if model is None:
            model = config.llm.model
            if not model:
                model = self._get_default_model(provider)

        self.model = model
        self.provider = provider
        self.logger = logger

        # Initialize OpenAI client (works with any OpenAI-compatible API)
        client_kwargs = {"api_key": api_key or "not-needed"}  # Ollama doesn't need API key
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for the given provider.

        Priority:
        1. Environment variable (for override)
        2. config.llm.api_key (from config.yaml)
        3. None (not set)
        """
        # First check environment variable (allows override)
        env_var = self._get_api_key_env_var(provider)
        env_key = os.environ.get(env_var)

        if env_key:
            return env_key

        # Fall back to config.yaml
        return config.llm.api_key

    def _get_api_key_env_var(self, provider: str) -> str:
        """Get the environment variable name for API key."""
        key_map = {
            "openai": "OPENAI_API_KEY",
            "azure": "AZURE_API_KEY",
            "custom": "CUSTOM_API_KEY",
        }
        return key_map.get(provider, "API_KEY")

    def _get_default_model(self, provider: str) -> str:
        """Get default model for the given provider."""
        defaults = {
            "openai": "gpt-4",
            "azure": "gpt-4",
            "ollama": "llama2",
            "local": "llama2",
            "custom": "gpt-4",
        }
        return defaults.get(provider, "gpt-4")

    def _estimate_prompt_tokens(self, messages: List[Dict]) -> int:
        """Estimate prompt tokens from messages.

        Uses a rough character-based estimate (char_count / 4) as a fallback
        when the API doesn't return usage information. This is approximate
        but sufficient for display purposes.

        Args:
            messages: List of message dicts with role and content

        Returns:
            Estimated token count
        """
        import json

        # Serialize messages to JSON to get a rough character count
        # This includes role, content, and JSON structure overhead
        text = json.dumps(messages)

        # Rough estimate: ~4 characters per token for English text
        # This is not exact but provides a reasonable approximation
        return max(1, len(text) // 4)

    def _build_chat_kwargs(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict:
        """Build kwargs for chat.completions.create API call.

        Args:
            messages: Message list
            tools: Optional tool schemas
            stream: Whether to stream response

        Returns:
            kwargs dict for API call
        """
        kwargs = {"model": self.model, "messages": messages, "stream": stream}

        if tools:
            kwargs["tools"] = tools
            kwargs["parallel_tool_calls"] = True

        return kwargs

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> tuple:
        """Send messages and get response with tool calls.

        Args:
            messages: Conversation history in OpenAI format
            tools: Optional list of tool schemas for function calling

        Returns:
            Tuple of (response_dict, metrics)
        """
        from src.metrics import LLMMetrics

        metrics = LLMMetrics(
            model=self.model,
            provider=self.provider,
            request_type="non-streaming"
        )

        # Log request (logger handles disabled state internally)
        if self.logger:
            self.logger.log_llm_request(messages, tools, self.model, self.provider)

        kwargs = self._build_chat_kwargs(messages, tools, stream=False)

        response = self.client.chat.completions.create(**kwargs)

        # Extract usage information
        if hasattr(response, 'usage') and response.usage:
            metrics.prompt_tokens = response.usage.prompt_tokens
            metrics.completion_tokens = response.usage.completion_tokens
            metrics.total_tokens = response.usage.total_tokens

            # Extract cached tokens if available
            if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                cached = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0)
                metrics.cached_tokens = cached or 0
        else:
            # Fallback: estimate tokens when API doesn't return usage
            metrics.prompt_tokens = self._estimate_prompt_tokens(messages)
            # For non-streaming, we can't accurately count completion tokens without usage
            # Set them to 0 and let total_tokens be just prompt_tokens
            metrics.completion_tokens = 0
            metrics.total_tokens = metrics.prompt_tokens

        metrics.finish()

        message = response.choices[0].message

        result = {
            "role": message.role,
            "content": message.content or "",
        }

        # Extract tool calls if present
        if message.tool_calls:
            result["tool_calls"] = []
            for tool_call in message.tool_calls:
                result["tool_calls"].append({
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                })

        # Log response
        if self.logger:
            self.logger.log_llm_response(result)

        return result, metrics

    def chat_stream(self, messages: List[Dict], tools: Optional[List[Dict]] = None):
        """Stream chat completion responses token-by-token.

        Args:
            messages: Conversation history in OpenAI format
            tools: Optional list of tool schemas for function calling

        Yields:
            Dicts with: {"delta": str, "role": str, "tool_calls": Optional[List], "finish_reason": str}
        """
        from src.metrics import LLMMetrics

        # Log request
        if self.logger:
            self.logger.log_llm_request(messages, tools, self.model, self.provider)

        kwargs = self._build_chat_kwargs(messages, tools, stream=True)

        # Create metrics tracker
        self._current_metrics = LLMMetrics(
            model=self.model,
            provider=self.provider,
            request_type="streaming"
        )

        # Start timing right before API call for accurate TTFT
        from time import perf_counter
        self._current_metrics.start_time = perf_counter()

        response = self.client.chat.completions.create(**kwargs)

        # Collect for logging
        full_content = ""
        full_role = "assistant"
        accumulated_tool_calls = {}
        first_content_token = True

        for chunk in response:
            choice = chunk.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason

            # Extract role (usually in first chunk)
            if delta.role:
                full_role = delta.role
                yield {"role": delta.role}

            # Extract content tokens
            if delta.content:
                full_content += delta.content

                # Mark first token arrival
                if first_content_token:
                    self._current_metrics.mark_first_token()
                    first_content_token = False

                # Record token timestamp
                self._current_metrics.add_token_timestamp()

                yield {"delta": delta.content, "role": "assistant"}

            # Extract tool calls (streamed in chunks)
            if delta.tool_calls:
                for tool_call_chunk in delta.tool_calls:
                    index = tool_call_chunk.index

                    if index not in accumulated_tool_calls:
                        accumulated_tool_calls[index] = {
                            "id": tool_call_chunk.id or f"call_{index}",
                            "name": "",
                            "arguments": ""
                        }

                    tc = accumulated_tool_calls[index]

                    if tool_call_chunk.function:
                        if tool_call_chunk.function.name:
                            tc["name"] = tool_call_chunk.function.name
                        if tool_call_chunk.function.arguments:
                            tc["arguments"] += tool_call_chunk.function.arguments

            # Signal when complete
            if finish_reason:
                yield {"finish_reason": finish_reason}

                # Extract usage from final chunk if available
                if hasattr(chunk, 'usage') and chunk.usage:
                    self._current_metrics.prompt_tokens = chunk.usage.prompt_tokens
                    self._current_metrics.completion_tokens = chunk.usage.completion_tokens
                    self._current_metrics.total_tokens = chunk.usage.total_tokens

                    # Extract cached tokens if available
                    if hasattr(chunk.usage, 'prompt_tokens_details') and chunk.usage.prompt_tokens_details:
                        cached = getattr(chunk.usage.prompt_tokens_details, 'cached_tokens', 0)
                        self._current_metrics.cached_tokens = cached or 0

        # Always call finish() to record end time, regardless of usage availability
        self._current_metrics.finish()

        # If no usage info was found in the stream, estimate it
        if self._current_metrics.prompt_tokens == 0:
            self._current_metrics.completion_tokens = self._current_metrics.token_count
            self._current_metrics.prompt_tokens = self._estimate_prompt_tokens(messages)
            self._current_metrics.total_tokens = self._current_metrics.prompt_tokens + self._current_metrics.completion_tokens

        # Log complete response
        tool_calls_list = list(accumulated_tool_calls.values())
        tool_calls_list.sort(key=lambda x: x["id"])  # Sort by id to maintain order

        if self.logger:
            self.logger.log_llm_response({
                "role": full_role,
                "content": full_content,
                "tool_calls": tool_calls_list if tool_calls_list else None
            })

        # Store tool calls for retrieval by caller (eliminates need for duplicate API call)
        self._last_stream_tool_calls = tool_calls_list if tool_calls_list else []

    def get_stream_metrics(self) -> Optional['LLMMetrics']:
        """Get metrics from the most recent stream request."""
        return getattr(self, '_current_metrics', None)

    def get_stream_tool_calls(self) -> List[Dict]:
        """Get tool calls from the most recent stream request.

        Returns:
            List of tool call dicts with id, name, arguments
        """
        return getattr(self, '_last_stream_tool_calls', [])
