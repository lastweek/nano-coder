"""LLM integration for Nano-Coder."""

import os
from typing import Optional, List, Dict, TYPE_CHECKING
from openai import OpenAI

if TYPE_CHECKING:
    from src.logger import ChatLogger


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
        provider = provider or os.environ.get("LLM_PROVIDER", "openai").lower()

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
            base_url = os.environ.get("BASE_URL", "")
            # Set default URLs for known providers
            if not base_url and provider == "ollama":
                base_url = "http://localhost:11434/v1"

        # Get model name with provider defaults
        if model is None:
            model = os.environ.get("MODEL")
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
        """Get API key for the given provider."""
        env_var = self._get_api_key_env_var(provider)
        return os.environ.get(env_var)

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

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
        """Send messages and get response with tool calls.

        Args:
            messages: Conversation history in OpenAI format
            tools: Optional list of tool schemas for function calling

        Returns:
            Response dict with role, content, and optional tool_calls
        """
        # Log request (logger handles disabled state internally)
        if self.logger:
            self.logger.log_llm_request(messages, tools, self.model, self.provider)

        kwargs = {"model": self.model, "messages": messages}

        if tools:
            kwargs["tools"] = tools
            # Enable parallel function calling
            kwargs["parallel_tool_calls"] = True

        response = self.client.chat.completions.create(**kwargs)

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

        return result
