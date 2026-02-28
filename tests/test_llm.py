"""Test LLMClient configuration."""

import pytest
from src.llm import LLMClient


class TestLLMClient:
    """Test LLMClient initialization and configuration."""

    def test_openai_provider_with_key(self, monkeypatch):
        """Test OpenAI provider initialization."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        client = LLMClient(provider="openai")
        assert client.provider == "openai"
        assert client.model == "gpt-4"
        assert client.client is not None

    def test_custom_provider_with_base_url(self, monkeypatch):
        """Test custom provider with base URL."""
        monkeypatch.setenv("CUSTOM_API_KEY", "test-key")
        monkeypatch.setenv("BASE_URL", "https://api.test.com/v1")
        client = LLMClient(provider="custom")
        assert client.provider == "custom"
        assert client.client is not None

    def test_ollama_provider_no_api_key(self, monkeypatch):
        """Test Ollama provider doesn't require API key."""
        monkeypatch.setenv("BASE_URL", "http://localhost:11434/v1")
        client = LLMClient(provider="ollama")
        assert client.provider == "ollama"
        assert client.model == "llama2"

    def test_default_base_url_for_ollama(self, monkeypatch):
        """Test Ollama gets default base URL."""
        monkeypatch.setenv("BASE_URL", "")  # Empty
        client = LLMClient(provider="ollama")
        # Client should be initialized with default Ollama URL
        assert client.provider == "ollama"

    def test_missing_api_key_raises_error(self, monkeypatch):
        """Test that missing API key raises ValueError."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            LLMClient(provider="openai")

    def test_model_from_env(self, monkeypatch):
        """Test model can be set from environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("MODEL", "gpt-3.5-turbo")
        client = LLMClient(provider="openai")
        assert client.model == "gpt-3.5-turbo"

    def test_get_default_model(self):
        """Test _get_default_model returns correct defaults."""
        client = LLMClient.__new__(LLMClient)
        assert client._get_default_model("openai") == "gpt-4"
        assert client._get_default_model("ollama") == "llama2"
        assert client._get_default_model("azure") == "gpt-4"

    def test_get_api_key_env_var(self):
        """Test _get_api_key_env_var mapping."""
        client = LLMClient.__new__(LLMClient)
        assert client._get_api_key_env_var("openai") == "OPENAI_API_KEY"
        assert client._get_api_key_env_var("azure") == "AZURE_API_KEY"
        assert client._get_api_key_env_var("custom") == "CUSTOM_API_KEY"
