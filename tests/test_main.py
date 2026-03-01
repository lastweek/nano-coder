"""Test main.py functions."""

import pytest
import os
import sys
from pathlib import Path
from rich.console import Console

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_print_banner_no_error():
    """Test that print_banner runs without errors."""
    from src.main import print_banner

    console = Console()
    # Should not raise any exceptions
    print_banner(console)


def test_print_banner_with_context_window(monkeypatch):
    """Test banner displays with context_window set."""
    # Enable test mode
    monkeypatch.setenv("NANO_CODER_TEST", "true")

    # Set test values
    monkeypatch.setenv("LLM_PROVIDER", "test-provider")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://test.example.com")
    monkeypatch.setenv("LLM_CONTEXT_WINDOW", "128000")

    # Reload config to pick up env vars
    from src.config import Config
    Config.reload()

    # Import after reload to get fresh config
    from src.main import print_banner

    console = Console()
    # Should not raise any exceptions
    print_banner(console)


def test_print_banner_minimal_config(monkeypatch):
    """Test banner with minimal config."""
    # Enable test mode
    monkeypatch.setenv("NANO_CODER_TEST", "true")

    # Clear any existing LLM env vars
    for key in list(os.environ.keys()):
        if key.startswith(("LLM_", "API_KEY")):
            monkeypatch.delenv(key, raising=False)

    # Reload config to pick up changes
    from src.config import Config
    Config.reload()

    # Import after reload
    from src.main import print_banner

    console = Console()
    # Should not raise any exceptions even with minimal config
    print_banner(console)
