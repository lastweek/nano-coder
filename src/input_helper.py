"""Input handling with prompt_toolkit for better UX."""

from typing import Optional
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from pathlib import Path


class InputHelper:
    """Enhanced input with bash-like editing."""

    def __init__(self, history_file: Optional[Path] = None):
        """Initialize input session with history support.

        Args:
            history_file: Path to command history file (default: ~/.nano-coder-history)
        """
        if history_file is None:
            history_file = Path.home() / ".nano-coder-history"

        self.history = FileHistory(str(history_file))
        self.session = PromptSession(
            history=self.history,
            enable_history_search=True,
            # Enable mouse support for better UX
            mouse_support=True,
            # Use sensible defaults
            enable_suspend=True,  # Allow Ctrl+Z to suspend
        )

    def get_input(self, prompt: str = "> ") -> str:
        """Get user input with enhanced editing.

        Args:
            prompt: Prompt string to display

        Returns:
            User input as string
        """
        import sys

        # Use HTML for colored prompt
        formatted_prompt = HTML(f"<style fg='green'>{prompt}</style>")

        try:
            result = self.session.prompt(
                formatted_prompt,
                # Enable common key bindings
                enable_open_in_editor=True,  # Ctrl+X Ctrl+E to edit in editor
            )
            # CRITICAL: Print newline to clear the prompt line for subsequent output
            # This ensures Rich's console output starts on a fresh line
            sys.stdout.write("\n")
            sys.stdout.flush()
            return result
        except (EOFError, KeyboardInterrupt):
            # User wants to exit
            raise EOFError
