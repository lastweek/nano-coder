"""Input handling with prompt_toolkit for better UX."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, CompleteEvent, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import has_completions
from prompt_toolkit.output.base import Output
from prompt_toolkit.shortcuts import CompleteStyle


def _get_active_slash_fragment(document: Document) -> Optional[Tuple[str, int]]:
    """Get the active slash fragment before the cursor."""
    text = document.text_before_cursor
    if not text:
        return None

    token_start = len(text)
    while token_start > 0 and not text[token_start - 1].isspace():
        token_start -= 1

    token = text[token_start:]
    slash_index = token.rfind("/")
    if slash_index == -1:
        return None

    fragment = token[slash_index:]
    if not fragment:
        return None

    return fragment, -len(fragment)


class SlashCommandCompleter(Completer):
    """Prefix-based slash command completer."""

    def __init__(self, command_names: List[str], descriptions: dict):
        """Initialize slash command completer.

        Args:
            command_names: List of command names (without /)
            descriptions: Dict mapping command names to short descriptions
        """
        self.command_names = list(command_names)
        self.descriptions = dict(descriptions)

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent
    ):
        """Get completions for the active slash fragment."""
        active_fragment = _get_active_slash_fragment(document)
        if active_fragment is None:
            return []

        fragment, start_position = active_fragment
        query = fragment[1:].lower()

        for command_name in self.command_names:
            if query and not command_name.lower().startswith(query):
                continue

            description = self.descriptions.get(
                command_name,
                f"Run {command_name} command"
            )
            yield Completion(
                text=f"/{command_name}",
                start_position=start_position,
                display=f"/{command_name}",
                display_meta=description,
            )


class InputHelper:
    """Enhanced input with bash-like editing and command menu."""

    def __init__(
        self,
        history_file: Optional[Path] = None,
        command_names: Optional[List[str]] = None,
        command_descriptions: Optional[dict] = None,
        mouse_support: bool = False,
        input: Optional[Input] = None,
        output: Optional[Output] = None,
    ):
        """Initialize input session with history and auto-command completion.

        Args:
            history_file: Path to command history file (default: ~/.nano-coder-history)
            command_names: List of command names for tab completion (without /)
            command_descriptions: Dict mapping command names to short descriptions
            mouse_support: Whether prompt_toolkit mouse tracking is enabled
            input: Optional prompt_toolkit input for testing
            output: Optional prompt_toolkit output for testing
        """
        if history_file is None:
            history_file = Path.home() / ".nano-coder-history"

        self.history = FileHistory(str(history_file))
        self.command_names = list(command_names or [])
        self.command_descriptions = dict(command_descriptions or {})

        self.command_completer = self._build_command_completer()
        kb = KeyBindings()

        @kb.add("/")
        def _(event):
            """Insert slash and immediately open the command menu."""
            self._handle_slash(event.app.current_buffer)

        @kb.add("down", filter=has_completions)
        def _(event):
            """Move to the next command completion."""
            self._move_completion_next(event.app.current_buffer)

        @kb.add("up", filter=has_completions)
        def _(event):
            """Move to the previous command completion."""
            self._move_completion_previous(event.app.current_buffer)

        @kb.add("tab", filter=has_completions)
        def _(event):
            """Insert the selected command completion."""
            self._accept_completion(event.app.current_buffer)

        @kb.add("escape", filter=has_completions)
        def _(event):
            """Dismiss the command completion menu."""
            self._cancel_completion(event.app.current_buffer)

        self.session = PromptSession(
            history=self.history,
            enable_history_search=True,
            mouse_support=mouse_support,
            enable_suspend=True,
            completer=self.command_completer,
            key_bindings=kb,
            complete_in_thread=False,
            complete_while_typing=False,
            complete_style=CompleteStyle.COLUMN,
            reserve_space_for_menu=8,
            input=input,
            output=output,
        )
        self.session.default_buffer.on_text_changed += self._on_buffer_text_changed

    def _build_command_completer(self) -> Optional[SlashCommandCompleter]:
        """Build the current slash command completer."""
        if not self.command_names:
            return None

        return SlashCommandCompleter(
            command_names=self.command_names,
            descriptions=self.command_descriptions,
        )

    def _get_document_completions(self, document: Document) -> List[Completion]:
        """Get completions for the current document."""
        if self.command_completer is None:
            return []

        complete_event = CompleteEvent(completion_requested=True)
        return list(self.command_completer.get_completions(document, complete_event))

    def _refresh_completion_menu(self, buffer, *, select_first: bool) -> None:
        """Refresh slash command completions for the current buffer."""
        if self.command_completer is None:
            return

        if _get_active_slash_fragment(buffer.document) is None:
            buffer.cancel_completion()
            return

        completions = self._get_document_completions(buffer.document)
        if not completions:
            buffer.cancel_completion()
            return

        state = buffer._set_completions(completions)
        if state is None:
            return

        if select_first:
            state.go_to_index(0)

        buffer.complete_state = state

    def _on_buffer_text_changed(self, buffer) -> None:
        """Keep the slash command menu in sync with the current token."""
        self._refresh_completion_menu(
            buffer,
            select_first=buffer.complete_state is not None,
        )

    def _handle_slash(self, buffer) -> None:
        """Insert slash and open command completions for the new fragment."""
        buffer.insert_text("/")
        self._refresh_completion_menu(buffer, select_first=True)

    def _move_completion_next(self, buffer) -> None:
        """Select the next completion."""
        if buffer.complete_state is not None:
            buffer.complete_next()

    def _move_completion_previous(self, buffer) -> None:
        """Select the previous completion."""
        if buffer.complete_state is not None:
            buffer.complete_previous()

    def _accept_completion(self, buffer) -> None:
        """Apply the currently selected completion to the buffer."""
        state = buffer.complete_state
        if state is None or not state.completions:
            return

        completion = state.current_completion or state.completions[0]
        buffer.apply_completion(completion)

    def _cancel_completion(self, buffer) -> None:
        """Dismiss the current completion menu."""
        buffer.cancel_completion()

    def update_commands(
        self,
        command_names: List[str],
        command_descriptions: Optional[Dict[str, str]] = None,
    ) -> None:
        """Update command completer with new command list.

        Args:
            command_names: List of command names (without /)
            command_descriptions: Optional descriptions for the new commands
        """
        self.command_names = list(command_names or [])
        if command_descriptions is not None:
            self.command_descriptions = dict(command_descriptions)
        else:
            self.command_descriptions = {
                name: self.command_descriptions.get(name, f"Run {name} command")
                for name in self.command_names
            }

        self.command_completer = self._build_command_completer()
        self.session.completer = self.command_completer
        self._refresh_completion_menu(
            self.session.default_buffer,
            select_first=self.session.default_buffer.complete_state is not None,
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
