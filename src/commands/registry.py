"""Slash command system for nano-coder meta-commands."""

from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass
from rich.console import Console


@dataclass
class Command:
    """Metadata for a slash command."""

    name: str
    description: str
    handler: Callable[..., None]
    args_description: Optional[str] = None
    short_desc: str = ""  # Short description for menu (50 chars max)


class CommandRegistry:
    """Registry for slash commands."""

    def __init__(self):
        self._commands: Dict[str, Command] = {}

    def register(
        self,
        name: str,
        description: str,
        args_description: str = None,
        short_desc: str = ""
    ):
        """Decorator to register a command.

        Args:
            name: Command name (without the /)
            description: What the command does
            args_description: Optional description of arguments
            short_desc: Short description for menu (50 chars max)

        Example:
            @registry.register("echo", "Print a message", short_desc="Echo text")
            def cmd_echo(console: Console, args: str, context: Any):
                console.print(args)
        """
        def decorator(func: Callable[..., None]):
            self._commands[name] = Command(
                name=name,
                description=description,
                handler=func,
                args_description=args_description,
                short_desc=short_desc
            )
            return func
        return decorator

    def execute(self, command_line: str, console: Console, context: Any) -> bool:
        """Execute a command line.

        Args:
            command_line: Full command line (e.g., "/tool" or "/mcp deepwiki")
            console: Rich console for output
            context: Application context (agent, mcp_manager, etc.)

        Returns:
            True if command was executed, False if not a command
        """
        stripped_command = command_line.lstrip()
        if not stripped_command.startswith("/"):
            return False

        parts = stripped_command[1:].split(maxsplit=1)
        command_name = parts[0]
        args = parts[1] if len(parts) > 1 else ""

        command = self._commands.get(command_name)
        if not command:
            console.print(f"[red]Unknown command: /{command_name}[/red]")
            console.print(f"[dim]Type /help for available commands[/dim]")
            return True  # It was a command, just unknown

        try:
            command.handler(console, args, context)
        except Exception as e:
            console.print(f"[red]Error executing /{command_name}: {e}[/red]")

        return True

    def list_commands(self) -> List[Command]:
        """Get all registered commands."""
        return list(self._commands.values())

    def get_command_names(self) -> List[str]:
        """Get all command names."""
        return list(self._commands.keys())
