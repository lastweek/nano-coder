"""Shared help rendering for slash commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from src.commands.registry import Command, CommandHelpSpec, CommandSubcommandHelp


def _build_fallback_help_spec(command: "Command") -> "CommandHelpSpec":
    """Build a minimal help spec if a command forgot to provide one."""
    from src.commands.registry import CommandHelpSpec

    usage = [f"/{command.name}" + (f" {command.args_description}" if command.args_description else "")]
    return CommandHelpSpec(summary=command.description, usage=usage)


def _get_help_spec(command: "Command") -> "CommandHelpSpec":
    """Return the configured help spec or a minimal fallback."""
    return command.help_spec or _build_fallback_help_spec(command)


def _find_subcommand_entries(command: "Command", subcommand: str) -> List["CommandSubcommandHelp"]:
    """Find exact or grouped subcommand help entries."""
    spec = _get_help_spec(command)
    target = subcommand.strip().lower()
    if not target:
        return []

    exact_matches = [entry for entry in spec.subcommands if entry.name.lower() == target]
    if exact_matches:
        return exact_matches

    return [entry for entry in spec.subcommands if entry.name.lower().startswith(f"{target} ")]


def _print_usage(console: Console, usage_lines: List[str]) -> None:
    """Render a usage block."""
    console.print("[bold]Usage[/bold]")
    for line in usage_lines:
        text = Text("  ")
        text.append(line, style="cyan")
        console.print(text)


def _print_examples(console: Console, examples: List[str]) -> None:
    """Render an examples block."""
    if not examples:
        return

    console.print()
    console.print("[bold]Examples[/bold]")
    for example in examples:
        text = Text("  ")
        text.append(example, style="cyan")
        console.print(text)


def _print_notes(console: Console, notes: List[str]) -> None:
    """Render a notes block."""
    if not notes:
        return

    console.print()
    console.print("[bold]Notes[/bold]")
    for note in notes:
        console.print(f"  [dim]{note}[/dim]")


def render_command_help(console: Console, command: "Command", subcommand: str | None = None) -> None:
    """Render command-level or targeted subcommand help."""
    spec = _get_help_spec(command)

    if subcommand:
        entries = _find_subcommand_entries(command, subcommand)
        if not entries:
            render_unknown_subcommand(console, command, subcommand)
            return

        title_suffix = f" ({subcommand})"
        console.print(Panel(spec.summary, title=f"Command: /{command.name}{title_suffix}", border_style="cyan"))
        for index, entry in enumerate(entries):
            if index:
                console.print()
            console.print(f"[bold]{entry.name}[/bold]")
            _print_usage(console, [entry.usage])
            console.print()
            console.print(entry.description)
            _print_examples(console, entry.examples)
            _print_notes(console, entry.notes)
        return

    console.print(Panel(spec.summary, title=f"Command: /{command.name}", border_style="cyan"))
    _print_usage(console, spec.usage)

    if spec.subcommands:
        console.print()
        table = Table(title="Subcommands", show_header=True, header_style="bold cyan")
        table.add_column("Subcommand", style="green", width=16)
        table.add_column("Usage", style="cyan", width=30)
        table.add_column("Description", style="white")
        for entry in spec.subcommands:
            table.add_row(entry.name, entry.usage, entry.description)
        console.print(table)

    _print_examples(console, spec.examples)
    _print_notes(console, spec.notes)


def render_unknown_subcommand(console: Console, command: "Command", subcommand: str) -> None:
    """Render a standard unknown-subcommand error followed by the full manual."""
    console.print(f"[red]Unknown /{command.name} subcommand: {subcommand}[/red]")
    render_command_help(console, command)
