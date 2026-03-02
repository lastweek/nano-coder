"""Slash commands for nano-coder."""

from src.commands.registry import CommandRegistry, Command
from src.commands import builtin

__all__ = ['CommandRegistry', 'Command', 'builtin', 'register_all']



