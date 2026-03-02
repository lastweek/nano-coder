"""Tool system for Nano-Coder."""

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING, Optional, List, Dict
from pathlib import Path

if TYPE_CHECKING:
    from src.context import Context


# Constants for message structure
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"

# Constants for request kind tracking
REQUEST_KIND_AGENT_TURN = "agent_turn"
REQUEST_KIND_CONTEXT_COMPACTION = "context_compaction"


@dataclass
class ToolResult:
    """Standardized tool output."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None


class Tool:
    """Base class for all agent tools."""

    name: str = ""
    description: str = ""
    parameters: dict = field(default_factory=dict)

    def execute(self, context: "Context", **kwargs) -> ToolResult:
        """Execute the tool with given arguments.

        Args:
            context: The session context
            **kwargs: Tool-specific arguments

        Returns:
            ToolResult with success status and data or error
        """
        raise NotImplementedError(f"{self.__class__.__name__}.execute() not implemented")

    def _require_param(self, kwargs: dict, name: str) -> Any:
        """Get a required parameter or raise ValueError.

        Args:
            kwargs: The arguments dictionary
            name: Parameter name to require

        Returns:
            The parameter value

        Raises:
            ValueError: If parameter is missing or empty
        """
        value = kwargs.get(name)
        if not value:
            raise ValueError(f"{name} is required")
        return value

    def _resolve_path(self, context: "Context", file_path: str) -> Path:
        """Resolve a file path relative to the current working directory.

        Args:
            context: The session context
            file_path: The file path to resolve

        Returns:
            Resolved Path object
        """
        return context.cwd / file_path

    def to_schema(self) -> Dict:
        """Convert tool to OpenAI function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class ToolRegistry:
    """Register and manage available tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_tool_schemas(self) -> List[Dict]:
        """Get all tools as OpenAI function schemas."""
        return [tool.to_schema() for tool in self._tools.values()]

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
