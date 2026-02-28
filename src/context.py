"""Context management for Nano-Coder."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Dict
import uuid


@dataclass
class Context:
    """Session context passed to all operations."""
    cwd: Path
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls, cwd: str = ".") -> 'Context':
        """Create a new context with resolved working directory."""
        return cls(cwd=Path(cwd).resolve())

    def add_message(self, role: str, content: Any) -> None:
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in the conversation history."""
        return self.messages

    def clear_messages(self) -> None:
        """Clear the conversation history."""
        self.messages.clear()
