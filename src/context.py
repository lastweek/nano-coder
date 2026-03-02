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
    active_skills: List[str] = field(default_factory=list)

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

    def activate_skill(self, name: str) -> None:
        """Pin a skill for the current session."""
        if name not in self.active_skills:
            self.active_skills.append(name)

    def deactivate_skill(self, name: str) -> None:
        """Unpin a skill for the current session."""
        if name in self.active_skills:
            self.active_skills.remove(name)

    def clear_skills(self) -> None:
        """Clear all pinned skills."""
        self.active_skills.clear()

    def get_active_skills(self) -> List[str]:
        """Return pinned skill names."""
        return list(self.active_skills)
