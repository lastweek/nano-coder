"""Live CLI turn progress display."""

from __future__ import annotations

import json
from typing import Any, List, Optional

from rich.console import Group, RenderableType
from rich.spinner import Spinner
from rich.text import Text

from src.turn_activity import TurnActivityEvent


class TurnProgressDisplay:
    """Track and render user-visible turn activity."""

    def __init__(self, *, skill_debug: bool = False, request_type: str = "streaming") -> None:
        self.current_status = "Preparing request"
        self.activity_lines: List[str] = []
        self.summary_lines: List[str] = []
        self.response_chunks: List[str] = []
        self.streaming_started = False
        self.skill_debug = skill_debug
        self.request_type = request_type

    def handle_event(self, event: TurnActivityEvent) -> None:
        """Update display state from one activity event."""
        self.current_status = self._status_for_event(event)

        live_line = self._format_event_line(event, summary=False)
        if live_line:
            self.activity_lines.append(live_line)

        summary_line = self._format_event_line(event, summary=True)
        if summary_line and (not self.summary_lines or self.summary_lines[-1] != summary_line):
            self.summary_lines.append(summary_line)

    def append_stream_chunk(self, token: str) -> None:
        """Append one streamed answer chunk."""
        if not token:
            return

        if not self.streaming_started:
            self.streaming_started = True
            self.handle_event(TurnActivityEvent(kind="answer_stream_started"))

        self.response_chunks.append(token)
        self.current_status = "Streaming answer"

    def render_live(self) -> RenderableType:
        """Render the in-progress live view."""
        renderables: List[RenderableType] = [
            Spinner("dots", Text(self.current_status, style="bold cyan"))
        ]

        if self.activity_lines:
            renderables.append(
                Group(
                    *[
                        Text(f"  • {line}", style="dim")
                        for line in self.activity_lines[-8:]
                    ]
                )
            )

        return Group(*renderables)

    def render_persisted(self) -> RenderableType:
        """Render the concise persisted summary."""
        if not self.summary_lines:
            return Text("")

        return Group(
            *[Text(f"  • {line}", style="dim") for line in self.summary_lines]
        )

    def final_response_text(self) -> str:
        """Return the accumulated streamed response text."""
        return "".join(self.response_chunks)

    def has_summary(self) -> bool:
        """Whether there is summary output to persist."""
        return bool(self.summary_lines)

    def _status_for_event(self, event: TurnActivityEvent) -> str:
        """Map one event to the current status line."""
        iteration = event.iteration + 1 if event.iteration is not None else None
        details = event.details

        if event.kind == "skill_preload":
            return f"Preparing skill context: {details.get('skill_name')}"
        if event.kind == "context_compaction_started":
            return "Compacting context"
        if event.kind == "context_compaction_completed":
            return "Context compacted"
        if event.kind == "context_compaction_failed":
            return "Context compaction failed"
        if event.kind == "skill_normalized":
            return "Preparing skill context"
        if event.kind == "skill_load_requested":
            return f"Loading skill: {details.get('skill_name')}"
        if event.kind == "skill_load_succeeded":
            return f"Loaded skill: {details.get('skill_name')}"
        if event.kind == "skill_load_failed":
            return f"Skill load failed: {details.get('skill_name')}"
        if event.kind == "llm_call_started":
            return f"Thinking: LLM call {iteration}"
        if event.kind == "llm_call_finished":
            if details.get("has_tool_calls"):
                return f"Working: {details.get('tool_call_count', 0)} tool(s) requested"
            return "Streaming answer" if details.get("stream") else "Final answer ready"
        if event.kind == "tool_call_started":
            return f"Working: running {self._format_tool_signature(details.get('tool_name'), details.get('arguments'))}"
        if event.kind == "tool_call_finished":
            if details.get("success", True):
                return f"Thinking: finished {details.get('tool_name')}"
            return "Working: tool failed"
        if event.kind == "answer_stream_started":
            return "Streaming answer"
        if event.kind == "turn_completed":
            return "Completed"
        if event.kind == "turn_error":
            return "Error"
        return self.current_status

    def _format_event_line(self, event: TurnActivityEvent, *, summary: bool) -> Optional[str]:
        """Map one event to a user-facing feed line."""
        iteration = event.iteration + 1 if event.iteration is not None else None
        details = event.details

        if event.kind == "skill_preload":
            line = f"Skill preloaded: {details.get('skill_name')} ({details.get('reason')})"
            if self.skill_debug:
                line += (
                    f" [source={details.get('source')}, "
                    f"catalog={'yes' if details.get('catalog_visible') else 'no'}]"
                )
            return line

        if event.kind == "context_compaction_started":
            return None if summary else (
                f"Compacting context: {details.get('covered_turn_count', 0)} older turns"
            )

        if event.kind == "context_compaction_completed":
            line = (
                f"Context compacted: {details.get('covered_turn_count', 0)} turns summarized, "
                f"{details.get('retained_turn_count', 0)} recent turns kept"
            )
            if summary:
                return f"Context compacted: {details.get('covered_turn_count', 0)} older turns summarized"
            return line

        if event.kind == "context_compaction_failed":
            return f"Context compaction failed: {details.get('error')}"

        if event.kind == "skill_normalized":
            return None if summary else "Using preloaded skill context for this request"

        if event.kind == "skill_load_requested":
            return None if summary else f"Loading skill: {details.get('skill_name')}"

        if event.kind == "skill_load_succeeded":
            return f"Loaded skill: {details.get('skill_name')}"

        if event.kind == "skill_load_failed":
            line = f"Failed to load skill: {details.get('skill_name')}"
            if self.skill_debug and details.get("error"):
                line += f" ({details.get('error')})"
            return line

        if event.kind == "llm_call_started":
            return None if summary else f"LLM call {iteration} started"

        if event.kind == "llm_call_finished":
            if details.get("has_tool_calls"):
                tool_call_count = details.get("tool_call_count", 0)
                tool_word = "tool" if tool_call_count == 1 else "tools"
                return f"LLM call {iteration} requested {tool_call_count} {tool_word}"
            if summary:
                return f"LLM call {iteration} produced final answer"
            return f"LLM call {iteration} returned final answer"

        if event.kind == "tool_call_started":
            signature = self._format_tool_signature(details.get("tool_name"), details.get("arguments"))
            return None if summary else f"Tool started: {signature}"

        if event.kind == "tool_call_finished":
            duration = float(details.get("duration_s", 0.0))
            signature = self._format_tool_signature(details.get("tool_name"), details.get("arguments"))
            if details.get("success", True):
                return f"Tool finished: {signature} ({duration:.2f}s)"

            line = f"Tool failed: {signature} ({duration:.2f}s)"
            if self.skill_debug and details.get("error"):
                line += f" ({details.get('error')})"
            return line

        if event.kind == "answer_stream_started":
            return None

        if event.kind == "turn_completed":
            return None

        if event.kind == "turn_error":
            return f"Turn failed: {details.get('message')}"

        return None

    def _format_tool_signature(self, tool_name: Optional[str], arguments: Any) -> str:
        """Format a compact user-facing tool call signature."""
        base = tool_name or "tool"
        if not isinstance(arguments, dict) or not arguments:
            return base

        parts = []
        for index, (key, value) in enumerate(arguments.items()):
            if index >= 2:
                parts.append("...")
                break
            parts.append(f"{key}={self._format_argument_value(value)}")

        signature = f"{base}({', '.join(parts)})"
        if len(signature) <= 96:
            return signature
        return signature[:93] + "..."

    def _format_argument_value(self, value: Any) -> str:
        """Format one tool argument for compact display."""
        if isinstance(value, str):
            compact = " ".join(value.split())
            if len(compact) > 36:
                compact = compact[:33] + "..."
            return repr(compact)
        if isinstance(value, (int, float, bool)) or value is None:
            return repr(value)

        try:
            compact = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        except TypeError:
            compact = repr(value)

        if len(compact) > 36:
            compact = compact[:33] + "..."
        return compact
