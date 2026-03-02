"""Local subagent runtime for delegated child-agent work."""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from time import perf_counter
from typing import Any, Literal, Optional

from src.config import config
from src.context import Context
from src.llm import LLMClient
from src.logger import SessionLogger
from src.tools import REQUEST_KIND_SUBAGENT_TURN
from src.tool_builder import clone_tool_registry


def _utc_now() -> str:
    """Return an ISO timestamp for run metadata."""
    return datetime.now().isoformat()


def _sanitize_label(label: str) -> str:
    """Convert a label into a filesystem-safe form."""
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in label.strip()).strip("-") or "subagent"


@dataclass(frozen=True)
class SubagentRequest:
    """A parent-authored task brief for a child agent."""

    task: str
    label: str
    context: str
    success_criteria: str
    files: list[str]
    output_hint: str


@dataclass(frozen=True)
class SubagentResult:
    """Structured result returned to the parent agent."""

    subagent_id: str
    label: str
    status: Literal["completed", "failed", "timed_out"]
    summary: str
    report: str
    session_dir: str
    llm_log: str
    events_log: str
    llm_call_count: int
    tool_call_count: int
    tools_used: list[str]
    error: str | None = None


@dataclass
class SubagentRun:
    """In-memory tracking for a child run in the current parent session."""

    subagent_id: str
    parent_turn_id: int | None
    label: str
    task: str
    status: Literal["running", "completed", "failed", "timed_out"]
    started_at: str
    ended_at: str | None
    duration_s: float | None
    result: SubagentResult | None


@dataclass(frozen=True)
class _PreparedSubagent:
    """Concrete prepared child-agent execution state."""

    request: SubagentRequest
    run: SubagentRun
    child_agent: Any
    task_message: str
    session_dir: str
    llm_log: str
    events_log: str

class SubagentManager:
    """Create and run local child agents with bounded parallel fan-out."""

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        max_parallel: Optional[int] = None,
        max_per_turn: Optional[int] = None,
        default_timeout_seconds: Optional[int] = None,
    ) -> None:
        subagent_config = config.subagents
        self.enabled = subagent_config.enabled if enabled is None else enabled
        self.max_parallel = subagent_config.max_parallel if max_parallel is None else max_parallel
        self.max_per_turn = subagent_config.max_per_turn if max_per_turn is None else max_per_turn
        self.default_timeout_seconds = (
            subagent_config.default_timeout_seconds
            if default_timeout_seconds is None
            else default_timeout_seconds
        )

        self._lock = Lock()
        self._run_counter = 0
        self._runs: list[SubagentRun] = []
        self._runs_by_id: dict[str, SubagentRun] = {}
        self._per_turn_counts: dict[tuple[str, int | None], int] = {}

    def list_runs(self) -> list[SubagentRun]:
        """Return runs created in this top-level session."""
        with self._lock:
            return list(self._runs)

    def get_run(self, subagent_id: str) -> SubagentRun | None:
        """Return one run by id."""
        with self._lock:
            return self._runs_by_id.get(subagent_id)

    def build_request(self, arguments: dict[str, Any]) -> SubagentRequest:
        """Normalize tool-call arguments into a subagent request."""
        task = str(arguments.get("task", "")).strip()
        if not task:
            raise ValueError("task is required")

        raw_files = arguments.get("files") or []
        files = [str(item) for item in raw_files if str(item).strip()] if isinstance(raw_files, list) else []
        return SubagentRequest(
            task=task,
            label=str(arguments.get("label", "")).strip(),
            context=str(arguments.get("context", "")).strip(),
            success_criteria=str(arguments.get("success_criteria", "")).strip(),
            files=files,
            output_hint=str(arguments.get("output_hint", "")).strip(),
        )

    def run_one(
        self,
        parent_agent,
        request: SubagentRequest,
        *,
        parent_turn_id: int | None,
        iteration: int,
        on_event=None,
    ) -> SubagentResult:
        """Run one child agent synchronously."""
        if not self.enabled:
            return self._disabled_result(request)

        allowance = self._reserve_turn_slots(parent_agent, parent_turn_id, requested=1)
        if allowance < 1:
            return self._limit_result(request, "Subagent per-turn limit reached")

        prepared = self._prepare_run(parent_agent, request, parent_turn_id)
        self._log_subagent_started(parent_agent, prepared, parent_turn_id, on_event)

        started_at = perf_counter()
        try:
            result = self._execute_prepared(prepared)
        except Exception as exc:
            result = self._failure_result(prepared, f"Subagent failed: {exc}")

        result = self._finalize_run(prepared.run, result, perf_counter() - started_at)
        self._log_subagent_finished(parent_agent, result, parent_turn_id, on_event)
        return result

    def run_batch(
        self,
        parent_agent,
        requests: list[SubagentRequest],
        *,
        parent_turn_id: int | None,
        iteration: int,
        on_event=None,
    ) -> list[SubagentResult]:
        """Run multiple child agents with bounded parallel fan-out."""
        if not requests:
            return []

        if not self.enabled:
            return [self._disabled_result(request) for request in requests]

        allowance = self._reserve_turn_slots(parent_agent, parent_turn_id, requested=len(requests))
        allowed_requests = requests[:allowance]
        overflow_requests = requests[allowance:]

        prepared_items = [
            self._prepare_run(parent_agent, request, parent_turn_id)
            for request in allowed_requests
        ]
        for prepared in prepared_items:
            self._log_subagent_started(parent_agent, prepared, parent_turn_id, on_event)

        results_by_id: dict[str, SubagentResult] = {}
        future_map: dict[Any, tuple[_PreparedSubagent, float]] = {}
        executor = ThreadPoolExecutor(max_workers=self.max_parallel, thread_name_prefix="subagent")
        try:
            for prepared in prepared_items:
                started_at = perf_counter()
                future = executor.submit(self._execute_prepared, prepared)
                future_map[future] = (prepared, started_at)

            for future, (prepared, started_at) in future_map.items():
                try:
                    raw_result = future.result(timeout=self.default_timeout_seconds)
                except TimeoutError:
                    future.cancel()
                    raw_result = self._timed_out_result(prepared, self.default_timeout_seconds)
                except Exception as exc:
                    raw_result = self._failure_result(prepared, f"Subagent failed: {exc}")

                finalized = self._finalize_run(
                    prepared.run,
                    raw_result,
                    perf_counter() - started_at,
                )
                results_by_id[prepared.run.subagent_id] = finalized
                self._log_subagent_finished(parent_agent, finalized, parent_turn_id, on_event)
        finally:
            executor.shutdown(wait=False, cancel_futures=False)

        ordered_results = [results_by_id[prepared.run.subagent_id] for prepared in prepared_items]
        ordered_results.extend(
            self._limit_result(
                request,
                (
                    "Subagent request rejected because this turn exceeded the configured "
                    f"max_per_turn={self.max_per_turn}"
                ),
            )
            for request in overflow_requests
        )
        return ordered_results

    def result_to_payload(self, result: SubagentResult) -> dict[str, Any]:
        """Convert a result into the parent tool-result payload."""
        return {
            "subagent_id": result.subagent_id,
            "label": result.label,
            "status": result.status,
            "summary": result.summary,
            "report": result.report,
            "session_dir": result.session_dir,
            "llm_log": result.llm_log,
            "events_log": result.events_log,
            "llm_call_count": result.llm_call_count,
            "tool_call_count": result.tool_call_count,
            "tools_used": result.tools_used,
            "error": result.error,
        }

    def _reserve_turn_slots(self, parent_agent, parent_turn_id: int | None, *, requested: int) -> int:
        """Reserve the remaining per-turn capacity for subagent runs."""
        if parent_turn_id is None:
            return min(requested, self.max_per_turn)

        key = (parent_agent.context.session_id, parent_turn_id)
        with self._lock:
            used = self._per_turn_counts.get(key, 0)
            remaining = max(self.max_per_turn - used, 0)
            allowance = min(requested, remaining)
            self._per_turn_counts[key] = used + allowance
            return allowance

    def _prepare_run(self, parent_agent, request: SubagentRequest, parent_turn_id: int | None) -> _PreparedSubagent:
        """Create the child agent and log paths for one prepared run."""
        subagent_id, label = self._allocate_identity(request.label)
        run = SubagentRun(
            subagent_id=subagent_id,
            parent_turn_id=parent_turn_id,
            label=label,
            task=request.task,
            status="running",
            started_at=_utc_now(),
            ended_at=None,
            duration_s=None,
            result=None,
        )
        with self._lock:
            self._runs.append(run)
            self._runs_by_id[subagent_id] = run

        child_context = Context.create(cwd=str(parent_agent.context.cwd))
        parent_session_dir = parent_agent.logger.ensure_session_dir()
        child_log_dir = parent_session_dir / "subagents"
        child_logger = SessionLogger(
            child_context.session_id,
            log_dir=str(child_log_dir),
            enabled=parent_agent.logger.enabled,
            async_mode=False,
            update_latest_symlinks=False,
            session_kind="subagent",
            parent_session_id=parent_agent.context.session_id,
            parent_turn_id=parent_turn_id,
            subagent_id=subagent_id,
            subagent_label=label,
        )
        child_logger.start_session(
            cwd=str(child_context.cwd),
            provider=getattr(parent_agent.llm, "provider", "unknown"),
            model=getattr(parent_agent.llm, "model", "unknown"),
            base_url=getattr(parent_agent.llm, "base_url", None),
            streaming_enabled=False,
        )
        child_logger.ensure_session_dir()

        child_llm = LLMClient(
            provider=getattr(parent_agent.llm, "provider", None),
            model=getattr(parent_agent.llm, "model", None),
            base_url=getattr(parent_agent.llm, "base_url", None),
        )
        child_tools = clone_tool_registry(parent_agent.tools, include_subagent_tool=False)

        from src.agent import Agent

        child_agent = Agent(
            child_llm,
            child_tools,
            child_context,
            skill_manager=parent_agent.skill_manager,
            logger=child_logger,
            request_kind=REQUEST_KIND_SUBAGENT_TURN,
        )
        task_message = self._build_task_message(request)

        session_dir = str(child_logger.session_dir)
        llm_log = str(child_logger.get_llm_log_path())
        events_log = str(child_logger.get_events_path())
        return _PreparedSubagent(
            request=request,
            run=run,
            child_agent=child_agent,
            task_message=task_message,
            session_dir=session_dir,
            llm_log=llm_log,
            events_log=events_log,
        )

    def _allocate_identity(self, requested_label: str) -> tuple[str, str]:
        """Allocate a stable run id and display label."""
        with self._lock:
            self._run_counter += 1
            index = self._run_counter
        label = requested_label.strip() or f"subagent-{index:03d}"
        subagent_id = f"sa_{index:04d}_{uuid.uuid4().hex[:8]}"
        return subagent_id, label

    def _build_task_message(self, request: SubagentRequest) -> str:
        """Render the child agent's delegated first user message."""
        lines = [
            "You are a delegated subagent working inside the same repository as the parent agent.",
            "Work from this task brief only. Inspect files and use tools yourself rather than assuming parent context.",
            "",
            "Task:",
            request.task,
        ]
        if request.context:
            lines.extend(["", "Context from parent:", request.context])
        if request.files:
            lines.extend(["", "Relevant files:"])
            lines.extend(f"- {path}" for path in request.files)
        if request.success_criteria:
            lines.extend(["", "Success criteria:", request.success_criteria])
        if request.output_hint:
            lines.extend(["", "Output hint:", request.output_hint])
        lines.extend([
            "",
            "Return a concise report. Put the executive summary in the first paragraph.",
        ])
        return "\n".join(lines)

    def _execute_prepared(self, prepared: _PreparedSubagent) -> SubagentResult:
        """Execute one prepared child agent synchronously."""
        try:
            report = prepared.child_agent.run(prepared.task_message)
            status: Literal["completed", "failed", "timed_out"] = "completed"
            error = None
        except Exception as exc:
            report = ""
            status = "failed"
            error = str(exc)
        finally:
            prepared.child_agent.logger.close(status="completed" if status == "completed" else "error")

        if status != "completed":
            return self._failure_result(prepared, error or "Subagent failed")

        session_data = self._read_child_session_data(prepared)
        return SubagentResult(
            subagent_id=prepared.run.subagent_id,
            label=prepared.run.label,
            status="completed",
            summary=self._extract_summary(report),
            report=report,
            session_dir=prepared.session_dir,
            llm_log=prepared.llm_log,
            events_log=prepared.events_log,
            llm_call_count=session_data["llm_call_count"],
            tool_call_count=session_data["tool_call_count"],
            tools_used=session_data["tools_used"],
            error=None,
        )

    def _read_child_session_data(self, prepared: _PreparedSubagent) -> dict[str, Any]:
        """Read child session aggregates after the logger closes."""
        session_path = prepared.child_agent.logger.session_dir / "session.json"
        events_path = prepared.child_agent.logger.session_dir / "events.jsonl"

        llm_call_count = 0
        tool_call_count = 0
        tools_used: list[str] = []

        if session_path.exists():
            session_json = json.loads(session_path.read_text(encoding="utf-8"))
            llm_call_count = int(session_json.get("llm_call_count", 0))
            tool_call_count = int(session_json.get("tool_call_count", 0))

        if events_path.exists():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("kind") == "turn_completed":
                    tools_used = [str(name) for name in event.get("tools_used", [])]

        return {
            "llm_call_count": llm_call_count,
            "tool_call_count": tool_call_count,
            "tools_used": tools_used,
        }

    def _finalize_run(
        self,
        run: SubagentRun,
        result: SubagentResult,
        duration_s: float | None,
    ) -> SubagentResult:
        """Persist run status in-memory and return the finalized result."""
        finalized = SubagentResult(
            **{
                **result.__dict__,
            }
        )
        with self._lock:
            run.status = finalized.status
            run.ended_at = _utc_now()
            run.duration_s = duration_s
            run.result = finalized
        return finalized

    def _log_subagent_started(self, parent_agent, prepared: _PreparedSubagent, parent_turn_id: int | None, on_event) -> None:
        """Record a parent-visible subagent start event."""
        parent_agent.logger.log_subagent_event(
            turn_id=parent_turn_id,
            stage="started",
            subagent_id=prepared.run.subagent_id,
            label=prepared.run.label,
            task=prepared.request.task,
            session_dir=prepared.session_dir,
            llm_log=prepared.llm_log,
            events_log=prepared.events_log,
        )
        if on_event is not None:
            from src.turn_activity import TurnActivityEvent

            on_event(
                TurnActivityEvent(
                    kind="subagent_started",
                    details={
                        "subagent_id": prepared.run.subagent_id,
                        "label": prepared.run.label,
                        "task": prepared.request.task,
                    },
                )
            )

    def _log_subagent_finished(self, parent_agent, result: SubagentResult, parent_turn_id: int | None, on_event) -> None:
        """Record a parent-visible subagent completion or failure event."""
        stage = "completed" if result.status == "completed" else "failed"
        parent_agent.logger.log_subagent_event(
            turn_id=parent_turn_id,
            stage=stage,
            subagent_id=result.subagent_id,
            label=result.label,
            session_dir=result.session_dir,
            llm_log=result.llm_log,
            events_log=result.events_log,
            summary=result.summary,
            error=result.error,
            status=result.status,
        )
        if on_event is not None:
            from src.turn_activity import TurnActivityEvent

            if result.status == "completed":
                on_event(
                    TurnActivityEvent(
                        kind="subagent_completed",
                        details={
                            "subagent_id": result.subagent_id,
                            "label": result.label,
                            "duration_s": self.get_run(result.subagent_id).duration_s or 0.0,
                            "summary": result.summary,
                        },
                    )
                )
            else:
                on_event(
                    TurnActivityEvent(
                        kind="subagent_failed",
                        details={
                            "subagent_id": result.subagent_id,
                            "label": result.label,
                            "duration_s": self.get_run(result.subagent_id).duration_s or 0.0,
                            "error": result.error or "Subagent failed",
                        },
                    )
                )

    def _extract_summary(self, report: str) -> str:
        """Derive a concise executive summary from a child report."""
        paragraphs = [paragraph.strip() for paragraph in report.split("\n\n") if paragraph.strip()]
        if paragraphs:
            summary = " ".join(paragraphs[0].split())
        else:
            summary = " ".join(report.split())
        if len(summary) > 240:
            summary = summary[:237].rstrip() + "..."
        return summary

    def _disabled_result(self, request: SubagentRequest) -> SubagentResult:
        """Return a structured failure when subagents are disabled."""
        label = request.label or "subagent"
        return SubagentResult(
            subagent_id="disabled",
            label=label,
            status="failed",
            summary="Subagents are disabled.",
            report="",
            session_dir="",
            llm_log="",
            events_log="",
            llm_call_count=0,
            tool_call_count=0,
            tools_used=[],
            error="Subagents are disabled in the current configuration",
        )

    def _limit_result(self, request: SubagentRequest, message: str) -> SubagentResult:
        """Return a structured failure when the per-turn limit is exceeded."""
        return SubagentResult(
            subagent_id=f"rejected_{uuid.uuid4().hex[:8]}",
            label=request.label or "subagent",
            status="failed",
            summary=message,
            report="",
            session_dir="",
            llm_log="",
            events_log="",
            llm_call_count=0,
            tool_call_count=0,
            tools_used=[],
            error=message,
        )

    def _failure_result(self, prepared: _PreparedSubagent, message: str) -> SubagentResult:
        """Return a structured failure tied to a prepared child run."""
        session_data = self._read_child_session_data(prepared)
        return SubagentResult(
            subagent_id=prepared.run.subagent_id,
            label=prepared.run.label,
            status="failed",
            summary=f"Subagent failed: {message}",
            report="",
            session_dir=prepared.session_dir,
            llm_log=prepared.llm_log,
            events_log=prepared.events_log,
            llm_call_count=session_data["llm_call_count"],
            tool_call_count=session_data["tool_call_count"],
            tools_used=session_data["tools_used"],
            error=message,
        )

    def _timed_out_result(self, prepared: _PreparedSubagent, timeout_seconds: int) -> SubagentResult:
        """Return a structured timeout result."""
        session_data = self._read_child_session_data(prepared)
        message = f"Subagent timed out after {timeout_seconds} seconds"
        return SubagentResult(
            subagent_id=prepared.run.subagent_id,
            label=prepared.run.label,
            status="timed_out",
            summary=message,
            report="",
            session_dir=prepared.session_dir,
            llm_log=prepared.llm_log,
            events_log=prepared.events_log,
            llm_call_count=session_data["llm_call_count"],
            tool_call_count=session_data["tool_call_count"],
            tools_used=session_data["tools_used"],
            error=message,
        )
