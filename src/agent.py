"""Agent orchestration for Nano-Coder."""

import json
from time import perf_counter
from typing import Callable, Optional, List, Dict, Tuple, Any
from src.tools import (
    ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL,
    REQUEST_KIND_AGENT_TURN, REQUEST_KIND_CONTEXT_COMPACTION
)
from src.logger import SessionLogger
from src.metrics import LLMMetrics
from src.config import Config, config
from src.context_compaction import ContextCompactionManager, ContextCompactionPolicy
from src.turn_activity import TurnActivityCallback, TurnActivityEvent


class Agent:
    """Main agent orchestration using ReAct loop."""

    def __init__(self, llm_client, tools, context, skill_manager=None):
        """Initialize the agent.

        Args:
            llm_client: LLMClient instance
            tools: ToolRegistry instance
            context: Context instance
            skill_manager: Optional SkillManager instance
        """
        self.llm = llm_client
        self.tools = tools
        self.context = context
        self.skill_manager = skill_manager
        current_config = Config.load()
        self.max_iterations = current_config.agent.max_iterations
        self.logger = SessionLogger(context.session_id)
        self.logger.start_session(
            cwd=self.context.cwd,
            provider=getattr(self.llm, "provider", "unknown"),
            model=getattr(self.llm, "model", "unknown"),
            base_url=getattr(self.llm, "base_url", None),
            streaming_enabled=current_config.ui.enable_streaming,
        )

        # Share logger with LLM client for request/response logging
        if hasattr(self.llm, 'logger'):
            self.llm.logger = self.logger

        # Accumulate metrics across LLM requests
        self.request_metrics: List[LLMMetrics] = []
        self.context_compaction = ContextCompactionManager(
            self.llm,
            self.context,
            self.skill_manager,
            ContextCompactionPolicy(
                auto_compact=current_config.context.auto_compact,
                auto_compact_threshold=current_config.context.auto_compact_threshold,
                target_usage_after_compaction=current_config.context.target_usage_after_compaction,
                min_recent_turns=current_config.context.min_recent_turns,
            ),
        )

        # Pre-build system message (hot-path optimization - avoids rebuilding on each request)
        tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in self.tools._tools.values()
        )

        self._cached_system_prompt_base = f"""You are a helpful coding assistant with access to tools.

Working directory: {self.context.cwd}

Available tools:
{tool_descriptions}

When asked to do something that requires tools, use them. Always explain what you're doing before using a tool.
Think step by step. If you make a mistake, try to recover.

Be concise and helpful."""

        # Tool schemas cached on first use (expensive to build)
        self._cached_tool_schemas = None
        self._skill_event_callback = None

    def set_skill_event_callback(self, callback: Optional[Callable]) -> None:
        """Set an optional callback for skill debug events."""
        self._skill_event_callback = callback

    def _get_system_message(self) -> Dict:
        """Build the system message for the current turn."""
        return {"role": ROLE_SYSTEM, "content": self._build_system_prompt()}

    def _build_system_prompt(self) -> str:
        """Build the full system prompt for the current turn."""
        sections = [self._cached_system_prompt_base]

        skill_catalog = self._build_skill_catalog_section()
        if skill_catalog:
            sections.append(skill_catalog)

        return "\n\n".join(section for section in sections if section)

    def _build_skill_catalog_section(self) -> str:
        """Build the compact catalog of discovered skills for this session."""
        if not self.skill_manager:
            return ""

        skills = self.skill_manager.list_catalog_skills()
        if not skills:
            return ""

        lines = ["Available skills in this session:"]
        for skill in skills:
            lines.append(f"- {skill.name}: {skill.short_description}")

        lines.extend([
            "",
            "Load a skill only when needed by calling load_skill.",
            "Users may also explicitly preload a skill by writing $skill-name.",
        ])
        return "\n".join(lines)

    def _emit_skill_event(self, turn_id: int, event: str, **details) -> None:
        """Log and optionally surface a skill-related debug event."""
        self.logger.log_skill_event(turn_id, event, **details)
        if self._skill_event_callback is not None:
            self._skill_event_callback(event, details)

    def _emit_turn_event(
        self,
        callback: Optional[TurnActivityCallback],
        kind: str,
        *,
        iteration: Optional[int] = None,
        **details: Any,
    ) -> None:
        """Emit a user-safe activity event for the CLI."""
        if callback is None:
            return
        callback(
            TurnActivityEvent(
                kind=kind,
                iteration=iteration,
                details=details,
            )
        )

    def _replay_pending_skill_events(
        self,
        turn_id: int,
        pending_skill_events: List[Tuple[str, Dict[str, Any]]],
        on_event: Optional[TurnActivityCallback],
        skills_used: List[str],
    ) -> None:
        """Write deferred skill events after a turn id exists."""
        for event_name, details in pending_skill_events:
            self._emit_skill_event(turn_id, event_name, **details)

            if event_name == "preload":
                skill_name = details.get("skill_name")
                if skill_name and skill_name not in skills_used:
                    skills_used.append(skill_name)
                self._emit_turn_event(on_event, "skill_preload", **details)
            elif event_name == "normalized_user_message":
                self._emit_turn_event(on_event, "skill_normalized", **details)

    def _metrics_event_details(self, metrics: Any) -> Dict[str, Any]:
        """Normalize metric fields for activity events."""
        return {
            "duration_s": getattr(metrics, "duration", 0.0),
            "prompt_tokens": getattr(metrics, "prompt_tokens", 0),
            "completion_tokens": getattr(metrics, "completion_tokens", 0),
            "total_tokens": getattr(metrics, "total_tokens", 0),
            "cached_tokens": getattr(metrics, "cached_tokens", 0),
        }

    def _record_prompt_metrics(self, metrics: Any) -> None:
        """Persist the most recent prompt usage for compaction decisions."""
        prompt_tokens = getattr(metrics, "prompt_tokens", None)
        if isinstance(prompt_tokens, int) and prompt_tokens > 0:
            self.context.last_prompt_tokens = prompt_tokens
            cached_tokens = getattr(metrics, "cached_tokens", None)
            self.context.last_prompt_cached_tokens = (
                cached_tokens if isinstance(cached_tokens, int) else None
            )
        else:
            self.context.last_prompt_tokens = None
            self.context.last_prompt_cached_tokens = None
        self.context.last_context_window = Config.load().llm.context_window

    def _initialize_turn(
        self,
        user_message: str,
        on_event: Optional[TurnActivityCallback] = None,
    ) -> Tuple[int, List[str], List[str], List[Dict[str, Any]], str]:
        """Initialize a new turn with message building and logging.

        Args:
            user_message: The user's input message
            on_event: Optional callback for events

        Returns:
            Tuple of (turn_id, tools_used list, skills_used list, normalized_user_message)
        """
        normalized_user_message, preload_skill_names, pending_skill_events = self._prepare_turn_inputs(user_message)
        turn_id = self.logger.start_turn(
            raw_user_input=user_message,
            normalized_user_input=normalized_user_message,
        )
        tools_used: List[str] = []
        skills_used: List[str] = []
        self._replay_pending_skill_events(turn_id, pending_skill_events, on_event, skills_used)

        # Cache tool schemas for hot-path optimization (only once)
        if self._cached_tool_schemas is None:
            self._cached_tool_schemas = self.tools.get_tool_schemas()

        self._maybe_auto_compact(turn_id, on_event)
        messages = self._build_messages(normalized_user_message, preload_skill_names)

        return turn_id, tools_used, skills_used, messages, normalized_user_message

    def _prepare_turn_inputs(
        self,
        user_message: str,
    ) -> Tuple[str, List[str], List[Tuple[str, Dict[str, Any]]]]:
        """Normalize the turn input and gather skill preloads without building messages."""
        pending_skill_events: List[Tuple[str, Dict[str, Any]]] = []
        normalized_user_message = user_message
        preload_skill_names: List[str] = []

        if self.skill_manager:
            pinned_skill_names = [
                skill_name
                for skill_name in self.context.get_active_skills()
                if self.skill_manager.get_skill(skill_name) is not None
            ]
            preload_skill_names.extend(pinned_skill_names)
            for skill_name in pinned_skill_names:
                skill = self.skill_manager.get_skill(skill_name)
                if skill is None:
                    continue
                pending_skill_events.append((
                    "preload",
                    {
                        "skill_name": skill.name,
                        "reason": "pinned",
                        "source": skill.source,
                        "catalog_visible": skill.catalog_visible,
                        "skill_file": str(skill.skill_file),
                    },
                ))

            mention_result = self.skill_manager.extract_skill_mentions(user_message)
            explicit_skill_names = [
                skill_name
                for skill_name in mention_result.skill_names
                if skill_name not in preload_skill_names
            ]
            preload_skill_names.extend(explicit_skill_names)
            for skill_name in explicit_skill_names:
                skill = self.skill_manager.get_skill(skill_name)
                if skill is None:
                    continue
                pending_skill_events.append((
                    "preload",
                    {
                        "skill_name": skill.name,
                        "reason": "explicit",
                        "source": skill.source,
                        "catalog_visible": skill.catalog_visible,
                        "skill_file": str(skill.skill_file),
                    },
                ))

            if mention_result.cleaned_text:
                normalized_user_message = mention_result.cleaned_text
            elif mention_result.skill_names:
                normalized_user_message = "Use the preloaded skill context for this request."
                pending_skill_events.append((
                    "normalized_user_message",
                    {
                        "reason": "explicit_skill_only",
                        "content": normalized_user_message,
                    },
                ))

        return normalized_user_message, preload_skill_names, pending_skill_events

    def _build_messages(self, normalized_user_message: str, preload_skill_names: List[str]) -> List[Dict]:
        """Build the message list for the next LLM call."""
        messages: List[Dict[str, Any]] = [self._get_system_message()]

        summary_message = self.context.get_summary_message()
        if summary_message is not None:
            messages.append(summary_message)

        messages.extend(self.context.get_messages())

        if self.skill_manager and preload_skill_names:
            messages.extend(self.skill_manager.build_preload_messages(preload_skill_names))

        messages.append({"role": ROLE_USER, "content": normalized_user_message})
        return messages

    def _maybe_auto_compact(
        self,
        turn_id: int,
        on_event: Optional[TurnActivityCallback],
    ) -> None:
        """Compact older turns before the first model call when policy requires it."""
        decision = self.context_compaction.build_decision(self)
        if not decision.should_compact:
            return

        plan = self.context_compaction._build_plan(self, force=False)
        if not plan.turns_to_compact:
            return

        self.logger.log_context_compaction_event(
            turn_id=turn_id,
            stage="started",
            reason=decision.reason,
            covered_turn_count=len(plan.turns_to_compact),
            retained_turn_count=len(plan.retained_turns),
        )
        self._emit_turn_event(
            on_event,
            "context_compaction_started",
            reason=decision.reason,
            covered_turn_count=len(plan.turns_to_compact),
            retained_turn_count=len(plan.retained_turns),
        )

        result = self.context_compaction.compact_now(
            self,
            decision.reason,
            turn_id=turn_id,
            force=False,
        )
        if result.error:
            self.logger.log_context_compaction_event(
                turn_id=turn_id,
                stage="failed",
                reason=decision.reason,
                error=result.error,
            )
            self._emit_turn_event(
                on_event,
                "context_compaction_failed",
                reason=decision.reason,
                error=result.error,
            )
            return

        self.logger.log_context_compaction_event(
            turn_id=turn_id,
            stage="completed",
            reason=decision.reason,
            covered_turn_count=result.covered_turn_count,
            retained_turn_count=result.retained_turn_count,
            before_tokens=result.before_tokens,
            after_tokens=result.after_tokens,
        )
        self._emit_turn_event(
            on_event,
            "context_compaction_completed",
            reason=decision.reason,
            covered_turn_count=result.covered_turn_count,
            retained_turn_count=result.retained_turn_count,
            before_tokens=result.before_tokens,
            after_tokens=result.after_tokens,
        )

    def _assistant_message_from_response(self, response: Dict) -> Dict:
        """Convert an assistant response into a replayable conversation message."""
        message = {
            "role": response["role"],
            "content": response.get("content", ""),
        }

        tool_calls = response.get("tool_calls", [])
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": tool_call["arguments"],
                    },
                }
                for tool_call in tool_calls
            ]

        return message

    def _execute_tool_call(self, tool_call: Dict, parsed_args: Dict) -> Tuple[Dict, Dict]:
        """Execute a single tool call.

        Args:
            tool_call: Tool call dict with id, name, and arguments
            parsed_args: Pre-parsed tool arguments

        Returns:
            Tuple of (tool result message for LLM, parsed result dict)
        """
        tool_name = tool_call["name"]
        tool_id = tool_call["id"]

        try:
            # Get tool and execute
            tool = self.tools.get(tool_name)
            if not tool:
                result = {"error": f"Unknown tool: {tool_name}"}
            else:
                result_obj = tool.execute(self.context, **parsed_args)
                if result_obj.success:
                    result = {"output": str(result_obj.data)}
                else:
                    result = {"error": result_obj.error or "Tool execution failed"}

        except json.JSONDecodeError:
            result = {"error": f"Invalid JSON in tool arguments: {tool_call['arguments']}"}
        except Exception as e:
            result = {"error": f"Error executing tool: {e}"}

        # Format as tool result message and return both message and parsed result
        tool_result_message = {
            "role": ROLE_TOOL,
            "tool_call_id": tool_id,
            "content": json.dumps(result)
        }
        return tool_result_message, result

    def _process_tool_calls(
        self,
        tool_calls: List[Dict],
        messages: List[Dict],
        turn_id: int,
        iteration: int,
        on_tool_call: Optional[Callable] = None,
        on_event: Optional[TurnActivityCallback] = None,
        tools_used: Optional[List[str]] = None,
        skills_used: Optional[List[str]] = None,
    ) -> int:
        """Process multiple tool calls and add results to messages.

        Args:
            tool_calls: List of tool call dicts
            messages: Message list to append results to
            on_tool_call: Optional callback for notification
        """
        processed_count = 0
        for tool_call in tool_calls:
            processed_count += 1
            parsed_args = json.loads(tool_call["arguments"])
            tool_name = tool_call["name"]
            tool_call_id = tool_call["id"]
            if tools_used is not None and tool_name not in tools_used:
                tools_used.append(tool_name)

            if tool_name == "load_skill":
                self._emit_skill_event(
                    turn_id,
                    "tool_load_requested",
                    skill_name=parsed_args.get("skill_name"),
                    iteration=iteration,
                )
                self._emit_turn_event(
                    on_event,
                    "skill_load_requested",
                    iteration=iteration,
                    skill_name=parsed_args.get("skill_name"),
                )

            if on_tool_call:
                on_tool_call(tool_name, parsed_args)

            self._emit_turn_event(
                on_event,
                "tool_call_started",
                iteration=iteration,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                arguments=parsed_args,
            )
            started_at = perf_counter()

            self.logger.log_tool_call(
                turn_id=turn_id,
                iteration=iteration,
                tool_name=tool_name,
                arguments=parsed_args,
                tool_call_id=tool_call_id,
            )

            tool_result_message, result_content = self._execute_tool_call(tool_call, parsed_args)
            messages.append(tool_result_message)
            duration_s = perf_counter() - started_at

            self.logger.log_tool_result(
                turn_id=turn_id,
                iteration=iteration,
                tool_name=tool_name,
                result=result_content,
                tool_call_id=tool_call_id,
            )
            self._emit_turn_event(
                on_event,
                "tool_call_finished",
                iteration=iteration,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                arguments=parsed_args,
                success="error" not in result_content,
                duration_s=duration_s,
                error=result_content.get("error"),
            )
            if tool_name == "load_skill":
                skill_event_name = "tool_load_succeeded"
                cli_event_name = "skill_load_succeeded"
                event_details = {
                    "skill_name": parsed_args.get("skill_name"),
                    "iteration": iteration,
                }
                if "error" in result_content:
                    skill_event_name = "tool_load_failed"
                    cli_event_name = "skill_load_failed"
                    event_details["error"] = result_content["error"]
                elif skills_used is not None:
                    skill_name = parsed_args.get("skill_name")
                    if skill_name and skill_name not in skills_used:
                        skills_used.append(skill_name)
                self._emit_skill_event(turn_id, skill_event_name, **event_details)
                cli_details = {key: value for key, value in event_details.items() if key != "iteration"}
                self._emit_turn_event(on_event, cli_event_name, iteration=iteration, **cli_details)
        return processed_count

    def _finalize_response(self, user_message: str, final_response: str) -> None:
        """Save and log final agent response.

        Args:
            user_message: Original user message
            final_response: Final response text
        """
        self.context.add_message(ROLE_USER, user_message)
        self.context.add_message(ROLE_ASSISTANT, final_response)
        # Note: Response is already logged by llm_client.log_llm_response()
        # No need to log again here

    def _handle_max_iterations(
        self,
        normalized_user_message: str,
        turn_id: int,
        tool_call_count: int,
        tools_used: List[str],
        skills_used: List[str],
        on_event: Optional[TurnActivityCallback] = None,
    ) -> str:
        """Handle max iterations error case.

        Args:
            normalized_user_message: The normalized user message
            turn_id: Turn ID for logging
            tool_call_count: Number of tool calls made
            tools_used: List of tools used
            skills_used: List of skills used
            on_event: Optional callback for events

        Returns:
            Error response message
        """
        error_response = "I reached the maximum number of iterations. Please try a simpler request."
        self.context.add_message(ROLE_USER, normalized_user_message)
        self.context.add_message(ROLE_ASSISTANT, error_response)
        self.logger.log_error(
            turn_id=turn_id,
            phase="agent.max_iterations",
            message=error_response,
            details={},
        )
        self.logger.finish_turn(
            turn_id,
            error_response,
            self.request_metrics,
            status="error",
            error={"phase": "agent.max_iterations", "message": error_response},
        )
        self._emit_turn_event(
            on_event,
            "turn_error",
            phase="agent.max_iterations",
            message=error_response,
        )
        self._emit_turn_event(
            on_event,
            "turn_completed",
            status="error",
            llm_call_count=len(self.request_metrics),
            tool_call_count=tool_call_count,
            tools_used=tools_used,
            skills_used=skills_used,
        )
        return error_response

    def _handle_exception(
        self,
        exc: Exception,
        turn_id: int,
        tool_call_count: int,
        tools_used: List[str],
        skills_used: List[str],
        turn_finished: bool,
        on_event: Optional[TurnActivityCallback] = None,
        phase: str = "agent.run",
    ) -> None:
        """Handle exception in agent loop.

        Args:
            exc: The exception that occurred
            turn_id: Turn ID for logging
            tool_call_count: Number of tool calls made
            tools_used: List of tools used
            skills_used: List of skills used
            turn_finished: Whether the turn was already finished
            on_event: Optional callback for events
            phase: Phase where exception occurred
        """
        self.logger.log_error(
            turn_id=turn_id,
            phase=phase,
            message=str(exc),
            details={"exception_type": type(exc).__name__},
        )
        if not turn_finished:
            self.logger.finish_turn(
                turn_id,
                "",
                self.request_metrics,
                status="error",
                error={"phase": phase, "message": str(exc)},
            )
        self._emit_turn_event(
            on_event,
            "turn_error",
            phase=phase,
            message=str(exc),
        )
        self._emit_turn_event(
            on_event,
            "turn_completed",
            status="error",
            llm_call_count=len(self.request_metrics),
            tool_call_count=tool_call_count,
            tools_used=tools_used,
            skills_used=skills_used,
        )

    def run(
        self,
        user_message: str,
        on_tool_call: Optional[Callable] = None,
        on_event: Optional[TurnActivityCallback] = None,
    ) -> str:
        """Main agent loop - process user message and return response.

        Args:
            user_message: The user's input message
            on_tool_call: Optional callback called with tool_name, args before execution
            on_event: Optional callback for user-safe live activity events

        Returns:
            The agent's final response as a string
        """
        self.request_metrics.clear()
        turn_id, tools_used, skills_used, messages, normalized_user_message = self._initialize_turn(
            user_message, on_event
        )
        tool_call_count = 0
        turn_finished = False

        try:
            for iteration in range(self.max_iterations):
                self._emit_turn_event(
                    on_event,
                    "llm_call_started",
                    iteration=iteration,
                    stream=False,
                    message_count=len(messages),
                    tool_schema_count=len(self._cached_tool_schemas),
                )
                response, metrics = self.llm.chat(
                    messages,
                    tools=self._cached_tool_schemas,
                    log_context={
                        "turn_id": turn_id,
                        "iteration": iteration,
                        "stream": False,
                        "request_kind": REQUEST_KIND_AGENT_TURN,
                    },
                )
                metrics.iteration = iteration
                self.request_metrics.append(metrics)
                self._record_prompt_metrics(metrics)
                messages.append(self._assistant_message_from_response(response))

                tool_calls = response.get("tool_calls", [])
                self._emit_turn_event(
                    on_event,
                    "llm_call_finished",
                    iteration=iteration,
                    stream=False,
                    **self._metrics_event_details(metrics),
                    has_tool_calls=bool(tool_calls),
                    tool_call_count=len(tool_calls),
                    result_kind="tool_calls" if tool_calls else "final_answer",
                )

                if not tool_calls:
                    final_response = response.get("content", "")
                    self._finalize_response(normalized_user_message, final_response)
                    self.logger.finish_turn(
                        turn_id, final_response, self.request_metrics, status="completed"
                    )
                    self._emit_turn_event(
                        on_event,
                        "turn_completed",
                        status="completed",
                        llm_call_count=len(self.request_metrics),
                        tool_call_count=tool_call_count,
                        tools_used=tools_used,
                        skills_used=skills_used,
                    )
                    turn_finished = True
                    return final_response

                tool_call_count += self._process_tool_calls(
                    tool_calls,
                    messages,
                    turn_id,
                    iteration,
                    on_tool_call,
                    on_event,
                    tools_used,
                    skills_used,
                )

            return self._handle_max_iterations(
                normalized_user_message,
                turn_id,
                tool_call_count,
                tools_used,
                skills_used,
                on_event,
            )
        except Exception as exc:
            self._handle_exception(
                exc, turn_id, tool_call_count, tools_used, skills_used, turn_finished, on_event, "agent.run"
            )
            raise

    def run_stream(
        self,
        user_message: str,
        on_tool_call: Optional[Callable] = None,
        on_event: Optional[TurnActivityCallback] = None,
    ):
        """Stream agent responses token-by-token.

        Args:
            user_message: The user's input message
            on_tool_call: Optional callback called with tool_name, args before execution
            on_event: Optional callback for user-safe live activity events

        Yields:
            Tokens/chunks of the response as they arrive
        """
        self.request_metrics.clear()
        turn_id, tools_used, skills_used, messages, normalized_user_message = self._initialize_turn(
            user_message, on_event
        )
        tool_call_count = 0
        turn_finished = False

        try:
            for iteration in range(self.max_iterations):
                self._emit_turn_event(
                    on_event,
                    "llm_call_started",
                    iteration=iteration,
                    stream=True,
                    message_count=len(messages),
                    tool_schema_count=len(self._cached_tool_schemas),
                )
                buffer = []
                current_role = "assistant"

                for chunk in self.llm.chat_stream(
                    messages,
                    tools=self._cached_tool_schemas,
                    log_context={
                        "turn_id": turn_id,
                        "iteration": iteration,
                        "stream": True,
                        "request_kind": REQUEST_KIND_AGENT_TURN,
                    },
                ):
                    if "role" in chunk:
                        current_role = chunk["role"]
                    if "delta" in chunk:
                        buffer.append(chunk["delta"])
                        yield chunk["delta"]

                stream_metrics = self.llm.get_stream_metrics()
                if stream_metrics:
                    stream_metrics.iteration = iteration
                    self.request_metrics.append(stream_metrics)
                    self._record_prompt_metrics(stream_metrics)

                tool_calls = self.llm.get_stream_tool_calls()
                metrics_for_event = stream_metrics or LLMMetrics()
                self._emit_turn_event(
                    on_event,
                    "llm_call_finished",
                    iteration=iteration,
                    stream=True,
                    **self._metrics_event_details(metrics_for_event),
                    has_tool_calls=bool(tool_calls),
                    tool_call_count=len(tool_calls),
                    result_kind="tool_calls" if tool_calls else "final_answer",
                )

                if not tool_calls:
                    final_response = "".join(buffer) if buffer else ""
                    self._finalize_response(normalized_user_message, final_response)
                    self.logger.finish_turn(
                        turn_id, final_response, self.request_metrics, status="completed"
                    )
                    self._emit_turn_event(
                        on_event,
                        "turn_completed",
                        status="completed",
                        llm_call_count=len(self.request_metrics),
                        tool_call_count=tool_call_count,
                        tools_used=tools_used,
                        skills_used=skills_used,
                    )
                    turn_finished = True
                    return

                messages.append(
                    self._assistant_message_from_response({
                        "role": current_role,
                        "content": "".join(buffer),
                        "tool_calls": tool_calls,
                    })
                )

                tool_call_count += self._process_tool_calls(
                    tool_calls,
                    messages,
                    turn_id,
                    iteration,
                    on_tool_call,
                    on_event,
                    tools_used,
                    skills_used,
                )

            error_response = "I reached the maximum number of iterations. Please try a simpler request."
            for char in error_response:
                yield char
            return self._handle_max_iterations(
                normalized_user_message,
                turn_id,
                tool_call_count,
                tools_used,
                skills_used,
                on_event,
            )
        except Exception as exc:
            self._handle_exception(
                exc, turn_id, tool_call_count, tools_used, skills_used, turn_finished, on_event, "agent.run_stream"
            )
            raise
