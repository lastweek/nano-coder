"""Agent orchestration for Nano-Coder."""

import json
from typing import Callable, Optional, List, Dict
from src.tools import ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL
from src.logger import ChatLogger
from src.metrics import LLMMetrics
from src.config import config


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
        self.max_iterations = config.agent.max_iterations
        self.logger = ChatLogger(context.session_id)

        # Share logger with LLM client for request/response logging
        if hasattr(self.llm, 'logger'):
            self.llm.logger = self.logger

        # Accumulate metrics across LLM requests
        self.request_metrics: List[LLMMetrics] = []

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

    def _get_system_message(self) -> Dict:
        """Build the system message for the current turn."""
        return {"role": ROLE_SYSTEM, "content": self._build_system_prompt()}

    def _build_system_prompt(self) -> str:
        """Build the full system prompt including dynamic skill context."""
        sections = [self._cached_system_prompt_base]

        skill_catalog = self._build_skill_catalog_section()
        if skill_catalog:
            sections.append(skill_catalog)

        pinned_skills = self._build_pinned_skills_section()
        if pinned_skills:
            sections.append(pinned_skills)

        return "\n\n".join(section for section in sections if section)

    def _build_skill_catalog_section(self) -> str:
        """Build the compact catalog of available skills."""
        if not self.skill_manager:
            return ""

        skills = self.skill_manager.list_skills()
        if not skills:
            return ""

        lines = ["Available skills (load with load_skill when relevant):"]
        for skill in skills:
            lines.append(f"- {skill.name}: {skill.short_description}")

        return "\n".join(lines)

    def _build_pinned_skills_section(self) -> str:
        """Build the pinned-skill prompt blocks for the current session."""
        if not self.skill_manager:
            return ""

        pinned_blocks = []
        for skill_name in self.context.get_active_skills():
            block = self.skill_manager.format_skill_for_prompt(skill_name)
            if block:
                pinned_blocks.append(block)

        if not pinned_blocks:
            return ""

        return "Pinned skills active for this session:\n\n" + "\n\n".join(pinned_blocks)

    def _build_messages(self, user_message: str) -> List[Dict]:
        """Build message list for LLM API call."""
        messages = [self._get_system_message()]

        for msg in self.context.get_messages():
            messages.append(msg)

        messages.append({"role": ROLE_USER, "content": user_message})

        return messages

    def _execute_tool_call(self, tool_call: Dict, parsed_args: Dict) -> Dict:
        """Execute a single tool call.

        Args:
            tool_call: Tool call dict with id, name, and arguments
            parsed_args: Pre-parsed tool arguments

        Returns:
            Tool result message for LLM
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

        # Format as tool result message
        return {
            "role": ROLE_TOOL,
            "tool_call_id": tool_id,
            "content": json.dumps(result)
        }

    def _process_tool_calls(
        self,
        tool_calls: List[Dict],
        messages: List[Dict],
        on_tool_call: Optional[Callable] = None
    ) -> None:
        """Process multiple tool calls and add results to messages.

        Args:
            tool_calls: List of tool call dicts
            messages: Message list to append results to
            on_tool_call: Optional callback for notification
        """
        for tool_call in tool_calls:
            parsed_args = json.loads(tool_call["arguments"])

            if on_tool_call:
                on_tool_call(tool_call["name"], parsed_args)

            self.logger.log_tool_call(tool_call["name"], parsed_args)

            tool_result = self._execute_tool_call(tool_call, parsed_args)
            messages.append(tool_result)

            result_content = json.loads(tool_result["content"])
            self.logger.log_tool_result(tool_call["name"], result_content)

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

    def run(self, user_message: str, on_tool_call: Optional[Callable] = None) -> str:
        """Main agent loop - process user message and return response.

        Args:
            user_message: The user's input message
            on_tool_call: Optional callback called with tool_name, args before execution

        Returns:
            The agent's final response as a string
        """
        # Clear previous metrics
        self.request_metrics.clear()

        # Build initial messages
        messages = self._build_messages(user_message)

        # Cache tool schemas for hot-path optimization
        if self._cached_tool_schemas is None:
            self._cached_tool_schemas = self.tools.get_tool_schemas()

        # ReAct loop
        for iteration in range(self.max_iterations):
            # Get LLM response with metrics
            response, metrics = self.llm.chat(messages, tools=self._cached_tool_schemas)
            metrics.iteration = iteration
            self.request_metrics.append(metrics)

            # Add assistant response to history
            messages.append({
                "role": response["role"],
                "content": response.get("content", "")
            })

            # Check for tool calls
            tool_calls = response.get("tool_calls", [])
            if not tool_calls:
                # No more tool calls, return final response
                final_response = response.get("content", "")
                self._finalize_response(user_message, final_response)
                return final_response

            # Process each tool call
            self._process_tool_calls(tool_calls, messages, on_tool_call)

        # Max iterations reached
        error_response = "I reached the maximum number of iterations. Please try a simpler request."
        self.context.add_message(ROLE_USER, user_message)
        self.context.add_message(ROLE_ASSISTANT, error_response)
        # Note: This is a locally generated error, not an LLM response
        # Log as error type for debugging
        self.logger.log({"type": "error", "message": error_response})
        return error_response

    def run_stream(self, user_message: str, on_tool_call: Optional[Callable] = None):
        """Stream agent responses token-by-token.

        Args:
            user_message: The user's input message
            on_tool_call: Optional callback called with tool_name, args before execution

        Yields:
            Tokens/chunks of the response as they arrive
        """
        # Clear previous metrics
        self.request_metrics.clear()

        # Build initial messages
        messages = self._build_messages(user_message)

        # Cache tool schemas for hot-path optimization
        if self._cached_tool_schemas is None:
            self._cached_tool_schemas = self.tools.get_tool_schemas()

        # ReAct loop with streaming
        for iteration in range(self.max_iterations):
            # Stream LLM response
            buffer = []
            current_role = "assistant"

            for chunk in self.llm.chat_stream(messages, tools=self._cached_tool_schemas):
                if "role" in chunk:
                    current_role = chunk["role"]

                if "delta" in chunk:
                    # Yield content tokens directly
                    buffer.append(chunk["delta"])
                    yield chunk["delta"]

                if "tool_calls" in chunk or "finish_reason" in chunk:
                    # Accumulate tool calls during streaming
                    # Tool calls come through differently in streaming mode
                    # They're accumulated in the LLM client
                    pass

            # Get streaming metrics
            stream_metrics = self.llm.get_stream_metrics()
            if stream_metrics:
                stream_metrics.iteration = iteration
                self.request_metrics.append(stream_metrics)

            # Get tool calls from the stream (no duplicate API call needed)
            tool_calls = self.llm.get_stream_tool_calls()

            if not tool_calls:
                # No more tool calls, this is the final response
                final_response = "".join(buffer) if buffer else ""
                self._finalize_response(user_message, final_response)
                return  # End of stream

            # We have tool calls - add the streamed content to messages
            messages.append({
                "role": current_role,
                "content": "".join(buffer)
            })

            # Process each tool call
            self._process_tool_calls(tool_calls, messages, on_tool_call)

        # Max iterations reached
        error_response = "I reached the maximum number of iterations. Please try a simpler request."
        for char in error_response:
            yield char
        self.context.add_message(ROLE_USER, user_message)
        self.context.add_message(ROLE_ASSISTANT, error_response)
        # Note: This is a locally generated error, not an LLM response
        # Log as error type for debugging
        self.logger.log({"type": "error", "message": error_response})
