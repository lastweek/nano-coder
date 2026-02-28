"""Agent orchestration for Nano-Coder."""

import json
from typing import Callable, Optional, List, Dict
from src.tools import ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL
from src.logger import ChatLogger


class Agent:
    """Main agent orchestration using ReAct loop."""

    DEFAULT_MAX_ITERATIONS = 10

    def __init__(self, llm_client, tools, context):
        """Initialize the agent.

        Args:
            llm_client: LLMClient instance
            tools: ToolRegistry instance
            context: Context instance
        """
        self.llm = llm_client
        self.tools = tools
        self.context = context
        self.max_iterations = self.DEFAULT_MAX_ITERATIONS
        self.logger = ChatLogger(context.session_id)

        # Share logger with LLM client for request/response logging
        if hasattr(self.llm, 'logger'):
            self.llm.logger = self.logger

        # Cache for hot-path optimization
        self._cached_system_message = None
        self._cached_tool_schemas = None

    def _get_system_message(self) -> Dict:
        """Get the cached system message for the agent."""
        if self._cached_system_message is None:
            # Dynamically build tool descriptions
            tool_descriptions = "\n".join(
                f"- {tool.name}: {tool.description}"
                for tool in self.tools._tools.values()
            )

            content = f"""You are a helpful coding assistant with access to tools.

Working directory: {self.context.cwd}

Available tools:
{tool_descriptions}

When asked to do something that requires tools, use them. Always explain what you're doing before using a tool.
Think step by step. If you make a mistake, try to recover.

Be concise and helpful."""

            self._cached_system_message = {"role": ROLE_SYSTEM, "content": content}

        return self._cached_system_message

    def _build_messages(self, user_message: str) -> List[Dict]:
        """Build message list for LLM API call."""
        messages = [self._get_system_message()]

        for msg in self.context.get_messages():
            messages.append(msg)

        messages.append({"role": ROLE_USER, "content": user_message})

        return messages

    def _execute_tool_call(self, tool_call: Dict, parsed_args: Optional[Dict] = None) -> Dict:
        """Execute a single tool call.

        Args:
            tool_call: Tool call dict with id, name, and arguments
            parsed_args: Pre-parsed arguments (to avoid duplicate JSON parsing)

        Returns:
            Tool result message for LLM
        """
        tool_name = tool_call["name"]
        tool_id = tool_call["id"]

        try:
            # Use pre-parsed args if provided, otherwise parse
            if parsed_args is None:
                arguments = json.loads(tool_call["arguments"])
            else:
                arguments = parsed_args

            # Get tool and execute
            tool = self.tools.get(tool_name)
            if not tool:
                result = {"error": f"Unknown tool: {tool_name}"}
            else:
                result_obj = tool.execute(self.context, **arguments)
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

    def run(self, user_message: str, on_tool_call: Optional[Callable] = None) -> str:
        """Main agent loop - process user message and return response.

        Args:
            user_message: The user's input message
            on_tool_call: Optional callback called with tool_name, args before execution

        Returns:
            The agent's final response as a string
        """
        # Log user message
        self.logger.log_user_message(user_message)

        # Build initial messages
        messages = self._build_messages(user_message)

        # Cache tool schemas for hot-path optimization
        if self._cached_tool_schemas is None:
            self._cached_tool_schemas = self.tools.get_tool_schemas()

        # ReAct loop
        for iteration in range(self.max_iterations):
            # Get LLM response
            response = self.llm.chat(messages, tools=self._cached_tool_schemas)

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
                # Save to context
                self.context.add_message(ROLE_USER, user_message)
                self.context.add_message(ROLE_ASSISTANT, final_response)
                # Log final agent response
                self.logger.log_agent_response(final_response)
                return final_response

            # Process each tool call
            for tool_call in tool_calls:
                # Parse arguments once
                parsed_args = json.loads(tool_call["arguments"])

                # Notify callback if provided
                if on_tool_call:
                    on_tool_call(tool_call["name"], parsed_args)

                # Log tool call
                self.logger.log_tool_call(tool_call["name"], parsed_args)

                # Execute tool and add result to messages
                tool_result = self._execute_tool_call(tool_call, parsed_args)
                messages.append(tool_result)

                # Log tool result
                result_content = json.loads(tool_result["content"])
                self.logger.log_tool_result(tool_call["name"], result_content)

        # Max iterations reached
        error_response = "I reached the maximum number of iterations. Please try a simpler request."
        self.context.add_message(ROLE_USER, user_message)
        self.context.add_message(ROLE_ASSISTANT, error_response)
        # Log agent response (error case)
        self.logger.log_agent_response(error_response)
        return error_response
