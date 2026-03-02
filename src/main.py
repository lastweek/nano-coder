"""Main CLI entry point for Nano-Coder."""

import os
import sys
import threading
import random
from pathlib import Path
from typing import List, Optional, Tuple

# Add parent directory to path to allow imports when running directly
if __name__ == "__main__" and __file__:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rich.console import Console, Group
from src.input_helper import InputHelper
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.live import Live

from src.context import Context
from src.llm import LLMClient
from src.tools import ToolRegistry
from tools.read import ReadTool
from tools.write import WriteTool
from tools.bash import BashTool
from src.agent import Agent
from src.metrics_display import display_metrics
from src.config import config
from src.mcp import MCPManager
from src.skills import LoadSkillTool, SkillManager
from src.commands import CommandRegistry
from src.commands import builtin

# Constants for loading indicator
LOADING_INDICATOR_ROTATION_INTERVAL = 0.8  # seconds between status word changes
DEFAULT_LOADING_STATUS_WORDS = [
    "Thinking...",
    "Processing...",
    "Analyzing...",
    "Generating...",
    "Working...",
    "Computing..."
]
REQUEST_TYPE_STREAMING = "streaming"
REQUEST_TYPE_NON_STREAMING = "non-streaming"


def print_banner(console: Console) -> None:
    """Print welcome banner."""
    from src.config import config

    # Build LLM info lines
    llm_info_lines = []
    if config.llm.base_url:
        llm_info_lines.append(Text(f"URL: {config.llm.base_url}", style="dim"))
    if config.llm.model:
        model_text = Text(f"Model: {config.llm.model}", style="dim")
        if config.llm.context_window:
            model_text.append(Text(f" (context: {config.llm.context_window:,} tokens)", style="dim"))
        llm_info_lines.append(model_text)
    if config.llm.provider:
        llm_info_lines.append(Text(f"Provider: {config.llm.provider}", style="dim"))

    # Build panel content
    panel_content = Text("Nano-Coder", style="bold cyan") + "\n" + \
                    Text("Minimalism Terminal Code Agent", style="dim")

    # Add LLM info if available
    if llm_info_lines:
        panel_content += Text("\n\n")
        panel_content += Text("\n").join(llm_info_lines)

    console.print(Panel(
        panel_content,
        border_style="cyan",
        padding=(1, 2)
    ))
    console.print("[dim]Type 'exit' or 'quit' to leave[/dim]\n")


def on_tool_call(console: Console, tool_name: str, args: dict) -> None:
    """Callback when a tool is called."""
    console.print(f"[dim]{_format_tool_call_line(tool_name, args)}[/dim]")


def _format_tool_call_line(tool_name: str, args: dict) -> str:
    """Format a tool call for display."""
    args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
    return f"  → {tool_name}({args_str})"


def _append_assistant_chunk(events: List[Tuple[str, str]], chunk: str) -> None:
    """Append streamed assistant text while preserving chronology."""
    if not chunk:
        return

    if events and events[-1][0] == "assistant":
        event_type, content = events[-1]
        events[-1] = (event_type, content + chunk)
        return

    events.append(("assistant", chunk))


def _build_stream_renderable(
    events: List[Tuple[str, str]],
    loading_text: Optional[Text] = None,
):
    """Build the current streaming preview renderable."""
    if not events:
        return loading_text if loading_text is not None else Text("")

    renderables = []
    for event_type, content in events:
        if event_type == "assistant":
            renderables.append(Markdown(content))
        elif event_type == "tool":
            renderables.append(Text(content, style="dim"))

    return Group(*renderables)


def create_loading_indicator() -> tuple:
    """Create a loading indicator with rotating status words.

    Returns:
        tuple: (Text object, list of status words)
    """
    return Text(DEFAULT_LOADING_STATUS_WORDS[0], style="dim italic"), DEFAULT_LOADING_STATUS_WORDS


def rotate_loading_indicator(
    live: Live,
    text_obj: Text,
    status_words: List[str],
    first_activity: List[bool]
) -> None:
    """Rotate loading indicator status words via background timer.

    Args:
        live: Rich Live display instance
        text_obj: Current Text object being displayed
        status_words: List of status word strings
        first_activity: Mutable list [bool] to track first output activity
    """
    # Stop once the stream has produced visible activity.
    if first_activity[0]:
        return

    # Pick random status word (different from current)
    current_text = text_obj.plain
    available_words = [w for w in status_words if w != current_text]
    new_word = random.choice(available_words)

    # Update display
    text_obj.plain = new_word
    live.update(text_obj, refresh=True)

    # Schedule next rotation
    timer = threading.Timer(
        LOADING_INDICATOR_ROTATION_INTERVAL,
        rotate_loading_indicator,
        args=(live, text_obj, status_words, first_activity)
    )
    timer.daemon = True
    timer.start()


def display_streaming_response(console: Console, agent, user_input: str) -> str:
    """Display streaming response with Claude-style loading indicator.

    Args:
        console: Rich console instance
        agent: Agent instance
        user_input: User's input message

    Returns:
        The complete response text
    """
    loading_text, status_words = create_loading_indicator()
    first_activity = [False]
    rotation_timer = None
    events: List[Tuple[str, str]] = []
    response_chunks: List[str] = []
    live_ref: List[Optional[Live]] = [None]

    def handle_tool_call(tool_name: str, args: dict) -> None:
        """Record tool calls in the live markdown preview."""
        first_activity[0] = True
        events.append(("tool", _format_tool_call_line(tool_name, args)))
        if live_ref[0] is not None:
            live_ref[0].update(
                _build_stream_renderable(events, loading_text),
                refresh=True,
            )

    stream = agent.run_stream(
        user_input,
        on_tool_call=handle_tool_call
    )

    try:
        with Live(
            _build_stream_renderable(events, loading_text),
            console=console,
            refresh_per_second=10,
            transient=True,
            auto_refresh=False,
        ) as live:
            live_ref[0] = live
            rotation_timer = threading.Timer(
                LOADING_INDICATOR_ROTATION_INTERVAL,
                rotate_loading_indicator,
                args=(live, loading_text, status_words, first_activity)
            )
            rotation_timer.daemon = True
            rotation_timer.start()

            for token in stream:
                first_activity[0] = True
                response_chunks.append(token)
                _append_assistant_chunk(events, token)
                live.update(
                    _build_stream_renderable(events, loading_text),
                    refresh=True,
                )
    finally:
        live_ref[0] = None
        if rotation_timer and rotation_timer.is_alive():
            rotation_timer.cancel()

    if events:
        console.print(_build_stream_renderable(events))

    return "".join(response_chunks)


def main() -> None:
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Load configuration
    enable_streaming = config.ui.enable_streaming
    provider = config.llm.provider

    # Use stderr for Rich output to avoid conflicts with prompt_toolkit using stdout
    import sys
    console = Console(stderr=True)

    # Check for API key (from config or environment variable)
    # Only check if provider requires it (ollama/local don't)
    if provider not in ("ollama", "local"):
        # Check if API key is in config or environment
        api_key_from_config = config.llm.api_key
        api_key_from_env = os.environ.get(f"{provider.upper()}_API_KEY") or os.environ.get("API_KEY")

        if not api_key_from_config and not api_key_from_env:
            console.print(f"[red]Error: API key not configured for provider '{provider}'[/red]")
            console.print(f"\n[dim]Set the API key in config.yaml:[/dim]")
            console.print(f"[dim]  llm:[/dim]")
            console.print(f"[dim]    provider: {provider}[/dim]")
            console.print(f"[dim]    api_key: your-key-here[/dim]\n")
            sys.exit(1)

    # Initialize components
    cwd = Path.cwd()
    context = Context.create(cwd=str(cwd))
    skill_manager = SkillManager(repo_root=cwd)
    skill_warnings = skill_manager.discover()

    for warning in skill_warnings:
        console.print(f"[yellow]Skill warning: {warning}[/yellow]")

    try:
        llm_client = LLMClient()
    except Exception as e:
        console.print(f"[red]Error initializing LLM client: {e}[/red]")
        sys.exit(1)

    # Register tools
    tools = ToolRegistry()
    tools.register(ReadTool())
    tools.register(WriteTool())
    tools.register(BashTool())
    tools.register(LoadSkillTool(skill_manager))

    # Initialize MCP manager and register MCP tools
    mcp_manager = None
    if config.mcp.servers:
        try:
            # Enable MCP debug mode via environment variable
            mcp_debug = os.environ.get("MCP_DEBUG", "").lower() in ("1", "true", "yes")

            if mcp_debug:
                console.print("[dim][MCP] Debug mode enabled[/dim]")

            # Show configured servers
            enabled_servers = [s.name for s in config.mcp.servers if s.enabled]
            if enabled_servers:
                if mcp_debug:
                    console.print(f"[dim][MCP] Configured servers: {', '.join(enabled_servers)}[/dim]")

            # Convert server configs to dicts for MCPManager
            servers_config = []
            for server in config.mcp.servers:
                servers_config.append({
                    "name": server.name,
                    "url": server.url,
                    "enabled": server.enabled,
                    "timeout": server.timeout
                })

            if mcp_debug:
                console.print("[dim][MCP] Creating MCP manager...[/dim]")

            mcp_manager = MCPManager(servers_config, debug=mcp_debug)

            if mcp_debug:
                console.print("[dim][MCP] Registering MCP tools...[/dim]")

            mcp_manager.register_tools(tools)

            # Log loaded MCP servers
            if enabled_servers:
                if mcp_debug:
                    console.print(f"[dim][MCP] Successfully loaded {len(enabled_servers)} MCP server(s)[/dim]")
                else:
                    console.print(f"[dim]Loaded {len(enabled_servers)} MCP server(s): {', '.join(enabled_servers)}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to initialize MCP manager: {e}[/yellow]")

    # Create agent
    agent = Agent(llm_client, tools, context, skill_manager=skill_manager)

    # Create command registry and register built-in commands
    registry = CommandRegistry()
    builtin.register_all(registry)

    # Create execution context for commands
    cmd_context = {
        "agent": agent,
        "mcp_manager": mcp_manager,
        "tools": tools,
        "skill_manager": skill_manager,
        "session_context": context,
    }

    # Extract command descriptions for completer
    command_descriptions = {}
    for cmd in registry.list_commands():
        command_descriptions[cmd.name] = cmd.short_desc or cmd.description

    # Get command names for completion
    command_names = registry.get_command_names()

    # Initialize input helper for bash-like editing with command completion
    input_helper = InputHelper(
        command_names=command_names,
        command_descriptions=command_descriptions
    )

    # Print banner
    print_banner(console)

    # Main REPL loop
    try:
        while True:
            try:
                # Get user input with bash-like editing
                user_input = input_helper.get_input("\nYou > ")

                # CRITICAL: Ensure we're on a fresh line after prompt_toolkit
                # This fixes the issue where subsequent responses don't display
                print()

                if not user_input.strip():
                    continue

                # Check for slash commands
                if registry.execute(user_input, console, cmd_context):
                    continue  # Command was handled

                if user_input.lower() in ['exit', 'quit']:
                    console.print("\n[yellow]Goodbye![/yellow]")
                    break

                # Process through agent
                console.print("\n[bold cyan]Agent[/bold cyan] >")

                if enable_streaming:
                    # Stream response token-by-token
                    response = display_streaming_response(console, agent, user_input)
                    # Display metrics after response (Live display already showed response)
                    display_metrics(console, agent.request_metrics, REQUEST_TYPE_STREAMING)
                else:
                    # Show loading indicator while agent processes
                    with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                        response = agent.run(
                            user_input,
                            on_tool_call=lambda name, args: on_tool_call(console, name, args)
                        )

                    # Display response
                    console.print("\n")
                    console.print(Markdown(response))
                    # Display metrics after response
                    display_metrics(console, agent.request_metrics, REQUEST_TYPE_NON_STREAMING)

            except KeyboardInterrupt:
                console.print("\n\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            except EOFError:
                console.print("\n\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
    finally:
        # Ensure logger is closed and logs are flushed
        if hasattr(agent, 'logger') and agent.logger:
            agent.logger.close()

        # Close MCP server connections
        if mcp_manager:
            mcp_manager.close_all()


if __name__ == "__main__":
    main()
