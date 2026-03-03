"""Main CLI entry point for Nano-Coder."""

import os
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path to allow imports when running directly
if __name__ == "__main__" and __file__:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rich.console import Console
from src.input_helper import InputHelper
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.live import Live

from src.context import Context
from src.llm import LLMClient
from src.metrics import LLMMetrics
from src.plan_mode import create_session_plan
from src.tools import build_tool_registry
from src.agent import Agent
from src.config import config
from src.mcp import MCPManager
from src.skills import SkillManager
from src.commands import CommandRegistry
from src.commands import builtin
from src.statusline import build_prompt_toolbar
from src.turn_display import TurnProgressDisplay
from src.turn_controls import LiveTurnControls
from src.subagents import SubagentManager

REQUEST_TYPE_STREAMING = "streaming"
REQUEST_TYPE_NON_STREAMING = "non-streaming"
DEBUG_ENABLED_VALUES = ("1", "true", "yes")
NANO_CODER_WORDMARK = r""" _   _                        ____          _
| \ | | __ _ _ __   ___      / ___|___   __| | ___ _ __
|  \| |/ _` | '_ \ / _ \____| |   / _ \ / _` |/ _ \ '__|
| |\  | (_| | | | | (_) |____| |__| (_) | (_| |  __/ |
|_| \_|\__,_|_| |_|\___/      \____\___/ \__,_|\___|_|
"""


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
    panel_content = Text(NANO_CODER_WORDMARK.rstrip("\n"), style="bold cyan")
    panel_content.append("\n")
    panel_content.append("Minimalism Terminal Code Agent", style="dim")

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


def _final_response_from_context(agent, fallback: str) -> str:
    """Prefer the finalized assistant response stored in context."""
    messages = agent.context.get_messages()
    if messages and messages[-1]["role"] == "assistant":
        return messages[-1]["content"]
    return fallback


def _calculate_aggregate_stream_tpot(metrics_list: list[LLMMetrics]) -> Optional[float]:
    """Aggregate TPOT across streamed requests using existing per-request semantics."""
    weighted_sum = 0.0
    weight_total = 0

    for metric in metrics_list:
        if metric.request_type != REQUEST_TYPE_STREAMING:
            continue
        if metric.tpot <= 0 or metric.token_count < 2:
            continue

        token_intervals = metric.token_count - 1
        weighted_sum += metric.tpot * token_intervals
        weight_total += token_intervals

    if weight_total == 0:
        return None
    return weighted_sum / weight_total


def display_metrics(console: Console, metrics_list: list[LLMMetrics], request_type: str) -> None:
    """Display aggregate request metrics after the final answer."""
    if not metrics_list:
        return

    total_prompt_tokens = sum(metric.prompt_tokens for metric in metrics_list)
    total_completion_tokens = sum(metric.completion_tokens for metric in metrics_list)
    total_duration = sum(metric.duration for metric in metrics_list)
    provider = metrics_list[0].provider if metrics_list else ""
    model = metrics_list[0].model if metrics_list else ""

    parts = [
        f"{total_prompt_tokens} prompt tokens",
        f"{total_completion_tokens} completion tokens",
        f"{total_duration:.2f}s",
    ]

    if request_type == REQUEST_TYPE_STREAMING:
        first_stream = next(
            (metric for metric in metrics_list if metric.request_type == REQUEST_TYPE_STREAMING),
            None,
        )
        if first_stream and first_stream.ttft > 0:
            parts.append(f"TTFT {first_stream.ttft:.2f}s")
        aggregate_stream_tpot = _calculate_aggregate_stream_tpot(metrics_list)
        if aggregate_stream_tpot is not None:
            parts.append(f"TPOT {aggregate_stream_tpot:.2f}s")

    if model and provider:
        parts.append(f"{model} ({provider})")

    console.print(Text(" • ".join(parts), style="dim"))


def run_agent_turn(
    console: Console,
    agent,
    user_input: str,
    *,
    enable_streaming: bool,
    skill_debug: bool,
) -> str:
    """Run one agent turn with a live chronological activity feed."""
    request_type = REQUEST_TYPE_STREAMING if enable_streaming else REQUEST_TYPE_NON_STREAMING
    display = TurnProgressDisplay(
        session_context=agent.context,
        skill_debug=skill_debug,
        request_type=request_type,
        live_activity_mode=config.ui.live_activity_mode,
        live_activity_details=config.ui.live_activity_details,
    )
    live_controls = LiveTurnControls(display)
    response = ""
    animate_live = console.is_terminal

    def handle_event(event) -> None:
        display.handle_event(event)

    try:
        with Live(
            display,
            console=console,
            transient=False,
            auto_refresh=animate_live,
            refresh_per_second=12 if animate_live else 4,
        ) as live:
            if animate_live:
                live_controls.start()
            if enable_streaming:
                for token in agent.run_stream(user_input, on_event=handle_event):
                    display.append_stream_chunk(token)
                response = _final_response_from_context(agent, display.final_response_text())
            else:
                response = agent.run(user_input, on_event=handle_event)
            live.update(display.render_persisted(), refresh=True)
    finally:
        live_controls.stop()

    if response:
        console.print()
        console.print(Markdown(response))

    return response


def execute_user_turn(
    console: Console,
    agent,
    user_input: str,
    *,
    enable_streaming: bool,
    skill_debug: bool,
) -> str:
    """Run one user-visible turn and print its final metrics summary."""
    console.print("\n[bold cyan]Agent[/bold cyan] >")
    response = run_agent_turn(
        console,
        agent,
        user_input,
        enable_streaming=enable_streaming,
        skill_debug=skill_debug,
    )
    request_type = REQUEST_TYPE_STREAMING if enable_streaming else REQUEST_TYPE_NON_STREAMING
    display_metrics(console, agent.request_metrics, request_type)
    return response


def display_streaming_response(console: Console, agent, user_input: str) -> str:
    """Backward-compatible streaming helper."""
    return run_agent_turn(
        console,
        agent,
        user_input,
        enable_streaming=True,
        skill_debug=False,
    )


def main() -> None:
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Load configuration
    enable_streaming = config.ui.enable_streaming
    provider = config.llm.provider
    skill_debug = os.environ.get("SKILL_DEBUG", "").lower() in DEBUG_ENABLED_VALUES

    # Use stderr for Rich output to avoid conflicts with prompt_toolkit using stdout
    import sys
    console = Console(stderr=True)

    if skill_debug:
        console.print("[dim][SKILL] Debug mode enabled[/dim]")

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

    if skill_debug:
        discovered_skills = skill_manager.list_skills()
        console.print(f"[dim][SKILL] discovered {len(discovered_skills)} skill(s)[/dim]")
        for skill in discovered_skills:
            console.print(
                "[dim]"
                f"[SKILL] discovered {skill.name} "
                f"(source={skill.source}, catalog={'yes' if skill.catalog_visible else 'no'})"
                "[/dim]"
            )

    try:
        llm_client = LLMClient()
    except Exception as e:
        console.print(f"[red]Error initializing LLM client: {e}[/red]")
        sys.exit(1)

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

            # Log loaded MCP servers
            if enabled_servers:
                if mcp_debug:
                    console.print(f"[dim][MCP] Successfully loaded {len(enabled_servers)} MCP server(s)[/dim]")
                else:
                    console.print(f"[dim]Loaded {len(enabled_servers)} MCP server(s): {', '.join(enabled_servers)}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to initialize MCP manager: {e}[/yellow]")

    subagent_manager = SubagentManager()
    tools = build_tool_registry(
        skill_manager=skill_manager,
        mcp_manager=mcp_manager,
        subagent_manager=subagent_manager,
        include_subagent_tool=config.subagents.enabled,
        tool_profile="build",
    )

    # Create agent
    agent = Agent(
        llm_client,
        tools,
        context,
        skill_manager=skill_manager,
        subagent_manager=subagent_manager,
    )

    # Create command registry and register built-in commands
    registry = CommandRegistry()
    builtin.register_all(registry)

    def set_tool_profile_callback(tool_profile: str) -> None:
        """Rebuild the parent tool registry for the requested session tool profile."""
        rebuilt_tools = build_tool_registry(
            skill_manager=skill_manager,
            mcp_manager=mcp_manager,
            subagent_manager=subagent_manager,
            include_subagent_tool=True,
            tool_profile=tool_profile,
        )
        agent.set_tool_registry(rebuilt_tools)
        cmd_context["tools"] = rebuilt_tools

    # Create execution context for commands
    cmd_context = {
        "agent": agent,
        "mcp_manager": mcp_manager,
        "tools": tools,
        "skill_manager": skill_manager,
        "session_context": context,
        "subagent_manager": subagent_manager,
    }

    # Extract command descriptions for completer
    command_descriptions = {}
    for cmd in registry.list_commands():
        command_descriptions[cmd.name] = cmd.short_desc or cmd.description

    # Get command names for completion
    command_names = registry.get_command_names()

    def build_idle_statusline():
        """Build the idle prompt statusline from session state and default view settings."""
        return build_prompt_toolbar(
            context,
            view_mode=config.ui.live_activity_mode,
            detail_mode=config.ui.live_activity_details,
        )

    def toggle_plan_mode() -> None:
        """Toggle the top-level session mode between build and plan."""
        if not config.plan.enabled:
            return

        if context.get_session_mode() == "plan":
            context.set_session_mode("build")
            set_tool_profile_callback("build")
            return

        if context.get_current_plan() is None:
            create_session_plan(
                context,
                task="Interactive planning session",
                plan_dir=config.plan.plan_dir,
            )
        context.set_session_mode("plan")
        set_tool_profile_callback("plan_main")

    # Initialize input helper for bash-like editing with command completion
    input_helper = InputHelper(
        command_names=command_names,
        command_descriptions=command_descriptions,
        skill_names=[skill.name for skill in skill_manager.list_skills()],
        bottom_toolbar_callback=build_idle_statusline,
        toggle_plan_mode_callback=toggle_plan_mode,
    )
    cmd_context["input_helper"] = input_helper
    cmd_context["prompt_input_callback"] = input_helper.get_input
    cmd_context["set_tool_profile_callback"] = set_tool_profile_callback
    cmd_context["run_agent_turn_callback"] = lambda prompt: execute_user_turn(
        console,
        agent,
        prompt,
        enable_streaming=enable_streaming,
        skill_debug=skill_debug,
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

                response = execute_user_turn(
                    console,
                    agent,
                    user_input,
                    enable_streaming=enable_streaming,
                    skill_debug=skill_debug,
                )

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
