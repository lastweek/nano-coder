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
from src.turn_display import TurnProgressDisplay

REQUEST_TYPE_STREAMING = "streaming"
REQUEST_TYPE_NON_STREAMING = "non-streaming"
DEBUG_ENABLED_VALUES = ("1", "true", "yes")


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


def _final_response_from_context(agent, fallback: str) -> str:
    """Prefer the finalized assistant response stored in context."""
    messages = agent.context.get_messages()
    if messages and messages[-1]["role"] == "assistant":
        return messages[-1]["content"]
    return fallback


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
    display = TurnProgressDisplay(skill_debug=skill_debug, request_type=request_type)
    response = ""

    live_ref: list[Optional[Live]] = [None]
    animate_live = console.is_terminal

    def refresh_live() -> None:
        if live_ref[0] is not None:
            live_ref[0].update(display.render_live(), refresh=True)

    def handle_event(event) -> None:
        display.handle_event(event)
        refresh_live()

    try:
        with Live(
            display.render_live(),
            console=console,
            transient=False,
            auto_refresh=animate_live,
            refresh_per_second=12 if animate_live else 4,
        ) as live:
            live_ref[0] = live
            if enable_streaming:
                for token in agent.run_stream(user_input, on_event=handle_event):
                    display.append_stream_chunk(token)
                    refresh_live()
                response = _final_response_from_context(agent, display.final_response_text())
            else:
                response = agent.run(user_input, on_event=handle_event)
            live.update(display.render_persisted(), refresh=True)
    finally:
        live_ref[0] = None

    if response:
        console.print()
        console.print(Markdown(response))

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
        command_descriptions=command_descriptions,
        skill_names=[skill.name for skill in skill_manager.list_skills()],
    )
    cmd_context["input_helper"] = input_helper

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
                    response = run_agent_turn(
                        console,
                        agent,
                        user_input,
                        enable_streaming=True,
                        skill_debug=skill_debug,
                    )
                    display_metrics(console, agent.request_metrics, REQUEST_TYPE_STREAMING)
                else:
                    response = run_agent_turn(
                        console,
                        agent,
                        user_input,
                        enable_streaming=False,
                        skill_debug=skill_debug,
                    )
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
