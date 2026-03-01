"""Main CLI entry point for Nano-Coder."""

import os
import sys
import threading
import random
from pathlib import Path
from typing import List

# Add parent directory to path to allow imports when running directly
if __name__ == "__main__" and __file__:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rich.console import Console
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
    console.print(Panel(
        Text("Nano-Coder", style="bold cyan") + "\n" +
        Text("Minimalism Terminal Code Agent", style="dim"),
        border_style="cyan",
        padding=(1, 2)
    ))
    console.print("[dim]Type 'exit' or 'quit' to leave[/dim]\n")


def on_tool_call(console: Console, tool_name: str, args: dict) -> None:
    """Callback when a tool is called."""
    # Format args for display
    args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
    console.print(f"[dim]  → {tool_name}({args_str})[/dim]")


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
    first_token_arrived: List[bool]
) -> None:
    """Rotate loading indicator status words via background timer.

    Args:
        live: Rich Live display instance
        text_obj: Current Text object being displayed
        status_words: List of status word strings
        first_token_arrived: Mutable list [bool] to track token arrival
    """
    # Stop if first token arrived
    if first_token_arrived[0]:
        return

    # Pick random status word (different from current)
    current_text = text_obj.plain
    available_words = [w for w in status_words if w != current_text]
    new_word = random.choice(available_words)

    # Update display
    text_obj.plain = new_word
    live.update(text_obj)

    # Schedule next rotation
    timer = threading.Timer(
        LOADING_INDICATOR_ROTATION_INTERVAL,
        rotate_loading_indicator,
        args=(live, text_obj, status_words, first_token_arrived)
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
    # Create loading indicator
    loading_text, status_words = create_loading_indicator()
    first_token_arrived = [False]  # Mutable list for closure access
    rotation_timer = None

    with Live(loading_text, console=console, refresh_per_second=10) as live:
        # Start rotation timer for loading indicator
        rotation_timer = threading.Timer(
            LOADING_INDICATOR_ROTATION_INTERVAL,
            rotate_loading_indicator,
            args=(live, loading_text, status_words, first_token_arrived)
        )
        rotation_timer.daemon = True
        rotation_timer.start()

        # Accumulate tokens when they arrive
        accumulated = Text()

        for token in agent.run_stream(
            user_input,
            on_tool_call=lambda name, args: on_tool_call(console, name, args)
        ):
            # First token: stop loading, switch to token display
            if not first_token_arrived[0]:
                first_token_arrived[0] = True

                # Cancel rotation timer
                if rotation_timer and rotation_timer.is_alive():
                    rotation_timer.cancel()

                # Clear loading indicator and switch to accumulated text
                live.update(accumulated)

            # Accumulate tokens
            accumulated.append(token)

        # Ensure timer is cancelled (handles edge cases)
        if rotation_timer and rotation_timer.is_alive():
            rotation_timer.cancel()

    return accumulated.plain  # Return full text for markdown rendering


def main() -> None:
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Load configuration
    enable_streaming = config.ui.enable_streaming
    provider = config.llm.provider

    console = Console()

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

    # Create agent
    agent = Agent(llm_client, tools, context)

    # Print banner
    print_banner(console)

    # Main REPL loop
    try:
        while True:
            try:
                # Get user input
                user_input = console.input("\n[bold green]You[/bold green] > ")

                if not user_input.strip():
                    continue

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


if __name__ == "__main__":
    main()