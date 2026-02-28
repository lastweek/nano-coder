"""Main CLI entry point for Nano-Coder."""

import os
import sys
from pathlib import Path

# Add parent directory to path to allow imports when running directly
if __name__ == "__main__" and __file__:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from src.context import Context
from src.llm import LLMClient
from src.tools import ToolRegistry
from tools.read import ReadTool
from tools.write import WriteTool
from tools.bash import BashTool
from src.agent import Agent


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


def main() -> None:
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    console = Console()

    # Check for provider configuration
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()

    # Map provider to its API key env var
    api_key_vars = {
        "openai": "OPENAI_API_KEY",
        "azure": "AZURE_API_KEY",
        "custom": "CUSTOM_API_KEY",
    }

    # Only check for API key if provider requires it (ollama/local don't)
    if provider not in ("ollama", "local"):
        api_key_var = api_key_vars.get(provider, "API_KEY")
        if not os.environ.get(api_key_var):
            console.print(f"[red]Error: {api_key_var} environment variable not set[/red]")
            console.print(f"\n[dim]Create a .env file with your {provider} API key:[/dim]")
            console.print(f"[dim]  {api_key_var}=your-key-here[/dim]\n")
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
            console.print("\n[bold cyan]Agent[/bold cyan] > ", end="")
            response = agent.run(
                user_input,
                on_tool_call=lambda name, args: on_tool_call(console, name, args)
            )

            # Display response
            console.print(Markdown(response))

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
        except EOFError:
            console.print("\n\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()