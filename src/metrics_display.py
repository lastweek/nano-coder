"""Display LLM metrics using Rich panels."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.metrics import LLMMetrics
from typing import List


def display_metrics(console: Console, metrics_list: List[LLMMetrics], request_type: str) -> None:
    """Display metrics in a Rich panel after agent response completes.

    Args:
        console: Rich console instance
        metrics_list: List of LLMMetrics for each request
        request_type: "streaming" or "non-streaming"
    """
    if not metrics_list:
        return

    # Calculate aggregates
    total_requests = len(metrics_list)
    total_tokens = sum(m.total_tokens for m in metrics_list)
    total_cached = sum(m.cached_tokens for m in metrics_list)
    total_duration = sum(m.duration for m in metrics_list)

    # Create table
    table = Table(show_header=True, header_style="bold cyan", show_edge=False)
    table.add_column("Metric", style="dim", width=20)
    table.add_column("Value", justify="right")

    if request_type == "streaming" and any(m.request_type == "streaming" for m in metrics_list):
        # Streaming mode
        table.add_row("Request Type", "Streaming")
        table.add_row("Total Requests", str(total_requests))
        table.add_row("Total Duration", f"{total_duration:.3f}s")

        # Show first streaming request metrics
        first_stream = next((m for m in metrics_list if m.request_type == "streaming"), None)
        if first_stream and first_stream.ttft > 0:
            table.add_row("TTFT (First Token)", f"{first_stream.ttft:.3f}s")
        if first_stream and first_stream.tpot > 0:
            table.add_row("TPOT (Per Token)", f"{first_stream.tpot:.3f}s")
        if first_stream and first_stream.tokens_per_second > 0:
            table.add_row("Tokens/s", f"{first_stream.tokens_per_second:.1f}")
    else:
        # Non-streaming mode
        table.add_row("Request Type", "Non-streaming")
        table.add_row("Total Requests", str(total_requests))
        table.add_row("Total Duration", f"{total_duration:.3f}s")

        if len(metrics_list) == 1:
            table.add_row("Request Duration", f"{metrics_list[0].duration:.3f}s")

    # Add token usage
    table.add_row("", "")  # Spacer
    table.add_row("Prompt Tokens", str(sum(m.prompt_tokens for m in metrics_list)))
    table.add_row("Completion Tokens", str(sum(m.completion_tokens for m in metrics_list)))
    table.add_row("Total Tokens", str(total_tokens))

    if total_cached > 0:
        table.add_row("Cached Tokens", f"[green]{total_cached}[/green]")
        total_prompt = sum(m.prompt_tokens for m in metrics_list)
        if total_prompt > 0:
            cache_pct = (total_cached / total_prompt) * 100
            table.add_row("Cache Hit Rate", f"[green]{cache_pct:.1f}%[/green]")

    # Add model info
    table.add_row("", "")  # Spacer
    table.add_row("Model", metrics_list[0].model)
    table.add_row("Provider", metrics_list[0].provider)

    # Create panel
    panel = Panel(
        table,
        title="[bold]LLM Request Metrics[/bold]",
        title_align="left",
        border_style="cyan",
        padding=(0, 1)
    )

    console.print("\n")
    console.print(panel)
