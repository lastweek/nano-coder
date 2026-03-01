"""Display LLM metrics using Rich panels."""

from rich.console import Console
from rich.text import Text
from src.metrics import LLMMetrics
from typing import List


def display_metrics(console: Console, metrics_list: List[LLMMetrics], request_type: str) -> None:
    """Display metrics as a single dimmed line after agent response completes.

    Args:
        console: Rich console instance
        metrics_list: List of LLMMetrics for each request
        request_type: "streaming" or "non-streaming"
    """
    if not metrics_list:
        return

    # Calculate aggregates
    total_prompt_tokens = sum(m.prompt_tokens for m in metrics_list)
    total_completion_tokens = sum(m.completion_tokens for m in metrics_list)
    total_duration = sum(m.duration for m in metrics_list)

    # Get model info from first request
    provider = metrics_list[0].provider if metrics_list else ""
    model = metrics_list[0].model if metrics_list else ""

    # Build simplified metrics line
    parts = []
    parts.append(f"{total_prompt_tokens} prompt tokens")
    parts.append(f"{total_completion_tokens} completion tokens")
    parts.append(f"{total_duration:.2f}s")

    # Add TTFT/TPOT for streaming
    if request_type == "streaming" and any(m.request_type == "streaming" for m in metrics_list):
        first_stream = next((m for m in metrics_list if m.request_type == "streaming"), None)
        if first_stream and first_stream.ttft > 0:
            parts.append(f"TTFT {first_stream.ttft:.2f}s")
        if first_stream and first_stream.tpot > 0:
            parts.append(f"TPOT {first_stream.tpot:.2f}s")

    # Add model info at the end
    if model and provider:
        parts.append(f"{model} ({provider})")

    # Create dimmed line
    text = Text(" • ".join(parts), style="dim")
    console.print(text)
