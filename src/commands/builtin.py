"""Built-in slash commands."""

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from typing import Any, Optional

from src.commands.registry import (
    CommandHelpSpec,
    CommandSubcommandHelp,
    render_command_help,
    render_unknown_subcommand,
)
from src.config import config
from src.context_usage import build_context_usage_snapshot, format_token_count
from src.plan_mode import (
    build_plan_execution_message,
    create_session_plan,
    mark_plan_approved,
    mark_plan_executing,
    mark_plan_rejected,
)


def _get_skill_dependencies(console: Console, context: Any):
    """Get skill-related dependencies from command context."""
    skill_manager = context.get("skill_manager")
    session_context = context.get("session_context")

    if not skill_manager or not session_context:
        console.print("[yellow]Skills are not initialized[/yellow]")
        return None, None

    return skill_manager, session_context


def _print_skill_list(console: Console, skill_manager, session_context) -> None:
    """Render the discovered skills table."""
    skills = skill_manager.list_skills()
    if not skills:
        console.print("[yellow]No skills discovered[/yellow]")
        return

    active = set(session_context.get_active_skills())
    table = Table(title=f"Available Skills ({len(skills)})", show_header=True, header_style="bold cyan")
    table.add_column("Skill", style="green", width=20)
    table.add_column("Description", style="white", width=50)
    table.add_column("Source", style="dim", width=10)
    table.add_column("Catalog", style="magenta", width=8)
    table.add_column("Active", style="cyan", width=8)

    for skill in skills:
        table.add_row(
            skill.name,
            skill.short_description,
            skill.source,
            "yes" if skill.catalog_visible else "no",
            "yes" if skill.name in active else "no",
        )

    console.print(table)


def _render_resource_inventory(paths) -> str:
    """Render a list of resource paths for display."""
    if not paths:
        return "  - none"
    return "\n".join(f"  - {path}" for path in paths)


def _format_percentage(value: Optional[float]) -> str:
    """Format a percentage for context usage tables."""
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def _get_compaction_dependencies(console: Console, context: Any):
    """Get context-compaction dependencies from command context."""
    agent = context.get("agent")
    session_context = context.get("session_context")
    if agent is None or session_context is None:
        console.print("[red]Agent and session context are required for /compact[/red]")
        return None, None, None

    compaction_manager = getattr(agent, "context_compaction", None)
    if compaction_manager is None:
        console.print("[red]Context compaction is not initialized[/red]")
        return None, None, None

    return agent, session_context, compaction_manager


def _get_subagent_dependencies(console: Console, context: Any):
    """Get subagent-related dependencies from command context."""
    agent = context.get("agent")
    subagent_manager = context.get("subagent_manager")
    if agent is None or subagent_manager is None:
        console.print("[red]Agent and subagent manager are required for /subagent[/red]")
        return None, None
    return agent, subagent_manager


def _get_plan_dependencies(console: Console, context: Any):
    """Get plan-workflow dependencies from command context."""
    agent = context.get("agent")
    session_context = context.get("session_context")
    run_agent_turn = context.get("run_agent_turn_callback")
    set_tool_profile = context.get("set_tool_profile_callback")
    if agent is None or session_context is None or run_agent_turn is None or set_tool_profile is None:
        console.print("[red]Agent, session context, and plan callbacks are required for /plan[/red]")
        return None, None, None, None
    return agent, session_context, run_agent_turn, set_tool_profile


def _prompt_for_plan_decision(console: Console, context: Any) -> Optional[bool]:
    """Prompt for plan approval when an interactive input callback is available."""
    prompt_input = context.get("prompt_input_callback")
    if prompt_input is None or not console.is_terminal:
        return None

    while True:
        decision = prompt_input("\nPlan Review [accept/reject] > ").strip().lower()
        if decision in {"accept", "a", "yes", "y"}:
            return True
        if decision in {"reject", "r", "no", "n"}:
            return False
        console.print("[yellow]Please enter accept or reject[/yellow]")


def register_all(registry):
    """Register all built-in commands."""

    help_help_spec = CommandHelpSpec(
        summary="List all available slash commands. Slash commands execute locally and do not go through the agent.",
        usage=["/help"],
        examples=["/help"],
    )

    tool_help_spec = CommandHelpSpec(
        summary="List the tools available in the current session. An optional filter matches tool names and descriptions.",
        usage=["/tool", "/tool <filter>"],
        examples=["/tool", "/tool read", "/tool deepwiki"],
    )

    context_help_spec = CommandHelpSpec(
        summary="Show an estimated next-call baseline for context usage in the current session.",
        usage=["/context"],
        examples=["/context"],
        notes=[
            "Baseline excludes the next user message.",
            "Baseline excludes any explicit $skill preload for a future turn.",
            "Token counts are rough estimates, not tokenizer-exact values.",
        ],
    )

    compact_help_spec = CommandHelpSpec(
        summary="Inspect or manage session-local context compaction for the current session.",
        usage=[
            "/compact",
            "/compact show",
            "/compact now",
            "/compact auto on",
            "/compact auto off",
        ],
        examples=["/compact", "/compact help now", "/compact auto off"],
        notes=[
            "Compaction is session-local.",
            "There is no restore or reset command in v1.",
            "/compact without arguments shows status.",
        ],
        subcommands=[
            CommandSubcommandHelp(
                name="show",
                usage="/compact show",
                description="Show the current rolling summary and its metadata.",
                examples=["/compact show"],
            ),
            CommandSubcommandHelp(
                name="now",
                usage="/compact now",
                description="Compact immediately using the current session state.",
                examples=["/compact now"],
            ),
            CommandSubcommandHelp(
                name="auto on",
                usage="/compact auto on",
                description="Enable auto-compaction for the current session.",
                examples=["/compact auto on"],
            ),
            CommandSubcommandHelp(
                name="auto off",
                usage="/compact auto off",
                description="Disable auto-compaction for the current session.",
                examples=["/compact auto off"],
            ),
        ],
    )

    subagent_help_spec = CommandHelpSpec(
        summary="Inspect or run local delegated child-agent tasks for the current top-level session.",
        usage=[
            "/subagent",
            "/subagent run <task>",
            "/subagent show <id>",
        ],
        examples=["/subagent", "/subagent run audit the logging flow", "/subagent show sa_0001_abcd1234"],
        notes=[
            "/subagent without arguments lists runs created in the current session.",
            "Slash-command subagent runs do not modify the parent conversation history.",
        ],
        subcommands=[
            CommandSubcommandHelp(
                name="run",
                usage="/subagent run <task>",
                description="Run one delegated child agent synchronously and print its structured result.",
                examples=["/subagent run review the logging flow and summarize problems"],
            ),
            CommandSubcommandHelp(
                name="show",
                usage="/subagent show <id>",
                description="Show the stored metadata and report for one prior subagent run.",
                examples=["/subagent show sa_0001_abcd1234"],
            ),
        ],
    )

    plan_help_spec = CommandHelpSpec(
        summary="Start, inspect, approve, or clear the session-local planning workflow.",
        usage=[
            "/plan",
            "/plan start <task>",
            "/plan show",
            "/plan apply",
            "/plan exit",
            "/plan clear",
        ],
        examples=[
            "/plan",
            "/plan start add a safe plan mode to the CLI",
            "/plan show",
            "/plan apply",
        ],
        notes=[
            "/plan start enters planning mode and runs a planning turn immediately.",
            "Accepting a submitted plan switches back to build mode and executes it immediately.",
            "Rejecting a submitted plan returns to build mode but keeps the plan file on disk.",
        ],
        subcommands=[
            CommandSubcommandHelp(
                name="start",
                usage="/plan start <task>",
                description="Enter planning mode, create or reuse the canonical session plan file, and run a planning turn.",
                examples=["/plan start redesign the context command and list test coverage"],
            ),
            CommandSubcommandHelp(
                name="show",
                usage="/plan show",
                description="Show the current session plan metadata and persisted plan content.",
                examples=["/plan show"],
            ),
            CommandSubcommandHelp(
                name="apply",
                usage="/plan apply",
                description="Apply the current reviewed or approved plan and execute it in build mode.",
                examples=["/plan apply"],
            ),
            CommandSubcommandHelp(
                name="exit",
                usage="/plan exit",
                description="Leave planning mode and return to build mode without executing the current plan.",
                examples=["/plan exit"],
            ),
            CommandSubcommandHelp(
                name="clear",
                usage="/plan clear",
                description="Clear the active approved plan contract while leaving the plan artifact on disk.",
                examples=["/plan clear"],
            ),
        ],
    )

    mcp_help_spec = CommandHelpSpec(
        summary="List configured MCP servers and the tools they expose. Optionally filter to one server.",
        usage=["/mcp", "/mcp <server_name>"],
        examples=["/mcp", "/mcp deepwiki"],
    )

    skill_help_spec = CommandHelpSpec(
        summary="List, inspect, pin, unpin, or reload skills for the current session.",
        usage=[
            "/skill",
            "/skill use <name>",
            "/skill clear <name|all>",
            "/skill show <name>",
            "/skill reload",
        ],
        examples=["/skill", "/skill help use", "/skill show pdf"],
        notes=[
            "/skill without arguments lists discovered skills.",
            "Pinned skills stay active for future turns in the current session.",
        ],
        subcommands=[
            CommandSubcommandHelp(
                name="use",
                usage="/skill use <name>",
                description="Pin a skill for future turns in the current session.",
                examples=["/skill use pdf"],
            ),
            CommandSubcommandHelp(
                name="clear",
                usage="/skill clear <name|all>",
                description="Unpin one skill or clear all pinned skills.",
                examples=["/skill clear pdf", "/skill clear all"],
            ),
            CommandSubcommandHelp(
                name="show",
                usage="/skill show <name>",
                description="Show metadata, source path, and resource inventory for a skill.",
                examples=["/skill show pdf"],
            ),
            CommandSubcommandHelp(
                name="reload",
                usage="/skill reload",
                description="Rescan skill directories and refresh the discovered skill list.",
                examples=["/skill reload"],
            ),
        ],
    )

    @registry.register(
        "help",
        "Show available commands",
        args_description="",
        short_desc="Show all commands",
        help_spec=help_help_spec,
    )
    def cmd_help(console: Console, args: str, context: Any):
        """Show help for available commands."""
        commands = registry.list_commands()

        # Create commands table
        table = Table(title="Available Commands", show_header=True, header_style="bold cyan")
        table.add_column("Command", style="green", width=20)
        table.add_column("Description", style="white", width=40)
        table.add_column("Arguments", style="dim", width=30)

        for cmd in commands:
            args_text = cmd.args_description or ""
            table.add_row(f"/{cmd.name}", cmd.description, args_text)

        console.print(table)
        console.print("\n[dim]Type /<command> to execute. Commands don't go through the agent.[/dim]")

    @registry.register(
        "tool",
        "List all available tools",
        args_description="[filter]",
        short_desc="List available tools",
        help_spec=tool_help_spec,
    )
    def cmd_tool(console: Console, args: str, context: Any):
        """List available tools in the current context."""
        agent = context.get("agent")
        if not agent:
            console.print("[red]Agent not available[/red]")
            return

        tools = agent.tools._tools

        # Filter by args if provided
        filter_str = args.strip().lower() if args.strip() else None
        if filter_str:
            tools = {
                name: tool for name, tool in tools.items()
                if filter_str in name.lower() or filter_str in tool.description.lower()
            }

        # Create tools table
        table = Table(
            title=f"Available Tools ({len(tools)})",
            show_header=True,
            header_style="bold cyan"
        )
        table.add_column("Tool Name", style="green", width=30)
        table.add_column("Description", style="white", width=50)
        table.add_column("Parameters", style="dim", width=30)

        for name, tool in sorted(tools.items()):
            # Format parameters - handle both old format and JSON schema format
            if tool.parameters:
                if isinstance(tool.parameters, dict):
                    # JSON Schema format
                    properties = tool.parameters.get("properties", {})
                    param_names = list(properties.keys()) if properties else []
                else:
                    # Legacy format
                    param_names = list(tool.parameters.keys()) if tool.parameters else []

                params_str = ", ".join(param_names) if param_names else "none"
            else:
                params_str = "none"

            table.add_row(name, tool.description, params_str)

        console.print(table)

    @registry.register(
        "context",
        "Show estimated context usage for the next LLM call",
        args_description="",
        short_desc="Show context usage",
        help_spec=context_help_spec,
    )
    def cmd_context(console: Console, args: str, context: Any):
        """Show a repo-native next-call context usage estimate."""
        agent = context.get("agent")
        session_context = context.get("session_context")
        skill_manager = context.get("skill_manager")

        if agent is None or session_context is None:
            console.print("[red]Agent and session context are required for /context[/red]")
            return

        snapshot = build_context_usage_snapshot(agent, session_context, skill_manager)

        total_text = format_token_count(snapshot.context_window)
        used_text = format_token_count(snapshot.used_tokens)
        percentage_text = _format_percentage(snapshot.used_percentage)
        summary_body = (
            f"[bold]Model:[/bold] {snapshot.model}\n"
            f"[bold]Tokens:[/bold] {used_text} / {total_text} ({percentage_text})"
        )
        console.print(Panel(summary_body, title="Context Usage", border_style="cyan"))

        category_table = Table(title="Estimated usage by category", show_header=True, header_style="bold cyan")
        category_table.add_column("Category", style="white", width=20)
        category_table.add_column("Tokens", style="green", width=10)
        category_table.add_column("Percentage", style="magenta", width=10)

        for row in snapshot.categories:
            category_table.add_row(
                row.category,
                format_token_count(row.tokens),
                _format_percentage(row.percentage),
            )

        console.print(category_table)

        if snapshot.tools:
            tool_table = Table(title="Tool Schemas", show_header=True, header_style="bold cyan")
            tool_table.add_column("Tool", style="green", width=30)
            tool_table.add_column("Kind", style="white", width=18)
            tool_table.add_column("Tokens", style="magenta", width=10)
            for row in snapshot.tools:
                tool_table.add_row(row.name, row.kind, format_token_count(row.tokens))
            console.print(tool_table)

        skill_rows = [row for row in snapshot.skills if row.tokens > 0]
        if skill_rows:
            skill_table = Table(title="Skills", show_header=True, header_style="bold cyan")
            skill_table.add_column("Skill", style="green", width=25)
            skill_table.add_column("Source", style="white", width=10)
            skill_table.add_column("Usage", style="cyan", width=10)
            skill_table.add_column("Tokens", style="magenta", width=10)
            for row in skill_rows:
                skill_table.add_row(row.name, row.source, row.usage_type, format_token_count(row.tokens))
            console.print(skill_table)

        if snapshot.messages:
            message_table = Table(title="Messages", show_header=True, header_style="bold cyan")
            message_table.add_column("#", style="white", width=4)
            message_table.add_column("Role", style="green", width=10)
            message_table.add_column("Tokens", style="magenta", width=10)
            message_table.add_column("Preview", style="white", width=60)
            for row in snapshot.messages:
                message_table.add_row(str(row.index), row.role, format_token_count(row.tokens), row.preview)
            console.print(message_table)

        for note in snapshot.notes:
            console.print(f"[dim]{note}[/dim]")

    @registry.register(
        "subagent",
        "List or run local subagents",
        args_description="[run <task>|show <id>]",
        short_desc="Manage subagent runs",
        help_spec=subagent_help_spec,
    )
    def cmd_subagent(console: Console, args: str, context: Any):
        """List, run, or inspect local subagent runs."""
        agent, subagent_manager = _get_subagent_dependencies(console, context)
        if agent is None or subagent_manager is None:
            return

        if not subagent_manager.enabled:
            console.print("[yellow]Subagents are disabled in the current configuration[/yellow]")
            return

        raw_args = args.strip()
        if not raw_args:
            runs = subagent_manager.list_runs()
            if not runs:
                console.print("[yellow]No subagent runs in this session[/yellow]")
                return

            table = Table(title=f"Subagent Runs ({len(runs)})", show_header=True, header_style="bold cyan")
            table.add_column("ID", style="green", width=18)
            table.add_column("Label", style="white", width=20)
            table.add_column("Status", style="cyan", width=12)
            table.add_column("Duration", style="magenta", width=10)
            table.add_column("Summary", style="white", width=46)

            for run in runs:
                summary = run.result.summary if run.result else ""
                if len(summary) > 80:
                    summary = summary[:77] + "..."
                duration = f"{run.duration_s:.2f}s" if run.duration_s is not None else "-"
                table.add_row(run.subagent_id, run.label, run.status, duration, summary)

            console.print(table)
            return

        parts = raw_args.split(maxsplit=1)
        subcommand = parts[0].lower()
        remainder = parts[1].strip() if len(parts) > 1 else ""
        command = registry.get_command("subagent")

        if subcommand == "run":
            if not remainder:
                console.print("[yellow]Missing task for /subagent run[/yellow]")
                if command is not None:
                    render_command_help(console, command, "run")
                return

            request = subagent_manager.build_subagent_request({"task": remainder})
            result = subagent_manager.run_subagents(
                agent,
                [request],
                parent_turn_id=None,
            )[0]
            metadata = "\n".join([
                f"[bold]ID:[/bold] {result.subagent_id}",
                f"[bold]Label:[/bold] {result.label}",
                f"[bold]Status:[/bold] {result.status}",
                f"[bold]LLM Log:[/bold] {result.llm_log}",
                f"[bold]Events Log:[/bold] {result.events_log}",
            ])
            console.print(Panel(metadata, title="Subagent Result", border_style="cyan"))
            if result.summary:
                console.print(f"[bold]Summary:[/bold] {result.summary}")
            if result.report:
                console.print(result.report)
            if result.error:
                console.print(f"[yellow]{result.error}[/yellow]")
            return

        if subcommand == "show":
            if not remainder:
                console.print("[yellow]Missing subagent id for /subagent show[/yellow]")
                if command is not None:
                    render_command_help(console, command, "show")
                return

            run = subagent_manager.get_run(remainder)
            if run is None:
                console.print(f"[red]Unknown subagent id: {remainder}[/red]")
                return

            result = run.result
            duration_str = f"{run.duration_s:.2f}s" if run.duration_s is not None else "-"
            metadata_lines = [
                f"[bold]ID:[/bold] {run.subagent_id}",
                f"[bold]Label:[/bold] {run.label}",
                f"[bold]Status:[/bold] {run.status}",
                f"[bold]Task:[/bold] {run.task}",
                f"[bold]Started:[/bold] {run.started_at}",
                f"[bold]Ended:[/bold] {run.ended_at or '-'}",
                f"[bold]Duration:[/bold] {duration_str}",
            ]
            if result is not None:
                metadata_lines.extend([
                    f"[bold]Session Dir:[/bold] {result.session_dir}",
                    f"[bold]LLM Log:[/bold] {result.llm_log}",
                    f"[bold]Events Log:[/bold] {result.events_log}",
                    f"[bold]Tools Used:[/bold] {', '.join(result.tools_used) if result.tools_used else 'none'}",
                ])
            console.print(Panel("\n".join(metadata_lines), title="Subagent Run", border_style="cyan"))
            if result:
                if result.summary:
                    console.print(f"[bold]Summary:[/bold] {result.summary}")
                if result.report:
                    console.print(result.report)
                if result.error:
                    console.print(f"[yellow]{result.error}[/yellow]")
            return

        if command is not None:
            render_unknown_subcommand(console, command, subcommand)
            return

        console.print(f"[red]Unknown /subagent subcommand: {subcommand}[/red]")

    @registry.register(
        "plan",
        "Start, inspect, or apply the session-local planning workflow",
        args_description="[start <task>|show|apply|exit|clear]",
        short_desc="Manage the planning workflow",
        help_spec=plan_help_spec,
    )
    def cmd_plan(console: Console, args: str, context: Any):
        """Start, inspect, or apply the session-local planning workflow."""
        agent, session_context, run_agent_turn_callback, set_tool_profile_callback = _get_plan_dependencies(console, context)
        if (
            agent is None
            or session_context is None
            or run_agent_turn_callback is None
            or set_tool_profile_callback is None
        ):
            return

        if not config.plan.enabled:
            console.print("[yellow]Plan mode is disabled in the current configuration[/yellow]")
            return

        raw_args = args.strip()
        command = registry.get_command("plan")
        current_plan = session_context.get_current_plan()

        if not raw_args:
            active_plan = session_context.get_active_approved_plan()
            status_lines = [
                f"[bold]Mode:[/bold] {session_context.get_session_mode()}",
                f"[bold]Plan present:[/bold] {'yes' if current_plan is not None else 'no'}",
                f"[bold]Active contract:[/bold] {'yes' if active_plan is not None else 'no'}",
            ]
            if current_plan is not None:
                status_lines.extend(
                    [
                        f"[bold]Plan ID:[/bold] {current_plan.plan_id}",
                        f"[bold]Status:[/bold] {current_plan.status}",
                        f"[bold]Path:[/bold] {current_plan.file_path}",
                        f"[bold]Task:[/bold] {current_plan.task}",
                    ]
                )
            console.print(Panel("\n".join(status_lines), title="Plan Status", border_style="cyan"))
            return

        parts = raw_args.split(maxsplit=1)
        subcommand = parts[0].lower()
        remainder = parts[1].strip() if len(parts) > 1 else ""

        if subcommand == "start":
            if not remainder:
                console.print("[yellow]Missing task for /plan start[/yellow]")
                if command is not None:
                    render_command_help(console, command, "start")
                return

            session_context.set_session_mode("plan")
            plan = create_session_plan(
                session_context,
                task=remainder,
                plan_dir=config.plan.plan_dir,
            )
            set_tool_profile_callback("plan_main")
            logger = getattr(agent, "logger", None)
            if logger is not None:
                logger.log_plan_event(
                    turn_id=None,
                    stage="started",
                    plan_id=plan.plan_id,
                    status=plan.status,
                    file_path=plan.file_path,
                    task=plan.task,
                )

            console.print(Panel(
                f"[bold]Planning task:[/bold] {plan.task}\n[bold]Plan file:[/bold] {plan.file_path}",
                title="Planning Mode",
                border_style="cyan",
            ))
            run_agent_turn_callback(plan.task)

            submitted_plan = session_context.get_current_plan()
            if submitted_plan is None:
                console.print("[yellow]Planning finished without creating a plan artifact[/yellow]")
                session_context.set_session_mode("build")
                set_tool_profile_callback("build")
                return

            if submitted_plan.status != "ready_for_review":
                console.print(
                    "[yellow]Planning finished without submit_plan. "
                    "Review the plan draft with /plan show or continue planning.[/yellow]"
                )
                return

            if submitted_plan.report:
                console.print()
                console.print(Panel(Markdown(submitted_plan.report), title="Plan Review", border_style="cyan"))

            decision = _prompt_for_plan_decision(console, context)
            if decision is None:
                session_context.set_session_mode("build")
                set_tool_profile_callback("build")
                console.print(
                    "[yellow]Plan ready for review. Use /plan apply to execute it or /plan exit to leave planning mode.[/yellow]"
                )
                return

            if decision:
                approved_plan = mark_plan_approved(session_context)
                session_context.set_session_mode("build")
                set_tool_profile_callback("build")
                if logger is not None:
                    logger.log_plan_event(
                        turn_id=None,
                        stage="approved",
                        plan_id=approved_plan.plan_id,
                        status=approved_plan.status,
                        file_path=approved_plan.file_path,
                    )
                executing_plan = mark_plan_executing(session_context)
                if logger is not None:
                    logger.log_plan_event(
                        turn_id=None,
                        stage="execution_started",
                        plan_id=executing_plan.plan_id,
                        status=executing_plan.status,
                        file_path=executing_plan.file_path,
                    )
                console.print("[green]Plan accepted. Executing the approved plan.[/green]")
                run_agent_turn_callback(build_plan_execution_message(executing_plan))
                return

            rejected_plan = mark_plan_rejected(session_context)
            session_context.set_session_mode("build")
            set_tool_profile_callback("build")
            if logger is not None:
                logger.log_plan_event(
                    turn_id=None,
                    stage="rejected",
                    plan_id=rejected_plan.plan_id,
                    status=rejected_plan.status,
                    file_path=rejected_plan.file_path,
                )
            console.print("[yellow]Plan rejected. Returned to build mode.[/yellow]")
            return

        if subcommand == "show":
            if current_plan is None:
                console.print("[yellow]No session plan exists yet[/yellow]")
                return

            metadata = "\n".join(
                [
                    f"[bold]Plan ID:[/bold] {current_plan.plan_id}",
                    f"[bold]Status:[/bold] {current_plan.status}",
                    f"[bold]Task:[/bold] {current_plan.task}",
                    f"[bold]Path:[/bold] {current_plan.file_path}",
                    f"[bold]Approved:[/bold] {current_plan.approved_at or '-'}",
                ]
            )
            console.print(Panel(metadata, title="Session Plan", border_style="cyan"))
            if current_plan.summary:
                console.print(f"[bold]Summary:[/bold] {current_plan.summary}")
            if current_plan.content:
                console.print(Markdown(current_plan.content))
            else:
                console.print("[yellow]Plan file is currently empty[/yellow]")
            return

        if subcommand == "apply":
            if current_plan is None:
                console.print("[yellow]No session plan exists yet[/yellow]")
                return
            if current_plan.status not in {"ready_for_review", "approved", "executing"}:
                console.print(
                    f"[yellow]Current plan status '{current_plan.status}' cannot be applied[/yellow]"
                )
                return

            approved_plan = current_plan
            logger = getattr(agent, "logger", None)
            if session_context.active_approved_plan_id != current_plan.plan_id:
                approved_plan = mark_plan_approved(session_context)
                if logger is not None:
                    logger.log_plan_event(
                        turn_id=None,
                        stage="approved",
                        plan_id=approved_plan.plan_id,
                        status=approved_plan.status,
                        file_path=approved_plan.file_path,
                    )

            executing_plan = mark_plan_executing(session_context)
            session_context.set_session_mode("build")
            set_tool_profile_callback("build")
            if logger is not None:
                logger.log_plan_event(
                    turn_id=None,
                    stage="execution_started",
                    plan_id=executing_plan.plan_id,
                    status=executing_plan.status,
                    file_path=executing_plan.file_path,
                )
            console.print("[green]Executing the approved session plan.[/green]")
            run_agent_turn_callback(build_plan_execution_message(executing_plan))
            return

        if subcommand == "exit":
            session_context.set_session_mode("build")
            set_tool_profile_callback("build")
            console.print("[yellow]Returned to build mode.[/yellow]")
            return

        if subcommand == "clear":
            if current_plan is None or session_context.active_approved_plan_id is None:
                console.print("[yellow]No active approved plan contract to clear[/yellow]")
                return
            cleared_plan_id = session_context.active_approved_plan_id
            session_context.clear_active_plan_contract()
            logger = getattr(agent, "logger", None)
            if logger is not None:
                logger.log_plan_event(
                    turn_id=None,
                    stage="cleared",
                    plan_id=cleared_plan_id,
                    status=current_plan.status,
                    file_path=current_plan.file_path,
                )
            console.print("[yellow]Cleared the active approved plan contract for this session.[/yellow]")
            return

        if command is not None:
            render_unknown_subcommand(console, command, subcommand)
            return

        console.print(f"[red]Unknown /plan subcommand: {subcommand}[/red]")

    @registry.register(
        "compact",
        "Inspect or manage session-local context compaction",
        args_description="[show|now|auto on|auto off]",
        short_desc="Manage context compaction",
        help_spec=compact_help_spec,
    )
    def cmd_compact(console: Console, args: str, context: Any):
        """Inspect or manage rolling context compaction."""
        agent, session_context, compaction_manager = _get_compaction_dependencies(console, context)
        if agent is None or session_context is None or compaction_manager is None:
            return

        raw_args = args.strip()
        parts = raw_args.split()

        def render_status() -> None:
            snapshot = compaction_manager.render_status_snapshot(agent)
            context_window = snapshot["context_window"]
            total_text = format_token_count(context_window)
            used_text = format_token_count(snapshot["current_used_tokens"])
            percentage_text = _format_percentage(snapshot["current_used_percentage"])
            body = "\n".join([
                f"[bold]Auto-compaction:[/bold] {'on' if snapshot['auto_compaction_enabled'] else 'off'}",
                f"[bold]Configured by config:[/bold] {'on' if snapshot['configured_auto_compact'] else 'off'}",
                f"[bold]Threshold:[/bold] {snapshot['auto_compact_threshold'] * 100:.0f}%",
                f"[bold]Target after compaction:[/bold] {snapshot['target_usage_after_compaction'] * 100:.0f}%",
                f"[bold]Configured recent turns retained:[/bold] {snapshot['min_recent_turns']}",
                f"[bold]Effective recent turns retained:[/bold] {snapshot['effective_retained_turns']}",
                f"[bold]Current baseline:[/bold] {used_text} / {total_text} ({percentage_text})",
                (
                    f"[bold]Auto decision:[/bold] "
                    f"{'compact now' if snapshot['decision_should_compact'] else 'skip'} "
                    f"({snapshot['decision_reason']})"
                ),
                f"[bold]Decision detail:[/bold] {snapshot['decision_reason_text']}",
                f"[bold]Summary present:[/bold] {'yes' if snapshot['summary_present'] else 'no'}",
                f"[bold]Summary-covered turns:[/bold] {snapshot['summary_covered_turn_count']}",
                f"[bold]Raw retained turns:[/bold] {snapshot['raw_retained_turn_count']}",
            ])
            console.print(Panel(body, title="Context Compaction", border_style="cyan"))

        def render_manual_steps(plan, details: dict[str, Any], result) -> None:
            complete_turns = details.get("complete_turn_count", 0)
            evictable_turns = details.get("evictable_turn_count", 0)
            configured_retained = details.get(
                "configured_min_recent_turns",
                compaction_manager.policy.min_recent_turns,
            )
            effective_retained = details.get("effective_retained_turns", len(plan.retained_turns))
            baseline_tokens = format_token_count(details.get("current_used_tokens"))

            lines = [
                "[bold]Manual compaction steps[/bold]",
                (
                    "1. Inspect current context: "
                    f"{complete_turns} complete turn(s), baseline {baseline_tokens}."
                ),
                (
                    "2. Apply adaptive retention: "
                    f"keep {effective_retained} raw turn(s) "
                    f"(configured cap {configured_retained}), "
                    f"leaving {evictable_turns} evictable older turn(s)."
                ),
            ]

            if result.status == "skipped":
                lines.append(
                    "3. Stop: "
                    f"{compaction_manager.describe_reason(result.reason, result.details)}"
                )
            else:
                lines.append(
                    "3. Select turns to summarize: "
                    f"force mode compacts all {len(plan.turns_to_compact)} evictable older turn(s)."
                )
                if result.error:
                    lines.append(
                        "4. Summary generation failed, so Nano-Coder used the deterministic fallback summary."
                    )
                else:
                    lines.append(
                        "4. Generate or update the rolling summary with the configured model."
                    )
                lines.append(
                    "5. Replace older raw history with the retained recent turns and store the new summary."
                )
                lines.append(
                    "6. Recalculate baseline usage: "
                    f"{format_token_count(result.before_tokens)} -> {format_token_count(result.after_tokens)}."
                )

            console.print(Panel("\n".join(lines), title="Manual Compaction", border_style="cyan"))

        if not parts:
            render_status()
            return

        if parts[0] == "show":
            summary = session_context.get_summary()
            if summary is None:
                console.print("[yellow]No compacted summary is available for this session[/yellow]")
                return

            metadata = "\n".join([
                f"[bold]Updated:[/bold] {summary.updated_at}",
                f"[bold]Compactions:[/bold] {summary.compaction_count}",
                f"[bold]Covered turns:[/bold] {summary.covered_turn_count}",
                f"[bold]Covered messages:[/bold] {summary.covered_message_count}",
            ])
            console.print(Panel(metadata, title="Compacted Summary", border_style="cyan"))
            console.print(summary.rendered_text)
            return

        if parts[0] == "now":
            logger = getattr(agent, "logger", None)
            plan_preview = compaction_manager._build_plan(agent, force=True)
            preview_details = compaction_manager._build_debug_details(
                plan_preview,
                current_used_tokens=plan_preview.before_tokens,
                threshold_tokens=(
                    int(plan_preview.context_window * compaction_manager.policy.auto_compact_threshold)
                    if plan_preview.context_window is not None
                    else None
                ),
                force=True,
            )
            if logger is not None:
                logger.log_context_compaction_event(
                    turn_id=None,
                    stage="started",
                    reason="manual_command",
                    covered_turn_count=len(plan_preview.turns_to_compact),
                    retained_turn_count=len(plan_preview.retained_turns),
                    **preview_details,
                )

            result = compaction_manager.compact_now(agent, "manual_command", force=True)
            if result.status == "skipped":
                if logger is not None:
                    logger.log_context_compaction_event(
                        turn_id=None,
                        stage="skipped",
                        reason=result.reason,
                        reason_text=compaction_manager.describe_reason(result.reason, result.details),
                        **result.details,
                    )
                console.print(
                    "[yellow]Context compaction skipped:[/yellow] "
                    f"{compaction_manager.describe_reason(result.reason, result.details)}"
                )
                render_manual_steps(plan_preview, result.details, result)
                for line in compaction_manager.render_debug_lines(result.details):
                    console.print(f"[dim]{line}[/dim]")
                return

            if result.error:
                if logger is not None:
                    logger.log_context_compaction_event(
                        turn_id=None,
                        stage="failed",
                        reason=result.reason,
                        error=result.error,
                        **result.details,
                    )
                console.print(f"[yellow]Compaction used fallback summary:[/yellow] {result.error}")
            elif logger is not None:
                logger.log_context_compaction_event(
                    turn_id=None,
                    stage="completed",
                    reason=result.reason,
                    covered_turn_count=result.covered_turn_count,
                    retained_turn_count=result.retained_turn_count,
                    before_tokens=result.before_tokens,
                    after_tokens=result.after_tokens,
                    **result.details,
                )

            render_manual_steps(plan_preview, result.details, result)
            console.print(
                "[green]Context compacted:[/green] "
                f"{result.covered_turn_count} turns summarized, "
                f"{result.retained_turn_count} recent turns kept, "
                f"{format_token_count(result.before_tokens)} -> {format_token_count(result.after_tokens)}"
            )
            return

        if parts[0] == "auto" and len(parts) == 2:
            if parts[1] == "on":
                session_context.set_auto_compaction(True)
                console.print("[green]Auto-compaction enabled for this session[/green]")
                return
            if parts[1] == "off":
                session_context.set_auto_compaction(False)
                console.print("[green]Auto-compaction disabled for this session[/green]")
                return

        command = registry.get_command("compact")
        if parts[0] == "auto" and command is not None:
            console.print("[yellow]Missing /compact auto action: expected on or off[/yellow]")
            render_command_help(console, command, "auto")
            return

        if command is not None:
            render_unknown_subcommand(console, command, parts[0])
            return

        console.print(f"[red]Unknown /compact subcommand: {parts[0]}[/red]")

    @registry.register(
        "mcp",
        "List MCP servers and their tools",
        args_description="[server_name]",
        short_desc="List MCP servers",
        help_spec=mcp_help_spec,
    )
    def cmd_mcp(console: Console, args: str, context: Any):
        """List MCP servers and their tools."""
        mcp_manager = context.get("mcp_manager")
        if not mcp_manager:
            console.print("[yellow]No MCP manager initialized[/yellow]")
            return

        agent = context.get("agent")
        if not agent:
            console.print("[red]Agent not available[/red]")
            return

        servers = mcp_manager._servers
        filter_str = args.strip().lower() if args.strip() else None

        if filter_str and filter_str not in servers:
            console.print(f"[yellow]Server '{filter_str}' not found[/yellow]")
            console.print(f"[dim]Available servers: {', '.join(servers.keys())}[/dim]")
            return

        # Show all servers or filtered
        servers_to_show = {filter_str: servers[filter_str]} if filter_str else servers

        for server_name, server in servers_to_show.items():
            # Get tools from this server
            server_tools = [
                tool for tool in agent.tools._tools.values()
                if hasattr(tool, '_server') and tool._server.name == server_name
            ]

            # Server panel
            panel_title = Text()
            panel_title.append(f"MCP Server: {server_name}", style="bold cyan")
            panel_title.append(f" ({server.url})", style="dim")

            console.print(Panel(
                f"Status: {'[green]Connected[/green]' if server_tools else '[yellow]No tools[/yellow]'}\n"
                f"Tools: {len(server_tools)}\n"
                f"Timeout: {server.timeout}s",
                title=panel_title,
                border_style="cyan"
            ))

            # List tools if any
            if server_tools:
                tool_table = Table(show_header=True, header_style="dim")
                tool_table.add_column("Tool", style="green", width=25)
                tool_table.add_column("Description", style="white")

                for tool in server_tools:
                    tool_name = tool._tool_def.get("name", "")
                    tool_desc = tool._tool_def.get("description", "")
                    tool_table.add_row(tool_name, tool_desc)

                console.print(tool_table)

            console.print()  # Blank line between servers

    @registry.register(
        "skill",
        "List, inspect, pin, or reload skills",
        args_description="[use|clear|show|reload] [name]",
        short_desc="Manage skills",
        help_spec=skill_help_spec,
    )
    def cmd_skill(console: Console, args: str, context: Any):
        """Manage discovered skills."""
        skill_manager, session_context = _get_skill_dependencies(console, context)
        if not skill_manager or not session_context:
            return

        raw_args = args.strip()
        if not raw_args:
            _print_skill_list(console, skill_manager, session_context)
            return

        parts = raw_args.split(maxsplit=1)
        subcommand = parts[0].lower()
        remainder = parts[1].strip() if len(parts) > 1 else ""

        if subcommand == "use":
            if not remainder:
                console.print("[yellow]Missing skill name for /skill use[/yellow]")
                command = registry.get_command("skill")
                if command is not None:
                    render_command_help(console, command, "use")
                return

            skill = skill_manager.get_skill(remainder)
            if skill is None:
                console.print(f"[red]Unknown skill: {remainder}[/red]")
                return

            session_context.activate_skill(skill.name)
            console.print(f"[green]Pinned skill:[/green] {skill.name}")
            return

        if subcommand == "clear":
            if not remainder:
                console.print("[yellow]Missing target for /skill clear[/yellow]")
                command = registry.get_command("skill")
                if command is not None:
                    render_command_help(console, command, "clear")
                return

            if remainder.lower() == "all":
                cleared = len(session_context.get_active_skills())
                session_context.clear_skills()
                console.print(f"[green]Cleared {cleared} pinned skill(s)[/green]")
                return

            if remainder not in session_context.get_active_skills():
                console.print(f"[yellow]Skill not pinned: {remainder}[/yellow]")
                return

            session_context.deactivate_skill(remainder)
            console.print(f"[green]Unpinned skill:[/green] {remainder}")
            return

        if subcommand == "show":
            if not remainder:
                console.print("[yellow]Missing skill name for /skill show[/yellow]")
                command = registry.get_command("skill")
                if command is not None:
                    render_command_help(console, command, "show")
                return

            skill = skill_manager.get_skill(remainder)
            if skill is None:
                console.print(f"[red]Unknown skill: {remainder}[/red]")
                return

            line_info = f"{skill.body_line_count}"
            if skill.is_oversized:
                line_info += " (over recommended 500 lines; consider moving detail into references/)"

            body = "\n".join([
                f"[bold]Description:[/bold] {skill.description}",
                f"[bold]Source:[/bold] {skill.source}",
                f"[bold]Catalog Visible:[/bold] {'yes' if skill.catalog_visible else 'no'}",
                f"[bold]Skill File:[/bold] {skill.skill_file}",
                f"[bold]Body Lines:[/bold] {line_info}",
                "",
                "[bold]Scripts:[/bold]",
                _render_resource_inventory(skill.scripts),
                "",
                "[bold]References:[/bold]",
                _render_resource_inventory(skill.references),
                "",
                "[bold]Assets:[/bold]",
                _render_resource_inventory(skill.assets),
            ])

            console.print(Panel(body, title=f"Skill: {skill.name}", border_style="cyan"))
            return

        if subcommand == "reload":
            warnings = skill_manager.discover()
            removed = []
            for skill_name in list(session_context.get_active_skills()):
                if skill_manager.get_skill(skill_name) is None:
                    session_context.deactivate_skill(skill_name)
                    removed.append(skill_name)

            for warning in warnings:
                console.print(f"[yellow]Skill warning: {warning}[/yellow]")

            for skill_name in removed:
                console.print(f"[yellow]Removed missing pinned skill: {skill_name}[/yellow]")

            input_helper = context.get("input_helper")
            if input_helper is not None:
                input_helper.update_skills([skill.name for skill in skill_manager.list_skills()])

            console.print(f"[green]Reloaded {len(skill_manager.list_skills())} skill(s)[/green]")
            return

        command = registry.get_command("skill")
        if command is not None:
            render_unknown_subcommand(console, command, subcommand)
            return

        console.print(f"[red]Unknown /skill subcommand: {subcommand}[/red]")
