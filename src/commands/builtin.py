"""Built-in slash commands."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from typing import Any


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
    table.add_column("Active", style="cyan", width=8)

    for skill in skills:
        table.add_row(
            skill.name,
            skill.short_description,
            skill.source,
            "yes" if skill.name in active else "no",
        )

    console.print(table)


def _render_resource_inventory(paths) -> str:
    """Render a list of resource paths for display."""
    if not paths:
        return "  - none"
    return "\n".join(f"  - {path}" for path in paths)


def register_all(registry):
    """Register all built-in commands."""

    @registry.register(
        "help",
        "Show available commands",
        args_description="",
        short_desc="Show all commands"
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
        short_desc="List available tools"
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
        "mcp",
        "List MCP servers and their tools",
        args_description="[server_name]",
        short_desc="List MCP servers"
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
        short_desc="Manage skills"
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
                console.print("[yellow]Usage: /skill use <name>[/yellow]")
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
                console.print("[yellow]Usage: /skill clear <name|all>[/yellow]")
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
                console.print("[yellow]Usage: /skill show <name>[/yellow]")
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

            console.print(f"[green]Reloaded {len(skill_manager.list_skills())} skill(s)[/green]")
            return

        console.print(f"[red]Unknown /skill subcommand: {subcommand}[/red]")
        console.print("[dim]Usage: /skill [use|clear|show|reload] [name][/dim]")
