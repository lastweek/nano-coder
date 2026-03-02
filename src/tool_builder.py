"""Shared tool registry construction helpers."""

from __future__ import annotations

from src.config import config
from src.tools import ToolRegistry
from src.tools.bash import BashTool
from src.tools.read import ReadTool
from src.tools.skill import LoadSkillTool
from src.tools.subagent import RunSubagentTool
from src.tools.write import WriteTool


def build_tool_registry(
    *,
    skill_manager,
    mcp_manager=None,
    subagent_manager=None,
    include_subagent_tool: bool = True,
) -> ToolRegistry:
    """Build the standard tool registry for a parent or child agent."""
    registry = ToolRegistry()
    registry.register(ReadTool())
    registry.register(WriteTool())
    registry.register(BashTool())
    registry.register(LoadSkillTool(skill_manager))

    if mcp_manager is not None:
        mcp_manager.register_tools(registry)

    if include_subagent_tool and subagent_manager is not None and config.subagents.enabled:
        registry.register(RunSubagentTool(subagent_manager))

    return registry


def clone_tool_registry(source: ToolRegistry, *, include_subagent_tool: bool = True) -> ToolRegistry:
    """Clone a registry by reusing tool instances from an existing registry."""
    registry = ToolRegistry()
    for tool in source._tools.values():
        if not include_subagent_tool and tool.name == "run_subagent":
            continue
        registry.register(tool)
    return registry
