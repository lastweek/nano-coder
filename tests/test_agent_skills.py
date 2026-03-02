"""Tests for agent integration with skills."""

import json
from pathlib import Path
from types import SimpleNamespace

from src.agent import Agent
from src.context import Context
from src.skills import LoadSkillTool, SkillManager
from src.tools import ToolRegistry


def write_skill(skill_dir: Path, body: str = "Prefer visual PDF checks.") -> Path:
    """Create a minimal skill bundle."""
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\n"
        "name: pdf\n"
        "description: Handle PDFs well\n"
        "metadata:\n"
        "  short-description: PDF workflows\n"
        "---\n\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return skill_file


class StubLLM:
    """Stub LLM that records messages and returns scripted responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.logger = None

    def chat(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        return self.responses.pop(0), SimpleNamespace(iteration=None)


def test_system_prompt_includes_skill_catalog_but_not_body_for_unpinned_skill(temp_dir):
    """The agent prompt should list available skills without injecting full bodies."""
    repo_root = temp_dir / "repo"
    write_skill(repo_root / ".nano-coder" / "skills" / "pdf")
    manager = SkillManager(repo_root=repo_root, user_root=temp_dir / "user-skills")
    manager.discover()

    tools = ToolRegistry()
    tools.register(LoadSkillTool(manager))
    llm = StubLLM([{"role": "assistant", "content": "Done"}])
    context = Context.create(cwd=str(repo_root))
    agent = Agent(llm, tools, context, skill_manager=manager)

    agent.run("help")

    system_prompt = llm.calls[0]["messages"][0]["content"]
    assert "Available skills (load with load_skill when relevant):" in system_prompt
    assert "- pdf: PDF workflows" in system_prompt
    assert "Prefer visual PDF checks." not in system_prompt


def test_pinned_skill_body_is_included_in_system_prompt(temp_dir):
    """Pinned skills should inject full instructions into every turn."""
    repo_root = temp_dir / "repo"
    write_skill(repo_root / ".nano-coder" / "skills" / "pdf", body="Always inspect layout before edits.")
    manager = SkillManager(repo_root=repo_root, user_root=temp_dir / "user-skills")
    manager.discover()

    tools = ToolRegistry()
    tools.register(LoadSkillTool(manager))
    llm = StubLLM([{"role": "assistant", "content": "Done"}])
    context = Context.create(cwd=str(repo_root))
    context.activate_skill("pdf")
    agent = Agent(llm, tools, context, skill_manager=manager)

    agent.run("help")

    system_prompt = llm.calls[0]["messages"][0]["content"]
    assert "Pinned skills active for this session:" in system_prompt
    assert "[Skill: pdf]" in system_prompt
    assert "Always inspect layout before edits." in system_prompt


def test_load_skill_result_is_ephemeral_to_current_turn(temp_dir):
    """Agent-loaded skill content should be available in-turn but not persisted."""
    repo_root = temp_dir / "repo"
    write_skill(repo_root / ".nano-coder" / "skills" / "pdf", body="Use pypdf when layout is irrelevant.")
    manager = SkillManager(repo_root=repo_root, user_root=temp_dir / "user-skills")
    manager.discover()

    tools = ToolRegistry()
    tools.register(LoadSkillTool(manager))
    llm = StubLLM(
        [
            {
                "role": "assistant",
                "content": "Loading the PDF skill.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "load_skill",
                        "arguments": json.dumps({"skill_name": "pdf"}),
                    }
                ],
            },
            {"role": "assistant", "content": "Used the skill."},
        ]
    )
    context = Context.create(cwd=str(repo_root))
    agent = Agent(llm, tools, context, skill_manager=manager)

    response = agent.run("help with pdfs")

    assert response == "Used the skill."
    assert len(llm.calls) == 2
    second_call_messages = llm.calls[1]["messages"]
    tool_messages = [message for message in second_call_messages if message["role"] == "tool"]
    assert tool_messages
    assert "Skill: pdf" in tool_messages[0]["content"]
    assert len(context.messages) == 2
    assert context.messages[0] == {"role": "user", "content": "help with pdfs"}
    assert context.messages[1] == {"role": "assistant", "content": "Used the skill."}
    assert all("Skill: pdf" not in str(message["content"]) for message in context.messages)
