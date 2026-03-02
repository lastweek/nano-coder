"""Skill discovery and loading for Nano-Coder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

import yaml

from src.tools import Tool, ToolResult

SkillSource = Literal["repo", "user"]

MAX_SKILL_BODY_LINES = 500


@dataclass
class SkillSpec:
    """A discovered skill bundle."""

    name: str
    description: str
    short_description: str
    body: str
    root_dir: Path
    skill_file: Path
    source: SkillSource
    scripts: List[Path]
    references: List[Path]
    assets: List[Path]

    @property
    def body_line_count(self) -> int:
        """Return the body line count for context budgeting."""
        return len(self.body.splitlines())

    @property
    def is_oversized(self) -> bool:
        """Return whether the body exceeds the recommended line budget."""
        return self.body_line_count > MAX_SKILL_BODY_LINES


class SkillManager:
    """Discover, inspect, and format Codex-style skill bundles."""

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        user_root: Optional[Path] = None,
    ) -> None:
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.user_root = (user_root or Path.home() / ".nano-coder" / "skills").expanduser().resolve()
        self.repo_skills_root = self.repo_root / ".nano-coder" / "skills"
        self._skills: Dict[str, SkillSpec] = {}
        self._warnings: List[str] = []

    def discover(self) -> List[str]:
        """Discover skills under the configured roots and return warnings."""
        skills: Dict[str, SkillSpec] = {}
        warnings: List[str] = []

        discovery_roots: List[tuple[SkillSource, Path]] = [
            ("user", self.user_root),
            ("repo", self.repo_skills_root),
        ]

        for source, root in discovery_roots:
            if not root.exists():
                continue

            for skill_file in sorted(root.rglob("SKILL.md")):
                spec, skill_warnings = self._load_skill_file(skill_file, source)
                warnings.extend(skill_warnings)
                if spec is None:
                    continue

                previous = skills.get(spec.name)
                if previous is not None:
                    warnings.append(
                        f"Duplicate skill '{spec.name}': {spec.skill_file} overrides {previous.skill_file}"
                    )
                skills[spec.name] = spec

        self._skills = skills
        self._warnings = warnings
        return list(warnings)

    def list_skills(self) -> List[SkillSpec]:
        """Return discovered skills."""
        return list(self._skills.values())

    def get_skill(self, name: str) -> Optional[SkillSpec]:
        """Return a discovered skill by name."""
        return self._skills.get(name)

    def get_warnings(self) -> List[str]:
        """Return warnings from the last discovery run."""
        return list(self._warnings)

    def format_skill_for_tool(self, name: str) -> str:
        """Format a skill payload for the agent tool result."""
        skill = self.get_skill(name)
        if skill is None:
            raise KeyError(name)
        return self._format_skill_payload(skill)

    def format_skill_for_prompt(self, name: str) -> Optional[str]:
        """Format a pinned skill block for the system prompt."""
        skill = self.get_skill(name)
        if skill is None:
            return None

        lines = [
            f"[Skill: {skill.name}]",
            f"Description: {skill.description}",
            "Instructions:",
            skill.body,
        ]

        resources = self._resource_lines(skill)
        if resources:
            lines.extend(["Resources:"] + resources)

        return "\n".join(lines).strip()

    def _load_skill_file(
        self,
        skill_file: Path,
        source: SkillSource,
    ) -> tuple[Optional[SkillSpec], List[str]]:
        warnings: List[str] = []

        try:
            raw_text = skill_file.read_text(encoding="utf-8")
        except Exception as exc:
            return None, [f"Failed to read skill file {skill_file}: {exc}"]

        try:
            metadata, body = self._parse_skill_markdown(raw_text)
        except ValueError as exc:
            return None, [f"Skipping invalid skill {skill_file}: {exc}"]

        name = metadata.get("name")
        description = metadata.get("description")
        if not isinstance(name, str) or not name.strip():
            return None, [f"Skipping skill {skill_file}: missing required frontmatter field 'name'"]
        if not isinstance(description, str) or not description.strip():
            return None, [f"Skipping skill {skill_file}: missing required frontmatter field 'description'"]

        metadata_block = metadata.get("metadata", {})
        short_description = description.strip()
        if isinstance(metadata_block, dict):
            short_candidate = metadata_block.get("short-description")
            if isinstance(short_candidate, str) and short_candidate.strip():
                short_description = short_candidate.strip()

        root_dir = skill_file.parent.resolve()
        skill = SkillSpec(
            name=name.strip(),
            description=description.strip(),
            short_description=short_description,
            body=body.strip(),
            root_dir=root_dir,
            skill_file=skill_file.resolve(),
            source=source,
            scripts=self._inventory_resources(root_dir / "scripts"),
            references=self._inventory_resources(root_dir / "references"),
            assets=self._inventory_resources(root_dir / "assets"),
        )

        if skill.is_oversized:
            warnings.append(
                f"Skill '{skill.name}' has {skill.body_line_count} body lines; consider moving detail into references/"
            )

        return skill, warnings

    def _parse_skill_markdown(self, text: str) -> tuple[dict, str]:
        lines = text.splitlines(keepends=True)
        if not lines or lines[0].strip() != "---":
            raise ValueError("missing YAML frontmatter")

        end_index = None
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                end_index = index
                break

        if end_index is None:
            raise ValueError("unterminated YAML frontmatter")

        frontmatter = "".join(lines[1:end_index])
        body = "".join(lines[end_index + 1 :])

        try:
            metadata = yaml.safe_load(frontmatter) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML frontmatter: {exc}") from exc

        if not isinstance(metadata, dict):
            raise ValueError("frontmatter must parse to a mapping")

        return metadata, body

    def _inventory_resources(self, root: Path) -> List[Path]:
        if not root.exists():
            return []
        return sorted(path.resolve() for path in root.rglob("*") if path.is_file())

    def _format_skill_payload(self, skill: SkillSpec) -> str:
        sections = [
            f"Skill: {skill.name}",
            f"Description: {skill.description}",
            f"Source: {skill.skill_file}",
            "",
            "Instructions:",
            skill.body,
            "",
            "Bundled resources:",
            *self._resource_group("Scripts", skill.scripts),
            *self._resource_group("References", skill.references),
            *self._resource_group("Assets", skill.assets),
            "",
            "Use read_file to inspect resource files as needed. Do not assume bundled scripts have already been executed.",
        ]
        return "\n".join(sections).strip()

    def _resource_lines(self, skill: SkillSpec) -> List[str]:
        lines: List[str] = []
        for label, paths in (
            ("Scripts", skill.scripts),
            ("References", skill.references),
            ("Assets", skill.assets),
        ):
            if not paths:
                continue
            lines.append(f"{label}:")
            lines.extend(f"- {path}" for path in paths)
        return lines

    def _resource_group(self, label: str, paths: Iterable[Path]) -> List[str]:
        lines = [f"{label}:"]
        entries = [f"- {path}" for path in paths]
        if entries:
            lines.extend(entries)
        else:
            lines.append("- none")
        return lines


class LoadSkillTool(Tool):
    """Tool for loading a skill's full instructions into the current turn."""

    name = "load_skill"
    description = (
        "Load a discovered skill's full instructions and bundled resource inventory "
        "for the current task."
    )
    parameters = {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "Name of the skill to load",
            }
        },
        "required": ["skill_name"],
        "additionalProperties": False,
    }

    def __init__(self, skill_manager: SkillManager) -> None:
        self.skill_manager = skill_manager

    def execute(self, context, **kwargs) -> ToolResult:
        """Return the formatted skill instructions for the requested skill."""
        skill_name = self._require_param(kwargs, "skill_name")

        try:
            payload = self.skill_manager.format_skill_for_tool(skill_name)
        except KeyError:
            return ToolResult(success=False, error=f"Unknown skill: {skill_name}")

        return ToolResult(success=True, data=payload)
