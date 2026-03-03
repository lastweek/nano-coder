# Nano-Coder

Nano-Coder is a terminal-based coding agent for working directly inside a repository. It combines an OpenAI-compatible LLM client with tool use for reading files, writing files, running shell commands, loading skills, and calling MCP servers from the same CLI session.

The project is built for practical repo work: you get a live activity feed while the agent is running, structured per-session logs after each turn, and slash commands for inspecting tools, skills, MCP servers, and estimated context usage.

## Why Nano-Coder

- Terminal-first workflow for staying in your shell and repository
- Tool-based code tasks with `read_file`, `write_file`, and `run_command`
- Streaming answers plus a live activity feed while the agent is thinking
- MCP support for external tool servers such as DeepWiki
- Local subagents for delegated repo tasks with a per-turn cap
- Local skill system with cataloging, pinning, and on-demand loading
- Per-session logging with `session.json`, `llm.log`, and `events.jsonl`
- `/context` command for estimating next-call baseline context usage

## Quickstart

### Install

```bash
pip install -r requirements.txt
git config core.hooksPath .githooks
```

### Configure

Start from the example config:

```bash
cp config.yaml.example config.yaml
cp .env.example .env
```

`config.yaml` is the main configuration file. `.env` is useful for secrets and local overrides. Environment variables override values from `config.yaml`.

Minimal OpenAI-compatible example:

```yaml
llm:
  provider: custom
  model: your-model-name
  base_url: https://api.example.com/v1
  api_key: your-api-key
```

### Run

```bash
python -m src.main
```

Alternative:

```bash
python src/main.py
```

## Example Session

```text
You > explain how the logging system works

Agent >
  • LLM call 1 requested 2 tools
  • Tool finished: run_command(cmd='rg "SessionLogger" -n src') (0.03s)
  • Tool finished: read_file(file_path='src/logger.py') (0.00s)
  • LLM call 2 produced final answer

Nano-Coder writes one session directory per CLI run, with `session.json` for metadata,
`llm.log` for the full execution timeline, and `events.jsonl` for structured events.

1234 prompt tokens • 87 completion tokens • 2.14s • TTFT 0.61s • glm-5 (custom)
```

## Core Capabilities

### Built-in tools

Nano-Coder ships with built-in tools for reading files, writing files, and running shell commands. The agent uses them through the same tool-calling loop as MCP tools and skills.

### MCP integration

MCP servers are configured in `config.yaml` and loaded at startup. Their tools appear alongside built-in tools and can be inspected with `/mcp`.

### Skills

Skills provide reusable local instructions through `SKILL.md` bundles. Nano-Coder keeps a compact skill catalog in context and loads full skill bodies only when they are pinned, explicitly requested with `$skill-name`, or loaded by the agent with `load_skill`.

### Subagents

Nano-Coder can delegate independent repo subtasks to fresh local subagents with `run_subagent`. Child agents run in the same repository with their own context and nested session logs, but they do not inherit the parent conversation history or spawn additional subagents.

### Logging

Each CLI run creates a per-session log directory. `llm.log` is the primary human-readable execution timeline, `events.jsonl` is the structured companion log, and `session.json` stores session metadata and aggregate counts.

### Context inspection

`/context` shows an estimated next-call baseline for the current session, broken down by system prompt, tool schemas, skill catalog, pinned skills, and persisted messages.

## How It Works

Nano-Coder uses a ReAct-style loop:

1. Build the system prompt, tool schemas, skill catalog, pinned skill preloads, and message history.
2. Send the request to the model.
3. If the model asks for tools, execute them and append the results.
4. Repeat until the model returns a final answer.
5. Stream or print the final answer in the CLI and log the turn.

Skills fit into the same flow: the catalog is available up front, and full `SKILL.md` bodies are loaded later only when needed. MCP tools participate through the same tool registry as built-in tools.

## Configuration

### Config sources and precedence

Nano-Coder uses:

- `config.yaml` for primary local configuration
- `.env` for secrets and values that are not committed
- environment variables for explicit overrides

In practice, treat `config.yaml` as the main config file. Use `.env` for credentials, and use environment variables when you need to override file-based settings for a specific run or machine.

### LLM providers

Supported provider modes:

- `openai`
- `azure`
- `ollama`
- `custom`

Core fields:

- `llm.provider`
- `llm.model`
- `llm.base_url`
- `llm.api_key`
- `llm.context_window` (optional)

Examples:

```yaml
llm:
  provider: openai
  model: gpt-4.1
  api_key: your-openai-key
```

```yaml
llm:
  provider: azure
  model: your-deployment-name
  base_url: https://your-resource.openai.azure.com
  api_key: your-azure-key
```

```yaml
llm:
  provider: ollama
  model: llama3
  base_url: http://localhost:11434/v1
```

```yaml
llm:
  provider: custom
  model: glm-5
  base_url: https://api.example.com/v1
  api_key: your-api-key
```

### UI and agent settings

Useful config knobs:

- `ui.enable_streaming`
- `agent.max_iterations`
- `logging.enabled`
- `logging.async_mode`
- `logging.log_dir`

### MCP server config

MCP servers live under `mcp.servers` in `config.yaml`:

```yaml
mcp:
  servers:
    - name: deepwiki
      url: https://mcp.deepwiki.com/mcp
      enabled: true
      timeout: 30
```

## Slash Commands

Nano-Coder supports built-in slash commands that bypass the agent:

- `/help` - list commands
- `/tool` - list available tools
- `/mcp` - inspect MCP servers and MCP-provided tools
- `/skill` - list, pin, unpin, inspect, and reload skills
- `/context` - estimate next-call baseline context usage
- `/subagent` - inspect or run local delegated child agents

Each slash command also supports built-in manual help:

- `/command help`
- `/command --help`
- `/command -h`

### `/skill`

Supported forms:

- `/skill`
- `/skill help`
- `/skill help <subcommand>`
- `/skill use <name>`
- `/skill clear <name>`
- `/skill clear all`
- `/skill show <name>`
- `/skill reload`

### `/compact`

Supported forms:

- `/compact`
- `/compact help`
- `/compact help <subcommand>`
- `/compact show`
- `/compact now`
- `/compact auto on`
- `/compact auto off`

For the full execution model, summary lifecycle, and logging flow, see [doc/context-compaction.md](doc/context-compaction.md).

### `/context`

`/context` estimates the baseline payload for the next LLM call before any new user message is added. The numbers are approximate and exclude the next user message and any explicit `$skill-name` preload for that future turn.

### `/subagent`

Supported forms:

- `/subagent`
- `/subagent help`
- `/subagent help <subcommand>`
- `/subagent run <task>`
- `/subagent show <id>`

## Skills

### What skills are

Skills are local instruction bundles stored in `SKILL.md`. They let the agent or user bring in domain-specific workflows without putting the full instructions into every turn up front.

### Discovery roots

Nano-Coder discovers skills from:

- `.nano-coder/skills`
- `~/.nano-coder/skills`

If the same skill name exists in both places, the repo-local copy wins.

### Skill layout

```text
skill-name/
├── SKILL.md
├── scripts/
├── references/
├── assets/
└── agents/
```

Only `SKILL.md` is required. `scripts/`, `references/`, and `assets/` are optional. `agents/` is ignored by the runtime.

### `SKILL.md` format

Each skill needs YAML frontmatter with `name` and `description`:

```md
---
name: pdf
description: Use for PDF tasks where layout and rendering matter
metadata:
  short-description: PDF workflows
---

Use `pdfplumber`, `pypdf`, and rendered page checks when layout matters.
```

### How skills enter context

Nano-Coder separates:

- **skill catalog**: skill name + short description
- **skill body**: full `SKILL.md`

The skill catalog is available up front. Full skill bodies are loaded on demand when:

- the agent calls `load_skill`
- you explicitly use `$skill-name`
- a skill is pinned for the session with `/skill use <name>`

Pinned skills remain available across turns, but their full bodies are injected later in the message list instead of being baked into the first system prompt.

### Slash command and `$skill-name` behavior

- Use `/skill` to inspect and manage discovered skills.
- Use `$skill-name` in a prompt to preload that skill for the current turn.
- Skill resources under `scripts/`, `references/`, and `assets/` are inventoried as paths only; their contents are not auto-loaded unless the agent reads them with tools.

## MCP

Nano-Coder can connect to MCP servers defined in `config.yaml`. Enabled servers are initialized at startup, and their tools are registered into the same tool system as built-in tools.

Current example configuration includes DeepWiki:

- `https://mcp.deepwiki.com/mcp`

You can inspect loaded MCP servers and tools with:

```text
/mcp
/mcp deepwiki
```

## Logging

Each CLI run writes a session directory under `logs/`:

- `session.json` - session metadata and aggregate counts
- `llm.log` - full human-readable execution timeline
- `events.jsonl` - structured event stream
- `artifacts/` - spilled large non-LLM payloads
- `subagents/` - nested child-agent session directories when delegation is used

Convenience symlinks:

- `logs/latest-session`
- `logs/latest.log`

### Reading logs

- Start with `session.json` for a quick summary.
- Open `llm.log` for the complete execution timeline.
- Use `events.jsonl` when you want structured event records.
- Check `artifacts/` only when an event or result references a spilled payload.

## Secret Guardrail

This repo includes git hooks that block commits and pushes when they contain values that look like real secrets.

Enable them locally:

```bash
git config core.hooksPath .githooks
```

Use `.env` for real credentials and keep tracked files on placeholders only.

## Development

### Running tests

```bash
pip install -r requirements-dev.txt
pytest
pytest --cov=src --cov-report=html
```

### Key directories

```text
src/                core CLI, agent, config, logging, commands, and built-in tools
tests/              pytest suite
.githooks/          local secret guard hooks
.nano-coder/skills/ optional repo-local skills
```

## License

MIT
