# Nano-Coder

A minimalism terminal-based AI code agent with support for multiple LLM providers.

## Features

- 🤖 Conversational coding assistant
- 📁 Read and write files
- 💻 Execute shell commands
- 🎯 Clean terminal interface
- 🔄 ReAct loop for tool orchestration

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up your environment
cp .env.example .env
# Edit .env with your configuration

# Install repo-local git hooks
git config core.hooksPath .githooks
```

## Configuration

Create a `.env` file from the example and configure your LLM provider:

### Using OpenAI

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
MODEL=gpt-4
```

### Using Azure OpenAI

```bash
LLM_PROVIDER=azure
AZURE_API_KEY=your-azure-key
BASE_URL=https://your-resource.openai.azure.com
MODEL=your-deployment-name
```

### Using Ollama (Local)

```bash
LLM_PROVIDER=ollama
BASE_URL=http://localhost:11434/v1
MODEL=llama2
# No API key needed for Ollama
```

### Using Custom OpenAI-Compatible Endpoint

```bash
LLM_PROVIDER=custom
CUSTOM_API_KEY=your-key-if-needed
BASE_URL=https://your-api-endpoint.com
MODEL=your-model-name
```

## Secret Guardrail

This repo includes git hooks that block commits and pushes when they contain values that look like real API keys or tokens.

Enable them in your local clone:

```bash
git config core.hooksPath .githooks
```

What the guard checks:

- Added lines in staged changes before `git commit`
- Added lines in commits that are about to be sent during `git push`

What to do instead of committing a secret:

- Put real credentials in `.env`
- Keep `.env.example` on placeholders only
- Reference secrets from environment variables in example scripts

## Logging

Nano-Coder automatically logs each CLI run into a session directory under `logs/`.

### Session Layout

Each new session creates:

- `logs/session-{timestamp}-{session_id}/session.json` - session metadata and aggregate counts
- `logs/session-{timestamp}-{session_id}/llm.log` - complete readable execution timeline with full LLM request/response JSON plus inline tool and skill activity
- `logs/session-{timestamp}-{session_id}/events.jsonl` - structured turn, tool, skill, and error events
- `logs/session-{timestamp}-{session_id}/artifacts/` - oversized non-LLM payloads that were spilled out of `events.jsonl`

Convenience symlinks are updated on each run:

- `logs/latest-session` - symlink to the newest session directory
- `logs/latest.log` - symlink to the newest session's `llm.log`

### What Gets Logged

- Session metadata and counters in `session.json`
- Full outgoing LLM request payload JSON in `llm.log`
- Full parsed or reconstructed LLM response JSON in `llm.log`
- Inline turn boundaries, tool calls/results, skill events, and errors in `llm.log`
- Turn lifecycle, tool calls/results, skill events, and structured errors in `events.jsonl`
- Large non-LLM payloads in `artifacts/` when they would otherwise bloat `events.jsonl`

### Reading Logs Quickly

- Start with `session.json` for the high-level session overview
- Open `llm.log` when you want the full chronological execution timeline for a session
- Use `events.jsonl` for structured non-LLM chronology
- Check `artifacts/` only when an event references a spilled payload

### Disabling Logging

Set `ENABLE_LOGGING=false` in your `.env` file to disable logging.

## Streaming Output

Nano-Coder supports streaming output for real-time token-by-token response display, similar to ChatGPT.

### Enabling Streaming

Streaming is enabled by default. To disable it:

```bash
# In your .env file
ENABLE_STREAMING=false
```

### How Streaming Works

When `ENABLE_STREAMING=true`:
- LLM responses appear token-by-token as they're generated
- You see the response in real-time instead of waiting for the complete message
- Tool executions still display normally during streaming

When `ENABLE_STREAMING=false`:
- A "Thinking..." indicator shows while waiting for the LLM
- The complete response appears at once when finished

### Performance

Streaming provides better perceived performance (instant feedback) but requires:
- OpenAI-compatible API that supports streaming (most do)
- Slightly more CPU for real-time rendering

## Skills

Nano-Coder supports local Codex-style skill bundles for domain-specific workflows and reusable instructions.

### Discovery Roots

- Repo-local: `.nano-coder/skills`
- User-global: `~/.nano-coder/skills`

If a repo-local skill and a user-global skill share the same `name`, the repo-local skill wins.

### Skill Layout

```text
skill-name/
├── SKILL.md
├── scripts/
├── references/
├── assets/
└── agents/
```

Only `SKILL.md` is required. `scripts/`, `references/`, and `assets/` are optional and are inventoried automatically. `agents/` is ignored by the runtime.

### `SKILL.md` Format

Each skill must use YAML frontmatter with `name` and `description`:

```md
---
name: pdf
description: Use for PDF tasks where layout and rendering matter
metadata:
  short-description: PDF workflows
---

Use `pdfplumber`, `pypdf`, and rendered page checks when layout matters.
```

### Slash Commands

- `/skill` lists discovered skills, whether they are part of the current session catalog, and whether they are pinned for the session
- `/skill use <name>` pins a skill for the session
- `/skill clear <name>` unpins one skill
- `/skill clear all` unpins all session skills
- `/skill show <name>` shows metadata, path, body size, and bundled resources
- `/skill reload` rescans the skill directories from disk

### Agent Behavior

- The system prompt contains a stable **skill catalog** for discovered skills in the current session:
  - skill name
  - short description
- This includes repo-local and user-global skills after duplicate resolution
- Full `SKILL.md` **skill bodies** are only loaded when:
  - the agent calls the built-in `load_skill` tool during the current turn
  - you explicitly preload a skill with `$skill-name`
- Pinned skills from `/skill use <name>` stay available across turns, but their full bodies are injected later in the message list instead of being baked into the first system prompt
- Loading a skill does not execute bundled scripts automatically
- `scripts/`, `references/`, and `assets/` are inventoried as paths only; their contents are loaded only if the agent later reads them with tools
- Set `SKILL_DEBUG=1` to print skill preload/load debug lines in the CLI while also recording structured `skill_event` entries in the session log

### Example Skill Session

```text
You > $pdf explain this PDF form workflow

Agent > Here is the PDF-specific workflow to follow...
```

## Usage

```bash
# Run from the project root directory (either method works)
python -m src.main
# OR
python src/main.py
```

## Testing

Run the test suite:

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov=tools --cov-report=html

# Run specific test file
pytest tests/test_tools.py -v
```

Test files are located in the `tests/` directory following pytest conventions.

## Project Structure

```
nano-coder/
├── src/
│   ├── main.py       # CLI entry point
│   ├── agent.py      # Core agent orchestration
│   ├── llm.py        # LLM API integration
│   ├── tools.py      # Tool registry and base classes
│   └── context.py    # Context management
├── tools/
│   ├── read.py       # File reading tool
│   ├── write.py      # File writing tool
│   └── bash.py       # Command execution tool
└── tests/
    └── test_tools.py
```

## Example Session

```text
You > list all python files

Agent > I'll search for Python files in the current directory.
  → run_command(command='find . -name "*.py" -type f')

Found 3 Python files:
- src/main.py
- src/agent.py
- src/tools.py
```

## Architecture

Nano-Coder uses the **ReAct pattern** (Reasoning + Acting):

1. User sends a message
2. Agent sends message + tools to LLM
3. If LLM requests tool calls, execute them
4. Feed tool results back to LLM
5. Repeat until no more tool calls
6. Return final response to user

## Available Tools

| Tool | Description |
| ---- | ----------- |
| `read_file` | Read file contents with line numbers |
| `write_file` | Create or overwrite files |
| `run_command` | Execute shell commands |
| `load_skill` | Load a skill's instructions and bundled resource inventory |

## Next Steps

Planned features for future releases:

- [ ] Grep/Glob tools for code search
- [ ] Edit tool for string replacement editing
- [ ] Session persistence
- [ ] Git integration
- [ ] Permission system for safe operations
- [ ] Todo tracking for multi-step tasks

## License

MIT
