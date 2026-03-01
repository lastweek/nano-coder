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

Nano-Coder automatically logs all chat completions and tool executions for debugging.

### Log Format

Logs are saved in JSONL format (one JSON object per line) in the `logs/` directory.

### Log Files

- `logs/session-{id}-{timestamp}.jsonl` - Per-session log file
- `logs/latest.jsonl` - Symlink to the most recent session

### What Gets Logged

- User messages
- LLM requests (messages, tools, model, provider)
- LLM responses (content, tool_calls)
- Tool executions (name, arguments, results)
- Final agent responses

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
