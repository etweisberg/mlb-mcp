[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/etweisberg-mlb-mcp-badge.png)](https://mseep.ai/app/etweisberg-mlb-mcp)

# MLB Stats MCP Server

[![Tests](https://github.com/etweisberg/mcp-baseball-stats/actions/workflows/test.yml/badge.svg)](https://github.com/etweisberg/baseball/mcp-baseball-stats/workflows/test.yml)
[![Pre-commit](https://github.com/etweisberg/mcp-baseball-stats/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/etweisberg/mcp-baseball-stats/actions/workflows/pre-commit.yml)
[![smithery badge](https://smithery.ai/badge/@etweisberg/mlb-mcp)](https://smithery.ai/server/@etweisberg/mlb-mcp)

A Python project that creates a Model Context Protocol (MCP) server for accessing MLB statistics data through the MLB Stats API and `pybaseball` library for statcast, fangraphs, and baseball reference statistics. This server provides structured API access to baseball statistics that can be used with MCP-compatible clients.

## Project Structure

- `mlb_stats_mcp/` - Main package directory
  - `server.py` - Core MCP server implementation
  - `tools/` - MCP tool implementations
    - `mlb_statsapi_tools.py` - MLB StatsAPI tool definitions
    - `statcast_tools.py` - Statcast data tool definitions
    - `pybaseball_plotting_tools.py` - Additional `pybaseball` tools provided for generating matplotlib plots and returning base64 encoded images
    - `pybaseball_supp_tools.py` - Supplemental `pybaseball` functions for interfacing with fangraphs, baseball reference, and other data sources
  - `utils/` - Utility modules
    - `logging_config.py` - Logging configuration
    - `images.py` - functions related to handling plot images
  - `tests/` - Test suite for verifying server functionality
- `pyproject.toml` - Project configuration and dependencies
- `.pre-commit-config.yaml` - Pre-commit hooks configuration
- `.github/` - GitHub Actions workflows

## Tools

## Setup

1. Install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:

```bash
uv pip install -e .
```

### Installing via Smithery

To install MLB Stats Server for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@etweisberg/mlb-mcp):

```bash
npx -y @smithery/cli install @etweisberg/mlb-mcp --client claude
```

### Running Tests

The project includes comprehensive pytest tests for the MCP server functionality:

```bash
uv run pytest -v
```

Tests verify all MLB StatsAPI tools work correctly with the MCP protocol, establishing connections, making API calls, and processing responses.

## Environment Variables

The project uses environment variables stored in `.env` to configure settings.

Use `ANTHROPIC_API_KEY` to enable MCP Server.

### Logging Configuration

The MLB Stats MCP Server supports configurable logging via environment variables:

- `MLB_STATS_LOG_LEVEL` - Sets the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `MLB_STATS_LOG_FILE` - Path to log file (if not set, logs to stdout)

## Claude Desktop Integration

To connect this MCP server to Claude Desktop, add a configuration to your `claude_desktop_config.json` file. Here's a template configuration:

```json
"mcp-baseball-stats": {
  "command": "{PATH_TO_UV}",
  "args": [
    "--directory",
    "{PROJECT_DIRECTORY}",
    "run",
    "python",
    "-m",
    "mlb_stats_mcp.server"
  ],
  "env": {
    "MLB_STATS_LOG_FILE": "{LOG_FILE_PATH}",
    "MLB_STATS_LOG_LEVEL": "DEBUG"
  }
}
```

Replace the following placeholders:

- `{PATH_TO_UV}`: Path to your uv installation (e.g., `~/.local/bin/uv`)
- `{PROJECT_DIRECTORY}`: Path to your project directory
- `{LOG_FILE_PATH}`: Path where you want to store the log file

## Technologies Used

- `mcp[cli]` - Machine-Learning Chat Protocol for tool definition
- `mlb-statsapi` - Python wrapper for the MLB Stats API
- `httpx` - HTTP client for making API requests
- `pytest` and `pytest-asyncio` - Test frameworks
- `uv` - Fast Python package manager and installer

## Linting

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and code formatting, with pre-commit hooks to ensure code quality.

### Setup Pre-commit Hooks

1. Install pre-commit:

```bash
pip install pre-commit
```

2. Initialize pre-commit hooks:

```bash
pre-commit install
```

Now, the linting checks will run automatically whenever you commit code. You can also run them manually:

```bash
pre-commit run --all-files
```

### Linting Configuration

Linting rules are configured in the `pyproject.toml` file under the `[tool.ruff]` section. The project follows PEP 8 style guidelines with some customizations.

### CI Integration

GitHub Actions workflows automatically run tests, linting, and pre-commit checks on all pull requests and pushes to the main branch.
