# Agent Instructions for OpenPTV

## Python Environment

Always use `uv` for Python package management. Do NOT use system python or pip directly.

### Commands

Instead of:
- `python` → use `uv run python` or activate the venv first
- `pip install` → use `uv pip install`
- `python -m pytest` → use `uv run pytest`

### Running Tests

```bash
uv run pytest algorithms/tests/test_parameter_converters.py -v
```

### Virtual Environment

If the project has a `.venv` directory, activate it before running Python commands:
```bash
source .venv/bin/activate
```

Or use `uv run` to execute in the project context.