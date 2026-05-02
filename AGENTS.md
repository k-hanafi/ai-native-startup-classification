# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Pure-Python CLI pipeline (`classify.py`) that classifies ~267K Crunchbase startups as AI-native or not using the OpenAI Batch API. No web server, no database, no Docker. See `coding_instructions.md` for code style, commit conventions, and testing standards.

### Running tests

```bash
OPENAI_API_KEY=sk-test-dummy pytest tests/ -v
```

Tests are offline (no API calls). The `OPENAI_API_KEY` env var must be set because `src/config.py` reads it at import time. Any dummy value works for tests.

### Linting

`ruff` is not declared in `pyproject.toml` but is referenced in `coding_instructions.md`. Install it with `pip install ruff` and run:

```bash
ruff check src/ tests/ classify.py
```

There are pre-existing unused-import warnings (F401) in the codebase.

### Running the CLI (dry-run mode)

The `prepare --dry-run` subcommand exercises the full pipeline logic (CSV parsing, message formatting, token counting, cost estimation) without calling the OpenAI API:

```bash
OPENAI_API_KEY=sk-test-dummy python3 classify.py prepare --dry-run --data <path-to-csv>
```

The input CSV (`data/company_us_short_long_desc_.csv`) is gitignored and not included in the repo. For testing, create a sample CSV with columns: `org_uuid`, `name`, `short_description`, `Long description`, `category_list`, `category_groups_list`, `founded_date`.

### API-dependent operations

All subcommands other than `prepare --dry-run` and `run --dry-run` require a valid `OPENAI_API_KEY` with sufficient billing tier. The key is loaded from `keys/openai.env` via `python-dotenv` at startup. Alternatively, export `OPENAI_API_KEY` directly in the environment.
