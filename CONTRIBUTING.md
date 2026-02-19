# Contributing

Code style and conventions for this project are defined in the **Python styling guide**:

- **Location:** `.cursor/rules/python-lang-styling.mdc`

Please read the relevant sections (formatting, naming, imports, type hints, docstrings, testing) before opening a pull request.

## Before you open a PR

1. **Install pre-commit** (one-time):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

2. **Run quality checks locally:**
   - `pre-commit run --all-files` — runs formatter, linter, type checker, and security scanner
   - `ruff check .` and `ruff format --check .` — lint and format
   - `mypy .` — type checking
   - `pytest` — tests (run from repo root after `pip install -e .`)

3. Ensure all of the above pass. The guide’s **“Before You Open a PR”** checklist (in the styling guide) applies.

## Running tests

From the repository root:

```bash
pip install -e .
pytest
```

Optional: install dev dependencies and run the full toolchain:

```bash
pip install -e ".[dev]"
pre-commit run --all-files
pytest
```
