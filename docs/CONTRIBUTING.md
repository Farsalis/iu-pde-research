# Contributing

## Project scope

This repository contains research code for **PDE initial temperature profile reconstruction**: algorithms live in `src/iu_pde_research/core/`, the CLI and demo entry point in `nhh_re.py`, and plotting in `src/iu_pde_research/visualization/`. For installation and usage, see [README.md](../README.md).

## Documentation

All public modules, classes, and functions must have docstrings. Format and conventions are defined in the **Python styling guide** (Rule D.1.4): one-line summary, blank line, then body with `:args:`, `:returns:`, `:raises:` as applicable. Omit return description when it is obvious from the type hint.

- **Location:** `./docs/RULES.md`

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
