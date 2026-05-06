# Contributing

## Project scope

This repository contains research code for PDE initial-temperature profile
reconstruction. Core algorithms live in `src/iu_pde_research/core/`, the CLI
entry point is `nhh_re.py`, and plotting/interactive views are in
`src/iu_pde_research/visualization/`.

For full setup and usage, see [README.md](../README.md).

## Documentation

All public modules, classes, and functions must have docstrings.

Format and conventions are defined in `docs/RULES.md` (Rule D.1.4):

- one-line summary,
- optional blank line,
- Google-style sections (`Args:`, `Returns:`, `Raises:`) where applicable.

For module-level technical notes, see `docs/core/`.

## Development setup

From the repository root:

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell:

  ```bash
  .\.venv\Scripts\Activate.ps1
  ```

- macOS/Linux:

  ```bash
  source .venv/bin/activate
  ```

Then install dependencies:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
pre-commit install
```

## Before you open a PR

1. Run the full local checks:

   ```bash
   pre-commit run --all-files
   pytest
   ```

2. If needed, run checks manually (same tools used by hooks):
   - `ruff check .`
   - `ruff format --check .`
   - `mypy src nhh_re.py`
   - `pytest`

3. Review the checklist in `docs/RULES.md` under **"Before You Open a PR"**
   and ensure each item is satisfied.

## Notes on optional backends

- CuPy-based tests are skipped automatically if `cupy` is not installed
  (`pytest.importorskip` in `tests/test_reconstruction_cupy.py`).
- JAX tests require JAX to import successfully in your environment.
