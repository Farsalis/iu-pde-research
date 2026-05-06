# iu-pde-research

PDE initial-temperature profile reconstruction for the 1D heat equation.

## Overview

This repository reconstructs an unknown initial condition \(f(x)\) from
temperature measurements at a single spatial point over time. The code provides
Fourier/sine-series based reconstruction methods for:

- modified \(t_j\) (homogeneous),
- modified \(t_j\) (non-homogeneous with forcing \(F(x,t)\)),
- modified \(t_j\) (complex-valued homogeneous case),
- linear time sequence (homogeneous).

The model assumes Dirichlet boundary conditions on \([0, L]\), with default
\(L = \pi\).

## Requirements

- Python 3.10+
- Runtime dependencies (from `pyproject.toml`): `numpy`, `scipy`,
  `matplotlib`, `jax`

## Setup (step-by-step)

1. Clone the repository and enter it.
2. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate it:
   - Windows PowerShell:

     ```bash
     .\.venv\Scripts\Activate.ps1
     ```

   - macOS/Linux:

     ```bash
     source .venv/bin/activate
     ```

4. Upgrade pip:

   ```bash
   python -m pip install --upgrade pip
   ```

5. Install the package:
   - Contributor setup (recommended):

     ```bash
     pip install -e ".[dev]"
     ```

   - Runtime-only setup:

     ```bash
     pip install -e .
     ```

6. Optional: install git hooks for local checks:

   ```bash
   pre-commit install
   ```

### Contributor setup (quick path)

If you plan to contribute code, use this exact sequence from the repo root:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
python -m pip install --upgrade pip
pip install -e ".[dev]"
pre-commit install
pytest
```

### Optional acceleration dependencies

- JAX is already a runtime dependency. If you need specific CPU/GPU builds or
  encounter install issues, follow the official JAX install guide:
  [https://jax.readthedocs.io/en/latest/installation.html](https://jax.readthedocs.io/en/latest/installation.html)
- CuPy is optional and only needed for `--backend gpu` (non-homogeneous path).
  Install a CUDA-matching wheel using the official docs:
  [https://docs.cupy.dev/en/stable/install.html](https://docs.cupy.dev/en/stable/install.html)

Note: `requirements.txt` is a pinned environment snapshot and not the canonical
project setup path. Use editable install from `pyproject.toml` as shown above.

## Verify your installation

From the repository root:

```bash
pytest
```

Optional full checks:

```bash
pre-commit run --all-files
```

## Usage

See all CLI options:

```bash
python nhh_re.py --help
```

### Default run (no flags)

```bash
python nhh_re.py
```

This runs:

- modified \(t_j\) homogeneous reconstruction (CPU path by default),
- linear-sequence homogeneous reconstruction,
- then prints command hints for explicit modes.

### Explicit demo modes

```bash
python nhh_re.py --interactive
python nhh_re.py --modified_homogeneous
python nhh_re.py --modified_nonhomogeneous --backend cpu
python nhh_re.py --modified_complex_homogeneous
python nhh_re.py --linear_homogeneous
```

Backend choices:

- `--backend cpu`
- `--backend jax`
- `--backend gpu` (CuPy installed; non-homogeneous GPU path)

Initial condition options:

- `--initial-condition default`
- `--initial-condition sine_series_5`
- `--initial-condition sine_series_10`
- `--initial-condition sine_series_15`

### Batch mode

Batch mode requires `--algorithm`, `--T`, `--x0`, and `--n`.

Valid `--algorithm` values:

- `modified_homogeneous`
- `modified_nonhomogeneous`
- `modified_complex_homogeneous`
- `linear_homogeneous`

Examples:

```bash
python nhh_re.py --algorithm modified_homogeneous --T 5 --x0 1.41 --n 20
python nhh_re.py --algorithm modified_nonhomogeneous --backend gpu --T 5 --x0 1.41 --n 20 --output-dir out
```

For `--T`, `--x0`, and `--n`, expressions are supported (for example
`np.pi/golden`, `sqrt(2)`, `2*10`).

## Project structure

- `nhh_re.py` - CLI entry point (default run, demos, interactive mode, batch)
- `src/iu_pde_research/core/`
  - `reconstruction.py`
  - `reconstruction_jax.py`
  - `reconstruction_cupy.py`
  - `forcing.py`
  - `forcing_autodiff.py`
  - `forcing_cupy.py`
  - `utils.py`
- `src/iu_pde_research/visualization/` - plotting and interactive visualization
- `tests/` - pytest suite for core, visualization, and CLI behavior
- `benchmarks/` - benchmark scripts
- `docs/core/` - module-level technical notes

## Contributing and standards

- Contributor workflow: [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)
- Project rules/style guide: [docs/RULES.md](docs/RULES.md)
