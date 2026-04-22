"""Main entry point for PDE initial temperature profile reconstruction.

CLI for running reconstruction methods: by default runs modified t_j (homogeneous)
and linear sequence demos; --interactive opens a plot with n, T, x0 controls;
--nonhomogeneous runs one non-homogeneous demo; batch mode uses --algorithm with
--T, --x0, --n (and optional --output-dir). Run with --help for full options.

NOTE: DIRICHLET BOUNDARY CONDITIONS ARE ASSUMED FOR EACH FUNCTIONS. DOES NOT SUPPORT PERIODIC OR NEUMANN.
"""

import argparse
import sys
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import golden

from iu_pde_research.core import (
    INITIAL_CONDITION_OPTIONS,
    compute_linear_t_j_homogeneous,
    compute_modified_t_j,
    compute_modified_t_j_homogeneous_complex,
    compute_modified_t_j_jax,
    compute_modified_t_j_nonhomogeneous,
)
from iu_pde_research.visualization import (
    create_interactive_plot,
    plot_reconstruction,
    plot_reconstruction_with_errors,
)

# from juliacall import Main as jl

# jl.include("./compute_constants.jl")

DEFAULT_OUTPUT_DIR = "reconstruction_output"

# Namespace for safe evaluation of CLI expressions (e.g. np.pi / golden).
_EXPR_NAMESPACE = {
    "np": np,
    "golden": golden,
    "pi": np.pi,
    "e": np.e,
    "sin": np.sin,
    "cos": np.cos,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "tan": np.tan,
}


def _parse_float_expr(s: str) -> float:
    """
    Parse a string as a float or as a safe expression (e.g. np.pi / golden).

    Args:
        s: A numeric string or expression using names from `_EXPR_NAMESPACE`.

    Returns:
        The evaluated float value.

    Raises:
        ValueError: If the string is not a valid number or expression.
    """
    s_stripped = s.strip()
    try:
        return float(s_stripped)
    except ValueError:
        pass
    try:
        # Restricted evaluation namespace to satisfy bandit (no arbitrary builtins).
        result = eval(s_stripped, {"__builtins__": {}}, _EXPR_NAMESPACE)  # nosec B307
        return float(result)
    except Exception as e:
        msg = f"Invalid expression for float: {s_stripped!r}: {e}"
        raise ValueError(msg) from e


def _parse_int_expr(s: str) -> int:
    """
    Parse a string as an int or as a safe expression (e.g. 2 * 10).

    Args:
        s: A numeric string or expression using names from `_EXPR_NAMESPACE`.

    Returns:
        The evaluated integer value.

    Raises:
        ValueError: If the string is not a valid integer or expression.
    """
    s_stripped = s.strip()
    try:
        return int(s_stripped)
    except ValueError:
        pass
    try:
        # Restricted evaluation namespace to satisfy bandit (no arbitrary builtins).
        result = eval(s_stripped, {"__builtins__": {}}, _EXPR_NAMESPACE)  # nosec B307
        result_float = float(result)
        if not result_float.is_integer():
            msg = f"Invalid expression for int: {s_stripped!r}: not an integer value"
            raise ValueError(msg)
        return int(result_float)
    except Exception as e:
        msg = f"Invalid expression for int: {s_stripped!r}: {e}"
        raise ValueError(msg) from e


def _default_f_true_nonhomogeneous(x: float | np.ndarray) -> float | np.ndarray:
    """Initial condition used for non-homogeneous demo and batch."""
    values = (1 / 8) * np.sin(2 * x) + (1 / 18) * np.sin(3 * x)
    if np.ndim(values) == 0:
        return float(values)
    return values


#############################################################################


def _default_F(x: float, t: float) -> float:
    """Forcing term used for non-homogeneous demo and batch.

    Uses JAX (jax.numpy) so that F_x = ∂F/∂x can be obtained via autodiff when
    F_x is not supplied. Returns a float-convertible value (Python float
    or 0-dim JAX array); call sites that need float use float().
    """
    # Delayed import so the CLI works even when JAX is not installed,
    # unless the user explicitly requests a JAX-based computation.
    import jax.numpy as jnp

    return cast(float, jnp.sin(x) * jnp.exp(-t))


############################################################################


def _default_f_true_linear(x: float | np.ndarray) -> float | np.ndarray:
    """Initial condition used for linear sequence demo and batch."""
    values = np.sin(2 * x) + 0.5 * np.sin(5 * x)
    if np.ndim(values) == 0:
        return float(values)
    return values


def _parse_args() -> argparse.Namespace:
    """
    Build and parse command-line arguments for demo and batch modes.

    Returns:
        Parsed namespace with optional interactive, nonhomogeneous, and
        batch fields (algorithm, T, x0, n, output_dir).
    """
    parser = argparse.ArgumentParser(
        description="PDE initial temperature profile reconstruction."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive plot with parameter controls for non-homogeneous modified t_j case.",
    )
    parser.add_argument(
        "--modified_homogeneous",
        action="store_true",
        help="Run homogeneous reconstruction once (demo).",
    )
    parser.add_argument(
        "--modified_nonhomogeneous",
        action="store_true",
        help="Run non-homogeneous reconstruction once (demo).",
    )
    parser.add_argument(
        "--modified_complex_homogeneous",
        action="store_true",
        help="Run homogeneous reconstruction for complex case once (demo).",
    )
    parser.add_argument(
        "--linear_homogeneous",
        action="store_true",
        help="Run homogeneous reconstruction for linear t_j (demo).",
    )
    parser.add_argument(
        "--algorithm",
        choices=[
            "modified_homogeneous",
            "modified_nonhomogeneous",
            "modified_complex_homogeneous",
            "linear_homogeneous",
        ],
        default=None,
        help="Reconstruction algorithm for batch mode.",
    )
    parser.add_argument(
        "--T",
        type=_parse_float_expr,
        nargs="*",
        default=None,
        metavar="T",
        help=(
            "Time parameter(s) for batch (T or t0 for linear). "
            "May be numbers or expressions (e.g. np.pi/golden)."
        ),
    )
    parser.add_argument(
        "--x0",
        type=_parse_float_expr,
        nargs="*",
        default=None,
        metavar="X0",
        help=(
            "Spatial measurement point(s) for batch. "
            "May be numbers or expressions (e.g. np.pi/golden, sqrt(2))."
        ),
    )
    parser.add_argument(
        "--n",
        type=_parse_int_expr,
        nargs="*",
        default=None,
        metavar="N",
        help=(
            "Number of time measurements for batch. "
            "May be integers or expressions (e.g. 2*10)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for batch plots (created if missing).",
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "jax", "gpu"],
        default="cpu",
        help=(
            "Backend for reconstruction: 'cpu' (NumPy/SciPy), 'jax' (JAX), "
            "or 'gpu' (CuPy for nonhomogeneous default forcing)."
        ),
    )
    parser.add_argument(
        "--initial-condition",
        choices=list(INITIAL_CONDITION_OPTIONS.keys()),
        default="default",
        help=(
            "True initial condition: 'default' (algorithm-specific), or "
            "'sine_series_5' / 'sine_series_10' / 'sine_series_15' for "
            "f(x)=sum_{j=1}^{n} (1/(2j^2))*sin(j*x)."
        ),
    )
    return parser.parse_args()


def _batch_filename_stem(algorithm: str, T: float, x0: float, n: int) -> str:
    """
    Build a filesystem-safe filename stem for one batch run.

    Args:
        algorithm: One of ``\"modified\"``, ``\"nonhomogeneous\"``, or ``\"linear\"``.
        T: Time parameter (or t0 for linear).
        x0: Spatial measurement point.
        n: Number of time measurements.

    Returns:
        Stem string suitable for a PNG filename
        (e.g. ``modified_T5_0_x0_1_41_n20``).
    """
    t_str = str(T).replace(".", "_").replace("-", "m")
    x0_str = str(round(x0, 4)).replace(".", "_")
    return f"{algorithm}_T{t_str}_x0{x0_str}_n{n}"


def _get_f_true_for_algorithm(
    algorithm: str,
    initial_condition: str,
) -> Callable[[float | np.ndarray], float | np.ndarray] | None:
    """Resolve f_true callable: use algorithm default if initial_condition is 'default', else the named option."""
    if initial_condition == "default":
        return None
    return INITIAL_CONDITION_OPTIONS[initial_condition]


def _run_single_reconstruction(
    algorithm: str,
    T: float,
    x0: float,
    n: int,
    output_dir: Path,
    x_grid: np.ndarray,
    backend: str = "cpu",
    initial_condition: str = "default",
) -> None:
    """
    Run one reconstruction for the given (algorithm, T, x0, n) and save a plot.

    Args:
        algorithm: One of ``\"modified\"``, ``\"nonhomogeneous\"``, or ``\"linear\"``.
        T: Time parameter (or t0 for linear).
        x0: Spatial measurement point.
        n: Number of time measurements.
        output_dir: Directory to write the PNG into.
        x_grid: Grid of x values for reconstruction.
    """
    stem = _batch_filename_stem(algorithm, T, x0, n)
    save_path = output_dir / f"{stem}.png"

    f_true_callable = _get_f_true_for_algorithm(algorithm, initial_condition)

    if algorithm in {"modified_homogeneous", "modified"}:
        f_true = f_true_callable if f_true_callable is not None else None
        if backend == "jax":
            f_true_vals, bar_f_n_vals = compute_modified_t_j_jax(n, T, x0, x_grid)
        else:
            f_true_vals, bar_f_n_vals = compute_modified_t_j(
                n, T, x0, x_grid, f_true=f_true
            )
        plot_reconstruction(
            x_grid,
            f_true_vals,
            bar_f_n_vals,
            title="Initial Temperature Profile Reconstruction (Modified t_j - Homogeneous)\n",
            save_path=str(save_path),
            show=False,
        )
    elif algorithm in {"modified_nonhomogeneous", "nonhomogeneous"}:
        if backend == "gpu":
            try:
                from iu_pde_research.core.forcing_cupy import (
                    default_F_cupy,
                    default_F_x_cupy,
                )
            except ImportError as exc:  # pragma: no cover - env dependent
                msg = (
                    "backend='gpu' requested for nonhomogeneous batch run, "
                    "but CuPy-based forcing kernels are not available. "
                    "Install CuPy or use --backend cpu."
                )
                raise RuntimeError(msg) from exc

            f_true_vals, bar_f_n_vals = compute_modified_t_j_nonhomogeneous(
                n,
                T,
                x0,
                x_grid,
                F=_default_F,
                f_true=f_true_callable or _default_f_true_nonhomogeneous,
                backend="gpu",
                F_gpu=default_F_cupy,
                F_x_gpu=default_F_x_cupy,
            )
        else:
            f_true_vals, bar_f_n_vals = compute_modified_t_j_nonhomogeneous(
                n,
                T,
                x0,
                x_grid,
                F=_default_F,
                f_true=f_true_callable or _default_f_true_nonhomogeneous,
            )
        plot_reconstruction_with_errors(
            x_grid,
            f_true_vals,
            bar_f_n_vals,
            title=(
                "Initial Temperature Profile Reconstruction (Modified t_j - Non-Homogeneous)\n"
                f"n={n}, T={T}, x0={x0:.4f}"
            ),
            save_path=str(save_path),
            show=False,
        )
    elif algorithm == "modified_complex_homogeneous":
        f_true_vals, bar_f_n_vals = compute_modified_t_j_homogeneous_complex(
            n, T, x0, x_grid
        )
        plot_reconstruction_with_errors(
            x_grid,
            f_true_vals,
            bar_f_n_vals,
            title=(
                "Initial Temperature Profile Complex Reconstruction (Modified t_j - Homogeneous)\n"
                f"n={n}, T={T}, x0={x0:.4f}"
            ),
            save_path=str(save_path),
            show=False,
        )
    else:
        if algorithm not in {"linear_homogeneous", "linear"}:
            raise ValueError(f"Unknown algorithm: {algorithm!r}")
        t0 = T
        f_true_vals, bar_f_n_vals = compute_linear_t_j_homogeneous(
            n,
            t0,
            x0,
            x_grid,
            f_true=f_true_callable or _default_f_true_linear,
        )
        plot_reconstruction(
            x_grid,
            f_true_vals,
            bar_f_n_vals,
            title="Initial Condition Reconstruction (Linear t_j - Homogeneous)",
            save_path=str(save_path),
            show=False,
        )
    plt.close()


############################################################################
# FUNCTIONS THAT RUN CONSTRUCTIONS WITH CUSTOM VARIABLES
# CALL SAME FUNCTIONS IN reconstruction.py
############################################################################


def modified_t_j() -> None:
    """
    Run factorial t_j method for homogeneous case.

    Demonstrates the reconstruction of initial condition using the modified
    factorial t_j for the homogeneous heat equation (CPU backend).

    Current implementation uses default f_true in the reconstruction function.
    """
    n = 20
    T = 5
    x0 = np.sqrt(2)
    x_grid = np.linspace(0, np.pi, 200)
    f_true_vals, bar_f_n_vals = compute_modified_t_j(n, T, x0, x_grid)
    plot_reconstruction(
        x_grid,
        f_true_vals,
        bar_f_n_vals,
        title="Initial Condition Reconstruction (Modified t_j)",
        save_path="modified_t_j.png",
    )


def modified_t_j_nonhomogeneous(
    backend: str = "cpu",
    f_true: (Callable[[float | np.ndarray], float | np.ndarray] | None) = None,
) -> None:
    """
    Run factorial t_j method for non-homogeneous case.

    Demonstrates the reconstruction of initial condition using the modified
    factorial t_j method for the non-homogeneous heat equation with a forcing term.
    """
    n = 20
    T = 5.0
    x0 = np.pi / golden
    x_grid = np.linspace(0, np.pi, 200)
    if f_true is None:
        f_true = _default_f_true_nonhomogeneous

    print("Computing non-homogeneous reconstruction...")
    print(f"Parameters: n={n}, T={T}, x0={x0:.4f}")
    print("Forcing term: F(x,t) = sin(x) * exp(-t)")

    if backend == "gpu":
        try:
            from iu_pde_research.core.forcing_cupy import (
                default_F_cupy,
                default_F_x_cupy,
            )
        except ImportError as exc:  # pragma: no cover - env dependent
            msg = (
                "backend='gpu' requested for nonhomogeneous demo, but CuPy-based "
                "forcing kernels are not available. Install CuPy or use "
                "--backend cpu."
            )
            raise RuntimeError(msg) from exc

        f_true_vals, bar_f_n_vals = compute_modified_t_j_nonhomogeneous(
            n,
            T,
            x0,
            x_grid,
            F=_default_F,
            f_true=f_true,
            backend="gpu",
            F_gpu=default_F_cupy,
            F_x_gpu=default_F_x_cupy,
        )
    else:
        f_true_vals, bar_f_n_vals = compute_modified_t_j_nonhomogeneous(
            n, T, x0, x_grid, F=_default_F, f_true=f_true
        )

    plot_reconstruction_with_errors(
        x_grid,
        f_true_vals,
        bar_f_n_vals,
        title=(
            "Initial Temperature Profile Reconstruction for Non-Homogeneous Case \n"
            f"n = {n}, T={T}\n"
            r"$f(x)=\frac{1}{8}\cdot \sin(2x)+\frac{1}{18}\cdot\sin(3x)$"
            "\n"
            r"$F(x,t)=\sin(x)\exp(-t)$"
        ),
        save_path="modified_t_j_nonhomogeneous.png",
    )


def modified_t_j_homogeneous_complex() -> None:
    """
    Run factorial t_j method for homogeneous complex case.

    Demonstrates the reconstruction of initial condition using the modified
    factorial t_j method for the homogeneous heat equation using complex notation.

    Current implementation uses default f_true in the reconstruction function.
    """
    n = 20
    T = 5.0
    x0 = 2 * np.sqrt(2)
    x_grid = np.linspace(0, np.pi, 200)

    print("Computing complex homogeneous reconstruction...")
    print(f"Parameters: n={n}, T={T}, x0={x0:.4f}")

    f_true_vals, bar_f_n_vals = compute_modified_t_j_homogeneous_complex(
        n, T, x0, x_grid
    )

    plot_reconstruction(
        x_grid,
        f_true_vals,
        bar_f_n_vals,
        title=(
            "Initial Temperature Profile Reconstruction for Homogeneous Complex Case \n"
            f"n = {n}, T={T}"
        ),
        save_path="modified_t_j_homogeneous_complex.png",
    )


def linear_sequence() -> None:
    """
    Run linear sequence method for reconstruction.

    Demonstrates the reconstruction of initial condition using a linear
    time sequence method.
    """
    n = 20
    t0 = 1e-3  # Tolerance
    x0 = np.sqrt(2)
    x_grid = np.linspace(0, np.pi, 200)

    f_true_vals, bar_f_n_vals = compute_linear_t_j_homogeneous(
        n, t0, x0, x_grid, f_true=_default_f_true_linear
    )

    plot_reconstruction(
        x_grid,
        f_true_vals,
        bar_f_n_vals,
        title="Initial Condition Reconstruction (Linear t_j)",
        save_path="linear_sequence.png",
    )


############################################################################


def main() -> None:
    """Main entry point for command-line interface."""
    args = _parse_args()

    if args.interactive:
        print("Starting interactive mode for non-homogeneous case...")
        print("Use the text boxes to adjust n, T, and x0 parameters.")
        create_interactive_plot()
        return
    elif args.modified_nonhomogeneous:
        print(f"Running non-homogeneous case (backend={args.backend})...")
        f_true = (
            None
            if args.initial_condition == "default"
            else INITIAL_CONDITION_OPTIONS[args.initial_condition]
        )
        modified_t_j_nonhomogeneous(backend=args.backend, f_true=f_true)
        return
    elif args.modified_homogeneous:
        print("Running homogeneous case...")
        modified_t_j()
    elif args.modified_complex_homogeneous:
        print("Running homogeneous complex case...")
        modified_t_j_homogeneous_complex()
        return
    elif args.linear_homogeneous:
        print("Running linear t_j homogeneous case...")
        linear_sequence()
        return

    batch = (
        args.algorithm is not None
        and args.T is not None
        and len(args.T) > 0
        and args.x0 is not None
        and len(args.x0) > 0
        and args.n is not None
        and len(args.n) > 0
    )
    if batch:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        x_grid = np.linspace(0, np.pi, 200)
        total = 0
        for T, x0, n in product(args.T, args.x0, args.n):
            _run_single_reconstruction(
                args.algorithm,
                T,
                x0,
                n,
                output_dir,
                x_grid,
                backend=args.backend,
                initial_condition=args.initial_condition,
            )
            total += 1
        print(f"Saved {total} plot(s) to {output_dir.resolve()}")
        return

    if (
        args.algorithm is not None
        or (args.T is not None and len(args.T) > 0)
        or (args.x0 is not None and len(args.x0) > 0)
        or (args.n is not None and len(args.n) > 0)
    ):
        print(
            "Batch mode requires --algorithm and at least one value each "
            "for --T, --x0, and --n.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Running Modified t_j method (homogeneous, backend={args.backend})...")
    if args.backend == "jax":
        n = 20
        T = 5
        x0 = np.sqrt(2)
        x_grid = np.linspace(0, np.pi, 200)
        f_true_vals, bar_f_n_vals = compute_modified_t_j_jax(n, T, x0, x_grid)
        plot_reconstruction(
            x_grid,
            f_true_vals,
            bar_f_n_vals,
            title="Initial Condition Reconstruction (Modified t_j, JAX backend)",
            save_path="modified_t_j_jax.png",
        )
    else:
        modified_t_j()
    print("\nRunning Linear sequence method...")
    linear_sequence()
    print("\nDone!")
    print("\nTo run interactive mode, use: python nhh_re.py --interactive")
    print(
        "\nTo run homogeneous case w/ modified t_j, use: python nhh_re.py --modified_homogeneous"
    )
    print(
        "\nTo run homogeneous case w/ linear t_j, use: python nhh_re.py --linear_homogeneous"
    )
    print(
        "\nTo run non-homogeneous case w/ modified t_j, use: python nhh_re.py --modified_nonhomogeneous"
    )
    print(
        "\nTo run complex homogeneous case w/ modified t_j, use: python nhh_re.py --modified_complex_homogeneous"
    )
    print(
        "For batch runs: python nhh_re.py --algorithm modified_homogeneous --T 5 --x0 1.41 "
        "--n 20 [--output-dir DIR]"
    )


if __name__ == "__main__":
    main()
