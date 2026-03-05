"""Main entry point for PDE initial temperature profile reconstruction.

CLI for running reconstruction methods: by default runs modified t_j (homogeneous)
and linear sequence demos; --interactive opens a plot with n, T, x0 controls;
--nonhomogeneous runs one non-homogeneous demo; batch mode uses --algorithm with
--T, --x0, --n (and optional --output-dir). Run with --help for full options.
"""

import argparse
import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import golden

from iu_pde_research.core import (
    compute_linear_t_j_homogeneous,
    compute_modified_t_j,
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

    :args:
        s: A numeric string or expression using names from _EXPR_NAMESPACE.

    :returns:
        The evaluated float value.

    :raises:
        ValueError: If the string is not a valid number or expression.
    """
    s_stripped = s.strip()
    try:
        return float(s_stripped)
    except ValueError:
        pass
    try:
        result = eval(s_stripped, _EXPR_NAMESPACE, {})  # nosec B307 — restricted namespace for CLI
        return float(result)
    except Exception as e:
        raise ValueError(f"Invalid expression for float: {s_stripped!r}: {e}") from e


def _parse_int_expr(s: str) -> int:
    """
    Parse a string as an int or as a safe expression (e.g. 2 * 10).

    :args:
        s: A numeric string or expression using names from _EXPR_NAMESPACE.

    :returns:
        The evaluated integer value.

    :raises:
        ValueError: If the string is not a valid integer or expression.
    """
    s_stripped = s.strip()
    try:
        return int(s_stripped)
    except ValueError:
        pass
    try:
        result = eval(s_stripped, _EXPR_NAMESPACE, {})  # nosec B307 — restricted namespace for CLI
        return int(round(result))
    except Exception as e:
        raise ValueError(f"Invalid expression for int: {s_stripped!r}: {e}") from e


def _default_f_true_nonhomogeneous(x: float) -> float:
    """Initial condition used for non-homogeneous demo and batch."""
    return (1 / 8) * np.sin(2 * x) + (1 / 18) * np.sin(3 * x)


#############################################################################


def _default_F(x: float, t: float) -> float:
    """Forcing term used for non-homogeneous demo and batch."""
    # julia_returned_result = jl._default_julia_F(x, t)
    return np.sin(x) * np.exp(-t)


############################################################################


def _default_f_true_linear(x: float) -> float:
    """Initial condition used for linear sequence demo and batch."""
    return np.sin(2 * x) + 0.5 * np.sin(5 * x)


def _parse_args() -> argparse.Namespace:
    """
    Build and parse command-line arguments for demo and batch modes.

    :returns:
        Parsed namespace with optional interactive, nonhomogeneous, and
        batch fields (algorithm, T, x0, n, output_dir).
    """
    parser = argparse.ArgumentParser(
        description="PDE initial temperature profile reconstruction."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive plot with parameter controls.",
    )
    parser.add_argument(
        "--nonhomogeneous",
        action="store_true",
        help="Run non-homogeneous reconstruction once (demo).",
    )
    parser.add_argument(
        "--algorithm",
        choices=["modified", "nonhomogeneous", "linear"],
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
    return parser.parse_args()


def _batch_filename_stem(algorithm: str, T: float, x0: float, n: int) -> str:
    """
    Build a filesystem-safe filename stem for one batch run.

    :args:
        algorithm: One of 'modified', 'nonhomogeneous', 'linear'.
        T: Time parameter (or t0 for linear).
        x0: Spatial measurement point.
        n: Number of time measurements.

    :returns:
        Stem string suitable for a PNG filename (e.g. modified_T5_0_x0_1_41_n20).
    """
    t_str = str(T).replace(".", "_").replace("-", "m")
    x0_str = str(round(x0, 4)).replace(".", "_")
    return f"{algorithm}_T{t_str}_x0{x0_str}_n{n}"


def _run_single_reconstruction(
    algorithm: str,
    T: float,
    x0: float,
    n: int,
    output_dir: Path,
    x_grid: np.ndarray,
) -> None:
    """
    Run one reconstruction for the given (algorithm, T, x0, n) and save plot.

    :args:
        algorithm: One of 'modified', 'nonhomogeneous', 'linear'.
        T: Time parameter (or t0 for linear).
        x0: Spatial measurement point.
        n: Number of time measurements.
        output_dir: Directory to write the PNG into.
        x_grid: Grid of x values for reconstruction.
    """
    stem = _batch_filename_stem(algorithm, T, x0, n)
    save_path = output_dir / f"{stem}.png"

    if algorithm == "modified":
        f_true_vals, bar_f_n_vals = compute_modified_t_j(n, T, x0, x_grid)
        plot_reconstruction(
            x_grid,
            f_true_vals,
            bar_f_n_vals,
            title=f"Initial Condition Reconstruction (Modified t_j) n={n}, T={T}",
            save_path=str(save_path),
            show=False,
        )
    elif algorithm == "nonhomogeneous":
        f_true_vals, bar_f_n_vals = compute_modified_t_j_nonhomogeneous(
            n,
            T,
            x0,
            x_grid,
            F=_default_F,
            f_true=_default_f_true_nonhomogeneous,
        )
        plot_reconstruction_with_errors(
            x_grid,
            f_true_vals,
            bar_f_n_vals,
            title=(
                "Initial Temperature Profile Reconstruction (Non-Homogeneous)\n"
                f"n={n}, T={T}, x0={x0:.4f}"
            ),
            save_path=str(save_path),
            show=False,
        )
    else:
        assert algorithm == "linear"
        t0 = T
        f_true_vals, bar_f_n_vals = compute_linear_t_j_homogeneous(
            n, t0, x0, x_grid, f_true=_default_f_true_linear
        )
        plot_reconstruction(
            x_grid,
            f_true_vals,
            bar_f_n_vals,
            title=f"Initial Condition Reconstruction (Linear t_j) n={n}, t0={t0}",
            save_path=str(save_path),
            show=False,
        )
    plt.close()


def modified_t_j() -> None:
    """
    Run factorial t_j method for homogeneous case.

    Demonstrates the reconstruction of initial condition using the modified
    factorial t_j  for the homogeneous heat equation.
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


def modified_t_j_nonhomogeneous() -> None:
    """
    Run factorial t_j method for non-homogeneous case.

    Demonstrates the reconstruction of initial condition using the modified
    factorial t_j method for the non-homogeneous heat equation with a forcing term.
    """
    n = 20
    T = 5.0
    x0 = np.pi / golden
    x_grid = np.linspace(0, np.pi, 200)

    print("Computing non-homogeneous reconstruction...")
    print(f"Parameters: n={n}, T={T}, x0={x0:.4f}")
    print("Forcing term: F(x,t) = sin(x) * exp(-t)")

    f_true_vals, bar_f_n_vals = compute_modified_t_j_nonhomogeneous(
        n, T, x0, x_grid, F=_default_F, f_true=_default_f_true_nonhomogeneous
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


def main() -> None:
    """Main entry point for command-line interface."""
    args = _parse_args()

    if args.interactive:
        print("Starting interactive mode...")
        print("Use the text boxes to adjust n, T, and x0 parameters.")
        create_interactive_plot()
        return

    if args.nonhomogeneous:
        print("Running non-homogeneous case...")
        modified_t_j_nonhomogeneous()
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
            _run_single_reconstruction(args.algorithm, T, x0, n, output_dir, x_grid)
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

    print("Running Modified t_j method (homogeneous)...")
    modified_t_j()
    print("\nRunning Linear sequence method...")
    linear_sequence()
    print("\nDone!")
    print("\nTo run interactive mode, use: python nhh_re.py --interactive")
    print("To run non-homogeneous case, use: python nhh_re.py --nonhomogeneous")
    print(
        "For batch runs: python nhh_re.py --algorithm modified --T 5 --x0 1.41 "
        "--n 20 [--output-dir DIR]"
    )


if __name__ == "__main__":
    main()
