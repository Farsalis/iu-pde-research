"""Main entry point for PDE initial temperature profile reconstruction.

CLI for running reconstruction methods: by default runs modified t_j (homogeneous)
and linear sequence demos; --interactive opens a plot with n, T, x0 controls;
--nonhomogeneous runs one non-homogeneous demo; batch mode uses --algorithm with
--T, --x0, --n (and optional --output-dir). Neumann BCs are supported on the
CPU path for modified t_j homogeneous and non-homogeneous workflows. Run with
--help for full options.
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
    get_default_f_true,
)
from iu_pde_research.core.basis import (
    BoundaryCondition,
    ZeroModeMethod,
    validate_boundary_condition,
    validate_zero_mode_method,
)
from iu_pde_research.visualization import (
    create_interactive_plot,
    plot_reconstruction,
    plot_reconstruction_with_errors,
)

DEFAULT_OUTPUT_DIR = "reconstruction_output"

# Named initial conditions in INITIAL_CONDITION_OPTIONS that use a sine-mode
# basis. These are mathematically incompatible with Neumann boundary conditions
# (∂f/∂x ≠ 0 at x=0,L for sin(j*x)) and are rejected at the CLI when --bc neumann
# is set.
_SINE_BASIS_INITIAL_CONDITIONS = frozenset(
    {
        "sine_series_5",
        "sine_series_10",
        "sine_series_15",
        "sine_series_new_5",
        "sine_series_new_10",
        "sine_series_new_15",
    }
)

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


def _default_f_true_neumann(x: float | np.ndarray) -> float | np.ndarray:
    """Initial condition used for Neumann demo and batch (cosine series + mean)."""
    values = 1.25 + 0.5 * np.cos(x)
    if np.ndim(values) == 0:
        return float(values)
    return values


def _validate_bc_cli_options(
    bc: BoundaryCondition,
    backend: str,
    zero_mode: ZeroModeMethod,
    f_hat_0: float | None,
    initial_condition: str = "default",
) -> None:
    """Validate Neumann-specific CLI combinations before running reconstruction."""
    if bc == "neumann" and backend in {"jax", "gpu"}:
        msg = (
            "Neumann boundary conditions are CPU-only; use --backend cpu "
            f"(got {backend!r})."
        )
        raise ValueError(msg)
    if bc == "neumann" and initial_condition in _SINE_BASIS_INITIAL_CONDITIONS:
        msg = (
            f"--initial-condition {initial_condition!r} uses a sine-mode basis "
            "and is incompatible with --bc neumann (sin(j*x) violates "
            "∂f/∂x|_boundary = 0). Choose a Neumann-compatible initial "
            "condition or omit --initial-condition to use the cosine default."
        )
        raise ValueError(msg)
    if bc == "dirichlet":
        if zero_mode != "known":
            msg = "--zero-mode is only valid with --bc neumann."
            raise ValueError(msg)
        if f_hat_0 is not None:
            msg = "--f-hat-0 is only valid with --bc neumann."
            raise ValueError(msg)
    if zero_mode == "measurement" and f_hat_0 is not None:
        msg = "--f-hat-0 cannot be combined with --zero-mode measurement."
        raise ValueError(msg)


def _default_f_true_for_run(
    base_algorithm: str,
    bc: BoundaryCondition,
) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """Return the algorithm-specific default initial condition when none is named."""
    del base_algorithm  # All current Dirichlet defaults share get_default_f_true.
    if bc == "neumann":
        return _default_f_true_neumann
    return get_default_f_true


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
        help=(
            "Reconstruction algorithm for batch mode. "
            "Use --bc neumann with modified_homogeneous or "
            "modified_nonhomogeneous for Neumann runs."
        ),
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
    parser.add_argument(
        "--bc",
        choices=["dirichlet", "neumann"],
        default="dirichlet",
        help=(
            "Boundary condition basis. Neumann uses cosine modes on the CPU path "
            "for modified t_j homogeneous and non-homogeneous workflows."
        ),
    )
    parser.add_argument(
        "--zero-mode",
        choices=["known", "measurement"],
        default="known",
        help=(
            "Neumann zero-mode handling: 'known' uses a supplied or quadrature "
            "mean; 'measurement' estimates the mean from the first sensor reading "
            "at t_1 = T (large T recommended)."
        ),
    )
    parser.add_argument(
        "--f-hat-0",
        type=_parse_float_expr,
        default=None,
        metavar="F_HAT_0",
        help=(
            "Known Neumann mean coefficient f_hat_0 when --bc neumann and "
            "--zero-mode known. If omitted, computed from f_true via quadrature."
        ),
    )
    parser.add_argument(
        "--L",
        type=_parse_float_expr,
        default=np.pi,
        metavar="L",
        help="Domain length (default: pi).",
    )
    return parser.parse_args()


def _batch_filename_stem(
    algorithm: str,
    T: float,
    x0: float,
    n: int,
    *,
    bc: BoundaryCondition = "dirichlet",
    zero_mode: ZeroModeMethod = "known",
) -> str:
    """
    Build a filesystem-safe filename stem for one batch run.

    Args:
        algorithm: CLI algorithm key (e.g. ``"modified_homogeneous"``,
            ``"modified_nonhomogeneous"``, ``"linear_homogeneous"``).
        T: Time parameter (or t0 for linear).
        x0: Spatial measurement point.
        n: Number of time measurements.
        bc: Boundary condition basis.
        zero_mode: Neumann zero-mode method when ``bc='neumann'``.

    Returns:
        Stem string suitable for a PNG filename
        (e.g. ``modified_homogeneous_T5_0_x0_1_41_n20``).
    """
    t_str = str(T).replace(".", "_").replace("-", "m")
    x0_str = str(round(x0, 4)).replace(".", "_")
    stem = f"{algorithm}_T{t_str}_x0{x0_str}_n{n}"
    if bc == "neumann":
        stem = f"{stem}_neumann_{zero_mode}"
    return stem


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
    *,
    bc: BoundaryCondition = "dirichlet",
    zero_mode: ZeroModeMethod = "known",
    f_hat_0: float | None = None,
    L: float = np.pi,
) -> None:
    """
    Run one reconstruction for the given (algorithm, T, x0, n) and save a plot.

    Args:
        algorithm: CLI algorithm key (e.g. ``"modified_homogeneous"``,
            ``"modified_nonhomogeneous"``, ``"linear_homogeneous"``).
        T: Time parameter (or t0 for linear).
        x0: Spatial measurement point.
        n: Number of time measurements.
        output_dir: Directory to write the PNG into.
        x_grid: Grid of x values for reconstruction.
        backend: Execution backend (``cpu``, ``jax``, or ``gpu``).
        initial_condition: Named initial condition or ``default``.
        bc: Boundary condition basis.
        zero_mode: Neumann zero-mode method.
        f_hat_0: Optional known Neumann mean coefficient.
        L: Domain length.
    """
    _validate_bc_cli_options(bc, backend, zero_mode, f_hat_0, initial_condition)

    stem = _batch_filename_stem(
        algorithm,
        T,
        x0,
        n,
        bc=bc,
        zero_mode=zero_mode,
    )
    save_path = output_dir / f"{stem}.png"

    f_true_callable = _get_f_true_for_algorithm(algorithm, initial_condition)
    f_true = f_true_callable or _default_f_true_for_run(algorithm, bc)
    neumann_kwargs = {
        "bc": bc,
        "zero_mode": zero_mode,
        "f_hat_0": f_hat_0,
        "L": L,
    }

    if algorithm == "modified_homogeneous":
        if bc == "neumann":
            f_true_vals, bar_f_n_vals = compute_modified_t_j(
                n,
                T,
                x0,
                x_grid,
                f_true=f_true,
                **neumann_kwargs,
            )
            title_bc = "Neumann"
        elif backend == "jax":
            f_true_vals, bar_f_n_vals = compute_modified_t_j_jax(n, T, x0, x_grid)
            title_bc = "Dirichlet"
        else:
            f_true_vals, bar_f_n_vals = compute_modified_t_j(
                n,
                T,
                x0,
                x_grid,
                f_true=f_true_callable,
                L=L,
            )
            title_bc = "Dirichlet"
        plot_reconstruction(
            x_grid,
            f_true_vals,
            bar_f_n_vals,
            title=(
                f"Initial Temperature Profile Reconstruction "
                f"(Modified t_j - Homogeneous, {title_bc})\n"
                f"n={n}, T={T}, x0={x0:.4f}"
            ),
            save_path=str(save_path),
            show=False,
        )
    elif algorithm == "modified_nonhomogeneous":
        if bc == "neumann":
            f_true_vals, bar_f_n_vals = compute_modified_t_j_nonhomogeneous(
                n,
                T,
                x0,
                x_grid,
                F=_default_F,
                f_true=f_true,
                backend="cpu",
                **neumann_kwargs,
            )
            title_bc = "Neumann"
        elif backend == "gpu":
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
                f_true=f_true_callable or get_default_f_true,
                backend="gpu",
                F_gpu=default_F_cupy,
                F_x_gpu=default_F_x_cupy,
            )
            title_bc = "Dirichlet"
        else:
            f_true_vals, bar_f_n_vals = compute_modified_t_j_nonhomogeneous(
                n,
                T,
                x0,
                x_grid,
                F=_default_F,
                f_true=f_true,
                L=L,
            )
            title_bc = "Dirichlet"
        plot_reconstruction(
            x_grid,
            f_true_vals,
            bar_f_n_vals,
            title=(
                "Initial Temperature Profile Reconstruction "
                f"(Modified t_j - Non-Homogeneous, {title_bc})\n"
                f"n={n}, T={T}, x0={x0:.4f}"
            ),
            save_path=str(save_path),
            show=False,
        )
    elif algorithm == "modified_complex_homogeneous":
        if bc == "neumann":
            msg = "Neumann boundary conditions are not supported for the complex homogeneous algorithm."
            raise ValueError(msg)
        f_true_vals, bar_f_n_vals = compute_modified_t_j_homogeneous_complex(
            n, T, x0, x_grid
        )
        plot_reconstruction(
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
        if algorithm != "linear_homogeneous":
            raise ValueError(f"Unknown algorithm: {algorithm!r}")
        if bc == "neumann":
            msg = "Neumann boundary conditions are not supported for the linear t_j algorithm."
            raise ValueError(msg)
        t0 = T
        f_true_vals, bar_f_n_vals = compute_linear_t_j_homogeneous(
            n,
            t0,
            x0,
            x_grid,
            f_true=f_true,
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
        f_true = get_default_f_true

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
        n, t0, x0, x_grid, f_true=get_default_f_true
    )

    plot_reconstruction(
        x_grid,
        f_true_vals,
        bar_f_n_vals,
        title="Initial Condition Reconstruction (Linear t_j)",
        save_path="linear_sequence.png",
    )


def _cli_bc_options(
    args: argparse.Namespace,
) -> tuple[BoundaryCondition, ZeroModeMethod, float | None, float]:
    """Parse and validate boundary-condition-related CLI options."""
    bc = validate_boundary_condition(args.bc)
    zero_mode = validate_zero_mode_method(args.zero_mode)
    f_hat_0 = args.f_hat_0
    domain_length = float(args.L)
    _validate_bc_cli_options(
        bc, args.backend, zero_mode, f_hat_0, args.initial_condition
    )
    return bc, zero_mode, f_hat_0, domain_length


def modified_t_j_neumann(
    *,
    zero_mode: ZeroModeMethod = "known",
    f_hat_0: float | None = None,
    L: float = np.pi,
    f_true: Callable[[float | np.ndarray], float | np.ndarray] | None = None,
) -> None:
    """
    Run modified t_j homogeneous reconstruction with Neumann boundary conditions.

    Uses a cosine-series initial condition and CPU-only Neumann workflow.
    """
    n = 20
    T = 20.0 if zero_mode == "measurement" else 0.1
    x0 = 0.7
    x_grid = np.linspace(0.01, L - 0.01, 200)
    if f_true is None:
        f_true = _default_f_true_neumann

    print("Computing Neumann homogeneous reconstruction...")
    print(f"Parameters: n={n}, T={T}, x0={x0:.4f}, zero_mode={zero_mode}, L={L:.4f}")

    f_true_vals, bar_f_n_vals = compute_modified_t_j(
        n,
        T,
        x0,
        x_grid,
        f_true=f_true,
        L=L,
        bc="neumann",
        f_hat_0=f_hat_0,
        zero_mode=zero_mode,
    )
    plot_reconstruction_with_errors(
        x_grid,
        f_true_vals,
        bar_f_n_vals,
        title=(
            "Initial Temperature Profile Reconstruction (Modified t_j - Homogeneous, Neumann)\n"
            f"n={n}, T={T}, zero_mode={zero_mode}\n"
            r"$f(x)=1.25+\frac{1}{2}\cos(x)$"
        ),
        save_path="modified_t_j_neumann.png",
    )


def modified_t_j_neumann_nonhomogeneous(
    *,
    zero_mode: ZeroModeMethod = "known",
    f_hat_0: float | None = None,
    L: float = np.pi,
    f_true: Callable[[float | np.ndarray], float | np.ndarray] | None = None,
) -> None:
    """
    Run modified t_j non-homogeneous reconstruction with Neumann boundary conditions.

    Uses the default forcing F(x,t) = sin(x) * exp(-t) on the CPU path.
    """
    n = 20
    T = 20.0 if zero_mode == "measurement" else 0.1
    x0 = 0.7
    x_grid = np.linspace(0.01, L - 0.01, 200)
    if f_true is None:
        f_true = _default_f_true_neumann

    print("Computing Neumann non-homogeneous reconstruction...")
    print(f"Parameters: n={n}, T={T}, x0={x0:.4f}, zero_mode={zero_mode}, L={L:.4f}")
    print("Forcing term: F(x,t) = sin(x) * exp(-t)")

    f_true_vals, bar_f_n_vals = compute_modified_t_j_nonhomogeneous(
        n,
        T,
        x0,
        x_grid,
        F=_default_F,
        f_true=f_true,
        L=L,
        backend="cpu",
        bc="neumann",
        f_hat_0=f_hat_0,
        zero_mode=zero_mode,
    )
    plot_reconstruction_with_errors(
        x_grid,
        f_true_vals,
        bar_f_n_vals,
        title=(
            "Initial Temperature Profile Reconstruction "
            "(Modified t_j - Non-Homogeneous, Neumann)\n"
            f"n={n}, T={T}, zero_mode={zero_mode}\n"
            r"$f(x)=1.25+\frac{1}{2}\cos(x)$"
            "\n"
            r"$F(x,t)=\sin(x)\exp(-t)$"
        ),
        save_path="modified_t_j_neumann_nonhomogeneous.png",
    )


############################################################################


def main() -> None:
    """Main entry point for command-line interface."""
    args = _parse_args()

    try:
        bc, zero_mode, f_hat_0, domain_length = _cli_bc_options(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    f_true_from_cli = (
        None
        if args.initial_condition == "default"
        else INITIAL_CONDITION_OPTIONS[args.initial_condition]
    )

    if args.interactive:
        print("Starting interactive mode for non-homogeneous case...")
        print("Use the text boxes to adjust n, T, and x0 parameters.")
        create_interactive_plot()
        return
    if args.modified_homogeneous and bc == "neumann":
        print(
            "Running Neumann homogeneous case "
            f"(zero_mode={zero_mode}, L={domain_length:.4f})..."
        )
        modified_t_j_neumann(
            zero_mode=zero_mode,
            f_hat_0=f_hat_0,
            L=domain_length,
            f_true=f_true_from_cli,
        )
        return
    if args.modified_nonhomogeneous and bc == "neumann":
        print(
            "Running Neumann non-homogeneous case "
            f"(zero_mode={zero_mode}, L={domain_length:.4f})..."
        )
        modified_t_j_neumann_nonhomogeneous(
            zero_mode=zero_mode,
            f_hat_0=f_hat_0,
            L=domain_length,
            f_true=f_true_from_cli,
        )
        return
    if args.modified_nonhomogeneous:
        print(f"Running non-homogeneous case (backend={args.backend})...")
        modified_t_j_nonhomogeneous(backend=args.backend, f_true=f_true_from_cli)
        return
    if args.modified_homogeneous:
        print("Running homogeneous case...")
        modified_t_j()
        return
    if args.modified_complex_homogeneous:
        print("Running homogeneous complex case...")
        modified_t_j_homogeneous_complex()
        return
    if args.linear_homogeneous:
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
                bc=bc,
                zero_mode=zero_mode,
                f_hat_0=f_hat_0,
                L=domain_length,
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
        "\nTo run Neumann homogeneous case, use: python nhh_re.py "
        "--modified_homogeneous --bc neumann "
        "[--zero-mode known|measurement] [--f-hat-0 VALUE] [--L VALUE]"
    )
    print(
        "\nTo run Neumann non-homogeneous case, use: python nhh_re.py "
        "--modified_nonhomogeneous --bc neumann "
        "[--zero-mode known|measurement] [--f-hat-0 VALUE] [--L VALUE]"
    )
    print(
        "For batch runs: python nhh_re.py --algorithm modified_homogeneous --T 5 --x0 1.41 "
        "--n 20 [--output-dir DIR]"
    )
    print(
        "Neumann batch example: python nhh_re.py --algorithm modified_homogeneous "
        "--bc neumann --T 0.1 --x0 0.7 --n 4"
    )


if __name__ == "__main__":
    main()
