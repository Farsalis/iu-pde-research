"""Plotting functions for PDE reconstruction visualization."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from typing import Callable, Optional, Tuple
from math import sqrt, sin, cos, tan, exp, log, pi

from core.reconstruction import compute_modified_t_j


def plot_reconstruction(
    x_grid: np.ndarray,
    f_true_vals: np.ndarray,
    bar_f_n_vals: np.ndarray,
    title: str = "Initial Condition Reconstruction",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot true and reconstructed initial conditions.

    Args:
        x_grid: Grid of x values.
        f_true_vals: True initial condition values.
        bar_f_n_vals: Reconstructed initial condition values.
        title: Plot title. Defaults to "Initial Condition Reconstruction".
        save_path: Path to save figure. If None, figure is not saved.
        show: Whether to display the plot. Defaults to True.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, f_true_vals, "k-", linewidth=2, label="True f(x)")
    plt.plot(
        x_grid, bar_f_n_vals, "r--", linewidth=2, label="Reconstructed f_n(x)"
    )
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(title)
    plt.legend()
    plt.ylim(np.min(f_true_vals), np.max(f_true_vals))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()


def plot_reconstruction_with_errors(
    x_grid: np.ndarray,
    f_true_vals: np.ndarray,
    bar_f_n_vals: np.ndarray,
    title: str = "Initial Condition Reconstruction",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[float, float, float]:
    """
    Plot reconstruction with error metrics.

    Args:
        x_grid: Grid of x values.
        f_true_vals: True initial condition values.
        bar_f_n_vals: Reconstructed initial condition values.
        title: Plot title. Defaults to "Initial Condition Reconstruction".
        save_path: Path to save figure. If None, figure is not saved.
        show: Whether to display the plot. Defaults to True.

    Returns:
        Tuple of (l2_error, linf_error, mse) error metrics.
    """

    # Compute error metrics
    error = bar_f_n_vals - f_true_vals
    l2_error = np.sqrt(np.trapezoid(error**2, x_grid))  # Trapezoidal rule for integration
    linf_error = np.max(np.abs(error))
    mse = np.mean(error**2)

    print(f"Error Metrics:")
    print(f"  L2 norm: {l2_error:.6e}")
    print(f"  Linf norm: {linf_error:.6e}")
    print(f"  MSE: {mse:.6e}")

    plt.figure(figsize=(12, 6))
    plt.plot(x_grid, f_true_vals, "k-", linewidth=2, label="True f(x)")
    plt.plot(
        x_grid, bar_f_n_vals, "r--", linewidth=2, label="Reconstructed f_n(x)"
    )
    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    plt.title(title, fontsize=22)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()

    return l2_error, linf_error, mse


def create_interactive_plot() -> None:
    """
    Create interactive plot with parameter controls.

    Creates an interactive matplotlib plot with text boxes for adjusting
    n, T, and x0 parameters in real-time.
    """
    n_init = 20
    T_init = 5.0
    x0_init = np.sqrt(2)
    x0_init_str = "sqrt(2)"  # Using string representation to find actual value
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.25)
    x_grid = np.linspace(0, np.pi, 200)

    f_true_vals, bar_f_n_vals = compute_modified_t_j(
        n_init, T_init, x0_init, x_grid
    )

    line_true, = ax.plot(
        x_grid, f_true_vals, "k-", linewidth=2, label="True f(x)"
    )

    line_recon, = ax.plot(
        x_grid, bar_f_n_vals, "r--", linewidth=2, label="Reconstructed f_n(x)"
    )

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f(x)", fontsize=12)
    ax.set_title(
        "Interactive Initial Condition Reconstruction (Modified t_j)", fontsize=14
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    y_min, y_max = np.min(f_true_vals), np.max(f_true_vals)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    current_params = {
        "n": n_init,
        "T": T_init,
        "x0": x0_init,
        "x0_str": x0_init_str,
    }

    # TODO: Make this asynchronous 
    def update_plot(n, T, x0):
        """Update plot with new parameters."""
        ax.set_title(
            f"Interactive Reconstruction: n={n}, T={T:.2f}, x0={x0:.3f} "
            f"(Computing...)",
            fontsize=14,
            color="blue",
        )

        fig.canvas.draw()
        fig.canvas.flush_events()  # Clear GUI events loop

        try:
            f_true_vals, bar_f_n_vals = compute_modified_t_j(n, T, x0, x_grid)  # Reconstruction algorithm

            line_true.set_ydata(f_true_vals)
            line_recon.set_ydata(bar_f_n_vals)
            y_min, y_max = np.min(f_true_vals), np.max(f_true_vals)

            ax.set_ylim(y_min - 0.5, y_max + 0.5)
            ax.set_title(
                f"Interactive Reconstruction: n={n}, T={T:.2f}, x0={x0:.3f}",
                fontsize=14,
                color="black",
            )

            current_params["n"] = n
            current_params["T"] = T
            current_params["x0"] = x0

            fig.canvas.draw_idle()  # Redraw title widget
        except Exception as e:
            ax.set_title(f"Error: {str(e)}. Please try again.", fontsize=12, color="red")
            fig.canvas.draw_idle() 

    # Create text box labels
    ax_n_label = plt.axes([0.15, 0.18, 0.1, 0.03])
    ax_n_label.axis("off")
    ax_n_label.text(
        0,
        0.5,
        "n:",
        fontsize=11,
        verticalalignment="center",
        horizontalalignment="left",
    )
    ax_n = plt.axes([0.25, 0.18, 0.15, 0.04])
    textbox_n = TextBox(ax_n, "", initial=str(n_init))

    ax_T_label = plt.axes([0.15, 0.12, 0.1, 0.03])
    ax_T_label.axis("off")
    ax_T_label.text(
        0,
        0.5,
        "T:",
        fontsize=11,
        verticalalignment="center",
        horizontalalignment="left",
    )
    ax_T = plt.axes([0.25, 0.12, 0.15, 0.04])
    textbox_T = TextBox(ax_T, "", initial=str(T_init))

    ax_x0_label = plt.axes([0.15, 0.06, 0.1, 0.03])
    ax_x0_label.axis("off")
    ax_x0_label.text(
        0,
        0.5,
        "x0:",
        fontsize=11,
        verticalalignment="center",
        horizontalalignment="left",
    )
    ax_x0 = plt.axes([0.25, 0.06, 0.15, 0.04])
    textbox_x0 = TextBox(ax_x0, "", initial=x0_init_str)

    ax_instructions = plt.axes([0.45, 0.06, 0.5, 0.15])
    ax_instructions.axis("off")
    ax_instructions.text(0, 0.8, "Instructions:", fontsize=11, fontweight="bold")
    ax_instructions.text(
        0, 0.6, "Type a value in any box and press Enter", fontsize=10
    )
    ax_instructions.text(
        0, 0.4, "n: integer (number of measurements)", fontsize=10
    )
    ax_instructions.text(0, 0.2, "T: float (time parameter)", fontsize=10)
    ax_instructions.text(
        0,
        0.0,
        "x0: expression (e.g., sqrt(2), 2*pi/3, np.sqrt(3))",
        fontsize=10,
    )

    def submit_n(text):
        """Handle n parameter submission."""
        try:
            n = int(float(text))

            if n < 1:
                raise ValueError("n must be >= 1")

            update_plot(n, current_params["T"], current_params["x0"])
        except ValueError as e:
            ax.set_title(f"Invalid input for n: {str(e)}", fontsize=12, color="red")
            textbox_n.set_val(str(current_params["n"]))
            fig.canvas.draw_idle()

    def submit_T(text):
        """Handle T parameter submission."""
        try:
            T = float(text)
            if T <= 0:
                raise ValueError("T must be > 0")
            update_plot(current_params["n"], T, current_params["x0"])
        except ValueError as e:
            ax.set_title(f"Invalid input for T: {str(e)}", fontsize=12, color="red")
            textbox_T.set_val(str(current_params["T"]))
            fig.canvas.draw_idle()

    def submit_x0(text):
        """Handle x0 parameter submission."""
        try:
            if text is None:
                text = ""

            text = str(text).strip()

            if not text:
                raise ValueError("Empty input")

            safe_dict = {
                "sqrt": sqrt,
                "sin": sin,
                "cos": cos,
                "tan": tan,
                "exp": exp,
                "log": log,
                "pi": pi,
                "np": np,
            }

            x0 = float(eval(text, safe_dict))

            if x0 <= 0 or x0 >= np.pi:
                raise ValueError(
                    f"x0 must be in (0, pi) = (0, {np.pi:.3f})"
                )

            current_params["x0_str"] = text

            update_plot(current_params["n"], current_params["T"], x0)
        except (
            ValueError,
            SyntaxError,
            NameError,
            TypeError,
            ZeroDivisionError,
        ) as e:
            error_msg = str(e)

            if isinstance(e, NameError):
                error_msg = f"Unknown name in expression: {error_msg}"
            elif isinstance(e, SyntaxError):
                error_msg = f"Syntax error in expression: {error_msg}"
            elif isinstance(e, TypeError):
                error_msg = f"Type error: {error_msg}"
            elif "Empty input" in error_msg:
                error_msg = "Please enter a value"

            ax.set_title(
                f"Invalid input for x0: {error_msg}", fontsize=12, color="red"
            )

            textbox_x0.set_val(current_params["x0_str"])
            fig.canvas.draw_idle()

    textbox_n.on_submit(submit_n)
    textbox_T.on_submit(submit_T)
    textbox_x0.on_submit(submit_x0)

    plt.show()

