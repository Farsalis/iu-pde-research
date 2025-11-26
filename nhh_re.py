import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from scipy.integrate import quad
from math import factorial, sqrt, sin, cos, tan, exp, log, pi

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

_f_hat_cache = {}

def _get_f_hat_true(j):
    if j not in _f_hat_cache:
        def f_true(x):
            return np.sin(2 * x) + 0.5 * np.sin(5 * x)
        integrand = lambda x: f_true(x) * np.sin(j * x)
        result, _ = quad(integrand, 0, np.pi)
        _f_hat_cache[j] = result * 2 / np.pi
    return _f_hat_cache[j]


def compute_modified_t_j(n, T, x0, x_grid):
    def compute_t_vec_optimized(n, T):
        t_vec = np.zeros(n)
        fact_cache = np.ones(2 * n + 1, dtype=np.float64)
        for i in range(1, 2 * n + 1):
            fact_cache[i] = fact_cache[i-1] * i
        for k in range(1, n + 1):
            numerator = fact_cache[2 * k - 1]
            denominator = (8**(k - 1)) * fact_cache[k] * fact_cache[k - 1]
            t_vec[k - 1] = (numerator / denominator) * T
        return t_vec
    if NUMBA_AVAILABLE:
        compute_t_vec_optimized = jit(nopython=True, cache=True)(compute_t_vec_optimized)
        t_vec = compute_t_vec_optimized(n, T)
    else:
        t_vec = compute_t_vec_optimized(n, T)
    def f_true(x):
        return np.sin(2 * x) + 0.5 * np.sin(5 * x)
    n_terms = 20
    j_vals = np.arange(1, n_terms + 1)
    f_hat_vals = np.array([_get_f_hat_true(j) for j in j_vals])
    j_squared = j_vals**2
    sin_j_x0 = np.sin(j_vals * x0)
    exp_matrix = np.exp(-np.outer(t_vec, j_squared))
    u_data = np.sum(exp_matrix * f_hat_vals * sin_j_x0, axis=1)
    bar_f_hat = np.zeros(int(np.ceil(n / 2)))
    sin_k_x0 = np.sin(np.arange(1, len(bar_f_hat) + 1) * x0)
    max_k = len(bar_f_hat)
    j_squared_cache = np.arange(1, max_k + 1)**2
    for k in range(1, max_k + 1):
        tk = t_vec[k - 1]
        if k == 1:
            bar_f_hat[0] = np.exp(tk) * u_data[0] / sin_k_x0[0]
        else:
            j_indices = np.arange(1, k)
            exp_terms = np.exp(-j_squared_cache[j_indices - 1] * tk)
            sum_prev = np.sum(exp_terms * bar_f_hat[j_indices - 1] * 
                            sin_j_x0[j_indices - 1])
            k_squared_tk = j_squared_cache[k - 1] * tk
            bar_f_hat[k - 1] = np.exp(k_squared_tk) * (u_data[k - 1] - sum_prev) / sin_k_x0[k - 1]
    k_vals = np.arange(1, len(bar_f_hat) + 1)
    sin_matrix = np.sin(np.outer(x_grid, k_vals))
    bar_f_n_vals = sin_matrix @ bar_f_hat
    f_true_vals = f_true(x_grid)
    return f_true_vals, bar_f_n_vals


def modified_t_j():
    n = 20
    T = 15
    x0 = 2*np.sqrt(3)
    x_grid = np.linspace(0, np.pi, 200)
    f_true_vals, bar_f_n_vals = compute_modified_t_j(n, T, x0, x_grid)
    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, f_true_vals, 'k-', linewidth=2, label='True f(x)')
    plt.plot(x_grid, bar_f_n_vals, 'r--', linewidth=2, label='Reconstructed f_n(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Initial Condition Reconstruction (Modified t_j)')
    plt.legend()
    plt.ylim(np.min(f_true_vals), np.max(f_true_vals))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('modified_t_j.png', dpi=150)
    plt.show()


def interactive_modified_t_j():
    n_init = 20
    T_init = 5.0
    x0_init = np.sqrt(2)
    x0_init_str = "sqrt(2)"
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.25)
    x_grid = np.linspace(0, np.pi, 200)
    f_true_vals, bar_f_n_vals = compute_modified_t_j(n_init, T_init, x0_init, x_grid)
    line_true, = ax.plot(x_grid, f_true_vals, 'k-', linewidth=2, label='True f(x)')
    line_recon, = ax.plot(x_grid, bar_f_n_vals, 'r--', linewidth=2, label='Reconstructed f_n(x)')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title('Interactive Initial Condition Reconstruction (Modified t_j)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    error_init = bar_f_n_vals - f_true_vals
    l2_error_init = np.sqrt(np.trapezoid(error_init**2, x_grid))
    linf_error_init = np.max(np.abs(error_init))
    mse_init = np.mean(error_init**2)
    error_text = ax.text(0.02, 0.98, 
                         f'Error Metrics:\n'
                         f'L2 norm: {l2_error_init:.6e}\n'
                         f'L∞ norm: {linf_error_init:.6e}\n'
                         f'MSE: {mse_init:.6e}',
                         transform=ax.transAxes, 
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    y_min, y_max = np.min(f_true_vals), np.max(f_true_vals)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    current_params = {'n': n_init, 'T': T_init, 'x0': x0_init, 'x0_str': x0_init_str}
    def update_plot(n, T, x0):
        ax.set_title(f'Interactive Reconstruction: n={n}, T={T:.2f}, x0={x0:.3f} (Computing...)', 
                     fontsize=14, color='blue')
        fig.canvas.draw()
        fig.canvas.flush_events()
        try:
            f_true_vals, bar_f_n_vals = compute_modified_t_j(n, T, x0, x_grid)
            error = bar_f_n_vals - f_true_vals
            l2_error = np.sqrt(np.trapezoid(error**2, x_grid))
            linf_error = np.max(np.abs(error))
            mse = np.mean(error**2)
            line_true.set_ydata(f_true_vals)
            line_recon.set_ydata(bar_f_n_vals)
            y_min, y_max = np.min(f_true_vals), np.max(f_true_vals)
            ax.set_ylim(y_min - 0.5, y_max + 0.5)
            ax.set_title(f'Interactive Reconstruction: n={n}, T={T:.2f}, x0={x0:.3f}', 
                         fontsize=14, color='black')
            error_text.set_text(
                f'Error Metrics:\n'
                f'L2 norm: {l2_error:.6e}\n'
                f'L∞ norm: {linf_error:.6e}\n'
                f'MSE: {mse:.6e}'
            )
            current_params['n'] = n
            current_params['T'] = T
            current_params['x0'] = x0
            fig.canvas.draw_idle()
        except Exception as e:
            ax.set_title(f'Error: {str(e)}', fontsize=12, color='red')
            error_text.set_text('Error: Could not compute reconstruction')
            fig.canvas.draw_idle()
    ax_n_label = plt.axes([0.15, 0.18, 0.1, 0.03])
    ax_n_label.axis('off')
    ax_n_label.text(0, 0.5, 'n:', fontsize=11, verticalalignment='center', horizontalalignment='left')
    ax_n = plt.axes([0.25, 0.18, 0.15, 0.04])
    textbox_n = TextBox(ax_n, '', initial=str(n_init))
    ax_T_label = plt.axes([0.15, 0.12, 0.1, 0.03])
    ax_T_label.axis('off')
    ax_T_label.text(0, 0.5, 'T:', fontsize=11, verticalalignment='center', horizontalalignment='left')
    ax_T = plt.axes([0.25, 0.12, 0.15, 0.04])
    textbox_T = TextBox(ax_T, '', initial=str(T_init))
    ax_x0_label = plt.axes([0.15, 0.06, 0.1, 0.03])
    ax_x0_label.axis('off')
    ax_x0_label.text(0, 0.5, 'x0:', fontsize=11, verticalalignment='center', horizontalalignment='left')
    ax_x0 = plt.axes([0.25, 0.06, 0.15, 0.04])
    textbox_x0 = TextBox(ax_x0, '', initial=x0_init_str)
    ax_instructions = plt.axes([0.45, 0.06, 0.5, 0.15])
    ax_instructions.axis('off')
    ax_instructions.text(0, 0.8, 'Instructions:', fontsize=11, fontweight='bold')
    ax_instructions.text(0, 0.6, '• Type a value in any box and press Enter', fontsize=10)
    ax_instructions.text(0, 0.4, '• n: integer (number of measurements)', fontsize=10)
    ax_instructions.text(0, 0.2, '• T: float (time parameter)', fontsize=10)
    ax_instructions.text(0, 0.0, '• x0: expression (e.g., sqrt(2), 2*pi/3, np.sqrt(3))', fontsize=10)
    def submit_n(text):
        try:
            n = int(float(text))
            if n < 1:
                raise ValueError("n must be >= 1")
            update_plot(n, current_params['T'], current_params['x0'])
        except ValueError as e:
            ax.set_title(f'Invalid input for n: {str(e)}', fontsize=12, color='red')
            textbox_n.set_val(str(current_params['n']))
            fig.canvas.draw_idle()
    def submit_T(text):
        try:
            T = float(text)
            if T <= 0:
                raise ValueError("T must be > 0")
            update_plot(current_params['n'], T, current_params['x0'])
        except ValueError as e:
            ax.set_title(f'Invalid input for T: {str(e)}', fontsize=12, color='red')
            textbox_T.set_val(str(current_params['T']))
            fig.canvas.draw_idle()
    def submit_x0(text):
        try:
            if text is None:
                text = ""
            text = str(text).strip()
            if not text:
                raise ValueError("Empty input")
            safe_dict = {
                'sqrt': sqrt,
                'sin': sin,
                'cos': cos,
                'tan': tan,
                'exp': exp,
                'log': log,
                'pi': pi,
                'np': np,
            }
            x0 = float(eval(text, safe_dict))
            if x0 <= 0 or x0 >= np.pi:
                raise ValueError(f"x0 must be in (0, π) ≈ (0, {np.pi:.3f})")
            current_params['x0_str'] = text
            update_plot(current_params['n'], current_params['T'], x0)
        except (ValueError, SyntaxError, NameError, TypeError, ZeroDivisionError) as e:
            error_msg = str(e)
            if isinstance(e, NameError):
                error_msg = f"Unknown name in expression: {error_msg}"
            elif isinstance(e, SyntaxError):
                error_msg = f"Syntax error in expression: {error_msg}"
            elif isinstance(e, TypeError):
                error_msg = f"Type error: {error_msg}"
            elif "Empty input" in error_msg:
                error_msg = "Please enter a value"
            ax.set_title(f'Invalid input for x0: {error_msg}', fontsize=12, color='red')
            textbox_x0.set_val(current_params['x0_str'])
            fig.canvas.draw_idle()
    textbox_n.on_submit(submit_n)
    textbox_T.on_submit(submit_T)
    textbox_x0.on_submit(submit_x0)
    plt.show()


def linear_sequence():
    n = 20
    t0 = 1e-3
    x0 = np.sqrt(2)
    x_grid = np.linspace(0, np.pi, 200)
    def t_k(k, T):
        return (n + k - 1) * T
    t_vec = np.array([t_k(k, t0) for k in range(1, n + 1)])
    def f_true(x):
        return np.sin(2 * x) + 0.5 * np.sin(5 * x)
    def f_hat_true(j):
        integrand = lambda x: f_true(x) * np.sin(j * x)
        result, _ = quad(integrand, 0, np.pi)
        return result * 2 / np.pi
    def u_xtk(x0, t_k, n_terms=50):
        results = []
        for t in t_k:
            sum_val = 0
            for j in range(1, n_terms + 1):
                sum_val += np.exp(-j**2 * t) * f_hat_true(j) * np.sin(j * x0)
            results.append(sum_val)
        return np.array(results)
    u_data = u_xtk(x0, t_vec)
    bar_f_hat = np.zeros(int(np.ceil(n / 2)))
    for k in range(1, len(bar_f_hat) + 1):
        tk = t_vec[k - 1]
        if k == 1:
            bar_f_hat[0] = np.exp(tk) * u_data[0] / np.sin(x0)
        else:
            sum_prev = sum([
                np.exp(-j**2 * tk) * bar_f_hat[j - 1] * np.sin(j * x0)
                for j in range(1, k)
            ])
            bar_f_hat[k - 1] = np.exp(k**2 * tk) * (u_data[k - 1] - sum_prev) / np.sin(k * x0)
    def bar_f_n(x):
        return sum([
            bar_f_hat[k - 1] * np.sin(k * x)
            for k in range(1, len(bar_f_hat) + 1)
        ])
    f_true_vals = f_true(x_grid)
    bar_f_n_vals = np.array([bar_f_n(x) for x in x_grid])
    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, f_true_vals, 'k-', linewidth=2, label='True f(x)')
    plt.plot(x_grid, bar_f_n_vals, 'r--', linewidth=2, label='Reconstructed f_n(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Initial Condition Reconstruction (Linear Sequence)')
    plt.legend()
    plt.ylim(np.min(f_true_vals), np.max(f_true_vals))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('linear_sequence.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        print("Starting interactive mode...")
        print("Use the sliders to adjust n, T, and x0 parameters.")
        interactive_modified_t_j()
    else:
        print("Running Modified t_j method...")
        modified_t_j()
        print("\nRunning Linear sequence method...")
        linear_sequence()
        print("\nDone!")
        print("\nTo run interactive mode, use: python nhh_re.py --interactive")