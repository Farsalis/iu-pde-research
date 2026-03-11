"""Benchmarks for non-homogeneous modified t_j (CPU vs CuPy).

Run with:

    python benchmarks/benchmark_modified_t_j_nonhomogeneous_cupy.py

This compares:
* CPU implementation:  compute_modified_t_j_nonhomogeneous (NumPy/SciPy)
* CuPy implementation: compute_modified_t_j_nonhomogeneous_cupy (GPU-capable)
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import numpy as np

from iu_pde_research.core import (
    compute_modified_t_j_nonhomogeneous,
    compute_modified_t_j_nonhomogeneous_cupy,
)

try:
    import cupy as cp
except ImportError:  # pragma: no cover - benchmark script only
    cp = None  # type: ignore[assignment]


@dataclass
class BenchmarkConfig:
    n: int = 20
    m: int = 4000
    T: float = 5.0
    x0: float = float(np.pi / 2.0)
    repeats: int = 3


def _time_call(func, *args, repeats: int = 3):
    """Return list of elapsed times in seconds for repeated calls."""
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def run_benchmark(cfg: BenchmarkConfig) -> None:
    """Run CPU vs CuPy benchmark for non-homogeneous modified t_j."""
    x_grid = np.linspace(0.0, np.pi, cfg.m)

    print("=== Benchmark: nonhomogeneous modified t_j reconstruction (CPU vs CuPy) ===")
    print(f"n = {cfg.n}, |x_grid| = {cfg.m}, T = {cfg.T}, x0 = {cfg.x0:.6f}")
    print()

    # Define default forcing F(x,t)=sin(x)*exp(-t) and its spatial derivative
    # F_x(x,t)=cos(x)*exp(-t) to match the CLI demo and avoid JAX autodiff in
    # the CPU path (we do not want make_F_x_from_F here).
    def F_default(x: float, t: float) -> float:
        return float(np.sin(x) * np.exp(-t))

    def F_x_default(x: float, t: float) -> float:
        return float(np.cos(x) * np.exp(-t))

    # Warm-up CPU implementation.
    print("Warming up CPU implementation...")
    compute_modified_t_j_nonhomogeneous(
        cfg.n, cfg.T, cfg.x0, x_grid, F=F_default, F_x=F_x_default
    )

    if cp is None:
        print("CuPy not installed; skipping CuPy benchmark.")
        return

    print("Warming up CuPy implementation (includes initial kernel setup)...")
    compute_modified_t_j_nonhomogeneous_cupy(
        cfg.n, cfg.T, cfg.x0, x_grid, F=F_default
    )

    # Time CPU implementation.
    print("\nTiming CPU (NumPy/SciPy) implementation...")
    cpu_times = _time_call(
        compute_modified_t_j_nonhomogeneous,
        cfg.n,
        cfg.T,
        cfg.x0,
        x_grid,
        F_default,
        F_x_default,
        repeats=cfg.repeats,
    )
    cpu_mean = statistics.mean(cpu_times)
    cpu_std = statistics.pstdev(cpu_times)

    # Time CuPy implementation.
    print("Timing CuPy implementation (GPU)...")
    cupy_times = _time_call(
        compute_modified_t_j_nonhomogeneous_cupy,
        cfg.n,
        cfg.T,
        cfg.x0,
        x_grid,
        F_default,
        repeats=cfg.repeats,
    )
    cupy_mean = statistics.mean(cupy_times)
    cupy_std = statistics.pstdev(cupy_times)

    print("\n--- Results (wall-clock, seconds) ---")
    print(f"CPU   mean = {cpu_mean:.4f}  std = {cpu_std:.4f}")
    print(f"CuPy  mean = {cupy_mean:.4f}  std = {cupy_std:.4f}")
    if cupy_mean > 0.0:
        print(f"Speedup (CPU / CuPy) ≈ {cpu_mean / cupy_mean:.2f}×")

    if cp is not None:
        dev = cp.cuda.runtime.getDeviceProperties(0)
        name = dev["name"].decode() if isinstance(dev["name"], bytes) else dev["name"]
        print(f"\nCuPy device: {name}")


if __name__ == "__main__":  # pragma: no cover - manual benchmark
    cfg = BenchmarkConfig()
    run_benchmark(cfg)

