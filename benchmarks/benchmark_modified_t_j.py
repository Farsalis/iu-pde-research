"""Benchmarks for modified t_j reconstruction (CPU vs JAX).

Run with:

    python benchmarks/benchmark_modified_t_j.py

This compares:
* CPU implementation:  compute_modified_t_j (NumPy/SciPy)
* JAX implementation:  compute_modified_t_j_jax (GPU-capable)
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import numpy as np

from iu_pde_research.core import (
    compute_modified_t_j,
    compute_modified_t_j_jax,
)

try:
    import jax
except ImportError:  # pragma: no cover - benchmark script only
    jax = None  # type: ignore[assignment]


@dataclass
class BenchmarkConfig:
    n: int = 10
    m: int = 4000
    T: float = 5.0
    x0: float = float(np.sqrt(2.0))
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
    """Run CPU vs JAX benchmark for modified t_j."""
    x_grid = np.linspace(0.0, np.pi, cfg.m)

    print("=== Benchmark: modified t_j reconstruction ===")
    print(f"n = {cfg.n}, |x_grid| = {cfg.m}, T = {cfg.T}, x0 = {cfg.x0:.6f}")
    print()

    # Warm-up both implementations (includes JAX compilation on first call).
    print("Warming up CPU implementation...")
    compute_modified_t_j(cfg.n, cfg.T, cfg.x0, x_grid)

    if jax is None:
        print("JAX not installed; skipping JAX benchmark.")
        return

    print("Warming up JAX implementation (includes compilation)...")
    compute_modified_t_j_jax(cfg.n, cfg.T, cfg.x0, x_grid)

    # Time CPU implementation.
    print("\nTiming CPU (NumPy/SciPy) implementation...")
    cpu_times = _time_call(
        compute_modified_t_j,
        cfg.n,
        cfg.T,
        cfg.x0,
        x_grid,
        repeats=cfg.repeats,
    )
    cpu_mean = statistics.mean(cpu_times)
    cpu_std = statistics.pstdev(cpu_times)

    # Time JAX implementation (compiled).
    print("Timing JAX implementation (jit-compiled)...")
    jax_times = _time_call(
        compute_modified_t_j_jax,
        cfg.n,
        cfg.T,
        cfg.x0,
        x_grid,
        repeats=cfg.repeats,
    )
    jax_mean = statistics.mean(jax_times)
    jax_std = statistics.pstdev(jax_times)

    print("\n--- Results (wall-clock, seconds) ---")
    print(f"CPU  mean = {cpu_mean:.4f}  std = {cpu_std:.4f}")
    print(f"JAX  mean = {jax_mean:.4f}  std = {jax_std:.4f}")
    if jax_mean > 0.0:
        print(f"Speedup (CPU / JAX) ≈ {cpu_mean / jax_mean:.2f}×")

    if jax is not None:
        device_str = ", ".join(str(d) for d in jax.devices())
        print(f"\nJAX devices: {device_str}")


if __name__ == "__main__":  # pragma: no cover - manual benchmark
    cfg = BenchmarkConfig()
    run_benchmark(cfg)

