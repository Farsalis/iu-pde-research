"""Automatic differentiation of the forcing term for n_k truncation.

Provides F_x (spatial derivative of F) from F using JAX when the user does not
supply F_x explicitly. See agent_references/JAX Automatic Differentiation Explained.md.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

# Enable float64 so F_x and quad use consistent precision (avoids truncation warning).
jax.config.update("jax_enable_x64", True)


def make_F_x_from_F(F: Callable[..., Any]) -> Callable[[float, float], float]:
    """
    Build F_x(x, s) = ∂F/∂x(x, s) from F using JAX automatic differentiation.

    F must be written with JAX-friendly operations (e.g. jax.numpy) so that
    jax.grad can differentiate with respect to the first argument (x).
    The returned callable is a plain Python function returning floats, suitable
    for use with scipy.integrate.quad and compute_forcing_truncation_constant_C.

    Args:
        F: Forcing function F(x, t). Must accept JAX arrays and use JAX ops
            (e.g. jnp.sin, jnp.exp) so that jax.grad(F, argnums=0) is valid.

    Returns:
        F_x(x, s) returning the spatial derivative as a Python float.
    """
    dF_dx = jax.grad(F, argnums=0)
    dF_dx_jit = jax.jit(dF_dx)

    def F_x(x: float, s: float) -> float:
        out = dF_dx_jit(
            jnp.array(x, dtype=jnp.float64), jnp.array(s, dtype=jnp.float64)
        )
        return float(out)

    return F_x
