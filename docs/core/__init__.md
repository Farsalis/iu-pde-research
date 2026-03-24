# `core.__init__`

## Module Purpose

`src/iu_pde_research/core/__init__.py` is the public export surface for the
project's reconstruction toolkit. It does not implement numerical methods by
itself. Instead, it gathers the CPU, JAX, and CuPy routines into a single
namespace so callers can access the same mathematical workflow through one
package.

At a high level, the package centers on the 1D heat equation with Dirichlet
boundary conditions:

$$
\begin{cases}
u_t(x,t) - u_{xx}(x,t) = F(x,t), & x \in (0,L), \ t > 0, \\
u(0,t) = u(L,t) = 0, & t > 0, \\
u(x,0) = f(x), & x \in (0,L).
\end{cases}
$$

The package exports functions that either:

- construct spectral coefficients,
- approximate forcing contributions,
- reconstruct the initial condition from pointwise-in-time measurements, or
- provide accelerated backends for the same reconstruction formulas.

## Public API Grouping

The module re-exports the following groups of functions:

- Forcing helpers:
  - `compute_exp_decay_cutoff`
  - `compute_F_hat_j`
  - `compute_forcing_contribution_at_time`
  - `compute_forcing_truncation_constant_C`
  - `compute_n_k`
- Reconstruction helpers:
  - `compute_modified_t_j`
  - `compute_modified_t_j_nonhomogeneous`
  - `compute_linear_t_j_homogeneous`
- Automatic differentiation helper:
  - `make_F_x_from_F`
- Accelerator backends:
  - `compute_modified_t_j_jax`
  - `compute_modified_t_j_nonhomogeneous_cupy`
- Environment capability:
  - `NUMBA_AVAILABLE`

## Mathematical Role

Although this file contains no numerical loops, it defines the conceptual
interface of the package. The exported functions collectively implement the
pipeline

$$
f(x)
\longrightarrow
u(x_0,t_k)
\longrightarrow
\bar f_{\hat k}
\longrightarrow
\bar f_n(x),
$$

where:

- $f(x)$ is the unknown initial condition,
- $u(x_0,t_k)$ is a measurement at one spatial location and many times,
- $\bar f_{\hat k}$ are reconstructed sine coefficients, and
- $\bar f_n(x)$ is the truncated reconstructed profile.

For the non-homogeneous problem, the same package-level surface also supports
the decomposition

$$
u(x,t) = u^{(0)}(x,t) + u^{(F)}(x,t),
$$

so the reconstruction code can separate the effect of the initial condition
from the effect of the forcing term.

## Optimization Relationship To The Mathematics

This file contains organizational optimization rather than numerical
optimization:

- It presents CPU and accelerator implementations through one API, which makes
  it easier to compare methods that are intended to represent the same
  mathematics.
- It keeps backend-specific execution details separate from the mathematical
  entry points. For example, a caller can choose a NumPy, JAX, or CuPy path
  without changing the governing PDE or the reconstruction formulas.

That separation matters mathematically because the accelerated modules are not
meant to solve a different problem. They are meant to evaluate the same
spectral expansions, recurrence relations, and truncation rules more
efficiently.

## What This Looks Like In Code

This file is structurally simple. Most of it looks like grouped imports and a
single public export list:

```python
from iu_pde_research.core.forcing import (
    compute_exp_decay_cutoff,
    compute_F_hat_j,
    compute_forcing_contribution_at_time,
)
from iu_pde_research.core.reconstruction import (
    compute_modified_t_j,
    compute_modified_t_j_nonhomogeneous,
)

__all__ = [
    "compute_exp_decay_cutoff",
    "compute_modified_t_j",
    "compute_modified_t_j_nonhomogeneous",
]
```

So when the docs say this file is the "public surface," the concrete coding
meaning is:

- it pulls functions in from the real implementation files,
- it lists which names are intended to be public, and
- it lets users import from `iu_pde_research.core` instead of remembering each
  submodule path.

Visually, the file acts like a hub:

```text
forcing.py ----------\
forcing_autodiff.py --\
reconstruction.py -----> core/__init__.py -----> user import
reconstruction_jax.py -/
reconstruction_cupy.py/
```

That is why this file has almost no numerical content of its own. Its coding
job is to connect the mathematically meaningful modules into one package-level
entry point.

## Constraints And Limitations

- This module only re-exports symbols; it does not validate mathematical
  assumptions on its own.
- Any backend-specific restrictions, such as CuPy assumptions about the forcing
  term or JAX assumptions about differentiability, come from the underlying
  modules rather than from `__init__.py`.
- The `__all__` list defines the intended public surface, so it also acts as a
  boundary between stable package-level math utilities and internal helpers.
