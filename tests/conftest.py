"""Shared pytest fixtures for PDE reconstruction tests."""

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path so "import nhh_re" works when running pytest
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from iu_pde_research.core.utils import get_default_f_true


@pytest.fixture
def x_grid() -> np.ndarray:
    """Grid of x values avoiding exact 0 and pi."""
    return np.linspace(0.01, np.pi - 0.01, 50)


@pytest.fixture
def small_n() -> int:
    """Small n for fast tests."""
    return 4


@pytest.fixture
def T() -> float:
    """Time scaling parameter."""
    return 1.0


@pytest.fixture
def x0() -> float:
    """Spatial measurement point."""
    return np.pi / 2


@pytest.fixture
def f_true_simple() -> Callable[[float], float]:
    """Simple callable for homogeneous tests (default f_true)."""
    return get_default_f_true


@pytest.fixture
def F_zero() -> Callable[[float, float], float]:
    """Zero forcing for homogeneous-via-nonhomogeneous tests."""

    def _zero(x: float, t: float) -> float:
        return 0.0

    return _zero
