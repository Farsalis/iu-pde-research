"""Tests for nhh_re CLI and helpers."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Import the module under test (run from repo root so nhh_re is on path)
import nhh_re as nhh


@pytest.mark.parametrize(
    "input_str,expected",
    [
        ("1.0", 1.0),
        ("0", 0.0),
        ("np.pi", np.pi),
        ("sqrt(2)", np.sqrt(2)),
        ("2*pi", 2 * np.pi),
    ],
)
def test_parse_float_expr_valid(input_str: str, expected: float) -> None:
    """_parse_float_expr parses valid numeric strings and expressions."""
    assert nhh._parse_float_expr(input_str) == pytest.approx(expected)


@pytest.mark.parametrize("invalid", ["not_a_number", "foo", "1+"])
def test_parse_float_expr_invalid(invalid: str) -> None:
    """_parse_float_expr raises ValueError for invalid input."""
    with pytest.raises(ValueError, match="Invalid expression"):
        nhh._parse_float_expr(invalid)


@pytest.mark.parametrize(
    "input_str,expected",
    [
        ("1", 1),
        ("20", 20),
        ("2*10", 20),
    ],
)
def test_parse_int_expr_valid(input_str: str, expected: int) -> None:
    """_parse_int_expr parses valid integer strings and expressions."""
    assert nhh._parse_int_expr(input_str) == expected


@pytest.mark.parametrize("invalid", ["1.5", "x", "1+2+"])
def test_parse_int_expr_invalid(invalid: str) -> None:
    """_parse_int_expr raises ValueError for invalid input."""
    with pytest.raises(ValueError, match="Invalid expression"):
        nhh._parse_int_expr(invalid)


@pytest.mark.parametrize(
    "algorithm,T,x0,n",
    [
        ("modified", 5.0, 1.41, 20),
        ("linear", 1e-3, np.pi / 2, 4),
        ("nonhomogeneous", 5.0, 2.0, 10),
    ],
)
def test_batch_filename_stem_safe_and_formatted(
    algorithm: str, T: float, x0: float, n: int
) -> None:
    """_batch_filename_stem returns a safe string with no path separators."""
    stem = nhh._batch_filename_stem(algorithm, T, x0, n)
    assert "/" not in stem
    assert "\\" not in stem
    assert algorithm in stem
    assert str(n) in stem


def test_run_single_reconstruction_modified(tmp_path: Path, x_grid: np.ndarray) -> None:
    """_run_single_reconstruction produces a PNG for algorithm=modified."""
    nhh._run_single_reconstruction(
        "modified", T=1.0, x0=np.pi / 2, n=4, output_dir=tmp_path, x_grid=x_grid
    )
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 1
    assert pngs[0].stat().st_size > 0


def test_run_single_reconstruction_linear(tmp_path: Path, x_grid: np.ndarray) -> None:
    """_run_single_reconstruction produces a PNG for algorithm=linear."""
    nhh._run_single_reconstruction(
        "linear", T=1e-3, x0=np.pi / 2, n=4, output_dir=tmp_path, x_grid=x_grid
    )
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 1
    assert pngs[0].stat().st_size > 0


def test_cli_help() -> None:
    """Running nhh_re.py --help exits 0 and prints 'reconstruction'."""
    result = subprocess.run(
        [sys.executable, "nhh_re.py", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0
    assert "reconstruction" in result.stdout.lower()
