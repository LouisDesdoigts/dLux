import pytest
from jax import numpy as np, config
from jax import random

config.update("jax_debug_nans", True)

from dLux.utils import math as math_utils


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def n():
    return 5.0


@pytest.fixture
def m():
    return 3


@pytest.fixture
def basis():
    key = random.PRNGKey(0)
    return random.normal(key, (5, 10, 10))


@pytest.fixture
def coefficients():
    key = random.PRNGKey(1)
    return random.normal(key, (5,))


@pytest.fixture
def a():
    key = random.PRNGKey(2)
    return random.normal(key, (10, 10))


@pytest.fixture
def b():
    key = random.PRNGKey(3)
    return random.normal(key, (10, 10))


@pytest.fixture
def fill():
    return np.inf


# ============================================================================
# Tests for factorial
# ============================================================================
class TestFactorial:
    """Tests for factorial evaluation."""

    def test_known_value(self, n):
        """factorial(5) evaluates to 120."""
        result = math_utils.factorial(n)
        assert np.allclose(result, 120.0)


# ============================================================================
# Tests for triangular_number
# ============================================================================
class TestTriangularNumber:
    """Tests for triangular number evaluation."""

    def test_formula(self, m):
        """Triangular numbers follow n(n+1)/2."""
        result = math_utils.triangular_number(m)
        assert result == m * (m + 1) / 2


# ============================================================================
# Tests for eval_basis
# ============================================================================
class TestEvalBasis:
    """Tests for linear basis evaluation."""

    def test_output_shape(self, basis, coefficients):
        """Evaluating a basis returns the spatial array shape."""
        result = math_utils.eval_basis(basis, coefficients)
        assert result.shape == (10, 10)

    def test_arbitrary_coefficient_and_output_dimensions(self):
        coefficients = np.arange(6.0).reshape(2, 3)
        basis = np.arange(2 * 3 * 4 * 5 * 6.0).reshape(2, 3, 4, 5, 6)

        result = math_utils.eval_basis(basis, coefficients)

        assert result.shape == (4, 5, 6)
        assert np.allclose(result, np.tensordot(basis, coefficients, ((0, 1), (0, 1))))

    def test_invalid_leading_dimensions(self):
        with pytest.raises(ValueError, match="leading basis dimensions"):
            math_utils.eval_basis(np.ones((2, 3, 4)), np.ones((3, 2)))


# ============================================================================
# Tests for solve_basis
# ============================================================================
class TestSolveBasis:
    """Tests for solving basis coefficients using least squares."""

    def test_recovers_coefficients(self, basis, coefficients):
        """Projection recovers coefficients used to evaluate a full-rank basis."""
        array = math_utils.eval_basis(basis, coefficients)
        result = math_utils.solve_basis(array, basis)

        assert np.allclose(result, coefficients, rtol=1e-5, atol=1e-5)

    def test_multidimensional_coefficients(self):
        """Projection restores multidimensional coefficient shapes."""
        key = random.PRNGKey(4)
        basis = random.normal(key, (2, 3, 4, 4))
        coefficients = np.arange(6.0).reshape((2, 3))
        array = math_utils.eval_basis(basis, coefficients)

        result = math_utils.solve_basis(array, basis)

        assert result.shape == coefficients.shape
        assert np.allclose(result, coefficients, rtol=1e-5, atol=1e-5)

    def test_least_squares_projection(self):
        """Projection returns the least-squares solution for an inexact instance."""
        basis = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        array = np.array([1.0, 2.0, 4.0])

        result = math_utils.solve_basis(array, basis)
        expected = np.linalg.lstsq(basis.T, array, rcond=None)[0]

        assert np.allclose(result, expected)

    def test_invalid_output_shape_raises(self):
        """Solving rejects basis output dimensions that do not match the array."""
        basis = np.ones((2, 2, 6))
        array = np.ones((3, 4))

        with pytest.raises(ValueError, match="trailing dimensions"):
            math_utils.solve_basis(array, basis)


# ============================================================================
# Tests for nandiv
# ============================================================================
class TestNaNDiv:
    """Tests for safe division with fill values."""

    def test_output_shape(self, a, b, fill):
        """Safe division preserves the input array shape."""
        result = math_utils.nandiv(a, b, fill)
        assert result.shape == (10, 10)


# ============================================================================
# Tests for gaussian
# ============================================================================
class TestGaussian:
    """Tests for separable Gaussian kernel generation."""

    def test_1d(self):
        """Scalar npixels returns a normalized 1D kernel."""
        result = math_utils.gaussian(npixels=16)
        assert result.shape == (16,)
        assert np.allclose(result.sum(), 1.0)

    def test_2d(self):
        """Tuple npixels returns a normalized 2D kernel."""
        result = math_utils.gaussian(npixels=(16, 16))
        assert result.shape == (16, 16)
        assert np.allclose(result.sum(), 1.0)

    def test_scalar_expansion(self):
        """Scalar npixels expands when mean implies a higher dimensionality."""
        result = math_utils.gaussian(mean=(0.0, 0.0), npixels=16)
        assert result.shape == (16, 16)
        assert np.allclose(result.sum(), 1.0)


# ============================================================================
# Tests for mv_gaussian
# ============================================================================
class TestMVGaussian:
    """Tests for multivariate Gaussian placeholder behavior."""

    def test_not_implemented(self):
        """mv_gaussian currently raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            math_utils.mv_gaussian(np.array([0.0, 0.0]), np.eye(2))
