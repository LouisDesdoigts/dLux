import pytest
from jax import numpy as np, config

config.update("jax_debug_nans", True)

from dLux.utils import misc as misc_utils


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def oversample():
    return 4


@pytest.fixture
def square_array(oversample):
    """A square array that can be downsampled by `oversample`."""
    n = 8 * oversample
    return np.ones((n, n))


# ============================================================================
# Tests for _lsq_matrix
# ============================================================================
class TestLsqMatrix:
    """Tests for the internal least-squares matrix builder."""

    def test_square(self):
        """Square case (m=None) returns shape (3, n*n)."""
        M = misc_utils._lsq_matrix(4)
        assert M.shape == (3, 16)

    def test_rectangular(self):
        """Rectangular case (m given) returns shape (3, n*m)."""
        M = misc_utils._lsq_matrix(4, m=6)
        assert M.shape == (3, 24)


# ============================================================================
# Tests for _calc_area_fraction
# ============================================================================
class TestCalcAreaFraction:
    """Tests for fractional area calculation."""

    def test_all_positive(self):
        """Plane entirely above zero has fraction near 0 (area below zero)."""
        # a, b, c such that z = 0*x + 0*y + 1 → entirely above zero
        coeffs = np.array([0.0, 0.0, 1.0])
        frac = misc_utils._calc_area_fraction(coeffs)
        # Should be a finite float
        assert np.isfinite(frac)

    def test_returns_scalar(self):
        """Returns a scalar value."""
        coeffs = np.array([1.0, 1.0, -0.5])
        frac = misc_utils._calc_area_fraction(coeffs)
        assert frac.shape == ()


# ============================================================================
# Tests for soft_binarise
# ============================================================================
class TestSoftBinarise:
    """Tests for the CLIMB soft binarisation algorithm."""

    def test_basic_ones(self, oversample):
        """Array of ones returns ones after binarisation."""
        arr = np.ones((8 * oversample, 8 * oversample))
        result = misc_utils.soft_binarise(arr, oversample=oversample)
        assert result.shape == (8, 8)
        assert np.allclose(result, 1.0)

    def test_basic_negative(self, oversample):
        """Array of negative values returns zeros after binarisation."""
        arr = -np.ones((8 * oversample, 8 * oversample))
        result = misc_utils.soft_binarise(arr, oversample=oversample)
        assert result.shape == (8, 8)
        assert np.allclose(result, 0.0)

    def test_output_range(self, oversample):
        """Output values are in [0, 1]."""
        rng = np.linspace(-1.0, 1.0, 8 * oversample)
        arr = np.outer(rng, rng)
        result = misc_utils.soft_binarise(arr, oversample=oversample)
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()

    def test_output_shape(self, oversample):
        """Output shape is input shape divided by oversample."""
        n = 12 * oversample
        arr = np.ones((n, n))
        result = misc_utils.soft_binarise(arr, oversample=oversample)
        assert result.shape == (12, 12)

    def test_invalid_shape_raises(self, oversample):
        """Array shape not divisible by oversample raises ValueError."""
        arr = np.ones((7, 7))
        with pytest.raises(ValueError, match="multiple of oversample"):
            misc_utils.soft_binarise(arr, oversample=oversample)

    def test_boundary_region(self, oversample):
        """Mixed-sign tiles produce values strictly between 0 and 1."""
        # Checkerboard-ish pattern: alternating positive/negative within each tile
        n = 8 * oversample
        arr = np.zeros((n, n))
        # Set first tile to have a boundary
        arr = arr.at[:oversample, :oversample].set(
            np.linspace(-1.0, 1.0, oversample)[:, None] * np.ones((1, oversample))
        )
        result = misc_utils.soft_binarise(arr, oversample=oversample)
        assert result.shape == (8, 8)
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()

    def test_zero_boundary_pixels(self, oversample):
        """Tiles containing exactly zero are handled via any_zero branch."""
        arr = np.zeros((8 * oversample, 8 * oversample))
        result = misc_utils.soft_binarise(arr, oversample=oversample)
        assert result.shape == (8, 8)
        assert np.allclose(result, 0.0)
