import pytest
from jax import numpy as np, config
from jax import random

config.update("jax_debug_nans", True)

from dLux.utils import interpolation as interpolation_utils
from dLux.utils import pixel_coords


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def array():
    key = random.PRNGKey(0)
    return random.normal(key, (10, 10))


@pytest.fixture
def knot_coords():
    return pixel_coords(10, 1)


@pytest.fixture
def sample_coords():
    return pixel_coords(20, 1)


@pytest.fixture
def npixels():
    return 20


@pytest.fixture
def ratio():
    return 2.0


@pytest.fixture
def angle():
    return np.pi / 4


@pytest.fixture
def method():
    return "linear"


@pytest.fixture
def fill():
    return 0.0


# ============================================================================
# Tests for interp
# ============================================================================
class TestInterp:
    """Tests for generic 2D interpolation."""

    def test_output_shape(self, array, knot_coords, sample_coords, method, fill):
        """Interpolating onto a 20x20 grid returns a 20x20 array."""
        result = interpolation_utils.interp(
            array, knot_coords, sample_coords, method, fill
        )
        assert result.shape == (20, 20)


# ============================================================================
# Tests for scale
# ============================================================================
class TestScale:
    """Tests for array resampling by a scale ratio."""

    def test_output_shape(self, array, npixels, ratio, method):
        """Scaling returns an array with the requested output size."""
        result = interpolation_utils.scale(array, npixels, ratio, method)
        assert result.shape == (npixels, npixels)


# ============================================================================
# Tests for rotate
# ============================================================================
class TestRotate:
    """Tests for interpolation-based array rotation."""

    def test_output_shape(self, array, angle, method):
        """Rotation preserves array shape."""
        result = interpolation_utils.rotate(array, angle, method)
        assert result.shape == (10, 10)
