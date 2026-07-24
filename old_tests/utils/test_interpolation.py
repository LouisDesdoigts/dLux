import pytest
from jax import numpy as np, config
from jax import random

config.update("jax_debug_nans", True)

from dLux.utils import interpolation as interpolation_utils
from dLux.utils import pixel_coords


def complex_components(array, complex):
    if complex:
        vals = np.array([array.real, array.imag])
        return vals, lambda x: x[0] + 1j * x[1]
    vals = np.array([np.abs(array), np.angle(array)])
    return vals, lambda x: x[0] * np.exp(1j * x[1])


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def array():
    key = random.PRNGKey(0)
    return random.normal(key, (10, 10))


@pytest.fixture
def complex_array(array):
    return array + 1j * np.flip(array, 0)


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

    @pytest.mark.parametrize("complex", [True, False])
    def test_complex_output_shape(
        self, complex_array, knot_coords, sample_coords, method, fill, complex
    ):
        """Complex interpolation supports Cartesian and polar components."""
        result = interpolation_utils.interp(
            complex_array,
            knot_coords,
            sample_coords,
            method,
            fill,
            complex=complex,
        )
        assert result.shape == (20, 20)
        assert np.iscomplexobj(result)

    @pytest.mark.parametrize("complex", [True, False])
    def test_complex_components(
        self, complex_array, knot_coords, sample_coords, method, fill, complex
    ):
        """Complex interpolation matches explicit component interpolation."""
        vals, return_fn = complex_components(complex_array, complex)
        expected = return_fn(
            np.array(
                [
                    interpolation_utils.interp(
                        val, knot_coords, sample_coords, method, fill
                    )
                    for val in vals
                ]
            )
        )
        actual = interpolation_utils.interp(
            complex_array,
            knot_coords,
            sample_coords,
            method,
            fill,
            complex=complex,
        )
        assert np.allclose(actual, expected)


# ============================================================================
# Tests for scale
# ============================================================================
class TestScale:
    """Tests for array resampling by a scale ratio."""

    def test_output_shape(self, array, npixels, ratio, method):
        """Scaling returns an array with the requested output size."""
        result = interpolation_utils.scale(array, npixels, ratio, method)
        assert result.shape == (npixels, npixels)

    @pytest.mark.parametrize("complex", [True, False])
    def test_complex_output_shape(self, complex_array, npixels, ratio, method, complex):
        """Scaling supports complex arrays."""
        result = interpolation_utils.scale(
            complex_array, npixels, ratio, method, complex=complex
        )
        assert result.shape == (npixels, npixels)
        assert np.iscomplexobj(result)

    @pytest.mark.parametrize("complex", [True, False])
    def test_complex_components(self, complex_array, npixels, ratio, method, complex):
        """Complex scaling matches explicit component scaling."""
        vals, return_fn = complex_components(complex_array, complex)
        expected = return_fn(
            np.array(
                [interpolation_utils.scale(val, npixels, ratio, method) for val in vals]
            )
        )
        actual = interpolation_utils.scale(
            complex_array, npixels, ratio, method, complex=complex
        )
        assert np.allclose(actual, expected)


# ============================================================================
# Tests for rotate
# ============================================================================
class TestRotate:
    """Tests for interpolation-based array rotation."""

    def test_output_shape(self, array, angle, method):
        """Rotation preserves array shape."""
        result = interpolation_utils.rotate(array, angle, method)
        assert result.shape == (10, 10)

    @pytest.mark.parametrize("complex", [True, False])
    def test_complex_output_shape(self, complex_array, angle, method, complex):
        """Rotation supports complex arrays."""
        result = interpolation_utils.rotate(
            complex_array, angle, method, complex=complex
        )
        assert result.shape == (10, 10)
        assert np.iscomplexobj(result)

    @pytest.mark.parametrize("complex", [True, False])
    def test_complex_components(self, complex_array, angle, method, complex):
        """Complex rotation matches explicit component rotation."""
        vals, return_fn = complex_components(complex_array, complex)
        expected = return_fn(
            np.array([interpolation_utils.rotate(val, angle, method) for val in vals])
        )
        actual = interpolation_utils.rotate(
            complex_array, angle, method, complex=complex
        )
        assert np.allclose(actual, expected)
