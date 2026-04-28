import pytest
from jax import numpy as np, config

config.update("jax_debug_nans", True)

from dLux.utils import array_ops as array_ops_utils


# ============================================================================
# Tests for pad_to
# ============================================================================
class TestPadTo:
    """Tests for symmetric array padding."""

    @pytest.mark.parametrize(
        "array, npixels",
        [
            (np.ones((10, 10)), 8),
            (np.ones((11, 11)), 9),
            (np.ones((10, 10)), 11),
            (np.ones((11, 11)), 12),
        ],
    )
    def test_invalid(self, array, npixels):
        """Parity mismatches raise ValueError."""
        with pytest.raises(ValueError):
            array_ops_utils.pad_to(array, npixels)

    @pytest.mark.parametrize(
        "array, npixels",
        [(np.ones((10, 10)), 12), (np.ones((11, 11)), 13)],
    )
    def test_output_shape(self, array, npixels):
        """Padding returns an array with the requested square shape."""
        actual = array_ops_utils.pad_to(array, npixels)
        assert actual.shape == (npixels, npixels)


# ============================================================================
# Tests for crop_to
# ============================================================================
class TestCropTo:
    """Tests for central array cropping."""

    @pytest.mark.parametrize(
        "array, npixels",
        [
            (np.ones((10, 10)), 12),
            (np.ones((11, 11)), 13),
            (np.ones((10, 10)), 9),
            (np.ones((11, 11)), 10),
        ],
    )
    def test_invalid(self, array, npixels):
        """Invalid crop sizes raise ValueError."""
        with pytest.raises(ValueError):
            array_ops_utils.crop_to(array, npixels)

    @pytest.mark.parametrize(
        "array, npixels",
        [(np.ones((10, 10)), 8), (np.ones((11, 11)), 9)],
    )
    def test_output_shape(self, array, npixels):
        """Cropping returns an array with the requested square shape."""
        actual = array_ops_utils.crop_to(array, npixels)
        assert actual.shape == (npixels, npixels)


# ============================================================================
# Tests for resize
# ============================================================================
class TestResize:
    """Tests for resize wrapper behavior."""

    @pytest.mark.parametrize(
        "array, shape",
        [
            (np.ones((10, 10)), 12),
            (np.ones((10, 10)), 8),
            (np.ones((11, 11)), 13),
            (np.ones((11, 11)), 9),
            (np.ones((10, 10)), 10),
        ],
    )
    def test_output_shape(self, array, shape):
        """Resize returns the requested square output shape."""
        actual = array_ops_utils.resize(array, shape)
        assert actual.shape == (shape, shape)


# ============================================================================
# Tests for downsample
# ============================================================================
class TestDownsample:
    """Tests for block downsampling."""

    @pytest.mark.parametrize(
        "array, n, mean",
        [
            (np.ones((10, 9)), 2, True),
            (np.ones((10, 9)), 2, False),
            (np.ones((10, 10)), 3, True),
            (np.ones((10, 10)), 3, False),
        ],
    )
    def test_invalid(self, array, n, mean):
        """Input shapes not divisible by the factor raise ValueError."""
        with pytest.raises(ValueError):
            array_ops_utils.downsample(array, n, mean)

    @pytest.mark.parametrize(
        "array, n, mean",
        [
            (np.ones((10, 10)), 2, True),
            (np.ones((9, 9)), 3, True),
            (np.ones((10, 10)), 5, True),
            (np.ones((15, 15)), 3, True),
            (np.ones((10, 10)), 2, False),
            (np.ones((9, 9)), 3, False),
            (np.ones((10, 10)), 5, False),
            (np.ones((15, 15)), 3, False),
        ],
    )
    def test_output_shape(self, array, n, mean):
        """Downsampling reduces each axis by the integer factor."""
        actual = array_ops_utils.downsample(array, n, mean)
        assert actual.shape == (array.shape[0] // n, array.shape[1] // n)
