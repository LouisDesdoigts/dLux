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

    def test_batched_output_shape(self):
        """Padding only affects the final two axes."""
        actual = array_ops_utils.pad_to(np.ones((2, 10, 10)), 12)
        assert actual.shape == (2, 12, 12)

    def test_batched_values(self):
        """Padding preserves each leading slice and fills the border."""
        array = np.arange(2 * 2 * 2).reshape(2, 2, 2)
        actual = array_ops_utils.pad_to(array, 4, fill=-1)
        assert np.all(actual[:, 1:3, 1:3] == array)
        assert np.all(actual[:, 0] == -1)
        assert np.all(actual[:, -1] == -1)
        assert np.all(actual[:, :, 0] == -1)
        assert np.all(actual[:, :, -1] == -1)


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

    def test_batched_output_shape(self):
        """Cropping only affects the final two axes."""
        actual = array_ops_utils.crop_to(np.ones((2, 10, 10)), 8)
        assert actual.shape == (2, 8, 8)

    def test_batched_values(self):
        """Cropping preserves the centred values for each leading slice."""
        array = np.arange(2 * 4 * 4).reshape(2, 4, 4)
        actual = array_ops_utils.crop_to(array, 2)
        assert np.all(actual == array[:, 1:3, 1:3])


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

    def test_complex_output_shape(self):
        """Resize preserves complex batched arrays."""
        array = np.ones((2, 10, 10)) * (1 + 1j)
        actual = array_ops_utils.resize(array, 12)
        assert actual.shape == (2, 12, 12)
        assert np.iscomplexobj(actual)

    def test_same_size_returns_input(self):
        """Resize is a no-op when the requested size is already present."""
        array = np.arange(2 * 4 * 4).reshape(2, 4, 4)
        actual = array_ops_utils.resize(array, 4)
        assert np.all(actual == array)


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

    def test_complex_batched_output_shape(self):
        """Downsampling only affects final two axes and preserves complex dtype."""
        array = np.ones((2, 10, 10)) * (1 + 1j)
        actual = array_ops_utils.downsample(array, 2)
        assert actual.shape == (2, 5, 5)
        assert np.iscomplexobj(actual)

    @pytest.mark.parametrize(
        "mean, expected",
        [
            (
                True,
                np.array(
                    [
                        [[2.5, 4.5], [10.5, 12.5]],
                        [[18.5, 20.5], [26.5, 28.5]],
                    ]
                ),
            ),
            (
                False,
                np.array(
                    [
                        [[10, 18], [42, 50]],
                        [[74, 82], [106, 114]],
                    ]
                ),
            ),
        ],
    )
    def test_batched_values(self, mean, expected):
        """Downsampling applies block reductions over the final two axes."""
        array = np.arange(2 * 4 * 4).reshape(2, 4, 4)
        actual = array_ops_utils.downsample(array, 2, mean)
        assert np.allclose(actual, expected)
