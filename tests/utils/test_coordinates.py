import pytest
from jax import numpy as np, config

config.update("jax_debug_nans", True)

from dLux.utils import coordinates as coordinates_utils

RTOL = 1e-5
ATOL = 1e-7


# ============================================================================
# Tests for translate_coords
# ============================================================================
class TestTranslateCoords:
    """Tests for coordinate translation."""

    def test_formula(self):
        """Translation subtracts the supplied center from each coordinate."""
        coords = np.array([[[0.0, 0.5, 1.0]], [[0.0, 0.5, 1.0]]])
        centre = np.array([0.5, 0.5])
        expected = np.array([[[-0.5, 0.0, 0.5]], [[-0.5, 0.0, 0.5]]])
        result = coordinates_utils.translate_coords(coords, centre)
        assert np.allclose(result, expected)


# ============================================================================
# Tests for compress_coords
# ============================================================================
class TestCompressCoords:
    """Tests for anisotropic coordinate compression."""

    def test_formula(self):
        """Compression scales each axis independently."""
        coords = np.array([[[0.0, 0.5, 1.0]], [[0.0, 0.5, 1.0]]])
        compress = np.array([0.5, 1.0])
        expected = np.array([[[0.0, 0.25, 0.5]], [[0.0, 0.5, 1.0]]])
        result = coordinates_utils.compress_coords(coords, compress)
        assert np.allclose(result, expected)


# ============================================================================
# Tests for shear_coords
# ============================================================================
class TestShearCoords:
    """Tests for coordinate shearing."""

    def test_formula(self):
        """Shearing a square coordinate grid gives the expected affine transform."""
        coords = np.array(
            [
                [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]],
                [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]],
            ]
        )
        shear = np.array([0.0, 0.5])
        expected = np.array(
            [
                [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]],
                [[0.0, 0.5, 1.0], [0.25, 0.75, 1.25], [0.5, 1.0, 1.5]],
            ]
        )
        result = coordinates_utils.shear_coords(coords, shear)
        assert np.allclose(result, expected)


# ============================================================================
# Tests for rotate_coords
# ============================================================================
class TestRotateCoords:
    """Tests for coordinate rotation."""

    def test_formula(self):
        """Rotation by π flips both axes."""
        coords = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        rotation = np.pi
        expected = np.array([[0.0, -0.5, -1.0], [0.0, -0.5, -1.0]])
        result = coordinates_utils.rotate_coords(coords, rotation)
        assert np.allclose(result, expected)


# ============================================================================
# Tests for gen_powers
# ============================================================================
class TestGenPowers:
    """Tests for polynomial power generation."""

    def test_output(self):
        """Generated power ordering matches the expected triangular pattern."""
        expected = np.array(
            [[0.0, 1.0, 0.0, 2.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0, 1.0, 2.0]]
        )
        powers = coordinates_utils.gen_powers(3)
        assert np.allclose(powers, expected)


# ============================================================================
# Tests for distort_coords
# ============================================================================
class TestDistortCoords:
    """Tests for polynomial coordinate distortion."""

    def test_output_shape(self):
        """Distortion preserves the coordinate array shape."""
        coords = np.array([[[0.0, 0.5, 1.0]], [[0.0, 0.5, 1.0]]])
        powers = coordinates_utils.gen_powers(3)
        distortion = coordinates_utils.distort_coords(coords, np.ones(6), powers)
        assert distortion.shape == coords.shape


# ============================================================================
# Tests for cart2polar
# ============================================================================
class TestCart2Polar:
    """Tests for Cartesian-to-polar conversion."""

    @pytest.mark.parametrize(
        "coordinates, expected",
        [
            (np.array([1, 0]), np.array([1, 0])),
            (np.array([0, 1]), np.array([1, np.pi / 2])),
            (np.array([-1, 0]), np.array([1, np.pi])),
            (np.array([0, -1]), np.array([1, -np.pi / 2])),
        ],
    )
    def test_known_points(self, coordinates, expected):
        """Axis-aligned points convert to the expected polar coordinates."""
        actual = coordinates_utils.cart2polar(coordinates)
        assert np.allclose(actual, expected, rtol=RTOL, atol=ATOL)


# ============================================================================
# Tests for polar2cart
# ============================================================================
class TestPolar2Cart:
    """Tests for polar-to-Cartesian conversion."""

    @pytest.mark.parametrize(
        "coordinates, expected",
        [
            (np.array([1, 0]), np.array([1, 0])),
            (np.array([1, np.pi / 2]), np.array([0, 1])),
            (np.array([1, np.pi]), np.array([-1, 0])),
            (np.array([1, -np.pi / 2]), np.array([0, -1])),
        ],
    )
    def test_known_points(self, coordinates, expected):
        """Axis-aligned polar points convert to the expected Cartesian coordinates."""
        actual = coordinates_utils.polar2cart(coordinates)
        assert np.allclose(actual, expected, rtol=RTOL, atol=ATOL)


# ============================================================================
# Tests for pixel_coords
# ============================================================================
class TestPixelCoords:
    """Tests for 2D pixel-center coordinate generation."""

    @pytest.mark.parametrize(
        "npixels, diameter, polar, expected_shape",
        [
            (10, 1.0, False, (2, 10, 10)),
            (10, 1.0, True, (2, 10, 10)),
            (20, 2.0, False, (2, 20, 20)),
            (20, 2.0, True, (2, 20, 20)),
        ],
    )
    def test_output_shape(self, npixels, diameter, polar, expected_shape):
        """Pixel coordinates return the expected shape across argument combinations."""
        actual = coordinates_utils.pixel_coords(npixels, diameter=diameter, polar=polar)
        assert actual.shape == expected_shape

    def test_no_scale_raises(self):
        """Exactly one scale specification is required."""
        with pytest.raises(ValueError, match="Exactly one"):
            coordinates_utils.pixel_coords(10)

    def test_radius(self):
        """Radius-based pixel scale generation returns a 2D coordinate grid."""
        result = coordinates_utils.pixel_coords(10, radius=0.5)
        assert result.shape == (2, 10, 10)

    def test_pixel_scale(self):
        """Explicit pixel scale generation returns a 2D coordinate grid."""
        result = coordinates_utils.pixel_coords(10, pixel_scale=0.1)
        assert result.shape == (2, 10, 10)

    def test_fft_style_even(self):
        """FFT-style centering works for even pixel counts."""
        result = coordinates_utils.pixel_coords(10, diameter=1.0, fft_style=True)
        assert result.shape == (2, 10, 10)


# ============================================================================
# Tests for nd_coords
# ============================================================================
class TestNDCoords:
    """Tests for n-dimensional pixel-center coordinate generation."""

    def test_1d_xy_indexing(self):
        """Scalar input returns a squeezed 1D coordinate array."""
        actual = coordinates_utils.nd_coords(10, 1.0, 0.0, "xy")
        assert actual.shape == (10,)

    def test_2d_ij_indexing(self):
        """Tuple input returns a 2D coordinate grid with ij indexing."""
        actual = coordinates_utils.nd_coords((10, 20), (1.0, 2.0), (0.0, 1.0), "ij")
        assert actual.shape == (2, 10, 20)

    def test_3d(self):
        """Three-dimensional inputs return a 3D coordinate grid."""
        actual = coordinates_utils.nd_coords(
            (10, 20, 30),
            (1.0, 2.0, 3.0),
            (0.0, 1.0, 2.0),
            "ij",
        )
        assert actual.shape == (3, 10, 20, 30)

    def test_invalid_indexing_raises(self):
        """Unsupported indexing conventions raise ValueError."""
        with pytest.raises(ValueError):
            coordinates_utils.nd_coords((10, 20, 30), (1.0, 2.0, 3.0), (0, 1, 2), "xi")

    def test_scalar_npixels_expand(self):
        """Scalar npixels expands when other parameters imply higher dimensionality."""
        result = coordinates_utils.nd_coords(10, pixel_scales=(1.0, 2.0))
        assert result.shape == (2, 10, 10)
