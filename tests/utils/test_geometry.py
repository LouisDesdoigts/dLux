import pytest
from jax import numpy as np, config
from jax import vmap

config.update("jax_debug_nans", True)

from dLux import utils
from dLux.utils import geometry as geometry_utils


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def coords():
    return utils.pixel_coords(10, 2)


@pytest.fixture
def distances():
    return np.array([-1, 0, 1, 2, 3])


@pytest.fixture
def clip_dist():
    return 0.5


@pytest.fixture
def radius():
    return 1.0


@pytest.fixture
def width():
    return 2.0


@pytest.fixture
def height():
    return 3.0


@pytest.fixture
def nsides():
    return 6


@pytest.fixture
def angles():
    return [0, 60, 120, 180, 240, 300]


# ============================================================================
# Tests for combine
# ============================================================================
class TestCombine:
    """Tests for oversampled geometry combination."""

    @pytest.mark.parametrize("oversample", [1, 2])
    def test_output_shape(self, oversample):
        """Combining arrays respects the oversample downsampling factor."""
        n = 6
        arrays = [np.ones((n, n)) for _ in range(3)]
        result = geometry_utils.combine(arrays, oversample)
        assert result.shape == (n // oversample, n // oversample)


# ============================================================================
# Tests for soften
# ============================================================================
class TestSoften:
    """Tests for edge softening."""

    @pytest.mark.parametrize("invert", [True, False])
    def test_output_shape(self, distances, clip_dist, invert):
        """Softening preserves the input distance array shape."""
        result = geometry_utils.soften(distances, clip_dist, invert)
        assert result.shape == distances.shape


# ============================================================================
# Tests for primitive aperture generators
# ============================================================================
class TestCircle:
    """Tests for circular aperture generation."""

    @pytest.mark.parametrize("invert", [True, False])
    def test_output_shape(self, coords, radius, invert):
        """Circle returns a 2D mask with the same spatial shape as the input grid."""
        result = geometry_utils.circle(coords, radius, invert)
        assert result.shape == (10, 10)

    def test_distance_relation(self, coords, radius):
        """Circle support matches the sign of the radial distance field."""
        result = geometry_utils.circle(coords, radius)
        distances = utils.cart2polar(coords)[0] - radius
        assert np.allclose(result, distances < 0)


class TestSquare:
    """Tests for square aperture generation."""

    @pytest.mark.parametrize("invert", [True, False])
    def test_output_shape(self, coords, width, invert):
        """Square returns a 2D mask with the same spatial shape as the input grid."""
        result = geometry_utils.square(coords, width, invert)
        assert result.shape == (10, 10)

    def test_distance_relation(self, coords, width):
        """Square support matches the sign of the square distance field."""
        result = geometry_utils.square(coords, width)
        distances = np.max(np.abs(coords), axis=0) - width / 2
        assert np.allclose(result, distances < 0)


class TestRectangle:
    """Tests for rectangular aperture generation."""

    @pytest.mark.parametrize("invert", [True, False])
    def test_output_shape(self, coords, width, height, invert):
        """Rectangle returns a 2D mask with the same spatial shape as the input grid."""
        result = geometry_utils.rectangle(coords, width, height, invert)
        assert result.shape == (10, 10)

    def test_distance_relation(self, coords, width, height):
        """Rectangle support matches the sign of the rectangular distance field."""
        result = geometry_utils.rectangle(coords, width, height)
        dist_from_vert = np.abs(coords[0]) - width / 2
        dist_from_horz = np.abs(coords[1]) - height / 2
        distances = np.maximum(dist_from_vert, dist_from_horz)
        assert np.allclose(result, distances < 0)


class TestRegPolygon:
    """Tests for regular polygon aperture generation."""

    @pytest.mark.parametrize("invert", [True, False])
    def test_output_shape(self, coords, radius, nsides, invert):
        """
        Regular polygons return a 2D mask with the same spatial shape as the input grid.
        """
        result = geometry_utils.reg_polygon(coords, radius, nsides, invert)
        assert result.shape == (10, 10)

    def test_distance_relation(self, coords, nsides, radius):
        """Polygon support matches the sign of the polygon edge distance field."""
        result = geometry_utils.reg_polygon(coords, radius, nsides)
        slopes, vertices = geometry_utils.reg_polygon_edges(nsides, radius)
        distances = vmap(geometry_utils.line_distance, (None, 0, 0))(
            coords, slopes, vertices
        ).max(0)
        assert np.allclose(result, distances < 0)


class TestSpider:
    """Tests for spider vane generation."""

    def test_output_shape(self, coords, width, angles):
        """Spider returns a 2D mask with the same spatial shape as the input grid."""
        result = geometry_utils.spider(coords, width, angles)
        assert result.shape == (10, 10)


# ============================================================================
# Tests for softened primitive generators
# ============================================================================
class TestSoftCircle:
    """Tests for softened circular aperture generation."""

    def test_output_shape(self, coords, radius, clip_dist):
        """Soft circle returns a 2D array with the expected spatial shape."""
        result = geometry_utils.soft_circle(coords, radius, clip_dist)
        assert result.shape == (10, 10)


class TestSoftSquare:
    """Tests for softened square aperture generation."""

    def test_output_shape(self, coords, width, clip_dist):
        """Soft square returns a 2D array with the expected spatial shape."""
        result = geometry_utils.soft_square(coords, width, clip_dist)
        assert result.shape == (10, 10)


class TestSoftRectangle:
    """Tests for softened rectangular aperture generation."""

    def test_output_shape(self, coords, width, height, clip_dist):
        """Soft rectangle returns a 2D array with the expected spatial shape."""
        result = geometry_utils.soft_rectangle(coords, width, height, clip_dist)
        assert result.shape == (10, 10)


class TestSoftRegPolygon:
    """Tests for softened regular polygon generation."""

    def test_output_shape(self, coords, radius, nsides, clip_dist):
        """Soft regular polygon returns a 2D array with the expected spatial shape."""
        result = geometry_utils.soft_reg_polygon(coords, radius, nsides, clip_dist)
        assert result.shape == (10, 10)


class TestSoftSpider:
    """Tests for softened spider vane generation."""

    def test_output_shape(self, coords, width, angles, clip_dist):
        """Soft spider returns a 2D array with the expected spatial shape."""
        result = geometry_utils.soft_spider(coords, width, angles, clip_dist)
        assert result.shape == (10, 10)


# ============================================================================
# Tests for polygon helpers
# ============================================================================
class TestRegPolygonEdges:
    """Tests for regular polygon edge helper generation."""

    def test_output_shapes(self, nsides, radius):
        """Edge helper arrays match the number of polygon sides."""
        slopes, vertices = geometry_utils.reg_polygon_edges(nsides, radius)
        assert slopes.shape == (nsides,)
        assert vertices.shape == (nsides, 2)
