from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
from jax import vmap
from dLux import utils
from dLux.utils.geometry import (
    combine,
    soften,
    circle,
    square,
    rectangle,
    reg_polygon,
    spider,
    soft_circle,
    soft_square,
    soft_rectangle,
    soft_reg_polygon,
    soft_spider,
    reg_polygon_edges,
    line_distance,
)


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
def diam():
    return 2.0


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


@pytest.mark.parametrize("oversample", [1, 2])
def test_combine(oversample):
    n = 6
    arrays = [np.ones((n, n)) for _ in range(3)]
    result = combine(arrays, oversample)
    assert result.shape == (n // oversample, n // oversample)


@pytest.mark.parametrize("invert", [True, False])
def test_soften(distances, clip_dist, invert):
    result = soften(distances, clip_dist, invert)
    assert result.shape == distances.shape


@pytest.mark.parametrize("invert", [True, False])
def test_circle(coords, radius, invert):
    result = circle(coords, radius, invert)
    assert result.shape == (10, 10)


@pytest.mark.parametrize("invert", [True, False])
def test_square(coords, width, invert):
    result = square(coords, width, invert)
    assert result.shape == (10, 10)


@pytest.mark.parametrize("invert", [True, False])
def test_rectangle(coords, width, height, invert):
    result = rectangle(coords, width, height, invert)
    assert result.shape == (10, 10)


@pytest.mark.parametrize("invert", [True, False])
def test_reg_polygon(coords, radius, nsides, invert):
    result = reg_polygon(coords, radius, nsides, invert)
    assert result.shape == (10, 10)


def test_spider(coords, width, angles):
    result = spider(coords, width, angles)
    assert result.shape == (10, 10)


def test_soft_circle(coords, radius, clip_dist):
    result = soft_circle(coords, radius, clip_dist)
    assert result.shape == (10, 10)


def test_soft_square(coords, width, clip_dist):
    result = soft_square(coords, width, clip_dist)
    assert result.shape == (10, 10)


def test_soft_rectangle(coords, width, height, clip_dist):
    result = soft_rectangle(coords, width, height, clip_dist)
    assert result.shape == (10, 10)


def test_soft_reg_polygon(coords, radius, nsides, clip_dist):
    result = soft_reg_polygon(coords, radius, nsides, clip_dist)
    assert result.shape == (10, 10)


def test_soft_spider(coords, width, angles, clip_dist):
    result = soft_spider(coords, width, angles, clip_dist)
    assert result.shape == (10, 10)


def test_circ_distance(coords, radius):
    result = circle(coords, 2 * radius)
    distances = utils.cart2polar(coords)[0] - radius
    assert np.allclose(result, distances < 0)


def test_square_distance(coords, width):
    result = square(coords, width)
    distances = np.max(np.abs(coords), axis=0) - width / 2
    assert np.allclose(result, distances < 0)


def test_rectangle_distance(coords, width, height):
    result = rectangle(coords, width, height)
    dist_from_vert = np.abs(coords[0]) - width / 2
    dist_from_horz = np.abs(coords[1]) - height / 2
    distances = np.maximum(dist_from_vert, dist_from_horz)
    assert np.allclose(result, distances < 0)


# def test_spider_distance(coords, width, angles):
#     result = spider(coords, width, angles)
#     angles = np.array(angles)
#     coords = vmap(utils.rotate_coords, (None, 0))(coords, utils.deg2rad(angles))

#     dist_from_vert = np.abs(coords[:, 0]) - width / 2
#     dist_from_horz = coords[:, 1]
#     distances = np.max(np.array([dist_from_vert, dist_from_horz]), axis=0)
#     assert np.allclose(result, distances < 0, atol=1e-4, rtol=1e-7)


def test_reg_polygon_edges(nsides, radius):
    m, xy = reg_polygon_edges(nsides, radius)
    assert m.shape == (nsides,)
    assert xy.shape == (nsides, 2)


def test_reg_polygon_distance(coords, nsides, radius):
    result = reg_polygon(coords, radius, nsides)
    m, xy = reg_polygon_edges(nsides, radius)
    distances = vmap(line_distance, (None, 0, 0))(coords, m, xy).max(0)
    assert np.allclose(result, distances < 0)
