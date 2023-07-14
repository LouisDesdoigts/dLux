import jax.numpy as np
from jax import config
import pytest
from dLux.utils import (
    coordinates,
    helpers,
    interpolation,
    optics,
    units,
)

config.update("jax_debug_nans", True)


"""
coordinates.py
"""


def test_polar_to_cart():
    """Tests the polar_to_cart function."""
    assert (coordinates.polar_to_cart([1, 0]) == np.array([1, 0])).all()


def test_cart_to_polar():
    """Tests the cart_to_polar function."""
    assert (coordinates.cart_to_polar([1, 0]) == np.array([1, 0])).all()


def test_pixel_coordinates():
    """Tests the pixel_coordinates function."""

    # 1d tests
    coordinates.pixel_coordinates(16, 1, 0.0)

    with pytest.raises(TypeError):
        coordinates.pixel_coordinates(16, 1, "Not an offset")

    with pytest.raises(TypeError):
        coordinates.pixel_coordinates(16, "not a pixel scale")

    coordinates.pixel_coordinates((16, 16), (1, 1), (0, 0))

    # 2d tests
    coordinates.pixel_coordinates((16, 16), (1, 1), (0, 0))
    coordinates.pixel_coordinates((16, 16), (1, 1), (0, 0), polar=True)

    with pytest.raises(TypeError):
        coordinates.pixel_coordinates((16, 16), (1, 1), "Not an offset")

    with pytest.raises(TypeError):
        coordinates.pixel_coordinates((16, 16), "not a pixel scale", (0, 0))

    # Others
    coordinates.pixel_coordinates(
        (16, 16), (1, 1), (0, 0), indexing="ij", polar=True
    )
    with pytest.raises(ValueError):
        coordinates.pixel_coordinates(16, 1, 0.0, indexing="ab")
    with pytest.raises(ValueError):
        coordinates.pixel_coordinates(16, 1, 0.0, polar=True)


"""
helpers.py
"""


def test_list_to_dict():
    """Tests the list_to_dict function."""
    inputs = [1, 2, (3, "name")]
    helpers.list_to_dictionary(inputs, ordered=True)

    with pytest.raises(ValueError):
        inputs = [1, 2, (3, "name with spaces")]
        helpers.list_to_dictionary(inputs, ordered=True)


"""
optics.py
"""


def test_phase_to_opd():
    """Tests the phase_to_opd function."""
    optics.phase_to_opd(1, 1e-6)


def test_opd_to_phase():
    """Tests the opd_to_phase function."""
    optics.opd_to_phase(1, 1e-6)


def test_get_fringe_size():
    """Tests the get_fringe_size function."""
    optics.get_fringe_size(1e-6, 1)


def test_get_pixels_per_fringe():
    """Tests the get_pixels_per_fringe function."""
    optics.get_pixels_per_fringe(1e-6, 1, 1)
    optics.get_pixels_per_fringe(1e-6, 1, 1, 1)


def test_get_pixel_scale():
    """Tests the get_pixel_scale function."""
    optics.get_pixel_scale(1e-6, 1, 1)
    optics.get_pixel_scale(1e-6, 1, 1, 1)


def test_get_airy_pixel_scale():
    """Tests the get_airy_pixel_scale function."""
    optics.get_airy_pixel_scale(1e-6, 1, 1)
    optics.get_airy_pixel_scale(1e-6, 1, 1, 1)


"""
units.py
"""


def test_rad_to_deg():
    """Tests the rad_to_deg function."""
    units.rad_to_deg(1)


def test_rad_to_arcmin():
    """Tests the rad_to_arcmin function."""
    units.rad_to_arcmin(1)


def test_rad_to_arcsec():
    """Tests the rad_to_arcsec function."""
    units.rad_to_arcsec(1)


def test_deg_to_rad():
    """Tests the deg_to_rad function."""
    units.deg_to_rad(1)


def test_deg_to_arcsec():
    """Tests the deg_to_arcsec function."""
    units.deg_to_arcsec(1)


def test_deg_to_arcmin():
    """Tests the deg_to_arcmin function."""
    units.deg_to_arcmin(1)


def test_arcmin_to_rad():
    """Tests the arcmin_to_rad function."""
    units.arcmin_to_rad(1)


def test_arcmin_to_deg():
    """Tests the arcmin_to_deg function."""
    units.arcmin_to_deg(1)


def test_arcmin_to_arcsec():
    """Tests the arcmin_to_arcsec function."""
    units.arcmin_to_arcsec(1)


def test_arcsec_to_rad():
    """Tests the arcsec_to_rad function."""
    units.arcsec_to_rad(1)


def test_arcsec_to_deg():
    """Tests the arcsec_to_deg function."""
    units.arcsec_to_deg(1)


def test_arcsec_to_arcmin():
    """Tests the arcsec_to_arcmin function."""
    units.arcsec_to_arcmin(1)


"""
interpolation.py
"""


def test_scale_array():
    """Tests the scale_array function."""
    interpolation.scale_array(np.ones((16, 16)), 8, 1)


def test_downsample():
    """Tests the downsample function."""
    interpolation.downsample(np.ones((16, 16)), 2)
    interpolation.downsample(np.ones((16, 16)), 2, "sum")
    with pytest.raises(ValueError):
        interpolation.downsample(np.ones((16, 16)), 2, "other")
