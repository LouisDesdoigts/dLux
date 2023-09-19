import pytest
import jax.numpy as np
from dLux.utils.units import (
    rad2arcsec,
    rad2deg,
    rad2arcmin,
    deg2rad,
    deg2arcmin,
    deg2arcsec,
    arcmin2rad,
    arcmin2deg,
    arcmin2arcsec,
    arcsec2rad,
    arcsec2deg,
    arcsec2arcmin,
)


@pytest.fixture
def values():
    return np.array([0.0, np.pi / 4, np.pi / 2])


def test_rad2arcsec(values):
    result = rad2arcsec(values)
    expected = values * 3600 * 180 / np.pi
    assert np.allclose(result, expected)


def test_rad2deg(values):
    result = rad2deg(values)
    expected = values * 180 / np.pi
    assert np.allclose(result, expected)


def test_rad2arcmin(values):
    result = rad2arcmin(values)
    expected = values * 60 * 180 / np.pi
    assert np.allclose(result, expected)


def test_deg2rad(values):
    result = deg2rad(values)
    expected = values * np.pi / 180
    assert np.allclose(result, expected)


def test_deg2arcmin(values):
    result = deg2arcmin(values)
    expected = values * 60
    assert np.allclose(result, expected)


def test_deg2arcsec(values):
    result = deg2arcsec(values)
    expected = values * 3600
    assert np.allclose(result, expected)


def test_arcmin2rad(values):
    result = arcmin2rad(values)
    expected = values * np.pi / (60 * 180)
    assert np.allclose(result, expected)


def test_arcmin2deg(values):
    result = arcmin2deg(values)
    expected = values / 60
    assert np.allclose(result, expected)


def test_arcmin2arcsec(values):
    result = arcmin2arcsec(values)
    expected = values * 60
    assert np.allclose(result, expected)


def test_arcsec2rad(values):
    result = arcsec2rad(values)
    expected = values * np.pi / (3600 * 180)
    assert np.allclose(result, expected)


def test_arcsec2deg(values):
    result = arcsec2deg(values)
    expected = values / 3600
    assert np.allclose(result, expected)


def test_arcsec2arcmin(values):
    result = arcsec2arcmin(values)
    expected = values / 60
    assert np.allclose(result, expected)
