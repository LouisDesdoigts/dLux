"""Tests for dLux.utils.units."""

import jax.numpy as np
import pytest

import dLux.utils as dlu

from tests.helpers import assert_jittable


@pytest.mark.parametrize(
    ("function", "factor"),
    [
        (dlu.rad2deg, 180 / np.pi),
        (dlu.rad2arcmin, 60 * 180 / np.pi),
        (dlu.rad2arcsec, 3600 * 180 / np.pi),
        (dlu.deg2rad, np.pi / 180),
        (dlu.deg2arcmin, 60),
        (dlu.deg2arcsec, 3600),
        (dlu.arcmin2rad, np.pi / (60 * 180)),
        (dlu.arcmin2deg, 1 / 60),
        (dlu.arcmin2arcsec, 60),
        (dlu.arcsec2rad, np.pi / (3600 * 180)),
        (dlu.arcsec2deg, 1 / 3600),
        (dlu.arcsec2arcmin, 1 / 60),
    ],
)
def test_angular_conversions(function, factor):
    values = np.asarray((0.0, 0.5, 1.0))
    output = assert_jittable(function, values)

    assert np.allclose(output, values * factor)


@pytest.mark.parametrize(
    ("unit", "factor"),
    [("mrad", 1e-3), ("nm", 1e-9), ("angstrom", 1e-10), ("kphoton", 1e3)],
)
def test_unit_factor(unit, factor):
    assert np.isclose(dlu.unit_factor(unit), factor)


def test_milliarcsecond_alias():
    assert np.isclose(dlu.unit_factor("mas"), np.pi / (180 * 3600 * 1000))


def test_prefixed_angular_factor():
    assert np.isclose(dlu.unit_factor_to_rad("mrad"), 1e-3)


def test_convert_roundtrip():
    value = np.asarray((1.0, 2.0))
    converted = dlu.convert(value, "arcsec", "rad")
    assert np.allclose(dlu.convert(converted, "rad", "arcsec"), value)


def test_canonical_units_and_dimensions():
    assert dlu.canonical_unit("arcseconds", dimension="angle") == "arcsec"
    assert dlu.canonical_unit("mas", dimension="angle") == "mas"
    assert dlu.canonical_unit("metres", dimension="length") == "m"

    assert np.isclose(dlu.convert(1.0, "nm", "m"), 1e-9)
    with pytest.raises(ValueError, match="Cannot convert"):
        dlu.convert(1.0, "nm", "rad")
    with pytest.raises(ValueError, match="position unit must have dimension 'angle'"):
        dlu.unit_factor("nm", dimension="angle", name="position unit")


@pytest.mark.parametrize("unit", [1, "", "unknown"])
def test_invalid_unit(unit):
    with pytest.raises((TypeError, ValueError)):
        dlu.unit_factor(unit)
