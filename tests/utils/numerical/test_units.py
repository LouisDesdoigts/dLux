"""Shared unit metadata keeps conversion owned by each numerical calculation."""

import equinox as eqx
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux.utils as dlu


@pytest.mark.parametrize("unit, factor", [(None, 1.0), ("nm", 1e-9), ("mm", 1e-3)])
def test_optional_unit_factor(unit, factor):
    assert dlu.unit_factor(unit) == factor


def test_unitful_sampling_and_expansion_keep_separate_units():
    axis = dlu.Axis(n=3, d=2.0, unit="mm")
    expansion = dlu.Polynomial(0, coeffs=[3.0], grid=axis, unit="nm")
    assert isinstance(axis, dlu.Unitful)
    assert isinstance(expansion, dlu.Unitful)
    assert np.allclose(axis.coordinates, np.array([[-2e-3, 0.0, 2e-3]]))
    evaluate = eqx.filter_jit(lambda value: value.resolve())
    output = evaluate(expansion)
    assert np.allclose(output / 1e-9, 3.0)
    assert np.allclose(evaluate(expansion.set(unit="mm")) / 1e-3, 3.0)
    assert expansion.coeffs[0] == 3.0
    assert expansion.grid.d[0] == 2.0


@pytest.mark.parametrize("curve", [dlu.Spectrum, dlu.Filter])
def test_curve_unit_converts_calibration_not_values(curve):
    sampling = dlu.Axis([500.0, 600.0, 700.0], unit="nm")
    response = curve([1.0, 3.0, 5.0], bandpass=sampling)
    assert isinstance(response.bandpass, dlu.Unitful)
    output = eqx.filter_jit(lambda value: value.sample(np.array([550e-9])))(response)
    assert np.allclose(output, 2.0, atol=1e-5)
    assert np.array_equal(response.value, np.array([1.0, 3.0, 5.0]))


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dlu.BandPass(1.0, unit="rad"),
        lambda: dlu.Filter(1.0, wavelengths=[1.0], unit="rad"),
        lambda: dlu.Spectrum(bandpass=dlu.Axis(n=1, c=1.0, unit="rad")),
        lambda: dlu.Blackbody(5000, bandpass=dlu.Axis(n=1, c=1.0, unit="rad")),
    ],
)
def test_wavelength_owners_reject_angular_units(constructor):
    with pytest.raises(ValueError, match="dimension"):
        constructor()


def test_unitful_constructor_validation_requires_unit_names():
    with pytest.raises(TypeError, match="unit must be a string or None"):
        dlu.Basis(np.ones((1, 2)), unit=zdx.Unit(x=10, unit="nm"))


def test_public_unit_helpers_use_si_and_preserve_shapes():
    assert dlu.unit_dimension(zdx.Unit(x=1, unit="nm")) == "length"
    assert dlu.unit_dimension("1/mm") == "spatial_frequency"
    assert np.isclose(dlu.unit_factor_to_rad("deg"), np.pi / 180)
    values = np.array([1.0, 2.0])
    assert np.allclose(
        dlu.convert(values, "mm", "um", scale=10), np.array([100.0, 200.0])
    )
    assert dlu.Unitful.validate_unit(None) is None
    assert dlu.Unitful.validate_unit("nm") == "nm"


def test_unsupported_optical_dimension_tail():
    """Coverage tail: the optical helpers exclude area dimensions."""
    with pytest.raises(ValueError, match="Unsupported"):
        dlu.unit_dimension("m^2")
