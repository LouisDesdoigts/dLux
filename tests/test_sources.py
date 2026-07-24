"""Tests for dLux.sources."""

import jax.numpy as np
import pytest

import dLux as dl
import dLux.utils as dlu

from tests.helpers import assert_differentiable


def test_spectrum_and_point_source_parameters():
    spectrum = dl.Spectrum([900, 1100], units={"wavelengths": "nm"})
    source = dl.PointSource(
        [900, 1100],
        position=[1, -2],
        flux=2,
        units={
            "wavelengths": "nm",
            "position": "arcsec",
            "flux": "kphoton",
        },
    )

    wavelengths, weights = spectrum.spectrum_params()
    parameters = source.params()
    assert np.allclose(wavelengths, np.array([0.9e-6, 1.1e-6]))
    assert np.allclose(weights, np.ones(2))
    assert np.allclose(parameters[2], np.array([1, -2]) * dlu.arcsec2rad(1))
    assert np.allclose(parameters[3], 2000)
    assert parameters[4] is None


def test_default_point_source():
    source = dl.PointSource([1e-6])
    _, _, position, flux, distribution = source.params()

    assert np.allclose(position, np.zeros(2))
    assert np.allclose(flux, 1)
    assert distribution is None


def test_binary_source_parameters():
    source = dl.BinarySource(
        [1e-6],
        centre=[0.1, -0.2],
        separation=0.4,
        position_angle=0.3,
        contrast=3.0,
        flux=2.0,
    )

    _, _, positions, fluxes, distribution = source.params()
    assert positions.shape == (2, 2)
    assert fluxes.shape == (2,)
    assert np.allclose(fluxes.mean(), 2)
    assert np.allclose(fluxes[0] / fluxes[1], 3)
    assert distribution is None

    assert_differentiable(
        lambda separation: source.set("separation", separation).params()[2],
        source.separation,
    )


def test_parametric_distribution():
    distribution = dl.ExplicitBasis(
        np.ones((2, 3, 3)),
        coefficients=[1.0, 2.0],
    )
    source = dl.BinarySource(
        [1e-6],
        distribution=distribution,
    )

    assert source.params()[-1].shape == (3, 3)


def test_log_flux_units():
    source = dl.PointSource(
        [1e-6],
        flux=np.log(1000),
        distribution=np.log(np.full((3, 3), 2.0)),
        units={"flux": "log_photon", "distribution": "log"},
    )
    assert np.allclose(source.params()[3], 1000)
    assert np.allclose(source.params()[4], 2)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: dl.Spectrum([[1e-6]]),
        lambda: dl.Spectrum([1e-6], [[1.0, 1.0], [1.0, 1.0]]),
        lambda: dl.Spectrum([1e-6], units={"unknown": "m"}),
        lambda: dl.PointSource([1e-6], position=np.zeros(3)),
        lambda: dl.PointSource([1e-6], flux=np.ones(2)),
        lambda: dl.PointSource([1e-6], distribution=np.ones(3)),
        lambda: dl.BinarySource([1e-6], centre=np.zeros(3)),
    ],
)
def test_source_validation(operation):
    with pytest.raises((TypeError, ValueError)):
        operation()
