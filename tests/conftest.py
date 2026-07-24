"""Shared factories for the dLux test suite."""

import jax.numpy as np
import pytest

import dLux as dl


@pytest.fixture
def make_spec():
    """Return a factory for small two-dimensional coordinate specifications."""

    def factory(n=8, d=0.1, c=0.0, unit="m"):
        return dl.CoordSpec(n=n, d=d, c=c, unit=unit).broadcast(2)

    return factory


@pytest.fixture
def make_wavefront(make_spec):
    """Return a factory for small scalar or polarised wavefronts."""

    def factory(
        wavelength=1e-6,
        spec=None,
        phasor=None,
        polarised=False,
    ):
        spec = make_spec() if spec is None else spec
        wavefront = dl.PolarisedWavefront if polarised else dl.Wavefront
        return wavefront(wavelength, spec, phasor)

    return factory


@pytest.fixture
def make_psf(make_spec):
    """Return a factory for small PSFs."""

    def factory(data=None, spec=None):
        spec = make_spec() if spec is None else spec
        data = np.ones(spec.shape) if data is None else data
        return dl.PSF(data, spec)

    return factory
