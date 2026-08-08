"""Shared factories for the dLux test suite."""

import jax.numpy as np
import pytest

import dLux as dl


@pytest.fixture
def make_grid():
    """Return a factory for small two-dimensional coordinate specifications."""

    def factory(n=8, d=0.1, c=0.0, unit="m"):
        return dl.GridSpec(n=n, d=d, c=c, unit=unit).broadcast(2)

    return factory


@pytest.fixture
def make_wavefront(make_grid):
    """Return a factory for small scalar or polarised wavefronts."""

    def factory(
        wavelength=1e-6,
        grid=None,
        phasor=None,
        polarised=False,
    ):
        grid = make_grid() if grid is None else grid
        wavefront = dl.PolarisedWavefront if polarised else dl.Wavefront
        return wavefront(wavelength, grid, phasor)

    return factory


@pytest.fixture
def make_psf(make_grid):
    """Return a factory for small PSFs."""

    def factory(data=None, grid=None):
        grid = make_grid() if grid is None else grid
        data = np.ones(grid.shape) if data is None else data
        return dl.PSF(data, grid)

    return factory
