from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
from dLux import PSF


@pytest.fixture
def psf():
    return PSF(np.ones((16, 16)), 1 / 16)


class TestPSF:
    def test_constructor(self, psf):
        assert psf.npixels == 16
        assert psf.pixel_scale == 1 / 16

    def test_properties(self, psf):
        assert psf.ndim == 0

    def test_methods(self, psf):
        assert psf.downsample(2).npixels == 8
        assert psf.downsample(2).pixel_scale == 1 / 8
        assert isinstance(psf.convolve(np.ones((2, 2))), PSF)
        assert isinstance(psf.convolve(np.ones((2, 2)), method="fft"), PSF)
        assert isinstance(psf.rotate(np.pi), PSF)
        assert isinstance(psf.resize(8), PSF)
        assert isinstance(psf.flip(0), PSF)

    def test_magic(self, psf):
        psf *= np.ones(1)
        assert isinstance(psf, PSF)

        psf += np.ones(1)
        assert isinstance(psf, PSF)

        psf -= np.ones(1)
        assert isinstance(psf, PSF)

        psf /= np.ones(1)
        assert isinstance(psf, PSF)
