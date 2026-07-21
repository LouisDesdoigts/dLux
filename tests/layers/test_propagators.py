from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
from dLux.layers import (
    MFT,
    FFT,
)
from dLux import Wavefront

wf = Wavefront(npixels=16, diameter=1, wavelength=1e-6)


@pytest.fixture
def pad():
    return 2


@pytest.fixture
def npixels():
    return 8


@pytest.fixture
def pixel_scale():
    return 1 / 16


@pytest.fixture
def shift():
    return np.ones(2)


@pytest.fixture
def focal_shift():
    return 1e-3


@pytest.fixture
def focal_length():
    return 1.0


def _test_apply(layer):
    assert isinstance(layer.apply(wf), Wavefront)


@pytest.mark.parametrize("focal_length", [None, 1e2])
def test_fft(focal_length, pad):
    _test_apply(FFT(focal_length=focal_length, pad=pad))
    _test_apply(FFT(focal_length=focal_length, pad=pad, center=False))


@pytest.mark.parametrize("focal_length", [None, 1e2])
def test_mft(focal_length, npixels, pixel_scale):
    propagator = MFT(
        npixels=npixels, pixel_scale=pixel_scale, focal_length=focal_length
    )
    _test_apply(propagator)
    assert propagator.pixel_scale.shape == ()
