from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
from dLux.layers import (
    MFT,
    FFT,
    ShiftedMFT,
    FarFieldFresnel,
)
from dLux import Wavefront


wf = Wavefront(16, 1, 1e-6)


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
    _test_apply(FFT(focal_length, pad))


@pytest.mark.parametrize("focal_length", [None, 1e2])
def test_mft(focal_length, npixels, pixel_scale):
    _test_apply(MFT(npixels, pixel_scale, focal_length))


@pytest.mark.parametrize("focal_length", [None, 1e2])
@pytest.mark.parametrize("pixel", [True, False])
def test_shifted_mft(focal_length, npixels, pixel_scale, shift, pixel):
    _test_apply(ShiftedMFT(npixels, pixel_scale, shift, focal_length, pixel))
    with pytest.raises(ValueError):
        ShiftedMFT(npixels, pixel_scale, [1.0], focal_length, True)


@pytest.mark.parametrize("pixel", [True, False])
def test_far_field_fresnel(
    npixels, pixel_scale, focal_length, focal_shift, shift, pixel
):
    _test_apply(
        FarFieldFresnel(
            npixels,
            pixel_scale,
            focal_length,
            focal_shift,
            shift,
            pixel,
        )
    )
