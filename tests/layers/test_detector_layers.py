from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
from dLux.layers import (
    ApplyPixelResponse,
    ApplyJitter,
    ApplySaturation,
    AddConstant,
    Downsample,
)
from dLux import PSF

psf = PSF(np.ones((16, 16)), 1 / 16)


def _test_apply(layer):
    assert isinstance(layer.apply(psf), PSF)


def test_apply_pixel_response():
    _test_apply(ApplyPixelResponse(np.ones((16, 16))))
    with pytest.raises(ValueError):
        ApplyPixelResponse(np.ones(16))


def test_apply_jitter():
    _test_apply(ApplyJitter(0.1, 5))


def test_apply_saturation():
    _test_apply(ApplySaturation(1e4))


def test_add_constant():
    _test_apply(AddConstant(1))


def test_downsample():
    _test_apply(Downsample(8))
