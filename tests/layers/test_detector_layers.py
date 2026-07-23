from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
import equinox as eqx
from dLux.layers import (
    ApplyPixelResponse,
    ApplyJitter,
    ApplySaturation,
    AddConstant,
    Downsample,
    ApplyInterpolation,
)
from dLux import Affine, PSF, Rotation, Scaling, Shearing, Translation

psf = PSF(np.ones((16, 16)), 1 / 16)


def _test_apply(layer):
    assert isinstance(layer.apply(psf), PSF)


def test_apply_pixel_response():
    _test_apply(ApplyPixelResponse(np.ones((16, 16))))
    with pytest.raises(ValueError):
        ApplyPixelResponse(np.ones(16))


def test_apply_jitter():
    layer = ApplyJitter(0.1, 5)
    _test_apply(layer)
    assert layer.sigma.shape == ()
    with pytest.raises(ValueError):
        ApplyJitter(0.1, 0)


def test_apply_saturation():
    layer = ApplySaturation(1e4)
    _test_apply(layer)
    assert layer.threshold.shape == ()


def test_add_constant():
    layer = AddConstant(1)
    _test_apply(layer)
    assert layer.value.shape == ()


def test_downsample():
    _test_apply(Downsample(8))
    with pytest.raises(ValueError):
        Downsample(0)


@pytest.mark.parametrize("method", ["nearest", "linear", "cubic"])
def test_apply_interpolation(method):
    layer = ApplyInterpolation(
        Affine(
            [
                Translation([0.01, -0.02]),
                Shearing([0.1, -0.1]),
                Scaling([0.9, 1.1]),
                Rotation(0.1),
            ]
        ),
        method=method,
    )
    _test_apply(layer)
    assert isinstance(eqx.filter_jit(layer)(psf), PSF)
    assert not hasattr(layer, "complex")
    assert layer.method == method
    assert layer.fill.shape == ()

    with pytest.raises(TypeError, match="transformation"):
        ApplyInterpolation(transformation="rotate")
