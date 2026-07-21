from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
from dLux.layers import Flip, Lambda, Resize, Rotate
from dLux import Wavefront, PSF

wf = Wavefront(npixels=16, diameter=1, wavelength=1e-6)
psf = PSF(np.ones((16, 16)), 1 / 16)


def _test_apply(layer):
    assert isinstance(layer.apply(wf), Wavefront)
    assert isinstance(layer.apply(psf), PSF)


@pytest.mark.parametrize("npixels", [8, 32])
def test_resize(npixels):
    _test_apply(Resize(npixels))


@pytest.mark.parametrize("angle", [np.pi])
@pytest.mark.parametrize("method", ["nearest", "linear", "cubic"])
@pytest.mark.parametrize("complex", [True, False])
def test_rotate(angle, method, complex):
    layer = Rotate(angle, method, complex)
    _test_apply(layer)
    assert layer.angle.shape == ()


@pytest.mark.parametrize("axes", [0, 1, (0, 1)])
def test_flip(axes):
    _test_apply(Flip(axes))
    with pytest.raises(ValueError):
        Flip((0.0, 1))
    with pytest.raises(ValueError):
        Flip(0.0)


def test_lambda():
    layer = Lambda()

    assert layer.apply(wf) is wf
    assert layer.apply(psf) is psf
