from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
import equinox as eqx
from dLux.layers import (
    DetectorLayer,
    Downsample,
    Flip,
    Interpolate,
    Lambda,
    Normalise,
    OpticalLayer,
    Resize,
)
from dLux import Affine, Wavefront, PSF

wf = Wavefront(npixels=16, diameter=1, wavelength=1e-6)
psf = PSF(np.ones((16, 16)), 1 / 16)


def _test_apply(layer):
    assert isinstance(layer, OpticalLayer)
    assert isinstance(layer, DetectorLayer)
    assert isinstance(layer.apply(wf), Wavefront)
    assert isinstance(layer.apply(psf), PSF)


@pytest.mark.parametrize("npixels", [8, 32])
def test_resize(npixels):
    _test_apply(Resize(npixels))


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


def test_downsample():
    _test_apply(Downsample(2))
    with pytest.raises(ValueError):
        Downsample(0)


@pytest.mark.parametrize("method", ["nearest", "linear", "cubic"])
def test_interpolate(method):
    layer = Interpolate(
        Affine(
            translation=[0.01, -0.02],
            shear=[0.1, -0.1],
            scale=[0.9, 1.1],
            rotation=0.1,
        ),
        method=method,
        complex=False,
    )
    _test_apply(layer)
    assert isinstance(eqx.filter_jit(layer)(psf), PSF)
    assert layer.method == method
    assert layer.fill.shape == ()
    with pytest.raises(TypeError, match="transformation"):
        Interpolate("rotate")


def test_normalise():
    layer = Normalise(mode="peak", value=2)
    assert np.allclose(layer(wf).psf.max(), 2)
    assert np.allclose(layer(psf).data.max(), 2)
