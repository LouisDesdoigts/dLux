import jax.numpy as np
import pytest
from dLux.layers import (
    Rotate,
    Flip,
    Resize,
)
from dLux import Wavefront, PSF

wf = Wavefront(16, 1, 1e-6)
psf = PSF(np.ones((16, 16)), 1 / 16)


def _test_apply(layer):
    assert isinstance(layer.apply(wf), Wavefront)
    assert isinstance(layer.apply(psf), PSF)


@pytest.mark.parametrize("npixels", [8, 32])
def test_resize(npixels):
    _test_apply(Resize(npixels))


@pytest.mark.parametrize("angle", [np.pi])
@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("complex", [True, False])
def test_rotate(angle, order, complex):
    _test_apply(Rotate(angle, order, complex))
    with pytest.raises(ValueError):
        Rotate(angle, 2, complex)


@pytest.mark.parametrize("axes", [0, 1, (0, 1)])
def test_flip(axes):
    _test_apply(Flip(axes))
    with pytest.raises(ValueError):
        Flip((0.0, 1))
    with pytest.raises(ValueError):
        Flip(0.0)


# @pytest.mark.parametrize("transmission", [None, np.ones((16, 16))])
# @pytest.mark.parametrize("normalise", [True, False])
# def test_transmissive_layer(transmission, normalise):
#     layer = TransmissiveLayer(transmission, normalise)
#     assert isinstance(layer.apply(wf), Wavefront)


# @pytest.mark.parametrize("opd", [None, np.ones((16, 16))])
# @pytest.mark.parametrize("phase", [None, np.ones((16, 16))])
# def test_aberrated_layer(opd, phase):
#     layer = AberratedLayer(opd, phase)
#     assert isinstance(layer.apply(wf), Wavefront)
#     with pytest.raises(ValueError):
#         layer = AberratedLayer(np.ones((4, 4)), np.ones((5, 5)))


# @pytest.mark.parametrize("basis", [np.ones((5, 16, 16))])
# @pytest.mark.parametrize("coefficients", [None, np.ones(5)])
# @pytest.mark.parametrize("as_phase", [False, True])
# def test_basis_layer(basis, coefficients, as_phase):
#     layer = BasisLayer(basis, coefficients, as_phase)
#     assert isinstance(layer.apply(wf), Wavefront)
#     with pytest.raises(ValueError):
#         layer = BasisLayer(np.ones((5, 16, 16)), np.ones(6))


# def test_tilt():
#     layer = Tilt(np.array([0.1, 0.2]))
#     assert isinstance(layer.apply(wf), Wavefront)
#     with pytest.raises(ValueError):
#         layer = Tilt(np.array([0.1, 0.2, 0.3]))


# def test_normalise():
#     layer = Normalise()
#     assert isinstance(layer.apply(wf), Wavefront)
