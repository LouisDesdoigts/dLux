from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
import dLux.utils as dlu
from dLux.layers import (
    TransmissiveLayer,
    AberratedLayer,
    BasisLayer,
    FourierBasis,
    Tilt,
    Normalise,
)
from dLux import Wavefront

wf = Wavefront(16, 1, 1e-6)


def _test_apply(layer):
    assert isinstance(layer.apply(wf), Wavefront)


@pytest.mark.parametrize("transmission", [None, np.ones((16, 16))])
@pytest.mark.parametrize("normalise", [True, False])
def test_transmissive_layer(transmission, normalise):
    _test_apply(TransmissiveLayer(transmission, normalise))


@pytest.mark.parametrize("opd", [None, np.ones((16, 16))])
@pytest.mark.parametrize("phase", [None, np.ones((16, 16))])
def test_aberrated_layer(opd, phase):
    _test_apply(AberratedLayer(opd, phase))
    with pytest.raises(ValueError):
        AberratedLayer(np.ones((4, 4)), np.ones((5, 5)))


@pytest.mark.parametrize("basis", [np.ones((5, 16, 16))])
@pytest.mark.parametrize("coefficients", [None, np.ones(5)])
@pytest.mark.parametrize("as_phase", [False, True])
def test_basis_layer(basis, coefficients, as_phase):
    _test_apply(BasisLayer(basis, coefficients, as_phase))
    with pytest.raises(ValueError):
        BasisLayer(np.ones((5, 16, 16)), np.ones(6))


def test_tilt():
    _test_apply(Tilt(np.array([0.1, 0.2])))
    with pytest.raises(ValueError):
        Tilt(np.array([0.1, 0.2, 0.3]))


def test_normalise():
    _test_apply(Normalise())


@pytest.mark.parametrize("coefficients", [None, np.ones((5, 5))])
def test_fourier_basis(coefficients):
    layer = FourierBasis(npix=16, n_modes=5, coefficients=coefficients)
    expected = dlu.eval_fourier_basis(layer.coefficients, *layer.kernels)
    updated = layer.update_kernels(8)
    applied = layer(wf)

    _test_apply(layer)
    assert layer.eval_basis().shape == (16, 16)
    assert np.allclose(layer.eval_basis(), expected)
    assert updated.kernels[0].shape == (8, 5)
    assert updated.kernels[1].shape == (8, 5)
    assert np.allclose(applied.phase, wf.add_opd(expected).phase)

    if coefficients is None:
        assert np.allclose(layer.eval_basis(), 0.0)


def test_fourier_basis_invalid_coefficients():
    with pytest.raises(ValueError):
        FourierBasis(npix=16, n_modes=5, coefficients=np.ones((4, 4)))
