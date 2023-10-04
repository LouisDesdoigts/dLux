from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
from dLux.layers import (
    TransmissiveLayer,
    AberratedLayer,
    BasisLayer,
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
