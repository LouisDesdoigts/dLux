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
@pytest.mark.parametrize("effect", ["opd", "phase", "amplitude"])
def test_basis_layer(basis, coefficients, effect):
    _test_apply(BasisLayer(basis, coefficients, effect))
    with pytest.raises(ValueError):
        BasisLayer(np.ones((5, 16, 16)), np.ones(6))


def test_basis_layer_errors_and_solve():
    basis = np.eye(4).reshape(4, 2, 2)
    coefficients = np.arange(4.0)
    layer = BasisLayer(basis, coefficients)

    assert np.allclose(layer.solve_basis(layer.eval_basis()), coefficients)
    with pytest.raises(ValueError, match="effect"):
        BasisLayer(basis, coefficients, effect="invalid")


def test_amplitude_basis_is_perturbation_from_unity():
    basis = np.ones((2, 16, 16))
    layer = BasisLayer(basis, np.zeros(2), effect="amplitude")

    result = layer(wf)

    assert np.allclose(result.amplitude, wf.amplitude)


def test_tilt():
    _test_apply(Tilt(np.array([0.1, 0.2])))
    with pytest.raises(ValueError):
        Tilt(np.array([0.1, 0.2, 0.3]))


def test_normalise():
    _test_apply(Normalise())
