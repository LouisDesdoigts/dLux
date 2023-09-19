import jax.numpy as np
import pytest
from dLux.layers import (
    Optic,
    BasisOptic,
)
from dLux import Wavefront

wf = Wavefront(16, 1, 1e-6)


def _test_apply(layer):
    assert isinstance(layer.apply(wf), Wavefront)


@pytest.mark.parametrize("transmission", [None, np.ones((16, 16))])
@pytest.mark.parametrize("opd", [None, np.ones((16, 16))])
@pytest.mark.parametrize("phase", [None, np.ones((16, 16))])
@pytest.mark.parametrize("normalise", [True, False])
def test_optic(transmission, opd, phase, normalise):
    _test_apply(Optic(transmission, opd, phase, normalise))
    with pytest.raises(ValueError):
        Optic(np.ones((4, 4)), opd=np.ones((5, 5)))
    with pytest.raises(ValueError):
        Optic(np.ones((4, 4)), phase=np.ones((5, 5)))


@pytest.mark.parametrize("basis", [np.ones((5, 16, 16))])
@pytest.mark.parametrize("transmission", [None, np.ones((16, 16))])
@pytest.mark.parametrize("coefficients", [None, np.ones(5)])
@pytest.mark.parametrize("as_phase", [True, False])
@pytest.mark.parametrize("normalise", [True, False])
def test_basis_optic(basis, transmission, coefficients, as_phase, normalise):
    _test_apply(
        BasisOptic(basis, transmission, coefficients, as_phase, normalise)
    )
