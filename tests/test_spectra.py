import jax.numpy as np
import pytest
from dLux import Spectrum, PolySpectrum


def test_spectrum():
    # Test 1d
    Spectrum(np.ones(3))
    spec = Spectrum(np.ones(3), np.ones(3))
    assert isinstance(spec.normalise(), Spectrum)

    # Test 2d
    spec = Spectrum(np.ones(3), np.ones((2, 3)))
    assert isinstance(spec.normalise(), Spectrum)

    # Test failure
    with pytest.raises(ValueError):
        Spectrum(np.ones((2, 3)), np.ones(3))
    with pytest.raises(ValueError):
        Spectrum(np.ones(3), np.ones((2, 4)))


def test_poly_spectrum():
    spec = PolySpectrum(np.ones(3), np.ones(3))
    assert isinstance(spec.weights, np.ndarray)
    assert isinstance(spec.normalise(), PolySpectrum)

    with pytest.raises(ValueError):
        PolySpectrum(np.ones(3), np.ones((2, 1)))
