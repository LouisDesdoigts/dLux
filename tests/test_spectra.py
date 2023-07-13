import pytest
from jax import config

config.update("jax_debug_nans", True)


def _test_normalise(constructor):
    """Tests the normalise method."""
    constructor().normalise()


class TestSpectrum:
    """Tests the Spectrum class."""

    def test_constructor(self, create_spectrum):
        """Tests the constructor."""
        create_spectrum()
        create_spectrum(weights=None)
        create_spectrum(wavelengths=[1, 2, 3], weights=[[1, 2, 3], [1, 2, 3]])
        with pytest.raises(ValueError):
            create_spectrum(wavelengths=[1], weights=[[1, 2], [1, 2]])
        with pytest.raises(ValueError):
            create_spectrum(wavelengths=[1], weights=[1, 2])

    def test_normalise(self, create_spectrum):
        """Tests the normalise method."""
        _test_normalise(create_spectrum)


class TestPolySpectrum:
    """Tests the PolySpectrum class."""

    def test_constructor(self, create_poly_spectrum):
        """Tests the constructor."""
        create_poly_spectrum()
        with pytest.raises(ValueError):
            create_poly_spectrum(coefficients=1)

    def test_normalise(self, create_poly_spectrum):
        """Tests the normalise method."""
        _test_normalise(create_poly_spectrum)

    def test_weights(self, create_poly_spectrum):
        """Tests the weights property."""
        create_poly_spectrum().weights
